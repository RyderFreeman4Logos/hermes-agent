"""Opt-in paired records for exact physical provider attempts.

The request capture file is the source of truth for each attempt.  This small
sidecar preserves pairing metadata and the same redacted request/body pair so
retry and fallback attempts remain inspectable without credentials.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.cache_request_capture import _redact, _serialize_body
from hermes_constants import get_hermes_home

__all__ = ["Attempt", "enabled", "prepare_cache_scope", "start_attempt"]

_RECORDS_FILE = "physical_attempt_digests.jsonl"
_CACHE_SCOPE_ENVELOPE = "_hermes_physical_attempt_cache_scope"
_LOCK = threading.Lock()
_LAST_ATTEMPT: dict[tuple[str, str, str, str], "Attempt"] = {}
# ponytail: fixed local caps; add rotation only if diagnostics volume matters.
_MAX_TRACKED_CORRELATIONS = 256
_MAX_RECORDS_BYTES = 8 * 1024 * 1024


@dataclass
class Attempt:
    request: dict[str, Any]
    body_bytes: dict[str, str]
    route: str
    provider: str
    model: str
    loop: int
    retry: int
    components: dict[str, Any]
    byte_lengths: dict[str, int]


def enabled() -> bool:
    """Return whether exact physical-attempt diagnostics are explicitly enabled."""
    try:
        from hermes_cli.config import read_raw_config_readonly

        config = read_raw_config_readonly() or {}
        value = config.get("observability", {}).get("physical_attempt_digests", {})
        return isinstance(value, dict) and value.get("enabled") is True
    except Exception:
        return False


def prepare_cache_scope(value: Any) -> dict[str, Any] | None:
    """Keep a redacted cache scope briefly for the next physical attempt."""
    if not enabled():
        return None
    return {"value": _redact(value)}


def take_cache_scope(request: dict[str, Any]) -> dict[str, Any] | None:
    """Remove the private cache-scope envelope from a provider request."""
    scope = request.pop(_CACHE_SCOPE_ENVELOPE, None)
    return scope if isinstance(scope, dict) else None


def start_attempt(
    request: dict[str, Any],
    *,
    api_mode: str,
    route: str,
    provider: str,
    model: str,
    retry: int,
    loop: int | None,
    correlation: str,
    scope: dict[str, Any] | None = None,
) -> Attempt | None:
    """Persist one redacted request/body pair and pair adjacent loop attempts."""
    if not enabled() or loop is None or loop < 0 or not correlation:
        return None
    try:
        safe_request = _redact(request)
        body = _serialize_body(safe_request)
        body_record = {
            "encoding": "base64",
            "data": _base64(body),
        }
        safe_route = _label(route)
        safe_provider = _label(provider)
        safe_model = _label(model, casefold=True)
        components = _components(safe_request, api_mode, scope)
        lengths = {name: len(_serialize_body(value)) for name, value in components.items()}
        attempt = Attempt(
            request=safe_request,
            body_bytes=body_record,
            route=safe_route,
            provider=safe_provider,
            model=safe_model,
            loop=loop,
            retry=max(0, _integer(retry)),
            components=components,
            byte_lengths=lengths,
        )
        _append(
            {
                "schema": "hermes.physical_attempt.v2",
                "phase": "attempt",
                "timestamp_ns": time.time_ns(),
                "route": safe_route,
                "provider": safe_provider,
                "model": safe_model,
                "loop": loop,
                "retry": attempt.retry,
                "request": safe_request,
                "body_bytes": body_record,
                "byte_lengths": lengths,
            }
        )
        _pair(attempt, correlation)
        return attempt
    except Exception:
        return None


def _pair(current: Attempt, correlation: str) -> None:
    identity = (_key(correlation), current.route, current.provider, current.model)
    with _LOCK:
        previous = _LAST_ATTEMPT.pop(identity, None)
        _LAST_ATTEMPT[identity] = current
        while len(_LAST_ATTEMPT) > _MAX_TRACKED_CORRELATIONS:
            _LAST_ATTEMPT.pop(next(iter(_LAST_ATTEMPT)))
    if previous is None or previous.loop != current.loop - 1:
        return
    names = ("prefix", "tools", "cache_scope", "later_history")
    equal = {name: previous.components[name] == current.components[name] for name in names}
    _append(
        {
            "schema": "hermes.physical_attempt.v2",
            "phase": "pair",
            "timestamp_ns": time.time_ns(),
            "route": current.route,
            "provider": current.provider,
            "model": current.model,
            "previous_loop": previous.loop,
            "current_loop": current.loop,
            "previous_attempt_retry": previous.retry,
            "previous": _snapshot(previous),
            "current": _snapshot(current),
            "equal": equal,
            "first_differing_segment": next(
                (name for name in names if not equal[name]), "none"
            ),
        }
    )


def _snapshot(attempt: Attempt) -> dict[str, Any]:
    return {
        "request": attempt.request,
        "body_bytes": attempt.body_bytes,
        "byte_lengths": attempt.byte_lengths,
    }


def _components(
    request: dict[str, Any], api_mode: str, scope: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "prefix": _prefix(request),
        "tools": request.get("tools") or request.get("toolConfig") or [],
        "cache_scope": {
            "scope": (scope or {}).get("value"),
            "key": _cache_key(request),
        },
        "later_history": _later_history(request, api_mode),
    }


def _prefix(request: dict[str, Any]) -> Any:
    if request.get("instructions") is not None:
        return request["instructions"]
    if request.get("system") is not None:
        return request["system"]
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    prefix = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
            break
        prefix.append(message)
    return prefix


def _later_history(request: dict[str, Any], api_mode: str) -> Any:
    if api_mode == "codex_responses":
        return request.get("input") or []
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    if request.get("instructions") is not None or request.get("system") is not None:
        return messages
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
            return messages[index:]
    return []


def _cache_key(request: dict[str, Any]) -> Any:
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict) and extra_body.get("prompt_cache_key") is not None:
        return extra_body["prompt_cache_key"]
    return request.get("prompt_cache_key")


def _label(value: Any, *, casefold: bool = False) -> str:
    text = str(value or "").strip()
    if casefold:
        text = text.casefold()
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else "unknown"


def _key(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _base64(value: bytes) -> str:
    import base64

    return base64.b64encode(value).decode("ascii")


def _root() -> Path:
    return get_hermes_home() / "observability"


def _append(record: dict[str, Any]) -> None:
    root = _root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.stat(follow_symlinks=False)
    euid = getattr(os, "geteuid", lambda: None)()
    if not stat.S_ISDIR(info.st_mode) or (euid is not None and info.st_uid != euid):
        raise PermissionError("unsafe physical attempt diagnostics directory")
    root.chmod(0o700)
    path = root / _RECORDS_FILE
    payload = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with _LOCK:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            if os.fstat(fd).st_size + len(payload) > _MAX_RECORDS_BYTES:
                os.ftruncate(fd, 0)
            os.write(fd, payload)
        finally:
            os.close(fd)
