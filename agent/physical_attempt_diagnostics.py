"""Opt-in, content-free digests for adjacent physical provider attempts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

__all__ = ["Attempt", "enabled", "prepare_cache_scope", "start_attempt"]

_KEY_FILE = "physical_attempt_digests.key"
_RECORDS_FILE = "physical_attempt_digests.jsonl"
_LOCK = threading.Lock()
_LAST_ATTEMPT: dict[tuple[str, str], "Attempt"] = {}


@dataclass
class Attempt:
    """Private in-memory identity for one provider request."""

    digest: str
    route: str
    loop: int
    retry: int
    correlation: str
    components: dict[str, str]


def enabled() -> bool:
    """Return whether local paired-attempt diagnostics were explicitly enabled."""
    try:
        from hermes_cli.config import read_raw_config_readonly

        config = read_raw_config_readonly() or {}
        value = config.get("observability", {}).get("physical_attempt_digests", {})
        return isinstance(value, dict) and value.get("enabled") is True
    except Exception:
        return False


def prepare_cache_scope(value: Any) -> dict[str, str] | None:
    """Digest cache-scope input without retaining it."""
    if not enabled():
        return None
    try:
        return {"digest": _digest(_key(), "cache_scope", value)}
    except Exception:
        return None


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
    scope: dict[str, str] | None = None,
) -> Attempt | None:
    """Record an opt-in HMAC-only request identity and adjacent-loop pair."""
    del api_mode, provider, model
    if not enabled() or loop is None or loop < 0 or not correlation:
        return None
    try:
        key = _key()
        components = {
            "prefix": _digest(key, "prefix", _prefix(request)),
            "tools": _digest(key, "tools", request.get("tools") or request.get("toolConfig") or []),
            "cache_scope": _digest(
                key,
                "cache_scope",
                {"scope": (scope or {}).get("digest"), "key": _cache_key(request)},
            ),
        }
        attempt = Attempt(
            digest=hmac.new(key, b"attempt\0" + secrets.token_bytes(32), hashlib.sha256).hexdigest(),
            route=_label(route),
            loop=loop,
            retry=max(0, int(retry)),
            correlation=correlation,
            components=components,
        )
        _append({
            "schema": "hermes.physical_attempt.v1",
            "phase": "attempt",
            "attempt_digest": attempt.digest,
            "route": attempt.route,
            "loop": attempt.loop,
            "retry": attempt.retry,
            "digests": components,
        })
        _pair(attempt)
        return attempt
    except Exception:
        return None


def _pair(current: Attempt) -> None:
    identity = (current.correlation, current.route)
    with _LOCK:
        previous = _LAST_ATTEMPT.get(identity)
        _LAST_ATTEMPT[identity] = current
    if previous is None or previous.loop != current.loop - 1:
        return
    _append({
        "schema": "hermes.physical_attempt.v1",
        "phase": "pair",
        "previous_attempt_digest": previous.digest,
        "current_attempt_digest": current.digest,
        "previous_loop": previous.loop,
        "current_loop": current.loop,
        "previous_attempt_retry": previous.retry,
        "digests": {
            name: current.components[name]
            for name in ("cache_scope", "prefix", "tools")
        },
    })


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


def _cache_key(request: dict[str, Any]) -> Any:
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict) and extra_body.get("prompt_cache_key") is not None:
        return extra_body["prompt_cache_key"]
    return request.get("prompt_cache_key")


def _digest(key: bytes, label: str, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hmac.new(key, label.encode() + b"\0" + encoded, hashlib.sha256).hexdigest()


def _label(value: Any) -> str:
    text = str(value or "")
    return text if text and len(text) <= 128 and "://" not in text else "unknown"


def _root() -> Path:
    return get_hermes_home() / "observability"


def _key() -> bytes:
    root = _root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise PermissionError("unsafe physical attempt diagnostics directory")
    if stat.S_IMODE(info.st_mode) != 0o700:
        root.chmod(0o700)
    path = root / _KEY_FILE
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        fd = _private_fd(path, os.O_RDONLY)
        try:
            value = os.read(fd, 33)
        finally:
            os.close(fd)
        if len(value) != 32:
            raise ValueError("invalid physical attempt digest key")
        return value
    value = secrets.token_bytes(32)
    try:
        os.write(fd, value)
    finally:
        os.close(fd)
    return value


def _private_fd(path: Path, flags: int) -> int:
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        os.close(fd)
        raise PermissionError("unsafe physical attempt diagnostics file")
    return fd


def _append(record: dict[str, Any]) -> None:
    path = _root() / _RECORDS_FILE
    fd = _private_fd(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    try:
        os.write(fd, (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    finally:
        os.close(fd)
