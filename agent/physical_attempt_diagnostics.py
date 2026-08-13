"""Opt-in, content-free diagnostics for physical provider attempts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

__all__ = ["Attempt", "enabled", "finish_attempt", "prepare_cache_scope", "start_attempt"]

_SCHEMA = "hermes.physical_attempt.v1"
_KEY_FILE = "physical_attempt_digests.key"
_RECORDS_FILE = "physical_attempt_digests.jsonl"
_PHASES = {"start", "terminal", "pair"}
_DIFFERENCES = {
    "system_or_static_prefix",
    "tools",
    "cache_key_or_scope",
    "later_history",
    "none",
}
_OUTCOMES = {"completed", "incomplete", "failed", "error", "cancelled", "interrupted"}
_ROOT_PARTS = ("observability",)
_PAIR_LOCK = threading.Lock()
_LAST_ATTEMPT: dict[tuple[str, str], "Attempt"] = {}


@dataclass
class Attempt:
    """Private in-memory identity for one physical provider request."""

    digest: str
    route: str
    loop: int | None
    correlation: str
    components: dict[str, tuple[str, int]]
    finished: bool = False


def enabled() -> bool:
    """Return whether the local diagnostic sink is explicitly enabled."""
    try:
        from hermes_cli.config import read_raw_config_readonly

        config = read_raw_config_readonly() or {}
        value = config.get("observability", {}).get("physical_attempt_digests", {})
        return isinstance(value, dict) and value.get("enabled") is True
    except Exception:
        return False


def prepare_cache_scope(value: Any) -> dict[str, Any] | None:
    """Digest a logical cache scope without persisting its contents."""
    if not enabled():
        return None
    try:
        digest, size = _digest_value(_digest_key(), "scope", value)
        return {"scope_digest": digest, "scope_bytes": size}
    except Exception:
        return None


def start_attempt(
    request: dict[str, Any],
    *,
    api_mode: str,
    route: str,
    provider: str,
    model: str,
    role: str,
    retry: int,
    continuation: int,
    scope: dict[str, Any] | None = None,
    loop: int | None = None,
    correlation: str = "",
) -> Attempt | None:
    """Record one final provider request and pair adjacent comparable loops."""
    if not enabled():
        return None
    try:
        key = _digest_key()
        components = _components(key, request, api_mode=api_mode, scope=scope)
        attempt = Attempt(
            digest=hmac.new(key, b"attempt\0" + secrets.token_bytes(32), hashlib.sha256).hexdigest(),
            route=_safe_label(route, 128),
            loop=loop if isinstance(loop, int) and loop >= 0 else None,
            correlation=str(correlation or ""),
            components=components,
        )
        record = {
            "schema": _SCHEMA,
            "phase": "start",
            "attempt_digest": attempt.digest,
            "monotonic_ns": time.monotonic_ns(),
            "route": attempt.route,
            "provider": _safe_label(provider, 128),
            "model": _safe_label(model, 128),
            "role": role if role in {"main", "subagent", "reviewer", "fallback", "auxiliary"} else "unknown",
            "retry": max(0, int(retry)),
            "continuation": max(0, int(continuation)),
            "scope_digest": (scope or {}).get("scope_digest"),
            "scope_bytes": (scope or {}).get("scope_bytes"),
            "key_digest": components["cache_key_or_scope"][0],
            "key_bytes": components["cache_key_or_scope"][1],
            "prefix_digest": components["system_or_static_prefix"][0],
            "prefix_bytes": components["system_or_static_prefix"][1],
            "tool_digest": components["tools"][0],
            "tool_bytes": components["tools"][1],
            "later_history_digest": components["later_history"][0],
            "later_history_bytes": components["later_history"][1],
        }
        _append(record)
        _pair(attempt)
        return attempt
    except Exception:
        return None


def finish_attempt(
    attempt: Attempt | None,
    *,
    usage: Any,
    outcome: str,
    api_mode: str,
    provider: str,
) -> None:
    """Record allowlisted terminal cache/usage scalars for an attempt."""
    del api_mode, provider
    if attempt is None or attempt.finished:
        return
    attempt.finished = True
    try:
        from agent.usage_pricing import normalize_usage

        canonical = normalize_usage(usage, api_mode="chat_completions", provider="")
        cache_present = bool(usage) and canonical.cache_telemetry_present
        _append(
            {
                "schema": _SCHEMA,
                "phase": "terminal",
                "attempt_digest": attempt.digest,
                "monotonic_ns": time.monotonic_ns(),
                "outcome": outcome if outcome in _OUTCOMES else "unknown",
                "cache_state": "unknown" if not cache_present else "hit" if canonical.cache_read_tokens > 0 else "miss",
                "input_tokens": canonical.input_tokens if usage else None,
                "output_tokens": canonical.output_tokens if usage else None,
                "cache_read_tokens": canonical.cache_read_tokens if cache_present else None,
                "cache_write_tokens": canonical.cache_write_tokens if cache_present else None,
            }
        )
    except Exception:
        return


def _components(
    key: bytes, request: dict[str, Any], *, api_mode: str, scope: dict[str, Any] | None
) -> dict[str, tuple[str, int]]:
    if api_mode == "codex_responses":
        static = request.get("instructions")
        later = request.get("input")
    else:
        messages = request.get("messages") if isinstance(request.get("messages"), list) else []
        static_messages = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
                break
            static_messages.append(message)
        static = request.get("system", static_messages)
        later = messages[len(static_messages):]
    tools = request.get("tools") or request.get("toolConfig") or []
    cache_key = _effective_cache_key(request)
    cache_scope = {
        "scope_digest": (scope or {}).get("scope_digest"),
        "scope_bytes": (scope or {}).get("scope_bytes"),
        "cache_key": cache_key,
    }
    return {
        "system_or_static_prefix": _digest_value(key, "static", static),
        "tools": _digest_value(key, "tools", tools),
        "cache_key_or_scope": _digest_value(key, "cache", cache_scope),
        "later_history": _digest_value(key, "later", later),
    }


def _pair(current: Attempt) -> None:
    if current.loop is None or not current.correlation:
        return
    identity = (current.correlation, current.route)
    with _PAIR_LOCK:
        previous = _LAST_ATTEMPT.get(identity)
        _LAST_ATTEMPT[identity] = current
    if previous is None or previous.loop != current.loop - 1:
        return
    difference = next(
        (
            name
            for name in ("system_or_static_prefix", "tools", "cache_key_or_scope", "later_history")
            if previous.components[name] != current.components[name]
        ),
        "none",
    )
    before = previous.components.get(difference)
    after = current.components.get(difference)
    _append(
        {
            "schema": _SCHEMA,
            "phase": "pair",
            "previous_attempt_digest": previous.digest,
            "current_attempt_digest": current.digest,
            "route": current.route,
            "first_differing_class": difference if difference in _DIFFERENCES else "none",
            "previous_digest": before[0] if before else None,
            "previous_bytes": before[1] if before else None,
            "current_digest": after[0] if after else None,
            "current_bytes": after[1] if after else None,
        }
    )


def _effective_cache_key(request: dict[str, Any]) -> Any:
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict) and extra_body.get("prompt_cache_key") is not None:
        return extra_body["prompt_cache_key"]
    if request.get("prompt_cache_key") is not None:
        return request["prompt_cache_key"]
    headers = request.get("extra_headers")
    return headers.get("x-client-request-id") if isinstance(headers, dict) else None


def _digest_value(key: bytes, label: str, value: Any) -> tuple[str, int]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8", errors="replace")
    return hmac.new(key, label.encode() + b"\0" + encoded, hashlib.sha256).hexdigest(), len(encoded)


def _safe_label(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > limit or "://" in text:
        return "unknown"
    try:
        from agent.redact import redact_sensitive_text

        if redact_sensitive_text(text, force=True, redact_url_credentials=True) != text:
            return "unknown"
    except Exception:
        return "unknown"
    return text if all(ch.isascii() and (ch.isalnum() or ch in "._/:-") for ch in text) else "unknown"


def _root() -> Path:
    return get_hermes_home().joinpath(*_ROOT_PARTS)


def _ensure_root() -> Path:
    root = _root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise PermissionError("unsafe physical-attempt diagnostics directory")
    if stat.S_IMODE(info.st_mode) != 0o700:
        root.chmod(0o700)
    return root


def _private_fd(path: Path, flags: int, *, create: bool = False) -> int:
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600) if create else os.open(path, flags | getattr(os, "O_NOFOLLOW", 0))
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        os.close(fd)
        raise PermissionError("unsafe physical-attempt diagnostics file")
    return fd


def _digest_key() -> bytes:
    path = _ensure_root() / _KEY_FILE
    try:
        fd = _private_fd(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, create=True)
    except FileExistsError:
        fd = _private_fd(path, os.O_RDONLY)
        try:
            value = os.read(fd, 33)
        finally:
            os.close(fd)
        if len(value) != 32:
            raise ValueError("invalid physical-attempt digest key")
        return value
    value = secrets.token_bytes(32)
    try:
        os.write(fd, value)
        os.fsync(fd)
    finally:
        os.close(fd)
    return value


def _append(record: dict[str, Any]) -> None:
    if record.get("phase") not in _PHASES:
        return
    fd = _private_fd(_ensure_root() / _RECORDS_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT, create=True)
    try:
        os.write(fd, (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    finally:
        os.close(fd)
