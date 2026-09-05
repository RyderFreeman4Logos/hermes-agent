"""Opt-in exact last-2 send-time dump on economically near-zero cache hits."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import deque
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

from agent import cache_request_capture
from agent.usage_pricing import CanonicalUsage

__all__ = [
    "MAX_DUMPS",
    "maybe_dump_on_usage",
    "remember_sent_request",
    "reset_for_tests",
]

MAX_DUMPS = 8
_LOCK = threading.Lock()
_LAST: deque[dict[str, Any]] = deque(maxlen=2)
_KEY = secrets.token_bytes(32)


def reset_for_tests() -> None:
    """Clear the in-memory last-2 buffer. Test-only."""
    with _LOCK:
        _LAST.clear()


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


def _serialized(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")


def _digest(key: bytes, label: str, value: Any) -> str:
    encoded = _serialized(value)
    return hmac.new(key, label.encode() + b"\0" + encoded, hashlib.sha256).hexdigest()


def _capture_enabled() -> bool:
    if cache_request_capture.enabled():
        return True
    with _LOCK:
        _LAST.clear()
    return False


def remember_sent_request(
    request: dict[str, Any], *, api_mode: str = "chat_completions"
) -> None:
    """Keep the last two opt-in send-time request/body pairs."""
    if not _capture_enabled():
        return
    redacted_request = cache_request_capture._redact(request)
    body_bytes = cache_request_capture._serialize_body(redacted_request)
    components = {
        "prefix": _prefix(request),
        "messages": request.get("messages"),
        "input": request.get("input"),
        "tools": request.get("tools") or request.get("toolConfig") or [],
        "prompt_cache_key": _cache_key(request),
        "later_history": _later_history(request, api_mode),
    }
    snapshot = {
        "fingerprint": _digest(_KEY, "cache_lowhit", components),
        "sizes": {
            f"{name}_bytes": len(_serialized(value)) for name, value in components.items()
        },
        "model": request.get("model"),
        "api_mode": api_mode,
        "request": redacted_request,
        "body_bytes": {
            "encoding": "base64",
            "data": base64.b64encode(body_bytes).decode("ascii"),
        },
    }
    with _LOCK:
        _LAST.append(snapshot)


def _is_near_zero(usage: CanonicalUsage) -> bool:
    if usage.cache_telemetry != "reported":
        return False
    cache_read = usage.cache_read_tokens
    prompt = usage.prompt_tokens
    if cache_read == 0:
        return True
    return cache_read > 0 and 100 * cache_read < prompt


def maybe_dump_on_usage(usage: CanonicalUsage) -> None:
    """Write the last two request/body pairs on economically near-zero hits."""
    if not _capture_enabled() or not _is_near_zero(usage):
        return
    with _LOCK:
        requests = list(_LAST)
    if not requests:
        return
    root = get_hermes_home() / "observability" / "cache_lowhit"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing = sorted(path for path in root.iterdir() if path.suffix == ".json")
    overflow = len(existing) + 1 - MAX_DUMPS
    for stale in existing[: max(0, overflow)]:
        stale.unlink(missing_ok=True)
    path = root / f"{time.time_ns()}.json"
    atomic_json_write(
        path,
        {
            "schema": "hermes.cache_lowhit.v2",
            "cache_read_tokens": usage.cache_read_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "requests": requests,
        },
    )
