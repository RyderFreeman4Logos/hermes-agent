"""Fingerprint-only last-2 send-time dump on economically near-zero cache hits."""

from __future__ import annotations

import contextvars
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

from agent.physical_attempt_diagnostics import (
    _cache_key,
    _digest,
    _key,
    _label,
    _later_history,
    _profile_lock,
    _prefix,
    _serialized,
    enabled,
)
from agent.usage_pricing import CanonicalUsage

__all__ = [
    "MAX_DUMPS",
    "maybe_dump_on_usage",
    "remember_sent_request",
    "reset_for_tests",
]

MAX_DUMPS = 8
_MAX_TRACKED_CALLS = 256
_LOCK = threading.Lock()
_LAST: dict[tuple[str, str], deque[dict[str, Any]]] = {}
_CURRENT_EVENT: contextvars.ContextVar["RequestEvent | None"] = contextvars.ContextVar(
    "hermes_cache_lowhit_event", default=None
)


@dataclass
class RequestEvent:
    """Send-time digest history owned by one response and one profile."""

    root: Path
    requests: tuple[dict[str, Any], ...]
    published: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def reset_for_tests() -> None:
    """Clear the in-memory last-2 buffer. Test-only."""
    with _LOCK:
        _LAST.clear()
    _CURRENT_EVENT.set(None)


def remember_sent_request(
    request: dict[str, Any], *, api_mode: str = "chat_completions", correlation: str = ""
) -> RequestEvent | None:
    """Return the digest-only history owned by this send and its eventual response."""
    if not enabled():
        return None
    profile_root = get_hermes_home()
    components = {
        "prefix": _prefix(request),
        "messages": request.get("messages"),
        "input": request.get("input"),
        "tools": request.get("tools") or request.get("toolConfig") or [],
        "prompt_cache_key": _cache_key(request),
        "later_history": _later_history(request, api_mode),
    }
    key = _key()
    snapshot = {
        "fingerprint": _digest(key, "cache_lowhit", components),
        "sizes": {
            f"{name}_bytes": len(_serialized(value)) for name, value in components.items()
        },
        "model": _label(request.get("model"), key),
    }
    with _LOCK:
        identity = (str(profile_root), correlation or "__default__")
        history = _LAST.setdefault(identity, deque(maxlen=2))
        history.append(snapshot)
        while len(_LAST) > _MAX_TRACKED_CALLS:
            _LAST.pop(next(iter(_LAST)))
        event = RequestEvent(profile_root, tuple(history))
    _CURRENT_EVENT.set(event)
    return event


def activate_event(event: RequestEvent | None) -> None:
    """Make a completed physical send's response event visible to accounting."""
    _CURRENT_EVENT.set(event)


def _is_near_zero(usage: CanonicalUsage, *, cache_telemetry: str = "unavailable") -> bool:
    # Official CanonicalUsage has no cache_telemetry field. LIVE dumps only when
    # the response actually reported cache buckets; callers pass that status.
    if cache_telemetry != "reported":
        return False
    cache_read = usage.cache_read_tokens
    prompt = usage.prompt_tokens
    if cache_read == 0:
        return True
    return cache_read > 0 and 100 * cache_read < prompt


def maybe_dump_on_usage(
    usage: CanonicalUsage, *, cache_telemetry: str = "unavailable",
    event: RequestEvent | None = None,
) -> None:
    """Write this response's send-time fingerprints once when its hit is near-zero."""
    if not enabled():
        return
    if not _is_near_zero(usage, cache_telemetry=cache_telemetry):
        return
    event = event or _CURRENT_EVENT.get()
    if event is None or not event.requests:
        return
    with event.lock:
        if event.published:
            return
        root = event.root / "observability" / "cache_lowhit"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with _profile_lock(root.parent / ".cache_lowhit.lock"):
            existing = sorted(path for path in root.iterdir() if path.suffix == ".json")
            overflow = len(existing) + 1 - MAX_DUMPS
            for stale in existing[: max(0, overflow)]:
                stale.unlink(missing_ok=True)
            path = root / f"{time.time_ns()}-{secrets.token_hex(4)}.json"
            atomic_json_write(
                path,
                {
                    "schema": "hermes.cache_lowhit.v1",
                    "cache_read_tokens": usage.cache_read_tokens,
                    "prompt_tokens": usage.prompt_tokens,
                    "requests": list(event.requests),
                },
            )
        event.published = True
