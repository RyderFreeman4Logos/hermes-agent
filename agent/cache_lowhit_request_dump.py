"""Unredacted last-2 send-time dump on economically near-zero cache hits."""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

from agent.physical_attempt_diagnostics import _cache_key, _later_history, _prefix
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


def reset_for_tests() -> None:
    """Clear the in-memory last-2 buffer. Test-only."""
    with _LOCK:
        _LAST.clear()


def remember_sent_request(
    request: dict[str, Any], *, api_mode: str = "chat_completions"
) -> None:
    """Keep the last two actually-sent cache-identity snapshots."""
    snapshot = copy.deepcopy(
        {
            "prefix": _prefix(request),
            "messages": request.get("messages"),
            "input": request.get("input"),
            "tools": request.get("tools") or request.get("toolConfig") or [],
            "prompt_cache_key": _cache_key(request),
            "model": request.get("model"),
            "later_history": _later_history(request, api_mode),
        }
    )
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
    """Write the last two unredacted send-time prefixes when the hit is near-zero."""
    if not _is_near_zero(usage):
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
            "schema": "hermes.cache_lowhit.v1",
            "cache_read_tokens": usage.cache_read_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "requests": requests,
        },
    )
