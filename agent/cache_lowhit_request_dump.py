"""Opt-in last-two exact request dumps for economically near-zero cache hits."""

from __future__ import annotations

import base64
import json
import threading
import time
from collections import deque
from typing import Any

from agent.cache_request_capture import _redact, _serialize_body, enabled
from agent.usage_pricing import CanonicalUsage
from hermes_constants import get_hermes_home
from utils import atomic_json_write

__all__ = ["MAX_DUMPS", "maybe_dump_on_usage", "remember_sent_request", "reset_for_tests"]

MAX_DUMPS = 8
_LOCK = threading.Lock()
_LAST: deque[dict[str, Any]] = deque(maxlen=2)


def reset_for_tests() -> None:
    with _LOCK:
        _LAST.clear()


def remember_sent_request(
    request: dict[str, Any],
    *,
    api_mode: str = "chat_completions",
    route: str = "unknown",
    provider: str = "unknown",
) -> None:
    """Keep the last two redacted payload/body pairs, never raw provider data."""
    if not enabled():
        return
    payload = _redact(request)
    body = _serialize_body(payload)
    snapshot = {
        "api_mode": str(api_mode),
        "route": str(route),
        "provider": str(provider),
        "model": payload.get("model"),
        "request": payload,
        "body_bytes": {
            "encoding": "base64",
            "data": base64.b64encode(body).decode("ascii"),
        },
    }
    with _LOCK:
        _LAST.append(snapshot)


def _is_near_zero(usage: CanonicalUsage) -> bool:
    if getattr(usage, "cache_telemetry", "unavailable") != "reported":
        return False
    cache_read = max(0, int(getattr(usage, "cache_read_tokens", 0)))
    prompt = max(0, int(getattr(usage, "prompt_tokens", 0)))
    return cache_read == 0 or (prompt > 0 and 100 * cache_read < prompt)


def maybe_dump_on_usage(usage: CanonicalUsage) -> None:
    """Persist the last two exact pairs only when capture is explicitly enabled."""
    if not enabled() or not _is_near_zero(usage):
        return
    with _LOCK:
        requests = list(_LAST)
    if not requests:
        return
    try:
        root = get_hermes_home() / "observability" / "cache_lowhit"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for path in sorted(root.glob("*.json"))[: max(0, len(list(root.glob("*.json"))) + 1 - MAX_DUMPS)]:
            path.unlink(missing_ok=True)
        atomic_json_write(
            root / f"{time.time_ns()}.json",
            {
                "schema": "hermes.cache_lowhit.v2",
                "cache_read_tokens": max(0, int(usage.cache_read_tokens)),
                "prompt_tokens": max(0, int(usage.prompt_tokens)),
                "requests": requests,
            },
        )
    except Exception:
        return
