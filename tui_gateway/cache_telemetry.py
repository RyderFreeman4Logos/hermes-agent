"""Passive TUI telemetry for the first real provider response of each wake.

Bodies are rebound onto server.py's globals at install time (method_ctx.bind_module),
so they reference server.py globals bare.
"""

from __future__ import annotations

from typing import Any

from .method_ctx import bind_module


def _attach_tui_cache_callback(agent, sid: str):
    """Publish the first provider cache response for each TUI wake."""
    agent._tui_cache_owner_session = sid

    def emit_cache_state(
        state: str, pct: int, _read: int, _prompt: int, record: dict | None = None
    ) -> None:
        if not getattr(agent, "_tui_first_provider_response_record_enabled", False):
            return
        if getattr(agent, "_tui_first_provider_response_recorded", False):
            return
        agent._tui_first_provider_response_recorded = True
        text = (
            f"cache {pct}%"
            if state == "hit"
            else "cache unavailable"
            if state == "no_field"
            else f"cache {state.upper()}"
        )
        payload: dict[str, Any] = {"kind": "cache_hit", "text": text}
        session = _sessions.get(sid)
        if isinstance(record, dict) and isinstance(session, dict) and session.get("agent") is agent:
            cache_record = {key: value for key, value in record.items() if value is not None}
            if getattr(
                getattr(agent, "context_compressor", None),
                "awaiting_real_usage_after_compression",
                False,
            ):
                cache_record["compression_bound"] = True
            for key, raw in (("read_tokens", _read), ("prompt_tokens", _prompt)):
                if key in cache_record:
                    continue
                try:
                    count = int(raw)
                except (TypeError, ValueError):
                    continue
                if count >= 0:
                    cache_record[key] = count
            cache_record.update(
                owner="tui_gateway",
                session=hashlib.sha256(
                    f"{sid}:{getattr(agent, 'session_id', '')}".encode()
                ).hexdigest(),
            )
            session["first_provider_response"] = cache_record
            payload["cache_record"] = cache_record
        _emit("status.update", sid, payload)

    agent._tui_cache_callback = emit_cache_state
    return agent


def _cache_info_from_first_call(record: Any) -> dict[str, int | str]:
    if not isinstance(record, dict):
        return {"state": "unavailable", "pct": 0}
    state = str(record.get("state") or "unavailable")
    if state == "no_field":
        state = "unavailable"
    try:
        pct = int(record["pct"]) if record.get("pct") is not None else 0
    except (TypeError, ValueError):
        pct = 0
    info: dict[str, int | str] = {"state": state, "pct": pct}
    if record.get("compression_bound") is True:
        info["compression_bound"] = True
    for key in ("read_tokens", "prompt_tokens"):
        value = record.get(key)
        if value is None:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count >= 0:
            info[key] = count
    return info


def _cache_info_from_usage(usage: Any) -> dict[str, int | str]:
    if not isinstance(usage, dict):
        return {"state": "unavailable", "pct": 0}
    try:
        read_tokens = max(0, int(usage.get("cache_read_tokens", 0) or 0))
        write_tokens = max(0, int(usage.get("cache_write_tokens", 0) or 0))
        prompt_tokens = max(0, int(usage.get("prompt_tokens", 0) or 0))
    except (TypeError, ValueError):
        return {"state": "unavailable", "pct": 0}
    telemetry = usage.get("cache_telemetry")
    if telemetry is None and (read_tokens or write_tokens):
        telemetry = "reported"
    if telemetry != "reported":
        return {"state": "unavailable", "pct": 0}
    if read_tokens:
        state = "hit"
        pct = round(100 * read_tokens / prompt_tokens) if prompt_tokens else 0
    elif write_tokens:
        state, pct = "cold_write", 0
    else:
        state, pct = "miss", 0
    return {
        "read_tokens": read_tokens,
        "prompt_tokens": prompt_tokens,
        "pct": pct,
        "state": state,
    }


def _stamp_loop_cache_info(sid: str, payload: dict) -> None:
    session = _sessions.get(sid) or {}
    if "cache_info" not in payload:
        payload["cache_info"] = _cache_info_from_first_call(
            session.get("first_provider_response")
        )
    record = session.get("first_provider_response")
    cache_info = payload.get("cache_info")
    if (
        isinstance(cache_info, dict)
        and isinstance(record, dict)
        and record.get("compression_bound") is True
    ):
        cache_info["compression_bound"] = True


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
