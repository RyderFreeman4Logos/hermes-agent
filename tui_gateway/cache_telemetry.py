"""TUI first-wake cache telemetry and bodyless cache-warm arming.

Bodies are rebound onto server.py's globals at install time (method_ctx.bind_module),
so they reference server.py globals bare.
"""

from __future__ import annotations

from typing import Any

from .method_ctx import bind_module

_CACHE_WARM_MIN_HIT_PCT = 80


def _tui_cache_warm_route(agent) -> str:
    provider = str(getattr(agent, "provider", "") or "").strip()
    model = str(getattr(agent, "model", "") or "").strip()
    return f"{provider}:{model}" if provider and model else ""


def _tui_cache_warm_interval_seconds(agent) -> int | None:
    from agent.prompt_caching import effective_cache_ttl
    from hermes_cli.heartbeat import parse_interval

    ttl = getattr(agent, "_cache_ttl", None)
    if ttl is None:
        ttl = ((_load_cfg().get("prompt_caching") or {}).get("cache_ttl"))
    if isinstance(ttl, (int, float)):
        ttl = f"{int(ttl)}s"
    ttl = effective_cache_ttl(
        str(ttl) if ttl is not None else None,
        provider=str(getattr(agent, "provider", "") or ""),
        model=str(getattr(agent, "model", "") or ""),
    )
    interval = parse_interval(ttl)
    return interval if interval and interval > 0 else None


def _cancel_tui_cache_warm(session: dict, *, retain_arm: bool = False) -> None:
    timer = session.pop("_cache_warm_timer", None)
    if timer is not None:
        timer.cancel()
    if retain_arm:
        route = session.get("_cache_warm_route")
        armed_at = session.get("_cache_warm_armed_at")
        interval = session.get("_cache_warm_interval")
        if (
            isinstance(route, str)
            and isinstance(armed_at, (int, float))
            and isinstance(interval, int)
        ):
            session["_cache_warm_previous_arm"] = (route, armed_at, interval)
    else:
        session.pop("_cache_warm_previous_arm", None)
    session.pop("_cache_warm_route", None)
    session.pop("_cache_warm_armed_at", None)
    session.pop("_cache_warm_due_at", None)
    session.pop("_cache_warm_interval", None)
    session_key = str(session.get("session_key") or "")
    if session_key:
        from hermes_cli.heartbeat import HeartbeatManager

        HeartbeatManager(session_key, purpose="cache_warm").clear()


def _tui_cache_warm_due(sid: str, session: dict, agent, route: str) -> None:
    if (
        session.get("agent") is not agent
        or session.get("running")
        or session.get("_cache_warm_route") != route
        or _tui_cache_warm_route(agent) != route
    ):
        return
    session_key = str(session.get("session_key") or "")
    if not session_key:
        return
    from hermes_cli.heartbeat import HeartbeatManager

    if HeartbeatManager(session_key, purpose="cache_warm").due_prompt() is not None:
        session["_cache_warm_due_at"] = time.monotonic()
        try:
            agent._interruptible_api_call({"model": agent.model, "messages": []})
        except Exception:
            logger.debug("bodyless TUI cache warm failed", exc_info=True)


def _arm_tui_cache_warm(sid: str, session: dict, agent, record: dict) -> None:
    if record.get("state") != "hit" or int(record.get("pct") or 0) < _CACHE_WARM_MIN_HIT_PCT:
        return
    route = _tui_cache_warm_route(agent)
    interval = _tui_cache_warm_interval_seconds(agent)
    session_key = str(session.get("session_key") or "")
    if not route or interval is None or not session_key:
        return
    from hermes_cli.heartbeat import HeartbeatManager

    manager = HeartbeatManager(session_key, purpose="cache_warm")
    if manager.state is not None and manager.state.route != route:
        _cancel_tui_cache_warm(session)
        return
    old_timer = session.pop("_cache_warm_timer", None)
    if old_timer is not None:
        old_timer.cancel()
    manager.arm_cache_warm(route, interval)
    session["_cache_warm_route"] = route
    session["_cache_warm_armed_at"] = time.monotonic()
    session["_cache_warm_interval"] = interval
    timer = threading.Timer(interval, _tui_cache_warm_due, args=(sid, session, agent, route))
    timer.daemon = True
    session["_cache_warm_timer"] = timer
    timer.start()


def _tui_cache_warm_miss_classification(session: dict, agent, record: dict) -> str | None:
    if record.get("state") == "hit":
        return None
    route = _tui_cache_warm_route(agent)
    arm = session.get("_cache_warm_previous_arm")
    if not isinstance(arm, tuple):
        arm = (
            session.get("_cache_warm_route"),
            session.get("_cache_warm_armed_at"),
            session.get("_cache_warm_interval"),
        )
    armed_route, armed_at, interval = arm
    if (
        route
        and route == armed_route
        and isinstance(armed_at, (int, float))
        and isinstance(interval, int)
        and 0 <= time.monotonic() - armed_at < interval
    ):
        return "cache_cold_idle_under_ttl"
    return None


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
            classification = _tui_cache_warm_miss_classification(session, agent, cache_record)
            session.pop("_cache_warm_previous_arm", None)
            if classification:
                cache_record["classification"] = classification
            session["first_provider_response"] = cache_record
            payload["cache_record"] = cache_record
            _arm_tui_cache_warm(sid, session, agent, cache_record)
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
