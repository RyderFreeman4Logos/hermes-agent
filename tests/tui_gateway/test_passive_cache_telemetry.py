from __future__ import annotations

import types

from tui_gateway import server


def test_first_real_response_publishes_passive_cache_telemetry_without_rearming(
    monkeypatch,
):
    sid = "passive-cache-sid"
    agent = types.SimpleNamespace(
        session_id="provider-session",
        _tui_first_provider_response_record_enabled=True,
        _tui_first_provider_response_recorded=False,
        context_compressor=None,
    )
    session = {"agent": agent}
    emitted: list[tuple[str, str, dict]] = []

    def forbidden_rearm(*_args, **_kwargs):
        raise AssertionError("passive cache telemetry must not re-arm active warming")

    monkeypatch.setattr(server, "_arm_tui_cache_warm", forbidden_rearm, raising=False)
    monkeypatch.setattr(server, "_emit", lambda event, key, payload: emitted.append((event, key, payload)))
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        agent._tui_cache_callback(
            "hit",
            90,
            900,
            1_000,
            {"state": "hit", "pct": 90, "read_tokens": 900, "prompt_tokens": 1_000},
        )
    finally:
        server._sessions.pop(sid, None)

    assert session["first_provider_response"]["state"] == "hit"
    assert session["first_provider_response"]["pct"] == 90
    assert not any(key.startswith("_cache_warm") for key in session)
    assert emitted[0][0:2] == ("status.update", sid)
    assert emitted[0][2]["cache_record"]["read_tokens"] == 900
