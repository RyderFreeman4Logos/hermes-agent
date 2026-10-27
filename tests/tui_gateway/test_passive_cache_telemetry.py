from __future__ import annotations

import types

from agent.turn_usage import _notify_tui_cache
from agent.usage_pricing import CanonicalUsage, format_cache_hit_pct
from tui_gateway import server


def test_cache_hit_pct_keeps_subpercent_and_true_zero_distinct():
    assert format_cache_hit_pct(1, 101) == "<1"
    assert format_cache_hit_pct(1, 100) == "1"
    assert format_cache_hit_pct(0, 100) == "0"


def _live_cache_status_text(monkeypatch, sid: str, usage: CanonicalUsage) -> str:
    agent = types.SimpleNamespace(
        session_id="provider-session",
        _tui_first_provider_response_record_enabled=True,
        _tui_first_provider_response_recorded=False,
        context_compressor=None,
    )
    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(server, "_emit", lambda event, key, payload: emitted.append((event, key, payload)))
    server._sessions[sid] = {"agent": agent}
    try:
        server._attach_tui_cache_callback(agent, sid)
        _notify_tui_cache(agent, usage)
    finally:
        server._sessions.pop(sid, None)
    return emitted[0][2]["text"]


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


def test_live_cache_status_keeps_positive_subpercent_hits(monkeypatch):
    text = _live_cache_status_text(
        monkeypatch,
        "subpercent-cache-sid",
        CanonicalUsage(input_tokens=99_872, cache_read_tokens=128, cache_telemetry="reported"),
    )

    assert text == "cache <1%"


def test_reported_zero_cache_reads_stay_a_miss(monkeypatch):
    text = _live_cache_status_text(
        monkeypatch,
        "zero-cache-sid",
        CanonicalUsage(input_tokens=100_000, cache_telemetry="reported"),
    )

    assert text == "cache MISS"
