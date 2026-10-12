"""message.complete carries frozen elapsed time and the turn's cache aggregate."""

import contextlib
from types import SimpleNamespace

import tui_gateway.server as srv


def test_complete_payload_freezes_elapsed_and_omits_incomplete_cache(monkeypatch):
    monkeypatch.setattr(srv, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(srv, "_clear_inflight_turn", lambda _session: None)
    monkeypatch.setattr(srv.time, "monotonic", lambda: 10.25)
    agent = SimpleNamespace(
        _reply_usage={"prompt_tokens": 1000},
        _reply_usage_complete=False,
        _last_turn_usage={"cache_read_tokens": 800, "prompt_tokens": 1000},
        _session_title_hint="Scratch",
        model="m",
    )
    session = {
        "pending_title": None, "session_key": "k", "history_lock": contextlib.nullcontext(), "agent": agent,
    }
    st = SimpleNamespace(
        result={"final_response": "kept partial", "interrupted": True},
        agent=agent, terminal_callback=None, receipt_committed=True, receipt_attempted=False,
        marker_key="", error_retained=False, error_detail="", prompt_text="ping", started_monotonic=6.0,
    )
    payload, _, status = srv._complete_turn_payload(session, st, None, 80)
    assert status == "interrupted"
    assert payload["text"] == "kept partial"
    assert payload["usage"]["reply_elapsed_s"] == 4.25
    assert "reply_cache_read" not in payload["usage"]
    assert "reply_prompt_tokens" not in payload["usage"]
    frame = srv._event_frame("message.complete", "sid", payload)
    assert frame["params"]["payload"]["usage"]["reply_elapsed_s"] == 4.25


def test_get_usage_reply_counts_are_turn_aggregate_not_session():
    """Reply footer uses this turn's requests. prompt_tokens already includes cache reads."""

    class _TurnAgent:
        model = "x"
        session_prompt_tokens = 10_000
        session_cache_read_tokens = 9_000
        _reply_usage = {"cache_read_tokens": 800, "prompt_tokens": 1000}
        _reply_usage_complete = True

    usage = srv._get_usage(_TurnAgent())
    assert usage["reply_cache_read"] == 800
    assert usage["reply_prompt_tokens"] == 1000
    assert usage["cache_hit_pct"] == 90  # session bar stays a separate number


def test_get_usage_omits_reply_counts_without_a_usage_record():
    class _NoUsage:
        model = "x"
        session_prompt_tokens = 100
        session_cache_read_tokens = 0

    usage = srv._get_usage(_NoUsage())
    assert "reply_cache_read" not in usage
    assert "reply_prompt_tokens" not in usage
