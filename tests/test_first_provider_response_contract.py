import threading
import types

import pytest

from tools import async_delegation as ad
from tui_gateway import server


@pytest.mark.parametrize("origin", ["user", "background_completion", "subagent_result"])
def test_first_provider_response_record_is_content_free_and_secondary_is_retained(origin):
    from agent.conversation_loop import _ingest_successful_provider_usage

    agent = types.SimpleNamespace(
        _first_turn_usage=None,
        _last_turn_usage=None,
        _cache_turn_origin=origin,
        _tui_cache_owner_session="root-session",
    )
    emitted = []
    agent._tui_cache_callback = lambda *args: emitted.append(args)

    _ingest_successful_provider_usage(
        agent,
        {
            "cache_telemetry_present": False,
            "prompt_tokens": 2_000,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "content": "secret-sentinel-must-not-persist",
        },
        first_call=True,
    )
    _ingest_successful_provider_usage(
        agent,
        {
            "cache_telemetry_present": True,
            "prompt_tokens": 2_000,
            "cache_read_tokens": 1_900,
            "cache_write_tokens": 0,
        },
        first_call=False,
    )

    first = agent._first_provider_response
    assert first == {
        "owner_session_id": "root-session",
        "turn_origin": origin,
        "request_index": 1,
        "state": "no_field",
        "timestamp": first["timestamp"],
        "prompt_tokens": 2_000,
    }
    assert first["timestamp"] > 0
    assert "secret-sentinel-must-not-persist" not in repr(first)
    assert agent._provider_response_records[1]["request_index"] == 2
    assert agent._provider_response_records[1]["state"] == "hit"
    assert agent._provider_response_records[1]["timestamp"] >= first["timestamp"]
    assert agent._first_provider_response is first
    assert len(emitted) == 2
    assert emitted[0][0:2] == ("no_field", None)


def test_provider_response_distinguishes_reported_miss_from_unknown():
    from agent.conversation_loop import _ingest_successful_provider_usage

    def record(usage):
        agent = types.SimpleNamespace(
            _first_turn_usage=None,
            _last_turn_usage=None,
            _cache_turn_origin="user",
            _tui_cache_owner_session="root-session",
        )
        _ingest_successful_provider_usage(agent, usage, first_call=True)
        return agent._first_provider_response

    assert record({"cache_telemetry_present": True, "prompt_tokens": 2_000})["state"] == "miss"
    unknown = record({"prompt_tokens": 2_000})
    assert unknown["state"] == "unknown"
    assert "pct" not in unknown


def test_post_compression_starts_a_new_cache_attribution_boundary():
    from agent.conversation_loop import _ingest_successful_provider_usage

    agent = types.SimpleNamespace(
        _first_turn_usage=None,
        _last_turn_usage=None,
        _cache_turn_origin="user",
        _tui_cache_owner_session="root-session",
    )
    _ingest_successful_provider_usage(
        agent, {"cache_telemetry_present": True, "prompt_tokens": 2_000}, first_call=True
    )
    first = agent._first_provider_response
    agent._awaiting_cache_usage_after_compression = True
    _ingest_successful_provider_usage(
        agent,
        {"cache_telemetry_present": True, "prompt_tokens": 2_000, "cache_write_tokens": 2_000},
        first_call=False,
    )

    warmup = agent._cache_attribution_response
    assert agent._first_provider_response is first
    assert warmup["request_index"] == 1
    assert warmup["cache_attribution"] == "post_compression"
    assert warmup["state"] == "cold_write"


def test_tui_cache_callback_persists_session_record_without_content(monkeypatch):
    class Agent:
        _first_turn_usage = {"cache_telemetry_present": False, "prompt_tokens": 2_000}

    sid = "root-session"
    agent = Agent()
    session = {"agent": agent}
    server._sessions[sid] = session
    emitted = []
    record = {
        "owner_session_id": sid,
        "turn_origin": "background_completion",
        "request_index": 1,
        "timestamp": 42.0,
        "state": "no_field",
        "prompt_tokens": 2_000,
    }
    try:
        monkeypatch.setattr(
            server, "_emit", lambda event, event_sid, payload: emitted.append((event, event_sid, payload))
        )
        server._attach_tui_cache_callback(agent, sid)
        agent._tui_cache_callback("no_field", None, 0, 2_000, record)
    finally:
        server._sessions.pop(sid, None)

    assert session["first_provider_response"] == record
    assert session["provider_response_records"] == [record]
    wire_record = emitted[0][2]["cache_record"]
    assert wire_record["owner"] == "tui_gateway"
    assert len(wire_record["session"]) == 64
    assert "owner_session_id" not in wire_record
    assert "content" not in repr(record)
    assert emitted[0][2]["text"] == "cache no telemetry"


@pytest.mark.parametrize(
    ("event_type", "origin"),
    [("completion", "background_completion"), ("async_delegation", "subagent_result")],
)
def test_completion_wakes_are_typed_before_the_provider_turn(monkeypatch, event_type, origin):
    from tools.process_registry import process_registry

    submitted = []
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_compose_completion_batch_prompt", lambda _items: ("wake", False))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, _text, **kwargs: submitted.append(kwargs["turn_origin"]),
    )
    monkeypatch.setattr(ad, "complete_event_delivery", lambda *_args: None)
    monkeypatch.setattr(process_registry, "complete_completion_delivery", lambda *_args: None)

    server._dispatch_completion_batch(
        "sid",
        {"history_lock": threading.Lock()},
        [{"evt": {"type": event_type}, "claim": "claim", "text": "wake"}],
        consumer="test",
    )

    assert submitted == [origin]


def test_child_cache_event_cannot_mutate_the_root_session_record(monkeypatch):
    class Agent:
        _first_turn_usage = {"cache_telemetry_present": True, "prompt_tokens": 2_000}

    sid = "root-session"
    root = Agent()
    child = Agent()
    session = {"agent": root}
    record = {
        "owner_session_id": sid,
        "turn_origin": "subagent_result",
        "request_index": 1,
        "timestamp": 42.0,
        "state": "miss",
        "prompt_tokens": 2_000,
    }
    server._sessions[sid] = session
    try:
        monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
        server._attach_tui_cache_callback(child, sid)
        child._tui_cache_callback("miss", 0, 0, 2_000, record)
    finally:
        server._sessions.pop(sid, None)

    assert "first_provider_response" not in session
    assert "provider_response_records" not in session
