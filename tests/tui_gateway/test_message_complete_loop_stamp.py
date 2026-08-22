"""message.complete must stamp loop end time + first-call cache_info (#126)."""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server


class _ImmediateThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, **_kw):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _StoppedTicker:
    def join(self):
        pass


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


@pytest.fixture()
def frames(monkeypatch):
    captured: list = []
    monkeypatch.setattr(server, "write_json", captured.append)
    return captured


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server, "_start_usage_ticker", lambda *_args: (threading.Event(), _StoppedTicker())
    )
    monkeypatch.setattr(server, "render_message", lambda *_args: "")
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: False)


def _complete_payloads(frames):
    out = []
    for frame in frames:
        params = frame.get("params") or {}
        if params.get("type") == "message.complete":
            out.append(params.get("payload") or {})
    return out


def test_emit_stamps_completed_at_on_message_complete(frames):
    before = server.time.time()
    server._emit("message.complete", "sid", {"text": "done", "status": "complete"})
    after = server.time.time()

    payload = _complete_payloads(frames)[0]
    stamp = payload["completed_at"]
    assert before <= stamp <= after
    assert frames[0]["params"]["payload"]["text"] == "done"


def test_emit_does_not_stamp_non_complete_events(frames):
    server._emit("status.update", "sid", {"kind": "cache_hit", "text": "cache 95%"})
    params = frames[0]["params"]
    assert params["type"] == "status.update"
    assert "completed_at" not in (params.get("payload") or {})


@pytest.mark.parametrize(
    ("state", "pct", "read", "expected_state", "expected_pct"),
    [
        ("hit", 95, 1_900, "hit", 95),
        ("cold_write", 0, 0, "cold_write", 0),
        ("no_field", 0, 0, "unavailable", 0),
    ],
)
def test_message_complete_stamps_first_call_cache_info(
    frames, turn_env, state, pct, read, expected_state, expected_pct
):
    class _Agent:
        def run_conversation(self, _prompt, *, turn_origin="user", **_kwargs):
            record = {
                "request_index": 1,
                "state": state,
                "pct": pct if state == "hit" else None,
                "timestamp": 1.0,
                "turn_origin": turn_origin,
            }
            callback = getattr(self, "_tui_cache_callback")
            callback(state, pct, read, 2_000, record)
            callback("hit", 99, 1_980, 2_000, {**record, "request_index": 2, "pct": 99})
            return {"final_response": "reply", "messages": []}

        def clear_interrupt(self):
            return None

    agent = _Agent()
    session = _session(agent=agent, running=True)
    sid = f"loop-stamp-{state}"
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        server._run_prompt_submit("rid", sid, session, "wake")
    finally:
        server._sessions.pop(sid, None)

    payload = _complete_payloads(frames)[0]
    assert isinstance(payload["completed_at"], (int, float))
    assert payload["cache_info"]["state"] == expected_state
    assert payload["cache_info"]["pct"] == expected_pct


def test_message_complete_unavailable_when_no_provider_usage(frames, turn_env):
    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=lambda *a, **k: {"final_response": "reply", "messages": []},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)
    sid = "loop-stamp-no-usage"
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        server._run_prompt_submit("rid", sid, session, "wake")
    finally:
        server._sessions.pop(sid, None)

    payload = _complete_payloads(frames)[0]
    assert isinstance(payload["completed_at"], (int, float))
    assert payload["cache_info"]["state"] == "unavailable"


def test_message_complete_does_not_reuse_prior_loop_cache_info(frames, turn_env):
    class _Agent:
        def __init__(self):
            self._calls = 0

        def run_conversation(self, _prompt, *, turn_origin="user", **_kwargs):
            self._calls += 1
            if self._calls == 1:
                getattr(self, "_tui_cache_callback")(
                    "hit",
                    95,
                    1_900,
                    2_000,
                    {
                        "request_index": 1,
                        "state": "hit",
                        "pct": 95,
                        "timestamp": 1.0,
                        "turn_origin": turn_origin,
                    },
                )
            return {"final_response": "reply", "messages": []}

        def clear_interrupt(self):
            return None

    agent = _Agent()
    session = _session(agent=agent, running=True)
    sid = "loop-stamp-no-reuse"
    server._sessions[sid] = session
    try:
        server._attach_tui_cache_callback(agent, sid)
        server._run_prompt_submit("rid-1", sid, session, "first")
        server._run_prompt_submit("rid-2", sid, session, "second")
    finally:
        server._sessions.pop(sid, None)

    payloads = _complete_payloads(frames)
    assert payloads[0]["cache_info"]["state"] == "hit"
    assert payloads[1]["cache_info"]["state"] == "unavailable"
