"""Receipt-lifetime regressions: transferred drain vs later clear, no-tool requeue vs reclaim."""
from __future__ import annotations

import contextlib
import queue
import threading
import types
from unittest.mock import patch

from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result
from run_agent import AIAgent
from tools.process_registry import process_registry
from tui_gateway import server


def _completion(session_id: str, *, output: str | None = None) -> dict:
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": "",
        "command": f"echo {session_id}",
        "exit_code": 0,
        "output": output if output is not None else f"out-{session_id}",
    }


def _session(agent=None, **extra) -> dict:
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "owner-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }


def _bare_agent() -> AIAgent:
    with patch("run_agent.AIAgent.__init__", return_value=None):
        agent = AIAgent.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._model_request_active = threading.Event()
    agent._executing_tools = True
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    agent._hard_interrupt_requested = threading.Event()
    agent._interrupt_thread_signal_pending = False
    agent._execution_thread_id = None
    agent.session_id = "owner-agent"
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.client = None
    agent._session_messages = None
    return agent


@contextlib.contextmanager
def _isolated_queue(monkeypatch):
    isolated = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_loop_tick", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_heartbeat_tick", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_notif_poll_kanban", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_poll_bot_live_delivery_once", lambda *_a, **_k: False)
    yield isolated


def _queued_ids(items: queue.Queue) -> list[str]:
    return [item.get("session_id", "") for item in list(items.queue)]


def _clear_ids(*session_ids: str) -> None:
    process_registry._completion_consumed.difference_update(session_ids)


def _dying_reclaim(sid: str, session: dict) -> None:
    session.update(_closing=True, _finalized=True)
    stop = threading.Event()
    stop.set()
    server._notification_poller_loop(stop, sid, session)


def test_clear_later_user_steer_preserves_structured_transfer(monkeypatch):
    event_id = "proc_receipt_a"
    _clear_ids(event_id)
    try:
        with _isolated_queue(monkeypatch) as isolated:
            agent = _bare_agent()
            session = _session(agent=agent)
            assert server._deliver_completions_via_steer(
                "receipt-ui", session, [_completion(event_id)], set()
            )
            assert agent.steer("later user steer") is True
            agent.clear_interrupt(hard_cancel=True)
            assert agent._pending_steer is None
            assert [event["session_id"] for event in session["_completion_transfer"]] == [event_id]

            monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a, **_k: False)
            server._run_post_turn_followups("rid", "receipt-ui", session, {}, None)
            assert process_registry.is_completion_consumed(event_id) is False
            assert event_id in session["queued_prompt"]["text"]
            assert "later user steer" not in session["queued_prompt"]["text"]

            _dying_reclaim("receipt-ui", session)
            assert process_registry.is_completion_consumed(event_id) is False
            assert event_id in _queued_ids(isolated)
    finally:
        _clear_ids(event_id)


def test_cancelled_drain_returns_receipt_to_queue_once(monkeypatch):
    from tui_gateway.session_auto_continue import _reclaim_queued_completion_receipts

    event = _completion("proc_cancelled_claim")
    envelope = {"text": "completion", "transport": None,
                "structured_completion": True, "completion_events": [event]}
    session = _session(running=False, queued_prompt=envelope)
    monkeypatch.setattr(server, "_session_uses_compute_host",
                        lambda _session: session.__setitem__("_queued_prompt_generation", 1) or False)
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("cancelled turn dispatched")))

    assert server._drain_queued_prompt("rid", "cancelled-ui", session)
    assert session["queued_prompt"] is envelope
    _reclaim_queued_completion_receipts(session)
    assert session["_completion_pending"] == [event]
    assert "_completion_active_receipt" not in session


def test_public_session_interrupt_does_not_restore_reclaimed_claim_receipt(monkeypatch, tmp_path):
    from tui_gateway.session_auto_continue import _reclaim_queued_completion_receipts

    event = _completion("proc_public_stop_claim")
    envelope = {"text": "completion", "transport": None,
                "structured_completion": True, "completion_events": [event]}
    first = {"text": "user follow-up", "transport": None}
    second = {"text": "later user follow-up", "transport": None}
    sid = "receipt-public-stop-claim"
    session = _session(running=False, profile_home=tmp_path, queued_prompt=envelope)
    calls = 0

    def stop_after_claim(_session):
        nonlocal calls
        calls += 1
        if calls == 1:
            response = server.handle_request({
                "id": "stop-at-claim", "method": "session.interrupt",
                "params": {"session_id": sid},
            })
            assert response["result"]["status"] == "interrupted"
            session["queued_prompt"] = first
            session["queued_prompts"] = [second]
        return False

    monkeypatch.setattr(server, "_session_uses_compute_host", stop_after_claim)
    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(server, "_resume_wake_after_interrupt", lambda: None)
    monkeypatch.setattr(server, "_retire_turn_marker", lambda *_args: None)
    monkeypatch.setattr(server, "_clear_pending", lambda *_args: None)
    dispatch_calls = []
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda *_args, **_kwargs: dispatch_calls.append(True))
    server._sessions[sid] = session
    try:
        assert server._drain_queued_prompt("rid", sid, session)
        assert not dispatch_calls
        _reclaim_queued_completion_receipts(session)
        assert session["_completion_pending"] == [event]
        assert session["queued_prompt"] is first
        assert session["queued_prompts"] == [second]
        assert "_completion_active_receipt" not in session
    finally:
        server._sessions.pop(sid, None)


def test_agent_reset_and_dying_poller_reclaim_preserve_user_prompt(monkeypatch):
    from tui_gateway.session_auto_continue import _reclaim_queued_completion_receipts

    event = _completion("proc_reset_claim")
    envelope = {"text": "completion", "transport": None,
                "structured_completion": True, "completion_events": [event]}
    followup = {"text": "user after reset", "transport": None}
    sid = "receipt-reset-claim"
    session = _session(running=False, queued_prompt=envelope)
    replacement = types.SimpleNamespace()
    for name, value in {
        "_load_show_reasoning": lambda: False,
        "_load_tool_progress_mode": lambda: "all",
        "_set_session_context": lambda *_args: None,
        "_clear_session_context": lambda *_args: None,
        "_session_source": lambda _session: "tui",
        "_context_cwd_is_launch_artifact": lambda _session: False,
        "_rebuild_session_agent": lambda *_args, **_kwargs: replacement,
        "_session_info": lambda *_args, **_kwargs: {},
        "_emit": lambda *_args, **_kwargs: None,
        "_restart_slash_worker": lambda *_args, **_kwargs: None,
    }.items():
        monkeypatch.setattr(server, name, value)
    calls = 0

    def reset_after_claim(_session):
        nonlocal calls
        calls += 1
        if calls == 1:
            server._reset_session_agent(sid, session)
            session["queued_prompt"] = followup
        return False

    monkeypatch.setattr(server, "_session_uses_compute_host", reset_after_claim)
    dispatch_calls = []
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda *_args, **_kwargs: dispatch_calls.append(True))
    assert server._drain_queued_prompt("rid", sid, session)
    assert not dispatch_calls
    _reclaim_queued_completion_receipts(session)
    assert session["_completion_pending"] == [event]
    assert session["queued_prompt"] is followup
    assert "_completion_active_receipt" not in session

    monkeypatch.undo()
    event = _completion("proc_poller_claim")
    event_id = event["session_id"]
    envelope = {"text": "completion", "transport": None,
                "structured_completion": True, "completion_events": [event]}
    followup = {"text": "user after close", "transport": None}
    sid = "receipt-poller-claim"
    session = _session(running=False, queued_prompt=envelope)
    _clear_ids(event_id)
    calls = 0

    def reap_after_claim(_session):
        nonlocal calls
        calls += 1
        if calls == 1:
            _dying_reclaim(sid, session)
            session["queued_prompt"] = followup
        return False

    monkeypatch.setattr(server, "_session_uses_compute_host", reap_after_claim)
    dispatch_calls = []
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda *_args, **_kwargs: dispatch_calls.append(True))
    try:
        with _isolated_queue(monkeypatch) as isolated:
            assert server._drain_queued_prompt("rid", sid, session)
            assert not dispatch_calls
            assert _queued_ids(isolated) == [event_id]
            assert session["queued_prompt"] is followup
            assert session["_completion_pending"] == []
            assert session.get("_completion_active_receipt") is None
            assert int(session.get("_queued_prompt_generation", 0)) == 0
    finally:
        _clear_ids(event_id)

def test_no_tool_requeue_then_reclaim_retains_each_owner(monkeypatch):
    event_id = "proc_ownerless_e"
    _clear_ids(event_id)
    try:
        with _isolated_queue(monkeypatch) as isolated:
            agent = _bare_agent()
            session = _session(agent=agent)
            assert server._deliver_completions_via_steer(
                "requeue-ui", session, [_completion(event_id)], set()
            )
            assert agent.steer("real user steer") is True
            drained = agent._drain_pending_steer()
            assert drained == "real user steer"
            rows = [{"role": "user", "content": "hello"}]
            _inject_steer_after_newest_tool_result(agent, rows, drained)
            assert agent._pending_steer == "real user steer"
            assert [event["session_id"] for event in session["_completion_transfer"]] == [event_id]

            _dying_reclaim("requeue-ui", session)
            assert process_registry.is_completion_consumed(event_id) is False
            assert _queued_ids(isolated) == [event_id]
            assert [row.get("role") for row in rows] == ["user"]
    finally:
        _clear_ids(event_id)
