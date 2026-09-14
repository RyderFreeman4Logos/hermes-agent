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
            agent.clear_interrupt()
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
