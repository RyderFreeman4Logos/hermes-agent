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


def test_clear_later_steer_before_ack_preserves_transferred_receipt(monkeypatch):
    """F-01: drain A, stage B, clear B before ACK; A's leftover still settles once."""
    first_id = "proc_receipt_a"
    later_id = "proc_receipt_b"
    _clear_ids(first_id, later_id)
    try:
        with _isolated_queue(monkeypatch) as isolated:
            agent = _bare_agent()
            session = _session(agent=agent)
            assert server._deliver_completions_via_steer(
                "receipt-ui", session, [_completion(first_id)], set()
            )
            leftover_a = agent._drain_pending_steer()
            assert leftover_a and first_id in leftover_a
            staged_a = session["_completion_pending"][0]
            assert staged_a["session_id"] == first_id
            assert staged_a.get("_steer_accepted") is True
            assert staged_a.get("_steer_drained") is True

            assert server._deliver_completions_via_steer(
                "receipt-ui", session, [_completion(later_id)], set()
            )
            assert agent._pending_steer and later_id in agent._pending_steer
            assert process_registry.is_completion_consumed(first_id) is False

            agent.clear_interrupt()
            assert agent._pending_steer is None
            by_id = {
                event["session_id"]: event
                for event in (session.get("_completion_pending") or [])
            }
            assert by_id[first_id].get("_steer_accepted") is True
            assert by_id[first_id].get("_steer_drained") is True
            assert process_registry.is_completion_consumed(later_id) is False

            monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a, **_k: False)
            server._run_post_turn_followups(
                "rid", "receipt-ui", session, {"pending_steer": leftover_a}, None
            )
            assert process_registry.is_completion_consumed(first_id) is True
            assert process_registry.is_completion_consumed(later_id) is False
            pending_ids = [
                event["session_id"]
                for event in (session.get("_completion_pending") or [])
            ]
            assert first_id not in pending_ids

            _dying_reclaim("receipt-ui", session)
            assert process_registry.is_completion_consumed(first_id) is True
            assert process_registry.is_completion_consumed(later_id) is False
            assert first_id not in _queued_ids(isolated)
            assert later_id in _queued_ids(isolated)
    finally:
        _clear_ids(first_id, later_id)


def test_no_tool_reclaim_before_requeue_does_not_consume_drained_uningested(monkeypatch):
    """F-02: ownerless spawn_local-style event, drain then dying reclaim before no-tool requeue."""
    event_id = "proc_ownerless_e"
    _clear_ids(event_id)
    try:
        with _isolated_queue(monkeypatch) as isolated:
            agent = _bare_agent()
            session = _session(agent=agent)
            assert server._deliver_completions_via_steer(
                "requeue-ui", session, [_completion(event_id)], set()
            )
            drained = agent._drain_pending_steer()
            assert drained and event_id in drained
            assert session["_completion_pending"][0].get("_steer_drained") is True
            assert agent._pending_steer is None

            _dying_reclaim("requeue-ui", session)
            assert process_registry.is_completion_consumed(event_id) is False
            assert _queued_ids(isolated) == [event_id]

            _inject_steer_after_newest_tool_result(
                agent, [{"role": "user", "content": "hello"}], drained
            )
            assert process_registry.is_completion_consumed(event_id) is False
            assert [row.get("role") for row in [{"role": "user", "content": "hello"}]] == ["user"]
    finally:
        _clear_ids(event_id)


def test_no_tool_requeue_gap_before_callback_does_not_consume_uningested(monkeypatch):
    """F-02: requeue restores live text, then reclaim before the requeue callback commits."""
    event_id = "proc_requeue_gap_e"
    _clear_ids(event_id)
    errors: list[BaseException] = []
    try:
        with _isolated_queue(monkeypatch) as isolated:
            agent = _bare_agent()
            session = _session(agent=agent)
            assert server._deliver_completions_via_steer(
                "requeue-gap-ui", session, [_completion(event_id)], set()
            )
            drained = agent._drain_pending_steer()
            assert drained and event_id in drained
            orig_requeued = agent._completion_steer_requeued
            slot_restored = threading.Event()
            reclaim_finished = threading.Event()

            def _gap_callback():
                slot_restored.set()
                assert reclaim_finished.wait(2), "dying reclaim missed the requeue gap"
                orig_requeued()

            agent._completion_steer_requeued = _gap_callback

            def _reclaim():
                try:
                    assert slot_restored.wait(2), "requeue did not restore the steer slot"
                    assert agent._pending_steer
                    _dying_reclaim("requeue-gap-ui", session)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    reclaim_finished.set()

            reclaim_thread = threading.Thread(target=_reclaim, daemon=True)
            reclaim_thread.start()
            _inject_steer_after_newest_tool_result(
                agent, [{"role": "user", "content": "hello"}], drained
            )
            reclaim_thread.join(2)
            assert not reclaim_thread.is_alive()
            assert errors == []
            assert process_registry.is_completion_consumed(event_id) is False
            assert event_id in _queued_ids(isolated)
    finally:
        _clear_ids(event_id)
