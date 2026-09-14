"""Completion ownership regressions at real TUI admission and steer-drain boundaries."""
from __future__ import annotations

import contextlib
import queue
import threading
import time
import types
from unittest.mock import patch

import pytest

from hermes_cli.active_sessions import active_session_registry_snapshot
from run_agent import AIAgent
from tools import async_delegation
from tools.process_registry import process_registry
from tui_gateway import server


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
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


def _completion(session_id: str, *, output: str | None = None) -> dict:
    return {
        "type": "completion",
        "session_id": session_id,
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
    yield isolated


def _queued_ids(items: queue.Queue) -> list[str]:
    return [item.get("session_id", "") for item in list(items.queue)]


def _clear_ids(*session_ids: str) -> None:
    process_registry._completion_consumed.difference_update(session_ids)


def _patch_inline_turn(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})


def _durable_delegation(delegation_id: str) -> dict:
    event = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "owner-session",
        "origin_ui_session_id": "owner-ui",
        "status": "completed",
    }
    async_delegation._persist_dispatch({
        "delegation_id": delegation_id,
        "session_key": "owner-session",
        "origin_ui_session_id": "owner-ui",
        "dispatched_at": time.time(),
    })
    async_delegation._persist_completion(event, {"status": "completed"})
    return event


def test_real_refusal_does_not_release_successor_turn(monkeypatch, tmp_path):
    profile_home = tmp_path / "profile"
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_concurrent_sessions": 1})
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    blocker, refusal = server._claim_active_session_slot(
        "blocker", live_session_id="blocker-ui", profile_home=profile_home
    )
    assert blocker is not None and refusal is None

    agent = _bare_agent()
    session = _session(
        agent=agent,
        active_session_lease=None,
        profile_home=profile_home,
        source="tui",
    )
    real_submit = server._run_prompt_submit
    refused_returned = threading.Event()
    resume_caller = threading.Event()
    outcome: dict[str, object] = {}

    def paused_submit(*args, **kwargs):
        accepted = real_submit(*args, **kwargs)
        assert accepted is False
        refused_returned.set()
        assert resume_caller.wait(2), "notification caller release timed out"
        return accepted

    def submit_notification():
        try:
            outcome["accepted"] = server._notif_submit(
                "rid", "owner-ui", session, "completion", "test notification"
            )
        except BaseException as exc:
            outcome["error"] = exc

    monkeypatch.setattr(server, "_run_prompt_submit", paused_submit)
    worker = threading.Thread(target=submit_notification)
    try:
        worker.start()
        assert refused_returned.wait(2), "real capacity refusal did not return"
        assert session["running"] is False
        blocker.release()
        session["running"] = True
        assert server._admit_prompt_turn("owner-ui", session, "successor", None, None) == (
            [], agent
        )
        successor_lease = session["active_session_lease"]
        assert len(active_session_registry_snapshot(registry_home=profile_home)) == 1

        resume_caller.set()
        worker.join(2)
        assert not worker.is_alive()
        assert outcome == {"accepted": False}
        assert session["running"] is True
        assert session["active_session_lease"] is successor_lease
        assert len(active_session_registry_snapshot(registry_home=profile_home)) == 1

        server._release_active_session_slot(session)
        blocker, refusal = server._claim_active_session_slot(
            "blocker-emit", live_session_id="blocker-emit-ui", profile_home=profile_home
        )
        assert blocker is not None and refusal is None
        emitted: list[str] = []

        def fail_refusal_emit(kind, *_args, **_kwargs):
            emitted.append(kind)
            if kind == "error":
                raise RuntimeError("emit failed")

        exception_session = _session(
            agent=_bare_agent(), active_session_lease=None, profile_home=profile_home,
            source="tui",
        )
        monkeypatch.setattr(server, "_run_prompt_submit", real_submit)
        monkeypatch.setattr(server, "_emit", fail_refusal_emit)
        with pytest.raises(RuntimeError, match="emit failed"):
            server._notif_submit(
                "rid", "emit-ui", exception_session, "completion", "test notification"
            )
        assert emitted == ["message.start", "error"]
        assert exception_session["running"] is False
        assert exception_session.get("active_session_lease") is None
        assert len(active_session_registry_snapshot(registry_home=profile_home)) == 1
    finally:
        resume_caller.set()
        worker.join(2)
        server._release_active_session_slot(session)
        blocker.release()


@pytest.mark.parametrize("caller", ["idle", "event", "batch"])
@pytest.mark.parametrize("accepted", [False, True])
def test_notification_claim_settlement_follows_submit_outcome(
    monkeypatch, tmp_path, caller: str, accepted: bool
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    delegation_id = f"delegation-{caller}-{accepted}"
    event = _durable_delegation(delegation_id)
    session = _session(running=caller != "batch")

    def submit(*args, **_kwargs):
        if not accepted:
            args[2]["running"] = False
        return accepted

    monkeypatch.setattr(server, "_notif_submit", submit)
    if caller == "idle":
        assert server._idle_completion_turn("owner-ui", session, event, "done") is accepted
    elif caller == "event":
        server._notif_dispatch_event("owner-ui", session, event, "done")
    else:
        server._notif_dispatch_completions(
            "owner-ui", session, [(event, "done")], process_registry, []
        )

    durable = async_delegation.get_durable_delegation(delegation_id)
    assert durable is not None
    assert durable["delivery_state"] == ("delivered" if accepted else "pending")
    if not accepted:
        retry_claim = async_delegation.claim_event_delivery(event, "retry-proof")
        assert retry_claim
        async_delegation.release_event_delivery(event, retry_claim)


def test_refused_idle_admission_requeues_once_and_normal_admission_settles_once(
    monkeypatch, tmp_path
):
    refused_id = "proc_refused_idle"
    admitted_id = "proc_admitted_idle"
    _clear_ids(refused_id, admitted_id)
    try:
        with _isolated_queue(monkeypatch) as isolated:
            refused = _session(
                running=False,
                _closing=True,
                _finalized=True,
                _completion_pending=[_completion(refused_id)],
            )
            stop = threading.Event()
            stop.set()
            server._notification_poller_loop(stop, "refused-ui", refused)

            assert process_registry.is_completion_consumed(refused_id) is False
            assert _queued_ids(isolated) == [refused_id]
            assert refused.get("_completion_pending") == []

            isolated.get_nowait()
            _patch_inline_turn(monkeypatch, tmp_path)
            agent = types.SimpleNamespace(
                session_id="owner-agent",
                run_conversation=lambda *_a, **_k: {"final_response": "done"},
                clear_interrupt=lambda: None,
            )
            admitted = _session(agent=agent, running=False)
            settlements: list[list[str]] = []
            real_settle = server._mark_completion_events_consumed

            def settle(events):
                settlements.append([event["session_id"] for event in events])
                real_settle(events)

            monkeypatch.setattr(server, "_mark_completion_events_consumed", settle)
            server._deliver_completion_notifications(
                "admitted-ui", admitted, [_completion(admitted_id)], set()
            )

            # Admission and the synthetic thread return are still before the
            # real turn-context row. The receipt therefore stays recoverable
            # until that insertion callback commits it.
            assert process_registry.is_completion_consumed(admitted_id) is False
            assert settlements == []
            assert [event["session_id"] for event in admitted["_completion_pending"]] == [admitted_id]
            assert admitted["running"] is False
            admitted.update(_closing=True, _finalized=True)
            server._notification_poller_loop(stop, "admitted-ui", admitted)
            assert _queued_ids(isolated) == [admitted_id]
    finally:
        _clear_ids(refused_id, admitted_id)


class _BlockingMessages(list):
    def __init__(self, rows):
        super().__init__(rows)
        self.user_appended = threading.Event()
        self.release = threading.Event()

    def append(self, item):
        super().append(item)
        if isinstance(item, dict) and item.get("role") == "user":
            self.user_appended.set()
            assert self.release.wait(2), "user-row release timed out"


def test_ingest_uses_structured_identity_not_formatted_text(monkeypatch):
    first_id = "proc_identity_a"
    staged_id = "proc_identity_b"
    restaged_id = "proc_identity_c"
    _clear_ids(first_id, staged_id, restaged_id)
    errors: list[BaseException] = []
    try:
        with _isolated_queue(monkeypatch) as isolated:
            agent = _bare_agent()
            session = _session(agent=agent)
            assert server._deliver_completions_via_steer(
                "owner-ui", session, [_completion(first_id)], set()
            )
            assert agent.steer(f"operator note also mentions {first_id}")
            messages = _BlockingMessages(
                [{"role": "tool", "content": "ok", "tool_call_id": "tool-1"}]
            )

            def apply():
                try:
                    agent._apply_pending_steer_to_tool_results(messages, 1)
                except BaseException as exc:
                    errors.append(exc)

            apply_thread = threading.Thread(target=apply, daemon=True)
            apply_thread.start()
            assert messages.user_appended.wait(2), "first completion was not ingested"
            messages.release.set()
            apply_thread.join(2)
            assert not apply_thread.is_alive()
            assert errors == []
            assert process_registry.is_completion_consumed(first_id) is True
            assert messages[-1]["content"].count(first_id) >= 2

            assert server._deliver_completions_via_steer(
                "owner-ui",
                session,
                [_completion(staged_id, output=f"ordinary output mentions {first_id} token")],
                set(),
            )
            process_registry._completion_consumed.add(staged_id)
            assert server._deliver_completions_via_steer(
                "owner-ui", session, [_completion(restaged_id)], set()
            )
            monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a, **_k: False)
            server._run_post_turn_followups("rid", "owner-ui", session, {}, None)

            payload = session["queued_prompt"]["text"]
            assert staged_id not in payload
            assert restaged_id in payload
            assert process_registry.is_completion_consumed(staged_id) is True
            # Queue staging preserves the structured receipt; only the core
            # insertion tests commit it. A lifecycle reclaim returns it once.
            assert process_registry.is_completion_consumed(restaged_id) is False
            session.update(_closing=True, _finalized=True)
            stop = threading.Event()
            stop.set()
            server._notification_poller_loop(stop, "owner-ui", session)
            assert _queued_ids(isolated) == [restaged_id]
    finally:
        _clear_ids(first_id, staged_id, restaged_id)
