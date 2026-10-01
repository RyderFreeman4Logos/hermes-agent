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
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
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
        assert server._admit_prompt_turn("owner-ui", session, "successor", None, None, None, None) == (
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


def test_idle_completion_claim_keeps_receipt_pending_when_prompt_submit_claims_after_status(
    monkeypatch,
):
    """The real submit claim between status output and idle receipt wins exactly once."""
    event = _completion("proc_idle_claim_race")
    session = _session(running=False)
    submitted: list[tuple] = []

    def emit(kind, sid, payload=None):
        if kind == "status.update":
            # This is the same history-lock transaction used by prompt.submit,
            # after its normal busy observation has seen the session idle.
            err, _fields = server._lock_in_submit_turn(
                "user-rid", sid, session, "actual user prompt", {}, False, None, None, None
            )
            assert err is None

    monkeypatch.setattr(server, "_emit", emit)
    monkeypatch.setattr(server, "_idle_completion_turn", lambda *args: submitted.append(args) or True)

    server._deliver_completion_notifications("owner-ui", session, [event], set())

    assert submitted == []
    assert session["running"] is True
    assert session.get("_completion_active_receipt") is None
    assert [item["session_id"] for item in session["_completion_pending"]] == ["proc_idle_claim_race"]


def test_live_poller_prompt_submit_claim_keeps_completion_receipt_unconsumed(
    monkeypatch,
):
    """A real prompt.submit admission wins the paused live-poller idle claim.

    The transport pause is deliberately at the status emission boundary.  The
    JSON-RPC handler therefore performs its normal history-lock admission
    before the poller reaches the final ownership transaction.
    """
    event = _completion("proc_live_poller_prompt_claim")
    _clear_ids(event["session_id"])
    session = _session(running=False, agent=_bare_agent(), agent_ready=threading.Event())
    submitted: list[dict] = []
    completion_submits: list[dict] = []
    sid = "owner-live-poller"

    class _OnePoll:
        checks = 0

        def is_set(self):
            self.checks += 1
            return self.checks > 1

    def emit(kind, emitted_sid, _payload=None):
        if kind != "status.update":
            return
        response = server.handle_request({
            "id": "real-user-admission",
            "method": "prompt.submit",
            "params": {"session_id": emitted_sid, "text": "actual user prompt"},
        })
        assert response["result"]["status"] == "streaming"

    try:
        with _isolated_queue(monkeypatch) as isolated:
            isolated.put(event)
            server._sessions[sid] = session
            monkeypatch.setattr(server, "_emit", emit)
            monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_a: None)
            monkeypatch.setattr(server, "_persist_session_row_for_submit", lambda *_a: None)
            monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a: True)
            monkeypatch.setattr(server, "_run_after_agent_ready", lambda *_a: submitted.append(dict(session)))
            monkeypatch.setattr(
                server,
                "_run_prompt_submit",
                lambda *_a, **kwargs: completion_submits.append(dict(kwargs)) or True,
            )
            monkeypatch.setattr(server.threading, "Thread", _InlineThread)

            server._notification_poller_loop(_OnePoll(), sid, session)

            assert len(submitted) == 1
            assert completion_submits == []
            assert session["running"] is True
            assert session.get("_completion_active_receipt") is None
            held = list(session.get("_completion_pending") or []) + list(
                session.get("_completion_transfer") or [])
            assert [item["session_id"] for item in held] == [event["session_id"]]
            assert process_registry.is_completion_consumed(event["session_id"]) is False
            assert _queued_ids(isolated) == []
    finally:
        server._sessions.pop(sid, None)
        _clear_ids(event["session_id"])


def test_public_user_core_row_does_not_ack_waiting_receipt_on_stop(monkeypatch, tmp_path):
    """A completed public user turn cannot consume its waiting completion receipt."""
    event = _completion("proc_user_core_stop_reclaim")
    _clear_ids(event["session_id"])

    class Agent:
        model = "test-model"
        provider = "test-provider"
        session_id = "owner-session"

        def clear_interrupt(self):
            return None

        def run_conversation(self, prompt, **_kwargs):
            return {"final_response": "user response", "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "user response"},
            ]}

    sid = "owner-user-core-stop"
    session = _session(agent=Agent(), running=False, agent_ready=threading.Event())
    session["agent_ready"].set()
    try:
        with _isolated_queue(monkeypatch) as isolated:
            server._sessions[sid] = session
            _patch_inline_turn(monkeypatch, tmp_path)
            monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a: None)
            monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a: None)
            monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a: None)
            monkeypatch.setattr(server, "_wire_callbacks", lambda *_a: None)
            monkeypatch.setattr(server, "_set_session_context", lambda *_a, **_k: [])
            monkeypatch.setattr(server, "_clear_session_context", lambda *_a: None)
            monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
            monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a: False)
            response = server.handle_request({"id": "user-core", "method": "prompt.submit", "params": {"session_id": sid, "text": "real user"}})
            assert response["result"]["status"] == "streaming"
            assert [row["role"] for row in session["history"]] == ["user", "assistant"]
            session["_completion_active_receipt"] = {"events": [event]}
            session.update(_closing=True, _finalized=True)
            stop = threading.Event(); stop.set()
            server._notification_poller_loop(stop, sid, session)
            assert process_registry.is_completion_consumed(event["session_id"]) is False
            assert _queued_ids(isolated) == [event["session_id"]]
    finally:
        server._sessions.pop(sid, None)
        _clear_ids(event["session_id"])


def test_idle_flush_keeps_suffix_pending_until_real_noncompletion_barrier_starts(monkeypatch):
    """C1/W/C2 must not merge C2 into C1 while W has not claimed its route."""
    c1, c2 = _completion("proc_barrier_first"), _completion("proc_barrier_later")
    session = _session(
        running=False,
        _completion_transfer=[c1],
        _completion_pending=[c2],
        _completion_transfer_barrier={"type": "watch_match", "session_id": "watch-boundary"},
    )
    reservations: list[list[str]] = []

    def enqueue(_session, _text, _transport, **kwargs):
        reservations.append([event["session_id"] for event in kwargs["completion_events"]])

    monkeypatch.setattr(server, "_enqueue_prompt", enqueue)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)

    server._flush_pending_completions_if_idle("owner-ui", session, set())

    assert reservations == [["proc_barrier_first"]]
    assert [event["session_id"] for event in session["_completion_pending"]] == ["proc_barrier_later"]


def test_structured_queued_receipt_uses_local_ingestion_when_compute_host_is_active(monkeypatch):
    """A compute-host flag cannot strand a local receipt on an incompatible bridge."""
    event = _completion("proc_local_receipt")
    session = _session(
        running=False,
        queued_prompt={
            "text": "completion text", "transport": None,
            "structured_completion": True, "completion_events": [event],
        },
        _compute_host_active=True,
    )
    local: list[dict] = []

    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(server, "_submit_prompt_to_compute_host", lambda *_a, **_k: pytest.fail("receipt reached host"))

    def submit(*_args, **kwargs):
        local.append(kwargs["completion_receipt"])
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    assert server._drain_queued_prompt("rid", "owner-ui", session)
    assert local and local[0] is session["_completion_active_receipt"]


def test_receipt_callback_binds_to_agent_selected_by_preparation(monkeypatch):
    """Capability preparation may replace the agent; only that final agent may consume."""
    original, replacement = types.SimpleNamespace(), types.SimpleNamespace()
    event = _completion("proc_rebuilt_agent")
    receipt = {"events": [event]}
    session = _session(agent=original, running=True)
    session["_completion_active_receipt"] = receipt
    consumed: list[bool] = []

    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_admit_prompt_turn", lambda *_a: ([], original))
    monkeypatch.setattr(server, "_record_turn_marker", lambda *_a, **_k: "marker")
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_finish_turn", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_retire_turn_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_clear_inflight_turn", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_run_post_turn_followups", lambda *_a, **_k: None)

    def prepare(_sid, current, st, *_args, **_kwargs):
        current["agent"] = replacement
        st.agent = replacement
        return "completion", "completion", 80, None

    def invoke(_sid, _session, st, *_args):
        consumed.append(st.agent._completion_queue_ingest())
        st.result = {"final_response": "done", "messages": []}

    monkeypatch.setattr(server, "_prepare_turn_input", prepare)
    monkeypatch.setattr(server, "_invoke_agent", invoke)
    monkeypatch.setattr(server, "_absorb_turn_result", lambda *_a: None)
    monkeypatch.setattr(server, "_complete_turn_payload", lambda *_a: ({"text": "done"}, "done", "complete"))
    monkeypatch.setattr(server, "_goal_followup_after_turn", lambda *_a: None)
    monkeypatch.setattr(server, "_after_complete_turn", lambda *_a: None)
    monkeypatch.setattr(server, "_publish_session_control_snapshot", lambda *_a, **_k: None)

    assert server._run_prompt_submit("rid", "owner-ui", session, "completion", completion_receipt=receipt)
    assert consumed == [True]
    assert session.get("_completion_active_receipt") is None
    assert getattr(original, "_completion_queue_ingest", None) is None
    assert getattr(replacement, "_completion_queue_ingest", None) is None


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
