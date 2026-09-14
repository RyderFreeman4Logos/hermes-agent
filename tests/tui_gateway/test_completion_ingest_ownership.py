"""Structured completion ownership at each real ingestion boundary (#315)."""
from __future__ import annotations

import contextlib
import queue
import threading
import time
import types
from unittest.mock import patch

import pytest

from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result
from run_agent import AIAgent
from tools.process_registry import ProcessSession, process_registry
from tui_gateway import server


def _completion(session_id: str, *, session_key: str = "", output: str | None = None) -> dict:
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": session_key,
        "command": "printf complete",
        "exit_code": 0,
        "output": output if output is not None else "done",
    }


def _watch(session_id: str, *, session_key: str) -> dict:
    return {
        "type": "watch_match",
        "session_id": session_id,
        "session_key": session_key,
        "pattern": "ready",
        "output": f"watch-{session_id}",
    }


def _session(agent: AIAgent, **extra) -> dict:
    return {
        "agent": agent,
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


def _agent() -> AIAgent:
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
def _isolated(monkeypatch):
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
    process_registry._poll_observed.difference_update(session_ids)


def _reclaim(sid: str, session: dict) -> None:
    session.update(_closing=True, _finalized=True)
    stop = threading.Event()
    stop.set()
    server._notification_poller_loop(stop, sid, session)


def _tool_messages() -> list[dict]:
    return [
        {"role": "assistant", "tool_calls": [{"id": "call", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call", "content": "done"},
    ]


class _InsertionBoundary:
    def __init__(self):
        self.inserted = threading.Event()
        self.release = threading.Event()

    def hit(self) -> None:
        self.inserted.set()
        assert self.release.wait(2), "insertion boundary release timed out"


class _BoundaryMessages(list):
    def __init__(self, rows, boundary: _InsertionBoundary):
        super().__init__(rows)
        self.boundary = boundary

    def append(self, row):
        super().append(row)
        if row.get("role") == "user":
            self.boundary.hit()

    def insert(self, index, row):
        super().insert(index, row)
        if row.get("role") == "user":
            self.boundary.hit()


class _ObservedRLock:
    def __init__(self, signals: queue.Queue | None = None):
        self._lock = threading.RLock()
        self.attempted = {name: threading.Event() for name in ("ingest", "reclaim")}
        self.signals = signals
        self.hold_name: str | None = None
        self.hold_acquired = threading.Event()
        self.hold_release = threading.Event()
        self.owner: int | None = None
        self.depth = 0

    def __enter__(self):
        name = threading.current_thread().name
        if event := self.attempted.get(name):
            event.set()
            if self.signals is not None:
                self.signals.put(name)
        self._lock.acquire()
        self.owner = threading.get_ident()
        self.depth += 1
        if name == self.hold_name:
            self.hold_acquired.set()
            assert self.hold_release.wait(2), "ownership boundary release timed out"
        return self

    def __exit__(self, *_exc):
        self.depth -= 1
        if self.depth == 0:
            self.owner = None
        self._lock.release()

    def held_by_current_thread(self) -> bool:
        return self.owner == threading.get_ident()


def _buffer_completion(sid: str, session: dict, event: dict, emitted: set) -> None:
    assert server._notif_handle_event(
        sid, session, event, emitted, process_registry,
        lambda item: item.get("session_id", ""), None, owned=True,
    )


@pytest.mark.parametrize("scenario", ["merge", "insert_false", "late", "filter"])
def test_idle_mixed_completion_transaction(monkeypatch, scenario: str):
    first_id = f"proc_mixed_a_{scenario}"
    second_id = f"proc_mixed_b_{scenario}"
    late_id = f"proc_mixed_c_{scenario}"
    _clear_ids(first_id, second_id, late_id)
    errors: list[BaseException] = []
    try:
        with _isolated(monkeypatch):
            agent = _agent()
            session = _session(agent)
            emitted: set = set()
            assert server._deliver_completions_via_steer(
                "owner-ui", session, [_completion(first_id)], emitted
            )
            _buffer_completion("owner-ui", session, _completion(second_id), emitted)
            session["running"] = False
            if scenario == "filter":
                process_registry._completion_consumed.add(first_id)

            ownership = _ObservedRLock()
            session["_completion_ownership_lock"] = ownership
            submitted: list[str] = []
            settlements: list[list[str]] = []
            real_settle = server._mark_completion_events_consumed
            real_format = server._format_completion_batch
            release_snapshot = threading.Event()
            phase = queue.Queue()

            def settle(events):
                settlements.append([event["session_id"] for event in events])
                real_settle(events)

            def submit(_rid, _sid, target_session, text, **_kwargs):
                submitted.append(text)
                return True

            def gated_format(events):
                ids = [event.get("session_id") for event in events]
                if scenario in {"insert_false", "late"} and ids == [first_id, second_id]:
                    phase.put("snapshot")
                    assert release_snapshot.wait(2), "mixed snapshot release timed out"
                return real_format(events)

            def emit(*_args, **_kwargs):
                assert not ownership.held_by_current_thread(), "status emitted under ownership lock"
                acquired = session["history_lock"].acquire(blocking=False)
                assert acquired, "status emitted under history lock"
                session["history_lock"].release()

            monkeypatch.setattr(server, "_mark_completion_events_consumed", settle)
            monkeypatch.setattr(server, "_run_prompt_submit", submit)
            monkeypatch.setattr(server, "_format_completion_batch", gated_format)
            monkeypatch.setattr(server, "_emit", emit)

            def flush():
                try:
                    server._flush_pending_completions_if_idle("owner-ui", session, emitted)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    phase.put("done")

            worker = threading.Thread(target=flush, name="ingest")
            worker.start()
            if scenario in {"insert_false", "late"}:
                reached = phase.get(timeout=2)
                if reached == "snapshot":
                    if scenario == "insert_false":
                        with session["history_lock"]:
                            session["running"] = True
                    else:
                        _buffer_completion(
                            "owner-ui", session, _completion(late_id), emitted
                        )
                    release_snapshot.set()
            worker.join(2)
            release_snapshot.set()
            assert not worker.is_alive()
            assert errors == []

            if scenario == "insert_false":
                assert [event["session_id"] for event in session["_completion_transfer"]] == [
                    first_id, second_id
                ]
                assert session.get("_completion_pending") == []
                assert submitted == [] and settlements == []
                assert process_registry.is_completion_consumed(first_id) is False
                assert process_registry.is_completion_consumed(second_id) is False
                return

            assert len(submitted) == 1
            payload = submitted[0]
            if scenario == "filter":
                assert first_id not in payload and payload.count(second_id) == 1
                assert settlements == [[second_id]]
            else:
                assert payload.index(first_id) < payload.index(second_id)
                assert settlements == [[first_id, second_id]]
            assert session.get("_completion_transfer") == []
            if scenario == "late":
                assert [event["session_id"] for event in session["_completion_pending"]] == [late_id]
                assert process_registry.is_completion_consumed(late_id) is False
            else:
                assert session.get("_completion_pending") == []
            assert process_registry.is_completion_consumed(first_id) is True
            assert process_registry.is_completion_consumed(second_id) is True
    finally:
        _clear_ids(first_id, second_id, late_id)


@pytest.mark.parametrize("winner", ["ingest", "reclaim"])
def test_idle_mixed_ingest_and_reclaim_have_one_winner(monkeypatch, winner: str):
    first_id = f"proc_mixed_winner_a_{winner}"
    second_id = f"proc_mixed_winner_b_{winner}"
    _clear_ids(first_id, second_id)
    errors: list[BaseException] = []
    try:
        with _isolated(monkeypatch) as isolated:
            agent = _agent()
            session = _session(agent)
            emitted: set = set()
            assert server._deliver_completions_via_steer(
                "owner-ui", session, [_completion(first_id)], emitted
            )
            _buffer_completion("owner-ui", session, _completion(second_id), emitted)
            session["running"] = False
            phase = queue.Queue()
            ownership = _ObservedRLock()
            session["_completion_ownership_lock"] = ownership
            boundary = _InsertionBoundary()
            real_enqueue = server._enqueue_prompt
            monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a, **_k: False)
            monkeypatch.setattr(server, "_run_prompt_submit", lambda *_a, **_k: True)

            def gated_enqueue(*args, **kwargs):
                real_enqueue(*args, **kwargs)
                phase.put("inserted")
                boundary.hit()

            monkeypatch.setattr(server, "_enqueue_prompt", gated_enqueue)

            def ingest():
                try:
                    server._flush_pending_completions_if_idle(
                        "owner-ui", session, emitted
                    )
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    phase.put("done")

            reclaim_entered = threading.Event()

            def reclaim():
                try:
                    reclaim_entered.set()
                    _reclaim("owner-ui", session)
                except BaseException as exc:
                    errors.append(exc)

            ingest_thread = threading.Thread(target=ingest, name="ingest")
            reclaim_thread = threading.Thread(target=reclaim, name="reclaim")
            reclaim_started = False
            if winner == "ingest":
                ingest_thread.start()
                first_phase = phase.get(timeout=2)
                if first_phase == "inserted":
                    reclaim_thread.start()
                    reclaim_started = True
                    assert reclaim_entered.wait(2), "reclaim thread did not start"
                    boundary.release.set()
            else:
                ownership.hold_name = "reclaim"
                reclaim_thread.start()
                reclaim_started = True
                assert ownership.hold_acquired.wait(2), "reclaim did not own completion state"
                ingest_thread.start()
                ingest_thread.join(2)
                ownership.hold_release.set()
            ingest_thread.join(2)
            if reclaim_started:
                reclaim_thread.join(2)
            boundary.release.set()
            ownership.hold_release.set()
            assert not ingest_thread.is_alive()
            assert not reclaim_started or not reclaim_thread.is_alive()
            assert errors == []

            if winner == "ingest":
                payload = session["queued_prompt"]["text"]
                assert payload.index(first_id) < payload.index(second_id)
                assert _queued_ids(isolated) == []
                assert process_registry.is_completion_consumed(first_id) is True
                assert process_registry.is_completion_consumed(second_id) is True
            else:
                assert _queued_ids(isolated) == [first_id, second_id]
                assert session.get("queued_prompt") is None
                assert process_registry.is_completion_consumed(first_id) is False
                assert process_registry.is_completion_consumed(second_id) is False
            assert session.get("_completion_transfer") == []
            assert session.get("_completion_pending") == []
    finally:
        _clear_ids(first_id, second_id)


def _payload(
    consumer: str,
    agent: AIAgent,
    session: dict,
    monkeypatch,
    boundary: _InsertionBoundary | None = None,
) -> str:
    if consumer == "tool":
        messages = (
            _BoundaryMessages(_tool_messages(), boundary)
            if boundary is not None
            else _tool_messages()
        )
        agent._apply_pending_steer_to_tool_results(messages, 1)
        return "\n".join(str(row.get("content", "")) for row in messages if row.get("role") == "user")
    if consumer == "pre_api":
        messages = (
            _BoundaryMessages(_tool_messages(), boundary)
            if boundary is not None
            else _tool_messages()
        )
        steer = agent._drain_pending_steer() or ""
        _inject_steer_after_newest_tool_result(agent, messages, steer)
        return "\n".join(str(row.get("content", "")) for row in messages if row.get("role") == "user")
    if boundary is not None:
        enqueue = getattr(server, "_enqueue_prompt")

        def gated_enqueue(*args, **kwargs):
            enqueue(*args, **kwargs)
            boundary.hit()

        monkeypatch.setattr(server, "_enqueue_prompt", gated_enqueue)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a, **_k: False)
    leftover = agent._drain_pending_steer()
    server._run_post_turn_followups("rid", "owner-ui", session, {"pending_steer": leftover}, None)
    queued = [session.get("queued_prompt"), *(session.get("queued_prompts") or [])]
    return "\n".join(str(entry.get("text", "")) for entry in queued if isinstance(entry, dict))


@pytest.mark.parametrize("consumer", ["tool", "pre_api", "post_turn"])
@pytest.mark.parametrize("winner", ["ingest", "reclaim"])
def test_ingest_and_reclaim_have_one_winner(monkeypatch, consumer: str, winner: str):
    event_id = f"proc_owner_{consumer}_{winner}"
    _clear_ids(event_id)
    try:
        with _isolated(monkeypatch) as isolated:
            agent = _agent()
            session = _session(agent)
            assert server._deliver_completions_via_steer(
                "owner-ui", session, [_completion(event_id)], set()
            )
            ownership = _ObservedRLock()
            session["_completion_ownership_lock"] = ownership
            boundary = _InsertionBoundary()
            payloads: list[str] = []
            errors: list[BaseException] = []
            reclaim_started = threading.Event()

            def ingest():
                try:
                    payloads.append(_payload(consumer, agent, session, monkeypatch, boundary))
                except BaseException as exc:
                    errors.append(exc)

            def reclaim():
                try:
                    reclaim_started.set()
                    _reclaim("owner-ui", session)
                except BaseException as exc:
                    errors.append(exc)

            ingest_thread = threading.Thread(target=ingest, name="ingest")
            reclaim_thread = threading.Thread(target=reclaim, name="reclaim")
            if winner == "ingest":
                ingest_thread.start()
                assert boundary.inserted.wait(2), "ingest did not reach insertion"
                reclaim_thread.start()
                assert reclaim_started.wait(2), "reclaim thread did not start"
                if consumer != "post_turn":
                    assert ownership.attempted["reclaim"].wait(2), "reclaim missed ownership lock"
                boundary.release.set()
            else:
                ownership.hold_name = "reclaim"
                reclaim_thread.start()
                assert ownership.hold_acquired.wait(2), "reclaim did not own the boundary"
                ingest_thread.start()
                assert ownership.attempted["ingest"].wait(2), "ingest missed ownership lock"
                ownership.hold_release.set()
            ingest_thread.join(2)
            reclaim_thread.join(2)
            boundary.release.set()
            ownership.hold_release.set()
            assert not ingest_thread.is_alive() and not reclaim_thread.is_alive()
            assert errors == []
            payload = payloads[0]
            if winner == "ingest":
                assert payload.count(event_id) == 1
                assert process_registry.is_completion_consumed(event_id) is True
                assert _queued_ids(isolated).count(event_id) == 0
            else:
                assert event_id not in payload
                assert process_registry.is_completion_consumed(event_id) is False
                assert _queued_ids(isolated).count(event_id) == 1
    finally:
        _clear_ids(event_id)


@pytest.mark.parametrize("consumer", ["tool", "pre_api", "post_turn"])
def test_rejected_closing_admission_retains_exact_event(monkeypatch, consumer: str):
    event_id = f"proc_closing_{consumer}"
    _clear_ids(event_id)
    try:
        with _isolated(monkeypatch) as isolated:
            agent = _agent()
            session = _session(agent)
            assert server._deliver_completions_via_steer(
                "owner-ui", session, [_completion(event_id)], set()
            )
            session["_closing"] = True
            payload = _payload(consumer, agent, session, monkeypatch)
            _reclaim("owner-ui", session)
            assert event_id not in payload
            assert process_registry.is_completion_consumed(event_id) is False
            assert _queued_ids(isolated).count(event_id) == 1
    finally:
        _clear_ids(event_id)


@pytest.mark.parametrize("consumer", ["tool", "pre_api", "post_turn"])
def test_restage_and_redirect_clear_do_not_reuse_old_authority(monkeypatch, consumer: str):
    event_id = f"proc_restage_{consumer}"
    _clear_ids(event_id)
    try:
        with _isolated(monkeypatch) as isolated:
            first = _agent()
            first_session = _session(first)
            assert server._deliver_completions_via_steer(
                "first-ui", first_session, [_completion(event_id)], set()
            )
            first.steer("real user steer")
            first._pending_redirect = "redirect"
            first.clear_interrupt(preserve_redirect=True)
            _reclaim("first-ui", first_session)
            event = isolated.get_nowait()

            second = _agent()
            second_session = _session(second)
            assert server._deliver_completions_via_steer(
                "second-ui", second_session, [event], set()
            )
            payload = _payload(consumer, second, second_session, monkeypatch)
            assert payload.count(event_id) == 1
            assert "real user steer" not in payload
            assert process_registry.is_completion_consumed(event_id) is True
            assert _queued_ids(isolated).count(event_id) == 0
    finally:
        _clear_ids(event_id)


@contextlib.contextmanager
def _finished_process(session_id: str):
    process = ProcessSession(
        id=session_id,
        command=f"printf {session_id}",
        exited=True,
        exit_code=0,
        output_buffer="first\nsecond\nthird\n",
    )
    process._completion_event.set()
    with process_registry._lock:
        previous = process_registry._finished.get(session_id)
        process_registry._finished[session_id] = process
    try:
        yield
    finally:
        with process_registry._lock:
            if previous is None:
                process_registry._finished.pop(session_id, None)
            else:
                process_registry._finished[session_id] = previous
        _clear_ids(session_id)


def _observe_completion(action: str, session_id: str) -> bool:
    if action == "wait":
        assert process_registry.wait(session_id, timeout=1)["status"] == "exited"
    elif action == "read_log":
        assert process_registry.read_log(session_id)["showing"] == "3 lines"
    elif action == "kill_consuming":
        assert process_registry.kill_process(session_id, consume_output=True)["status"] == "already_exited"
    elif action == "partial_log":
        assert process_registry.read_log(session_id, offset=0, limit=1)["showing"] == "1 lines"
    elif action == "poll":
        assert process_registry.poll(session_id)["status"] == "exited"
    else:
        assert action == "kill_nonconsuming"
        assert process_registry.kill_process(session_id, consume_output=False)["status"] == "already_exited"
    return action in {"wait", "read_log", "kill_consuming"}


@pytest.mark.parametrize("consumer", ["tool", "pre_api", "post_turn"])
@pytest.mark.parametrize(
    "action",
    ["wait", "read_log", "kill_consuming", "partial_log", "poll", "kill_nonconsuming"],
)
def test_explicit_consumption_is_reconciled_at_insertion(
    monkeypatch, consumer: str, action: str
):
    first_id = f"proc_e_{consumer}_{action}"
    later_id = f"proc_f_{consumer}_{action}"
    _clear_ids(first_id, later_id)
    try:
        with _isolated(monkeypatch), _finished_process(first_id):
            agent = _agent()
            session = _session(agent)
            assert server._deliver_completions_via_steer(
                "owner-ui",
                session,
                [_completion(first_id), _completion(later_id)],
                set(),
            )
            assert agent.steer("keep this user steer") is True
            consumed = _observe_completion(action, first_id)
            payload = _payload(consumer, agent, session, monkeypatch)
            assert payload.count(first_id) == (0 if consumed else 1)
            assert payload.count(later_id) == 1
            assert payload.count("keep this user steer") == 1
            assert process_registry.is_completion_consumed(first_id) is True
            assert process_registry.is_completion_consumed(later_id) is True
    finally:
        _clear_ids(first_id, later_id)


def test_later_user_clear_cannot_erase_earlier_transfer(monkeypatch):
    event_id = "proc_transfer_before_user_clear"
    _clear_ids(event_id)
    try:
        with _isolated(monkeypatch):
            agent = _agent()
            session = _session(agent)
            assert server._deliver_completions_via_steer(
                "owner-ui", session, [_completion(event_id)], set()
            )
            assert agent.steer("later user steer") is True
            agent.clear_interrupt()
            payload = _payload("tool", agent, session, monkeypatch)
            assert payload.count(event_id) == 1
            assert "later user steer" not in payload
            assert process_registry.is_completion_consumed(event_id) is True
    finally:
        _clear_ids(event_id)


class _BlockingMessages(list):
    def __init__(self, rows):
        super().__init__(rows)
        self.inserted = threading.Event()
        self.release = threading.Event()

    def append(self, row):
        super().append(row)
        if row.get("role") == "user":
            self.inserted.set()
            assert self.release.wait(2), "shutdown overlap release timed out"


def test_actual_tui_shutdown_cannot_requeue_inserted_event(monkeypatch):
    event_id = "proc_shutdown_overlap"
    _clear_ids(event_id)
    errors: list[BaseException] = []
    try:
        with _isolated(monkeypatch) as isolated:
            agent = _agent()
            session = _session(agent)
            assert server._deliver_completions_via_steer(
                "shutdown-ui", session, [_completion(event_id)], set()
            )
            messages = _BlockingMessages(_tool_messages())

            def ingest():
                try:
                    agent._apply_pending_steer_to_tool_results(messages, 1)
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=ingest)
            worker.start()
            assert messages.inserted.wait(1)
            session.update(_closing=True, _finalized=True)
            stop = threading.Event()
            stop.set()
            closer = threading.Thread(
                target=lambda: server._notification_poller_loop(stop, "shutdown-ui", session)
            )
            closer.start()
            time.sleep(0.05)
            messages.release.set()
            worker.join(2)
            closer.join(2)
            assert not worker.is_alive() and not closer.is_alive()
            assert errors == []
            payload = "\n".join(row.get("content", "") for row in messages if row.get("role") == "user")
            assert payload.count(event_id) == 1
            assert process_registry.is_completion_consumed(event_id) is True
            assert _queued_ids(isolated).count(event_id) == 0
    finally:
        _clear_ids(event_id)


def test_successful_snapshot_flushes_idle_completion_despite_foreign_busy_watch(monkeypatch):
    completion_id = "proc_idle_a_completion"
    watch_id = "proc_busy_b_watch"
    _clear_ids(completion_id)
    turns_a: list[str] = []
    turns_b: list[str] = []
    release_completion_snapshot = threading.Event()
    completion_snapshot_claimed = threading.Event()
    foreign_snapshot_handled = threading.Event()
    release_foreign_snapshot = threading.Event()
    busy_watch_claimed = threading.Event()
    stop_a = threading.Event()
    stop_b = threading.Event()
    poller_a = None
    poller_b = None
    try:
        with _isolated(monkeypatch) as isolated:
            idle_agent = _agent()
            busy_agent = _agent()
            idle = _session(idle_agent, session_key="session-a", running=False)
            busy = _session(busy_agent, session_key="session-b", running=True)
            server._sessions.update({"idle-a": idle, "busy-b": busy})

            def submit(_rid, sid, _session, text, **_kwargs):
                (turns_a if sid == "idle-a" else turns_b).append(text)
                return True

            real_handle = getattr(server, "_notif_handle_ready")

            def synchronized_handle(sid, session, events, *args, **kwargs):
                ids = {event.get("session_id") for event in events}
                if sid == "idle-a" and completion_id in ids:
                    completion_snapshot_claimed.set()
                    assert release_completion_snapshot.wait(2)
                result = real_handle(sid, session, events, *args, **kwargs)
                if sid == "idle-a" and watch_id in ids:
                    foreign_snapshot_handled.set()
                    assert release_foreign_snapshot.wait(2)
                elif sid == "busy-b" and watch_id in ids:
                    busy_watch_claimed.set()
                return result

            monkeypatch.setattr(server, "_run_prompt_submit", submit)
            monkeypatch.setattr(server, "_notif_handle_ready", synchronized_handle)
            isolated.put(_completion(completion_id, session_key="session-a"))
            poller_a = threading.Thread(
                target=server._notification_poller_loop,
                args=(stop_a, "idle-a", idle),
            )
            poller_a.start()
            assert completion_snapshot_claimed.wait(2), "idle poller did not claim completion"
            isolated.put(_watch(watch_id, session_key="session-b"))
            release_completion_snapshot.set()
            assert foreign_snapshot_handled.wait(2), "idle poller missed successful foreign snapshot"
            poller_b = threading.Thread(
                target=server._notification_poller_loop,
                args=(stop_b, "busy-b", busy),
            )
            poller_b.start()
            assert busy_watch_claimed.wait(2), "busy poller did not claim its watch"
            delivered_before_stop = bool(turns_a)
            assert delivered_before_stop is True
            assert len(turns_a) == 1
            assert turns_a[0].count(completion_id) == 1
            assert watch_id not in turns_a[0]
            assert turns_b == []
            assert process_registry.is_completion_consumed(completion_id) is True
    finally:
        stop_a.set()
        stop_b.set()
        release_completion_snapshot.set()
        release_foreign_snapshot.set()
        for poller in (poller_a, poller_b):
            if poller is not None:
                poller.join(4)
                assert not poller.is_alive()
        server._sessions.pop("idle-a", None)
        server._sessions.pop("busy-b", None)
        _clear_ids(completion_id)
