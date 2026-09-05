"""Busy process completions coalesce to one ingest batch (#122)."""

from __future__ import annotations

import queue as queue_mod
import threading
import time
import types
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.process_registry import process_registry
from tui_gateway import server


class _SteerAgent:
    """Minimal AIAgent.steer stand-in used by mid-loop ingest tests."""

    def __init__(self):
        self.steers: list[str] = []
        self._pending_steer = None
        self._model_request_active = threading.Event()
        self._executing_tools = False

    def steer(self, text: str) -> bool:
        if not text or not str(text).strip():
            return False
        cleaned = str(text).strip()
        self.steers.append(cleaned)
        self._pending_steer = (
            f"{self._pending_steer}\n{cleaned}" if self._pending_steer else cleaned
        )
        return True


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(),
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
        **extra,
    }


def _completion(session_id: str, exit_code: int, command: str) -> dict:
    return {
        "type": "completion",
        "session_id": session_id,
        "command": command,
        "exit_code": exit_code,
        "output": f"out-{session_id}",
    }


def _routine_child_completion(session_id: str) -> dict:
    return {
        **_completion(session_id, 0, f"echo {session_id}"),
        "started_at": 1.0,
        "completion_reason": "exited",
        "termination_source": "",
        "delegated_child": True,
    }


def _process_status(emitted: list) -> list:
    return [args for args in emitted if args and args[0] == "status.update"]


def test_admitted_completion_steer_does_not_deadlock_next_turn(monkeypatch):
    """A completion accepted by steer must not wedge the next admission."""
    from run_agent import AIAgent

    class _NotStartedThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    agent = AIAgent.__new__(AIAgent)
    agent._pending_steer = None
    setattr(agent, "_pending_steer_lock", threading.Lock())
    agent._pending_redirect = None
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    agent._interrupt_thread_signal_pending = False
    setattr(agent, "_execution_thread_id", None)
    session = _session(agent=agent, running=True)
    event = _completion("proc-next-turn-deadlock", 0, "echo done")
    emitted = []

    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    assert server._deliver_completions_via_steer("sid-next-turn", session, [event], set())
    assert agent._pending_steer

    with session["history_lock"]:
        session["running"] = False

    real_thread = threading.Thread
    monkeypatch.setattr(server.threading, "Thread", _NotStartedThread)
    monkeypatch.setattr(
        server,
        "_start_usage_ticker",
        lambda *args, **kwargs: (threading.Event(), types.SimpleNamespace(join=lambda: None)),
    )
    result = {}

    def admit_next_turn():
        result["accepted"] = server._run_prompt_submit(
            "rid-next-turn", "sid-next-turn", session, "next prompt"
        )

    thread = real_thread(target=admit_next_turn, daemon=True)
    thread.start()
    assert thread.join(1.0) is None
    assert not thread.is_alive(), "next-turn admission deadlocked"
    assert result["accepted"] is True
    assert agent._pending_steer is None
    assert "_steer_accepted" not in session["_completion_pending"][0]


def test_notification_turn_releases_claim_when_prompt_admission_fails(monkeypatch):
    import tools.async_delegation as async_delegation

    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_admission_false",
    }
    sess = _session(running=True)
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        async_delegation,
        "claim_event_delivery",
        lambda *_args: "claim-token",
    )
    monkeypatch.setattr(
        async_delegation,
        "complete_event_delivery",
        lambda *args: calls.append(("complete", args)),
    )
    monkeypatch.setattr(
        async_delegation,
        "release_event_delivery",
        lambda *args: calls.append(("release", args)),
    )
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: False,
    )

    assert server._submit_process_notification_turn(
        "sid", sess, evt, "completion"
    ) == [evt]
    assert calls == [("release", (evt, "claim-token"))]


def test_midloop_routine_child_successes_are_silent_before_batch_projection(monkeypatch):
    agent = _SteerAgent()
    child_events = [
        _routine_child_completion("proc_child_a"),
        _routine_child_completion("proc_child_b"),
    ]
    sess = _session(running=True, agent=agent, _completion_pending=child_events)
    emitted: list = []
    monkeypatch.setattr(server, "_emit", lambda *args, **_kw: emitted.append(args))

    server._flush_pending_completions_if_idle("sid_child_midloop", sess, set())

    assert agent.steers == []
    assert _process_status(emitted) == []
    assert sess.get("_completion_pending") == []
    assert all(evt["session_id"] in process_registry._completion_consumed for evt in child_events)


def test_idle_routine_child_successes_do_not_submit_parent_turn(monkeypatch):
    child_events = [
        _routine_child_completion("proc_child_idle_a"),
        _routine_child_completion("proc_child_idle_b"),
    ]
    sess = _session(running=False, _completion_pending=child_events)
    turns: list = []
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kw: turns.append("submitted") or True,
    )

    server._flush_pending_completions_if_idle("sid_child_idle", sess, set())

    assert turns == []
    assert sess["running"] is False
    assert sess.get("_completion_pending") == []
    assert all(evt["session_id"] in process_registry._completion_consumed for evt in child_events)


def test_mixed_completion_batch_keeps_parent_order_and_ack_path(monkeypatch):
    agent = _SteerAgent()
    child = _routine_child_completion("proc_child_mixed")
    parent = _completion("proc_parent_mixed", 0, "echo parent")
    sess = _session(running=True, agent=agent, _completion_pending=[child, parent])
    emitted: list = []
    monkeypatch.setattr(server, "_emit", lambda *args, **_kw: emitted.append(args))

    server._flush_pending_completions_if_idle("sid_mixed", sess, set())

    assert len(agent.steers) == 1
    assert "proc_parent_mixed" in agent.steers[0]
    assert "proc_child_mixed" not in agent.steers[0]
    assert len(_process_status(emitted)) == 1
    assert "proc_parent_mixed" in _process_status(emitted)[0][2]["text"]
    assert "proc_child_mixed" not in _process_status(emitted)[0][2]["text"]
    assert "proc_child_mixed" in process_registry._completion_consumed
    assert "proc_parent_mixed" not in process_registry._completion_consumed
    assert [evt["session_id"] for evt in sess["_completion_pending"]] == ["proc_parent_mixed"]

    agent._pending_steer = None
    server._ack_steered_completion_ingest(sess)
    assert "proc_parent_mixed" in process_registry._completion_consumed


def _run_poller_until(sid: str, sess: dict, pred, timeout: float = 2.0) -> None:
    stop = threading.Event()
    thread = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, sid, sess),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if pred():
                break
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=2.0)


def test_busy_completions_coalesce_to_one_ingest_batch_idle_stays_immediate(
    monkeypatch,
):
    """N completions while running become one batch at next ingest; idle stays immediate for a lone completion."""
    isolated: queue_mod.Queue = queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    # --- idle: one completion still delivers immediately ---
    idle_emitted: list = []
    idle_turns: list = []
    idle_sess = _session()
    idle_sid = "sid_idle_coalesce"
    server._sessions[idle_sid] = idle_sess
    monkeypatch.setattr(server, "_emit", lambda *args, **_kw: idle_emitted.append(args))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, session, text, **_kw: idle_turns.append(text)
        or session.__setitem__("running", False)
        or True,
    )
    idle_evt = _completion("proc_idle_one", 0, "echo idle")
    process_registry._completion_consumed.discard("proc_idle_one")
    isolated.put(idle_evt)
    try:
        _run_poller_until(
            idle_sid,
            idle_sess,
            lambda: len(idle_turns) == 1,
        )
        idle_status = _process_status(idle_emitted)
        assert len(idle_status) == 1
        assert idle_status[0][2]["kind"] == "process"
        assert "proc_idle_one" in idle_status[0][2]["text"]
        assert len(idle_turns) == 1
        assert "proc_idle_one" in idle_turns[0]
        assert isolated.empty()
    finally:
        server._sessions.pop(idle_sid, None)
        process_registry._completion_consumed.discard("proc_idle_one")
        while not isolated.empty():
            isolated.get_nowait()

    # --- busy: N completions buffer; next ingest is one structured batch ---
    busy_emitted: list = []
    busy_turns: list = []
    busy_sess = _session(running=True)
    busy_sid = "sid_busy_coalesce"
    server._sessions[busy_sid] = busy_sess
    monkeypatch.setattr(server, "_emit", lambda *args, **_kw: busy_emitted.append(args))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, session, text, **_kw: busy_turns.append(text)
        or session.__setitem__("running", False)
        or True,
    )
    events = [
        _completion("proc_busy_a", 0, "echo a"),
        _completion("proc_busy_b", 1, "echo b"),
        _completion("proc_busy_c", 0, "echo c"),
    ]
    for evt in events:
        process_registry._completion_consumed.discard(evt["session_id"])

    stop = threading.Event()
    thread = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, busy_sid, busy_sess),
        daemon=True,
    )
    thread.start()
    try:
        for evt in events:
            isolated.put(evt)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not isolated.empty():
            time.sleep(0.01)

        assert _process_status(busy_emitted) == []
        assert busy_turns == []

        busy_sess["running"] = False
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(busy_turns) < 1:
            time.sleep(0.01)

        batch_status = _process_status(busy_emitted)
        assert len(batch_status) == 1
        assert batch_status[0][2]["kind"] == "process"
        batch_text = batch_status[0][2]["text"]
        assert len(busy_turns) == 1
        turn_text = busy_turns[0]
        for evt in events:
            assert evt["session_id"] in batch_text
            assert str(evt["exit_code"]) in batch_text
            assert evt["session_id"] in turn_text
            assert str(evt["exit_code"]) in turn_text
        assert isolated.empty()
    finally:
        stop.set()
        thread.join(timeout=2.0)
        server._sessions.pop(busy_sid, None)
        for evt in events:
            process_registry._completion_consumed.discard(evt["session_id"])
        while not isolated.empty():
            isolated.get_nowait()


def test_idle_completions_coalesce_to_one_ingest_batch(monkeypatch):
    """N idle/between-turn completions in one window become one parent ingest."""
    isolated: queue_mod.Queue = queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    emitted: list = []
    turns: list = []
    sess = _session(running=False)
    sid = "sid_idle_batch"
    server._sessions[sid] = sess
    monkeypatch.setattr(server, "_emit", lambda *args, **_kw: emitted.append(args))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, session, text, **_kw: turns.append(text)
        or session.__setitem__("running", False)
        or True,
    )
    events = [
        _completion("proc_idle_a", 0, "echo a"),
        _completion("proc_idle_b", 1, "echo b"),
        _completion("proc_idle_c", 0, "echo c"),
    ]
    for evt in events:
        process_registry._completion_consumed.discard(evt["session_id"])
        isolated.put(evt)
    try:
        _run_poller_until(sid, sess, lambda: len(turns) >= 1)
        status = _process_status(emitted)
        assert len(status) == 1
        assert status[0][2]["kind"] == "process"
        assert len(turns) == 1
        batch_text = status[0][2]["text"]
        turn_text = turns[0]
        for evt in events:
            assert evt["session_id"] in batch_text
            assert str(evt["exit_code"]) in batch_text
            assert evt["session_id"] in turn_text
            assert str(evt["exit_code"]) in turn_text
        assert isolated.empty()
    finally:
        server._sessions.pop(sid, None)
        for evt in events:
            process_registry._completion_consumed.discard(evt["session_id"])
        while not isolated.empty():
            isolated.get_nowait()


def test_child_tails_after_compress_do_not_each_start_a_parent_turn(monkeypatch):
    """After restart/compress, N idle child terminal() tails are one ingest.

    Production terminal() stores task_id=\"default\" for parent and child.
    A live native delegate may exist only as a durable SQLite row after
    process restart. #131's 0.1s window is too short for spaced tails;
    they must still become one parent ingest, and a later parent-owned
    tail must still deliver.
    """
    import tools.async_delegation as ad

    isolated: queue_mod.Queue = queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    ad._reset_for_tests()

    old_key = "sess_pre_compress"
    new_key = "sess_post_compress"
    sid = "sid_after_compress"

    class _Db:
        def resolve_resume_session_id(self, target):
            return new_key if target == old_key else target

    monkeypatch.setattr(server, "_get_db", lambda: _Db())

    # Production spawn identity: both parent and child collapse to "default".
    procs = {
        "proc_child_a": types.SimpleNamespace(
            task_id="default", session_key=old_key, parent_session_id=old_key
        ),
        "proc_child_b": types.SimpleNamespace(
            task_id="default", session_key=old_key, parent_session_id=old_key
        ),
        "proc_parent": types.SimpleNamespace(
            task_id="default", session_key=new_key, parent_session_id=new_key
        ),
    }
    monkeypatch.setattr(process_registry, "get", lambda pid: procs.get(pid))

    # Process-restart residual: durable running row. Same-process compress
    # may also have an in-memory record; either must be enough.
    ad._persist_dispatch(
        {
            "delegation_id": "deleg_live",
            "session_key": old_key,
            "origin_ui_session_id": "old_tab",
            "parent_session_id": old_key,
            "origin_session_id": "",
            "dispatched_at": time.time(),
            "goal": "child work",
        }
    )
    with ad._records_lock:
        ad._records.clear()

    emitted: list = []
    turns: list = []
    sess = _session(running=False, session_key=new_key)
    sess["agent"] = types.SimpleNamespace(session_id=new_key)
    server._sessions[sid] = sess
    monkeypatch.setattr(server, "_emit", lambda *args, **_kw: emitted.append(args))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, session, text, **_kw: turns.append(text)
        or session.__setitem__("running", False)
        or True,
    )

    child_events = [
        {**_completion("proc_child_a", 0, "echo child-a"), "session_key": old_key},
        {**_completion("proc_child_b", 0, "echo child-b"), "session_key": old_key},
    ]
    parent_evt = {
        **_completion("proc_parent", 0, "echo parent"),
        "session_key": new_key,
    }
    for evt in (*child_events, parent_evt):
        process_registry._completion_consumed.discard(evt["session_id"])

    stop = threading.Event()
    thread = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, sid, sess),
        daemon=True,
    )
    thread.start()
    try:
        isolated.put(child_events[0])
        # Give #131's 0.1s singleton flush a chance to fire before the
        # second tail. The contract is still one ingest for both.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            time.sleep(0.02)
        isolated.put(child_events[1])
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if turns and all(
                pid in turns[0] for pid in ("proc_child_a", "proc_child_b")
            ):
                break
            time.sleep(0.01)

        assert len(turns) == 1
        child_text = turns[0]
        assert "proc_child_a" in child_text
        assert "proc_child_b" in child_text
        assert "proc_parent" not in child_text
        child_status = _process_status(emitted)
        assert len(child_status) == 1

        isolated.put(parent_evt)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(turns) < 2:
            time.sleep(0.01)

        assert len(turns) == 2
        assert "proc_parent" in turns[1]
        assert isolated.empty()
    finally:
        stop.set()
        thread.join(timeout=2.0)
        server._sessions.pop(sid, None)
        ad._delete_durable_delegation("deleg_live")
        ad._reset_for_tests()
        for evt in (*child_events, parent_evt):
            process_registry._completion_consumed.discard(evt["session_id"])
        while not isolated.empty():
            isolated.get_nowait()


def test_midloop_completions_use_steer_rail_not_new_turns(monkeypatch):
    """N completions while mid-tool-batch land on _pending_steer, not N turns.

    #138: insert into the current loop at least as timely as steer. A live
    parent executing tools must ingest every child notify_on_complete via
    AIAgent.steer (one structured batch when N>1) before any new idle turn.
    """
    isolated: queue_mod.Queue = queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    agent = _SteerAgent()
    agent._executing_tools = True
    emitted: list = []
    turns: list = []
    sess = _session(running=True, agent=agent)
    sid = "sid_midloop_steer"
    server._sessions[sid] = sess
    monkeypatch.setattr(server, "_emit", lambda *args, **_kw: emitted.append(args))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, session, text, **_kw: turns.append(text)
        or session.__setitem__("running", False)
        or True,
    )
    events = [
        _completion("proc_mid_a", 0, "echo a"),
        _completion("proc_mid_b", 1, "echo b"),
        _completion("proc_mid_c", 0, "echo c"),
    ]
    for evt in events:
        process_registry._completion_consumed.discard(evt["session_id"])

    stop = threading.Event()
    thread = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, sid, sess),
        daemon=True,
    )
    thread.start()
    try:
        for evt in events:
            isolated.put(evt)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if agent.steers:
                break
            time.sleep(0.01)

        assert turns == []
        assert len(agent.steers) == 1
        steered = agent.steers[0]
        for evt in events:
            assert evt["session_id"] in steered
            assert str(evt["exit_code"]) in steered
        status = _process_status(emitted)
        assert len(status) == 1
        assert all(evt["session_id"] in status[0][2]["text"] for evt in events)
        assert isolated.empty()
        # Accept is not ingest: keep events pending until leftover/tool-result ACK.
        assert "proc_mid_a" not in process_registry._completion_consumed
        pending = sess.get("_completion_pending") or []
        assert {evt.get("session_id") for evt in pending} == {
            "proc_mid_a",
            "proc_mid_b",
            "proc_mid_c",
        }
    finally:
        stop.set()
        thread.join(timeout=2.0)
        server._sessions.pop(sid, None)
        for evt in events:
            process_registry._completion_consumed.discard(evt["session_id"])
        while not isolated.empty():
            isolated.get_nowait()


def test_llm_blocked_pileup_is_one_steer_batch_zero_drops(monkeypatch):
    """Parent blocked on LLM + N notifies → one structured steer; no drops.

    Completions that pile up while _model_request_active must not each
    become a parent turn, and must not wait for the whole loop to stop.
    """
    isolated: queue_mod.Queue = queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    agent = _SteerAgent()
    agent._model_request_active.set()
    emitted: list = []
    turns: list = []
    sess = _session(running=True, agent=agent)
    sid = "sid_llm_blocked_batch"
    server._sessions[sid] = sess
    monkeypatch.setattr(server, "_emit", lambda *args, **_kw: emitted.append(args))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, session, text, **_kw: turns.append(text)
        or session.__setitem__("running", False)
        or True,
    )
    events = [
        _completion("proc_llm_a", 0, "echo a"),
        _completion("proc_llm_b", 0, "echo b"),
    ]
    for evt in events:
        process_registry._completion_consumed.discard(evt["session_id"])

    stop = threading.Event()
    thread = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, sid, sess),
        daemon=True,
    )
    thread.start()
    try:
        isolated.put(events[0])
        time.sleep(0.15)
        isolated.put(events[1])
        deadline = time.monotonic() + 3.5
        while time.monotonic() < deadline:
            if agent.steers:
                break
            time.sleep(0.01)

        assert turns == []
        assert sess.get("running") is True
        assert len(agent.steers) == 1
        steered = agent.steers[0]
        assert "proc_llm_a" in steered and "proc_llm_b" in steered
        assert isolated.empty()
        # steer() True is accept/staging only — ACK happens at ingest.
        consumed = process_registry._completion_consumed
        assert "proc_llm_a" not in consumed
        assert "proc_llm_b" not in consumed
        pending = sess.get("_completion_pending") or []
        pending_ids = {evt.get("session_id") for evt in pending}
        assert pending_ids == {"proc_llm_a", "proc_llm_b"}
    finally:
        stop.set()
        thread.join(timeout=2.0)
        server._sessions.pop(sid, None)
        for evt in events:
            process_registry._completion_consumed.discard(evt["session_id"])
        while not isolated.empty():
            isolated.get_nowait()


def test_single_completion_steers_while_parent_waits_on_tools(monkeypatch):
    """One notify while mid-loop is ingested immediately; loop need not stop."""
    isolated: queue_mod.Queue = queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    agent = _SteerAgent()
    agent._executing_tools = True
    turns: list = []
    sess = _session(running=True, agent=agent)
    sid = "sid_single_midloop"
    server._sessions[sid] = sess
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, session, text, **_kw: turns.append(text)
        or session.__setitem__("running", False)
        or True,
    )
    evt = _completion("proc_wait_one", 0, "echo wait")
    process_registry._completion_consumed.discard("proc_wait_one")
    isolated.put(evt)
    try:
        _run_poller_until(sid, sess, lambda: bool(agent.steers))
        assert turns == []
        assert sess.get("running") is True
        assert len(agent.steers) == 1
        assert "proc_wait_one" in agent.steers[0]
        assert isolated.empty()
        assert "proc_wait_one" not in process_registry._completion_consumed
        pending = sess.get("_completion_pending") or []
        assert {evt.get("session_id") for evt in pending} == {"proc_wait_one"}
    finally:
        server._sessions.pop(sid, None)
        process_registry._completion_consumed.discard("proc_wait_one")
        while not isolated.empty():
            isolated.get_nowait()


def test_interrupt_after_steer_accept_does_not_drop_completion():
    """clear_interrupt after accept must not lose the event or block replay."""
    agent = _SteerAgent()

    def clear_interrupt(self, *, preserve_redirect=False):
        self._pending_steer = None
        return True

    agent.clear_interrupt = types.MethodType(clear_interrupt, agent)
    sess = _session(running=True, agent=agent)
    evt = _completion("proc_int_drop", 1, "echo int")
    process_registry._completion_consumed.discard("proc_int_drop")
    try:
        assert server._deliver_completions_via_steer(
            "sid_int_drop", sess, [evt], set()
        )
        assert agent._pending_steer
        assert "proc_int_drop" not in process_registry._completion_consumed

        agent.clear_interrupt()
        assert agent._pending_steer is None

        pending = sess.get("_completion_pending") or []
        assert {item.get("session_id") for item in pending} == {"proc_int_drop"}
        assert process_registry.is_completion_consumed("proc_int_drop") is False
    finally:
        process_registry._completion_consumed.discard("proc_int_drop")


def test_shared_apply_can_ack_before_steer_returns(monkeypatch):
    inserted: list[str] = []
    marked: list[str] = []

    class _ApplyBarrierAgent(_SteerAgent):
        def __init__(self):
            super().__init__()

            def _apply(messages, _num_tool_msgs):
                text = self._pending_steer
                self._pending_steer = None
                messages[-1]["content"] += text

            self._apply_pending_steer_to_tool_results = _apply

        def steer(self, text: str) -> bool:
            assert super().steer(text)
            messages = [{"role": "tool", "content": "result:"}]
            self._apply_pending_steer_to_tool_results(messages, 1)
            inserted.append(messages[0]["content"])
            return True

    agent = _ApplyBarrierAgent()
    sess = _session(running=True, agent=agent)
    event = _completion("proc-publish-apply", 0, "echo apply")
    monkeypatch.setattr(
        server,
        "_mark_completion_events_consumed",
        lambda events: marked.extend(evt["session_id"] for evt in events),
    )

    assert server._deliver_completions_via_steer("sid-apply", sess, [event], set())
    assert "proc-publish-apply" in inserted[0]
    assert marked == [event["session_id"]]
    assert sess.get("_completion_pending") == []


def test_final_leftover_can_ack_before_steer_returns(monkeypatch):
    marked: list[str] = []
    session_ref = {}

    class _LeftoverBarrierAgent(_SteerAgent):
        def steer(self, text: str) -> bool:
            assert super().steer(text)
            leftover = self._pending_steer
            self._pending_steer = None
            assert leftover is not None
            sess = session_ref["session"]
            snapshot = server._take_steered_completion_snapshot(sess, leftover)
            with sess["history_lock"]:
                server._enqueue_prompt(
                    sess,
                    leftover,
                    None,
                    on_admitted=lambda: server._ack_steered_completion_ingest(
                        sess, snapshot
                    ),
                )
                sess["running"] = False
            assert server._drain_queued_prompt("rid", "sid-leftover", sess)
            return True

    agent = _LeftoverBarrierAgent()
    sess = _session(running=True, agent=agent)
    session_ref["session"] = sess
    event = _completion("proc-publish-leftover", 0, "echo leftover")
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        server,
        "_mark_completion_events_consumed",
        lambda events: marked.extend(evt["session_id"] for evt in events),
    )

    assert server._deliver_completions_via_steer("sid-leftover", sess, [event], set())
    assert marked == [event["session_id"]]
    assert sess.get("_completion_pending") == []


@pytest.mark.parametrize("raises", [False, True], ids=["false", "raises"])
def test_failed_steer_rolls_back_only_its_publication(monkeypatch, raises):
    prior = _completion("proc-prior", 0, "echo prior")
    later = _completion("proc-later", 0, "echo later")
    event = _completion("proc-failed-publication", 0, "echo failed")
    session_ref = {}

    class _FailingAgent(_SteerAgent):
        def steer(self, text: str) -> bool:
            assert getattr(self, "_completion_steer_guards", False)
            sess = session_ref["session"]
            with sess["history_lock"]:
                sess["_completion_pending"].append(later)
            if raises:
                raise RuntimeError("injected steer failure")
            return False

    sess = _session(running=True, agent=_FailingAgent(), _completion_pending=[prior])
    session_ref["session"] = sess
    marked: list[str] = []
    monkeypatch.setattr(
        server,
        "_mark_completion_events_consumed",
        lambda events: marked.extend(evt["session_id"] for evt in events),
    )

    assert not server._deliver_completions_via_steer("sid-failed", sess, [event], set())
    assert [item["session_id"] for item in sess["_completion_pending"]] == [
        prior["session_id"],
        later["session_id"],
    ]
    assert marked == []


def test_leftover_ack_does_not_consume_later_steer():
    """Leftover harvest ACKs the snapshot only; a later steer stays replayable."""
    agent = _SteerAgent()

    def _drain(self):
        text = self._pending_steer
        self._pending_steer = None
        return text

    def clear_interrupt(self, *, preserve_redirect=False):
        self._pending_steer = None
        return True

    agent._drain_pending_steer = types.MethodType(_drain, agent)
    agent.clear_interrupt = types.MethodType(clear_interrupt, agent)
    sess = _session(running=True, agent=agent)
    evt_a = _completion("proc_left_a", 0, "echo a")
    evt_b = _completion("proc_left_b", 0, "echo b")
    process_registry._completion_consumed.discard("proc_left_a")
    process_registry._completion_consumed.discard("proc_left_b")
    try:
        assert server._deliver_completions_via_steer("sid_left", sess, [evt_a], set())
        leftover = agent._drain_pending_steer()
        assert leftover and "proc_left_a" in leftover

        assert server._deliver_completions_via_steer("sid_left", sess, [evt_b], set())
        assert agent._pending_steer and "proc_left_b" in agent._pending_steer

        with sess["history_lock"]:
            server._enqueue_prompt(sess, leftover, sess.get("transport"))
        server._ack_steered_completion_ingest(sess)

        assert "proc_left_a" in process_registry._completion_consumed
        assert "proc_left_b" not in process_registry._completion_consumed
        pending_ids = {evt.get("session_id") for evt in (sess.get("_completion_pending") or [])}
        assert "proc_left_b" in pending_ids

        agent.clear_interrupt()
        assert agent._pending_steer is None
        assert "proc_left_b" not in process_registry._completion_consumed
        assert process_registry.is_completion_consumed("proc_left_b") is False
    finally:
        process_registry._completion_consumed.discard("proc_left_a")
        process_registry._completion_consumed.discard("proc_left_b")


def test_leftover_ack_toctou_does_not_consume_later_steer():
    """Stale empty live snapshot must not ACK a concurrent later steer."""

    class HookLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.hook: Callable[[], None] | None = None

        def __enter__(self):
            if self.hook:
                fn, self.hook = self.hook, None
                fn()
            self._lock.acquire()
            return self

        def __exit__(self, *exc):
            self._lock.release()
            return False

    agent = _SteerAgent()
    lock = HookLock()
    sess = _session(running=True, agent=agent, history_lock=lock)
    evt_a = _completion("proc_race_a", 0, "echo a")
    evt_b = _completion("proc_race_b", 0, "echo b")
    process_registry._completion_consumed.discard("proc_race_a")
    process_registry._completion_consumed.discard("proc_race_b")
    try:
        assert server._deliver_completions_via_steer("sid_race", sess, [evt_a], set())
        leftover = agent._pending_steer
        agent._pending_steer = None
        assert leftover and "proc_race_a" in leftover
        with lock._lock:
            server._enqueue_prompt(sess, leftover, sess.get("transport"))

        def _inject_b():
            assert server._deliver_completions_via_steer("sid_race", sess, [evt_b], set())

        lock.hook = _inject_b
        server._ack_steered_completion_ingest(sess)

        assert "proc_race_a" in process_registry._completion_consumed
        assert "proc_race_b" not in process_registry._completion_consumed
        assert process_registry.is_completion_consumed("proc_race_b") is False
        pending_ids = {
            evt.get("session_id") for evt in (sess.get("_completion_pending") or [])
        }
        assert "proc_race_b" in pending_ids
        assert agent._pending_steer and "proc_race_b" in agent._pending_steer
    finally:
        process_registry._completion_consumed.discard("proc_race_a")
        process_registry._completion_consumed.discard("proc_race_b")


def test_pre_api_completion_ingest_acks_only_inserted_snapshot(tmp_path, monkeypatch):
    """The real conversation loop ACKs only the completion inserted before API."""
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            session_id="pre-api-ack-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=2,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "stable test prompt"
    agent._session_db = None
    agent._session_json_enabled = False
    agent.save_trajectories = False
    agent.compression_enabled = False
    agent._cleanup_task_resources = lambda *_a, **_kw: None
    agent._save_trajectory = lambda *_a, **_kw: None

    sess = _session(running=True, agent=agent)
    first = _completion("proc_pre_api_a", 0, "echo a")
    later = _completion("proc_pre_api_b", 0, "echo b")
    no_tool = _completion("proc_pre_api_no_tool", 0, "echo no-tool")
    ids = {evt["session_id"] for evt in (first, later, no_tool)}
    process_registry._completion_consumed.difference_update(ids)
    marked: list[str] = []
    original_mark = server._mark_completion_events_consumed

    def record_mark(events):
        marked.extend(evt["session_id"] for evt in events)
        original_mark(events)

    monkeypatch.setattr(server, "_mark_completion_events_consumed", record_mark)
    original_ack = server._ack_steered_completion_ingest
    ack_calls = 0

    def ack_with_concurrent_later(session, snapshot=None):
        nonlocal ack_calls
        ack_calls += 1
        if ack_calls == 1:
            assert server._deliver_completions_via_steer(
                "sid_pre_api", sess, [later], set()
            )
        original_ack(session, snapshot)

    monkeypatch.setattr(server, "_ack_steered_completion_ingest", ack_with_concurrent_later)
    seen_requests: list[list[dict]] = []

    def response(content: str, finish_reason: str):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=None),
                    finish_reason=finish_reason,
                )
            ],
            model="test/model",
            usage=None,
        )

    def model_call(kwargs):
        seen_requests.append(kwargs["messages"])
        if len(seen_requests) == 1:
            assert server._deliver_completions_via_steer(
                "sid_pre_api", sess, [first], set()
            )
            return response("partial", "length")
        return response("done", "stop")

    agent._interruptible_api_call = model_call
    history = [
        {"role": "user", "content": "run a tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "prior output", "tool_call_id": "tc1"},
    ]

    try:
        result = agent.run_conversation("continue", conversation_history=history)

        assert len(seen_requests) == 2
        provider_text = str(seen_requests[1])
        assert "proc_pre_api_a" in provider_text
        assert "proc_pre_api_b" not in provider_text
        assert marked.count("proc_pre_api_a") == 1
        assert "proc_pre_api_a" in process_registry._completion_consumed
        assert "proc_pre_api_b" not in process_registry._completion_consumed
        assert [
            evt["session_id"] for evt in sess.get("_completion_pending") or []
        ] == ["proc_pre_api_b"]
        assert "proc_pre_api_b" in result.get("pending_steer", "")

        assert server._deliver_completions_via_steer(
            "sid_pre_api", sess, [no_tool], set()
        )
        agent._apply_pending_steer_to_tool_results(
            [{"role": "user", "content": "no tool result"}], 1
        )
        assert "proc_pre_api_no_tool" in agent._pending_steer
        assert "proc_pre_api_no_tool" not in process_registry._completion_consumed
    finally:
        process_registry._completion_consumed.difference_update(ids)
