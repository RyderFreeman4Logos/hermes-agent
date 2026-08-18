"""Busy process completions coalesce to one ingest batch (#122)."""

from __future__ import annotations

import queue as queue_mod
import threading
import time
import types

from tools.process_registry import process_registry
from tui_gateway import server


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


def _process_status(emitted: list) -> list:
    return [args for args in emitted if args and args[0] == "status.update"]


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
        or session.__setitem__("running", False),
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
        or session.__setitem__("running", False),
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
        or session.__setitem__("running", False),
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
        or session.__setitem__("running", False),
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
