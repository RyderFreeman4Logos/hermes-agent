"""Native delegated-child lifecycle beyond ordinary formatter silence.

Spawn freezes the child stamp from the process context. A routine child
watch is dropped before the queue; an explicit handoff still notifies.
Gateway completion events and checkpoints carry the stamp and handoff note.
"""

import json
import os
import subprocess
import sys
import time

import pytest

from agent.delegation_context import delegated_child_context
from gateway.run_notifications import GatewayNotificationsMixin
from tools.process_registry import ProcessRegistry, ProcessSession


def _events(registry):
    found = []
    while not registry.completion_queue.empty():
        found.append(registry.completion_queue.get_nowait())
    return found


def test_spawn_stamps_delegated_child_from_context_not_label(monkeypatch):
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER

    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "unowned-ambient-marker")
    registry = ProcessRegistry()
    with delegated_child_context():
        child = registry._new_session("echo child", "t1", "", "sk", "/tmp")
    parent = registry._new_session("echo parent", "t1", "", "sk", "/tmp")
    assert child.delegated_child is True
    assert parent.delegated_child is False


def test_owning_profile_optin_surfaces_routine_native_watch_and_completion(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.process_registry_notifications import format_process_notification

    home = tmp_path / "owner"
    home.mkdir()
    (home / "config.yaml").write_text(
        "delegation:\n  surface_child_process_notifications: true\n"
    )
    token = set_hermes_home_override(home)
    try:
        registry = ProcessRegistry()
        child = ProcessSession(
            id="proc_optin_watch", command="echo child", task_id="sa-child",
            owner_task_id="sa-child", started_at=time.time(),
            watch_patterns=["READY"], delegated_child=True,
        )
        registry._check_watch_patterns(child, "READY\n")
        queued = _events(registry)
        assert len(queued) == 1 and queued[0]["type"] == "watch_match"
        event = {
            "type": "completion", "session_id": "proc_optin", "command": "echo child",
            "started_at": 1.0, "exit_code": 0, "completion_reason": "exited",
            "termination_source": "", "delegated_child": True, "owner_task_id": "sa-child",
        }
        assert format_process_notification(event) is not None
        registry.completion_queue.put(event)
        assert len(registry.drain_notifications()) == 1
    finally:
        reset_hermes_home_override(token)


def test_native_watch_is_suppressed_before_queue_unless_handed_off():
    registry = ProcessRegistry()
    child = ProcessSession(
        id="proc_child_watch", command="tail -f app.log", task_id="native",
        started_at=time.time(), watch_patterns=["ERROR"], delegated_child=True,
    )
    registry._check_watch_patterns(child, "ERROR: disk full\n")
    assert _events(registry) == []

    handed = ProcessSession(
        id="proc_handed_watch", command="tail -f app.log", task_id="parent",
        started_at=time.time(), watch_patterns=["ERROR"], delegated_child=True,
        handoff_note="watch the deploy",
    )
    registry._check_watch_patterns(handed, "ERROR: disk full\n")
    events = _events(registry)
    assert len(events) == 1
    assert events[0]["type"] == "watch_match"
    assert events[0]["delegated_child"] is True
    assert events[0]["handoff_note"] == "watch the deploy"

    ordinary = ProcessSession(
        id="proc_parent_watch", command="tail -f app.log", task_id="parent",
        started_at=time.time(), watch_patterns=["ERROR"],
    )
    registry._check_watch_patterns(ordinary, "ERROR: still visible\n")
    visible = _events(registry)
    assert len(visible) == 1 and visible[0]["session_id"] == "proc_parent_watch"


def test_accepted_handoff_routes_completion_to_live_parent(tmp_path, monkeypatch):
    import weakref

    import tools.process_registry as process_registry_module
    from tools.delegate_tool_registry import _register_subagent, _unregister_subagent
    from tools.process_registry import _handle_process

    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(process_registry_module, "CHECKPOINT_PATH", checkpoint)
    registry = process_registry_module.process_registry
    writes = []
    real_write = registry._write_checkpoint

    def counted_write(*args, **kwargs):
        writes.append(registry._lock.locked())
        return real_write(*args, **kwargs)

    monkeypatch.setattr(registry, "_write_checkpoint", counted_write)

    class Parent:
        session_id = "parent-db"
        _current_task_id = "parent-owner"

    parent = Parent()

    class Child:
        session_id = "child-db"
        _delegate_parent_ref = weakref.ref(parent)

    child = Child()
    sid = "sa-handoff-route"
    session = ProcessSession(
        id="proc_handoff_route", command="echo child", task_id=sid,
        owner_task_id=sid, session_key="child-key", parent_session_id="child-db",
        delegated_child=True, notify_on_complete=True, started_at=time.time(),
    )
    session.watcher_platform = "telegram"
    session.watcher_chat_id = "123"
    session.watcher_interval = 5
    registry._running[session.id] = session
    watcher = {"session_id": session.id, "parent_session_id": "child-db", "session_key": "child-key"}
    registry.pending_watchers.append(watcher)
    _register_subagent({
        "subagent_id": sid, "parent_id": None, "depth": 0, "goal": "route",
        "model": "none", "started_at": time.time(), "status": "running",
        "tool_count": 0, "agent": child,
    })
    try:
        result = json.loads(_handle_process(
            {"action": "handoff", "session_id": session.id, "data": "retain public result"},
            task_id=sid,
        ))
        assert result["status"] == "handed_off"
        assert writes == [False]
        assert session.parent_session_id == "parent-db"
        assert session.session_key == "child-key"
        assert watcher["parent_session_id"] == "parent-db"
        assert watcher["session_key"] == "child-key"
        event = GatewayNotificationsMixin._build_process_completion_event(watcher, session, session.id)
        assert event["parent_session_id"] == "parent-db"
        assert event["session_key"] == "child-key"
        stored = json.loads(checkpoint.read_text())
        assert stored[0]["parent_session_id"] == "parent-db"
    finally:
        _unregister_subagent(sid)
        registry._running.pop(session.id, None)
        if watcher in registry.pending_watchers:
            registry.pending_watchers.remove(watcher)


def test_gateway_completion_event_carries_stamp_and_handoff():
    session = ProcessSession(
        id="proc_gw", command="echo child", task_id="child", started_at=1.0,
        delegated_child=True, handoff_note="finish the deploy", exit_code=0,
    )
    event = GatewayNotificationsMixin._build_process_completion_event(
        {}, session, session.id,
    )
    assert event["delegated_child"] is True
    assert event["handoff_note"] == "finish the deploy"


def test_checkpoint_roundtrip_restores_child_flag_and_handoff(tmp_path, monkeypatch):
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
    )
    try:
        start = ProcessRegistry._safe_host_start_time(proc.pid)
        assert start is not None
        writer = ProcessRegistry()
        session = ProcessSession(
            id="proc_ckpt_child", command="sleep-for-child-stamp", task_id="child",
            owner_task_id="parent", started_at=time.time(), pid=proc.pid,
            pid_scope="host", host_start_time=start, delegated_child=True,
            handoff_note="owned by parent",
        )
        with writer._lock:
            writer._running[session.id] = session
        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr("tools.process_registry.CHECKPOINT_PATH", checkpoint)
        writer._write_checkpoint()
        stored = json.loads(checkpoint.read_text())
        assert stored[0]["delegated_child"] is True
        assert stored[0]["handoff_note"] == "owned by parent"

        reader = ProcessRegistry()
        assert reader.recover_from_checkpoint() == 1
        restored = reader.get("proc_ckpt_child")
        assert restored.delegated_child is True
        assert restored.handoff_note == "owned by parent"
        assert proc.poll() is None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.parametrize("event", [
    {"type": "completion", "session_id": "p", "command": "c", "started_at": 1.0,
     "exit_code": "0", "completion_reason": "exited", "termination_source": "",
     "delegated_child": True},
    {"type": "completion", "session_id": "p", "command": "c", "started_at": 1.0,
     "exit_code": 1, "completion_reason": "exited", "termination_source": "",
     "delegated_child": True},
    {"type": "watch_match", "session_id": "p", "command": "c", "delegated_child": True,
     "pattern": "ERROR", "output": "ERROR: boom"},
])
def test_malformed_or_visible_child_events_are_not_silenced(event):
    from tools.process_registry_notifications import format_process_notification
    assert format_process_notification(event) is not None


def test_cross_profile_delayed_checkpoint_keeps_both_files(tmp_path, monkeypatch):
    """A later write to profile B must not drop a delayed write to profile A."""
    import threading
    from unittest.mock import patch
    from gateway.run import _profile_runtime_scope
    from tools import process_registry as pr
    import utils

    monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", False)
    a, b = tmp_path / "a", tmp_path / "b"
    for home in (a, b):
        home.mkdir()
        (home / "config.yaml").write_text("display:\n  background_process_notifications: concise\n")
    reg = pr.ProcessRegistry()
    reached, release, results = threading.Event(), threading.Event(), {}
    real = utils._dump_json

    def delayed(payload, handle, *args, **kw):
        if threading.current_thread().name == "profile-a":
            reached.set()
            assert release.wait(10)
        return real(payload, handle, *args, **kw)

    def spawn(home, key):
        with _profile_runtime_scope(home, {}):
            results[key] = reg.spawn_local("sleep 30", cwd=str(home), task_id=key, owner_task_id=key)

    worker = threading.Thread(target=spawn, args=(a, "a"), name="profile-a")
    try:
        with patch.object(utils, "_dump_json", delayed):
            worker.start()
            assert reached.wait(10)
            spawn(b, "b")
            release.set()
            worker.join(10)
        assert (a / "processes.json").exists()
        assert (b / "processes.json").exists()
        with _profile_runtime_scope(a, {}):
            restored = pr.ProcessRegistry()
            restored.recover_from_checkpoint()
            assert restored.get(results["a"].id) is not None
    finally:
        release.set()
        worker.join(10)
        for session in results.values():
            reg.kill_process(session.id, source="test")
            if session._reader_thread:
                session._reader_thread.join(10)


def test_native_descendant_inherits_positive_stamp_without_context(tmp_path, monkeypatch):
    """A real child of _spawn_env is positive; a forged session env is not."""
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER, DELEGATED_CHILD_SESSION_ENV

    monkeypatch.delenv(DELEGATED_CHILD_SESSION_ENV, raising=False)
    registry = ProcessRegistry()
    script = (
        "import json, os, sys\n"
        "sys.path.insert(0, os.environ['SRC'])\n"
        "from tools.process_registry import ProcessRegistry\n"
        "session = ProcessRegistry._new_session('echo', 't', '', 'sk', '/')\n"
        "print(json.dumps({'delegated_child': bool(session.delegated_child)}))\n"
    )
    with delegated_child_context("native-child-session"):
        env = registry._spawn_env({})
    child = subprocess.run(
        [sys.executable, "-c", script],
        env={**env, "SRC": os.getcwd(), "HERMES_HOME": str(tmp_path)},
        cwd=os.getcwd(), capture_output=True, text=True, check=False,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout.strip().splitlines()[-1])["delegated_child"] is True
    forged = dict(os.environ)
    forged[DELEGATED_CHILD_SESSION_ENV] = "forged-session"
    forged.pop(DELEGATED_CHILD_ENV_MARKER, None)
    forged["SRC"] = os.getcwd()
    forged["HERMES_HOME"] = str(tmp_path)
    bad = subprocess.run(
        [sys.executable, "-c", script], env=forged, cwd=os.getcwd(),
        capture_output=True, text=True, check=False,
    )
    assert bad.returncode == 0, bad.stderr
    assert json.loads(bad.stdout.strip().splitlines()[-1])["delegated_child"] is False
