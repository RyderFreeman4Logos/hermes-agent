"""Native delegated-child lifecycle beyond ordinary formatter silence.

Spawn freezes the child stamp from the process context. A routine child
watch is dropped before the queue; an explicit handoff still notifies.
Gateway completion events and checkpoints carry the stamp and handoff note.
"""

import json
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


def test_spawn_stamps_delegated_child_from_context_not_label():
    registry = ProcessRegistry()
    with delegated_child_context():
        child = registry._new_session("echo child", "t1", "", "sk", "/tmp")
    parent = registry._new_session("echo parent", "t1", "", "sk", "/tmp")
    assert child.delegated_child is True
    assert parent.delegated_child is False


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
