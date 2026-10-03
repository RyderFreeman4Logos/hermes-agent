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


_NATIVE_COMPLETION_PROBE = """
import json, os, sys
sys.path.insert(0, os.environ['SRC'])
from agent.delegation_context import is_delegated_child_context
from tools.process_registry import ProcessRegistry
registry = ProcessRegistry()
session = registry.spawn_local('printf native-ok', cwd=os.environ['HERMES_HOME'],
    task_id='ordinary', owner_task_id='ordinary', notify_on_complete=True)
try:
    assert session._completion_event.wait(10)
    session._reader_thread.join(10)
    assert not session._reader_thread.is_alive()
    print('RESULT:' + json.dumps([is_delegated_child_context(),
        session.delegated_child, len(registry.drain_notifications()), session.exit_code]), flush=True)
finally:
    if not session.exited:
        registry.kill_process(session.id, source='test')
"""


def _run_native_probe(env, script=_NATIVE_COMPLETION_PROBE):
    child = subprocess.run(
        [sys.executable, '-c', script], env=env, cwd=os.getcwd(),
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert child.returncode == 0, child.stderr
    return json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith('RESULT:')))


@pytest.mark.platforms('linux')
def test_fork_during_first_issuance_does_not_inherit_listener():
    """Fork at the first issuer's start-return boundary in a fresh process."""
    probe_env = os.environ.copy()
    probe_env['PYTHONPATH'] = os.getcwd() + os.pathsep + probe_env.get('PYTHONPATH', '')
    result = subprocess.run(
        [sys.executable, __file__, '--fork-first-issuance-probe'],
        capture_output=True, text=True, timeout=30, env=probe_env,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed['issuer_threads'] == 0
    assert observed['after'] == observed['before']


def _run_fork_first_issuance_probe():
    import gc
    import select
    import signal
    import threading
    from unittest.mock import patch

    import agent.delegation_lineage as lineage

    assert lineage._issuer is None and lineage._pending is None
    ready = threading.Event()
    release = threading.Event()
    errors = []
    original = threading.Thread.start

    def started(self):
        original(self)
        if self.name == 'delegation-lineage':
            ready.set()
            assert release.wait(10)

    def issue():
        try:
            with delegated_child_context('race-fixture'):
                ProcessRegistry._spawn_env({})
        except BaseException as exc:
            errors.append(type(exc).__name__)

    def sockets():
        found = []
        for entry in os.scandir('/proc/self/fd'):
            try:
                target = os.readlink(entry.path)
            except FileNotFoundError:
                continue
            if target.startswith('socket:'):
                found.append(target)
        return found

    before = sockets()
    read_fd, write_fd = os.pipe()
    child = None
    reaped = False
    with patch.object(threading.Thread, 'start', started):
        issuer = threading.Thread(target=issue, name='fixture-issuer')
        issuer.start()
        try:
            assert ready.wait(10), errors
            child = os.fork()
            if child == 0:
                os.close(read_fd)
                try:
                    gc.collect()
                    inherited = sockets()
                    payload = json.dumps({
                        'before': len(before),
                        'after': len(inherited),
                        'issuer_threads': sum(
                            thread.name == 'delegation-lineage'
                            for thread in threading.enumerate()
                        ),
                    }).encode()
                    os.write(write_fd, payload)
                finally:
                    os.close(write_fd)
                    os._exit(0)
            os.close(write_fd)
            write_fd = -1
            assert select.select([read_fd], [], [], 10)[0]
            observed = json.loads(os.read(read_fd, 4096))
            _, status = os.waitpid(child, 0)
            reaped = True
            assert status == 0
        finally:
            if write_fd >= 0:
                os.close(write_fd)
            os.close(read_fd)
            release.set()
            issuer.join(10)
            if child is not None and not reaped:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.waitpid(child, 0)
    assert not issuer.is_alive() and not errors
    print(json.dumps(observed))


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('transport', ['direct', 'nested', 'fork', 'pipe', 'pty'])
def test_native_descendant_inherits_positive_stamp_without_context(tmp_path, transport):
    """Issued authority survives real shell/nested spawns, never child ContextVar reinjection."""
    import shlex

    extra = {'SRC': os.getcwd(), 'HERMES_HOME': str(tmp_path)}
    registry = ProcessRegistry()
    with delegated_child_context('native-child-session'):
        env = registry._spawn_env(extra)
    if transport in ('direct', 'nested', 'fork'):
        script = _NATIVE_COMPLETION_PROBE
        if transport in ('nested', 'fork'):
            script = (
                "import os, subprocess, sys\n"
                "sys.path.insert(0, os.environ['SRC'])\n"
                "from tools.process_registry import ProcessRegistry\n"
                "from agent.delegation_context import is_delegated_child_context\n"
                "assert not is_delegated_child_context()\n"
                + ("ProcessRegistry._spawn_env({})\npid = os.fork()\nif pid:\n    _, status = os.waitpid(pid, 0)\n    sys.exit(os.waitstatus_to_exitcode(status))\n" if transport == 'fork' else '')
                + "p = subprocess.run([sys.executable, '-c', " + repr(_NATIVE_COMPLETION_PROBE)
                + "], env=ProcessRegistry._spawn_env({}), capture_output=True, text=True, timeout=20)\n"
                "assert p.returncode == 0, p.stderr\nprint(p.stdout, flush=True)\n"
            )
        result = _run_native_probe(env, script)
    else:
        command = shlex.quote(sys.executable) + ' -c ' + shlex.quote(_NATIVE_COMPLETION_PROBE) + '; :'
        with delegated_child_context('native-transport-session'):
            session = registry.spawn_local(command, cwd=os.getcwd(), env_vars=extra, use_pty=transport == 'pty')
        try:
            assert session._completion_event.wait(30)
            assert session._reader_thread is not None
            session._reader_thread.join(10)
            assert not session._reader_thread.is_alive()
            assert bool(session._pty) == (transport == 'pty')
            result = json.loads(next(line[7:] for line in session.output_buffer.splitlines() if line.startswith('RESULT:')))
        finally:
            if not session.exited:
                registry.kill_process(session.id, source='test')
    assert result == [False, True, 0, 0]


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('mutation', ['ordinary', 'live-parent', 'session', 'proof', 'pid', 'start'])
def test_native_forged_authority_keeps_public_completion_visible(tmp_path, mutation):
    """Neither a genuine parent tuple nor tampering with issued authority may silence ordinary work."""
    from gateway.status import get_process_start_time

    env = {k: v for k, v in os.environ.items() if not k.startswith('HERMES_DELEGATED_CHILD_')}
    env.update(SRC=os.getcwd(), HERMES_HOME=str(tmp_path))
    if mutation == 'live-parent':
        env.update(HERMES_DELEGATED_CHILD_SESSION='forged',
                   HERMES_DELEGATED_CHILD_PARENT_PID=str(os.getpid()),
                   HERMES_DELEGATED_CHILD_PARENT_START=str(get_process_start_time(os.getpid())))
    elif mutation != 'ordinary':
        with delegated_child_context('native-issued-session'):
            issued = ProcessRegistry._spawn_env({})
        env.update({k: v for k, v in issued.items() if k.startswith('HERMES_DELEGATED_CHILD_')})
        key, value = {
            'session': ('HERMES_DELEGATED_CHILD_SESSION', 'different-session'),
            'proof': ('HERMES_DELEGATED_CHILD_PROOF', '0' * 64),
            'pid': ('HERMES_DELEGATED_CHILD_PARENT_PID', '999999999'),
            'start': ('HERMES_DELEGATED_CHILD_PARENT_START', '0'),
        }[mutation]
        env[key] = value
    assert _run_native_probe(env) == [False, False, 1, 0]


if __name__ == '__main__' and sys.argv[1:] == ['--fork-first-issuance-probe']:
    _run_fork_first_issuance_probe()
