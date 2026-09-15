"""Agent teardown owns processes, not their shared terminal environment."""
import json
import shlex
import sys
import threading
import time

import pytest

from agent.turn_context import _bind_turn_identity
from run_agent import AIAgent
from tools.process_registry import ProcessRegistry
from tools.terminal_tool import terminal_tool


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _agent():
    return AIAgent(api_key="test", base_url="http://127.0.0.1:9/v1",
                   provider="openai-compat", model="test", enabled_toolsets=[],
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


def _spawn(agent, task_id, tmp_path):
    _bind_turn_identity(agent, task_id, None, None, None, None)
    command = shlex.quote(sys.executable) + " -c " + shlex.quote("import time; time.sleep(60)")
    result = json.loads(terminal_tool(command, background=True, task_id=task_id,
                                     workdir=str(tmp_path), notify_on_complete=True))
    return result["session_id"]


def test_child_close_kills_only_its_processes(tmp_path, monkeypatch):
    import tools.process_registry as processes
    registry = ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    parent, child, sibling, unstarted = [_agent() for _ in range(4)]
    try:
        parent_id = _spawn(parent, "parent-owner", tmp_path)
        child_id = _spawn(child, "sa-child-owner", tmp_path)
        sibling_id = _spawn(sibling, "sa-sibling-owner", tmp_path)
        assert len({registry._running[s].task_id for s in (parent_id, child_id, sibling_id)}) == 1
        unstarted.close()
        assert all(registry.poll(s)["status"] == "running" for s in (parent_id, child_id, sibling_id))
        child.close()
        assert registry.poll(child_id)["status"] != "running"
        assert registry.poll(parent_id)["status"] == "running"
        assert registry.poll(sibling_id)["status"] == "running"
        assert child_id in registry._completion_consumed
        child.close()
        assert registry.poll(parent_id)["status"] == "running"
    finally:
        registry.kill_all()
        for agent in (parent, child, sibling, unstarted):
            agent.close()


def test_close_reclaims_processes_from_previous_turns(tmp_path, monkeypatch):
    import tools.process_registry as processes
    registry = ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    agent = _agent()
    try:
        first = _spawn(agent, "turn-one-owner", tmp_path)
        second = _spawn(agent, "turn-two-owner", tmp_path)
        agent.close()
        assert _wait_until(
            lambda: all(registry.poll(s)["status"] != "running" for s in (first, second))
        )
    finally:
        registry.kill_all()
        agent.close()


@pytest.mark.linux_only
def test_close_reclaims_descendant_created_during_termination(tmp_path, monkeypatch):
    import psutil
    import tools.process_registry as processes

    registry = ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    agent = _agent()
    owner = "signal-fork-owner"
    late_pid_path = tmp_path / "late-child.pid"

    def late_child_alive(pid):
        try:
            return ProcessRegistry._proc_alive(psutil.Process(pid))
        except psutil.NoSuchProcess:
            return False

    _bind_turn_identity(agent, owner, None, None, None, None)
    code = (
        "import pathlib, signal, subprocess, sys, time\n"
        f"late_pid_path = pathlib.Path({str(late_pid_path)!r})\n"
        "def stop(_signum, _frame):\n"
        "    child = subprocess.Popen(\n"
        "        [sys.executable, '-c', 'import time; time.sleep(8)'],\n"
        "        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        "    )\n"
        "    late_pid_path.write_text(str(child.pid), encoding='ascii')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print('READY-SIGNAL-FORK', flush=True)\n"
        "time.sleep(60)\n"
    )
    result = json.loads(terminal_tool(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
        background=True, task_id=owner, workdir=str(tmp_path), notify_on_complete=True,
    ))
    session_id = result["session_id"]
    late_pid = None
    try:
        assert _wait_until(
            lambda: "READY-SIGNAL-FORK" in registry.poll(session_id)["output_preview"]
        )

        agent.close()

        assert _wait_until(late_pid_path.exists), "SIGTERM handler did not create its descendant"
        late_pid = int(late_pid_path.read_text(encoding="ascii"))
        assert registry.poll(session_id)["status"] != "running"
        assert _wait_until(lambda: not late_child_alive(late_pid))
    finally:
        if late_pid is not None:
            assert _wait_until(lambda: not late_child_alive(late_pid), timeout=10)
        registry.kill_all()
        agent.close()


@pytest.mark.linux_only
def test_close_preserves_tail_when_reader_settles_after_return(tmp_path, monkeypatch):
    import tools.process_registry as processes
    registry = ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(processes, "save_completed_result", lambda _session: None)
    agent = _agent()
    _bind_turn_identity(agent, "delayed-reader-owner", None, None, None, None)
    entered_finish = threading.Event()
    release_finish = threading.Event()
    original_finish = registry._finish_reader

    def held_finish(*args, **kwargs):
        entered_finish.set()
        assert release_finish.wait(5), "test did not release terminal reader"
        return original_finish(*args, **kwargs)

    monkeypatch.setattr(registry, "_finish_reader", held_finish)
    code = (
        "import signal, time\n"
        "def stop(_signum, _frame):\n"
        "    print('FINAL-DELAYED-READER', flush=True)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print('READY-DELAYED-READER', flush=True)\n"
        "time.sleep(60)\n"
    )
    result = json.loads(terminal_tool(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
        background=True, task_id="delayed-reader-owner",
        workdir=str(tmp_path), notify_on_complete=True,
    ))
    session = registry.get(result["session_id"])
    assert session is not None
    assert _wait_until(lambda: "READY-DELAYED-READER" in session.output_buffer)
    assert session.process.poll() is None
    monkeypatch.setattr(
        session._reader_settlement_ready,
        "wait",
        lambda timeout=None: False,
    )
    try:
        agent.close()

        assert entered_finish.wait(2), "reader did not reach terminal settlement"
        assert session.process.wait(timeout=2) is not None
        assert registry.poll(session.id)["status"] == "running"

        release_finish.set()
        assert _wait_until(lambda: registry.poll(session.id)["status"] != "running")
        assert "FINAL-DELAYED-READER" in session.output_buffer
    finally:
        release_finish.set()
        registry.kill_all()
        agent.close()
