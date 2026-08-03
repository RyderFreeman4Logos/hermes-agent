"""Foreground-release regressions for long terminal calls."""

from __future__ import annotations

import json
import os
import select
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _config(**overrides):
    config = {
        "env_type": "local",
        "timeout": 21,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "auto_background_timeout_threshold": 20,
    }
    config.update(overrides)
    return config


def _session():
    return SimpleNamespace(
        id="proc_auto_bg",
        pid=4242,
        notify_on_complete=False,
        watch_patterns=[],
        watcher_platform="",
        watcher_chat_id="",
        watcher_user_id="",
        watcher_user_name="",
        watcher_thread_id="",
        watcher_message_id="",
        watcher_interval=0,
    )


def _run(
    *, command="make build", config=None, async_delivery=True,
    fresh_environment=False, **kwargs,
):
    from tools.terminal_tool import terminal_tool

    env = MagicMock(env={})
    env.execute.return_value = {"output": "foreground", "returncode": 0}
    proc = _session()
    registry = MagicMock(pending_watchers=[])
    def fake_spawn(**spawn_kwargs):
        for key, value in spawn_kwargs.get("notification_metadata", {}).items():
            setattr(proc, key, value)
        return proc

    registry.spawn_local.side_effect = fake_spawn
    registry.spawn_via_env.side_effect = fake_spawn
    registry.wait_for_promotion.return_value = "running"

    def fake_promote(session, notification_metadata=None):
        for key, value in (notification_metadata or {}).items():
            setattr(session, key, value)
        return True

    registry.promote.side_effect = fake_promote
    registry.discard.return_value = {"status": "killed"}
    create_environment = MagicMock(return_value=env)
    active_environments = {} if fresh_environment else {"default": env}

    async_probe = MagicMock()
    if isinstance(async_delivery, Exception):
        async_probe.side_effect = async_delivery
    else:
        async_probe.return_value = async_delivery

    with (
        patch("tools.terminal_tool._get_env_config", return_value=config or _config()),
        patch("tools.terminal_tool._start_cleanup_thread"),
        patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}),
        patch("tools.terminal_tool._active_environments", active_environments),
        patch("tools.terminal_tool._last_activity", {"default": 0}),
        patch("tools.terminal_tool._create_environment", create_environment),
        patch("tools.process_registry.process_registry", registry),
        patch("tools.approval.get_current_session_key", return_value=""),
        patch("gateway.session_context.async_delivery_supported", async_probe),
        patch("gateway.session_context.get_session_env", return_value=""),
        patch("tools.runtime_heartbeat.preflight_current_heartbeat", return_value=None),
        patch("tools.runtime_heartbeat.runtime_heartbeat.arm"),
    ):
        result = json.loads(terminal_tool(command=command, **kwargs))

    return result, env, proc, registry, create_environment


@pytest.fixture
def terminal_runtime(monkeypatch):
    import tools.async_delegation as async_delegation
    import tools.process_registry as process_module
    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        async_delegation, "restore_undelivered_completions", lambda _queue: 0
    )
    registry = process_module.ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(process_module, "process_registry", registry)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: _config(timeout=2, auto_background_timeout_threshold=1),
    )
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {"default": SimpleNamespace(env={})},
    )
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0})
    monkeypatch.setattr(
        terminal_tool,
        "_create_environment",
        lambda **_kwargs: pytest.fail("cached local environment should be reused"),
    )
    monkeypatch.setattr("tools.approval.get_current_session_key", lambda default="": "")
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: True
    )
    monkeypatch.setattr(
        "gateway.session_context.get_session_env", lambda *_args, **_kwargs: ""
    )
    monkeypatch.setattr(
        "tools.runtime_heartbeat.preflight_current_heartbeat", lambda: None
    )
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.arm",
        lambda *_args, **_kwargs: False,
    )
    return terminal_tool, registry


def _wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _pid_is_running(pid):
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or stat.read_text(encoding="utf-8").split()[2] != "Z"
    except OSError:
        return False


class _TimedRemoteEnvironment:
    def __init__(
        self,
        *,
        done_after=30.0,
        kill_result=None,
        poll_failure_stage=None,
        poll_failures=0,
    ):
        self.done_after = done_after
        self.kill_result = kill_result or {"output": "", "returncode": 0}
        self.poll_failure_stage = poll_failure_stage
        self.poll_failures = poll_failures
        self.started_at = None
        self.killed = False
        self.poll_calls = 0
        self.poll_times = []
        self.stage_calls = {"status": 0, "log": 0, "exit": 0}
        self.kill_calls = []

    def _done(self):
        return self.killed or (
            self.started_at is not None
            and time.monotonic() - self.started_at >= self.done_after
        )

    def _poll_stage(self, stage):
        self.stage_calls[stage] += 1
        if stage == "status":
            self.poll_calls += 1
            self.poll_times.append(time.monotonic())
        if self.poll_failure_stage == stage and self.poll_failures:
            self.poll_failures -= 1
            raise RuntimeError(f"transient {stage} transport failure")

    def execute(self, command, **kwargs):
        if "nohup setsid bash -lc" in command:
            self.started_at = time.monotonic()
            return {"output": "4242\n", "returncode": 0}
        if command.startswith("kill -TERM"):
            self.kill_calls.append({"command": command, **kwargs})
            result = dict(self.kill_result)
            if result.get("returncode") == 0:
                self.killed = True
            return result
        if command.startswith("kill -0"):
            self._poll_stage("status")
            return {"output": "1\n" if self._done() else "0\n", "returncode": 0}
        if command.startswith("cat ") and ".exit" in command:
            self._poll_stage("exit")
            return {"output": "0\n" if self._done() else "", "returncode": 0}
        if command.startswith("cat ") and ".log" in command:
            self._poll_stage("log")
            return {
                "output": "remote done\n" if self._done() else "",
                "returncode": 0,
            }
        raise AssertionError(f"unexpected remote command: {command}")


class _ShellRemoteEnvironment:
    """Execute the nonlocal shell contract without faking artifact reads."""

    def __init__(self, temp_dir):
        self.temp_dir = temp_dir
        self.commands = []
        self.transports = []

    def get_temp_dir(self):
        return str(self.temp_dir)

    def execute(self, command, timeout=10, **_kwargs):
        self.commands.append(command)
        if "nohup setsid bash -lc" in command and "while [ ! -s" in command:
            transport = subprocess.Popen(
                ["/bin/bash", "-lc", command],
                cwd=self.temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.transports.append(transport)
            assert transport.stdout is not None
            ready, _, _ = select.select([transport.stdout], [], [], timeout)
            output = transport.stdout.readline() if ready else ""
            return {
                "output": output,
                "returncode": 0 if output.strip().isdigit() else 124,
            }

        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=self.temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {"output": completed.stdout, "returncode": completed.returncode}

    def cleanup(self):
        for transport in self.transports:
            try:
                transport.wait(timeout=2)
            except subprocess.TimeoutExpired:
                transport.kill()
                transport.wait(timeout=2)


@pytest.mark.parametrize("requested_timeout", [2, None], ids=["requested", "configured"])
def test_short_command_with_large_timeout_stays_inline(
    terminal_runtime, requested_timeout
):
    terminal_tool, registry = terminal_runtime
    kwargs = {"timeout": requested_timeout} if requested_timeout is not None else {}

    result = json.loads(
        terminal_tool.terminal_tool(
            command="printf short",
            notify_on_complete=True,
            **kwargs,
        )
    )

    assert result["output"] == "short"
    assert result["exit_code"] == 0
    assert "session_id" not in result
    assert registry.list_sessions() == []
    assert registry.completion_queue.empty()


def test_completion_followup_does_not_start_a_completion_self_loop(terminal_runtime):
    terminal_tool, registry = terminal_runtime
    command = shlex.join(
        [sys.executable, "-c", "import time; time.sleep(1.2); print('long')"]
    )

    promoted = json.loads(terminal_tool.terminal_tool(command=command, timeout=2))
    assert promoted["notify_on_complete"] is True
    completion = registry.completion_queue.get(timeout=5)
    assert completion["session_id"] == promoted["session_id"]
    assert registry.completion_queue.empty()

    followup = json.loads(
        terminal_tool.terminal_tool(
            command="printf status",
            timeout=2,
        )
    )
    assert followup["output"] == "status"
    assert "session_id" not in followup
    time.sleep(0.2)
    assert registry.completion_queue.empty()
    assert [item["session_id"] for item in registry.list_sessions()] == [
        promoted["session_id"]
    ]


def test_unsupported_async_delivery_kills_unregistered_candidate(
    terminal_runtime, monkeypatch, tmp_path
):
    terminal_tool, registry = terminal_runtime
    pid_path = tmp_path / "candidate.pid"
    command = shlex.join(
        [
            sys.executable,
            "-c",
            (
                "import os,pathlib,time; "
                f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
                "time.sleep(30)"
            ),
        ]
    )
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: False
    )

    result = json.loads(terminal_tool.terminal_tool(command=command, timeout=2))

    assert "cannot deliver" in result["error"]
    assert pid_path.exists()
    assert _wait_until(lambda: not _pid_is_running(int(pid_path.read_text())))
    assert registry.list_sessions() == []
    assert registry.completion_queue.empty()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-tree regression")
@pytest.mark.parametrize("stage", ["wait", "promote", "deadline"])
@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit]
)
def test_deferred_candidate_is_contained_on_lifecycle_exception(
    terminal_runtime, monkeypatch, tmp_path, stage, error_type
):
    import tools.process_registry as process_module

    terminal_tool, registry = terminal_runtime
    child_pid_path = tmp_path / f"{stage}-{error_type.__name__}.pid"
    child_code = "import time; time.sleep(30)"
    parent_code = (
        "import pathlib,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid)); "
        "time.sleep(30)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_code)}"
    original_spawn = registry.spawn_local
    original_promote = registry.promote
    original_prune = registry._prune_if_needed
    original_timer = process_module.threading.Timer
    captured = []

    def capture_spawn(**kwargs):
        session = original_spawn(**kwargs)
        captured.append(session)
        return session

    def reach_exception_point(*_args, **_kwargs):
        assert _wait_until(child_pid_path.exists)
        if stage == "wait":
            raise error_type(f"injected {error_type.__name__}")
        return "running"

    def fail_prune():
        if stage == "promote":
            raise error_type(f"injected {error_type.__name__}")
        if stage == "deadline":
            return original_prune()
        raise RuntimeError("persistent cleanup promotion failure")

    class FailingDeadlineTimer:
        daemon = False

        def start(self):
            raise error_type(f"injected {error_type.__name__}")

        def cancel(self):
            pass

    def timer_factory(*args, **kwargs):
        if stage == "deadline":
            return FailingDeadlineTimer()
        return original_timer(*args, **kwargs)

    monkeypatch.setattr(registry, "spawn_local", capture_spawn)
    monkeypatch.setattr(registry, "wait_for_promotion", reach_exception_point)
    monkeypatch.setattr(registry, "_prune_if_needed", fail_prune)
    monkeypatch.setattr(process_module.threading, "Timer", timer_factory)
    monkeypatch.setattr(registry, "_daemon_term_grace_seconds", lambda: 0.1)

    try:
        if issubclass(error_type, Exception):
            result = json.loads(
                terminal_tool.terminal_tool(command=command, timeout=2)
            )
            assert f"injected {error_type.__name__}" in result["error"]
        else:
            with pytest.raises(error_type, match=f"injected {error_type.__name__}"):
                terminal_tool.terminal_tool(command=command, timeout=2)

        session = captured[0]
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert _wait_until(lambda: not _pid_is_running(session.pid))
        assert _wait_until(lambda: not _pid_is_running(child_pid))
        assert registry.list_sessions() == []
        assert registry.completion_queue.empty()
    finally:
        monkeypatch.setattr(registry, "_prune_if_needed", original_prune)
        if captured and _pid_is_running(captured[0].pid):
            original_promote(captured[0])
            registry.kill_process(
                captured[0].id,
                source="test_emergency_cleanup",
                consume_output=False,
            )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-tree regression")
@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit]
)
def test_outer_rescue_keeps_managed_identity_when_discard_and_kill_fail(
    terminal_runtime, monkeypatch, tmp_path, error_type
):
    terminal_tool, registry = terminal_runtime
    child_pid_path = tmp_path / f"outer-rescue-{error_type.__name__}.pid"
    child_code = "import time; time.sleep(30)"
    parent_code = (
        "import pathlib,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid)); "
        "time.sleep(30)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_code)}"
    original_spawn = registry.spawn_local
    original_terminate = registry._terminate_host_pid
    captured = []

    def capture_spawn(**kwargs):
        session = original_spawn(**kwargs)
        captured.append(session)
        return session

    def fail_wait(*_args, **_kwargs):
        assert _wait_until(child_pid_path.exists)
        raise error_type(f"injected {error_type.__name__}")

    monkeypatch.setattr(registry, "spawn_local", capture_spawn)
    monkeypatch.setattr(registry, "wait_for_promotion", fail_wait)
    monkeypatch.setattr(
        registry,
        "discard",
        MagicMock(side_effect=RuntimeError("persistent discard failure")),
    )
    monkeypatch.setattr(
        registry,
        "_terminate_host_pid",
        MagicMock(side_effect=RuntimeError("cleanup kill status unknown")),
    )

    try:
        if issubclass(error_type, Exception):
            result = json.loads(
                terminal_tool.terminal_tool(command=command, timeout=2)
            )
            assert result["session_id"] == captured[0].id
            assert "cleanup could not be confirmed" in result["error"]
            assert "persistent discard failure" in result["cleanup_error"]
            assert "cleanup kill status unknown" in result["cleanup_error"]
        else:
            with pytest.raises(error_type, match=f"injected {error_type.__name__}"):
                terminal_tool.terminal_tool(command=command, timeout=2)

        session = captured[0]
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert _pid_is_running(session.pid)
        assert _pid_is_running(child_pid)
        assert registry.get(session.id) is session
        assert session._deadline_timer is not None
        assert session._deadline_timer.is_alive()
    finally:
        if captured:
            monkeypatch.setattr(registry, "_terminate_host_pid", original_terminate)
            if registry.get(captured[0].id) is not None:
                registry.kill_process(
                    captured[0].id,
                    source="test_emergency_cleanup",
                    consume_output=False,
                )
            if _pid_is_running(captured[0].pid):
                original_terminate(captured[0].pid, captured[0].host_start_time)


@pytest.mark.parametrize(
    ("kill_result", "provenance"),
    [
        pytest.param(
            {"output": "permission denied", "returncode": 1},
            "permission denied",
            id="nonzero",
        ),
        pytest.param(
            {"output": "transport timed out", "returncode": 124},
            "returncode=124",
            id="timeout",
        ),
        pytest.param(
            {"output": "transport status unknown"},
            "status unknown",
            id="uncertain",
        ),
    ],
)
def test_nonlocal_discard_failure_keeps_managed_identity(
    terminal_runtime, kill_result, provenance
):
    from tools.process_registry import ProcessSession

    _, registry = terminal_runtime
    env = _TimedRemoteEnvironment(kill_result=kill_result)
    session = ProcessSession(
        id="proc_remote_kill_failure",
        command="sleep 30",
        pid=4242,
        env_ref=env,
        pid_scope="sandbox",
    )

    result = registry.discard(session, source="background_promotion_failed")

    assert result["status"] == "error"
    assert result["session_id"] == session.id
    assert result["termination_source"] == "background_promotion_failed"
    assert provenance in result["error"]
    assert registry.get(session.id) is session
    assert session.exited is False
    assert session.completion_reason == "exited"
    assert session.termination_source == ""

    env.kill_result = {"output": "", "returncode": 0}
    assert registry.discard(session, source="retry_cleanup")["status"] == "killed"
    assert registry.get(session.id) is None


def test_nonlocal_discard_success_removes_registry_ownership(terminal_runtime):
    from tools.process_registry import ProcessSession

    _, registry = terminal_runtime
    env = _TimedRemoteEnvironment(kill_result={"output": "", "returncode": 0})
    session = ProcessSession(
        id="proc_remote_kill_success",
        command="sleep 30",
        pid=4242,
        env_ref=env,
        pid_scope="sandbox",
    )

    result = registry.discard(session, source="background_promotion_failed")

    assert result["status"] == "killed"
    assert result["termination_source"] == "background_promotion_failed"
    assert session.exited is True
    assert registry.get(session.id) is None
    assert len(env.kill_calls) == 1


def test_nonlocal_completion_before_minimum_threshold_stays_inline(
    terminal_runtime, monkeypatch
):
    terminal_tool, registry = terminal_runtime
    env = _TimedRemoteEnvironment(done_after=0.05)
    terminal_tool._active_environments["default"] = env
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: _config(
            env_type="docker", timeout=2, auto_background_timeout_threshold=1
        ),
    )

    result = json.loads(
        terminal_tool.terminal_tool(command="printf remote", timeout=2)
    )
    try:
        assert result["output"] == "remote done"
        assert result["exit_code"] == 0
        assert "session_id" not in result
        assert registry.list_sessions() == []
        assert registry.completion_queue.empty()
        assert 1 <= env.poll_calls <= 2
    finally:
        registry.kill_all()


def test_nonlocal_completion_after_threshold_promotes_and_notifies_once(
    terminal_runtime, monkeypatch
):
    terminal_tool, registry = terminal_runtime
    env = _TimedRemoteEnvironment(done_after=1.2)
    terminal_tool._active_environments["default"] = env
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: _config(
            env_type="docker", timeout=4, auto_background_timeout_threshold=1
        ),
    )

    result = json.loads(
        terminal_tool.terminal_tool(command="printf remote", timeout=4)
    )
    session_id = result["session_id"]
    try:
        assert result["notify_on_complete"] is True
        event = registry.completion_queue.get(timeout=4)
        assert event["session_id"] == session_id
        assert event["output"] == "remote done\n"
        time.sleep(0.1)
        assert registry.completion_queue.empty()
        assert env.poll_calls <= 4
    finally:
        if session_id in registry._running:
            registry.kill_process(session_id, consume_output=False)


@pytest.mark.parametrize("poll_stage", ["status", "log", "exit"])
@pytest.mark.parametrize(
    ("done_after", "promoted"),
    [(0.0, False), (1.2, True)],
    ids=["before-threshold-inline", "after-threshold-promoted"],
)
def test_nonlocal_transient_poll_error_recovers_without_forging_exit(
    terminal_runtime, monkeypatch, poll_stage, done_after, promoted
):
    terminal_tool, registry = terminal_runtime
    env = _TimedRemoteEnvironment(
        done_after=done_after,
        poll_failure_stage=poll_stage,
        poll_failures=1,
    )
    terminal_tool._active_environments["default"] = env
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: _config(
            env_type="docker", timeout=6, auto_background_timeout_threshold=1
        ),
    )

    result = json.loads(
        terminal_tool.terminal_tool(command="printf remote", timeout=6)
    )
    try:
        if not promoted:
            assert result["output"] == "remote done"
            assert result["exit_code"] == 0
            assert "session_id" not in result
            assert registry.list_sessions() == []
            assert registry.completion_queue.empty()
        else:
            session_id = result["session_id"]
            session = registry.get(session_id)
            assert session is not None
            assert session.exited is False
            assert session.execution_deadline == pytest.approx(
                session.started_at + 6, abs=0.05
            )
            assert session._deadline_timer is not None
            event = registry.completion_queue.get(timeout=6)
            assert event["session_id"] == session_id
            assert event["exit_code"] == 0
            assert event["output"] == "remote done\n"
            final = registry.poll(session_id)
            assert final["status"] == "exited"
            assert poll_stage in final["poll_error"]
            assert registry.completion_queue.empty()
        assert env.poll_failures == 0
        assert env.poll_calls <= 4
        assert all(
            later - earlier >= 0.5
            for earlier, later in zip(env.poll_times, env.poll_times[1:])
        )
    finally:
        registry.kill_all()


def test_continuous_nonlocal_poll_errors_stay_managed_and_killable(
    terminal_runtime, monkeypatch
):
    terminal_tool, registry = terminal_runtime
    env = _TimedRemoteEnvironment(
        done_after=30,
        poll_failure_stage="status",
        poll_failures=50,
    )
    terminal_tool._active_environments["default"] = env
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: _config(
            env_type="docker", timeout=4, auto_background_timeout_threshold=1
        ),
    )

    result = json.loads(
        terminal_tool.terminal_tool(command="sleep 30", timeout=4)
    )
    session_id = result["session_id"]
    try:
        session = registry.get(session_id)
        assert session is not None
        assert session.exited is False
        assert session.execution_deadline == pytest.approx(
            session.started_at + 4, abs=0.05
        )
        assert session._deadline_timer is not None
        assert session._deadline_timer.is_alive()
        assert "status" in result["poll_error"]
        assert "status" in registry.poll(session_id)["poll_error"]
        assert env.poll_calls <= 2

        killed = registry.kill_process(
            session_id,
            source="test_continuous_poll_cleanup",
            consume_output=False,
        )
        assert killed["status"] == "killed"
        event = registry.completion_queue.get(timeout=2)
        assert event["session_id"] == session_id
        assert event["termination_source"] == "test_continuous_poll_cleanup"
        assert registry.completion_queue.empty()
    finally:
        registry.kill_all()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX nonlocal shell contract")
@pytest.mark.parametrize(
    ("sleep_seconds", "promoted"),
    [(0.15, False), (1.4, True)],
    ids=["before-threshold-inline", "after-threshold-promoted"],
)
def test_shell_backed_nonlocal_threshold_contract(
    terminal_runtime, monkeypatch, tmp_path, sleep_seconds, promoted
):
    terminal_tool, registry = terminal_runtime
    env = _ShellRemoteEnvironment(tmp_path)
    terminal_tool._active_environments["default"] = env
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: _config(
            env_type="docker",
            timeout=4,
            cwd=str(tmp_path),
            auto_background_timeout_threshold=1,
        ),
    )
    marker = "remote-after-threshold" if promoted else "remote-before-threshold"
    command = f"sleep {sleep_seconds}; printf '%s\\n' {shlex.quote(marker)}"

    try:
        result = json.loads(terminal_tool.terminal_tool(command=command, timeout=4))
        if not promoted:
            assert result["output"].strip() == marker
            assert result["exit_code"] == 0
            assert "session_id" not in result
            assert registry.list_sessions() == []
            time.sleep(0.25)
            assert registry.completion_queue.empty()
        else:
            session_id = result["session_id"]
            assert result["notify_on_complete"] is True
            event = registry.completion_queue.get(timeout=5)
            assert event["session_id"] == session_id
            assert event["output"].strip() == marker
            time.sleep(0.25)
            assert registry.completion_queue.empty()
            assert [session["session_id"] for session in registry.list_sessions()] == [
                session_id
            ]
        assert sum(command.startswith("kill -0") for command in env.commands) <= 4
    finally:
        registry.kill_all()
        env.cleanup()


@pytest.mark.parametrize(
    ("config", "kwargs"),
    [
        pytest.param(_config(timeout=21), {}, id="omitted-timeout-and-flags"),
        pytest.param(_config(timeout=5), {"timeout": 21}, id="explicit-timeout"),
        pytest.param(
            _config(timeout=5),
            {"timeout": 21, "background": False, "notify_on_complete": False},
            id="explicit-unsafe-false",
        ),
    ],
)
def test_long_foreground_calls_promote_to_managed_background(config, kwargs):
    result, env, proc, registry, _ = _run(config=config, **kwargs)

    assert result["session_id"] == proc.id
    assert result["notify_on_complete"] is True
    assert proc.notify_on_complete is True
    registry.spawn_local.assert_called_once()
    registry.wait_for_promotion.assert_called_once_with(proc, 20)
    registry.promote.assert_called_once()
    env.execute.assert_not_called()


def test_auto_promotion_preserves_requested_execution_deadline():
    result, _, _, registry, create_environment = _run(
        timeout=21,
        background=False,
        notify_on_complete=False,
        fresh_environment=True,
    )

    assert result["session_id"] == "proc_auto_bg"
    assert create_environment.call_args.kwargs["timeout"] == 21
    assert registry.spawn_local.call_args.kwargs["execution_timeout"] == 21
    registry.spawn_local.assert_called_once()


def test_auto_promotion_carries_deadline_to_remote_spawn():
    result, _, _, registry, _ = _run(
        config=_config(env_type="docker"),
        timeout=21,
        background=False,
    )

    assert result["session_id"] == "proc_auto_bg"
    assert registry.spawn_via_env.call_args.kwargs["execution_timeout"] == 21
    assert registry.spawn_via_env.call_args.kwargs["defer_registration"] is True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-tree regression")
def test_promoted_execution_deadline_kills_process_tree_once(
    terminal_runtime, tmp_path
):
    terminal_tool, registry = terminal_runtime
    child_pid_path = tmp_path / "child.pid"
    child_code = "import time; time.sleep(30)"
    parent_code = (
        "import pathlib,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid)); "
        "print('parent-ready', flush=True); time.sleep(30)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_code)}"

    result = json.loads(
        terminal_tool.terminal_tool(command=command, timeout=2, background=False)
    )
    session_id = result["session_id"]
    child_pid = None
    try:
        assert _wait_until(child_pid_path.exists)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert _pid_is_running(child_pid)
        session = registry._running.get(session_id) or registry._finished[session_id]
        assert session._completion_event.wait(5)
        assert _wait_until(lambda: not _pid_is_running(child_pid))
        assert registry.poll(session_id)["status"] == "exited"

        events = []
        while not registry.completion_queue.empty():
            events.append(registry.completion_queue.get_nowait())
        matching = [event for event in events if event.get("session_id") == session_id]
        assert len(matching) == 1
        assert matching[0]["completion_reason"] == "killed"
        assert matching[0]["termination_source"] == "execution_timeout"
        assert "parent-ready" in matching[0]["output"]
    finally:
        if session_id in registry._running:
            registry.kill_process(session_id)
        if child_pid is not None and _pid_is_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_completion_before_spawn_returns_stays_inline_without_notification(
    terminal_runtime, monkeypatch
):
    terminal_tool, registry = terminal_runtime
    original_spawn = registry.spawn_local

    def complete_before_return(**kwargs):
        session = original_spawn(**kwargs)
        assert session._completion_event.wait(3)
        return session

    monkeypatch.setattr(registry, "spawn_local", complete_before_return)
    result = json.loads(
        terminal_tool.terminal_tool(command="true", timeout=2, background=False)
    )

    assert result["exit_code"] == 0
    assert "session_id" not in result
    assert registry.list_sessions() == []
    assert registry.completion_queue.empty()


def test_timeout_at_threshold_stays_foreground():
    result, env, _, registry, _ = _run(timeout=20)

    assert result["output"] == "foreground"
    env.execute.assert_called_once()
    registry.spawn_local.assert_not_called()


def test_explicit_background_call_is_not_rewritten_or_forced_to_notify():
    result, env, proc, registry, _ = _run(
        timeout=21,
        background=True,
        notify_on_complete=False,
    )

    assert result["session_id"] == proc.id
    assert "notify_on_complete" not in result
    assert proc.notify_on_complete is False
    registry.spawn_local.assert_called_once()
    assert "execution_timeout" not in registry.spawn_local.call_args.kwargs
    assert "defer_registration" not in registry.spawn_local.call_args.kwargs
    env.execute.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 21, "command": "python -m http.server"},
        {"timeout": 21, "watch_patterns": ["ready"]},
    ],
)
def test_server_and_watch_requests_keep_explicit_background_guidance(kwargs):
    command = kwargs.pop("command", "make build")
    result, env, _, registry, _ = _run(command=command, **kwargs)

    assert "background=true" in result["error"]
    env.execute.assert_not_called()
    registry.spawn_local.assert_not_called()


@pytest.mark.parametrize(
    "async_delivery",
    [False, RuntimeError("probe failed")],
    ids=["unsupported", "probe-error"],
)
def test_auto_promotion_stops_without_completion_route(async_delivery):
    result, env, proc, registry, _ = _run(
        timeout=21,
        background=False,
        async_delivery=async_delivery,
    )

    assert "stopped" in result["error"].lower() or "setup failed" in result[
        "error"
    ].lower()
    env.execute.assert_not_called()
    registry.spawn_local.assert_called_once()
    registry.wait_for_promotion.assert_called_once_with(proc, 20)
    registry.discard.assert_called_once()
    registry.promote.assert_not_called()


def test_threshold_is_read_from_active_config_without_an_env_bridge(tmp_path, monkeypatch):
    import tools.terminal_tool as terminal_tool

    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "terminal:\n  auto_background_timeout_threshold: 7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(terminal_tool, "_ensure_terminal_env_bridged", lambda: None)

    assert terminal_tool._get_env_config()["auto_background_timeout_threshold"] == 7


def test_terminal_schema_describes_auto_promotion():
    from hermes_cli.config import DEFAULT_CONFIG
    from tools.terminal_tool import TERMINAL_SCHEMA, TERMINAL_TOOL_DESCRIPTION

    description = TERMINAL_SCHEMA["parameters"]["properties"]["timeout"]["description"]
    assert DEFAULT_CONFIG["terminal"]["auto_background_timeout_threshold"] == 200
    assert "auto_background_timeout_threshold" in description
    assert "still running" in description
    assert "execution deadline" in description
    assert "auto_background_timeout_threshold" in TERMINAL_TOOL_DESCRIPTION
