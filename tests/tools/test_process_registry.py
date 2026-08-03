"""Tests for tools/process_registry.py — ProcessRegistry query methods, pruning, checkpoint."""

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from tools.environments.local import _HERMES_PROVIDER_ENV_FORCE_PREFIX
from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    FINISHED_TTL_SECONDS,
    MAX_PROCESSES,
    MAX_ACTIVE_PROCESS_AGE,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


def _make_session(
    sid="proc_test123",
    command="echo hello",
    task_id="t1",
    exited=False,
    exit_code=None,
    output="",
    started_at=None,
) -> ProcessSession:
    """Helper to create a ProcessSession for testing."""
    s = ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=started_at or time.time(),
        exited=exited,
        exit_code=exit_code,
        output_buffer=output,
    )
    return s


def test_terminal_claim_cancels_heartbeat_before_completion_publication(
    registry, monkeypatch
):
    from tools.runtime_heartbeat import runtime_heartbeat

    order = []
    session = _make_session(sid="proc-heartbeat", exited=True, exit_code=0)
    session.notify_on_complete = True
    session.session_key = "owner"
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(
        runtime_heartbeat,
        "cancel",
        lambda target_id: order.append(("cancel", target_id)) or True,
    )

    class RecordingQueue:
        def put(self, event):
            order.append(("publish", event["session_id"]))

    registry.completion_queue = RecordingQueue()
    registry._move_to_finished(session)

    assert order == [
        ("cancel", "proc-heartbeat"),
        ("publish", "proc-heartbeat"),
    ]


def test_completion_event_fences_checkpoint_and_notification_publication(
    registry, monkeypatch
):
    session = _make_session(sid="proc-publication-fence", exited=True, exit_code=0)
    session.notify_on_complete = True
    registry._running[session.id] = session
    checkpoint_started = threading.Event()
    release_checkpoint = threading.Event()

    def blocking_checkpoint():
        checkpoint_started.set()
        assert release_checkpoint.wait(2)

    monkeypatch.setattr(registry, "_write_checkpoint", blocking_checkpoint)
    worker = threading.Thread(target=registry._move_to_finished, args=(session,))
    worker.start()
    assert checkpoint_started.wait(2)
    try:
        assert not session._completion_event.is_set()
        assert registry.completion_queue.empty()
    finally:
        release_checkpoint.set()
        worker.join(2)

    assert not worker.is_alive()
    assert session._completion_event.is_set()
    assert registry.completion_queue.get_nowait()["session_id"] == session.id
    assert registry.completion_queue.empty()


def _spawn_python_sleep(seconds: float) -> subprocess.Popen:
    """Spawn a portable short-lived Python sleep process."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
    )


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll a predicate until it returns truthy or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_write_stdin_uses_str_for_windows_pty(monkeypatch, registry):
    """pywinpty expects str input; bytes raises a PyString conversion error."""
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-win")
    session._pty = _FakePty()
    registry._running[session.id] = session
    monkeypatch.setattr("tools.process_registry._IS_WINDOWS", True)

    result = registry.write_stdin(session.id, "hello\n")

    assert result == {"status": "ok", "bytes_written": 6}
    assert written == ["hello\n"]
    assert isinstance(written[0], str)


# =========================================================================
# Get / Poll
# =========================================================================

class TestGetAndPoll:
    def test_poll_running(self, registry):
        s = _make_session(output="some output here")
        registry._running[s.id] = s
        result = registry.poll(s.id)
        assert result["status"] == "running"
        assert "some output" in result["output_preview"]
        assert result["command"] == "echo hello"

    def test_poll_exited(self, registry):
        s = _make_session(exited=True, exit_code=0, output="done")
        registry._finished[s.id] = s
        result = registry.poll(s.id)
        assert result["status"] == "exited"
        assert result["exit_code"] == 0


def test_request_close_terminal_invokes_sink_without_killing(registry):
    """With a sink wired, close routes (session, process_id) to the UI and leaves
    the process running — close is a view drop, not a kill."""
    s = _make_session(sid="proc_close_live")
    registry._running[s.id] = s
    calls = []
    registry.on_close = lambda session, pid: calls.append((session, pid))

    result = registry.request_close_terminal(s.id)

    assert result["status"] == "ok"
    assert result["closed"] == "proc_close_live"
    assert calls == [(s, "proc_close_live")]
    # Still tracked as running — closing the tab must not reap the process.
    assert s.id in registry._running


def test_reader_loop_streams_incremental_chunks_from_read1(registry, monkeypatch):
    """Local reader must emit live chunks, not one EOF burst.

    Regression for desktop agent terminals: ``stdout.read(4096)`` can buffer
    until process exit for small periodic output. ``buffer.read1(4096)`` should
    surface each chunk as it arrives.
    """

    class _FakeBuffer:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        def read1(self, _n):
            if self._chunks:
                return self._chunks.pop(0)
            return b""

    class _FakeStdout:
        def __init__(self, chunks):
            self.buffer = _FakeBuffer(chunks)

    class _FakeProcess:
        def __init__(self, chunks):
            self.stdout = _FakeStdout(chunks)
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

    session = _make_session(sid="proc_reader_live")
    session.process = _FakeProcess([b"tick 1\n", b"tick 2\n", b"tick 3\n", b""])
    emitted = []
    moved = []

    monkeypatch.setattr(registry, "_check_watch_patterns", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_emit_output", lambda _s, chunk: emitted.append(chunk))
    monkeypatch.setattr(registry, "_move_to_finished", lambda _s: moved.append(_s.id))

    registry._reader_loop(session)

    assert emitted == ["tick 1\n", "tick 2\n", "tick 3\n"]
    assert session.output_buffer == "tick 1\ntick 2\ntick 3\n"
    assert session.output_size == len("tick 1\ntick 2\ntick 3\n")
    assert session.exited is True
    assert session.exit_code == 0
    assert moved == ["proc_reader_live"]


def test_reader_activity_counter_advances_after_rolling_buffer_saturates(
    registry, monkeypatch
):
    class _Buffer:
        def __init__(self):
            self.chunks = [b"abcde", b"fghij", b""]

        def read1(self, _size):
            return self.chunks.pop(0)

    process = MagicMock()
    process.stdout.buffer = _Buffer()
    process.wait.return_value = 0
    process.returncode = 0
    session = _make_session(sid="proc-saturated")
    session.process = process
    session.max_output_chars = 5
    monkeypatch.setattr(registry, "_check_watch_patterns", lambda *_args: None)
    monkeypatch.setattr(registry, "_emit_output", lambda *_args: None)
    monkeypatch.setattr(registry, "_move_to_finished", lambda *_args: None)

    registry._reader_loop(session)

    assert session.output_buffer == "fghij"
    assert session.output_size == 10


# =========================================================================
# Orphaned-pipe reconciliation (issue #17327)
# =========================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: uses setsid/fcntl")
class TestOrphanedPipeReconciliation:
    """A launcher exit is non-terminal while its owned process group is live."""

    def test_reconcile_waits_for_detached_child(self, registry):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os,time; pid=os.fork(); "
                "time.sleep(.8) if pid == 0 else os._exit(7)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        s = _make_session(sid="proc_orphan_test")
        s.process = proc
        s.pid = proc.pid
        s.process_group_id = proc.pid
        registry._running[s.id] = s

        assert _wait_until(lambda: proc.poll() is not None, timeout=5.0), (
            "launcher should exit before its detached child"
        )
        assert s.exit_code is None
        assert registry.poll(s.id)["status"] == "running"
        assert _wait_until(lambda: registry.poll(s.id)["status"] == "exited")
        assert registry.poll(s.id)["exit_code"] == 7

    def test_wait_blocks_until_descendant_settles(self, registry):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os,time; pid=os.fork(); "
                "time.sleep(.6) if pid == 0 else os._exit(0)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        s = _make_session(sid="proc_wait_orphan")
        s.process = proc
        s.pid = proc.pid
        s.process_group_id = proc.pid
        registry._running[s.id] = s

        assert _wait_until(lambda: proc.poll() is not None, timeout=5.0)

        start = time.monotonic()
        result = registry.wait(s.id, timeout=5)
        elapsed = time.monotonic() - start

        assert result["status"] == "exited", result
        assert result["exit_code"] == 0
        assert elapsed >= 0.2

    def test_wait_wakes_when_session_moves_to_finished(self, registry):
        """wait() should not sleep for the old 1s polling tick after exit."""
        s = _make_session(sid="proc_wait_event", output="done")
        registry._running[s.id] = s

        def finish_later():
            time.sleep(0.05)
            s.exited = True
            s.exit_code = 0
            with patch.object(registry, "_write_checkpoint"):
                registry._move_to_finished(s)

        t = threading.Thread(target=finish_later)
        t.start()
        start = time.monotonic()
        try:
            result = registry.wait(s.id, timeout=5)
        finally:
            t.join(timeout=1)
        elapsed = time.monotonic() - start

        assert result["status"] == "exited", result
        assert result["exit_code"] == 0
        assert elapsed < 0.9  # must stay under the old 1s poll tick being regression-tested, f"wait() should wake on completion; took {elapsed:.3f}s"


# =========================================================================
# Read log
# =========================================================================

class TestReadLog:
    def test_read_full_log(self, registry):
        lines = "\n".join([f"line {i}" for i in range(50)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id)
        assert result["total_lines"] == 50

    def test_read_with_offset(self, registry):
        lines = "\n".join([f"line {i}" for i in range(100)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id, offset=10, limit=5)
        assert "5 lines" in result["showing"]


# =========================================================================
# Stdin helpers
# =========================================================================

class TestStdinHelpers:
    def test_close_stdin_pipe_mode(self, registry):
        proc = MagicMock()
        proc.stdin = MagicMock()
        s = _make_session()
        s.process = proc
        registry._running[s.id] = s

        result = registry.close_stdin(s.id)

        proc.stdin.close.assert_called_once()
        assert result["status"] == "ok"

    def test_close_stdin_allows_eof_driven_process_to_finish(self, registry, tmp_path):
        """PTY mode: writing data + sending EOF lets an EOF-driven child finish.

        Background non-PTY mode used to expose subprocess stdin via a pipe,
        but PR #214b95392 detached non-PTY stdin to DEVNULL to fix keyboard
        lockout (#17959). For interactive stdin → PTY mode is now the only
        supported path.
        """
        session = registry.spawn_local(
            'python3 -c "import sys; print(sys.stdin.read().strip())"',
            cwd=str(tmp_path),
            use_pty=True,
        )

        try:
            # Wait for the PTY child to be up rather than sleeping blindly.
            assert _wait_until(
                lambda: registry.poll(session.id)["status"] == "running",
                timeout=5.0,
                interval=0.02,
            ), "PTY session never reached running"
            assert registry.submit_stdin(session.id, "hello")["status"] == "ok"
            assert registry.close_stdin(session.id)["status"] == "ok"

            deadline = time.time() + 5
            while time.time() < deadline:
                poll = registry.poll(session.id)
                if poll["status"] == "exited":
                    assert poll["exit_code"] == 0
                    assert "hello" in poll["output_preview"]
                    return
                time.sleep(0.02)

            pytest.fail("process did not exit after stdin was closed")
        finally:
            registry.kill_process(session.id)


# =========================================================================
# List sessions
# =========================================================================

class TestListSessions:
    def test_filter_by_task_id(self, registry):
        s1 = _make_session(sid="proc_1", task_id="t1")
        s2 = _make_session(sid="proc_2", task_id="t2")
        registry._running[s1.id] = s1
        registry._running[s2.id] = s2
        result = registry.list_sessions(task_id="t1")
        assert len(result) == 1
        assert result[0]["session_id"] == "proc_1"

    def test_session_key_surfaces_cross_task_processes(self, registry):
        """A bg process under the same gateway session but a DIFFERENT task is
        surfaced when session_key is passed, and flagged session_scoped (#29177).
        """
        # Current turn's task = "t_now"; forgotten preview server = "t_old"
        # but both share gateway session_key "gw1".
        own = _make_session(sid="proc_own", task_id="t_now")
        own.session_key = "gw1"
        forgotten = _make_session(sid="proc_forgotten", task_id="t_old")
        forgotten.session_key = "gw1"
        other = _make_session(sid="proc_other", task_id="t_x")
        other.session_key = "gw_other"
        registry._running[own.id] = own
        registry._running[forgotten.id] = forgotten
        registry._running[other.id] = other

        # Task-only (legacy) view sees just the current task's process.
        legacy = registry.list_sessions(task_id="t_now")
        assert {r["session_id"] for r in legacy} == {"proc_own"}

        # With session_key, the forgotten process under the same gateway
        # session is surfaced and flagged; the unrelated session is not.
        result = registry.list_sessions(task_id="t_now", session_key="gw1")
        by_id = {r["session_id"]: r for r in result}
        assert set(by_id) == {"proc_own", "proc_forgotten"}
        assert by_id["proc_forgotten"].get("session_scoped") is True
        assert "session_scoped" not in by_id["proc_own"]

# =========================================================================
# Active process queries
# =========================================================================

class TestActiveQueries:
    def test_has_active_processes(self, registry):
        s = _make_session(task_id="t1")
        registry._running[s.id] = s
        assert registry.has_active_processes("t1") is True
        assert registry.has_active_processes("t2") is False

    def test_has_active_for_session_with_max_age_stale(self, registry):
        """Stale process (older than max_active_age) is ignored."""
        s = _make_session(started_at=time.time() - 90000)  # 25 hours ago
        s.session_key = "gw_session_1"
        registry._running[s.id] = s
        assert registry.has_active_for_session("gw_session_1", max_active_age=86400) is False

# =========================================================================
# Pruning
# =========================================================================

class TestPruning:
    def test_prune_expired_finished(self, registry):
        old_session = _make_session(
            sid="proc_old",
            exited=True,
            started_at=time.time() - FINISHED_TTL_SECONDS - 100,
        )
        registry._finished[old_session.id] = old_session
        registry._prune_if_needed()
        assert "proc_old" not in registry._finished

    def test_prune_over_max_removes_oldest(self, registry):
        # Fill up to MAX_PROCESSES
        for i in range(MAX_PROCESSES):
            s = _make_session(
                sid=f"proc_{i}",
                exited=True,
                started_at=time.time() - i,  # older as i increases
            )
            registry._finished[s.id] = s

        # Add one more running to trigger prune
        s = _make_session(sid="proc_new")
        registry._running[s.id] = s
        registry._prune_if_needed()

        total = len(registry._running) + len(registry._finished)
        assert total <= MAX_PROCESSES


# =========================================================================
# Spawn env sanitization
# =========================================================================

class TestSpawnEnvSanitization:
    def test_spawn_local_strips_blocked_vars_from_background_env(self, registry):
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["env"] = kwargs["env"]
            proc = MagicMock()
            proc.pid = 4321
            proc.stdout = iter([])
            proc.stdin = MagicMock()
            proc.poll.return_value = None
            return proc

        fake_thread = MagicMock()

        with patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "USER": "tester",
            "TELEGRAM_BOT_TOKEN": "bot-secret",
            "FIRECRAWL_API_KEY": "fc-secret",
        }, clear=True), \
            patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
            patch("subprocess.Popen", side_effect=fake_popen), \
            patch("threading.Thread", return_value=fake_thread), \
            patch.object(registry, "_write_checkpoint"):
            registry.spawn_local(
                "echo hello",
                cwd="/tmp",
                env_vars={
                    "MY_CUSTOM_VAR": "keep-me",
                    "TELEGRAM_BOT_TOKEN": "drop-me",
                    f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}TELEGRAM_BOT_TOKEN": "forced-bot-token",
                },
            )

        env = captured["env"]
        assert env["MY_CUSTOM_VAR"] == "keep-me"
        assert env["TELEGRAM_BOT_TOKEN"] == "forced-bot-token"
        assert "FIRECRAWL_API_KEY" not in env
        assert f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}TELEGRAM_BOT_TOKEN" not in env
        assert env["PYTHONUNBUFFERED"] == "1"

    def test_spawn_via_env_checks_returncode_when_wrapper_fails(self, registry):
        class FakeEnv:
            def __init__(self):
                self.commands = []

            def execute(self, command, **kwargs):
                self.commands.append((command, kwargs))
                return {"output": "syntax error", "returncode": 2}

        env = FakeEnv()
        fake_thread = MagicMock()

        with patch("tools.process_registry.threading.Thread", return_value=fake_thread), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_via_env(env, "echo hello")

        assert session.exited is True
        assert session.exit_code == 2
        assert session.pid is None
        assert session.output_buffer == "syntax error"
        fake_thread.start.assert_not_called()
        # A failed launch must not be exposed as a running/tracked session.
        assert session.id not in registry._running

    def test_spawn_via_env_carries_deadline_and_notification_metadata(self, registry):
        commands = []

        class FakeEnv:
            def execute(self, command, **kwargs):
                commands.append(command)
                return {"output": "4321", "returncode": 0}
        fake_thread = MagicMock()
        fake_timer = MagicMock()

        def assert_registered():
            assert len(registry._running) == 1

        fake_thread.start.side_effect = assert_registered
        fake_timer.start.side_effect = assert_registered
        with patch(
            "tools.process_registry.threading.Thread", return_value=fake_thread
        ), patch(
            "tools.process_registry.threading.Timer", return_value=fake_timer
        ), patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_via_env(
                FakeEnv(),
                "echo hello",
                execution_timeout=12,
                notification_metadata={"notify_on_complete": True},
            )

        assert session.execution_deadline == pytest.approx(session.started_at + 12)
        assert session.notify_on_complete is True
        assert session.id in registry._running
        assert "nohup setsid bash -lc" in commands[0]
        fake_timer.start.assert_called_once()
        fake_thread.start.assert_called_once()

    def test_env_poller_quotes_temp_paths_with_spaces(self, registry):
        session = _make_session(sid="proc_space")
        session.exited = False

        class FakeEnv:
            def __init__(self):
                self.commands = []
                self._responses = iter([
                    {"output": "1\n", "returncode": 0},
                    {"output": "hello\n", "returncode": 0},
                    {"output": "0\n", "returncode": 0},
                ])

            def execute(self, command, **kwargs):
                self.commands.append((command, kwargs))
                return next(self._responses)

        env = FakeEnv()

        registry._env_poller_loop(
            session,
            env,
            "/path with spaces/hermes_bg.log",
            "/path with spaces/hermes_bg.pid",
            "/path with spaces/hermes_bg.exit",
        )

        assert env.commands[0][0] == (
            "pid=$(head -n 1 '/path with spaces/hermes_bg.pid' 2>/dev/null); "
            'kill -0 "$pid" 2>/dev/null; echo $?'
        )
        assert env.commands[1][0] == (
            "if [ -f '/path with spaces/hermes_bg.log' ]; then "
            "bytes=$(wc -c < '/path with spaces/hermes_bg.log'); "
            "printf '__HERMES_LOG_BYTES__:%s\\n' \"$bytes\"; "
            "tail -c 200000 -- '/path with spaces/hermes_bg.log'; fi"
        )
        assert env.commands[2][0] == (
            "head -c 5 -- '/path with spaces/hermes_bg.exit' 2>/dev/null"
        )


# =========================================================================
# Popen leak prevention
# =========================================================================

class TestPopenLeakOnSetupFailure:
    """Regression for issue #2749: subprocess orphaned when post-Popen setup raises."""

    def test_popen_killed_when_thread_creation_fails(self, registry):
        """If Thread() raises after Popen, proc must be killed — not orphaned."""
        proc = MagicMock()
        proc.pid = 9999
        proc.stdout = iter([])
        proc.stdin = MagicMock()
        proc.returncode = None
        proc.poll.side_effect = lambda: proc.returncode

        def reap(**_kwargs):
            proc.returncode = -15
            return -15

        proc.wait.side_effect = reap
        terminate = MagicMock(return_value=True)

        def boom(*args, **kwargs):
            raise RuntimeError("Thread creation failed")

        with (
            patch("tools.process_registry._find_shell", return_value="/bin/bash"),
            patch("subprocess.Popen", return_value=proc),
            patch("threading.Thread", side_effect=boom),
            patch.object(registry, "_terminate_host_pid", terminate),
            patch.object(registry, "_write_checkpoint"),
        ):
            with pytest.raises(RuntimeError, match="Thread creation failed"):
                registry.spawn_local("echo hello", cwd="/tmp")

        terminate.assert_called_once_with(9999, None)
        assert not registry._running
        assert not registry._finished


# =========================================================================
# Spawn rewrite regression (issue #68915)
# =========================================================================


class TestSpawnRewriteCompoundBackground:
    """Verify that spawn_local rewrites `A && B &` patterns to avoid subshell deadlocks.

    Issue #68915: when bash parses ``A && B &`` it forks a subshell ``(A && B) &``.
    If B is a long-running server, the subshell never exits and holds the stdout
    pipe open, causing a permanent deadlock. The rewriter wraps the tail to
    ``A && { B & }`` so no subshell fork occurs.
    """

    def test_compound_and_background_gets_rewritten(self, registry):
        """A && B & must be rewritten to A && { B & } before Popen."""
        captured_cmd = []

        def fake_popen(args, **kwargs):
            captured_cmd.append(args)
            proc = MagicMock()
            proc.pid = 1111
            proc.stdout = MagicMock()
            return proc

        fake_thread = MagicMock()
        fake_thread.daemon = False

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            registry.spawn_local("cd /app && node server.js &>/tmp/srv.log &", cwd="/tmp")

        assert len(captured_cmd) == 1
        shell_cmd = captured_cmd[0]
        # The command passed to Popen should be the REWRITTEN version
        assert "&& { node server.js &>/tmp/srv.log & }" in shell_cmd[-1]

    def test_simple_background_preserved(self, registry):
        """Simple cmd & (no &&) must NOT be rewritten — no subshell bug."""
        captured_cmd = []

        def fake_popen(args, **kwargs):
            captured_cmd.append(args)
            proc = MagicMock()
            proc.pid = 2222
            proc.stdout = MagicMock()
            return proc

        fake_thread = MagicMock()
        fake_thread.daemon = False

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            registry.spawn_local("sleep 5 &", cwd="/tmp")

        assert len(captured_cmd) == 1
        shell_cmd = captured_cmd[0][-1]
        # Simple background must remain as-is
        assert "sleep 5 &" in shell_cmd

    def test_pty_path_uses_rewritten_command(self, registry):
        """PTY spawn path must also use the rewritten command (issue #68915)."""
        mock_pty_proc = MagicMock()
        mock_pty_proc.pid = 5555

        mock_pty_module = MagicMock()
        mock_pty_module.PtyProcess.spawn = MagicMock(return_value=mock_pty_proc)

        fake_thread = MagicMock()
        fake_thread.daemon = False

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("tools.process_registry._IS_WINDOWS", False), \
             patch.dict("sys.modules", {"ptyprocess": mock_pty_module}), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_local(
                "cd /app && node server.js &",
                cwd="/tmp",
                use_pty=True,
            )

        assert mock_pty_module.PtyProcess.spawn.called, \
            "PTY spawn should have been attempted"
        pty_args = mock_pty_module.PtyProcess.spawn.call_args[0][0]
        assert "&& { node server.js & }" in pty_args[2], \
            f"PTY path should use rewritten command, got: {pty_args[2]}"
        assert session.command == "cd /app && node server.js &"


# =========================================================================
# Checkpoint
# =========================================================================

class TestCheckpoint:
    def test_recover_dead_pid(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_dead",
            "command": "sleep 999",
            "pid": 999999999,  # almost certainly not running
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0


    def test_recovery_skips_explicit_sandbox_backed_entries(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        original = [{
            "session_id": "proc_remote",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "pid_scope": "sandbox",
        }]
        checkpoint.write_text(json.dumps(original))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0
            assert registry.get("proc_remote") is None

            data = json.loads(checkpoint.read_text())
            assert data == []

# =========================================================================
# Kill process
# =========================================================================

class TestKillProcess:
    def test_kill_already_exited(self, registry):
        s = _make_session(exited=True, exit_code=0)
        registry._finished[s.id] = s
        result = registry.kill_process(s.id)
        assert result["status"] == "already_exited"

    def test_kill_pty_uses_process_tree_termination(self, registry, monkeypatch):
        s = _make_session(sid="proc_pty_tree")
        s.pid = 424242
        s.host_start_time = 99
        s._pty = MagicMock()
        registry._running[s.id] = s
        terminate = MagicMock()
        monkeypatch.setattr(registry, "_terminate_host_pid", terminate)

        result = registry.kill_process(s.id)

        assert result["status"] == "killed"
        terminate.assert_called_once_with(424242, 99)
        s._pty.terminate.assert_not_called()

    def test_kill_remote_uses_process_group(self, registry, monkeypatch):
        env = MagicMock()
        env.execute.return_value = {
            "output": "__HERMES_TERMINATED__\n",
            "returncode": 0,
        }
        s = _make_session(sid="proc_remote_tree")
        s.pid = 4321
        s.host_start_time = 99
        s.pid_scope = "sandbox"
        s.env_ref = env
        registry._running[s.id] = s
        monkeypatch.setattr(registry, "_daemon_term_grace_seconds", lambda: 0.0)

        result = registry.kill_process(s.id)

        assert result["status"] == "killed"
        command = next(
            call.args[0]
            for call in env.execute.call_args_list
            if "kill -TERM" in call.args[0]
        )
        assert "kill -TERM -- -4321" in command
        assert "kill -KILL -- -4321" in command

    def test_kill_detached_session_uses_host_pid(self, registry):
        s = _make_session(sid="proc_detached", command="sleep 999")
        s.pid = 424242
        s.detached = True
        registry._running[s.id] = s

        terminate_calls = []

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
            def children(self, recursive=False):
                return []
            def terminate(self):
                terminate_calls.append(("terminate", self.pid))

            def is_running(self):
                return False

            def status(self):
                return _psutil.STATUS_ZOMBIE

        import psutil as _psutil

        try:
            # Post-#21561: liveness probe routes through
            # ``ProcessRegistry._is_host_pid_alive`` (→
            # ``gateway.status._pid_exists``), and the actual kill on POSIX
            # routes through ``psutil.Process(pid).terminate()``. Neither
            # touches ``os.kill`` directly. Mock both seams.  Disable the
            # SIGKILL-escalation step (grace=0) so it doesn't call
            # ``psutil.wait_procs`` on the FakeProcess.
            with patch("gateway.status._pid_exists", return_value=True), \
                 patch.object(ProcessRegistry, "_daemon_term_grace_seconds",
                              staticmethod(lambda: 0.0)), \
                 patch.object(_psutil, "Process", side_effect=lambda pid: FakeProcess(pid)):
                result = registry.kill_process(s.id)

            assert result["status"] == "killed"
            assert ("terminate", 424242) in terminate_calls
        finally:
            registry._running.pop(s.id, None)

    def test_kill_intent_wins_completion_race(self, registry, monkeypatch):
        class FakeProcess:
            pid = 424242
            returncode = None
            stdout = None
            stdin = None

            def poll(self):
                return self.returncode

        proc = FakeProcess()
        s = _make_session(sid="proc_kill_race")
        s.process = proc
        s.pid = proc.pid
        s.notify_on_complete = True
        s._subreaper_managed = True
        registry._running[s.id] = s
        registry._move_to_finished(s)

        terminate_entered = threading.Event()
        release_terminate = threading.Event()
        result = {}

        def terminate(*_args):
            terminate_entered.set()
            assert release_terminate.wait(5)

        monkeypatch.setattr(registry, "_terminate_host_pid", terminate)
        thread = threading.Thread(
            target=lambda: result.update(registry.kill_process(s.id))
        )
        thread.start()
        try:
            assert terminate_entered.wait(2)
            proc.returncode = -15
            # Completion publication is delayed until the signal outcome is
            # known; terminal reason/source are not pre-written.
            time.sleep(0.1)
            assert registry.completion_queue.empty()
            assert s.completion_reason == "exited"
            assert s.termination_source == ""
            release_terminate.set()
            thread.join(5)
        finally:
            release_terminate.set()
            thread.join(5)

        assert not thread.is_alive()
        assert result["status"] == "killed"
        event = registry.completion_queue.get_nowait()
        assert event["completion_reason"] == "killed"
        assert event["exit_code"] == -15
        assert registry.completion_queue.empty()
        assert s.completion_reason == "killed"
        assert s.exit_code == -15
        assert registry.is_completion_consumed(s.id)
        registry._move_to_finished(s)
        assert registry.completion_queue.empty()


# =========================================================================
# Tool handler
# =========================================================================

class TestProcessToolHandler:
    def test_unknown_action(self):
        from tools.process_registry import _handle_process
        result = json.loads(_handle_process({"action": "unknown_action"}))
        assert "error" in result


# =========================================================================
# format_process_notification + drain_notifications (shared helpers)
# =========================================================================

from tools.process_registry import format_process_notification


def test_drain_notifications_completion_callback_exception_fails_closed(registry):
    event = {
        "type": "completion",
        "session_id": "proc_callback_error",
        "session_key": "session-a",
        "command": "safe-test-command",
        "exit_code": 0,
        "output": "done",
    }
    registry.completion_queue.put(event)

    def broken(_event):
        raise RuntimeError("ownership check exploded")

    results = registry.drain_notifications(
        session_key="session-a",
        owns_event=broken,
    )

    assert results == []
    assert registry.completion_queue.get_nowait() == event
    assert registry.completion_queue.empty()


def test_drain_notifications_filters_async_delegation_by_session_key():
    """Async-delegation events should only be consumed by the matching session's drain.

    Regression test for issue #58684: background delegation results delivered
    to the wrong session when the user switches sessions while a subagent runs.
    """
    from tools.process_registry import process_registry

    # Clear the queue first
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    try:
        # Put events for different sessions
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_session_a",
            "session_key": "telegram:dm:111:user_a",
            "goal": "task A",
            "status": "completed",
            "summary": "done A",
            "api_calls": 1,
            "duration_seconds": 0.5,
        })
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_session_b",
            "session_key": "telegram:dm:222:user_b",
            "goal": "task B",
            "status": "completed",
            "summary": "done B",
            "api_calls": 1,
            "duration_seconds": 0.3,
        })

        # Drain for session A — should only get deleg_session_a
        results_a = process_registry.drain_notifications(session_key="telegram:dm:111:user_a")
        assert len(results_a) == 1, (
            f"Expected 1 event for session A, got {len(results_a)}"
        )
        assert results_a[0][0]["delegation_id"] == "deleg_session_a"
        assert "done A" in results_a[0][1]

        # Session B's event should have been re-queued — drain for session B
        results_b = process_registry.drain_notifications(session_key="telegram:dm:222:user_b")
        assert len(results_b) == 1, (
            f"Expected 1 event for session B, got {len(results_b)}"
        )
        assert results_b[0][0]["delegation_id"] == "deleg_session_b"
        assert "done B" in results_b[0][1]

        # No more events should remain
        assert process_registry.completion_queue.empty()
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_drain_notifications_owns_event_callback_beats_key_equality():
    """The positive-proof ownership callback consumes ONLY approved events —
    including across a compression rotation where bare key equality would
    wrongly re-queue the session's own pre-compression dispatch (#55578)."""
    from tools.process_registry import process_registry

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    try:
        # Pre-compression dispatch: event carries the OLD key.
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_precompress",
            "session_key": "old_parent_key",
            "goal": "task", "status": "completed", "summary": "mine",
            "api_calls": 1, "duration_seconds": 0.1,
        })
        # Foreign event that plain key equality would also reject.
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_foreign",
            "session_key": "someone_else",
            "goal": "task", "status": "completed", "summary": "not mine",
            "api_calls": 1, "duration_seconds": 0.1,
        })

        # Chain-aware ownership: this session's lineage includes old_parent_key.
        lineage = {"old_parent_key", "new_child_key"}
        results = process_registry.drain_notifications(
            session_key="new_child_key",
            owns_event=lambda e: e.get("session_key") in lineage,
        )
        assert [r[0]["delegation_id"] for r in results] == ["deleg_precompress"]

        # The foreign event was re-queued, not consumed.
        leftover = process_registry.completion_queue.get_nowait()
        assert leftover["delegation_id"] == "deleg_foreign"
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


# ---------------------------------------------------------------------------
# _terminate_host_pid — cross-platform process-tree termination
# ---------------------------------------------------------------------------


class TestTerminateHostPidWindows:
    """Windows branch uses ``taskkill /T /F`` — the documented MS tree-kill
    primitive. We can't use psutil's ``children(recursive=True)`` /
    ``.terminate()`` path on Windows because (1) Windows doesn't maintain
    a Unix-style process tree so the walk is unreliable, and (2)
    ``Process.terminate()`` on Windows is ``TerminateProcess()`` for the
    target handle only, not the tree.
    """

    def test_windows_invokes_taskkill_with_tree_and_force_flags(self, monkeypatch):
        """The Windows branch must shell out to ``taskkill /PID N /T /F``."""
        from tools import process_registry as pr

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(pr, "_IS_WINDOWS", True)
        monkeypatch.setattr(pr.subprocess, "run", fake_run)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert captured["args"][0] == "taskkill"
        assert "/PID" in captured["args"]
        assert "12345" in captured["args"]
        assert "/T" in captured["args"], "Tree flag required to reach descendants"
        assert "/F" in captured["args"], "Force flag required for headless Chromium"

class TestTerminateHostPidPosix:
    """POSIX branch walks the tree via psutil and SIGTERMs children first."""

    def test_posix_walks_tree_and_terminates_children_then_parent(self, monkeypatch):
        from tools import process_registry as pr
        import psutil

        terminate_order = []

        class _FakeChild:
            def __init__(self, pid):
                self.pid = pid

            def terminate(self):
                terminate_order.append(self.pid)

        class _FakeParent:
            def __init__(self, pid):
                self.pid = pid

            def children(self, recursive=False):
                assert recursive is True
                return [_FakeChild(101), _FakeChild(102), _FakeChild(103)]

            def terminate(self):
                terminate_order.append(self.pid)

        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setattr(psutil, "Process", _FakeParent)
        # This test covers only the SIGTERM tree-walk ordering; disable the
        # SIGKILL-escalation step (which would call psutil.wait_procs on the
        # fakes) by setting the grace to 0.
        monkeypatch.setattr(pr.ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.0))
        monkeypatch.setattr(
            pr.ProcessRegistry,
            "_proc_confirmed_gone",
            staticmethod(lambda _proc: True),
        )

        assert pr.ProcessRegistry._terminate_host_pid(12345) is True

        assert terminate_order == [101, 102, 103, 12345], (
            "Children must be terminated before the parent"
        )

    def test_posix_oserror_falls_back_to_os_kill(self, monkeypatch):
        from tools import process_registry as pr
        import psutil

        def boom(pid):
            raise PermissionError("can't read /proc")

        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))

        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setattr(psutil, "Process", boom)
        monkeypatch.setattr(pr.os, "kill", fake_kill)
        monkeypatch.setattr(
            pr.ProcessRegistry,
            "_is_host_pid_alive",
            staticmethod(lambda _pid: True),
        )

        confirmed = pr.ProcessRegistry._terminate_host_pid(12345)

        assert kill_calls == [(12345, signal.SIGTERM)]
        assert confirmed is False


# =========================================================================
# PID-reuse guard — a recycled PID/PGID must never be signalled.
#
# Regression: once a background-session process exits and is reaped, the kernel
# can recycle its PID onto an unrelated process (observed in the wild landing on
# a desktop browser's session leader, whose whole tree we then SIGTERMed —
# Firefox dying at irregular intervals).  Identity is re-validated via the
# kernel start time captured at spawn before any signal is sent.
# =========================================================================

class TestPidReuseGuard:
    def test_terminate_refuses_when_start_time_mismatches(self, registry):
        """A live PID whose start time changed (recycled) is NOT killed."""
        proc = _spawn_python_sleep(30)
        try:
            real_start = ProcessRegistry._safe_host_start_time(proc.pid)
            assert real_start is not None, "no /proc start time on this platform?"
            # Simulate recycling: the recorded baseline no longer matches.
            registry._terminate_host_pid(proc.pid, expected_start=real_start + 1)
            # The process must still be alive — the guard refused to signal it.
            assert not _wait_until(lambda: proc.poll() is not None, timeout=0.3)
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()


    def test_refresh_detached_marks_recycled_pid_exited(self, registry):
        """A detached session whose PID got recycled is moved to finished."""
        wrong_start = (ProcessRegistry._safe_host_start_time(os.getpid()) or 0) + 999
        s = _make_session(sid="proc_detached")
        s.pid = os.getpid()          # alive, but...
        s.pid_scope = "host"
        s.detached = True
        s.host_start_time = wrong_start  # ...identity no longer matches
        registry._running[s.id] = s
        refreshed = registry._refresh_detached_session(s)
        assert refreshed.exited is True
        assert s.id in registry._finished


@pytest.mark.skipif(sys.platform == "win32",
                    reason="POSIX SIGTERM→SIGKILL escalation; Windows uses taskkill /F")
class TestSigkillEscalation:
    """Bounded SIGTERM→SIGKILL escalation in _terminate_host_pid.

    A daemon that ignores/stalls on SIGTERM must be force-killed after the
    configured grace window so it can't leak indefinitely — while well-behaved
    processes still exit cleanly on SIGTERM and the recycled-PID guard is never
    bypassed.
    """

    # A process that traps SIGTERM (ignores it): only SIGKILL stops it.
    # It prints "ready" AFTER installing the handler so the parent never
    # signals it during the startup window (before SIG_IGN is in place).
    _TRAP = (
        "import signal, sys, time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "sys.stdout.write('ready\\n'); sys.stdout.flush();"
        "[time.sleep(0.2) for _ in iter(int, 1)]"
    )

    def _spawn_trap(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", self._TRAP],
            stdout=subprocess.PIPE, text=True,
        )
        # Wait until the handler is installed before returning.
        line = proc.stdout.readline()
        assert line.strip() == "ready", "trap process failed to start"
        return proc

    def test_sigterm_ignoring_daemon_is_sigkilled(self, monkeypatch):
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.3))
        proc = self._spawn_trap()
        try:
            ProcessRegistry._terminate_host_pid(proc.pid)
            assert _wait_until(lambda: proc.poll() is not None, timeout=4.0), \
                "SIGTERM-ignoring daemon should be SIGKILLed after grace"
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_escalation_does_not_bypass_recycled_pid_guard(self, monkeypatch):
        """A start-time mismatch must still spare the PID — no SIGTERM, no SIGKILL."""
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.3))
        proc = self._spawn_trap()
        try:
            real_start = ProcessRegistry._safe_host_start_time(proc.pid)
            ProcessRegistry._terminate_host_pid(
                proc.pid, expected_start=(real_start or 0) + 1)
            assert not _wait_until(lambda: proc.poll() is not None, timeout=0.3)
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()

    def test_grace_reader_floors_at_zero(self, monkeypatch):
        """A negative configured grace is clamped to 0 (no escalation)."""
        import hermes_cli.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "read_raw_config",
                            lambda: {"terminal": {"daemon_term_grace_seconds": -5}})
        assert ProcessRegistry._daemon_term_grace_seconds() == 0.0

    @pytest.mark.live_system_guard_bypass
    def test_entire_tree_is_sigkilled_not_just_parent(self, monkeypatch):
        """A SIGTERM-ignoring parent + children are ALL force-killed.

        Regression: an earlier implementation trusted psutil.wait_procs's
        gone/alive partition, which mis-partitioned across a parent/child tree
        and left survivors un-killed (flaky — sometimes the parent lived,
        sometimes a child). The escalation now re-probes every target directly.
        """
        import psutil
        # 2.0s grace (not 1.0): with three interpreters mid-startup on a
        # loaded runner, a 1s SIGTERM->partition window races child spawn and
        # is how a child PID escaped the live-system guard in CI.
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 2.0))
        # Parent spawns 2 children; all trap SIGTERM. Parent prints child pids
        # after the handler is installed.
        parent_src = (
            "import signal, subprocess, sys, time;"
            "child='import signal,time\\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
            "[time.sleep(0.2) for _ in iter(int,1)]';"
            "kids=[subprocess.Popen([sys.executable,'-c',child]) for _ in range(2)];"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "sys.stdout.write(' '.join(str(k.pid) for k in kids)+'\\n'); sys.stdout.flush();"
            "[time.sleep(0.2) for _ in iter(int,1)]"
        )
        parent = subprocess.Popen([sys.executable, "-c", parent_src],
                                  stdout=subprocess.PIPE, text=True)
        # Bound the readline: if the parent wedges before printing, fail THIS
        # test with a clear message instead of letting the per-file timeout
        # SIGKILL the whole pytest process (opaque rc=124 in CI).
        import select as _select
        ready, _, _ = _select.select([parent.stdout], [], [], 20.0)
        assert ready, "parent process failed to print child pids within 20s"
        child_pids = [int(x) for x in parent.stdout.readline().split()]
        all_pids = [parent.pid] + child_pids
        try:
            ProcessRegistry._terminate_host_pid(parent.pid)

            def _pid_dead(p: int) -> bool:
                # A pid is "dead" for our purposes if it no longer exists OR
                # exists only as an unreaped zombie (already terminated, just
                # not reaped by its reparented parent yet). psutil can also
                # raise mid-probe if the pid vanishes between the existence
                # check and the status read — treat any such race as dead.
                try:
                    if not psutil.pid_exists(p):
                        return True
                    return not ProcessRegistry._proc_alive(psutil.Process(p))
                except Exception:
                    return True

            def _all_dead():
                return all(_pid_dead(p) for p in all_pids)

            # _terminate_host_pid SIGKILLs synchronously before returning, so
            # the kill signals are already delivered here. The only remaining
            # wait is the kernel tearing down 3 processes and the reparented
            # children transitioning to zombie — which can lag on a loaded CI
            # runner. Give a generous budget (matches the wait() test's 10s)
            # so this asserts the escalation BEHAVIOR, not the runner's
            # scheduling latency. The assertion itself never weakens: every
            # tree member must end up dead/zombie.
            assert _wait_until(_all_dead, timeout=15.0, interval=0.02), (
                "entire SIGTERM-ignoring tree (parent + children) must be SIGKILLed"
            )
        finally:
            for p in all_pids:
                try:
                    os.kill(p, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            parent.wait()


class TestHandleProcessRedaction:
    """`_handle_process` redacts background-process output before it reaches the
    model / session.db / CLI display — issue #43025.

    Mirrors the foreground `terminal` redaction so the two surfaces can't
    diverge. Env-dump commands (`printenv`/`env`) get the ENV-assignment pass
    so opaque tokens are masked; other commands stay on the code_file path.
    """

    def _setup(self, monkeypatch, command, output):
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", True)
        from tools import process_registry as pr
        reg = ProcessRegistry()
        sess = _make_session(sid="proc_redact1", command=command)
        sess.output_buffer = output
        sess.exited = True
        sess.exit_code = 0
        reg._running.clear()
        reg._finished[sess.id] = sess
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)
        return pr, sess

    def test_log_redacts_env_dump_opaque_token(self, monkeypatch):
        pr, sess = self._setup(
            monkeypatch, "printenv",
            "MY_SERVICE_TOKEN=abc123randomopaquetokenvalue999\nHOME=/home/u",
        )
        out = json.loads(pr._handle_process({"action": "log", "session_id": sess.id}))
        assert "abc123randomopaquetokenvalue999" not in out["output"]
        assert "HOME=/home/u" in out["output"]

    def test_poll_redacts_prefix_key(self, monkeypatch):
        pr, sess = self._setup(
            monkeypatch, "python app.py",
            "leaked OPENAI_API_KEY sk-proj-abc123def456ghi789jkl012 here",
        )
        out = json.loads(pr._handle_process({"action": "poll", "session_id": sess.id}))
        assert "abc123def456" not in out["output_preview"]

    def test_disabled_passes_through(self, monkeypatch):
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", False)
        from tools import process_registry as pr
        reg = ProcessRegistry()
        sess = _make_session(sid="proc_redact2", command="printenv")
        sess.output_buffer = "CUSTOM_TOKEN=zzzopaque1234567890abcdef"
        sess.exited = True
        sess.exit_code = 0
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)
        out = json.loads(pr._handle_process({"action": "log", "session_id": sess.id}))
        assert "zzzopaque1234567890abcdef" in out["output"]


# =========================================================================
# Reader loop: orphaned grandchild holding the stdout pipe (issue #68915)
# =========================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: select() on pipes")
class TestReaderLoopOrphanedPipe:
    """The reader supervises the complete owned process group."""

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="Linux-only: requires PR_SET_CHILD_SUBREAPER",
    )
    def test_spawn_waits_for_setsid_worker_before_one_completion(
        self, registry, tmp_path
    ):
        done = tmp_path / "worker.done"
        code = (
            "import os,time; pid=os.fork(); "
            "(os.setsid(), print('worker-start', flush=True), os.close(1), "
            f"os.close(2), time.sleep(.8), open({str(done)!r}, 'w').close(), "
            "os._exit(0)) "
            "if pid == 0 else os._exit(23)"
        )
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
        session = registry.spawn_local(command, cwd=str(tmp_path))
        session.notify_on_complete = True

        try:
            assert _wait_until(lambda: "worker-start" in session.output_buffer)
            assert registry.poll(session.id)["status"] == "running"
            assert session.id in registry._running
            assert registry.completion_queue.empty()
            assert not done.exists()

            assert _wait_until(done.exists)
            assert _wait_until(
                lambda: registry.poll(session.id)["status"] == "exited"
            )
            assert _wait_until(lambda: not registry.completion_queue.empty())
            event = registry.completion_queue.get_nowait()
            assert event["session_id"] == session.id
            assert event["exit_code"] == 23
            assert event["output"] == "worker-start\n"
            registry._move_to_finished(session)
            assert registry.completion_queue.empty()
        finally:
            if not session.exited:
                registry.kill_process(session.id)

    def test_final_notify_waits_for_child_and_is_exactly_once(self, registry):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os,time; print('launcher', flush=True); pid=os.fork(); "
                "(time.sleep(.8), print('child-final', flush=True), os._exit(0)) "
                "if pid == 0 else os._exit(9)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        s = _make_session(sid="proc_orphan_notify")
        s.process = proc
        s.pid = proc.pid
        s.process_group_id = proc.pid
        s.notify_on_complete = True
        registry._running[s.id] = s

        done = threading.Event()
        t = threading.Thread(
            target=lambda: (registry._reader_loop(s), done.set()), daemon=True
        )
        t.start()
        assert _wait_until(lambda: proc.poll() is not None)
        assert not done.wait(timeout=0.2)
        assert registry.completion_queue.empty()
        assert done.wait(timeout=5)

        item = registry.completion_queue.get_nowait()
        assert item["type"] == "completion"
        assert item["session_id"] == s.id
        assert item["exit_code"] == 9
        assert item["output"] == "launcher\nchild-final\n"
        registry._move_to_finished(s)
        assert registry.completion_queue.empty()

    def test_final_buffered_tail_is_complete_and_notified_once(self, registry):
        payload = "".join(f"tail-{i:05d}:abcdefghij\n" for i in range(1200))
        payload += "TAIL-SUFFIX-IS-PRESENT\n"
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1])", payload],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        assert proc.stdout.buffer.peek(1), "expected a real BufferedReader tail"
        s = _make_session(sid="proc_buffered_tail")
        s.process = proc
        s.pid = proc.pid
        s.process_group_id = proc.pid
        s.notify_on_complete = True
        s.max_output_chars = len(payload) + 1
        registry._running[s.id] = s

        registry._reader_loop(s)

        assert s.output_buffer == payload
        event = registry.completion_queue.get_nowait()
        assert event["exit_code"] == 0
        assert event["output"].endswith("TAIL-SUFFIX-IS-PRESENT\n")
        registry._move_to_finished(s)
        assert registry.completion_queue.empty()

    def test_buffered_and_delayed_tail_finish_with_external_writer_open(self, registry):
        read_fd, write_fd = os.pipe()
        proc = subprocess.Popen(
            [sys.executable, "-c", "print('prefetched-tail', flush=True)"],
            stdout=write_fd,
            stderr=write_fd,
        )
        stdout = os.fdopen(read_fd, "r", encoding="utf-8")
        proc.stdout = stdout
        proc.wait()
        assert stdout.buffer.peek(1), "expected a real BufferedReader tail"
        s = _make_session(sid="proc_external_writer")
        s.process = proc
        s.pid = proc.pid
        s.notify_on_complete = True
        registry._running[s.id] = s

        delayed = threading.Thread(
            target=lambda: (time.sleep(0.3), os.write(write_fd, b"delayed-tail\n")),
            daemon=True,
        )
        done = threading.Event()
        reader = threading.Thread(
            target=lambda: (registry._reader_loop(s), done.set()), daemon=True
        )
        delayed.start()
        reader.start()
        try:
            assert done.wait(timeout=3)
            assert s.output_buffer == "prefetched-tail\ndelayed-tail\n"
            event = registry.completion_queue.get_nowait()
            assert event["exit_code"] == 0
            registry._move_to_finished(s)
            assert registry.completion_queue.empty()
        finally:
            delayed.join(timeout=1)
            os.close(write_fd)
            reader.join(timeout=1)
            stdout.close()

    def test_known_launcher_exit_is_kept_until_descendant_settles(self, registry):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os,time; pid=os.fork(); "
                "time.sleep(.7) if pid == 0 else os._exit(4)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        s = _make_session(sid="proc_zombie_notify", exited=True)
        s.process = proc
        s.pid = proc.pid
        s.process_group_id = proc.pid
        s.notify_on_complete = True
        registry._running[s.id] = s

        import psutil

        assert _wait_until(lambda: psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE)
        assert s.exit_code is None
        registry._move_to_finished(s)
        assert s.exited is False
        assert s.exit_code == 4
        assert s.id in registry._running
        assert s.id not in registry._finished
        assert registry.poll(s.id)["status"] == "running"
        assert s.exit_code == 4
        assert registry.completion_queue.empty()
        assert _wait_until(lambda: not registry.completion_queue.empty(), timeout=5)
        assert s.exited is True
        assert s.id not in registry._running
        assert s.id in registry._finished
        item = registry.completion_queue.get_nowait()
        assert item["exit_code"] == 4
        assert registry.completion_queue.empty()


# =========================================================================
# Tier-4 stale lifecycle ownership / termination regressions
# =========================================================================


def _detached_host_session(registry, proc, sid, *, deadline=None):
    session = _make_session(sid=sid)
    session.pid = proc.pid
    session.pid_scope = "host"
    session.host_start_time = registry._safe_host_start_time(proc.pid)
    session.detached = True
    session.execution_deadline = deadline
    registry._running[session.id] = session
    return session


def test_local_kill_noop_retains_owner_deadline_and_nonterminal_metadata(
    registry, monkeypatch
):
    proc = _spawn_python_sleep(30)
    original_terminate = registry._terminate_host_pid
    deadline = time.time() + 30
    session = _detached_host_session(
        registry, proc, "proc_local_noop", deadline=deadline
    )
    timer = MagicMock()
    session._deadline_timer = timer
    reason_before = session.completion_reason
    source_before = session.termination_source
    monkeypatch.setattr(registry, "_terminate_host_pid", lambda *_args: False)

    try:
        result = registry.kill_process(session.id, source="manual_noop")

        assert result["status"] == "error"
        assert registry._running[session.id] is session
        assert session.exited is False
        assert session.exit_code is None
        assert session.completion_reason == reason_before
        assert session.termination_source == source_before
        assert session.execution_deadline == deadline
        assert session._deadline_timer is timer
        timer.cancel.assert_not_called()
        assert proc.poll() is None
    finally:
        original_terminate(proc.pid, session.host_start_time)
        proc.wait(timeout=5)


def test_local_kill_confirms_physical_exit_before_terminal_commit(registry):
    proc = _spawn_python_sleep(30)
    session = _detached_host_session(registry, proc, "proc_local_confirmed")

    try:
        result = registry.kill_process(session.id, source="manual_confirmed")

        assert result["status"] == "killed"
        assert _wait_until(lambda: proc.poll() is not None)
        assert session.id not in registry._running
        assert registry._finished[session.id] is session
        assert session.exited is True
        assert session.completion_reason == "killed"
        assert session.termination_source == "manual_confirmed"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


class _RemoteKillEnv:
    def __init__(self, output="", returncode=0, on_kill=None):
        self.output = output
        self.returncode = returncode
        self.on_kill = on_kill
        self.commands = []

    def execute(self, command, timeout=None):
        self.commands.append((command, timeout))
        if "kill -TERM" in command:
            if self.on_kill is not None:
                self.on_kill()
            return {"output": self.output, "returncode": self.returncode}
        if "kill -0" in command:
            return {"output": "1\n", "returncode": 0}
        return {"output": "", "returncode": 0}


def _remote_session(registry, env, sid="proc_remote_confirm"):
    session = _make_session(sid=sid)
    session.env_ref = env
    session.pid = 4321
    session.host_start_time = 99
    session.pid_scope = "sandbox"
    registry._running[session.id] = session
    return session


def test_remote_kill_transport_rc_zero_without_marker_retains_owner(registry):
    env = _RemoteKillEnv(output="", returncode=0)
    session = _remote_session(registry, env, "proc_remote_false_success")
    reason_before = session.completion_reason
    source_before = session.termination_source

    result = registry.kill_process(session.id, source="remote_false_success")

    assert result["status"] == "error"
    assert registry._running[session.id] is session
    assert session.exited is False
    assert session.completion_reason == reason_before
    assert session.termination_source == source_before


def test_remote_kill_transport_124_with_marker_retains_owner(registry):
    env = _RemoteKillEnv(output="__HERMES_TERMINATED__\n", returncode=124)
    session = _remote_session(registry, env, "proc_remote_transport_timeout")

    result = registry.kill_process(session.id, source="remote_transport_timeout")

    assert result["status"] == "error"
    assert registry._running[session.id] is session
    assert session.exited is False
    assert session.termination_source == ""


def test_remote_kill_requires_gone_marker_and_avoids_shell_false_success(registry):
    env = _RemoteKillEnv(output="__HERMES_TERMINATED__\n", returncode=0)
    session = _remote_session(registry, env, "proc_remote_gone")

    result = registry.kill_process(session.id, source="remote_confirmed")

    command = env.commands[0][0]
    assert result["status"] == "killed"
    assert "|| true" not in command
    assert "__HERMES_TERMINATED__" in command
    assert "/proc/4321/stat" in command
    assert '"$current" != "99"' in command
    assert session.id not in registry._running
    assert registry._finished[session.id] is session


def test_remote_kill_without_spawn_identity_retains_owner(registry):
    env = _RemoteKillEnv(output="__HERMES_TERMINATED__\n", returncode=0)
    session = _remote_session(registry, env, "proc_remote_missing_identity")
    session.host_start_time = None

    result = registry.kill_process(session.id, source="remote_missing_identity")

    assert result["status"] == "error"
    assert registry._running[session.id] is session
    assert session.exited is False
    assert env.commands == []


def test_remote_natural_exit_during_kill_keeps_exit_metadata_and_one_completion(
    registry, monkeypatch
):
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    holder = {}

    def finish_naturally():
        session = holder["session"]
        with session._lock:
            session.exit_code = 124
            session.exited = True
            session.completion_reason = "exited"
            session.termination_source = ""
        registry._move_to_finished(session)

    env = _RemoteKillEnv(output="__HERMES_TERMINATED__\n", on_kill=finish_naturally)
    session = _remote_session(registry, env, "proc_remote_natural_race")
    holder["session"] = session
    session.notify_on_complete = True

    result = registry.kill_process(session.id, source="remote_racing_kill")

    assert result["status"] == "already_exited"
    assert session.exit_code == 124
    assert session.completion_reason == "exited"
    assert session.termination_source == ""
    assert registry.completion_queue.qsize() == 1


def test_promote_collision_never_overwrites_owner_or_arms_candidate_deadline(
    registry, monkeypatch
):
    owner = _make_session(sid="proc_collision", command="owner")
    candidate = _make_session(sid=owner.id, command="candidate")
    candidate.execution_deadline = time.time() + 60
    registry._running[owner.id] = owner
    arm = MagicMock()
    monkeypatch.setattr(registry, "_arm_execution_deadline", arm)

    with pytest.raises(RuntimeError, match="collision"):
        registry.promote(
            candidate,
            notification_metadata={"notify_on_complete": True},
        )

    assert registry._running[owner.id] is owner
    assert candidate.notify_on_complete is False
    assert candidate._deadline_timer is None
    arm.assert_not_called()


def test_unregistered_completion_collision_never_removes_existing_owner(
    registry, monkeypatch
):
    owner = _make_session(sid="proc_finish_collision", command="owner")
    candidate = _make_session(
        sid=owner.id, command="candidate", exited=True, exit_code=0
    )
    registry._running[owner.id] = owner
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

    registry._move_to_finished(candidate)

    assert registry._running[owner.id] is owner
    assert owner.id not in registry._finished
    assert registry.completion_queue.empty()
    assert not candidate._completion_event.is_set()


def test_promote_clears_stale_completion_event_for_live_session(registry, monkeypatch):
    session = _make_session(sid="proc_stale_completion_event")
    session._completion_event.set()
    monkeypatch.setattr(registry, "_arm_execution_deadline", lambda _session: None)

    assert registry.promote(session)

    assert registry._running[session.id] is session
    assert not session._completion_event.is_set()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-tree regression")
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_spawn_reader_start_baseexception_contains_child_and_propagates(
    registry, monkeypatch, tmp_path, error_type
):
    from tools import process_registry as process_registry_module

    created = []
    containment = []
    real_popen = subprocess.Popen
    real_rescue_discard = registry.rescue_discard

    def capture_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        return proc

    def capture_rescue(*args, **kwargs):
        result = real_rescue_discard(*args, **kwargs)
        containment.append(result)
        return result

    def fail_start(_thread):
        raise error_type("reader start failed")

    monkeypatch.setattr(process_registry_module.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(process_registry_module.threading.Thread, "start", fail_start)
    monkeypatch.setattr(registry, "rescue_discard", capture_rescue)

    with pytest.raises(error_type, match="reader start failed"):
        registry.spawn_local(
            f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(30)'",
            cwd=str(tmp_path),
            execution_timeout=30,
            defer_registration=True,
        )

    assert len(created) == 1
    proc = created[0]
    start = registry._safe_host_start_time(proc.pid)
    try:
        assert _wait_until(lambda: proc.poll() is not None)
        assert containment and containment[0]["status"] in {
            "killed",
            "already_exited",
        }
        assert not registry._running
        assert not registry._finished
    finally:
        if proc.poll() is None:
            ProcessRegistry._terminate_host_pid(proc.pid, start)
            proc.wait(timeout=5)


class _ArtifactEnv:
    def __init__(self, payload, exit_returncode=0):
        self.payload = payload
        self.exit_returncode = exit_returncode
        self.commands = []

    def execute(self, command, timeout=None):
        self.commands.append(command)
        if "kill -0" in command:
            return {"output": "1\n", "returncode": 0}
        if ".exit" in command:
            return {"output": self.payload, "returncode": self.exit_returncode}
        return {"output": "", "returncode": 0}


@pytest.mark.parametrize(
    ("payload", "accepted"),
    [
        pytest.param("1", False, id="partial-no-newline"),
        pytest.param("-1\n", False, id="negative"),
        pytest.param("x" * 1024 + "\n1\n", False, id="oversize"),
        pytest.param("124\n", True, id="valid-posix-code"),
    ],
)
def test_nonlocal_exit_artifact_is_strict_complete_and_bounded(
    registry, monkeypatch, payload, accepted
):
    env = _ArtifactEnv(payload)
    session = _remote_session(registry, env, f"proc_artifact_{accepted}_{len(payload)}")
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    log_path, pid_path, exit_path = registry._env_artifact_paths(env, session.id)

    registry._poll_env_once(
        session,
        env,
        log_path,
        pid_path,
        exit_path,
        0,
    )

    assert session.exited is accepted
    assert session.exit_code == (124 if accepted else None)
    assert (session.id in registry._finished) is accepted
    exit_commands = [command for command in env.commands if ".exit" in command]
    assert len(exit_commands) == 1
    assert "cat " not in exit_commands[0]


def test_nonlocal_exit_artifact_transport_124_is_unknown(registry, monkeypatch):
    env = _ArtifactEnv("124\n", exit_returncode=124)
    session = _remote_session(registry, env, "proc_artifact_transport_timeout")
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    paths = registry._env_artifact_paths(env, session.id)

    registry._poll_env_once(session, env, *paths, 0)

    assert session.exited is False
    assert registry._running[session.id] is session


def test_nonlocal_exit_artifact_is_written_by_atomic_rename(registry, monkeypatch):
    class LaunchEnv:
        def __init__(self):
            self.commands = []

        def execute(self, command, timeout=None, **_kwargs):
            self.commands.append(command)
            return {"output": "4321\n", "returncode": 0}

    env = LaunchEnv()
    fake_reader = MagicMock()
    monkeypatch.setattr(
        "tools.process_registry.threading.Thread", lambda **_kw: fake_reader
    )

    registry.spawn_via_env(env, "echo hello", defer_registration=True)

    launch = env.commands[0]
    assert ".exit.tmp" in launch
    assert "mv -f" in launch
    fake_reader.start.assert_called_once()


def test_nonlocal_log_transport_and_payload_stay_bounded_after_retention_limit(
    registry, monkeypatch
):
    payload = ["x" * (2 * 1024 * 1024)]
    responses = []
    commands = []

    class LogEnv:
        def execute(self, command, timeout=None):
            commands.append(command)
            if "kill -0" in command:
                return {"output": "0\n", "returncode": 0}
            if ".log" in command:
                if "tail -c" in command:
                    output = (
                        f"__HERMES_LOG_BYTES__:{len(payload[0].encode())}\n"
                        + payload[0][-session.max_output_chars :]
                    )
                else:
                    output = payload[0]
                responses.append(len(output))
                return {"output": output, "returncode": 0}
            return {"output": "", "returncode": 0}

    env = LogEnv()
    session = _remote_session(registry, env, "proc_large_remote_log")
    log_path, pid_path, exit_path = registry._env_artifact_paths(env, session.id)
    emitted = []
    monkeypatch.setattr(
        registry, "_emit_output", lambda _session, chunk: emitted.append(chunk)
    )

    previous = registry._poll_env_once(
        session,
        env,
        log_path,
        pid_path,
        exit_path,
        0,
    )
    payload[0] += "END"
    registry._poll_env_once(
        session,
        env,
        log_path,
        pid_path,
        exit_path,
        previous,
    )

    log_commands = [command for command in commands if ".log" in command]
    assert all(
        f"tail -c {session.max_output_chars}" in command for command in log_commands
    )
    assert max(responses) <= session.max_output_chars + 64
    assert len(session.output_buffer) <= session.max_output_chars
    assert emitted[-1] == "END"


def test_local_kill_exception_restores_nonterminal_metadata_and_owner(
    registry, monkeypatch
):
    proc = _spawn_python_sleep(30)
    original_terminate = registry._terminate_host_pid
    session = _detached_host_session(
        registry,
        proc,
        "proc_local_kill_fault",
        deadline=time.time() + 30,
    )
    timer = MagicMock()
    session._deadline_timer = timer
    reason_before = session.completion_reason
    source_before = session.termination_source
    monkeypatch.setattr(
        registry,
        "_terminate_host_pid",
        MagicMock(side_effect=PermissionError("signal denied")),
    )

    try:
        result = registry.kill_process(session.id, source="faulting_kill")

        assert result["status"] == "error"
        assert registry._running[session.id] is session
        assert session.exited is False
        assert session.exit_code is None
        assert session.completion_reason == reason_before
        assert session.termination_source == source_before
        assert session._deadline_timer is timer
        timer.cancel.assert_not_called()
    finally:
        if proc.poll() is None:
            original_terminate(proc.pid, session.host_start_time)
        proc.wait(timeout=5)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-tree regression")
def test_broken_deadline_timer_and_first_kill_retry_until_one_real_completion(
    registry, monkeypatch
):
    proc = _spawn_python_sleep(30)
    deadline = time.time() + 0.4
    session = _detached_host_session(
        registry, proc, "proc_deadline_fallback", deadline=deadline
    )
    registry._running.pop(session.id)
    session.notify_on_complete = True
    reason_before = session.completion_reason
    source_before = session.termination_source
    original_terminate = registry._terminate_host_pid
    terminate_calls = []

    def fail_once_then_terminate(pid, expected_start=None):
        terminate_calls.append((pid, expected_start))
        if len(terminate_calls) == 1:
            return False
        return original_terminate(pid, expected_start)

    class BrokenTimer:
        def __init__(self, *_args, **_kwargs):
            self.daemon = False

        def start(self):
            raise RuntimeError("timer start failed")

        def cancel(self):
            pass

    monkeypatch.setattr(registry, "_terminate_host_pid", fail_once_then_terminate)
    monkeypatch.setattr("tools.process_registry.threading.Timer", BrokenTimer)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

    try:
        with pytest.raises(RuntimeError, match="timer start failed"):
            registry.promote(session)

        first = registry.kill_process(session.id, source="first_kill_fault")
        assert first["status"] == "error"
        assert registry._running[session.id] is session
        assert session.exited is False
        assert session.completion_reason == reason_before
        assert session.termination_source == source_before
        assert session.execution_deadline == deadline
        assert not session._completion_event.is_set()

        assert session._completion_event.wait(timeout=5)
        assert _wait_until(lambda: proc.poll() is not None)
        assert len(terminate_calls) >= 2
        assert registry._finished[session.id] is session
        assert session.completion_reason == "killed"
        assert session.termination_source == "execution_timeout"
        assert registry.completion_queue.qsize() == 1
    finally:
        if proc.poll() is None:
            original_terminate(proc.pid, session.host_start_time)
        proc.wait(timeout=5)
