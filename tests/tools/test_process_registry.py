"""Tests for tools/process_registry.py — ProcessRegistry query methods, pruning, checkpoint."""

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress

import pytest
from unittest.mock import MagicMock, patch

from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_FORCE_PREFIX
from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
    FINISHED_TTL_SECONDS,
    MAX_PROCESSES,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


@pytest.fixture(autouse=True)
def _reset_systemd_scope_cache():
    """Reset the cached ``systemd-run --user --scope`` availability flag
    before each test so a probe run on a real systemd host (where
    ``INVOCATION_ID`` is set) doesn't leak into tests that mock
    ``subprocess.Popen``. Tests that exercise the probe directly reset the
    cache themselves."""
    import tools.process_registry as _pr

    original = _pr._SYSTEMD_SCOPE_AVAILABLE
    _pr._SYSTEMD_SCOPE_AVAILABLE = False
    yield
    _pr._SYSTEMD_SCOPE_AVAILABLE = original


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


def _spawn_python_sleep(seconds: float) -> subprocess.Popen:
    """Spawn a portable short-lived Python sleep process."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
    )


def test_kill_started_since_preserves_preexisting_and_foreign_processes(registry):
    old = _make_session(sid="proc_old", task_id="session-a")
    finished = _make_session(
        sid="proc_finished", task_id="session-a", exited=True, exit_code=0
    )
    registry._running[old.id] = old
    registry._finished[finished.id] = finished
    baseline = registry.snapshot_running_ids("session-a")

    new = _make_session(sid="proc_new", task_id="session-a")
    foreign = _make_session(sid="proc_foreign", task_id="session-b")
    registry._running[new.id] = new
    registry._running[foreign.id] = foreign

    calls = []

    def fake_kill(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {"status": "killed"}

    registry.kill_process = fake_kill

    assert baseline == frozenset({"proc_old"})
    assert registry.kill_started_since(
        "session-a", baseline, source="gateway_turn_timeout"
    ) == 1
    assert calls == [
        (
            "proc_new",
            {
                "source": "gateway_turn_timeout",
                "consume_output": True,
            },
        )
    ]


def test_kill_all_backward_compat_and_exclude_ids(registry):
    """kill_all keeps its historical default behavior (kill everything for
    the task, consume_output=False, source='kill_all') and honors the new
    exclude_ids kwarg that kill_started_since delegates through (#76188)."""
    a = _make_session(sid="proc_a", task_id="session-a")
    b = _make_session(sid="proc_b", task_id="session-a")
    registry._running[a.id] = a
    registry._running[b.id] = b

    calls = []

    def fake_kill(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {"status": "killed"}

    registry.kill_process = fake_kill

    assert registry.kill_all("session-a", exclude_ids=frozenset({"proc_a"})) == 1
    assert calls == [
        ("proc_b", {"source": "kill_all", "consume_output": False})
    ]

    calls.clear()
    assert registry.kill_all("session-a") == 2
    assert sorted(c[0] for c in calls) == ["proc_a", "proc_b"]


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll a predicate until it returns truthy or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.mark.windows_only
def test_write_stdin_uses_str_for_windows_pty(registry):
    """pywinpty expects str input; bytes raises a PyString conversion error.

    Windows-only: the str-vs-bytes choice IS the ``_IS_WINDOWS`` branch, and
    the real pty handle it must satisfy (pywinpty) does not exist elsewhere.
    """
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-win")
    session._pty = _FakePty()
    registry._running[session.id] = session

    result = registry.write_stdin(session.id, "hello\n")

    assert result == {"status": "ok", "bytes_written": 6}
    assert written == ["hello\n"]
    assert isinstance(written[0], str)


@pytest.mark.linux_only
def test_write_stdin_uses_bytes_for_posix_pty(registry):
    """The POSIX counterpart: ptyprocess expects bytes, not str."""
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-posix")
    session._pty = _FakePty()
    registry._running[session.id] = session

    result = registry.write_stdin(session.id, "hello\n")

    assert result == {"status": "ok", "bytes_written": 6}
    assert written == [b"hello\n"]


@pytest.mark.windows_only
def test_submit_stdin_uses_crlf_for_windows_pty(registry):
    """Enter on a Windows PTY is a carriage return, not a bare LF.

    ConPTY cooked input only ends a line on ``\\r``; a bare ``\\n`` through
    pywinpty is never delivered to a blocking line read (Python readline,
    Go bufio.Scanner — the exact hang seen live with ``gh auth login``'s
    "Press Enter to open the browser" prompt). submit_stdin must append
    ``\\r\\n`` for Windows PTY sessions.
    """
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-win-submit")
    session._pty = _FakePty()
    registry._running[session.id] = session

    result = registry.submit_stdin(session.id, "Y")

    assert result["status"] == "ok"
    assert written == ["Y\r\n"]


@pytest.mark.windows_only
def test_submit_stdin_keeps_lf_for_windows_pipe(registry):
    """Non-PTY (Popen pipe) sessions keep the plain LF on Windows."""
    session = _make_session(sid="pipe-win-submit")
    fake_stdin = MagicMock()
    session.process = MagicMock()
    session.process.stdin = fake_stdin
    registry._running[session.id] = session

    result = registry.submit_stdin(session.id, "Y")

    assert result["status"] == "ok"
    fake_stdin.write.assert_called_once_with("Y\n")


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
    assert session.exited is True
    assert session.exit_code == 0
    assert moved == ["proc_reader_live"]


# =========================================================================
# Incremental UTF-8 decoding across chunk boundaries
# (ported from openclaw/openclaw#112325)
# =========================================================================


class _FakeChunkBuffer:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read1(self, _n):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeChunkStdout:
    def __init__(self, chunks):
        self.buffer = _FakeChunkBuffer(chunks)


class _FakeChunkProcess:
    def __init__(self, chunks):
        self.stdout = _FakeChunkStdout(chunks)
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


def _run_reader(registry, monkeypatch, chunks, sid="proc_utf8"):
    session = _make_session(sid=sid)
    session.process = _FakeChunkProcess(chunks)
    monkeypatch.setattr(registry, "_check_watch_patterns", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_emit_output", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_move_to_finished", lambda _s: None)
    registry._reader_loop(session)
    return session


def test_reader_loop_reassembles_multibyte_char_split_across_chunks(registry, monkeypatch):
    """A UTF-8 char split across two read1() chunks must not become U+FFFD.

    Before the incremental decoder, each chunk was decoded statelessly with
    ``errors="replace"``, so ``é`` (0xC3 0xA9) straddling a 4096-byte read
    boundary decoded as two replacement characters.
    """
    session = _run_reader(registry, monkeypatch, [b"caf\xc3", b"\xa9 ok\n"])
    assert session.output_buffer == "café ok\n"
    assert "\ufffd" not in session.output_buffer


def test_reader_loop_reassembles_four_byte_char_split_three_ways(registry, monkeypatch):
    """A 4-byte emoji fragmented across three reads reassembles cleanly."""
    session = _run_reader(registry, monkeypatch, [b"\xf0", b"\x9f\x92", b"\xa9\n"])
    assert session.output_buffer == "\U0001f4a9\n"


def test_reader_loop_flushes_truncated_multibyte_tail_at_eof(registry, monkeypatch):
    """A sequence truncated by process exit flushes as a single U+FFFD."""
    session = _run_reader(registry, monkeypatch, [b"ok \xe2\x82"])
    assert session.output_buffer == "ok \ufffd"


def test_reader_loop_still_replaces_genuinely_invalid_bytes(registry, monkeypatch):
    """Truly invalid bytes keep the errors="replace" behavior."""
    session = _run_reader(registry, monkeypatch, [b"ok\xffdone\n"])
    assert session.output_buffer == "ok\ufffddone\n"


def test_pty_reader_loop_reassembles_multibyte_char_split_across_chunks(registry, monkeypatch):
    """The PTY reader gets the same incremental-decode treatment."""

    class _FakePty:
        def __init__(self, chunks):
            self._chunks = list(chunks)
            self.exitstatus = 0

        def isalive(self):
            return bool(self._chunks)

        def read(self, _n):
            if self._chunks:
                return self._chunks.pop(0)
            raise EOFError

        def wait(self):
            return 0

    session = _make_session(sid="proc_pty_utf8")
    session._pty = _FakePty([b"caf\xc3", b"\xa9\n"])
    monkeypatch.setattr(registry, "_check_watch_patterns", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_emit_output", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_move_to_finished", lambda _s: None)

    registry._pty_reader_loop(session)

    assert session.output_buffer == "café\n"
    assert "\ufffd" not in session.output_buffer


# =========================================================================
# Orphaned-pipe reconciliation (issue #17327)
# =========================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: uses setsid/fcntl")
class TestOrphanedPipeReconciliation:
    """Regression tests for issue #17327.

    `hermes update` in Feishu spawned a background subprocess that restarted
    the gateway; the direct child exited quickly but a descendant daemon
    held the stdout pipe open. `_reader_loop.finally` never ran, so
    `session.exited` stayed False and the agent polled 74 times over 7
    minutes, all returning `status: running`.

    The fix is `_reconcile_local_exit()`: poll() and wait() now check the
    direct `Popen.poll()` before trusting `session.exited`.
    """

    def test_reconcile_flips_exited_when_direct_child_done(self, registry):
        """Direct child exited but reader thread is blocked on orphaned pipe."""
        # Simulate the orphaned-pipe scenario: direct child exited, but a
        # descendant holds stdout open so the reader never sees EOF.
        # Approach: spawn `sh -c 'sleep 10 &'` with setsid — sh forks the
        # sleep into a new session group, exits immediately, but sleep
        # inherits the stdout pipe and keeps it open.
        proc = subprocess.Popen(
            ["sh", "-c", "exec 1>&2; ( sleep 30 ) & disown; exit 0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        s = _make_session(sid="proc_orphan_test")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        # Wait for the direct child to exit. We don't start a reader thread,
        # so session.exited stays False (mimicking the stuck-reader state).
        assert _wait_until(lambda: proc.poll() is not None, timeout=5.0), (
            "Direct child should exit quickly (sh exits, sleep descendant "
            "holds the pipe open)"
        )

        # Before the fix: poll would return "running" forever.
        # After the fix: poll reconciles against proc.poll() and flips.
        assert s.exited is False  # Precondition: reader hasn't updated it.
        result = registry.poll(s.id)
        assert result["status"] == "exited", (
            f"Expected reconciled 'exited' status; got {result!r}. "
            "This is issue #17327 — reader is blocked on orphaned pipe."
        )
        assert result["exit_code"] == 0
        assert s.exited is True
        assert s.id in registry._finished
        assert s.id not in registry._running

        # Clean up the orphaned descendant.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def test_wait_returns_when_reader_blocked(self, registry):
        """wait() must also reconcile — not just poll()."""
        proc = subprocess.Popen(
            ["sh", "-c", "( sleep 30 ) & disown; exit 0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        s = _make_session(sid="proc_wait_orphan")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        assert _wait_until(lambda: proc.poll() is not None, timeout=5.0)

        start = time.monotonic()
        result = registry.wait(s.id, timeout=10)
        elapsed = time.monotonic() - start

        assert result["status"] == "exited", result
        assert elapsed < 5.0, (
            f"wait() should return ~immediately via reconcile; took {elapsed:.1f}s"
        )

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

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

    def test_env_poller_quotes_temp_paths_with_spaces(self, registry):
        session = _make_session(sid="proc_space")
        session.exited = False

        class FakeEnv:
            def __init__(self):
                self.commands = []
                self._responses = iter([
                    {"output": "6 0\nhello\n"},
                    {"output": "1\n"},
                    {"output": "0\n"},
                ])

            def execute(self, command, **kwargs):
                self.commands.append((command, kwargs))
                return next(self._responses)

        env = FakeEnv()

        with patch("tools.process_registry.time.sleep", return_value=None), \
            patch.object(registry, "_move_to_finished"):
            registry._env_poller_loop(
                session,
                env,
                "/path with spaces/hermes_bg.log",
                "/path with spaces/hermes_bg.pid",
                "/path with spaces/hermes_bg.exit",
            )

        assert "'/path with spaces/hermes_bg.log'" in env.commands[0][0]
        assert "cat '/path with spaces/hermes_bg.log'" not in env.commands[0][0]
        assert env.commands[1][0] == "kill -0 \"$(cat '/path with spaces/hermes_bg.pid' 2>/dev/null)\" 2>/dev/null; echo $?"
        assert env.commands[2][0] == "cat '/path with spaces/hermes_bg.exit' 2>/dev/null"


class TestEnvPollerIncrementalRead:
    """The sandbox log poller must read only new bytes, not the whole file.

    Reading the whole file every poll made one poll cost grow with the total
    output so far, so a long noisy job re-sent all of its output over the
    docker or SSH channel every two seconds.
    """

    @staticmethod
    def _run_poller(registry, session, responses):
        """Drive one poll cycle and hand back the commands the env saw."""

        class FakeEnv:
            def __init__(self):
                self.commands = []
                self._responses = iter(responses)

            def execute(self, command, **kwargs):
                self.commands.append(command)
                return next(self._responses)

        env = FakeEnv()
        with patch("tools.process_registry.time.sleep", return_value=None), \
            patch.object(registry, "_move_to_finished"):
            registry._env_poller_loop(
                session, env, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit"
            )
        return env.commands

    def test_read_command_asks_only_for_new_bytes(self):
        cmd = ProcessRegistry._log_delta_command("'/tmp/bg.log'", 4096)
        # The offset is carried into the command, and the file is opened with
        # tail rather than cat.
        assert "O=4096" in cmd
        assert "tail -c +$((O+1)) '/tmp/bg.log'" in cmd
        assert "cat '/tmp/bg.log'" not in cmd

    def test_read_command_starts_from_zero_on_first_poll(self):
        cmd = ProcessRegistry._log_delta_command("'/tmp/bg.log'", 0)
        assert "O=0" in cmd

    @pytest.mark.skipif(not shutil.which("sh"), reason="needs a POSIX sh")
    def test_read_command_holds_back_a_split_utf8_sequence(self, tmp_path):
        """A multibyte character straddling two polls must not be split.

        The backend decodes each execute() result on its own, so returning
        the first byte of an 'é' in one poll and the rest in the next would
        yield replacement characters in the transcript (and break watch
        patterns at the seam). Every prefix of a mixed ASCII/2/3/4-byte
        string must come back decodable, with at most 3 bytes held back and
        nothing held back once the trailing character is complete.
        """
        full = "hé😀中a\n€bz🚀".encode()
        log = tmp_path / "bg.log"
        quoted = shlex.quote(str(log))
        for n in range(1, len(full) + 1):
            log.write_bytes(full[:n])
            out = subprocess.run(
                ["sh", "-c", ProcessRegistry._log_delta_command(quoted, 0)],
                capture_output=True, timeout=30,
            ).stdout
            header, _, delta = out.partition(b"\n")
            size, _offset = map(int, header.split())
            delta.decode("utf-8")  # must not raise
            assert delta == full[:size]
            complete = full[:n].decode("utf-8", "ignore").encode() == full[:n]
            assert (n - size) == 0 if complete else 0 < (n - size) <= 3

    def test_first_poll_reads_from_the_start(self, registry):
        session = _make_session(sid="proc_delta")
        session.exited = False
        commands = self._run_poller(
            registry,
            session,
            [
                {"output": "11 0\nfirst chunk"},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert "O=0" in commands[0]
        assert session.output_buffer == "first chunk"

    def test_delta_is_appended_not_replaced(self, registry):
        session = _make_session(sid="proc_append", output="already here ")
        session.exited = False
        self._run_poller(
            registry,
            session,
            [
                {"output": "8 0\nand new"},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "already here and new"

    def test_second_poll_asks_from_where_the_first_one_stopped(self, registry):
        session = _make_session(sid="proc_two_polls")
        session.exited = False
        commands = self._run_poller(
            registry,
            session,
            [
                {"output": "11 0\nfirst chunk"},
                {"output": "0\n"},          # still running, poll again
                {"output": "17 11\n and more"},
                {"output": "1\n"},          # gone now
                {"output": "0\n"},
            ],
        )
        assert "O=0" in commands[0]
        # The second read starts at byte 11, so the first chunk is not sent
        # a second time.
        assert "O=11" in commands[2]
        assert session.output_buffer == "first chunk and more"

    def test_truncated_log_drops_the_stale_buffer(self, registry):
        session = _make_session(sid="proc_rotate")
        session.exited = False
        # The second read reports offset 0 even though the first one left off
        # at byte 11. The file no longer reaches that byte, so it was rotated
        # or truncated and the buffer we hold no longer matches it.
        self._run_poller(
            registry,
            session,
            [
                {"output": "11 0\nfirst chunk"},
                {"output": "0\n"},          # still running, poll again
                {"output": "5 0\nfresh"},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "fresh"

    def test_unreadable_header_leaves_the_buffer_alone(self, registry):
        session = _make_session(sid="proc_bad", output="keep me")
        session.exited = False
        # No header at all, for example when the shell is missing one of the
        # tools the command needs.
        self._run_poller(
            registry,
            session,
            [
                {"output": ""},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "keep me"

    def test_buffer_stays_within_the_cap(self, registry):
        session = _make_session(sid="proc_cap")
        session.exited = False
        session.max_output_chars = 10
        self._run_poller(
            registry,
            session,
            [
                {"output": "20 0\n" + "x" * 20},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "x" * 10


# =========================================================================
# Popen leak prevention
# =========================================================================

class TestPopenLeakOnSetupFailure:
    """Regression for issue #2749: subprocess orphaned when post-Popen setup raises."""

    def test_popen_killed_when_thread_creation_fails(self, registry):
        """If Thread() raises after Popen, proc must be killed — not orphaned."""
        killed = []

        proc = MagicMock()
        proc.pid = 9999
        proc.stdout = iter([])
        proc.stdin = MagicMock()
        proc.poll.return_value = None

        def fake_kill():
            killed.append(True)

        proc.kill = fake_kill
        proc.wait = MagicMock()

        def boom(*args, **kwargs):
            raise RuntimeError("Thread creation failed")

        # proc.pid is a MagicMock-backed fake; os.getpgid(fake_pid) would query
        # the real OS for an arbitrary PID. On a busy host that PID may exist,
        # in which case spawn_local's primary cleanup path
        # (os.killpg(os.getpgid(pid), SIGKILL)) succeeds against an UNRELATED
        # real process group and proc.kill() is never reached — flaky failure,
        # and a real risk of SIGKILLing an innocent process group. Force the
        # ProcessLookupError fallback so the test deterministically exercises
        # proc.kill() and never issues a real killpg.
        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("threading.Thread", side_effect=boom), \
             patch("os.getpgid", side_effect=ProcessLookupError), \
             patch.object(registry, "_write_checkpoint"):
            with pytest.raises(RuntimeError, match="Thread creation failed"):
                registry.spawn_local("echo hello", cwd="/tmp")

        assert killed, "proc.kill() must be called when post-Popen setup raises"

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
        assert "&& { node server.js &>/tmp/srv.log & }" in shell_cmd[2]

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
        shell_cmd = captured_cmd[0][2]
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

    def test_recover_dead_wrapper_retries_unreaped_systemd_scope(
        self, registry, tmp_path, monkeypatch
    ):
        checkpoint = tmp_path / "procs.json"
        entry = {
            "session_id": "proc_dead_scope",
            "command": "daemonize",
            "pid": 999999999,
            "pid_scope": "host",
            "host_start_time": 123.0,
            "systemd_unit": "hermes-worker-proc_dead_scope.scope",
        }
        checkpoint.write_text(json.dumps([entry]))
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(registry, "_is_host_pid_alive", lambda *_args: False)

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), patch(
            "tools.process_registry._stop_systemd_unit", return_value=False
        ) as stop_unit:
            assert registry.recover_from_checkpoint() == 0

        stop_unit.assert_called_once_with(entry["systemd_unit"])
        assert json.loads(checkpoint.read_text()) == [entry]

    def test_recover_dead_wrapper_drops_reaped_systemd_scope(
        self, registry, tmp_path, monkeypatch
    ):
        checkpoint = tmp_path / "procs.json"
        entry = {
            "session_id": "proc_dead_scope",
            "command": "daemonize",
            "pid": 999999999,
            "pid_scope": "host",
            "host_start_time": 123.0,
            "systemd_unit": "hermes-worker-proc_dead_scope.scope",
        }
        checkpoint.write_text(json.dumps([entry]))
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(registry, "_is_host_pid_alive", lambda *_args: False)

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), patch(
            "tools.process_registry._stop_systemd_unit", return_value=True
        ) as stop_unit:
            assert registry.recover_from_checkpoint() == 0

        stop_unit.assert_called_once_with(entry["systemd_unit"])
        assert json.loads(checkpoint.read_text()) == []


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

    def test_checkpoint_redacts_command_with_inline_secret(self, registry, tmp_path):
        """Issue #77484: the checkpoint file persists raw commands; inline
        credentials (e.g. ``curl -H 'Authorization: Bearer sk-...'``) must be
        redacted before write. Recovery only uses command for display/logging
        (the process is already running), so masking is lossless."""
        checkpoint = tmp_path / "procs.json"
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            secret = "sk-secret1234567890"
            command = f"curl -H 'Authorization: Bearer {secret}' http://x"
            s = _make_session(sid="proc_secret", command=command)
            s.pid = 12345
            s.host_start_time = int(time.time())
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads(checkpoint.read_text())
            assert data[0]["session_id"] == "proc_secret"
            assert secret not in data[0]["command"]
            assert data[0]["command"] != command

# =========================================================================
# Kill process
# =========================================================================

class TestKillProcess:
    def test_kill_already_exited(self, registry):
        s = _make_session(exited=True, exit_code=0)
        registry._finished[s.id] = s
        result = registry.kill_process(s.id)
        assert result["status"] == "already_exited"


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

    @staticmethod
    def _bind_natural_pty_exit_race(registry, monkeypatch, sid):
        """Pause the PTY reader after it observes natural death but before settlement."""
        observed = threading.Event()
        release = threading.Event()

        class NaturalExitPty:
            exitstatus = None

            def isalive(self):
                observed.set()
                assert release.wait(2), "test did not release the PTY reader"
                self.exitstatus = 0
                return False

            def terminate(self, force=False):
                raise OSError("ECHILD after natural exit")

            def wait(self):
                return self.exitstatus

        session = _make_session(sid=sid, command="exit 0", output="complete output")
        session.pid = 424242
        session.pid_scope = "host"
        session.host_start_time = 1
        session._pty = NaturalExitPty()
        session.notify_on_complete = True
        registry._running[session.id] = session
        monkeypatch.setattr(registry, "_is_host_pid_alive", lambda _pid: False)
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        monkeypatch.setattr("tools.process_registry.save_completed_result", lambda _session: None)

        reader = threading.Thread(target=registry._pty_reader_loop, args=(session,))
        reader.start()
        assert observed.wait(2), "PTY reader did not observe natural exit"
        return session, reader, release

    def test_kill_error_does_not_consume_natural_pty_completion(self, registry, monkeypatch):
        session, reader, release = self._bind_natural_pty_exit_race(
            registry, monkeypatch, "proc_natural_kill_error"
        )
        try:
            with patch("tools.process_registry.os.kill", side_effect=ProcessLookupError):
                result = registry.kill_process(session.id, consume_output=True)

            assert result["status"] == "error"
            assert session.id not in registry._completion_consumed
            assert session.exited is False
        finally:
            release.set()
            reader.join(timeout=2)

        assert not reader.is_alive()
        assert session.exit_code == 0
        assert session.completion_reason == "exited"
        event = registry.completion_queue.get_nowait()
        assert event["output"] == "complete output"
        assert event["exit_code"] == 0
        assert event["completion_reason"] == "exited"

    def test_kill_all_preserves_natural_pty_exit_provenance(self, registry, monkeypatch):
        session, reader, release = self._bind_natural_pty_exit_race(
            registry, monkeypatch, "proc_natural_kill_all"
        )
        try:
            with patch("tools.process_registry.os.kill", side_effect=ProcessLookupError):
                assert registry.kill_all(source="kill_all", consume_output=False) == 0
        finally:
            release.set()
            reader.join(timeout=2)

        assert not reader.is_alive()
        assert session.exit_code == 0
        assert session.completion_reason == "exited"
        assert session.termination_source == ""
        event = registry.completion_queue.get_nowait()
        assert event["exit_code"] == 0
        assert event["completion_reason"] == "exited"
        assert event["termination_source"] == ""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX local-pipe reader ownership")
    def test_direct_kill_waits_for_pipe_reader_final_tail(self, registry, monkeypatch, tmp_path):
        """A delivered signal is provenance; the real reader publishes the tail once."""
        code = (
            "import signal, sys, time\n"
            "def finish(_signum, _frame):\n"
            "    print('FINAL-TAIL-41', flush=True)\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, finish)\n"
            "print('READY-41', flush=True)\n"
            "while True: time.sleep(1)\n"
        )
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
        # Keep this real public spawn/read/queue path isolated from persistent test
        # state; the terminal owner and completion queue are not mocked.
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        monkeypatch.setattr("tools.process_registry.save_completed_result", lambda _s: None)
        session = registry.spawn_local(command, cwd=str(tmp_path))
        session.notify_on_complete = True
        try:
            assert _wait_until(lambda: "READY-41" in session.output_buffer), "reader never received readiness"
            result = registry.kill_process(session.id, source="test.pipe.kill", consume_output=True)
            assert result["status"] == "killed"
            assert result["output"].endswith("FINAL-TAIL-41\n")
            assert result["exit_code"] == 0
            assert result["completion_reason"] == "killed"
            assert result["termination_source"] == "test.pipe.kill"
            event = registry.completion_queue.get_nowait()
            assert event["output"].endswith("FINAL-TAIL-41\n")
            assert event["exit_code"] == 0
            assert event["completion_reason"] == "killed"
            assert event["termination_source"] == "test.pipe.kill"
            assert registry.completion_queue.empty()
        finally:
            self._reap_child(session.process)

    @staticmethod
    def _spawn_pipe_parent_with_detached_writer(registry, monkeypatch, tmp_path, *, tail, waits_for_signal):
        """Exercise the public pipe reader after its direct child is gone.

        The small double-fork keeps a *test-owned* stdout writer alive after the
        direct shell/Python child exits.  That is the production orphaned-pipe
        shape: the real reader, not a fake finalizer, decides when its bounded
        drain is complete.
        """
        mode = (
            "def finish(_signum, _frame):\n"
            f"    print({tail!r}, flush=True)\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, finish)\n"
            f"print('READY-{tail}', flush=True)\n"
            "while True: time.sleep(0.05)\n"
            if waits_for_signal else
            f"print('READY-{tail}', flush=True)\nprint({tail!r}, flush=True)\n"
        )
        code = (
            "import os, signal, sys, time\n"
            "child = os.fork()\n"
            "if child == 0:\n"
            "    os.setsid()\n"
            "    grandchild = os.fork()\n"
            "    if grandchild:\n"
            "        os._exit(0)\n"
            "    time.sleep(1.2)\n"
            "    os._exit(0)\n"
            "os.waitpid(child, 0)\n"
            + mode
        )
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
        # Keep durable state out of this fixture.  The reader/finalizer and
        # completion queue remain the real public path under test.
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        monkeypatch.setattr("tools.process_registry.save_completed_result", lambda _s: None)
        session = registry.spawn_local(command, cwd=str(tmp_path))
        session.notify_on_complete = True
        assert _wait_until(lambda: f"READY-{tail}" in session.output_buffer), "reader never received readiness"
        return session

    @pytest.mark.linux_only
    def test_pipe_reader_owns_post_death_probe_error_and_natural_tail(self, registry, monkeypatch, tmp_path):
        """P1: a post-death probe fault cannot consume or relabel a held pipe tail."""
        session = self._spawn_pipe_parent_with_detached_writer(
            registry, monkeypatch, tmp_path, tail="FINAL-P1-41", waits_for_signal=False,
        )

        def dead_probe_then_raise(pid, expected_start, on_direct_signal=None, **_kwargs):
            assert _wait_until(lambda: session.process.poll() is not None), "direct child did not exit"
            assert not session._completion_event.is_set(), "reader was not held by its inherited pipe"
            raise RuntimeError("post-death host probe failure")

        monkeypatch.setattr(registry, "_terminate_host_pid", dead_probe_then_raise)
        try:
            result = registry.kill_process(session.id, source="test.p1", consume_output=True)
            assert result == {"status": "error", "error": "post-death host probe failure"}
            assert session.id not in registry._completion_consumed
            assert session._completion_event.wait(timeout=3), "reader did not finish its bounded drain"
            assert session.output_buffer.endswith("FINAL-P1-41\n")
            assert (session.exit_code, session.completion_reason, session.termination_source) == (0, "exited", "")
            event = registry.completion_queue.get_nowait()
            assert event["output"].endswith("FINAL-P1-41\n")
            assert (event["exit_code"], event["completion_reason"], event["termination_source"]) == (0, "exited", "")
            assert registry.completion_queue.empty()
        finally:
            self._reap_child(session.process)

    @pytest.mark.linux_only
    def test_pipe_reader_owns_post_signal_adapter_error_and_delivered_provenance(self, registry, monkeypatch, tmp_path):
        """P2: delivery is real evidence, while the held reader owns terminal publication."""
        session = self._spawn_pipe_parent_with_detached_writer(
            registry, monkeypatch, tmp_path, tail="FINAL-P2-41", waits_for_signal=True,
        )
        original_terminate = registry._terminate_host_pid

        def deliver_then_raise(*args, **kwargs):
            original_terminate(*args, **kwargs)
            assert not session._completion_event.is_set(), "reader was not held after real signal delivery"
            raise RuntimeError("post-signal termination adapter failure")

        monkeypatch.setattr(registry, "_terminate_host_pid", deliver_then_raise)
        try:
            result = registry.kill_process(session.id, source="test.p2", consume_output=True)
            assert result == {"status": "error", "error": "post-signal termination adapter failure"}
            assert session.id not in registry._completion_consumed
            assert session._completion_event.wait(timeout=3), "reader did not publish the delivered tail"
            assert session.id not in registry._completion_consumed
            assert session.output_buffer.endswith("FINAL-P2-41\n")
            assert (session.exit_code, session.completion_reason, session.termination_source) == (0, "killed", "test.p2")
            drained = registry.drain_notifications(skip_poll_observed=False)
            assert len(drained) == 1
            event, _text = drained[0]
            assert event["output"].endswith("FINAL-P2-41\n")
            assert (event["exit_code"], event["completion_reason"], event["termination_source"]) == (0, "killed", "test.p2")
            assert registry.drain_notifications(skip_poll_observed=False) == []
            repeat = registry.kill_process(session.id, source="test.p2", consume_output=False)
            assert repeat["status"] == "already_exited"
            assert repeat["output"].endswith("FINAL-P2-41\n")
            assert registry.drain_notifications(skip_poll_observed=False) == []
        finally:
            self._reap_child(session.process)

    @pytest.mark.linux_only
    def test_pipe_reader_settles_kill_disposition_after_keyboard_interrupt(
        self, registry, monkeypatch, tmp_path
    ):
        """A caller interruption cannot leave the terminal reader parked forever."""
        session = self._spawn_pipe_parent_with_detached_writer(
            registry, monkeypatch, tmp_path, tail="FINAL-INTERRUPT-41", waits_for_signal=True,
        )
        original_terminate = registry._terminate_host_pid

        def deliver_then_interrupt(*args, **kwargs):
            original_terminate(*args, **kwargs)
            assert not session._completion_event.is_set()
            raise KeyboardInterrupt

        monkeypatch.setattr(registry, "_terminate_host_pid", deliver_then_interrupt)
        try:
            with pytest.raises(KeyboardInterrupt):
                registry.kill_process(session.id, source="test.interrupt", consume_output=True)

            assert session.id not in registry._completion_consumed
            assert session._completion_event.wait(timeout=3), "interrupted caller abandoned reader disposition"
            assert session.output_buffer.endswith("FINAL-INTERRUPT-41\n")
            assert (session.exit_code, session.completion_reason, session.termination_source) == (
                0, "killed", "test.interrupt"
            )
            event = registry.completion_queue.get_nowait()
            assert event["output"].endswith("FINAL-INTERRUPT-41\n")
            assert (event["exit_code"], event["completion_reason"], event["termination_source"]) == (
                0, "killed", "test.interrupt"
            )
            assert registry.completion_queue.empty()
            repeat = registry.kill_process(session.id, source="test.interrupt.retry", consume_output=False)
            assert repeat["status"] == "already_exited"
        finally:
            self._reap_child(session.process)

    @pytest.mark.linux_only
    def test_pipe_stopping_does_not_consume_reader_completion(self, registry, monkeypatch, tmp_path):
        """A stopping result has not returned the reader's final output."""
        entered_finish = threading.Event()
        release_finish = threading.Event()
        original_finish = registry._finish_reader

        def held_finish(*args, **kwargs):
            entered_finish.set()
            assert release_finish.wait(7)
            return original_finish(*args, **kwargs)

        monkeypatch.setattr(registry, "_finish_reader", held_finish)
        session = self._spawn_pipe_parent_with_detached_writer(
            registry, monkeypatch, tmp_path, tail="FINAL-STOPPING-41", waits_for_signal=True,
        )
        try:
            result = registry.kill_process(session.id, source="test.stopping", consume_output=True)
            assert entered_finish.is_set()
            assert result["status"] == "stopping"
            assert session.id not in registry._completion_consumed
            release_finish.set()
            assert session._completion_event.wait(timeout=3)
            assert session.id not in registry._completion_consumed
            drained = registry.drain_notifications(skip_poll_observed=False)
            assert len(drained) == 1
            event, _text = drained[0]
            assert event["output"].endswith("FINAL-STOPPING-41\n")
            assert (event["exit_code"], event["completion_reason"], event["termination_source"]) == (
                0, "killed", "test.stopping"
            )
            assert registry.drain_notifications(skip_poll_observed=False) == []
        finally:
            release_finish.set()
            self._reap_child(session.process)

    @pytest.mark.linux_only
    @pytest.mark.parametrize("reader_finishes_first", [True, False])
    def test_pipe_reader_guard_race_preserves_natural_owner(self, registry, monkeypatch, tmp_path, reader_finishes_first):
        """P4: both observer/reader schedules leave the reader as the sole owner."""
        tail = f"FINAL-P4-41-{'reader' if reader_finishes_first else 'observer'}"
        session = self._spawn_pipe_parent_with_detached_writer(
            registry, monkeypatch, tmp_path, tail=tail, waits_for_signal=False,
        )
        observed_death = threading.Event()

        def observe_death_then_raise(pid, expected_start, on_direct_signal=None, **_kwargs):
            assert _wait_until(lambda: session.process.poll() is not None), "observer never saw actual direct-child death"
            observed_death.set()
            if reader_finishes_first:
                assert session._completion_event.wait(timeout=3), "reader did not win its scheduled race"
            else:
                assert not session._completion_event.is_set(), "reader did not remain draining for observer-first race"
            raise RuntimeError("post-poll guard failure")

        monkeypatch.setattr(registry, "_terminate_host_pid", observe_death_then_raise)
        try:
            result = registry.kill_process(session.id, source="test.p4", consume_output=True)
            assert observed_death.is_set()
            assert result == {"status": "error", "error": "post-poll guard failure"}
            assert session.id not in registry._completion_consumed
            assert session._completion_event.wait(timeout=3), "reader did not finish its scheduled race"
            assert session.output_buffer.endswith(f"{tail}\n")
            assert (session.exit_code, session.completion_reason, session.termination_source) == (0, "exited", "")
            event = registry.completion_queue.get_nowait()
            assert event["output"].endswith(f"{tail}\n")
            assert (event["exit_code"], event["completion_reason"], event["termination_source"]) == (0, "exited", "")
            assert registry.completion_queue.empty()
        finally:
            self._reap_child(session.process)

    def _bind_real_child(self, registry, sid: str):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        s = _make_session(sid=sid, command="sleep 60")
        s.process = proc
        s.pid = proc.pid
        s.host_start_time = ProcessRegistry._safe_host_start_time(proc.pid)
        registry._running[s.id] = s
        return s, proc

    @staticmethod
    def _reap_child(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass
            with suppress(Exception):
                proc.wait(timeout=2)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX real-child kill/liveness matrix")
    def test_kill_process_pre_signal_exception_keeps_child_running(self, registry):
        """A raise before any signal must not mark the OS child killed."""
        s, proc = self._bind_real_child(registry, "proc_pre_signal")
        pid = proc.pid
        orig = registry._signal_kill

        def boom(_session, _session_id, _consume_output):
            raise PermissionError("EPERM before signal")

        try:
            registry._signal_kill = boom
            result = registry.kill_process(s.id, source="test", consume_output=True)
            still = ProcessRegistry._is_host_pid_alive(pid)
            poll = registry.poll(s.id)
            listed = [p for p in registry.list_sessions() if p["session_id"] == s.id]
            assert result["status"] == "error"
            assert s.exited is False
            assert still is True
            assert proc.poll() is None
            assert poll["status"] == "running"
            assert s.id in registry._running
            assert s.id not in registry._completion_consumed
            assert listed and listed[0]["status"] == "running"
        finally:
            registry._signal_kill = orig
            self._reap_child(proc)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX real-child kill/liveness matrix")
    def test_kill_process_failed_kill_still_alive_stays_retryable(self, registry):
        """Failed signal with a living child must remain running so close/kill_all retry."""
        s, proc = self._bind_real_child(registry, "proc_failed_kill")
        pid = proc.pid
        orig = registry._signal_kill

        def fail_without_death(_session, _session_id, _consume_output):
            raise OSError("kill failed; child still alive")

        try:
            registry._signal_kill = fail_without_death
            counted = registry.kill_all(source="test", consume_output=False)
            still = ProcessRegistry._is_host_pid_alive(pid)
            poll = registry.poll(s.id)
            would_retry = any(
                p["session_id"] == s.id and p["status"] == "running"
                for p in registry.list_sessions()
            )
            assert counted == 0
            assert still is True
            assert proc.poll() is None
            assert s.exited is False
            assert poll["status"] == "running"
            assert s.id not in registry._completion_consumed
            assert would_retry is True
        finally:
            registry._signal_kill = orig
            self._reap_child(proc)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX real-child kill/liveness matrix")
    def test_kill_process_post_death_exception_marks_waitable_exit(self, registry):
        """After a waitable child is dead, a later exception may finish metadata.

        Does not treat a mock RuntimeError as proof of a historic race. The child
        must be reaped via the Popen handle (or matching host identity gone).
        """
        s, proc = self._bind_real_child(registry, "proc_post_death")
        pid = proc.pid
        orig = registry._signal_kill

        def kill_then_raise(session, _session_id, _consume_output):
            os.kill(session.process.pid, signal.SIGKILL)
            session.process.wait(timeout=2)
            raise RuntimeError("psutil after SIGKILL")

        try:
            registry._signal_kill = kill_then_raise
            result = registry.kill_process(s.id, source="test", consume_output=True)
            marked_on_kill_return = s.exited
            still = ProcessRegistry._is_host_pid_alive(pid)
            poll = registry.poll(s.id)
            assert result["status"] == "error"
            assert still is False
            assert proc.poll() is not None
            assert marked_on_kill_return is True
            assert s.exited is True
            assert poll["status"] != "running"
            assert s.exit_code == proc.poll()
            assert s.exit_code != -15
        finally:
            registry._signal_kill = orig
            self._reap_child(proc)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX real-child kill/liveness matrix")
    def test_kill_process_post_death_identity_gone_without_waitable_handle(self, registry):
        """Matching host identity gone (PID reuse-safe) may finish even without Popen.wait."""
        s, proc = self._bind_real_child(registry, "proc_identity_gone")
        pid = proc.pid
        start = s.host_start_time
        orig = registry._signal_kill

        def kill_reap_then_drop_handle(session, _session_id, _consume_output):
            os.kill(session.process.pid, signal.SIGKILL)
            session.process.wait(timeout=2)
            session.process = None
            raise OSError("checkpoint after death")

        try:
            registry._signal_kill = kill_reap_then_drop_handle
            result = registry.kill_process(s.id, source="test", consume_output=True)
            marked_on_kill_return = s.exited
            still = ProcessRegistry._host_pid_is_ours(pid, start)
            assert result["status"] == "error"
            assert still is False
            assert marked_on_kill_return is True
            assert s.exited is True
            assert registry.poll(s.id)["status"] != "running"
        finally:
            registry._signal_kill = orig
            self._reap_child(proc)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX real-child kill/liveness matrix")
    def test_kill_process_unknown_start_time_keeps_live_host_child_running(self, registry):
        """Alive host PID + unreadable start time is UNKNOWN, not proven identity-gone."""
        s, proc = self._bind_real_child(registry, "proc_unknown_start")
        pid = proc.pid
        orig = registry._signal_kill
        assert s.host_start_time is not None
        assert ProcessRegistry._is_host_pid_alive(pid) is True

        def boom(_session, _session_id, _consume_output):
            raise OSError("failed kill; start-time unreadable")

        try:
            registry._signal_kill = boom
            with patch.object(
                ProcessRegistry, "_safe_host_start_time", staticmethod(lambda _pid=None: None)
            ):
                result = registry.kill_process(s.id, source="test", consume_output=True)
            still = ProcessRegistry._is_host_pid_alive(pid)
            poll = registry.poll(s.id)
            listed = [p for p in registry.list_sessions() if p["session_id"] == s.id]
            assert result["status"] == "error"
            assert still is True
            assert proc.poll() is None
            assert s.exited is False
            assert poll["status"] == "running"
            assert s.id not in registry._completion_consumed
            assert listed and listed[0]["status"] == "running"
        finally:
            registry._signal_kill = orig
            self._reap_child(proc)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX real-child kill/liveness matrix")
    def test_kill_process_pty_unknown_start_time_keeps_live_child_running(self, registry):
        """PTY / dropped handle + UNKNOWN start time must not mark a live host PID."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        s = _make_session(sid="proc_pty_unknown", command="pty")
        s.process = None
        s.pid = proc.pid
        s.pid_scope = "host"
        s.host_start_time = ProcessRegistry._safe_host_start_time(proc.pid)
        s._pty = object()
        registry._running[s.id] = s
        pid = proc.pid
        orig = registry._signal_kill
        assert s.host_start_time is not None

        def boom(_session, _session_id, _consume_output):
            raise OSError("pty terminate failed")

        try:
            registry._signal_kill = boom
            with patch.object(
                ProcessRegistry, "_safe_host_start_time", staticmethod(lambda _pid=None: None)
            ):
                result = registry.kill_process(s.id, source="test", consume_output=True)
            still = ProcessRegistry._is_host_pid_alive(pid)
            poll = registry.poll(s.id)
            listed = [p for p in registry.list_sessions() if p["session_id"] == s.id]
            assert result["status"] == "error"
            assert still is True
            assert proc.poll() is None
            assert s.exited is False
            assert poll["status"] == "running"
            assert s.id not in registry._completion_consumed
            assert listed and listed[0]["status"] == "running"
        finally:
            registry._signal_kill = orig
            self._reap_child(proc)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX real-child kill/liveness matrix")
    def test_kill_process_waitable_zero_exit_marks_zero_not_minus15(self, registry):
        """A waitable poll() of 0 is authoritative; do not invent -15."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(0)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait(timeout=2)
        s = _make_session(sid="proc_waitable_zero", command="true")
        s.process = proc
        s.pid = proc.pid
        s.pid_scope = "host"
        s.host_start_time = 1
        registry._running[s.id] = s
        orig = registry._signal_kill

        def boom(_session, _session_id, _consume_output):
            raise RuntimeError("after natural zero exit")

        try:
            registry._signal_kill = boom
            result = registry.kill_process(s.id, source="test", consume_output=True)
            assert result["status"] == "error"
            assert s.exited is True
            assert s.exit_code == 0
            assert s.exit_code == proc.poll()
            assert s.exit_code != -15
        finally:
            registry._signal_kill = orig

    def test_kill_process_sandbox_pid_not_host_identity_gone(self, registry):
        """Sandbox PIDs must not be host-checked even if host_start_time is set."""
        s = _make_session(sid="proc_sandbox_scope", command="sandbox")
        s.process = None
        s.pid = 1
        s.pid_scope = "sandbox"
        s.host_start_time = 1
        s.env_ref = MagicMock()
        registry._running[s.id] = s
        orig = registry._signal_kill

        def boom(_session, _session_id, _consume_output):
            raise OSError("remote kill failed")

        try:
            registry._signal_kill = boom
            result = registry.kill_process(s.id, source="test", consume_output=True)
            listed = [p for p in registry.list_sessions() if p["session_id"] == s.id]
            assert result["status"] == "error"
            assert s.exited is False
            assert registry.poll(s.id)["status"] == "running"
            assert s.id not in registry._completion_consumed
            assert listed and listed[0]["status"] == "running"
        finally:
            registry._signal_kill = orig

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX real-child kill/liveness matrix")
    def test_kill_process_pid_reuse_mismatch_marks_without_killing_stranger(self, registry):
        """Live stranger with mismatched start time: mark ours gone, do not signal the stranger."""
        stranger = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        s = _make_session(sid="proc_pid_reuse", command="old")
        s.process = None
        s.pid = stranger.pid
        s.pid_scope = "host"
        s.host_start_time = 1
        registry._running[s.id] = s
        pid = stranger.pid
        orig = registry._signal_kill
        assert ProcessRegistry._is_host_pid_alive(pid) is True

        def boom(_session, _session_id, _consume_output):
            raise OSError("checkpoint after original death")

        try:
            registry._signal_kill = boom
            result = registry.kill_process(s.id, source="test", consume_output=True)
            stranger_alive = ProcessRegistry._is_host_pid_alive(pid)
            assert result["status"] == "error"
            assert s.exited is True
            assert s.exit_code != -15
            assert stranger_alive is True
            assert stranger.poll() is None
        finally:
            registry._signal_kill = orig
            self._reap_child(stranger)

    @pytest.mark.linux_only
    def test_recovered_natural_death_during_kill_error_stays_unconsumed(self, registry, tmp_path, monkeypatch):
        """R1/R2: recovery observes death, but a failed operation did not deliver it."""
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.25)"])
        checkpoint = tmp_path / "processes.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_recovered_natural", "command": "sleep", "pid": proc.pid,
            "pid_scope": "host", "host_start_time": ProcessRegistry._safe_host_start_time(proc.pid),
            "task_id": "t1", "notify_on_complete": True,
        }]))
        monkeypatch.setattr("tools.process_registry.save_completed_result", lambda _s: None)
        try:
            with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
                assert registry.recover_from_checkpoint() == 1
                session = registry.get("proc_recovered_natural")

                def natural_then_error(*_args):
                    proc.wait(timeout=2)
                    raise RuntimeError("adapter failed after natural death")

                monkeypatch.setattr(registry, "_signal_kill", natural_then_error)
                result = registry.kill_process(session.id, source="test.failed", consume_output=True)

            assert result == {"status": "error", "error": "adapter failed after natural death"}
            assert (session.exit_code, session.completion_reason, session.termination_source) == (None, "exited", "")
            assert session.id not in registry._completion_consumed
            event = registry.completion_queue.get_nowait()
            assert (event["completion_reason"], event["termination_source"]) == ("exited", "")
        finally:
            self._reap_child(proc)

    @pytest.mark.linux_only
    @pytest.mark.parametrize("consume_output", [True, False])
    def test_pipe_kill_commits_disposition_before_notification_drain(
        self, registry, monkeypatch, tmp_path, consume_output
    ):
        """R6: consumption is committed before an actual notification drain."""
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        monkeypatch.setattr("tools.process_registry.save_completed_result", lambda _s: None)
        session = registry.spawn_local(
            f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)'", cwd=str(tmp_path)
        )
        session.notify_on_complete = True
        visible = []
        drained = []
        original_put = registry.completion_queue.put

        def inspect_put(event):
            visible.append(registry.is_completion_consumed(session.id))
            original_put(event)
            drained.append(registry.drain_notifications(skip_poll_observed=False))

        monkeypatch.setattr(registry.completion_queue, "put", inspect_put)
        try:
            result = registry.kill_process(session.id, source="test.scope", consume_output=consume_output)
            assert result["status"] == "killed"
            assert visible == [consume_output]
            assert [len(batch) for batch in drained] == [0 if consume_output else 1]
        finally:
            self._reap_child(session.process)

    @pytest.mark.linux_only
    def test_pipe_kill_stops_owned_scope_before_completed_return(self, registry, monkeypatch, tmp_path):
        """R5: completed reader return cannot skip its owned scope cleanup."""
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        monkeypatch.setattr("tools.process_registry.save_completed_result", lambda _s: None)
        stopped = []
        monkeypatch.setattr("tools.process_registry._stop_systemd_unit", lambda unit: stopped.append(unit) or True)
        session = registry.spawn_local(
            f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)'", cwd=str(tmp_path)
        )
        session.systemd_unit = "hermes-worker-test.scope"
        try:
            result = registry.kill_process(session.id, source="test.scope", consume_output=True)
            assert result["status"] == "killed"
            assert stopped == ["hermes-worker-test.scope"]
        finally:
            self._reap_child(session.process)

    def test_concurrent_failed_source_cannot_replace_delivering_source(self, registry, monkeypatch):
        """R7: pending provenance belongs to the invocation whose signal succeeded."""
        session = _make_session(sid="proc_source_race")
        session.process = MagicMock(pid=424242)
        session._reader_thread = MagicMock()
        session.notify_on_complete = True
        registry._running[session.id] = session
        success_entered = threading.Event()
        failed_done = threading.Event()
        results = {}

        def terminate(_pid, _start, on_direct_signal=None, **_kwargs):
            if threading.current_thread().name == "success-kill":
                success_entered.set()
                assert failed_done.wait(2)
                on_direct_signal()
                session._reader_settlement_ready.set()
                registry._finish_exited(session, 0)
                return
            assert success_entered.wait(2)
            failed_done.set()
            raise PermissionError("failed competitor")

        monkeypatch.setattr(registry, "_terminate_host_pid", terminate)
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        monkeypatch.setattr("tools.process_registry.save_completed_result", lambda _s: None)
        threads = [
            threading.Thread(name="success-kill", target=lambda: results.setdefault(
                "success", registry.kill_process(session.id, source="delivered", consume_output=False))),
            threading.Thread(name="failed-kill", target=lambda: results.setdefault(
                "failed", registry.kill_process(session.id, source="failed", consume_output=True))),
        ]
        threads[0].start()
        threads[1].start()
        for thread in threads:
            thread.join(timeout=3)
            assert not thread.is_alive()
        assert results["failed"]["status"] == "error"
        assert session.termination_source == "delivered"
        assert registry.completion_queue.get_nowait()["termination_source"] == "delivered"

    @pytest.mark.linux_only
    @pytest.mark.parametrize("signal_path", ["psutil", "posix_fallback"])
    def test_direct_signal_and_provenance_are_atomic_for_reader_publication(
        self, registry, monkeypatch, tmp_path, signal_path
    ):
        """The real signal and its provenance are one terminal-lock operation."""
        import psutil

        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        saved = []
        monkeypatch.setattr(
            "tools.process_registry.save_completed_result",
            lambda session: saved.append((
                session.exit_code, session.completion_reason, session.termination_source,
                session.output_buffer,
            )),
        )
        code = (
            "import signal,time\n"
            "def finish(_signum, _frame):\n"
            " raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, finish)\n"
            "print('ATOMIC-READY-41', flush=True)\n"
            "while True: time.sleep(0.05)\n"
        )
        session = registry.spawn_local(
            f"exec {shlex.quote(sys.executable)} -c {shlex.quote(code)}", cwd=str(tmp_path)
        )
        session.notify_on_complete = True
        assert _wait_until(lambda: "ATOMIC-READY-41" in session.output_buffer)
        if signal_path == "posix_fallback":
            original_process = psutil.Process

            def inaccessible_parent(pid):
                if pid == session.pid:
                    raise OSError("psutil unavailable")
                return original_process(pid)

            monkeypatch.setattr(psutil, "Process", inaccessible_parent)

        delivered = threading.Event()
        allow_record = threading.Event()
        reader_attempted = threading.Event()
        original_terminate = registry._terminate_host_pid
        original_finish_exited = registry._finish_exited

        def observe_reader(*args, **kwargs):
            reader_attempted.set()
            return original_finish_exited(*args, **kwargs)

        def hold_between_signal_and_record(pid, expected_start, on_direct_signal=None, **kwargs):
            def held_record(*callback_args):
                delivered.set()
                assert allow_record.wait(3)
                on_direct_signal(*callback_args)

            return original_terminate(pid, expected_start, held_record, **kwargs)

        monkeypatch.setattr(registry, "_finish_exited", observe_reader)
        monkeypatch.setattr(registry, "_terminate_host_pid", hold_between_signal_and_record)
        result = {}
        killer = threading.Thread(target=lambda: result.update(
            registry.kill_process(session.id, source=f"test.atomic.{signal_path}", consume_output=False)
        ))
        killer.start()
        try:
            assert delivered.wait(2), "real direct signal was not delivered"
            assert reader_attempted.wait(2), "reader did not attempt terminal publication"
            published_before_record = session._completion_event.is_set()
        finally:
            allow_record.set()
            killer.join(timeout=3)
            self._reap_child(session.process)

        assert not killer.is_alive()
        assert not published_before_record
        assert result["status"] == "killed"
        expected = (0, "killed", f"test.atomic.{signal_path}")
        assert (session.exit_code, session.completion_reason, session.termination_source) == expected
        assert saved == [(*expected, session.output_buffer)]
        event = registry.completion_queue.get_nowait()
        assert (event["exit_code"], event["completion_reason"], event["termination_source"]) == expected
        assert event["output"].endswith("ATOMIC-READY-41\n")
        assert registry.completion_queue.empty()

    def test_pipe_stopping_result_still_stops_owned_scope(self, registry, monkeypatch):
        """R5 bounded-stopping return cannot bypass its cgroup obligation."""
        session = _make_session(sid="proc_scope_stopping")
        session.process = MagicMock(pid=424242)
        session._reader_thread = MagicMock()
        session._completion_event = MagicMock()
        session._completion_event.wait.return_value = False
        session.systemd_unit = "hermes-worker-stopping.scope"
        registry._running[session.id] = session
        stopped = []
        monkeypatch.setattr(registry, "_signal_kill", lambda *_args: None)
        monkeypatch.setattr("tools.process_registry._stop_systemd_unit", lambda unit: stopped.append(unit) or True)

        result = registry.kill_process(session.id, source="test.stopping", consume_output=True)

        assert result["status"] == "stopping"
        assert stopped == ["hermes-worker-stopping.scope"]

    @pytest.mark.linux_only
    def test_posix_fallback_signal_records_delivering_source(self, registry, monkeypatch, tmp_path):
        """R7: os.kill fallback is a successful delivery, not missing provenance."""
        import psutil

        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        monkeypatch.setattr("tools.process_registry.save_completed_result", lambda _s: None)
        session = registry.spawn_local(
            f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)'", cwd=str(tmp_path)
        )
        original_process = psutil.Process

        def inaccessible_parent(pid):
            if pid == session.pid:
                raise OSError("psutil unavailable")
            return original_process(pid)

        monkeypatch.setattr(psutil, "Process", inaccessible_parent)
        try:
            result = registry.kill_process(session.id, source="test.posix.fallback", consume_output=True)
            assert result["status"] == "killed"
            assert result["termination_source"] == "test.posix.fallback"
            assert session.termination_source == "test.posix.fallback"
        finally:
            self._reap_child(session.process)

    @pytest.mark.linux_only
    @pytest.mark.parametrize("observer", ["poll", "wait"])
    def test_public_observer_waits_for_pipe_reader_terminal_owner(
        self, registry, monkeypatch, tmp_path, observer
    ):
        """R8: poll/wait cannot publish while a signalled reader owns cutoff."""
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        saved = []
        monkeypatch.setattr(
            "tools.process_registry.save_completed_result",
            lambda session: saved.append((
                session.exit_code, session.completion_reason, session.termination_source,
                session.output_buffer,
            )),
        )
        entered_finish = threading.Event()
        release_finish = threading.Event()
        original_finish = registry._finish_reader

        def held_finish(*args, **kwargs):
            entered_finish.set()
            assert release_finish.wait(2)
            return original_finish(*args, **kwargs)

        monkeypatch.setattr(registry, "_finish_reader", held_finish)
        session = registry.spawn_local(
            f"{shlex.quote(sys.executable)} -c "
            "'import signal,time; "
            "signal.signal(signal.SIGTERM, lambda *_: (print(\"final-tail\", flush=True), raise_exit())); "
            "exec(\"def raise_exit():\\n raise SystemExit(0)\"); "
            "print(\"ready\", flush=True); time.sleep(60)'",
            cwd=str(tmp_path),
        )
        session.notify_on_complete = True
        assert _wait_until(lambda: "ready" in session.output_buffer)
        kill_result = {}
        killer = threading.Thread(target=lambda: kill_result.update(
            registry.kill_process(session.id, source="test.reader.owner", consume_output=False)
        ))
        killer.start()
        assert entered_finish.wait(2)
        observer_result = {}
        observe = registry.poll if observer == "poll" else lambda sid: registry.wait(sid, timeout=2)
        poller = threading.Thread(target=lambda: observer_result.update(observe(session.id)))
        poller.start()
        try:
            published_before_owner = session._completion_event.wait(timeout=1.0)
            observer_returned_before_owner = not poller.is_alive()
            observer_marked_exit = session.exited
            observer_queued_completion = not registry.completion_queue.empty()
        finally:
            release_finish.set()
            poller.join(timeout=3)
            killer.join(timeout=3)
            self._reap_child(session.process)

        assert not poller.is_alive()
        assert not killer.is_alive()
        assert not published_before_owner
        assert not observer_marked_exit
        assert not observer_queued_completion
        if observer == "poll":
            assert observer_returned_before_owner
            assert observer_result["status"] == "running"
            observer_result = registry.poll(session.id)
        else:
            assert not observer_returned_before_owner
        assert observer_result["status"] == "exited"
        expected = (
            0, "killed", "test.reader.owner"
        )
        assert (session.exit_code, session.completion_reason, session.termination_source) == expected
        assert "final-tail" in observer_result.get("output", observer_result.get("output_preview", ""))
        assert kill_result["termination_source"] == "test.reader.owner"
        assert saved == [(*expected, session.output_buffer)]
        event = registry.completion_queue.get_nowait()
        assert (event["exit_code"], event["completion_reason"], event["termination_source"]) == expected
        assert event["output"].endswith("final-tail\n")
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

from tools.process_registry_notifications import format_process_notification


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

    @pytest.mark.windows_only
    def test_windows_invokes_taskkill_with_tree_and_force_flags(self, monkeypatch):
        """The Windows branch must shell out to ``taskkill /PID N /T /F``.

        Windows-only: ``taskkill.exe`` is the thing under test and only exists
        here — with a faked ``_IS_WINDOWS`` the argv was asserted against a
        binary that could never have run.
        """
        from tools import process_registry as pr

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return MagicMock(returncode=0, stderr="", stdout="")

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

        monkeypatch.setattr(psutil, "Process", _FakeParent)
        # This test covers only the SIGTERM tree-walk ordering; disable the
        # SIGKILL-escalation step (which would call psutil.wait_procs on the
        # fakes) by setting the grace to 0.
        monkeypatch.setattr(pr.ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.0))

        pr.ProcessRegistry._terminate_host_pid(12345)

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

        monkeypatch.setattr(psutil, "Process", boom)
        monkeypatch.setattr(pr.os, "kill", fake_kill)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert kill_calls == [(12345, signal.SIGTERM)]


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

    def test_list_redacts_command_and_output(self, monkeypatch):
        """`process(action=list)` redacts command + output_preview — issue #77484.

        The list branch previously returned raw ``command[:200]`` and
        ``output_preview[-200:]`` with no redaction wrap, leaking inline
        secrets (unlike poll/log/wait/kill).
        """
        pr, sess = self._setup(
            monkeypatch, "curl -H 'Authorization: Bearer sk-abc123def456ghi789jkl012345'",
            "opaque token sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGG output",
        )
        out = json.loads(pr._handle_process({"action": "list"}))
        assert len(out["processes"]) >= 1
        entry = out["processes"][0]
        assert "sk-abc123def456ghi789jkl012345" not in entry["command"]
        assert "sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGG" not in entry["output_preview"]
        assert "curl" in entry["command"]

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
    """Regression tests for issue #68915.

    When an agent command backgrounds a long-lived process (``node server.js
    &``), the grandchild inherits the write end of the reader's stdout pipe.
    The direct bash child exits, but the pipe never EOFs — the old blocking
    ``read1()`` parked the reader thread forever, ``session.exited`` never
    flipped on its own, and ``notify_on_complete`` never fired. The reader
    must instead terminate shortly after the direct child exits, even while
    a descendant still holds the pipe open.
    """

    def test_reader_exits_when_orphan_holds_pipe(self, registry):
        """Reader loop must return promptly after the direct child exits,
        even though a backgrounded descendant keeps the pipe open."""
        proc = subprocess.Popen(
            ["sh", "-c", "echo started; sleep 30 & exit 0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            preexec_fn=os.setsid,
        )
        s = _make_session(sid="proc_orphan_reader")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        done = threading.Event()

        def _run():
            registry._reader_loop(s)
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            # The direct child exits immediately; the reader must notice and
            # return well before the 30s descendant releases the pipe.
            assert done.wait(timeout=10.0), (
                "_reader_loop is still blocked on the orphan-held pipe "
                "(issue #68915) — session.exited would never flip and "
                "notify_on_complete would never fire"
            )
            assert s.exited is True
            assert s.exit_code == 0
            assert s.completion_reason == "exited"
            assert "started" in s.output_buffer
            assert s.id in registry._finished
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def test_reader_exit_fires_notify_on_complete(self, registry):
        """The autonomous completion notification must not depend on a
        poll()/wait() call when an orphan holds the pipe."""
        proc = subprocess.Popen(
            ["sh", "-c", "sleep 30 & echo bg-started"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            preexec_fn=os.setsid,
        )
        s = _make_session(sid="proc_orphan_notify")
        s.process = proc
        s.pid = proc.pid
        s.notify_on_complete = True
        registry._running[s.id] = s

        done = threading.Event()

        def _run():
            registry._reader_loop(s)
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            assert done.wait(timeout=10.0), (
                "_reader_loop blocked — completion notification lost (#68915)"
            )
            # Exactly one completion event must have been queued.
            item = registry.completion_queue.get_nowait()
            assert item["type"] == "completion"
            assert item["session_id"] == s.id
            assert item["exit_code"] == 0
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def test_normal_eof_retains_large_direct_child_tail(self, registry, monkeypatch, tmp_path):
        """The orphan deadline cannot truncate a direct child's normal EOF tail."""
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        monkeypatch.setattr("tools.process_registry.save_completed_result", lambda _s: None)
        payload = "DIRECT-LARGE-TAIL-41:" + ("x" * 32_768) + ":END-41\n"
        code = f"import sys; sys.stdout.write({payload!r}); sys.stdout.flush(); raise SystemExit(9)"
        session = registry.spawn_local(
            f"exec {shlex.quote(sys.executable)} -c {shlex.quote(code)}", cwd=str(tmp_path)
        )
        session.notify_on_complete = True

        assert session._completion_event.wait(timeout=5), "normal EOF reader did not finish"
        result = registry.wait(session.id, timeout=1)

        assert result["status"] == "exited"
        assert result["exit_code"] == 9
        assert session.output_buffer == payload
        assert result["output"].endswith(":END-41\n")
        event = registry.completion_queue.get_nowait()
        assert event["exit_code"] == 9
        assert event["output"].endswith(":END-41\n")
        assert registry.completion_queue.empty()

    @pytest.mark.parametrize("public_entry", ["spawn_local", "adopt_local"])
    def test_public_poll_wait_finishes_while_orphan_keeps_writing(
        self, registry, monkeypatch, tmp_path, public_entry
    ):
        """A busy inherited pipe cannot extend the direct command's lifetime."""
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
        monkeypatch.setattr("tools.process_registry.save_completed_result", lambda _s: None)
        code = (
            "import os,time\n"
            "child=os.fork()\n"
            "if child == 0:\n"
            " for i in range(100):\n"
            "  print(f'ORPHAN-WRITE-{i}', flush=True)\n"
            "  time.sleep(0.05)\n"
            " os._exit(0)\n"
            "print(f'DESCENDANT={child}', flush=True)\n"
            "print('DIRECT-DONE-41', flush=True)\n"
            "raise SystemExit(7)\n"
        )
        if public_entry == "spawn_local":
            session = registry.spawn_local(
                f"exec {shlex.quote(sys.executable)} -c {shlex.quote(code)}", cwd=str(tmp_path)
            )
            proc = session.process
        else:
            proc = subprocess.Popen(
                [sys.executable, "-c", code], cwd=str(tmp_path), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
            )
            session = registry.adopt_local(
                proc, command="writing orphan", cwd=str(tmp_path), notify_on_complete=True,
            )
        session.notify_on_complete = True
        try:
            assert _wait_until(lambda: proc.poll() is not None, timeout=3), "direct child did not exit"
            assert _wait_until(
                lambda: "ORPHAN-WRITE-8" in session.output_buffer, timeout=2
            ), "descendant did not exercise ready reads"
            assert proc.stdout is not None and proc.stdout.closed is False

            started = time.monotonic()
            polled = registry.poll(session.id)
            waited = registry.wait(session.id, timeout=3)
            elapsed = time.monotonic() - started

            assert polled["status"] in {"running", "exited"}
            assert waited["status"] == "exited", waited
            assert elapsed < 2.5
            assert waited["exit_code"] == 7
            assert "DIRECT-DONE-41" in waited["output"]
            assert "ORPHAN-WRITE-" in waited["output"]
            assert session._reader_thread is not None and not session._reader_thread.is_alive()
            event = registry.completion_queue.get_nowait()
            assert event["exit_code"] == 7
            assert event["output"] == waited["output"]
            assert registry.completion_queue.empty()
        finally:
            with suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()

# =========================================================================
# systemd cgroup isolation for gateway-spawned local executors (#70716)
# =========================================================================
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: systemd scopes")
class TestSystemdCgroupIsolation:
    """Verify spawn_local wraps the worker in ``systemd-run --user --scope``
    when running under a supervisor and systemd-run is available, and falls
    back to the legacy ``start_new_session`` path otherwise.

    Issue #70716: local background terminal executors inherit the gateway's
    cgroup, so an OOM in a memory-heavy worker lets systemd-oomd kill the
    ENTIRE gateway cgroup, taking down the messaging control plane.
    """

    @pytest.fixture()
    def _gateway_identity(self, monkeypatch):
        """Opt-in: mark this test as running AS the live gateway process."""
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda *, cleanup_stale=False: os.getpid(),
        )

    def _fake_popen_capture(self):
        """Return (fake_popen, captured) where captured["argv"] gets the
        argv passed to subprocess.Popen."""
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = list(argv)
            captured["start_new_session"] = kwargs.get("start_new_session")
            proc = MagicMock()
            proc.pid = 4321
            proc.stdout = iter([])
            proc.stdin = MagicMock()
            proc.poll.return_value = None
            return proc

        return fake_popen, captured

    @pytest.mark.linux_only
    def test_wraps_in_systemd_scope_when_supervisor_and_available(
        self, registry, monkeypatch, _gateway_identity
    ):
        """Under a supervisor with systemd-run available, the spawn argv is
        wrapped in ``systemd-run --user --scope --unit=hermes-worker-<id>``."""
        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        # _build_systemd_scope_argv calls shutil.which — point it at a stub.
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            session = registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        assert argv[0] == "/usr/bin/systemd-run", argv
        assert "--user" in argv
        assert "--scope" in argv
        assert "--quiet" in argv, (
            "systemd-run argv must include --quiet (#70716 gap #3)"
        )
        assert "--unit" in argv
        unit_idx = argv.index("--unit")
        assert argv[unit_idx + 1].startswith("hermes-worker-"), argv
        assert argv[unit_idx + 1] == f"hermes-worker-{session.id}", (
            argv
        )  # _build_systemd_scope_argv uses bare name
        properties = [
            argv[index + 1]
            for index, value in enumerate(argv[:-1])
            if value == "--property"
        ]
        assert "MemoryAccounting=yes" in properties
        # systemd rejects OOMPolicy= on transient --scope units across the versions
        # users run (239/245/249, #102486); emitting it fails the probe and every
        # cron worker dispatch. MemoryMax + MemoryAccounting carry the isolation.
        assert not any(p.startswith("OOMPolicy=") for p in properties), properties
        memory_max = next(
            value for value in properties if value.startswith("MemoryMax=")
        )
        assert int(memory_max.split("=", 1)[1]) > 0
        # The original shell command must still be present at the tail,
        # after the ``--`` separator that prevents systemd-run from
        # interpreting command flags as its own.
        assert "--" in argv, "systemd-run argv must use -- to separate command"
        sep_idx = argv.index("--")
        assert "/bin/bash" in argv[sep_idx:]
        assert "set +m; echo hello" in argv[sep_idx:]
        # systemd-run --scope gives the worker a new cgroup but NOT a new
        # session (#70716 regression: start_new_session was False, so the
        # worker kept the parent's session + controlling terminal → SIGTTIN/
        # SIGTTOU stopped the TUI).  start_new_session=True gives systemd-run
        # (and the scoped worker below it) a private session.
        assert captured["start_new_session"] is True
        # The session must record the unit name so kill_process can stop it.
        assert session.systemd_unit == f"hermes-worker-{session.id}.scope"

    def test_falls_back_when_systemd_run_unavailable(self, registry, monkeypatch, _gateway_identity):
        """Under a supervisor but without systemd-run, fall back to the
        legacy ``start_new_session=True`` path (worker shares the gateway
        cgroup)."""
        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: False,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        # No systemd-run wrapping — direct shell invocation.
        assert argv == ["/bin/bash", "-lic", "set +m; echo hello"], argv
        assert captured["start_new_session"] is True

    def test_falls_back_when_not_under_supervisor(self, registry, monkeypatch):
        """CLI mode (no supervisor) must NOT wrap in a systemd scope even if
        systemd-run is available — isolation is a gateway concern."""
        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: False,
        )

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        assert argv == ["/bin/bash", "-lic", "set +m; echo hello"], argv
        assert captured["start_new_session"] is True

    @pytest.mark.parametrize("use_pty", [False, True])
    def test_inherited_systemd_marker_does_not_scope_interactive_cli(
        self, registry, monkeypatch, use_pty
    ):
        """A CLI inside a supervised terminal must keep workers off its tty.

        INVOCATION_ID is inherited by every descendant, so its presence
        alone must not activate the gateway-only systemd scope path.
        """
        monkeypatch.setenv("INVOCATION_ID", "herdr-service-inherited-marker")
        monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        if use_pty:
            from ptyprocess import PtyProcess

            fake_pty = MagicMock(pid=4321)
            with (
                patch.object(PtyProcess, "spawn", return_value=fake_pty) as pty_spawn,
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)
            assert pty_spawn.call_args.args[0] == [
                "/bin/bash", "-lic", "set +m; codex",
            ]
        else:
            fake_popen, captured = self._fake_popen_capture()
            with (
                patch("subprocess.Popen", side_effect=fake_popen),
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("echo hello", cwd="/tmp")
            assert captured["argv"] == [
                "/bin/bash", "-lic", "set +m; echo hello",
            ]
            assert captured["start_new_session"] is True

        assert session.systemd_unit == ""

    @pytest.mark.parametrize("use_pty", [False, True])
    def test_inherited_gateway_tree_markers_do_not_scope_child_cli(
        self, registry, monkeypatch, use_pty
    ):
        """Gateway descendants are not the gateway process that owns the PID file.

        _HERMES_GATEWAY is inherited (and set by importing gateway.run), so
        both it and INVOCATION_ID may be present in a child process. The
        PID-ownership gate must still keep the scope path off.
        """
        monkeypatch.setenv("INVOCATION_ID", "inherited-systemd-marker")
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda *, cleanup_stale=False: os.getpid() + 1,
        )
        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        if use_pty:
            from ptyprocess import PtyProcess

            fake_pty = MagicMock(pid=4321)
            with (
                patch.object(PtyProcess, "spawn", return_value=fake_pty) as pty_spawn,
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)
            assert pty_spawn.call_args.args[0] == [
                "/bin/bash", "-lic", "set +m; codex",
            ]
        else:
            fake_popen, captured = self._fake_popen_capture()
            with (
                patch("subprocess.Popen", side_effect=fake_popen),
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("echo hello", cwd="/tmp")
            assert captured["argv"] == [
                "/bin/bash", "-lic", "set +m; echo hello",
            ]
            assert captured["start_new_session"] is True

        assert session.systemd_unit == ""

    @pytest.mark.linux_only
    def test_systemd_post_spawn_failure_never_kills_gateway_process_group(
        self, registry, monkeypatch, _gateway_identity
    ):
        """Cleanup must not killpg: scope teardown is the authoritative path."""
        fake_popen, _captured = self._fake_popen_capture()
        fake_proc = fake_popen(["placeholder"])

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        broken_reader = MagicMock()
        broken_reader.start.side_effect = RuntimeError("reader failed")

        with patch("subprocess.Popen", return_value=fake_proc), \
            patch("threading.Thread", return_value=broken_reader), \
            patch("tools.process_registry._stop_systemd_unit", return_value=True) as stop_unit, \
            patch("os.killpg") as killpg, \
            patch.object(registry, "_write_checkpoint"):
            with pytest.raises(RuntimeError, match="reader failed"):
                registry.spawn_local("echo hello", cwd="/tmp")

        stop_unit.assert_called_once()
        assert stop_unit.call_args.args[0].startswith("hermes-worker-proc_")
        assert stop_unit.call_args.args[0].endswith(".scope")
        killpg.assert_not_called()

    @pytest.mark.linux_only
    def test_pty_spawn_is_wrapped_in_systemd_scope(self, registry, monkeypatch, _gateway_identity):
        """Interactive executors receive the same sibling-cgroup isolation."""
        from ptyprocess import PtyProcess

        fake_pty = MagicMock()
        fake_pty.pid = 4321

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch.object(PtyProcess, "spawn", return_value=fake_pty) as pty_spawn, \
            patch("threading.Thread", return_value=MagicMock()), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)

        argv = pty_spawn.call_args.args[0]
        assert argv[0] == "/usr/bin/systemd-run"
        assert "--scope" in argv
        assert "--unit" in argv
        assert "--" in argv
        assert argv[-3:] == ["/bin/bash", "-lic", "set +m; codex"]
        assert session.systemd_unit == f"hermes-worker-{session.id}.scope"

    @pytest.mark.linux_only
    def test_pty_spawn_failure_reaps_scope_before_distinct_pipe_fallback(
        self, registry, monkeypatch, _gateway_identity
    ):
        """A failed PTY scope must not collide with the pipe fallback scope."""
        from ptyprocess import PtyProcess

        events = []
        fake_proc = MagicMock()
        fake_proc.pid = 4321
        fake_proc.stdout = iter([])
        fake_proc.stdin = MagicMock()
        fake_proc.poll.return_value = None

        def fake_popen(argv, **_kwargs):
            events.append(("pipe", list(argv)))
            return fake_proc

        def fake_stop(unit_name):
            events.append(("stop", unit_name))
            return True

        def fail_pty(*_args, **_kwargs):
            events.append(("pty", None))
            raise RuntimeError("PTY wrapper failed after scope creation")

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch.object(PtyProcess, "spawn", side_effect=fail_pty), \
            patch("subprocess.Popen", side_effect=fake_popen), \
            patch("tools.process_registry._stop_systemd_unit", side_effect=fake_stop), \
            patch("threading.Thread", return_value=MagicMock()), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)

        assert [event[0] for event in events] == ["pty", "stop", "pipe"]
        stopped_unit = events[1][1]
        fallback_argv = events[2][1]
        assert stopped_unit == f"hermes-worker-{session.id}.scope"
        unit_idx = fallback_argv.index("--unit")
        assert fallback_argv[unit_idx + 1] == (
            f"hermes-worker-{session.id}-pipe-fallback"
        )
        assert session.systemd_unit == (
            f"hermes-worker-{session.id}-pipe-fallback.scope"
        )

    @pytest.mark.linux_only
    def test_pty_spawn_failure_does_not_fallback_when_scope_reap_fails(
        self, registry, monkeypatch, _gateway_identity
    ):
        """Do not launch a duplicate command while the failed PTY scope may live."""
        from ptyprocess import PtyProcess

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch.object(
            PtyProcess,
            "spawn",
            side_effect=RuntimeError("PTY wrapper failed after scope creation"),
        ), patch("subprocess.Popen") as pipe_spawn, patch(
            "tools.process_registry._stop_systemd_unit", return_value=False
        ) as stop_unit:
            with pytest.raises(RuntimeError, match="could not be reaped"):
                registry.spawn_local("codex", cwd="/tmp", use_pty=True)

        stop_unit.assert_called_once()
        pipe_spawn.assert_not_called()

    def test_worker_memory_limit_honors_local_guard_mb_override(self, monkeypatch):
        import tools.process_registry as pr

        monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "123")
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch("tools.process_registry.logger.warning") as warning:
            argv = pr._build_systemd_scope_argv(
                ["/bin/bash", "-lc", "true"],
                unit_suffix="test",
            )

        warning.assert_not_called()
        assert f"MemoryMax={123 * 1024 * 1024}" in argv

    def test_worker_memory_limit_caps_oversized_local_guard_override(
        self, monkeypatch
    ):
        import tools.process_registry as pr

        monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "999999")
        monkeypatch.setattr(
            pr.Path,
            "read_text",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no cgroup")),
        )
        monkeypatch.setattr(
            pr.os,
            "sysconf",
            lambda *_args: (_ for _ in ()).throw(OSError("no sysconf")),
        )

        assert pr._worker_memory_max_bytes() == pr._DEFAULT_WORKER_MEMORY_MAX_BYTES

    def test_kill_recovered_detached_already_exited_stops_persisted_scope(
        self, registry, monkeypatch
    ):
        """Recovered detached sessions whose wrapper PID is gone/recycled must
        still stop their persisted systemd scope before the already_exited
        return, while retaining the PID-reuse guard (no PID tree kill)."""
        session = _make_session(sid="proc_recovered_scope", command="daemonize")
        session.detached = True
        session.pid_scope = "host"
        session.pid = 12345
        session.host_start_time = 67890
        session.systemd_unit = "hermes-worker-proc_recovered_scope.scope"
        registry._running[session.id] = session

        stopped = []
        terminated = []
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda pid, start: False)
        monkeypatch.setattr(
            registry, "_terminate_host_pid",
            lambda pid, start, on_direct_signal=None: terminated.append((pid, start)),
        )
        monkeypatch.setattr("tools.process_registry._stop_systemd_unit", lambda unit: stopped.append(unit) or True)

        with patch.object(registry, "_write_checkpoint"):
            result = registry.kill_process(session.id)

        assert result["status"] == "already_exited"
        assert stopped == ["hermes-worker-proc_recovered_scope.scope"]
        assert terminated == []
        assert session.exited is True
        assert session.id in registry._finished
        assert session.id not in registry._running

    @pytest.mark.linux_only
    def test_systemd_run_user_scope_available_caches_after_probe(
        self, registry, monkeypatch
    ):
        """The availability check probes once and caches — a second call must
        not re-probe (and must return the same value)."""
        import tools.process_registry as pr

        # Reset the cache.
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        probe_calls = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            return subprocess.CompletedProcess(args=args[0], returncode=0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", fake_run)

        first = pr._systemd_run_user_scope_available()
        second = pr._systemd_run_user_scope_available()
        assert first is True
        assert second is True
        assert len(probe_calls) == 1, "probe must run only once (cached)"
        # The probe must not carry OOMPolicy= either: that is the argv systemd
        # rejected on scope units and cached as "unavailable" (#102486).
        probe_argv = probe_calls[0][0]
        assert not any(
            value.startswith("OOMPolicy=") for value in probe_argv if isinstance(value, str)
        ), probe_argv

    @pytest.mark.linux_only
    def test_systemd_probe_derives_owned_user_bus_env_for_system_gateway(
        self, registry, monkeypatch, request
    ):
        """A system service running as an unprivileged user has no login env,
        but may still have a valid lingering user manager and D-Bus socket."""
        import socket
        import tempfile

        import tools.process_registry as pr

        # Short path: AF_UNIX socket paths are capped at ~104 bytes, longer than most tmp_path values.
        runtime_dir = pr.Path(tempfile.mkdtemp(prefix="hbus-", dir="/tmp"))
        runtime_dir.chmod(0o700)
        bus_path = runtime_dir / "bus"
        bus_socket = socket.socket(socket.AF_UNIX)
        bus_socket.bind(str(bus_path))

        def _cleanup():
            bus_socket.close()
            bus_path.unlink(missing_ok=True)
            runtime_dir.rmdir()

        request.addfinalizer(_cleanup)

        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_default_user_runtime_dir", lambda: runtime_dir)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        derived = pr.systemd_user_bus_env(
            {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/tmp/untrusted-bus"}
        )
        assert derived["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={bus_path}"
        probe_kwargs = []

        def fake_run(*args, **kwargs):
            probe_kwargs.append(kwargs)
            return subprocess.CompletedProcess(args=args[0], returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        assert pr._systemd_run_user_scope_available() is True
        env = probe_kwargs[0]["env"]
        assert env["XDG_RUNTIME_DIR"] == str(runtime_dir)
        assert env["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={bus_path}"
        assert "XDG_RUNTIME_DIR" not in os.environ
        assert "DBUS_SESSION_BUS_ADDRESS" not in os.environ

    @pytest.mark.linux_only
    def test_probe_succeeds_without_bin_true(self, monkeypatch):
        """An absent ``/bin/true`` must not make a usable scope fail its probe."""
        import tools.process_registry as pr

        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_PROBED_AT", 0.0)
        real_run = subprocess.run
        executed = []

        def systemd_run_on_nixos_shaped_root(argv, **kwargs):
            # Simulate NixOS's missing executable, but run the selected replacement.
            payload = argv[argv.index("--") + 1 :]
            if payload[0] == "/bin/true":
                return subprocess.CompletedProcess(payload, 127, stderr=b"No such file or directory")
            executed.append(payload)
            return real_run(payload, **kwargs)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", systemd_run_on_nixos_shaped_root)

        assert pr._systemd_run_user_scope_available() is True
        assert len(executed) == 1, "payload must really run (exit 0) on the host, not just be spelled right"

    @pytest.mark.linux_only
    def test_systemd_scope_first_probe_is_serialized(self, monkeypatch):
        """Concurrent first-use callers must wait for one definitive probe.

        A temporary cached ``False`` would let a racing worker spawn inside the
        gateway cgroup, defeating the OOM isolation guarantee.
        """
        import tools.process_registry as pr

        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        probe_started = threading.Event()
        release_probe = threading.Event()
        probe_calls = []
        results = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            probe_started.set()
            assert release_probe.wait(timeout=2)
            return subprocess.CompletedProcess(args=args[0], returncode=0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", fake_run)

        first = threading.Thread(
            target=lambda: results.append(pr._systemd_run_user_scope_available())
        )
        second = threading.Thread(
            target=lambda: results.append(pr._systemd_run_user_scope_available())
        )
        first.start()
        assert probe_started.wait(timeout=2)
        second.start()

        # The racing caller must be blocked behind the probe, not observe a
        # temporary False cache value.
        second.join(timeout=0.05)
        assert second.is_alive()

        release_probe.set()
        first.join(timeout=2)
        second.join(timeout=2)

        assert not first.is_alive()
        assert not second.is_alive()
        assert results == [True, True]
        assert len(probe_calls) == 1

    @pytest.mark.linux_only
    def test_failed_systemd_probe_retries_after_cache_ttl(self, monkeypatch):
        import tools.process_registry as pr

        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_PROBED_AT", 0.0, raising=False)
        clock = [100.0]
        probe_results = [1, 0]
        probe_calls = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            return subprocess.CompletedProcess(
                args=args[0], returncode=probe_results.pop(0)
            )

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("tools.process_registry.time.monotonic", lambda: clock[0])
        monkeypatch.setattr("subprocess.run", fake_run)

        assert pr._systemd_run_user_scope_available() is False
        assert pr._systemd_run_user_scope_available() is False
        assert len(probe_calls) == 1

        clock[0] += 61
        assert pr._systemd_run_user_scope_available() is True
        assert len(probe_calls) == 2

    def test_stop_systemd_unit_treats_absent_unit_as_clean(self, monkeypatch):
        import tools.process_registry as pr

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=5,
                stderr=b"Unit hermes-worker-gone.scope not loaded.\n",
            ),
        )

        assert pr._stop_systemd_unit("hermes-worker-gone.scope") is True

    def test_darwin_never_takes_scope_path_even_with_systemd_run_on_path(
        self, registry, monkeypatch, _gateway_identity
    ):
        """macOS no-op guarantee (#70716 cross-platform audit).

        With ``_IS_LINUX = False`` (darwin), the spawn path must be
        byte-identical to the legacy path even when a ``systemd-run``
        binary is somehow on PATH and the gateway identity checks pass:
        no probe, no wrapping, no unit recorded.
        """
        import tools.process_registry as pr

        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr(pr, "_IS_LINUX", False)
        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process", lambda: True
        )
        # If any branch consults the probe or builds a scope argv on darwin,
        # fail loudly.
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/systemd-run")
        scope_builds = []
        real_build = pr._build_systemd_scope_argv
        monkeypatch.setattr(
            pr,
            "_build_systemd_scope_argv",
            lambda *a, **k: scope_builds.append(a) or real_build(*a, **k),
        )
        probe_runs = []

        def fake_probe_run(argv, **kwargs):
            probe_runs.append(argv)
            return subprocess.CompletedProcess(args=argv, returncode=0)

        monkeypatch.setattr("subprocess.run", fake_probe_run)

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            session = registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        assert argv == ["/bin/bash", "-lic", "set +m; echo hello"], argv
        assert captured["start_new_session"] is True
        assert session.systemd_unit == ""
        assert scope_builds == [], "darwin must never build a systemd scope argv"
        assert probe_runs == [], "darwin must never run the systemd-run probe"

    def test_probe_returns_false_off_linux(self, monkeypatch):
        """``_systemd_run_user_scope_available`` is False on non-Linux even
        when a ``systemd-run`` binary exists on PATH."""
        import tools.process_registry as pr

        monkeypatch.setattr(pr, "_IS_LINUX", False)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/systemd-run")
        probe_runs = []
        monkeypatch.setattr(
            "subprocess.run",
            lambda argv, **kwargs: probe_runs.append(argv)
            or subprocess.CompletedProcess(args=argv, returncode=0),
        )

        assert pr._systemd_run_user_scope_available() is False
        assert probe_runs == [], "non-Linux must not exec the probe"


class TestNotificationRedaction:
    """Background-process notification delivery (completion_queue) applies the
    same redaction as the explicit process tool — issue #43025 gap.

    The _move_to_finished() and _check_watch_patterns() paths enqueue raw
    output into the completion_queue.  After the fix, _redact_process_result()
    is called before enqueueing so secrets are masked in the [IMPORTANT: ...]
    messages delivered to the LLM.
    """

    def test_completion_notification_redacts_secret(self, monkeypatch):
        """_move_to_finished completion notification redacts API keys."""
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", True)
        from tools import process_registry as pr

        reg = ProcessRegistry()
        sess = _make_session(sid="proc_notif1", command="env")
        sess.output_buffer = "OPENAI_API_KEY=sk-proj-secret123\nHOME=/home/u"
        sess.notify_on_complete = True
        sess.exited = True
        sess.exit_code = 0
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)

        reg._move_to_finished(sess)

        # Drain and check the notification
        results = reg.drain_notifications()
        assert len(results) == 1
        _evt, text = results[0]
        assert "sk-proj-secret123" not in text
        assert "REDACTED" in text or "sk-proj" not in text

    def test_watch_match_notification_redacts_secret(self, monkeypatch):
        """_check_watch_patterns watch_match notification redacts secrets."""
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", True)
        from tools import process_registry as pr

        reg = ProcessRegistry()
        sess = _make_session(sid="proc_notif2", command="python server.py")
        sess.output_buffer = "Server started\nAPI_TOKEN=ghp_abc123def456\nListening on :8080"
        sess.watch_patterns = ["API_TOKEN"]
        sess._watch_disabled = False
        sess._watch_hits = 0
        sess._watch_suppressed = 0
        sess.watcher_platform = None
        sess.watcher_chat_id = None
        sess.watcher_user_id = None
        sess.watcher_user_name = None
        sess.watcher_thread_id = None
        sess.watcher_message_id = None
        sess.exited = False
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)

        reg._check_watch_patterns(sess, "API_TOKEN=ghp_abc123def456\n")

        results = reg.drain_notifications()
        assert len(results) == 1
        _evt, text = results[0]
        assert "ghp_abc123def456" not in text
        assert "ghp_" not in text or "REDACTED" in text


# ── Prefix resolution (Factory Droid-inspired task-ID prefixes) ──────────────


class TestGetByPrefix:
    """ProcessRegistry.get() resolves unique ID prefixes like git short hashes."""

    def test_full_id_still_exact(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_4dae56ca81f6") is s

    def test_unique_prefix_resolves(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_4dae5") is s

    def test_bare_suffix_resolves(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("4dae56") is s

    def test_finished_sessions_also_resolve(self, registry):
        s = _make_session(sid="proc_9bee77aa0011", exited=True, exit_code=0)
        registry._finished[s.id] = s
        assert registry.get("proc_9bee") is s

    def test_ambiguous_prefix_returns_none(self, registry):
        a = _make_session(sid="proc_4dae56ca81f6")
        b = _make_session(sid="proc_4dae99999999")
        registry._running[a.id] = a
        registry._running[b.id] = b
        assert registry.get("proc_4dae") is None

    def test_too_short_prefix_returns_none(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_4da") is None
        assert registry.get("4da") is None
        assert registry.get("proc_") is None
        assert registry.get("") is None

    def test_exact_id_wins_over_prefix_scan(self, registry):
        # A session whose FULL id happens to be a prefix of another's must
        # resolve to itself, never trigger the ambiguity path.
        short = _make_session(sid="proc_4dae")
        long = _make_session(sid="proc_4dae56ca81f6")
        registry._running[short.id] = short
        registry._running[long.id] = long
        assert registry.get("proc_4dae") is short

    def test_no_match_returns_none(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_ffff") is None

    def test_poll_accepts_prefix(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6", output="hello world")
        registry._running[s.id] = s
        result = registry.poll("4dae56ca")
        assert result["session_id"] == "proc_4dae56ca81f6"
        assert result["status"] == "running"


# ---------------------------------------------------------------------------
# Config-level model_not_found notice in delegation batch reports (#97654)
# ---------------------------------------------------------------------------


def _make_delegation_batch_evt(results):
    """A batch async-delegation event carrying a per-task ``results`` list."""
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_97654",
        "is_batch": True,
        "results": results,
        "goals": [r.get("goal") or "" for r in results],
        "session_key": "agent:main:cli:dm:local",
        "status": "completed",
        "model": "upstage/solar-pro-4",
    }


def _patch_delegation_config(
    monkeypatch, model="upstage/solar-pro-4", provider="openrouter", **over
):
    import tools.process_registry_notifications as _prn

    cfg = {"model": model, "provider": provider}
    cfg.update(over)
    monkeypatch.setattr(_prn, "_delegation_config", lambda: cfg)
    return cfg


def _format_async(evt) -> str:
    from tools.process_registry_notifications import format_process_notification

    text = format_process_notification(evt)
    assert text is not None, "format_process_notification returned None"
    return text


def test_model_not_found_notice_single_failure_once(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "exit_reason": "error",
            "goal": "Create bridge module",
            "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
            "summary": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        }
    ])
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert text is not None
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "upstage/solar-pro-4" in text
    assert "openrouter" in text
    assert "No fallback chain is configured" in text


def test_model_not_found_notice_mixed_batch_named_model(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "exit_reason": "error",
            "goal": "A",
            "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
            "summary": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        },
        {
            "task_index": 1,
            "status": "completed",
            "goal": "B",
            "summary": "ok",
            "api_calls": 3,
        },
    ])
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "upstage/solar-pro-4" in text


def test_model_not_found_notice_absent_for_non_model_errors(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "goal": "A",
            "error": "HTTP 429: rate limit exceeded",
        },
        {
            "task_index": 1,
            "status": "failed",
            "goal": "B",
            "error": "Connection timed out",
        },
    ])
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert "SUBAGENT MODEL REJECTED" not in text


def test_model_not_found_notice_absent_when_configured_model_not_named(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "goal": "A",
            "error": "HTTP 400: gpt-99 is not a valid model ID",
        }
    ])
    # Configured model is upstage/solar-pro-4; the rejection names gpt-99.
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert "SUBAGENT MODEL REJECTED" not in text


def test_model_not_found_notice_single_dispatch(monkeypatch):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_single",
        "session_key": "agent:main:cli:dm:local",
        "goal": "task A",
        "model": "upstage/solar-pro-4",
        "status": "failed",
        "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        "summary": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
    }
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "upstage/solar-pro-4" in text


def test_model_not_found_notice_absent_when_fallback_chain_configured(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "goal": "A",
            "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        }
    ])
    _patch_delegation_config(
        monkeypatch,
        fallback_providers=[{"provider": "openrouter", "model": "upstage/solar-pro4"}],
    )
    text = _format_async(evt)
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "No fallback chain is configured" not in text
