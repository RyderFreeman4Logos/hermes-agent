"""Verify scripts/run_tests_parallel.py kills test-spawned grandchildren.

Setup
-----
A test in this file spawns a long-lived Python grandchild that writes
its PID + a nonce to a tempfile, then exits without cleaning up.
With the old ``subprocess.run`` runner, that grandchild would orphan
and outlive the test (and the whole runner). On Linux the runner now puts
that subprocess in a transient user-systemd service with a task budget and
``KillMode=control-group``; other POSIX platforms use the captured process
group. The verifier runs the runner over the leaker file in a subprocess,
then waits for the grandchild PID to vanish from the kernel's process table.

POSIX-only: Windows has its own grandchild lifecycle (no shared session,
``taskkill /F /T`` semantics). Marked accordingly.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest


@pytest.fixture(autouse=True)
def isolated_probe_environment(monkeypatch):
    # Probe files exercise pytest/runner mechanics, not installed third-party
    # plugins. Autoloading the developer environment changes their startup cost.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def _load_runner_module() -> ModuleType:
    """Load the runner for a direct lifecycle seam test."""
    runner = Path(__file__).resolve().parents[2] / "scripts" / "run_tests_parallel.py"
    spec = importlib.util.spec_from_file_location("runner_lifecycle_test", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module




def _probe_root(tmp_path):
    root = tmp_path / "runner-root"
    scripts = root / "scripts"
    (scripts / "ci").mkdir(parents=True, exist_ok=True)
    real = Path(__file__).resolve().parents[2] / "scripts"
    shutil.copy2(real / "run_tests_parallel.py", scripts)
    shutil.copy2(real / "test_temp_root.py", scripts)
    shutil.copy2(real.parent / "hermes_constants.py", root)
    # The runner shares the platforms() spec resolver with the CI lane selector.
    shutil.copy2(real / "ci" / "list_os_marked_tests.py", scripts / "ci")
    return root


def _pid_alive(pid: int) -> bool:
    """POSIX: send signal 0 to probe whether ``pid`` is still alive.

    ``os.kill(pid, 0)`` raises ``ProcessLookupError`` if the process is
    gone, ``PermissionError`` if it exists but we can't signal it
    (someone else's pid). We treat PermissionError as "alive" because
    the process exists and that's all we need to know.
    """
    if sys.platform == "win32":  # pragma: no cover — POSIX-only test
        # On Windows we'd use OpenProcess + GetExitCodeProcess; this
        # test is skipped on Windows so the path is unreachable.
        raise RuntimeError("_pid_alive POSIX-only")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _ProcessIdentity(NamedTuple):
    pid: int
    start_time: int
    pidfd: int


def _proc_start_time(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return int(stat[stat.rfind(")") + 2:].split()[19])
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None


def _open_process_identity(pid: int, expected_start: int | None) -> _ProcessIdentity:
    if expected_start is None or _proc_start_time(pid) != expected_start:
        raise RuntimeError(f"fixture process {pid} no longer has its recorded identity")
    pidfd = os.pidfd_open(pid)
    if _proc_start_time(pid) != expected_start:
        os.close(pidfd)
        raise RuntimeError(f"fixture process {pid} changed identity while opening pidfd")
    return _ProcessIdentity(pid, expected_start, pidfd)


def _process_identity_alive(identity: _ProcessIdentity) -> bool:
    poller = select.poll()
    poller.register(identity.pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
    return not poller.poll(0)


def _signal_process_identity(identity: _ProcessIdentity, sig: int) -> bool:
    if not _process_identity_alive(identity):
        return False
    try:
        signal.pidfd_send_signal(identity.pidfd, sig)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd identity")
def test_stale_process_identity_is_not_signalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exited fixture identity cannot target a reused numeric PID."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
    )
    identity = _open_process_identity(proc.pid, _proc_start_time(proc.pid))
    assert proc.stdin is not None
    proc.stdin.close()
    proc.wait(timeout=10)
    calls = []
    monkeypatch.setattr(
        signal,
        "pidfd_send_signal",
        lambda *args: calls.append(args),
    )
    try:
        assert not _signal_process_identity(identity, signal.SIGKILL)
        assert calls == []
    finally:
        os.close(identity.pidfd)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux systemd completion")
def test_linux_normal_completion_never_signals_stale_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A reaped systemd-run client cannot authorize numeric-PGID cleanup."""
    runner = _load_runner_module()
    killpg_calls: list[tuple[int, int]] = []

    class ReapedClient:
        pid = 4242
        returncode = 0
        kill_calls = 0

        def communicate(self, timeout: float) -> tuple[str, None]:
            return "", None

        def kill(self) -> None:
            self.kill_calls += 1

    client = ReapedClient()
    monkeypatch.setattr(runner, "_spawn_test_process", lambda *args: (client, None))
    monkeypatch.setattr(runner, "_read_linux_completion", lambda *args: (0, True, ""))
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: 4242)
    monkeypatch.setattr(
        runner.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )

    result = runner._run_one_file_once(tmp_path / "test_probe.py", [], tmp_path, 1)

    assert result.returncode == 0
    assert killpg_calls == []
    assert client.kill_calls == 0


@pytest.mark.parametrize("prefill_pipe", [False, True])
def test_environment_pipe_backpressure_is_bounded_and_private(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prefill_pipe: bool,
) -> None:
    """A retained nonreading supervisor pipe cannot block or expose its environment."""
    runner = _load_runner_module()
    secret = "watch-pipe-backpressure-sentinel-" + "x" * (1024 * 1024)
    writes: list[int] = []
    commands: list[list[str]] = []
    launcher_envs: list[dict[str, str]] = []
    original_pipe = runner.os.pipe
    original_write = runner.os.write

    class NonreadingClient:
        def __init__(self, stdin_fd: int) -> None:
            self.read_fd = os.dup(stdin_fd)
            self.kill_calls = 0
            self.communicate_timeouts: list[float] = []

        def kill(self) -> None:
            self.kill_calls += 1

        def communicate(self, timeout: float) -> tuple[str, None]:
            self.communicate_timeouts.append(timeout)
            return "", None

    client: NonreadingClient | None = None

    def pipe() -> tuple[int, int]:
        read_fd, write_fd = original_pipe()
        if prefill_pipe:
            os.set_blocking(write_fd, False)
            while True:
                try:
                    original_write(write_fd, b"x" * 65536)
                except BlockingIOError:
                    return read_fd, write_fd
        return read_fd, write_fd

    def write(fd: int, payload: bytes) -> int:
        count = original_write(fd, payload)
        writes.append(count)
        return count

    def popen(*args: object, **kwargs: object) -> NonreadingClient:
        nonlocal client
        command = args[0]
        stdin_fd = kwargs["stdin"]
        launcher_env = kwargs["env"]
        assert isinstance(command, list)
        assert isinstance(stdin_fd, int)
        assert isinstance(launcher_env, dict)
        commands.append(command)
        launcher_envs.append(launcher_env)
        client = NonreadingClient(stdin_fd)
        return client

    monkeypatch.setenv("WATCH_PIPE_BACKPRESSURE_SECRET", secret)
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/env")
    monkeypatch.setattr(runner.os, "pipe", pipe)
    monkeypatch.setattr(runner.os, "write", write)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)

    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="environment pipe write timed out") as exc_info:
            runner._run_one_file_once(tmp_path / "test_probe.py", [], tmp_path, 0.1)
    finally:
        if client is not None:
            os.close(client.read_fd)

    assert time.monotonic() - started < 2
    assert client is not None
    assert client.kill_calls == 1
    assert client.communicate_timeouts
    assert secret not in str(exc_info.value)
    assert len(commands) == len(launcher_envs) == 1
    assert secret not in " ".join(commands[0])
    assert secret not in launcher_envs[0].values()
    if prefill_pipe:
        assert writes == []
    else:
        assert writes and writes[0] < len(secret)


def test_environment_pipe_write_failure_removes_secret_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A failed supervisor-pipe write leaves neither root nor live client."""
    runner = _load_runner_module()
    root = tmp_path / "attempt-root"

    class SpawnedClient:
        def __init__(self) -> None:
            self.kill_calls = 0
            self.communicate_calls = 0

        def kill(self) -> None:
            self.kill_calls += 1

        def communicate(self, timeout: float) -> tuple[str, None]:
            self.communicate_calls += 1
            return "", None

    client = SpawnedClient()

    def make_root(*args: object, **kwargs: object) -> str:
        root.mkdir()
        return str(root)

    monkeypatch.setattr(runner.tempfile, "mkdtemp", make_root)
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/env")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: client)
    monkeypatch.setattr(runner.os, "write", lambda *args: (_ for _ in ()).throw(OSError("pipe failed")))

    with pytest.raises(OSError, match="pipe failed"):
        runner._run_one_file_once(tmp_path / "test_probe.py", [], tmp_path, 1)

    assert client.kill_calls == 1
    assert client.communicate_calls == 1
    assert not root.exists()


def test_runner_sigkill_before_spawn_leaves_no_secret_transport(tmp_path: Path) -> None:
    """A killed runner cannot leave a pre-spawn environment file behind."""
    repo_root = Path(__file__).resolve().parents[2]
    runner_path = repo_root / "scripts" / "run_tests_parallel.py"
    root = tmp_path / "attempts"
    root.mkdir()
    ready = tmp_path / "pre-spawn-ready"
    secret = "pre-spawn-secret-sentinel"
    child_code = textwrap.dedent(
        f"""
        import importlib.util
        import sys
        import time
        from pathlib import Path

        spec = importlib.util.spec_from_file_location("runner_probe", {str(runner_path)!r})
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        def stop_before_spawn(*args):
            Path({str(ready)!r}).write_text("ready", encoding="utf-8")
            time.sleep(600)

        runner._spawn_test_process = stop_before_spawn
        runner._run_one_file_once(Path("test_probe.py"), [], Path({str(repo_root)!r}), 1)
        """
    )
    env = os.environ.copy()
    env["TMPDIR"] = str(root)
    env["PARENT_LOSS_SECRET"] = secret
    proc = subprocess.Popen([sys.executable, "-c", child_code], env=env)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "runner did not reach the pre-spawn boundary"
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        transport_files = list(root.rglob("environment.json"))
        retained = b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
        assert transport_files == []
        assert secret.encode() not in retained
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        shutil.rmtree(root, ignore_errors=True)


def test_supervisor_exits_after_watch_parent_loss(tmp_path: Path) -> None:
    """The scope-side supervisor exits without exposing its pipe environment."""
    repo_root = Path(__file__).resolve().parents[2]
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    secret = "parent-loss-secret-sentinel"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(runner),
            "--_hermes-supervise",
            str(repo_root),
            str(tmp_path / "completion.json"),
            sys.executable,
            "-c",
            "import time; time.sleep(0.2)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdin is not None
    proc.stdin.write(json.dumps({"PARENT_LOSS_SECRET": secret}) + "\n")
    proc.stdin.close()
    assert proc.wait(timeout=10) == 137
    assert proc.stdout is not None
    assert secret not in proc.stdout.read()


def test_linux_descendant_reap_has_an_absolute_deadline(tmp_path: Path) -> None:
    """An unreapable adopted child cannot block the supervisor forever."""
    repo_root = Path(__file__).resolve().parents[2]
    runner_path = repo_root / "scripts" / "run_tests_parallel.py"
    child_code = textwrap.dedent(
        f"""
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("runner_probe", {str(runner_path)!r})
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        runner.os.waitpid = lambda *args: (0, 0)
        runner.os.kill = lambda *args: None
        runner.Path.read_text = lambda *args, **kwargs: "123"
        assert runner._linux_terminate_and_reap_descendants() == 1
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", child_code])
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        pytest.fail("descendant cleanup blocked in waitpid")
    assert proc.returncode == 0


def test_linux_cleanup_consumes_watch_close_before_reap_deadline(tmp_path: Path) -> None:
    """Watch-parent loss interrupts a cleanup that cannot reap immediately."""
    repo_root = Path(__file__).resolve().parents[2]
    runner_path = repo_root / "scripts" / "run_tests_parallel.py"
    ready = tmp_path / "cleanup-ready"
    child_code = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys
        from pathlib import Path

        spec = importlib.util.spec_from_file_location("runner_probe", {str(runner_path)!r})
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        class FinishedChild:
            returncode = 0
            def poll(self):
                return 0

        runner._linux_enable_subreaper = lambda: True
        runner._read_linux_resource_events = lambda: {{
            "pids.events": 0, "memory.events": 0, "memory.events.oom": 0,
            "memory.events.oom_kill": 0,
        }}
        runner.subprocess.Popen = lambda *args, **kwargs: FinishedChild()
        runner.os.waitpid = lambda *args: (0, 0)
        runner.os.kill = lambda *args: None
        def children(*args, **kwargs):
            Path({str(ready)!r}).write_text("ready", encoding="utf-8")
            return "123"
        runner.Path.read_text = children
        raise SystemExit(runner._linux_supervise({str(repo_root)!r}, {str(tmp_path / 'completion.json')!r}, []))
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", child_code], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    proc.stdin.write(json.dumps({"SAFE": "value"}).encode() + b"\n")
    proc.stdin.flush()
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "supervisor did not enter descendant cleanup"
        proc.stdin.close()
        assert proc.wait(timeout=1) == 137
    except subprocess.TimeoutExpired:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        pytest.fail("watch close did not interrupt descendant cleanup")
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def test_progress_output_tolerates_legacy_stdout_encoding(tmp_path: Path) -> None:
    """Progress glyphs must not crash the runner on non-UTF-8 consoles."""
    repo_root = _probe_root(tmp_path)
    runner = repo_root / "scripts" / "run_tests_parallel.py"

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe = probe_dir / "test_probe_smoke.py"
    probe.write_text("def test_smoke():\n    assert True\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252:strict"

    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--paths",
            str(probe_dir),
            "-j",
            "1",
            "--file-timeout",
            "30",
        ],
        cwd=probe_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    assert "UnicodeEncodeError" not in proc.stdout
    assert "1 tests passed" in proc.stdout


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
def test_grandchild_leak_is_killed_by_runner(tmp_path: Path) -> None:
    """Run the parallel runner over a probe file and verify cleanup.

    1. Materialize a probe file that spawns a long-lived grandchild and
       writes its PID to disk before exiting.
    2. Invoke ``scripts/run_tests_parallel.py`` against the probe file.
    3. Wait for the grandchild PID to vanish (poll for ~5s).
    4. Assert the runner exited cleanly AND the grandchild is dead.
    """
    repo_root = _probe_root(tmp_path)
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    assert runner.exists(), f"runner missing at {runner}"

    # Probe lives in a temp dir, NOT under tests/, so the regular suite
    # never picks it up — only our explicit invocation does.
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe = probe_dir / "test_probe_leaker.py"
    nonce = f"{os.getpid()}-{int(time.time() * 1000)}"
    handoff = tmp_path / f"grandchild-{nonce}.json"

    probe_src = textwrap.dedent(f"""
        import json, os, subprocess, sys, time
        from pathlib import Path

        HANDOFF = Path({str(handoff)!r})

        def test_spawns_grandchild_and_walks_away():
            # Long-lived grandchild: detached, ignores SIGTERM (we want
            # SIGKILL or process-group kill to be the only thing that
            # works, simulating a misbehaving server).
            child = subprocess.Popen(
                [
                    sys.executable, "-c",
                    "import os, signal, sys, time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "sys.stdout.write(f'gc-pgid={{os.getpgid(0)}} gc-pid={{os.getpid()}}\\\\n'); "
                    "sys.stdout.flush(); "
                    "time.sleep(600)",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # IMPORTANT: do NOT pass start_new_session here. We want
                # the grandchild to inherit the pytest subprocess's
                # process group, so when the runner kills the group the
                # grandchild dies too.
            )
            # Read the first line so we can record gc's pgid in the
            # handoff, then walk away — don't close the pipe (would
            # signal EOF and let the child see SIGPIPE on next write).
            first_line = child.stdout.readline().decode().strip()
            HANDOFF.write_text(json.dumps({{
                "pid": child.pid,
                "diag": first_line,
                "test_pid": os.getpid(),
                "test_pgid": os.getpgid(0),
            }}))
            assert child.pid > 0
    """).strip()
    probe.write_text(probe_src + "\n")

    # Run the parallel runner against just the probe file. The runner
    # discovers under ``tests/`` by default, so we override via --paths.
    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--paths",
            str(probe_dir),
            "-j",
            "1",
            # Tight per-file timeout: the probe finishes in <1s, no
            # need for 10min.
            "--file-timeout",
            "30",
        ],
        cwd=probe_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # The runner declares its stdio UTF-8 (see _make_stdio_glyph_safe);
        # decode the same way so ✓-glyph assertions hold on Windows, where
        # text=True alone would decode with the locale codec (cp1252).
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert handoff.exists(), (
        f"probe never wrote handoff file; runner output:\n{proc.stdout}"
    )
    handoff_data = json.loads(handoff.read_text())
    grandchild_pid = handoff_data["pid"]
    diag = handoff_data.get("diag", "(no diag)")
    test_pid = handoff_data.get("test_pid")
    test_pgid = handoff_data.get("test_pgid")
    handoff.unlink()

    # The runner must have exited cleanly (probe test passes).
    assert proc.returncode == 0, (
        f"runner exited {proc.returncode}; output:\n{proc.stdout}"
    )

    # The grandchild must be gone. Poll for a bit because process-group
    # SIGKILL + reaping isn't synchronous; on a loaded box it can take
    # a beat.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _pid_alive(grandchild_pid):
            break
        time.sleep(0.05)
    else:
        # Test cleanup: kill the leaked grandchild ourselves so a
        # FAILED assertion doesn't leave a sleep(600) running.
        try:
            os.kill(grandchild_pid, 9)
        except ProcessLookupError:
            pass
        pytest.fail(
            f"grandchild PID {grandchild_pid} survived runner exit; "
            f"diag={diag!r} test_pid={test_pid} test_pgid={test_pgid}; "
            f"runner output:\n{proc.stdout}"
        )



@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass
def test_detached_grandchild_is_killed_by_runner(tmp_path: Path) -> None:
    """A child in its OWN session (an auto-started ``gateway run``) escapes the process-group
    kill; the runner still finds it by the attempt's temp root in its environment."""
    repo_root = _probe_root(tmp_path)
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    handoff = tmp_path / "detached.json"
    (probe_dir / "test_probe_detached.py").write_text(textwrap.dedent(f"""
        import json, os, subprocess, sys
        from pathlib import Path

        def test_spawns_detached_child_and_walks_away():
            child = subprocess.Popen(
                [sys.executable, "-c",
                 "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"],
                start_new_session=True,
            )
            Path({str(handoff)!r}).write_text(json.dumps({{"pid": child.pid}}), encoding="utf-8")
    """).lstrip(), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_tests_parallel.py"),
         "--paths", str(probe_dir), "-j", "1", "--file-timeout", "30"],
        cwd=probe_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", timeout=60, check=False,
    )
    pid = json.loads(handoff.read_text(encoding="utf-8-sig"))["pid"]
    deadline = time.monotonic() + 5.0
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _pid_alive(pid):
        os.kill(pid, 9)
        pytest.fail(f"detached child {pid} survived the runner; output:\n{proc.stdout}")
    assert proc.returncode == 0, proc.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only probe")
@pytest.mark.live_system_guard_bypass
def test_runner_sigkill_contains_descendant_tree(tmp_path: Path) -> None:
    """A SIGKILL'd runner must not orphan a bounded descendant tree."""
    repo_root = Path(__file__).resolve().parents[2]
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    probe_dir = repo_root / f".runner-sigkill-probe-{os.getpid()}"
    probe_dir.mkdir()
    handoff = tmp_path / "tree.json"
    release = tmp_path / "release"
    output_path = tmp_path / "runner.out"
    probe = probe_dir / "test_probe_tree.py"
    leaf_code = textwrap.dedent(
        f"""
        import json, os, subprocess, sys, time
        from pathlib import Path

        def start_time(pid):
            stat = Path(f"/proc/{{pid}}/stat").read_text()
            return int(stat[stat.rfind(")") + 2:].split()[19])

        leaf = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        Path({str(handoff)!r}).write_text(json.dumps({{
            "tree": os.getpid(),
            "tree_start": start_time(os.getpid()),
            "leaf": leaf.pid,
            "leaf_start": start_time(leaf.pid),
        }}))
        time.sleep(600)
        """
    )
    probe.write_text(
        textwrap.dedent(
            f"""
            import os, subprocess, sys, time
            from pathlib import Path

            HANDOFF = Path({str(handoff)!r})
            RELEASE = Path({str(release)!r})

            def test_tree_waits_for_runner():
                subprocess.Popen(
                    [sys.executable, "-c", {leaf_code!r}],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                deadline = time.monotonic() + 10
                while not HANDOFF.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert HANDOFF.exists(), "descendant tree never started"
                while not RELEASE.exists():
                    time.sleep(0.01)
            """
        ),
        encoding="utf-8",
    )

    with output_path.open("wb") as output:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(runner),
                "--paths",
                str(probe),
                "--file-retries",
                "0",
                "-j",
                "1",
                "--file-timeout",
                "30",
                "-q",
                f"--rootdir={repo_root}",
            ],
            cwd=repo_root,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        identities: list[_ProcessIdentity] = []
        try:
            deadline = time.monotonic() + 10
            while not handoff.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert handoff.exists(), "runner did not start the synthetic tree"
            data = json.loads(handoff.read_text(encoding="utf-8"))
            identities.append(_open_process_identity(data["tree"], data["tree_start"]))
            identities.append(_open_process_identity(data["leaf"], data["leaf_start"]))
            os.kill(proc.pid, 9)
            proc.wait(timeout=10)
            assert proc.returncode in (-9, 137)
            assert "=== Summary:" not in output_path.read_text(encoding="utf-8")
            deadline = time.monotonic() + 5
            while (
                any(_process_identity_alive(identity) for identity in identities)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert not any(
                _process_identity_alive(identity) for identity in identities
            ), f"descendants survived: {[identity.pid for identity in identities]}"
        finally:
            if proc.poll() is None:
                os.kill(proc.pid, 9)
                proc.wait(timeout=10)
            try:
                for identity in identities:
                    _signal_process_identity(identity, signal.SIGKILL)
                deadline = time.monotonic() + 5
                while (
                    any(_process_identity_alive(identity) for identity in identities)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
            finally:
                for identity in identities:
                    os.close(identity.pidfd)
            probe.unlink(missing_ok=True)
            shutil.rmtree(probe_dir, ignore_errors=True)


# ── Bare pytest-flag passthrough ─────────────────────────────────────────────
#
# The runner routes any token starting with ``-`` that isn't one of its own
# options (``-j``/``--jobs``, ``--paths``, ``--slice``, ``--file-timeout``,
# ``--generate-slices``, ``--files``, ``--include-integration``,
# ``--files-from``) straight
# through to each per-file pytest invocation — no ``--`` separator required.
# Before this, a bare ``-q`` errored out with "unrecognized arguments",
# forcing a retry on every run. These tests are behavior contracts, not
# snapshots: they assert that bare flags reach pytest and that value-taking
# flags (``-k expr``) keep their value instead of having it stolen by the
# positional-path discovery.


def _make_probe_dir(tmp_path: Path) -> Path:
    """Two trivial passing tests, one named test_alpha, one test_beta."""
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "test_flagprobe.py").write_text(
        "def test_alpha():\n    assert True\n\n"
        "def test_beta():\n    assert True\n"
    )
    return probe_dir


def _run_runner(probe_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    repo_root = _probe_root(probe_dir.parent)
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    return subprocess.run(
        [sys.executable, str(runner), "--paths", str(probe_dir),
         "-j", "1", "--file-timeout", "30", *extra],
        cwd=probe_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # The runner declares its stdio UTF-8 (see _make_stdio_glyph_safe);
        # decode the same way so ✓-glyph assertions hold on Windows, where
        # text=True alone would decode with the locale codec (cp1252).
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )


def test_out_of_tree_probe_does_not_collect_host_tmp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested pytest must not scandir host /tmp for an out-of-tree probe."""
    host_tmp = Path(tempfile.gettempdir())
    probe_root = Path(tempfile.mkdtemp(prefix="hermes-nested-probe-", dir=str(host_tmp)))
    try:
        probe_dir = probe_root / "probe"
        probe_dir.mkdir()
        (probe_dir / "test_nested_isolation.py").write_text(
            "def test_smoke():\n    assert True\n", encoding="utf-8"
        )
        proc = _run_runner(probe_dir)
        assert proc.returncode == 0, proc.stdout
        assert "PermissionError" not in proc.stdout
        assert "1✓" in proc.stdout or "1 tests passed" in proc.stdout, proc.stdout
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)


def test_out_of_tree_symlink_probe_file_matches_collection_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pytest file and --rootdir/--confcutdir must share one path form.

    A lexical symlink probe (TMPDIR=$HOME/tmp on this host) is not under
    file.resolve().parent, so mixed argv still walks an ancestor conftest.
    """
    physical = tmp_path / "physical"
    physical.mkdir()
    (physical / "conftest.py").write_text(
        "def pytest_configure(config):\n"
        "    raise RuntimeError('WALKED_ANCESTOR_CONFTEST')\n",
        encoding="utf-8",
    )
    probe = physical / "probe"
    probe.mkdir()
    test_file = probe / "test_smoke.py"
    test_file.write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    link_root = tmp_path / "link"
    link_root.symlink_to(physical)
    lexical_file = link_root / "probe" / "test_smoke.py"
    assert str(lexical_file) != str(lexical_file.resolve())

    runner = _load_runner_module()
    repo_root = Path(__file__).resolve().parents[2]
    captured: dict[str, list[str]] = {}

    def fake_spawn(cmd, _repo_root, _env, completion_path=None, deadline=None):
        captured["cmd"] = list(cmd)
        raise RuntimeError("CAPTURED_SPAWN")

    monkeypatch.setattr(runner, "_spawn_test_process", fake_spawn)
    with pytest.raises(RuntimeError, match="CAPTURED_SPAWN"):
        runner._run_one_file_once(lexical_file, ["-q"], repo_root, 30.0)

    cmd = captured["cmd"]
    assert cmd[1:3] == ["-m", "pytest"]
    file_arg = Path(cmd[3])
    rootdir = next(a.split("=", 1)[1] for a in cmd if a.startswith("--rootdir="))
    confcutdir = next(a.split("=", 1)[1] for a in cmd if a.startswith("--confcutdir="))
    assert "--noconftest" not in cmd
    assert rootdir == confcutdir
    file_arg.relative_to(rootdir)

    proc = _run_runner(lexical_file.parent)
    assert proc.returncode == 0, proc.stdout
    assert "WALKED_ANCESTOR_CONFTEST" not in proc.stdout
    assert "1✓" in proc.stdout or "1 tests passed" in proc.stdout, proc.stdout


@pytest.mark.parametrize("help_flag", ["-h", "--help"])
def test_help_prints_usage_without_discovering_or_running_tests(
    tmp_path: Path, help_flag: str
) -> None:
    """Runner help stays in argparse instead of becoming a pytest sweep."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"

    proc = subprocess.run(
        [sys.executable, str(runner), "--paths", str(tmp_path / "no-tests"), help_flag],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    assert "usage:" in proc.stdout
    assert "Discovered" not in proc.stdout


def test_unknown_bare_flag_errors_with_usage_instead_of_sweeping(tmp_path: Path) -> None:
    """A typo'd flag fails once, up front, and never reaches a per-file pytest.

    Bare tokens are checked against pytest's own option set, so real pytest
    forms (attached short value ``-rA``, bare ``-x``) still pass through and
    run, while ``--jbs`` is rejected with this runner's usage before discovery.
    """
    probe_dir = _make_probe_dir(tmp_path)

    proc = _run_runner(probe_dir, "--jbs")
    assert proc.returncode == 2, proc.stdout
    assert "usage:" in proc.stdout and "unrecognized arguments: --jbs" in proc.stdout
    assert "Discovered" not in proc.stdout

    proc = _run_runner(probe_dir, "-rA", "-x")
    assert proc.returncode == 0, proc.stdout
    assert "2✓" in proc.stdout or "2 passed" in proc.stdout, proc.stdout


def test_known_flag_missing_value_errors_with_usage_instead_of_sweeping(
    tmp_path: Path,
) -> None:
    """``--tb`` with no value is a pytest UsageError, not a per-file sweep.

    The flag itself is known, so an unknown-token check alone lets it through;
    pytest's own parser must be allowed to reject it up front.
    """
    probe_dir = _make_probe_dir(tmp_path)

    proc = _run_runner(probe_dir, "--tb")
    assert proc.returncode == 2, proc.stdout
    assert "usage:" in proc.stdout and "--tb: expected one argument" in proc.stdout
    assert "Discovered" not in proc.stdout


def test_file_retry_self_heals_and_prints_both_attempts(tmp_path: Path) -> None:
    """A pass-on-retry is green, loud, and retains the failing traceback."""
    repo_root = _probe_root(tmp_path)
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    marker = tmp_path / "ran-once"
    probe = tmp_path / "test_flaky_probe.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            def test_flaky_once():
                marker = Path({str(marker)!r})
                if not marker.exists():
                    marker.write_text("failed once")
                    assert False, "simulated first-attempt flake"
                assert True
            """
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--files",
            str(probe),
            "--file-retries",
            "1",
            "-j",
            "1",
            "-q",
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    assert "FLAKY file" in proc.stdout
    assert "simulated first-attempt flake" in proc.stdout
    assert "first-attempt output" in proc.stdout
    assert "retry output" in proc.stdout


def _fake_systemd_run(
    tmp_path: Path, tasksmax_event: bool = False, late_descendant_event: bool = False,
) -> Path:
    """Create a non-unit launcher with an optional fake cgroup event reader."""
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    launcher = fake_bin / "systemd-run"
    launcher.write_text(
        textwrap.dedent(
            """
            #!/usr/bin/python3
            import json
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            args = sys.argv[1:]
            cmd = args[args.index("--") + 1:]
            flag = cmd.index("--_hermes-supervise")
            completion = Path(cmd[flag + 2])
            transport = sys.stdin.buffer.readline()
            try:
                private_env = json.loads(transport)
            except (OSError, ValueError):
                private_env = os.environ
            state = Path(private_env["FAKE_SYSTEMD_STATE"])
            attempt = int(state.read_text()) + 1 if state.exists() else 1
            state.write_text(str(attempt))
            mode = private_env["FAKE_SYSTEMD_MODE"]
            if attempt == 1 and mode == "fatal-137":
                print("service killed by resource containment", flush=True)
                raise SystemExit(137)
            if attempt == 1 and mode == "startup":
                print("Failed to start transient service unit", flush=True)
                raise SystemExit(125)
            if attempt == 1 and mode == "missing-status":
                print("1 failed in 0.01s", flush=True)
                raise SystemExit(1)
            if attempt == 1 and mode == "malformed-status":
                completion.write_text("not json")
                raise SystemExit(1)
            if attempt == 1 and mode == "mismatched-status":
                completion.write_text(json.dumps({
                    "kind": "pytest", "returncode": 0,
                    "resource_events": {
                        "pids.events": 0,
                        "memory.events": 0,
                        "memory.events.oom": 0,
                        "memory.events.oom_kill": 0,
                    },
                }))
                raise SystemExit(1)
            if attempt == 1 and mode == "timeout":
                time.sleep(2)
                raise SystemExit(1)
            if attempt == 1 and mode == "typed-exit4":
                completion.write_text(json.dumps({
                    "kind": "pytest",
                    "returncode": 4,
                    "resource_events": {
                        "pids.events": 0,
                        "memory.events": 0,
                        "memory.events.oom": 0,
                        "memory.events.oom_kill": 0,
                    },
                }))
                print("ERROR: file or directory not found", flush=True)
                raise SystemExit(4)
            record = private_env.get("FAKE_SYSTEMD_ARGV_RECORD")
            if record:
                Path(record).write_text(json.dumps(sys.argv))
            child = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            child.stdin.write(transport)
            child.stdin.flush()
            status = child.wait()
            child.stdin.close()
            raise SystemExit(status)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    if tasksmax_event or late_descendant_event:
        site_dir = tmp_path / "fake-site"
        site_dir.mkdir()
        (site_dir / "sitecustomize.py").write_text(
            textwrap.dedent(
                f"""
                    import pathlib
                    import subprocess
                    import sys

                    _read_text = pathlib.Path.read_text
                    _pids_events_reads = 0

                    def _fake_cgroup_read_text(self, *args, **kwargs):
                        global _pids_events_reads
                        if str(self) == "/proc/self/cgroup":
                            return "0::/fake\\n"
                        if str(self) == "/sys/fs/cgroup/fake/pids.events":
                            _pids_events_reads += 1
                            if _pids_events_reads == 2 and {late_descendant_event!r}:
                                subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
                                return "max 0\\n"
                            return f"max {{_pids_events_reads - 1}}\\n"
                        if str(self) == "/sys/fs/cgroup/fake/memory.events":
                            return "max 0\\noom 0\\noom_kill 0\\n"
                        return _read_text(self, *args, **kwargs)

                    pathlib.Path.read_text = _fake_cgroup_read_text
                    """
                ).lstrip(),
            encoding="utf-8",
        )
        env_binary = fake_bin / "env"
        env_binary.write_text(
            "#!/bin/sh\nshift\nexec /usr/bin/env -i PYTHONPATH=" + str(site_dir) + " \"$@\"\n",
            encoding="utf-8",
        )
        env_binary.chmod(0o755)
    return fake_bin


def _run_fake_systemd_retry(
    tmp_path: Path,
    mode: str,
    probe_source: str = "def test_probe():\n    assert True\n",
    *,
    file_timeout: float = 30,
    tasksmax_event: bool = False,
    late_descendant_event: bool = False,
    secret: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], int]:
    repo_root = Path(__file__).resolve().parents[2]
    probe_dir = repo_root / f".runner-retry-probe-{os.getpid()}-{tmp_path.name}"
    probe_dir.mkdir()
    probe = probe_dir / "test_retry_boundary.py"
    probe.write_text(probe_source, encoding="utf-8")
    state = tmp_path / "attempts"
    env = os.environ.copy()
    env["FAKE_SYSTEMD_MODE"] = mode
    env["FAKE_SYSTEMD_STATE"] = str(state)
    if secret is not None:
        env["DIRECT_RUNNER_SECRET"] = secret
        env["FAKE_SYSTEMD_ARGV_RECORD"] = str(tmp_path / "launcher-argv.json")
    env["PATH"] = os.pathsep.join((
        str(_fake_systemd_run(tmp_path, tasksmax_event, late_descendant_event)), "/usr/bin", "/bin",
    ))
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_tests_parallel.py"),
                "--files",
                str(probe),
                "--file-retries",
                "1",
                "--file-timeout",
                str(file_timeout),
                "-j",
                "1",
                "-q",
                f"--rootdir={repo_root}",
            ],
            check=False,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    return proc, int(state.read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", ("fatal-137", "startup", "missing-status"))
def test_non_pytest_attempt_cannot_retry_to_green(tmp_path: Path, mode: str) -> None:
    """A fatal systemd-boundary result is terminal even if retry would pass."""
    proc, attempts = _run_fake_systemd_retry(tmp_path, mode)

    assert proc.returncode == 1, proc.stdout
    assert attempts == 1, proc.stdout
    assert "1 file failed" in proc.stdout
    assert "FLAKY" not in proc.stdout


def test_typed_pytest_failure_can_retry_to_green(tmp_path: Path) -> None:
    """A real pytest assertion failure remains explicitly retry-eligible."""
    marker = tmp_path / "failed-once"
    proc, attempts = _run_fake_systemd_retry(
        tmp_path,
        "pytest",
        textwrap.dedent(
            f"""
            from pathlib import Path

            def test_probe():
                marker = Path({str(marker)!r})
                if not marker.exists():
                    marker.write_text("failed")
                    assert False, "retry-eligible pytest failure"
            """
        ),
    )

    assert proc.returncode == 0, proc.stdout
    assert attempts == 2, proc.stdout
    assert "FLAKY file" in proc.stdout
    assert "retry-eligible pytest failure" in proc.stdout


def test_tasksmax_event_cannot_retry_to_green(tmp_path: Path) -> None:
    """A pids-controller denial is terminal even when its pytest retry passes."""
    marker = tmp_path / "failed-once"
    proc, attempts = _run_fake_systemd_retry(
        tmp_path,
        "pytest",
        textwrap.dedent(
            f"""
            from pathlib import Path

            def test_probe():
                marker = Path({str(marker)!r})
                if not marker.exists():
                    marker.write_text("failed")
                    assert False, "simulated TasksMax EAGAIN"
            """
        ),
        tasksmax_event=True,
    )

    assert proc.returncode == 1, proc.stdout
    assert attempts == 1, proc.stdout
    assert "fatal cgroup resource event" in proc.stdout
    assert "FLAKY" not in proc.stdout


def test_post_sample_descendant_event_cannot_retry_to_green(tmp_path: Path) -> None:
    """A descendant created after the final counter sample is terminal."""
    marker = tmp_path / "failed-once"
    proc, attempts = _run_fake_systemd_retry(
        tmp_path,
        "pytest",
        textwrap.dedent(
            f"""
            from pathlib import Path
            import subprocess
            import sys

            def test_probe():
                marker = Path({str(marker)!r})
                if not marker.exists():
                    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
                    marker.write_text("failed")
                    assert False, "late cgroup event must be terminal"
            """
        ),
        late_descendant_event=True,
    )

    assert proc.returncode == 1, proc.stdout
    assert attempts == 1, proc.stdout
    assert "FLAKY" not in proc.stdout


@pytest.mark.parametrize("mode", ("malformed-status", "mismatched-status", "timeout"))
def test_non_authoritative_attempt_siblings_cannot_retry_to_green(
    tmp_path: Path, mode: str,
) -> None:
    """Malformed, mismatched, and timeout boundaries stay terminal."""
    proc, attempts = _run_fake_systemd_retry(
        tmp_path, mode, file_timeout=0.1 if mode == "timeout" else 30,
    )

    assert proc.returncode == 1, proc.stdout
    assert attempts == 1, proc.stdout
    assert "FLAKY" not in proc.stdout


def test_direct_linux_runner_keeps_secrets_out_of_launcher_argv(tmp_path: Path) -> None:
    """The direct Python entrypoint never places inherited secrets in argv."""
    secret = "direct-runner-secret-sentinel"
    proc, attempts = _run_fake_systemd_retry(tmp_path, "pytest", secret=secret)
    argv = (tmp_path / "launcher-argv.json").read_text(encoding="utf-8")

    assert proc.returncode == 0, proc.stdout
    assert attempts == 1, proc.stdout
    assert secret not in argv
    assert secret not in proc.stdout


def test_payload_env_omits_session_bus_and_wraps_bwrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Launcher keeps bus coords; pytest env and default /run socket do not."""
    runner = _load_runner_module()
    captured: dict[str, object] = {}

    class ImmediateChild:
        def poll(self) -> int:
            return 0

        returncode = 0

    def popen(cmd: object, **kwargs: object) -> ImmediateChild:
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return ImmediateChild()

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "_linux_enable_subreaper", lambda: True)
    monkeypatch.setattr(
        runner,
        "_read_linux_resource_events",
        lambda: {field: 0 for field in runner._LINUX_RESOURCE_EVENT_FIELDS},
    )
    monkeypatch.setattr(runner, "_linux_terminate_and_reap_descendants", lambda *a, **k: 0)
    monkeypatch.setattr(
        runner.shutil,
        "which",
        lambda name, path=None: "/usr/bin/bwrap" if name == "bwrap" else None,
    )

    payload = json.dumps({
        "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": "/run/user/1001",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus",
        "PYTEST_DEBUG_TEMPROOT": str(tmp_path),
    }) + "\n"
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload.encode())
    os.close(write_fd)
    stdin_file = os.fdopen(read_fd)
    monkeypatch.setattr(sys, "stdin", stdin_file)
    completion = tmp_path / "completion.json"
    try:
        rc = runner._linux_supervise(str(tmp_path), str(completion), [sys.executable, "-c", "pass"])
    finally:
        stdin_file.close()
    env = captured["env"]
    cmd = captured["cmd"]
    assert rc == 0
    assert isinstance(env, dict)
    assert "DBUS_SESSION_BUS_ADDRESS" not in env
    assert "XDG_RUNTIME_DIR" not in env
    assert isinstance(cmd, list)
    assert cmd[0] == "/usr/bin/bwrap"
    assert "--tmpfs" in cmd and "/run" in cmd
    assert cmd[-3:] == [sys.executable, "-c", "pass"]


def test_missing_bwrap_fails_closed_without_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Isolation setup failure must not start pytest or write a green receipt."""
    runner = _load_runner_module()
    spawned: list[object] = []

    class ImmediateChild:
        def poll(self) -> int:
            return 0

        returncode = 0

    def popen(*args: object, **kwargs: object) -> ImmediateChild:
        spawned.append(args)
        return ImmediateChild()

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "_linux_enable_subreaper", lambda: True)
    monkeypatch.setattr(runner.shutil, "which", lambda name, path=None: None)
    payload = json.dumps({"PATH": str(tmp_path / "empty")}) + "\n"
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload.encode())
    os.close(write_fd)
    stdin_file = os.fdopen(read_fd)
    monkeypatch.setattr(sys, "stdin", stdin_file)
    completion = tmp_path / "completion.json"
    try:
        rc = runner._linux_supervise(str(tmp_path), str(completion), [sys.executable, "-c", "pass"])
    finally:
        stdin_file.close()
    assert rc == 1
    assert spawned == []
    assert not completion.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux nested-harness policy")
def test_nested_runner_harness_keeps_host_supervision_without_payload_bus_restore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Harness pytest may use the user bus; ordinary probe pytest must not.

    Outer bwrap --tmpfs /run plus stripped bus env makes nested systemd-run
    fail closed (No medium found). Do not restore bus to arbitrary pytest.
    """
    runner = _load_runner_module()
    captured: dict[str, object] = {}
    repo_root = Path(__file__).resolve().parents[2]
    harness = Path(__file__).resolve()

    class ImmediateChild:
        def poll(self) -> int:
            return 0

        returncode = 0

    def popen(cmd: object, **kwargs: object) -> ImmediateChild:
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return ImmediateChild()

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "_linux_enable_subreaper", lambda: True)
    monkeypatch.setattr(
        runner,
        "_read_linux_resource_events",
        lambda: {field: 0 for field in runner._LINUX_RESOURCE_EVENT_FIELDS},
    )
    monkeypatch.setattr(runner, "_linux_terminate_and_reap_descendants", lambda *a, **k: 0)
    monkeypatch.setattr(
        runner.shutil,
        "which",
        lambda name, path=None: "/usr/bin/bwrap" if name == "bwrap" else None,
    )

    payload = json.dumps({
        "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": "/run/user/1001",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus",
        "PYTEST_DEBUG_TEMPROOT": str(tmp_path),
    }) + "\n"
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload.encode())
    os.close(write_fd)
    stdin_file = os.fdopen(read_fd)
    monkeypatch.setattr(sys, "stdin", stdin_file)
    completion = tmp_path / "completion.json"
    cmd = [sys.executable, "-m", "pytest", str(harness), "-q"]
    try:
        rc = runner._linux_supervise(str(repo_root), str(completion), cmd)
    finally:
        stdin_file.close()
    env = captured["env"]
    child_cmd = captured["cmd"]
    assert rc == 0
    assert isinstance(env, dict)
    assert env.get("XDG_RUNTIME_DIR") == "/run/user/1001"
    assert env.get("DBUS_SESSION_BUS_ADDRESS") == "unix:path=/run/user/1001/bus"
    assert isinstance(child_cmd, list)
    assert child_cmd == cmd
    assert "--tmpfs" not in child_cmd


def test_nested_runner_harness_pipe_keeps_launcher_bus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Spawn must hand the harness pytest the bus; systemd-run argv stays clean."""
    if sys.platform != "linux":
        pytest.skip("Linux nested-harness policy")
    runner = _load_runner_module()
    repo_root = Path(__file__).resolve().parents[2]
    harness = Path(__file__).resolve()
    captured_argv: list[str] = []
    captured_launcher: dict[str, str] = {}
    read_fd = -1

    class FakeClient:
        def kill(self) -> None:
            return None

        def communicate(self, timeout: float | None = None) -> tuple[str, None]:
            return "", None

    def popen(*args: object, **kwargs: object) -> FakeClient:
        nonlocal read_fd, captured_argv, captured_launcher
        command = args[0]
        stdin_fd = kwargs["stdin"]
        launcher_env = kwargs["env"]
        assert isinstance(command, list)
        assert isinstance(stdin_fd, int)
        assert isinstance(launcher_env, dict)
        captured_argv = [str(part) for part in command]
        captured_launcher = {str(key): str(value) for key, value in launcher_env.items()}
        read_fd = os.dup(stdin_fd)
        return FakeClient()

    monkeypatch.setattr(runner.shutil, "which", lambda name, path=None: {
        "systemd-run": "/usr/bin/systemd-run",
        "env": "/usr/bin/env",
    }.get(name))
    monkeypatch.setattr(runner.subprocess, "Popen", popen)

    env = {
        "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": "/run/user/1001",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus",
        "DIRECT_RUNNER_SECRET": "harness-pipe-secret",
    }
    cmd = [sys.executable, "-m", "pytest", str(harness), "-q"]
    completion = tmp_path / "completion.json"
    proc, watch_w = runner._spawn_test_process(
        cmd, repo_root, env, completion, time.monotonic() + 2,
    )
    assert proc is not None
    if watch_w is not None:
        os.close(watch_w)
    raw = os.read(read_fd, 65536)
    os.close(read_fd)
    payload = json.loads(raw.decode())
    assert payload["XDG_RUNTIME_DIR"] == "/run/user/1001"
    assert payload["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1001/bus"
    assert payload["DIRECT_RUNNER_SECRET"] == "harness-pipe-secret"
    assert set(captured_launcher) <= {"XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"}
    assert "DIRECT_RUNNER_SECRET" not in captured_launcher
    assert "harness-pipe-secret" not in " ".join(captured_argv)


def test_typed_exit4_for_existing_file_can_retry_to_green(tmp_path: Path) -> None:
    """Preserve the intentional loaded-runner exit-4 retry contract."""
    proc, attempts = _run_fake_systemd_retry(tmp_path, "typed-exit4")

    assert proc.returncode == 0, proc.stdout
    assert attempts == 2, proc.stdout
    assert "FLAKY file" in proc.stdout
    assert "file or directory not found" in proc.stdout


# ---------------------------------------------------------------------------
# Zero-collection is not a pass; node ids are translated, not dropped.
#
# Both behaviors were real foot-guns: a run where NOTHING was collected printed
# "0 tests passed, 0 failed (100% complete)" (reads green), and a pytest node id
# (`file.py::Class::test`) was silently discarded by path discovery so the run
# ended with "No test files to run" while looking like an accepted selector.


def test_zero_collected_across_run_fails_and_says_so(tmp_path: Path) -> None:
    """A -k that matches nothing must FAIL, not report a green summary."""
    probe_dir = _make_probe_dir(tmp_path)
    proc = _run_runner(probe_dir, "-k", "zzz_matches_nothing")
    assert proc.returncode == 1, proc.stdout
    assert "NO TESTS RAN" in proc.stdout
    assert "NOT a pass" in proc.stdout




def test_fail_on_skip_fails_a_run_with_any_skip_even_beside_an_xfail(tmp_path: Path) -> None:
    """--fail-on-skip reads the parsed skip count, not the progress line.

    The remote-backends lane once grepped the line for an ``<n>s,`` suffix,
    which an xfail printed after the skip (``1✓ 1s 1xf, ...``) hid.
    """
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "test_skipprobe.py").write_text(
        "import pytest\n\n"
        "def test_runs():\n    assert True\n\n"
        "def test_skips():\n    pytest.skip('host not wired')\n\n"
        "@pytest.mark.xfail(strict=True)\ndef test_xfails():\n    assert False\n"
    )
    plain = _run_runner(probe_dir)
    assert plain.returncode == 0, plain.stdout
    assert "1s 1xf," in plain.stdout, plain.stdout

    gated = _run_runner(probe_dir, "--fail-on-skip")
    assert gated.returncode == 1, gated.stdout
    assert "1 test SKIPPED under --fail-on-skip" in gated.stdout
    assert "test_skipprobe.py  (1 skipped)" in gated.stdout


def test_fail_on_skip_passes_a_run_without_skips(tmp_path: Path) -> None:
    proc = _run_runner(_make_probe_dir(tmp_path), "--fail-on-skip")
    assert proc.returncode == 0, proc.stdout
    assert "SKIPPED" not in proc.stdout


@pytest.mark.parametrize("form,expected", [
    ("positional", ["alpha", "beta"]), ("bare-k", ["alpha"]),
    ("node-id", ["alpha"]), ("explicit-k", ["beta"]),
    ("pathsep", ["alpha", "beta", "gamma"]),
])
def test_runner_selection_records_actual_test_identity(tmp_path, form, expected):
    probe = tmp_path / "probe"
    other = tmp_path / "other"
    probe.mkdir()
    other.mkdir()
    receipt = tmp_path / "witnesses"
    receipt.mkdir()
    for directory, filename, names in ((probe, "test_flags.py", ["alpha", "beta"]),
                                       (other, "test_other.py", ["gamma"])):
        (directory / filename).write_text("from pathlib import Path\n" + "".join(
            f"def test_{name}():\n    Path({str(receipt / name)!r}).touch()\n" for name in names
        ), encoding="utf-8")
    target = str(probe / "test_flags.py")
    arguments = {
        "positional": [str(probe), "-q"],
        "bare-k": ["--paths", str(probe), "-k", "test_alpha"],
        "node-id": [target + "::test_alpha"],
        "explicit-k": [target + "::test_alpha", "-k", "test_beta"],
        "pathsep": ["--paths", os.pathsep.join([str(probe), str(other)])],
    }[form]
    runner = _probe_root(tmp_path) / "scripts/run_tests_parallel.py"
    result = subprocess.run([sys.executable, str(runner), *arguments, "-j", "1", "--file-timeout", "30"],
                            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(path.name for path in receipt.iterdir()) == expected


@pytest.mark.parametrize("selector", ["files", "discovery"])
@pytest.mark.parametrize("ignore", ["glob", "glob-spaced", "path"])
def test_passthrough_ignore_drops_files_the_runner_hands_pytest_explicitly(
    tmp_path, selector, ignore,
):
    """Each file reaches its own pytest as an explicit argument, which pytest's
    --ignore/--ignore-glob never filter; the runner must apply them itself. The
    Windows lane's ``--ignore-glob='*test_desktop_update_windows_*.py'`` gate was a
    no-op, so those files ran (and hit the per-file cap) on PRs it meant to spare."""
    root = _probe_root(tmp_path)   # relative globs anchor at the repo root (per-file pytest's cwd)
    probe = root / "tests" / "probe"
    probe.mkdir(parents=True)
    receipt = tmp_path / "witnesses"
    receipt.mkdir()
    for name in ("keep", "gated_skipme"):
        (probe / f"test_{name}.py").write_text(
            f"from pathlib import Path\ndef test_{name}():\n    Path({str(receipt / name)!r}).touch()\n",
            encoding="utf-8")
    pick = (["--files", os.pathsep.join(f"tests/probe/test_{n}.py" for n in ("keep", "gated_skipme"))]
            if selector == "files" else ["--paths", str(probe)])
    flag = {"glob": ["--ignore-glob=*test_gated_*.py"],
            "glob-spaced": ["--ignore-glob", "*test_gated_*.py"],
            "path": ["--ignore=tests/probe/test_gated_skipme.py"]}[ignore]
    runner = root / "scripts/run_tests_parallel.py"
    result = subprocess.run([sys.executable, str(runner), *pick, "-j", "1", "--file-timeout", "30", "--", *flag],
                            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(path.name for path in receipt.iterdir()) == ["keep"], result.stdout
    assert "excluded 1 test file" in result.stdout, result.stdout




@pytest.mark.platforms("posix")  # POSIX signal death; Windows has no SIGSEGV exit
def test_interpreter_crash_is_reported_as_a_crash_not_as_no_tests_ran(tmp_path: Path) -> None:
    """A file whose interpreter dies by signal is classified as CRASHED (#113186).

    A native fault after some tests passed leaves no pytest summary line, so
    every count parses to 0. The runner used to file that under "no tests ran
    (collection/import error)" beneath a summary reading ``0 failed`` — two
    wrong diagnoses for one real bug. The crash must be named on the summary
    line and in the failure buckets, and the run must still exit non-zero.
    """
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "test_probe_crash.py").write_text(
        textwrap.dedent(
            """
            import os, signal

            def test_before():
                assert True

            def test_crash():
                os.kill(os.getpid(), signal.SIGSEGV)
            """
        )
    )

    proc = _run_runner(probe_dir, "--file-retries", "0")

    assert proc.returncode != 0
    assert "1 file CRASHED" in proc.stdout
    assert "SIGSEGV" in proc.stdout
    assert "where no tests ran" not in proc.stdout
    assert "NO TESTS RAN" not in proc.stdout


# ── --files-from: file-backed explicit file lists ───────────────────────────
#
# --files carries the whole list as ONE argv element, and Linux caps a
# single argument at MAX_ARG_STRLEN (128 KiB) — a much smaller limit than
# ARG_MAX. The whole-suite list (~210 KB) dies with E2BIG in execve before
# the runner's first line runs. --files-from takes the same explicit list
# from a file (or stdin via '-'), one path per line, so the list is bounded
# by the filesystem instead of one argv element.


def test_files_from_runs_exactly_the_listed_files(tmp_path: Path) -> None:
    """A newline-separated list file bypasses discovery like --files."""
    probe_dir = _make_probe_dir(tmp_path)
    extra = tmp_path / "probe_extra"
    extra.mkdir()
    (extra / "test_extra.py").write_text("def test_extra():\n    assert True\n")

    list_file = tmp_path / "files.txt"
    list_file.write_text(
        f"{probe_dir / 'test_flagprobe.py'}\n\n{extra / 'test_extra.py'}\n",
        encoding="utf-8",
    )

    # --paths points at the probe dir too; an explicit list must win and
    # NOT discover the extra file by accident.
    proc = _run_runner(probe_dir, "--files-from", str(list_file), "-q")
    assert proc.returncode == 0, proc.stdout
    assert "Running 2 test files" in proc.stdout, proc.stdout
    assert "3 passed" not in proc.stdout, proc.stdout


def test_files_from_dash_reads_the_list_from_stdin(tmp_path: Path) -> None:
    """--files-from - reads the list from stdin."""
    probe_dir = _make_probe_dir(tmp_path)

    repo_root = Path(__file__).resolve().parent.parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    proc = subprocess.run(
        [sys.executable, str(runner), "--files-from", "-",
         "-j", "1", "--file-timeout", "30", "-q"],
        input=f"{probe_dir / 'test_flagprobe.py'}\n",
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    assert "Running 1 test files" in proc.stdout, proc.stdout
    assert "✓2" in proc.stdout or "2 passed" in proc.stdout, proc.stdout


def test_scratch_root_is_per_user(tmp_path: Path, monkeypatch) -> None:
    """Two users on one host must never share the runner's scratch root.

    A fixed literal in a world-writable sticky dir belongs to whoever created it first: a
    root-owned root (a container or system-service run) makes every later ``mkdtemp`` there
    fail with EPERM for every other user, with no non-root way back.
    """
    import importlib

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    runner = importlib.import_module("run_tests_parallel")

    monkeypatch.delenv("HERMES_TEST_TMP_ROOT", raising=False)
    # Exercise the non-/var/tmp arm so the probe never mints roots in the real shared dir.
    # (Narrow: on 3.14 ``Path.is_dir()`` itself goes through ``os.path.isdir``.)
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(tmp_path))
    real_isdir = os.path.isdir
    monkeypatch.setattr(runner.os.path, "isdir",
                        lambda path: False if str(path) == "/var/tmp" else real_isdir(path))

    monkeypatch.setattr(runner.os, "getuid", lambda: 1000)
    mine = Path(runner._runner_scratch_root())
    monkeypatch.setattr(runner.os, "getuid", lambda: 0)
    theirs = Path(runner._runner_scratch_root())

    assert mine != theirs
    assert mine.is_dir() and theirs.is_dir()
    assert mine.parent == tmp_path and theirs.parent == tmp_path


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership/mode validation")
def test_explicit_runner_temp_root_is_validated_and_used(tmp_path: Path, monkeypatch) -> None:
    import importlib
    from scripts import test_temp_root

    root = tmp_path.resolve() / "runner-root"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    monkeypatch.setattr(test_temp_root, "SOCKET_TMPDIR_MAX_LEN", len(os.fsencode(str(root))) + 11)
    monkeypatch.setenv("HERMES_TEST_TMP_ROOT", str(root))

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    runner = importlib.import_module("run_tests_parallel")

    assert Path(runner._runner_scratch_root()) == root


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink/ownership validation")
def test_explicit_runner_temp_root_rejects_symlinks_and_foreign_owners(
    tmp_path: Path, monkeypatch,
) -> None:
    import importlib
    from scripts import test_temp_root

    base = tmp_path.resolve()
    target = base / "target"
    target.mkdir(mode=0o700)
    target.chmod(0o700)
    link = base / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(test_temp_root, "SOCKET_TMPDIR_MAX_LEN", len(os.fsencode(str(target))) + 11)
    monkeypatch.setenv("HERMES_TEST_TMP_ROOT", str(link))

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    runner = importlib.import_module("run_tests_parallel")
    with pytest.raises(ValueError, match="symlink"):
        runner._runner_scratch_root()

    target.chmod(0o777)
    monkeypatch.setenv("HERMES_TEST_TMP_ROOT", str(target))
    with pytest.raises(ValueError, match="mode 0700"):
        runner._runner_scratch_root()

    target.chmod(0o700)
    unsafe_parent = base / "unsafe-parent"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    unsafe_root = unsafe_parent / "runner-root"
    unsafe_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        test_temp_root, "SOCKET_TMPDIR_MAX_LEN", len(os.fsencode(str(unsafe_root))) + 11,
    )
    monkeypatch.setenv("HERMES_TEST_TMP_ROOT", str(unsafe_root))
    try:
        with pytest.raises(ValueError, match="unsafe writable ancestor"):
            runner._runner_scratch_root()
    finally:
        unsafe_parent.chmod(0o700)

    monkeypatch.setenv("HERMES_TEST_TMP_ROOT", str(target))
    actual_uid = os.getuid()
    monkeypatch.setattr(test_temp_root.os, "getuid", lambda: actual_uid + 1)
    with pytest.raises(ValueError, match="owned by the current user"):
        runner._runner_scratch_root()


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership/mode validation")
def test_explicit_runner_temp_root_enforces_socket_path_budget(tmp_path: Path, monkeypatch) -> None:
    import importlib
    from scripts import test_temp_root

    root = tmp_path.resolve() / "runner-root"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    monkeypatch.setattr(test_temp_root, "SOCKET_TMPDIR_MAX_LEN", len(os.fsencode(str(root))) + 10)
    monkeypatch.setenv("HERMES_TEST_TMP_ROOT", str(root))

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    runner = importlib.import_module("run_tests_parallel")
    with pytest.raises(ValueError, match="socket-safe"):
        runner._runner_scratch_root()


def test_off_host_note_names_platforms_specs_that_exclude_this_host(tmp_path: Path) -> None:
    """A green local run must say which platforms() tests were skipped and where
    they run; specs are resolved (posix is not off-host on Linux or macOS)."""
    probe_dir = _make_probe_dir(tmp_path)
    (probe_dir / "test_gated.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.platforms('not linux')\ndef test_elsewhere():\n    assert True\n\n"
        "@pytest.mark.platforms('posix')\ndef test_posix():\n    assert True\n\n"
        "@pytest.mark.platforms('windows')\ndef test_windows():\n    assert True\n",
        encoding="utf-8",
    )
    proc = _run_runner(probe_dir)
    notes = [line for line in proc.stdout.splitlines() if "SKIPPED on this host" in line]
    host = {"linux": "linux", "darwin": "macos", "win32": "windows"}[sys.platform]
    off_host = {"windows"} - {host}
    if host == "linux":
        off_host.add("not linux")
    if host == "windows":
        off_host.add("posix")
    assert {n.split("platforms(")[1].split(")")[0].strip("'") for n in notes} == off_host, proc.stdout
    assert all("they run on the" in n for n in notes), proc.stdout


def test_slices_partition_the_suite_exactly_even_with_different_duration_caches(capsys) -> None:
    """Each CI slice job restores its own duration cache; the slices must still cover every file once."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rtp_slices", Path(__file__).resolve().parents[2] / "scripts" / "run_tests_parallel.py")
    rtp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rtp)
    root = Path("/repo")
    files = [root / f"tests/t_{i:03d}.py" for i in range(101)]
    caches = [{f"tests/t_{i:03d}.py": float(i) for i in range(101)}, {"tests/t_000.py": 500.0}, {}]
    slices = [rtp._slice_files(list(reversed(files)) if i % 2 else files, i % 3 + 1, 3, caches[i % 3], root)
              for i in range(3)]
    assert sorted(f for s in slices for f in s) == sorted(files)

def test_huge_requested_worker_count_is_capped(tmp_path: Path) -> None:
    """A caller cannot turn the file runner into an unbounded process fanout."""
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe = probe_dir / "test_worker_cap.py"
    probe.write_text("def test_smoke():\n    assert True\n", encoding="utf-8")

    repo_root = _probe_root(tmp_path)
    env = os.environ.copy()
    env.pop("HERMES_TEST_MAX_WORKERS", None)
    env.pop("HERMES_TEST_WORKERS", None)
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_tests_parallel.py"),
            "--files",
            str(probe),
            "-j",
            "40",
            "--file-retries",
            "0",
            "-q",
        ],
        check=False,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    expected = min(os.cpu_count() or 1, 8)
    assert proc.returncode == 0, proc.stdout
    assert f"with -j {expected}" in proc.stdout, proc.stdout


def test_worker_path_excludes_mise_tool_shims(tmp_path: Path) -> None:
    """A normal worker cannot resolve or execute mise's Rust-tool shims."""
    repo_root = Path(__file__).resolve().parents[2]
    probe_dir = repo_root / f".runner-mise-probe-{os.getpid()}"
    probe_dir.mkdir()
    capture = tmp_path / "worker-env.json"
    executions = tmp_path / "mise-tool-executions"
    mise_shims = tmp_path / "mise" / "shims"
    mise_executable = tmp_path / "mise-runtime" / "mise"
    safe_bin = tmp_path / "safe-bin"
    mise_shims.mkdir(parents=True)
    mise_executable.parent.mkdir()
    safe_bin.mkdir()
    mise_executable.write_text(
        "#!/bin/sh" + chr(10) + f"echo \"$0\" >> {executions!s}" + chr(10),
        encoding="utf-8",
    )
    mise_executable.chmod(0o755)
    exposed_bins = []
    for tool in ("rustup", "cargo", "rustc"):
        exposed = tmp_path / f"mise-exposing-{tool}"
        exposed.mkdir()
        (exposed / tool).symlink_to(mise_executable)
        (safe_bin / tool).write_text("#!/bin/sh" + chr(10) + "exit 0" + chr(10), encoding="utf-8")
        (safe_bin / tool).chmod(0o755)
        exposed_bins.append(exposed)

    aliased_bins = []
    suffix_bins = []
    for tool in ("rustup", "cargo", "rustc"):
        real_mise = tmp_path / f"regular-{tool}" / "mise"
        real_shims = real_mise / "shims"
        real_shims.mkdir(parents=True)
        regular_shim = real_shims / tool
        regular_shim.write_text(
            "#!/bin/sh" + chr(10) + f"echo \"$0\" >> {executions!s}" + chr(10),
            encoding="utf-8",
        )
        regular_shim.chmod(0o755)
        alias = tmp_path / f"aliased-{tool}"
        alias.symlink_to(real_mise, target_is_directory=True)
        aliased_bins.append(alias / "shims")

        suffix_bin = tmp_path / f"suffix-{tool}"
        suffix_bin.mkdir()
        (suffix_bin / f"{tool}.exe").symlink_to(mise_executable)
        suffix_bins.append(suffix_bin)

    uncertain_bin = tmp_path / "uncertain-bin"
    uncertain_bin.mkdir()
    (uncertain_bin / "cargo").symlink_to(tmp_path / "missing-mise-target")

    probe = probe_dir / "test_worker_path.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            import json
            import os
            import shutil
            from pathlib import Path

            def test_worker_path_is_sanitized():
                Path({str(capture)!r}).write_text(json.dumps({{
                    "path": os.environ.get("PATH", ""),
                    "tools": {{tool: shutil.which(tool) for tool in ("rustup", "cargo", "rustc")}},
                }}))
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        (
            str(mise_shims),
            *(str(path) for path in exposed_bins),
            *(str(path) for path in aliased_bins),
            *(str(path) for path in suffix_bins),
            str(uncertain_bin),
            str(safe_bin),
            "/usr/bin",
            "/bin",
        )
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_tests_parallel.py"),
            "--files",
            str(probe),
            "-j",
            "1",
            "--file-retries",
            "0",
            "-q",
            f"--rootdir={repo_root}",
        ],
        check=False,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout
    data = json.loads(capture.read_text(encoding="utf-8"))
    path_entries = data["path"].split(os.pathsep)
    assert not any(Path(entry).as_posix().rstrip("/").endswith("/mise/shims") for entry in path_entries)
    assert not any(str(path) in path_entries for path in exposed_bins)
    assert not any(str(path) in path_entries for path in aliased_bins)
    assert not any(str(path) in path_entries for path in suffix_bins)
    assert str(uncertain_bin) not in path_entries
    assert data["tools"] == {tool: str(safe_bin / tool) for tool in ("rustup", "cargo", "rustc")}
    assert not executions.exists(), "ordinary worker executed a mise Rust-tool shim"
    shutil.rmtree(probe_dir, ignore_errors=True)


def test_direct_runner_resolves_empty_and_relative_path_at_repo_root(tmp_path: Path) -> None:
    """Direct invocation filters empty and dot PATH entries using the child CWD."""
    repo_root = Path(__file__).resolve().parents[2]
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    probe_dir = repo_root / f".runner-relative-mise-probe-{os.getpid()}-{tmp_path.name}"
    parent_cwd = tmp_path / "safe-parent"
    capture = tmp_path / "worker-env.json"
    executions = tmp_path / "mise-tool-executions"
    safe_bin = tmp_path / "safe-bin"
    mise = tmp_path / "mise"
    root_tools = [repo_root / tool for tool in ("rustup", "cargo", "rustc")]
    probe_dir.mkdir()
    parent_cwd.mkdir()
    safe_bin.mkdir()
    try:
        assert not any(path.exists() or path.is_symlink() for path in root_tools)
        mise.write_text(
            "#!/bin/sh\n" + f"echo \"$0\" >> {executions!s}\n",
            encoding="utf-8",
        )
        mise.chmod(0o755)
        for tool, root_tool in zip(("rustup", "cargo", "rustc"), root_tools):
            root_tool.symlink_to(mise)
            safe_tool = safe_bin / tool
            safe_tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            safe_tool.chmod(0o755)
        probe = probe_dir / "test_worker_path.py"
        probe.write_text(
            textwrap.dedent(
                f"""
                import json
                import os
                import shutil
                from pathlib import Path

                def test_worker_path_is_sanitized():
                    Path({str(capture)!r}).write_text(json.dumps({{
                        "path": os.environ["PATH"],
                        "tools": {{tool: shutil.which(tool) for tool in ("rustup", "cargo", "rustc")}},
                    }}))
                """
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(("", str(safe_bin), ".", "/usr/bin", "/bin"))
        proc = subprocess.run(
            [
                sys.executable,
                str(runner),
                "--files",
                str(probe),
                "--file-retries",
                "0",
                "-j",
                "1",
                "-q",
                f"--rootdir={repo_root}",
            ],
            check=False,
            cwd=parent_cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )

        assert proc.returncode == 0, proc.stdout
        data = json.loads(capture.read_text(encoding="utf-8"))
        assert "" not in data["path"].split(os.pathsep)
        assert "." not in data["path"].split(os.pathsep)
        assert data["tools"] == {tool: str(safe_bin / tool) for tool in ("rustup", "cargo", "rustc")}
        assert not executions.exists(), "ordinary worker executed a mise Rust-tool shim"
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
        for root_tool in root_tools:
            root_tool.unlink(missing_ok=True)
