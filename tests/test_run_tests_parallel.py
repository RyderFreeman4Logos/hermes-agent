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
import os
import select
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import ModuleType
from typing import NamedTuple, TextIO

import pytest


def _load_runner_module() -> ModuleType:
    """Load the runner for a direct lifecycle seam test."""
    runner = Path(__file__).resolve().parent.parent / "scripts" / "run_tests_parallel.py"
    spec = importlib.util.spec_from_file_location("runner_lifecycle_test", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Both tests share the same handoff file: the leaker writes here, the
# verifier reads here. We park it in $TMPDIR with a unique-per-run name
# so concurrent invocations of the suite don't clobber each other.
_HANDOFF_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "hermes-isolation-probe"
_HANDOFF_DIR.mkdir(exist_ok=True)


def _handoff_path_for(nonce: str) -> Path:
    return _HANDOFF_DIR / f"grandchild-{nonce}.json"


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


def test_environment_writer_failure_removes_secret_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A partial environment transport is removed before spawn can begin."""
    runner = _load_runner_module()
    root = tmp_path / "attempt-root"

    def make_root(*args: object, **kwargs: object) -> str:
        root.mkdir()
        return str(root)

    def fail_after_partial_write(value: object, target: TextIO) -> None:
        target.write("partial-secret")
        raise OSError("injected environment write failure")

    monkeypatch.setattr(runner.tempfile, "mkdtemp", make_root)
    monkeypatch.setattr(runner.json, "dump", fail_after_partial_write)

    with pytest.raises(OSError, match="injected environment write failure"):
        runner._run_one_file_once(tmp_path / "test_probe.py", [], tmp_path, 1)

    assert not root.exists()


def test_supervisor_removes_transport_after_watch_parent_loss(tmp_path: Path) -> None:
    """The scope-side supervisor deletes secrets when its runner disappears."""
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    transport = tmp_path / "environment.json"
    secret = "parent-loss-secret-sentinel"
    transport.write_text(json.dumps({"PARENT_LOSS_SECRET": secret}), encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(runner),
            "--_hermes-supervise",
            str(repo_root),
            str(tmp_path / "completion.json"),
            str(transport),
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
    proc.stdin.close()
    assert proc.wait(timeout=10) == 137
    assert proc.stdout is not None
    assert secret not in proc.stdout.read()
    assert not transport.exists()


def test_progress_output_tolerates_legacy_stdout_encoding(tmp_path: Path) -> None:
    """Progress glyphs must not crash the runner on non-UTF-8 consoles."""
    repo_root = Path(__file__).resolve().parent.parent
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
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout
    assert "UnicodeEncodeError" not in proc.stdout
    assert "1 tests passed" in proc.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only probe")
@pytest.mark.live_system_guard_bypass
def test_grandchild_leak_is_killed_by_runner(tmp_path: Path) -> None:
    """Run the parallel runner over a probe file and verify cleanup.

    1. Materialize a probe file that spawns a long-lived grandchild and
       writes its PID to disk before exiting.
    2. Invoke ``scripts/run_tests_parallel.py`` against the probe file.
    3. Wait for the grandchild PID to vanish (poll for ~5s).
    4. Assert the runner exited cleanly AND the grandchild is dead.
    """
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    assert runner.exists(), f"runner missing at {runner}"

    # Probe lives in a temp dir, NOT under tests/, so the regular suite
    # never picks it up — only our explicit invocation does.
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe = probe_dir / "test_probe_leaker.py"
    nonce = f"{os.getpid()}-{int(time.time() * 1000)}"
    handoff = _handoff_path_for(nonce)
    if handoff.exists():
        handoff.unlink()

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
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # The runner declares its stdio UTF-8 (see _make_stdio_glyph_safe);
        # decode the same way so ✓-glyph assertions hold on Windows, where
        # text=True alone would decode with the locale codec (cp1252).
        encoding="utf-8",
        errors="replace",
        timeout=60,
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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only probe")
@pytest.mark.live_system_guard_bypass
def test_runner_sigkill_contains_descendant_tree(tmp_path: Path) -> None:
    """A SIGKILL'd runner must not orphan a bounded descendant tree."""
    repo_root = Path(__file__).resolve().parent.parent
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
# ``--generate-slices``, ``--files``, ``--include-integration``) straight
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
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    return subprocess.run(
        [sys.executable, str(runner), "--paths", str(probe_dir),
         "-j", "1", "--file-timeout", "30", *extra],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # The runner declares its stdio UTF-8 (see _make_stdio_glyph_safe);
        # decode the same way so ✓-glyph assertions hold on Windows, where
        # text=True alone would decode with the locale codec (cp1252).
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )




def test_bare_value_flag_keeps_its_value(tmp_path: Path) -> None:
    """``-k test_alpha`` reaches pytest as a selector, not as a path.

    The value token (``test_alpha``) must NOT be swallowed by the runner's
    positional-path discovery — if it were, discovery would look for a path
    named ``test_alpha``, find nothing, and the run would degrade. We assert
    the run succeeds AND only one of the two tests was selected (proving the
    ``-k`` filter actually applied inside pytest).
    """
    probe_dir = _make_probe_dir(tmp_path)
    proc = _run_runner(probe_dir, "-k", "test_alpha")
    assert proc.returncode == 0, proc.stdout
    # Exactly one test selected: the per-file summary shows "1✓" (1 passed).
    # test_beta is deselected by the -k filter.
    assert "1✓" in proc.stdout or "1 passed" in proc.stdout, proc.stdout
    assert "2✓" not in proc.stdout, (
        f"both tests ran — -k filter did not apply:\n{proc.stdout}"
    )




def test_positional_path_not_treated_as_flag(tmp_path: Path) -> None:
    """A positional path arg still overrides discovery (not routed to pytest)."""
    probe_dir = _make_probe_dir(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    # Pass the probe dir positionally (no --paths), plus a bare -q.
    proc = subprocess.run(
        [sys.executable, str(runner), str(probe_dir), "-j", "1",
         "--file-timeout", "30", "-q"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", timeout=60,
    )
    assert proc.returncode == 0, proc.stdout
    # Discovery found the probe file (2 tests), proving the positional path
    # was consumed as a root, not forwarded to pytest as a bad flag.
    assert "test_flagprobe.py" in proc.stdout, proc.stdout


def test_file_retry_self_heals_and_prints_both_attempts(tmp_path: Path) -> None:
    """A pass-on-retry is green, loud, and retains the failing traceback."""
    repo_root = Path(__file__).resolve().parent.parent
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
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
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
            transport = Path(cmd[flag + 3])
            try:
                private_env = json.loads(transport.read_text())
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
            os.execv(cmd[0], cmd)
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
    repo_root = Path(__file__).resolve().parent.parent
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




def test_node_id_selector_runs_the_named_test(tmp_path: Path) -> None:
    """``file.py::test_alpha`` runs that test instead of discovering nothing."""
    probe_dir = _make_probe_dir(tmp_path)
    target = probe_dir / "test_flagprobe.py"
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_tests_parallel.py"),
         f"{target}::test_alpha", "-j", "1", "--file-timeout", "30"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout
    assert "No test files to run" not in proc.stdout
    assert "node id" in proc.stdout  # explains the translation
    # Ran exactly the one selected test, not both in the file.
    assert "1 tests passed" in proc.stdout


def test_explicit_k_wins_over_node_id_inference(tmp_path: Path) -> None:
    """A caller's own ``-k`` is not overridden by the node-id translation."""
    probe_dir = _make_probe_dir(tmp_path)
    target = probe_dir / "test_flagprobe.py"
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_tests_parallel.py"),
         f"{target}::test_alpha", "-k", "test_beta",
         "-j", "1", "--file-timeout", "30"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=60,
    )
    # -k test_beta wins: one test ran, and it wasn't filtered to nothing.
    assert proc.returncode == 0, proc.stdout
    assert "1 tests passed" in proc.stdout


def test_multiple_absolute_paths_split_on_pathsep(tmp_path: Path) -> None:
    """``--paths`` accepts ``os.pathsep``-joined absolute paths.

    On Windows the absolute paths contain drive-letter colons, so a naive
    ``split(":")`` shreds them into phantom roots and only one (or neither)
    of the two probe dirs would be discovered.
    """
    dir_a = _make_probe_dir(tmp_path)
    dir_b = tmp_path / "probe_b"
    dir_b.mkdir()
    (dir_b / "test_flagprobe_b.py").write_text(
        "def test_gamma():\n    assert True\n"
    )
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / "scripts" / "run_tests_parallel.py"
    proc = subprocess.run(
        [sys.executable, str(runner),
         "--paths", os.pathsep.join([str(dir_a), str(dir_b)]),
         "-j", "1", "--file-timeout", "30", "-q"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", timeout=60,
    )
    assert proc.returncode == 0, proc.stdout
    assert "Discovered 2 test files" in proc.stdout, proc.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="drive-letter paths")
def test_drive_letter_colon_is_not_a_path_separator(tmp_path: Path) -> None:
    """An absolute ``--paths`` value stays one root on Windows.

    The naive split used to produce a phantom relative root ``'C'`` (the
    drive letter) alongside the real path; discovery only worked by the
    accident of ``repo_root / '\\rooted\\rest'`` re-anchoring onto the
    repo's drive.
    """
    probe_dir = _make_probe_dir(tmp_path)
    proc = _run_runner(probe_dir, "-q")
    assert proc.returncode == 0, proc.stdout
    drive = str(probe_dir)[0]
    assert f"['{drive}', " not in proc.stdout, (
        f"drive letter split off as a phantom root:\n{proc.stdout}"
    )
    assert "Discovered 1 test files" in proc.stdout, proc.stdout


def test_huge_requested_worker_count_is_capped(tmp_path: Path) -> None:
    """A caller cannot turn the file runner into an unbounded process fanout."""
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    probe = probe_dir / "test_worker_cap.py"
    probe.write_text("def test_smoke():\n    assert True\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parent.parent
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
    repo_root = Path(__file__).resolve().parent.parent
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
    repo_root = Path(__file__).resolve().parent.parent
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
