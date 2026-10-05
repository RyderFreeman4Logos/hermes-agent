"""Verify scripts/run_tests_parallel.py kills test-spawned grandchildren.

Setup
-----
A test in this file spawns a long-lived Python grandchild that writes
its PID + a nonce to a tempfile, then exits without cleaning up.
With the old ``subprocess.run`` runner, that grandchild would orphan
and outlive the test (and the whole runner). With the current Popen +
``start_new_session`` + ``_kill_tree`` runner, the grandchild gets
SIGKILL'd via process-group kill when its file's pytest exits.

The leaker test always passes — its only job is to spawn a grandchild
and walk away. The verifier runs the runner over the leaker file in a
subprocess, then waits for the grandchild PID to disappear from the
kernel's process table.

POSIX-only: Windows has its own grandchild lifecycle (no shared session,
``taskkill /F /T`` semantics). Marked accordingly.
"""

from __future__ import annotations

import json
import shutil
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_probe_environment(monkeypatch):
    # Probe files exercise pytest/runner mechanics, not installed third-party
    # plugins. Autoloading the developer environment changes their startup cost.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def _probe_root(tmp_path):
    root = tmp_path / "runner-root"
    scripts = root / "scripts"
    (scripts / "ci").mkdir(parents=True, exist_ok=True)
    real = Path(__file__).resolve().parents[2] / "scripts"
    shutil.copy2(real / "run_tests_parallel.py", scripts)
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
    )


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
    )

    assert proc.returncode == 0, proc.stdout
    assert "FLAKY file" in proc.stdout
    assert "simulated first-attempt flake" in proc.stdout
    assert "first-attempt output" in proc.stdout
    assert "retry output" in proc.stdout




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
                            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(path.name for path in receipt.iterdir()) == expected




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
    )
    assert proc.returncode == 0, proc.stdout
    assert "Running 1 test files" in proc.stdout, proc.stdout
    assert "✓2" in proc.stdout or "2 passed" in proc.stdout, proc.stdout


@pytest.mark.platforms("posix")
def test_scratch_root_is_private_and_honors_tmpdir(monkeypatch) -> None:
    """Explicit temp locations survive cached defaults, with private per-user roots."""
    import importlib
    import tempfile

    runner = importlib.import_module("scripts.run_tests_parallel")
    # Stay short enough for real AF_UNIX paths, even inside this file's worker.
    base = Path(os.environ["TMPDIR"]).parent.parent
    with tempfile.TemporaryDirectory(prefix="t-", dir=base) as directory:
        monkeypatch.setenv("TMPDIR", directory)
        monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: "must-not-fallback")
        root = Path(runner._runner_scratch_root())
        assert root.parent == Path(directory).resolve()
        assert root.name.endswith(str(os.getuid()))
        assert root.stat().st_mode & 0o777 == 0o700
        assert root.stat().st_uid == os.getuid()
        link = Path(directory) / "link"
        link.symlink_to(directory, target_is_directory=True)
        monkeypatch.setenv("TMPDIR", str(link))
        assert Path(runner._runner_scratch_root()) == root
        assert link.is_symlink()
        root.chmod(0o755)
        with pytest.raises(PermissionError):
            runner._runner_scratch_root()


@pytest.mark.platforms("posix")
def test_unset_tmpdir_uses_hermes_scratch_not_system_temp(monkeypatch):
    import tempfile
    from scripts.run_tests_parallel import _runner_scratch_root

    base = Path(os.environ["TMPDIR"]).parent.parent
    with tempfile.TemporaryDirectory(prefix="t-", dir=base) as directory:
        home = Path(directory) / "home"
        link = Path(directory) / "link"
        link.symlink_to(directory, target_is_directory=True)
        monkeypatch.setenv("HERMES_HOME", str(link / "home"))
        monkeypatch.delenv("TMPDIR")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: pytest.fail("system temp fallback"))
        root = Path(_runner_scratch_root())
        assert root.parent == home / "cache/scratch"
        assert root.stat().st_mode & 0o777 == 0o700
        assert root.stat().st_uid == os.getuid()
        assert link.is_symlink()
        assert Path(_runner_scratch_root()) == root


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("caller", ["root", "main", "worker"])
@pytest.mark.parametrize("case", ["missing-home", "missing-cache", "existing-scratch", "symlink", "explicit"])
def test_rejected_scratch_ancestor_is_never_mutated(monkeypatch, caller, case):
    import tempfile
    from scripts import run_tests_parallel as runner

    base = Path(os.environ["TMPDIR"]).parent.parent
    with tempfile.TemporaryDirectory(prefix="t-", dir=base) as directory:
        directory = Path(directory)
        unsafe = directory / "unsafe"
        unsafe.mkdir(mode=0o700)
        home = unsafe / "home"
        if case in ("missing-cache", "existing-scratch", "explicit"):
            home.mkdir(mode=0o700)
        if case in ("existing-scratch", "explicit"):
            (home / "cache/scratch").mkdir(parents=True, mode=0o755)
        unsafe.chmod(0o777)
        selected = home
        if case == "symlink":
            selected = directory / "link"
            selected.symlink_to(home, target_is_directory=True)
        monkeypatch.setenv("HERMES_HOME", str(selected))
        monkeypatch.delenv("TMPDIR")
        if case == "explicit":
            monkeypatch.setenv("TMPDIR", str(home / "cache/scratch"))
        monkeypatch.setattr(tempfile, "gettempdir", lambda: pytest.fail("system temp fallback"))
        probe = directory / "test_probe.py"
        probe.write_text("def test_probe():\n    assert True\n", encoding="utf-8")
        before = {p.relative_to(unsafe): p.lstat() for p in (unsafe, *unsafe.rglob("*"))}
        monkeypatch.setattr(sys, "argv", [str(runner.__file__), "--files", str(probe), "-j", "1"])
        with pytest.raises(PermissionError, match="unsafe temp ancestor"):
            if caller == "root":
                runner._runner_scratch_root()
            elif caller == "worker":
                runner._run_one_file_once(probe, [], directory, 30)
            else:
                runner.main()
        after = {p.relative_to(unsafe): p.lstat() for p in (unsafe, *unsafe.rglob("*"))}
        assert after.keys() == before.keys(), "rejected ancestry must not create descendants"
        for path, info in before.items():
            assert (after[path].st_ino, after[path].st_mode, after[path].st_ctime_ns) == (
                info.st_ino, info.st_mode, info.st_ctime_ns
            ), "rejected ancestry must not replace or chmod descendants"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("creation", ["created", "eexist", "existing-child"])
@pytest.mark.parametrize("boundary", ["foreign", "sticky", "private", "fstat-error"])
def test_missing_scratch_chain_requires_private_owner(monkeypatch, creation, boundary):
    """Synthetic stat-owner boundary, not a privileged cross-UID attack."""
    import errno
    import tempfile
    from scripts import run_tests_parallel as runner

    base = Path(os.environ["TMPDIR"]).parent.parent
    with tempfile.TemporaryDirectory(prefix="t-", dir=base) as directory:
        parent = Path(directory)
        parent.chmod(0o1777)  # Legitimate pre-existing sticky traversal is allowed.
        home = parent / "home"
        suspect = home / "cache" if creation == "existing-child" else home
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("TMPDIR")
        original_mkdir, original_fstat = os.mkdir, os.fstat
        seen = []
        def mkdir(path, *args, **kwargs):
            if path == "home" and "dir_fd" in kwargs:
                original_mkdir(path, *args, **kwargs)
                if creation == "existing-child":
                    (home / "cache").mkdir(mode=0o700)
                if creation == "eexist":
                    raise FileExistsError(errno.EEXIST, "concurrent winner")
                return None
            return original_mkdir(path, *args, **kwargs)
        def fstat(fd):
            info = original_fstat(fd)
            if suspect.exists() and info.st_ino == suspect.stat().st_ino:
                seen.append(fd)
                if boundary == "fstat-error":
                    raise OSError(errno.EIO, "injected owner stat failure")
                fields = list(info)
                if boundary == "foreign":
                    fields[4] = os.getuid() + 1
                if boundary == "sticky":
                    fields[0] = (fields[0] & ~0o7777) | 0o1777
                return os.stat_result(fields)
            return info
        before = set(Path("/proc/self/fd").iterdir())
        with monkeypatch.context() as patch:
            patch.setattr(runner.os, "mkdir", mkdir)
            patch.setattr(runner.os, "fstat", fstat)
            if boundary == "private":
                assert Path(runner._runner_scratch_root()).is_dir()
            elif boundary == "fstat-error":
                with pytest.raises(OSError, match="injected owner stat failure"):
                    runner._runner_scratch_root()
            else:
                with pytest.raises(PermissionError):
                    runner._runner_scratch_root()
        assert seen, "must reach the opened missing-chain component"
        assert set(Path("/proc/self/fd").iterdir()) == before
        if boundary != "private":
            assert not list(suspect.iterdir()), "reject before creating any child"
            assert suspect.is_dir(), "do not remove a concurrent worker's name"


@pytest.mark.platforms("posix")
def test_preexisting_foreign_sticky_scratch_parent_is_traversable(monkeypatch):
    import tempfile
    from scripts import run_tests_parallel as runner

    base = Path(os.environ["TMPDIR"]).parent.parent
    with tempfile.TemporaryDirectory(prefix="t-", dir=base) as directory:
        parent = Path(directory)
        parent.chmod(0o1777)
        inode = parent.stat().st_ino
        original_fstat = os.fstat
        seen = []
        def fstat(fd):
            info = original_fstat(fd)
            if info.st_ino == inode:
                fields = list(info)
                fields[4] = os.getuid() + 1
                seen.append(fd)
                return os.stat_result(fields)
            return info
        monkeypatch.setenv("TMPDIR", directory)
        monkeypatch.setattr(runner.os, "fstat", fstat)
        root = Path(runner._runner_scratch_root())
        assert seen and root.is_dir() and root.parent == parent
        assert root.stat().st_uid == os.getuid() and root.stat().st_mode & 0o077 == 0


@pytest.mark.platforms("posix")
def test_invalid_tmpdir_never_falls_back(tmp_path, monkeypatch) -> None:
    import importlib

    runner = importlib.import_module("scripts.run_tests_parallel")
    for value in ("relative", "", str(tmp_path / "missing")):
        monkeypatch.setenv("TMPDIR", value)
        with pytest.raises((ValueError, FileNotFoundError)):
            runner._runner_scratch_root()
    tmp_path.chmod(0o777)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with pytest.raises(PermissionError):
        runner._runner_scratch_root()


@pytest.mark.parametrize("platform,limit", [("linux", 107), ("darwin", 103)])
def test_scratch_socket_path_budget(platform, limit):
    from scripts.run_tests_parallel import _check_scratch_socket_path

    # Include the production apt-stage GNUPGHOME nested under each r-XXXXXXXX
    # temp root, with every socket name reported by gpgconf --list-dirs.
    # The test-only apt-test-gnupg- prefix is shorter than this production one.
    agent_home = "/r-XXXXXXXX/apt-stage-gnupg-XXXXXXXX"
    suffixes = (
        "/r-XXXXXXXX/socket.sock",
        *(f"{agent_home}/S.gpg-agent{variant}" for variant in ("", ".extra", ".ssh", ".browser")),
    )
    suffix = max(suffixes, key=lambda value: len(os.fsencode(value)))
    root = Path("/" + "x" * (limit - len(os.fsencode(suffix)) - 1))
    _check_scratch_socket_path(root, platform)

    too_long = Path(str(root) + "x")
    assert len(os.fsencode(too_long)) + len(os.fsencode(suffixes[0])) <= limit
    assert len(os.fsencode(too_long)) + len(os.fsencode(suffix)) > limit
    with pytest.raises(ValueError, match="too long for AF_UNIX"):
        _check_scratch_socket_path(too_long, platform)

    multibyte_root_budget = limit - len(os.fsencode(suffix))
    multibyte = Path("/" + "\N{LATIN SMALL LETTER E WITH ACUTE}" * (multibyte_root_budget // 2 + 1))
    assert len(os.fsencode(multibyte)) > multibyte_root_budget
    with pytest.raises(ValueError, match="too long for AF_UNIX"):
        _check_scratch_socket_path(multibyte, platform)


@pytest.mark.platforms("windows")
def test_windows_scratch_root_rejects_permissive_existing_acl(tmp_path, monkeypatch):
    import win32security
    from scripts.run_tests_parallel import _runner_scratch_root

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    root = tmp_path / "h"
    root.mkdir()
    descriptor = win32security.ConvertStringSecurityDescriptorToSecurityDescriptor(
        "D:P(A;;FA;;;WD)", win32security.SDDL_REVISION_1)
    win32security.SetFileSecurity(str(root), win32security.DACL_SECURITY_INFORMATION, descriptor)
    with pytest.raises(OSError, match="permissive DACL"):
        _runner_scratch_root()
    root.rmdir()
    assert Path(_runner_scratch_root()) == root
    descriptor = win32security.GetFileSecurity(str(root), win32security.DACL_SECURITY_INFORMATION)
    dacl = descriptor.GetSecurityDescriptorDacl()
    from hermes_cli.windows_ssh_runtime import _allowed_sids
    assert dacl is not None
    assert {win32security.ConvertSidToStringSid(dacl.GetAce(i)[-1])
            for i in range(dacl.GetAceCount())} == _allowed_sids()
    # Same creation/verification boundary must work for the native default as well.
    monkeypatch.delenv("TMPDIR")
    assert Path(_runner_scratch_root()).is_dir()


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
