"""The canonical shell must run tests and propagate a failing test's status."""
import getpass
from pathlib import Path
import os
import subprocess
import sys

import pytest

from scripts import run_tests_parallel as test_runner


@pytest.mark.parametrize("selection", ["explicit", "default"])
@pytest.mark.platforms("linux")
def test_runner_scratch_root_uses_explicit_tmpdir_without_redirect(tmp_path, monkeypatch, selection):
    private_base = tmp_path.resolve()
    requested = tmp_path / "private-alias"
    requested.symlink_to(private_base, target_is_directory=True)
    real_mkdir = os.mkdir

    def contained_mkdir(path, mode=0o777, *, dir_fd=None):
        assert Path(path).parent == requested, f"runner escaped approved TMPDIR: {path}"
        return real_mkdir(path, mode, dir_fd=dir_fd)

    if selection == "explicit":
        monkeypatch.setenv("TMPDIR", str(requested))
    else:
        monkeypatch.delenv("TMPDIR", raising=False)
        real_isdir = os.path.isdir
        monkeypatch.setattr(test_runner.os.path, "isdir",
                            lambda p: False if p == "/var/tmp" else real_isdir(p))
        monkeypatch.setattr(test_runner.tempfile, "gettempdir", lambda: str(requested))
    monkeypatch.setattr(test_runner.os, "mkdir", contained_mkdir)

    root = Path(test_runner._runner_scratch_root())
    assert root == requested / f"hermes-pytest-{os.getuid()}"
    assert root.is_dir()


@pytest.mark.platforms("linux")
def test_runner_scratch_root_preserves_symlink_dotdot_destination(tmp_path, monkeypatch):
    lexical = tmp_path / "lexical"
    target = tmp_path / "target"
    lexical.mkdir(mode=0o700)
    (lexical / "base").mkdir(mode=0o700)
    target.mkdir(mode=0o700)
    (target / "inner").mkdir(mode=0o700)
    resolved_base = target / "base"
    resolved_base.mkdir(mode=0o700)
    (lexical / "link").symlink_to(target / "inner", target_is_directory=True)
    requested = lexical / "link" / ".." / "base"
    assert requested.resolve(strict=True) == resolved_base
    monkeypatch.setenv("TMPDIR", str(requested))
    wrong_parent_root = lexical / "base" / f"hermes-pytest-{os.getuid()}"

    try:
        root = Path(test_runner._runner_scratch_root())
    except ValueError:
        assert not wrong_parent_root.exists(), "scratch root was created in the lexically normalized parent"
        raise

    expected_root = requested / wrong_parent_root.name
    assert root == expected_root
    assert root.resolve(strict=True) == resolved_base / wrong_parent_root.name
    assert root.is_dir()
    assert not wrong_parent_root.exists(), "scratch root was created in the wrong parent"


@pytest.mark.platforms("linux")
def test_runner_scratch_root_rejects_socket_path_over_budget(tmp_path, monkeypatch):
    temproot = tmp_path / ("x" * 80)
    temproot.mkdir()
    socket_path = temproot / f"pytest-of-{getpass.getuser()}" / "pytest-0" / "test_socket0" / "s.sock"
    assert len(os.fsencode(socket_path)) > 107  # Linux sun_path has 108 bytes including NUL.

    with pytest.raises(ValueError, match="socket path"):
        test_runner._validate_runner_socket_path(str(temproot))
    monkeypatch.setenv("TMPDIR", str(temproot))
    real_mkdtemp = test_runner.tempfile.mkdtemp
    workers = []

    def track_worker(*args, **kwargs):
        assert kwargs["prefix"] == "r"
        worker = real_mkdtemp(*args, **kwargs)
        workers.append(Path(worker))
        return worker

    monkeypatch.setattr(test_runner.tempfile, "mkdtemp", track_worker)
    monkeypatch.setattr(test_runner.subprocess, "Popen",
                        lambda *args, **kwargs: pytest.fail("over-budget worker must not spawn"))
    result = test_runner._run_one_file_once(Path(__file__), [], tmp_path, 30)
    assert result[1] == 2
    assert "socket path" in result[2]
    assert len(workers) == 1
    assert not workers[0].exists()


@pytest.mark.platforms("linux")
def test_runner_scratch_root_refuses_link_planted_at_creation(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    target = private / "owned-child"
    target.mkdir(mode=0o700)
    root = private / f"hermes-pytest-{os.getuid()}"
    monkeypatch.setenv("TMPDIR", str(private))

    real_mkdir = os.mkdir
    planted = False

    def plant_link():
        nonlocal planted
        if not os.path.lexists(root):
            os.symlink(target, root)
            planted = True

    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        if Path(path) == root:
            plant_link()
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(test_runner.os, "mkdir", racing_mkdir)

    with pytest.raises(ValueError, match="symlink|canonical-path"):
        test_runner._runner_scratch_root()

    assert planted
    assert root.is_symlink()
    assert target.is_dir()


@pytest.mark.parametrize("shared_mode", [0o1777, 0o1703])
@pytest.mark.platforms("linux")
def test_runner_scratch_root_rejects_shared_tmpdir_before_creation(
        tmp_path, monkeypatch, shared_mode):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(shared_mode)
    root = shared / f"hermes-pytest-{os.getuid()}"
    monkeypatch.setenv("TMPDIR", str(shared))

    with pytest.raises(ValueError, match="group/other-writable"):
        test_runner._runner_scratch_root()

    assert not root.exists()


@pytest.mark.platforms("linux")
def test_runner_scratch_root_rejects_shared_default_parent_before_creation(
        tmp_path, monkeypatch):
    shared = tmp_path / "shared-default"
    shared.mkdir()
    shared.chmod(0o1777)
    root = shared / f"hermes-pytest-{os.getuid()}"
    monkeypatch.delenv("TMPDIR", raising=False)
    real_isdir = os.path.isdir
    monkeypatch.setattr(
        test_runner.os.path, "isdir",
        lambda path: False if path == "/var/tmp" else real_isdir(path),
    )
    monkeypatch.setattr(test_runner.tempfile, "gettempdir", lambda: str(shared))
    created = []

    def track_makedirs(path, mode=0o777, exist_ok=False):
        created.append(Path(path))

    monkeypatch.setattr(test_runner.os, "makedirs", track_makedirs)

    with pytest.raises(ValueError, match="group/other-writable"):
        test_runner._runner_scratch_root()

    assert not created
    assert not root.exists()


@pytest.mark.parametrize("invalid_tmpdir", ["relative-root", "missing"])
@pytest.mark.platforms("linux")
def test_runner_scratch_root_fails_closed_for_invalid_explicit_tmpdir(
        tmp_path, monkeypatch, invalid_tmpdir):
    invalid = invalid_tmpdir if invalid_tmpdir == "relative-root" else str(tmp_path / invalid_tmpdir)
    monkeypatch.setenv("TMPDIR", invalid)
    monkeypatch.setattr(
        test_runner.os, "makedirs",
        lambda *args, **kwargs: pytest.fail("invalid TMPDIR must not create a fallback root"),
    )

    with pytest.raises(ValueError, match="TMPDIR"):
        test_runner._runner_scratch_root()


@pytest.mark.parametrize("shared_mode", [0o1777, 0o1703])
@pytest.mark.parametrize("selection", ["explicit", "default"])
@pytest.mark.parametrize("ancestry", ["lexical", "hop", "destination"])
@pytest.mark.platforms("linux")
def test_runner_main_rejects_alias_ancestry_without_mutation(
        tmp_path, monkeypatch, shared_mode, selection, ancestry):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(shared_mode)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir(mode=0o700)
    second.mkdir(mode=0o700)
    alias = (shared if ancestry == "lexical" else tmp_path) / "alias"
    if ancestry == "hop":
        hop = shared / "hop"
        hop.symlink_to("../first", target_is_directory=True)
        alias.symlink_to("shared/hop", target_is_directory=True)
    elif ancestry == "destination":
        first = shared / "private"
        first.mkdir(mode=0o700)
        alias.symlink_to(first, target_is_directory=True)
    else:
        alias.symlink_to(first, target_is_directory=True)
    name = f"hermes-pytest-{os.getuid()}"
    real_mkdir = os.mkdir
    created, spawned, swept = [], [], []

    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        if Path(path) == alias / name:
            created.append(str(path))
            alias.unlink()
            alias.symlink_to(second, target_is_directory=True)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    def stop_sweep(root):
        swept.append(root)
        raise ValueError("stop unsafe baseline before sweep")

    def stop_spawn(*args, **kwargs):
        spawned.append(args)
        pytest.fail("denied ancestry must not spawn a worker")

    monkeypatch.setattr(test_runner.os, "mkdir", racing_mkdir)
    monkeypatch.setattr(test_runner, "_sweep_killed_run_roots", stop_sweep)
    monkeypatch.setattr(test_runner.subprocess, "Popen", stop_spawn)
    monkeypatch.setattr(sys, "argv", ["runner", "-j", "1", __file__])
    if selection == "explicit":
        monkeypatch.setenv("TMPDIR", str(alias))
    else:
        monkeypatch.delenv("TMPDIR", raising=False)
        real_isdir = os.path.isdir
        monkeypatch.setattr(test_runner.os.path, "isdir",
                            lambda p: False if p == "/var/tmp" else real_isdir(p))
        monkeypatch.setattr(test_runner.tempfile, "gettempdir", lambda: str(alias))
    with pytest.raises(ValueError):
        test_runner.main()
    assert not (second / name).exists(), "runner mutated redirected target before refusal"
    assert not (first / name).exists(), "runner mutated original target before refusal"
    assert not created
    assert not swept
    assert not spawned


@pytest.mark.parametrize("selection", ["explicit", "default"])
@pytest.mark.parametrize("leaf", ["symlink", "bad-mode"])
@pytest.mark.platforms("linux")
def test_runner_scratch_root_validates_existing_leaf(tmp_path, monkeypatch, selection, leaf):
    root = tmp_path / f"hermes-pytest-{os.getuid()}"
    if leaf == "symlink":
        root.symlink_to(tmp_path, target_is_directory=True)
    else:
        root.mkdir(mode=0o755)
        root.chmod(0o755)
    if selection == "explicit":
        monkeypatch.setenv("TMPDIR", str(tmp_path))
    else:
        monkeypatch.delenv("TMPDIR", raising=False)
        real_isdir = os.path.isdir
        monkeypatch.setattr(test_runner.os.path, "isdir",
                            lambda p: False if p == "/var/tmp" else real_isdir(p))
        monkeypatch.setattr(test_runner.tempfile, "gettempdir", lambda: str(tmp_path))
    with pytest.raises(ValueError, match="scratch root"):
        test_runner._runner_scratch_root()
    assert root.is_symlink() if leaf == "symlink" else root.stat().st_mode & 0o777 == 0o755


@pytest.mark.platforms("posix")
def test_shell_runner_executes_tests_and_propagates_failure(tmp_path):
    case = tmp_path / "test_runner_canary.py"
    marker = tmp_path / "executed"
    requested = tmp_path / "private-alias"
    configured = Path.home() / "tmp"
    requested.symlink_to(configured, target_is_directory=True)
    case.write_text(
        "from pathlib import Path\nimport os\n"
        "def test_canary():\n"
        f"    Path({str(marker)!r}).write_text(os.environ['TMPDIR'])\n"
        "    assert os.environ['PATHEXT'] == '.COM;.EXE;.BAT;.CMD'\n"
        "    assert False, 'runner failure propagation canary'\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "scripts/run_tests.sh"), "-j", "1", str(case)],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
        env={**os.environ, "HERMES_PYTHON": sys.executable, "HERMES_TEST_FILE_RETRIES": "0",
             "PATHEXT": ".COM;.EXE;.BAT;.CMD", "TMPDIR": str(requested)},
    )
    assert marker.exists(), result.stdout + result.stderr
    worker = Path(marker.read_text(encoding="utf-8"))
    assert worker.parent == requested / f"hermes-pytest-{os.getuid()}"
    assert worker.parent.resolve() == configured.resolve() / f"hermes-pytest-{os.getuid()}"
    assert worker.name.startswith("r")
    assert not worker.exists()
    assert requested.is_symlink()
    assert result.returncode != 0, result.stdout + result.stderr
    assert "runner failure propagation canary" in result.stdout + result.stderr
