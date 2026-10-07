"""The canonical shell must run tests and propagate a failing test's status."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

from scripts import run_tests_parallel as test_runner


@pytest.mark.platforms("linux")
def test_runner_scratch_root_uses_explicit_tmpdir_without_redirect(tmp_path, monkeypatch):
    requested = tmp_path.resolve()
    real_makedirs = os.makedirs

    def contained_makedirs(path, *args, **kwargs):
        assert Path(path).parent == requested, f"runner escaped approved TMPDIR: {path}"
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setenv("TMPDIR", str(requested))
    monkeypatch.setattr(test_runner.os, "makedirs", contained_makedirs)

    root = Path(test_runner._runner_scratch_root())
    assert root == requested / f"hermes-pytest-{os.getuid()}"
    assert root.is_dir()


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


@pytest.mark.platforms("posix")
def test_shell_runner_executes_tests_and_propagates_failure(tmp_path):
    case = tmp_path / "test_runner_canary.py"
    marker = tmp_path / "executed"
    case.write_text(
        "from pathlib import Path\nimport os\n"
        "def test_canary():\n"
        f"    Path({str(marker)!r}).write_text('executed')\n"
        "    assert os.environ['PATHEXT'] == '.COM;.EXE;.BAT;.CMD'\n"
        "    assert False, 'runner failure propagation canary'\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "scripts/run_tests.sh"), "-j", "1", str(case)],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
        env={**os.environ, "HERMES_PYTHON": sys.executable, "HERMES_TEST_FILE_RETRIES": "0",
             "PATHEXT": ".COM;.EXE;.BAT;.CMD"},
    )
    assert marker.read_text(encoding="utf-8") == "executed", result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
    assert "runner failure propagation canary" in result.stdout + result.stderr
