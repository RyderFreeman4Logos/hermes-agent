"""The canonical shell must run tests and propagate a failing test's status."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


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
             check=False,
    )
    assert marker.read_text(encoding="utf-8") == "executed", result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
    assert "runner failure propagation canary" in result.stdout + result.stderr


@pytest.mark.platforms("posix")
def test_shell_runner_uses_configured_root_for_child(tmp_path):
    configured_root = os.environ.get("HERMES_TEST_TMP_ROOT")
    if not configured_root:
        pytest.skip("requires an explicitly configured supported temp root")
    assert configured_root is not None
    root_path = Path(configured_root).resolve()
    case = tmp_path / "test_runner_temp_root_canary.py"
    marker = tmp_path / "child-temp-root"
    case.write_text(
        "import os, tempfile\nfrom pathlib import Path\n"
        "def test_canary():\n"
        f"    root = Path({str(root_path)!r})\n"
        "    assert os.environ['HERMES_TEST_TMP_ROOT'] == str(root)\n"
        "    actual = Path(tempfile.gettempdir()).resolve()\n"
        "    assert actual.is_relative_to(root), actual\n"
        f"    Path({str(marker)!r}).write_text(str(actual), encoding='utf-8')\n",
        encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(repo / "scripts/run_tests.sh"), "-j", "1", str(case)],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
        env={**os.environ, "HERMES_PYTHON": sys.executable,
             "HERMES_TEST_FILE_RETRIES": "0"},
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert marker.is_file(), output
    assert Path(marker.read_text(encoding="utf-8")).is_relative_to(root_path)


@pytest.mark.platforms("posix")
def test_shell_runner_forwards_invalid_temp_root_to_runner(tmp_path):
    case = tmp_path / "test_runner_temp_root_canary.py"
    case.write_text("def test_canary():\n    assert True\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "scripts/run_tests.sh"), "-j", "1", str(case)],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
        env={**os.environ, "HERMES_PYTHON": sys.executable,
             "HERMES_TEST_FILE_RETRIES": "0", "HERMES_TEST_TMP_ROOT": "relative"},
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "HERMES_TEST_TMP_ROOT" in output
    assert "absolute path" in output
