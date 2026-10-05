"""Optional diagnostics do not make the agent depend on POSIX flock at import."""
import os
import subprocess
import sys


def test_missing_fcntl_keeps_diagnostics_importable_and_fail_open(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run([sys.executable, "-c", """
import builtins
import hermes_constants
original = builtins.__import__
def without_fcntl(name, *args, **kwargs):
    if name == "fcntl":
        raise ModuleNotFoundError("fixture: no flock capability")
    return original(name, *args, **kwargs)
builtins.__import__ = without_fcntl
from agent import cache_prefix_diagnostics as diagnostics
diagnostics._enabled = lambda: True
assert diagnostics.begin_attempt({}, session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0) is None
"""], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "cache" / "codex-cache-prefix.jsonl").exists()
