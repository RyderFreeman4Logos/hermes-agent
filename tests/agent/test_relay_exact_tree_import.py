"""Keep relay imports hermetic to the source tree."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_relay_llm_imports_without_installed_agent_fallback() -> None:
    root = Path(__file__).resolve().parents[2]
    env = {"PYTHONPATH": os.fspath(root), "PYTHONNOUSERSITE": "1"}
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import agent.relay_llm"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
