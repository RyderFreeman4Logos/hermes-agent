import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import process_subreaper


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux-only: requires PR_SET_CHILD_SUBREAPER",
)


@pytest.mark.parametrize("signum", [signal.SIGKILL, signal.SIGTERM])
def test_subreaper_mirrors_launcher_signal(signum):
    wrapper = Path(__file__).parents[2] / "tools" / "process_subreaper.py"
    result = subprocess.run(
        [
            sys.executable,
            str(wrapper),
            "--",
            sys.executable,
            "-c",
            "import os,signal; os.kill(os.getpid(), int(signal.Signals(%d)))" % signum,
        ],
        check=False,
        timeout=10,
    )

    assert result.returncode == -signum


def test_subreaper_encodes_sigstop_without_suspending(monkeypatch):
    monkeypatch.setattr(process_subreaper, "_enable_subreaper", lambda: None)
    monkeypatch.setattr(
        process_subreaper.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(wait=lambda: -signal.SIGSTOP),
    )
    monkeypatch.setattr(
        process_subreaper.os,
        "waitpid",
        lambda *args: (_ for _ in ()).throw(ChildProcessError),
    )

    assert process_subreaper.main(["true"]) == 128 + signal.SIGSTOP
