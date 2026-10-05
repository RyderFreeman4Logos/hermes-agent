"""Content-free Git metadata for abandoned-delegation recovery.

Recovery keeps private evidence in canonical task records and live transcripts;
only locators and best-effort working-tree status belong in diagnostic envelopes.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

_GIT_TIMEOUT_S = 5


def git_state_hint(cwd: Optional[str]) -> Optional[str]:
    """``sha (N uncommitted file(s))``; omit private paths, branch names and commit text."""
    if not cwd or not os.path.isdir(cwd):
        return None

    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", cwd, *args], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace",
                                 stdin=subprocess.DEVNULL, timeout=_GIT_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout if out.returncode == 0 else None

    head = run("rev-parse", "--short", "HEAD")
    if head is None:
        return None
    status = run("status", "--porcelain")
    dirty = len(status.splitlines()) if status is not None else "?"
    return f"{head.strip()} ({dirty} uncommitted file(s))"
