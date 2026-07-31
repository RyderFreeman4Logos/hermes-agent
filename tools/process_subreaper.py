#!/usr/bin/env python3
"""Linux subreaper for one managed background command.

Every descendant of the launcher is owned until it exits, including children
that call ``setsid()``.  Service-manager launches are outside this process tree
and therefore are not owned.  The kernel child relationship scopes waiting to
this command, so concurrent sessions and recycled PIDs cannot be confused.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys

_PR_SET_CHILD_SUBREAPER = 36


def _enable_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def main(argv: list[str] | None = None) -> int:
    command = list(sys.argv[1:] if argv is None else argv)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("process_subreaper: no command given after '--'", file=sys.stderr)
        return 2

    try:
        _enable_subreaper()
    except OSError as exc:
        print(f"process_subreaper: cannot enable subreaper: {exc}", file=sys.stderr)
        return 125

    launcher = subprocess.Popen(
        command,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    launcher_exit = launcher.wait()

    # Orphaned descendants are synchronously adopted by this subreaper when
    # their parent exits. waitpid therefore covers setsid/double-fork children
    # without ancestry polling, a global process scan, or PID-reuse races.
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break

    if launcher_exit < 0:
        signum = -launcher_exit
        # Mirroring SIGSTOP would suspend this wrapper forever.  Match the
        # conventional shell encoding instead; terminating signals still reach
        # the parent as a negative Popen return code.
        if signum == getattr(signal, "SIGSTOP", None):
            return 128 + signum
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (OSError, ValueError):
            # SIGKILL cannot have a handler, but can still terminate us.
            if signum != getattr(signal, "SIGKILL", None):
                return 128 + signum
        os.kill(os.getpid(), signum)
        return 128 + signum
    return launcher_exit


if __name__ == "__main__":
    sys.exit(main())
