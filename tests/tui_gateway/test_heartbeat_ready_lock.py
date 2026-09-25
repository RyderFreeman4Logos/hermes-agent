"""A held state.db write lock must not delay gateway.ready.

The stdio entrypoint upserts its heartbeat before the ready frame. That
write waits up to the shared 20s patience, which is longer than the TUI's
15s startup timer, so a foreign holder makes the parent kill a healthy child.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

READY_BUDGET_S = 2.0
RPC_BUDGET_S = 5.0


def _hold_write_lock(db_path: Path) -> subprocess.Popen:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.touch()
    holder = subprocess.Popen(
        [
            sys.executable, "-c",
            "import fcntl,sys,time; "
            "f=open(sys.argv[1],'a+b'); "
            "fcntl.flock(f.fileno(), fcntl.LOCK_EX); "
            "sys.stdout.write('held\\n'); sys.stdout.flush(); "
            "time.sleep(120)",
            str(db_path) + ".write.lock",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "held"
    return holder


def _spawn_gateway(home: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    return subprocess.Popen(
        [sys.executable, "-m", "tui_gateway.entry"],
        cwd=str(home),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _first_line(proc: subprocess.Popen, budget_s: float) -> tuple[str, float]:
    assert proc.stdout is not None
    started = time.monotonic()
    line = proc.stdout.readline()
    return line, time.monotonic() - started if line else budget_s + 1


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.kill()
    proc.wait(timeout=5)


def test_ready_and_rpc_under_held_write_lock(tmp_path: Path):
    home = tmp_path / "home"
    holder = gateway = None
    try:
        holder = _hold_write_lock(home / "state.db")
        gateway = _spawn_gateway(home)
        line, elapsed = _first_line(gateway, READY_BUDGET_S)
        assert elapsed < READY_BUDGET_S, f"ready waited {elapsed:.2f}s: {line!r}"
        frame = json.loads(line)
        assert frame["params"]["type"] == "gateway.ready"

        assert gateway.stdin is not None
        gateway.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "config.get"}) + "\n")
        gateway.stdin.flush()
        reply, rpc_elapsed = _first_line(gateway, RPC_BUDGET_S)
        assert rpc_elapsed < RPC_BUDGET_S, f"rpc waited {rpc_elapsed:.2f}s: {reply!r}"
        body = json.loads(reply)
        assert body.get("id") == 1
        assert "error" not in body or body["error"]["code"] != -32000
    finally:
        _stop(gateway)
        _stop(holder)


def test_heartbeat_row_lands_after_lock_release(tmp_path: Path):
    home = tmp_path / "home"
    holder = gateway = None
    try:
        holder = _hold_write_lock(home / "state.db")
        gateway = _spawn_gateway(home)
        line, elapsed = _first_line(gateway, READY_BUDGET_S)
        assert elapsed < READY_BUDGET_S and "gateway.ready" in line

        holder.terminate()
        holder.wait(timeout=5)
        holder = None

        from hermes_state import SessionDB

        deadline = time.monotonic() + 8
        rows = []
        while time.monotonic() < deadline:
            rows = SessionDB(home / "state.db").list_backend_heartbeats()
            if rows:
                break
            time.sleep(0.1)
        assert rows, "heartbeat never landed after the lock was released"
        assert rows[0]["pid"] == gateway.pid
    finally:
        _stop(gateway)
        _stop(holder)
