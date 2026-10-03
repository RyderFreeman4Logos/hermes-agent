"""Fork discards every lineage socket, not just its published listener."""
import gc
import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("side", ["accepted", "outgoing"])
def test_fork_discards_inflight_validation_sockets(side):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, __file__, "--issuer", side], env=env,
        capture_output=True, text=True, timeout=40,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["issuer_threads"] == 0
    assert observed["after"] == observed["before"]


def _socket_count():
    count = 0
    for entry in os.scandir("/proc/self/fd"):
        try:
            target = os.readlink(entry.path)
        except FileNotFoundError:
            continue
        count += target.startswith("socket:")
    return count


def _probe(side):
    from agent.delegation_context import delegated_child_context
    from tools.process_registry import ProcessRegistry

    before = _socket_count()
    ready, release = threading.Event(), threading.Event()
    errors, issued = [], []
    original = socket.socket.recv
    target = "delegation-lineage" if side == "accepted" else "fixture-validator"

    def recv(self, *args, **kwargs):
        data = original(self, *args, **kwargs)
        if threading.current_thread().name == target:
            ready.set()
            assert release.wait(10)
        return data

    def validate():
        try:
            issued.append(ProcessRegistry._spawn_env({}))
        except BaseException as exc:
            errors.append(type(exc).__name__)

    read_fd, write_fd = os.pipe()
    fork_pid = None
    reaped = False
    worker = None
    native = None
    stderr = b""
    with patch.object(socket.socket, "recv", recv):
        try:
            if side == "accepted":
                with delegated_child_context("fork-connection-fixture"):
                    env = ProcessRegistry._spawn_env({})
                native = subprocess.Popen(
                    [sys.executable, "-c",
                     "from tools.process_registry import ProcessRegistry; ProcessRegistry._spawn_env({})"],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            else:
                worker = threading.Thread(target=validate, name=target)
                worker.start()
            assert ready.wait(10), errors
            fork_pid = os.fork()
            if fork_pid == 0:
                os.close(read_fd)
                try:
                    gc.collect()
                    os.write(write_fd, json.dumps({
                        "before": before, "after": _socket_count(),
                        "issuer_threads": sum(t.name == "delegation-lineage" for t in threading.enumerate()),
                    }).encode())
                finally:
                    os.close(write_fd)
                    os._exit(0)
            os.close(write_fd)
            write_fd = -1
            assert select.select([read_fd], [], [], 10)[0]
            observed = json.loads(os.read(read_fd, 4096))
            _, status = os.waitpid(fork_pid, 0)
            reaped = True
            assert status == 0
        finally:
            release.set()
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
            if fork_pid is not None and not reaped:
                os.kill(fork_pid, signal.SIGKILL)
                os.waitpid(fork_pid, 0)
            if worker is not None:
                worker.join(10)
            if native is not None:
                try:
                    _, stderr = native.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    native.kill()
                    native.communicate()
                    raise
    if worker is not None:
        assert not worker.is_alive() and not errors
        assert issued[0].get("HERMES_DELEGATED_CHILD_SESSION") == "fork-connection-fixture"
    if native is not None:
        assert native.returncode == 0, stderr
    print(json.dumps(observed))


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    if sys.argv[1] == "--issuer" and sys.argv[2] == "outgoing":
        from agent.delegation_context import delegated_child_context
        from tools.process_registry import ProcessRegistry

        with delegated_child_context("fork-connection-fixture"):
            env = ProcessRegistry._spawn_env({})
        child = subprocess.run(
            [sys.executable, __file__, "--probe", "outgoing"], env=env,
            capture_output=True, text=True, timeout=30,
        )
        assert child.returncode == 0, child.stderr
        print(child.stdout, end="")
    else:
        _probe(sys.argv[2])
