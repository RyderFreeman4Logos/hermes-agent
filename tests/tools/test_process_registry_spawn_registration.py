"""Registration-order regressions for local managed process spawns."""

from unittest.mock import MagicMock

import pytest

from tools.process_registry import ProcessRegistry


class _InspectingThread:
    """Observe registry ownership at the reader-thread publication boundary."""

    def __init__(self, *, args, observations, registry, **_kwargs):
        self._session = args[0]
        self._observations = observations
        self._registry = registry

    def start(self):
        self._observations.append(
            self._registry.get(self._session.id) is self._session
        )


@pytest.fixture()
def registry(monkeypatch):
    instance = ProcessRegistry()
    monkeypatch.setattr(instance, "_write_checkpoint", MagicMock(return_value=True))
    monkeypatch.setattr(
        "tools.process_registry._is_supervised_gateway_process",
        lambda: False,
    )
    return instance


def test_pipe_spawn_registers_before_reader_start(registry, monkeypatch, tmp_path):
    observations = []
    process = MagicMock(pid=424242)
    process.stdout = MagicMock()
    process.stdin = MagicMock()
    process.poll.return_value = None

    monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/sh")
    monkeypatch.setattr(
        "tools.process_registry.subprocess.Popen",
        lambda *_a, **_kw: process,
    )
    monkeypatch.setattr(
        "tools.process_registry.threading.Thread",
        lambda **kwargs: _InspectingThread(
            **kwargs,
            observations=observations,
            registry=registry,
        ),
    )

    session = registry.spawn_local("sleep 30", cwd=str(tmp_path))

    assert observations == [True]
    assert registry.get(session.id) is session


def test_pty_spawn_registers_before_reader_start(registry, monkeypatch, tmp_path):
    observations = []
    process = MagicMock(pid=434343)

    monkeypatch.setattr("ptyprocess.PtyProcess.spawn", lambda *_a, **_kw: process)
    monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/sh")
    monkeypatch.setattr(
        "tools.process_registry.threading.Thread",
        lambda **kwargs: _InspectingThread(
            **kwargs,
            observations=observations,
            registry=registry,
        ),
    )

    session = registry.spawn_local("sleep 30", cwd=str(tmp_path), use_pty=True)

    assert observations == [True]
    assert registry.get(session.id) is session


def test_pty_reader_start_failure_reaps_child_without_pipe_fallback(
    registry, monkeypatch, tmp_path
):
    process = MagicMock(pid=444444)

    class _FailingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("reader start failed")

    monkeypatch.setattr("ptyprocess.PtyProcess.spawn", lambda *_a, **_kw: process)
    monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/sh")
    monkeypatch.setattr("tools.process_registry.threading.Thread", _FailingThread)

    def _unexpected_pipe_fallback(*_args, **_kwargs):
        raise AssertionError("must not start a second process after PTY ownership")

    monkeypatch.setattr(
        "tools.process_registry.subprocess.Popen", _unexpected_pipe_fallback
    )

    with pytest.raises(RuntimeError, match="PTY setup failed after process spawn"):
        registry.spawn_local("sleep 30", cwd=str(tmp_path), use_pty=True)

    process.terminate.assert_called_once_with(force=True)
    assert registry.list_sessions() == []
