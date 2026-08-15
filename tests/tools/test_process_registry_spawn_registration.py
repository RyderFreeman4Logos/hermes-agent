"""Registration-order regressions for local managed process spawns."""

import queue

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


class _SynchronousThread:
    """Run the reader inline so completion deterministically wins the race."""

    def __init__(self, *, target, args, **_kwargs):
        self._target = target
        self._args = args

    def start(self):
        self._target(*self._args)


def _finish_immediately(registry, session):
    with session._lock:
        session.exited = True
        session.exit_code = 0
        session.completion_reason = "exited"
    registry._move_to_finished(session)


def _assert_one_completion(registry, session):
    event = registry.completion_queue.get_nowait()
    assert event["session_id"] == session.id
    assert event["exit_code"] == 0

    # A later kill/reader reconciliation must not publish the same lifecycle twice.
    registry._move_to_finished(session)
    with pytest.raises(queue.Empty):
        registry.completion_queue.get_nowait()


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


def test_pipe_fast_exit_queues_completion_once(registry, monkeypatch, tmp_path):
    process = MagicMock(pid=454545)
    process.stdout = MagicMock()
    process.stdin = MagicMock()
    process.poll.return_value = 0

    monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/sh")
    monkeypatch.setattr(
        "tools.process_registry.subprocess.Popen",
        lambda *_a, **_kw: process,
    )
    monkeypatch.setattr(
        registry,
        "_reader_loop",
        lambda session: _finish_immediately(registry, session),
    )
    monkeypatch.setattr("tools.process_registry.threading.Thread", _SynchronousThread)

    session = registry.spawn_local(
        "true",
        cwd=str(tmp_path),
        notification_metadata={"notify_on_complete": True},
    )

    _assert_one_completion(registry, session)


def test_fast_exit_cancels_execution_deadline(registry, monkeypatch, tmp_path):
    process = MagicMock(pid=474747)
    process.stdout = MagicMock()
    process.stdin = MagicMock()
    process.poll.return_value = 0
    timer = MagicMock()

    monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/sh")
    monkeypatch.setattr(
        "tools.process_registry.subprocess.Popen",
        lambda *_a, **_kw: process,
    )
    monkeypatch.setattr(
        registry,
        "_reader_loop",
        lambda session: _finish_immediately(registry, session),
    )
    monkeypatch.setattr("tools.process_registry.threading.Thread", _SynchronousThread)
    monkeypatch.setattr(
        "tools.process_registry.threading.Timer", lambda *_a, **_kw: timer
    )

    session = registry.spawn_local(
        "true",
        cwd=str(tmp_path),
        execution_timeout=7200,
    )

    timer.start.assert_called_once_with()
    timer.cancel.assert_called_once_with()
    assert session._deadline_timer is None


def test_pty_fast_exit_queues_completion_once(registry, monkeypatch, tmp_path):
    process = MagicMock(pid=464646)

    monkeypatch.setattr("ptyprocess.PtyProcess.spawn", lambda *_a, **_kw: process)
    monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/sh")
    monkeypatch.setattr(
        registry,
        "_pty_reader_loop",
        lambda session: _finish_immediately(registry, session),
    )
    monkeypatch.setattr("tools.process_registry.threading.Thread", _SynchronousThread)

    session = registry.spawn_local(
        "true",
        cwd=str(tmp_path),
        use_pty=True,
        notification_metadata={"notify_on_complete": True},
    )

    _assert_one_completion(registry, session)


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
