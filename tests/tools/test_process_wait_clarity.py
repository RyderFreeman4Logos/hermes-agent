"""Tests for process wait timeout-result clarity and notification handoff."""

import threading
from unittest.mock import patch

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return ProcessRegistry()


def _spawn_sleeper(registry, notify=False):
    session = registry.spawn_local("sleep 30", cwd="/tmp", task_id="t-waitclar")
    session.notify_on_complete = notify
    return session.id


class _ForegroundWaitAttempted(threading.Event):
    def wait(self, timeout=None):
        raise AssertionError("foreground wait attempted")


class TestWaitTimeoutClarity:
    def test_wait_timeout_marks_process_running(self, registry):
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=1)
            assert r["status"] == "timeout"
            assert r["process_running"] is True
            assert "not an error" in r["timeout_note"]
            assert "Uptime" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_wait_timeout_suggests_notify_when_unset(self, registry):
        sid = _spawn_sleeper(registry, notify=False)
        try:
            r = registry.wait(sid, timeout=1)
            assert "notify_on_complete=true" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_wait_defers_to_notify_without_blocking(self, registry):
        sid = _spawn_sleeper(registry, notify=True)
        try:
            r = registry.wait(sid, timeout=1)
            assert r["status"] == "running"
            assert r["wait_deferred"] is True
            assert r["notify_on_complete"] is True
            assert "you will be notified exactly once" in r["note"]
        finally:
            registry.kill_process(sid)

    def test_clamped_wait_keeps_clamp_note_and_running_semantics(self, registry, monkeypatch):
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=600)
            assert r["status"] == "timeout"
            assert "clamped" in r["timeout_note"]
            assert "not an error" in r["timeout_note"]
            assert r["process_running"] is True
        finally:
            registry.kill_process(sid)

    def test_exited_process_unaffected(self, registry):
        session = registry.spawn_local("true", cwd="/tmp", task_id="t-waitclar")
        r = registry.wait(session.id, timeout=10)
        assert r["status"] == "exited"
        assert "process_running" not in r

    def test_notified_running_wait_returns_without_blocking(self, registry):
        session = ProcessSession(
            id="proc_wait_deferred",
            command="long-running command",
            notify_on_complete=True,
        )
        session._completion_event = _ForegroundWaitAttempted()
        registry._running[session.id] = session

        result = registry.wait(session.id, timeout=1)

        assert result["status"] == "running"
        assert result["wait_deferred"] is True
        assert result["notify_on_complete"] is True
        assert result["process_running"] is True
        assert not registry.is_completion_consumed(session.id)

    @pytest.mark.parametrize("exit_code", [0, 7])
    def test_deferred_wait_keeps_one_success_or_failure_notification(
        self, registry, exit_code
    ):
        session = ProcessSession(
            id=f"proc_wait_deferred_{exit_code}",
            command="long-running command",
            started_at=1.0,
            notify_on_complete=True,
            output_buffer="finished",
        )
        registry._running[session.id] = session

        result = registry.wait(session.id, timeout=1)
        assert result["wait_deferred"] is True

        session.exited = True
        session.exit_code = exit_code
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(session)

        notification = registry.completion_queue.get_nowait()
        assert notification["type"] == "completion"
        assert notification["exit_code"] == exit_code
        assert registry.completion_queue.empty()
