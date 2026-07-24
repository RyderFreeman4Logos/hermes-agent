"""Regression tests for the process(wait) auto-background threshold guard."""

import json
from unittest.mock import MagicMock


def _configure_auto_background(monkeypatch, *, enabled=True, threshold=200):
    """Make process(wait) read a deterministic terminal timeout config."""
    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "auto_background_long_timeout": enabled,
            "auto_background_timeout_threshold": threshold,
        },
    )


def test_wait_above_threshold_polls_without_blocking(monkeypatch):
    """Long waits return the current background-process status immediately."""
    import tools.process_registry as process_module

    _configure_auto_background(monkeypatch, threshold=200)
    wait = MagicMock(return_value={"status": "waited"})
    poll = MagicMock(return_value={"status": "running"})
    monkeypatch.setattr(process_module.process_registry, "wait", wait)
    monkeypatch.setattr(process_module.process_registry, "poll", poll)

    result = json.loads(
        process_module._handle_process(
            {"action": "wait", "session_id": "proc-threshold", "timeout": 300}
        )
    )

    assert result["status"] == "running"
    assert "backgrounded" in result["note"].lower()
    poll.assert_called_once_with("proc-threshold")
    wait.assert_not_called()


def test_wait_at_or_below_threshold_blocks_normally(monkeypatch):
    """Short waits retain the existing wait behavior."""
    import tools.process_registry as process_module

    _configure_auto_background(monkeypatch, threshold=200)
    wait = MagicMock(return_value={"status": "waited"})
    poll = MagicMock(return_value={"status": "running"})
    monkeypatch.setattr(process_module.process_registry, "wait", wait)
    monkeypatch.setattr(process_module.process_registry, "poll", poll)

    result = json.loads(
        process_module._handle_process(
            {"action": "wait", "session_id": "proc-short", "timeout": 10}
        )
    )

    assert result["status"] == "waited"
    wait.assert_called_once_with("proc-short", timeout=10)
    poll.assert_not_called()


def test_wait_blocks_normally_when_auto_background_is_disabled(monkeypatch):
    """Disabling terminal auto-background also disables the wait guard."""
    import tools.process_registry as process_module

    _configure_auto_background(monkeypatch, enabled=False, threshold=200)
    wait = MagicMock(return_value={"status": "waited"})
    poll = MagicMock(return_value={"status": "running"})
    monkeypatch.setattr(process_module.process_registry, "wait", wait)
    monkeypatch.setattr(process_module.process_registry, "poll", poll)

    result = json.loads(
        process_module._handle_process(
            {"action": "wait", "session_id": "proc-auto-off", "timeout": 300}
        )
    )

    assert result["status"] == "waited"
    wait.assert_called_once_with("proc-auto-off", timeout=300)
    poll.assert_not_called()


def test_wait_without_timeout_guards_its_300_second_effective_timeout(monkeypatch):
    """An omitted wait timeout uses 300 seconds for the threshold decision."""
    import tools.process_registry as process_module

    _configure_auto_background(monkeypatch, threshold=200)
    wait = MagicMock(return_value={"status": "waited"})
    poll = MagicMock(return_value={"status": "running"})
    monkeypatch.setattr(process_module.process_registry, "wait", wait)
    monkeypatch.setattr(process_module.process_registry, "poll", poll)

    result = json.loads(
        process_module._handle_process(
            {"action": "wait", "session_id": "proc-omitted-timeout"}
        )
    )

    assert result["status"] == "running"
    assert "timeout=300s" in result["note"]
    poll.assert_called_once_with("proc-omitted-timeout")
    wait.assert_not_called()
