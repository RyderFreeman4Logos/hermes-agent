"""Regression tests for the bounded process(wait) foreground guard."""

import json
from unittest.mock import MagicMock

import pytest


def _configure(monkeypatch, *, timeout=180, threshold=20):
    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "timeout": timeout,
            "auto_background_timeout_threshold": threshold,
        },
    )


@pytest.mark.parametrize(
    ("requested_timeout", "configured_timeout"),
    [
        pytest.param(80, 100, id="explicit-long-wait"),
        pytest.param(None, 80, id="omitted-long-effective-wait"),
    ],
)
def test_long_wait_returns_current_status_without_blocking(
    monkeypatch, requested_timeout, configured_timeout
):
    import tools.process_registry as process_module

    _configure(monkeypatch, timeout=configured_timeout)
    wait = MagicMock(return_value={"status": "waited"})
    poll = MagicMock(return_value={"status": "running"})
    monkeypatch.setattr(process_module.process_registry, "wait", wait)
    monkeypatch.setattr(process_module.process_registry, "poll", poll)
    args = {"action": "wait", "session_id": "proc-long"}
    if requested_timeout is not None:
        args["timeout"] = requested_timeout

    result = json.loads(process_module._handle_process(args))

    assert result["status"] == "running"
    assert "notify_on_complete" in result["note"]
    poll.assert_called_once_with("proc-long")
    wait.assert_not_called()


@pytest.mark.parametrize(
    ("requested_timeout", "configured_timeout", "expected_timeout"),
    [
        pytest.param(20, 100, 20, id="exact-threshold"),
        pytest.param(80, 10, 10, id="configured-cap-makes-effective-wait-short"),
        pytest.param(None, 10, 10, id="omitted-short-effective-wait"),
    ],
)
def test_short_effective_wait_keeps_bounded_synchronous_wait(
    monkeypatch, requested_timeout, configured_timeout, expected_timeout
):
    import tools.process_registry as process_module

    _configure(monkeypatch, timeout=configured_timeout)
    wait = MagicMock(return_value={"status": "waited"})
    poll = MagicMock(return_value={"status": "running"})
    monkeypatch.setattr(process_module.process_registry, "wait", wait)
    monkeypatch.setattr(process_module.process_registry, "poll", poll)

    result = json.loads(
        process_module._handle_process(
            {"action": "wait", "session_id": "proc-short", "timeout": requested_timeout}
        )
    )

    assert result["status"] == "waited"
    wait.assert_called_once_with("proc-short", timeout=expected_timeout)
    poll.assert_not_called()


def test_process_schema_describes_long_wait_as_immediate_status():
    from tools.process_registry import PROCESS_SCHEMA

    description = PROCESS_SCHEMA["description"]
    timeout_description = PROCESS_SCHEMA["parameters"]["properties"]["timeout"][
        "description"
    ]
    assert "notify_on_complete" in description
    assert "current status immediately" in timeout_description
    assert "block until done" not in description
