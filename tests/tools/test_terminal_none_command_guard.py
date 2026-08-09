"""Regression tests for invalid/None terminal command handling."""

import json

from tools.terminal_tool import _transform_sudo_command, terminal_tool


def test_transform_sudo_command_none_returns_cleanly():
    transformed, sudo_stdin = _transform_sudo_command(None)

    assert transformed is None
    assert sudo_stdin is None


def test_terminal_tool_none_command_returns_clean_error():
    result = json.loads(terminal_tool(None))  # type: ignore[arg-type]

    assert result["exit_code"] == -1
    assert result["status"] == "error"
    assert "expected string" in result["error"].lower()
    assert "nonetype" in result["error"].lower()


def test_terminal_tool_rejects_ui_truncation_payload_before_environment_creation(
    monkeypatch,
):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("truncated command reached environment creation")

    monkeypatch.setattr("tools.terminal_tool._get_env_config", fail_if_called)

    for command in ("[truncated]", "…[truncated]", "echo stale...[truncated]"):
        result = json.loads(terminal_tool(command))

        assert result["exit_code"] == -1
        assert result["status"] == "error"
        assert "truncated" in result["error"].lower()

    assert not called


def test_terminal_tool_allows_marker_inside_a_complete_command(monkeypatch):
    def config_reached(*args, **kwargs):
        raise RuntimeError("config reached")

    monkeypatch.setattr("tools.terminal_tool._get_env_config", config_reached)

    result = json.loads(terminal_tool("printf '[truncated]'"))

    assert "config reached" in result["error"]
