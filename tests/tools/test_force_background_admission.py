"""Regression coverage for issue #30's non-blocking admission contract."""

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from run_agent import AIAgent
from tools.process_registry import _handle_process, process_registry


def test_force_background_keeps_nested_model_dispatch_detached():
    """Configured force_background must override the nested sync default."""
    agent = SimpleNamespace(_delegate_depth=1)
    with (
        patch("tools.delegate_tool._load_config", return_value={"force_background": True}),
        patch("tools.delegate_tool.delegate_task", return_value="dispatched") as delegate_task,
    ):
        result = AIAgent._dispatch_delegate_task(cast(Any, agent), {"goal": "work"})

    assert result == "dispatched"
    assert delegate_task.call_args.kwargs["background"] is True
    assert delegate_task.call_args.kwargs["force_background"] is True


def test_long_admission_never_reblocks_through_process_wait(monkeypatch):
    """The configured 19-second threshold must skip a re-blocking wait."""
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {
            "timeout": 180,
            "auto_background_long_timeout": True,
            "auto_background_timeout_threshold": 19,
        },
    )
    with (
        patch.object(process_registry, "poll", return_value={"status": "running"}) as poll,
        patch.object(process_registry, "wait") as wait,
    ):
        result = json.loads(
            _handle_process({"action": "wait", "session_id": "proc_test", "timeout": 20})
        )

    assert result["status"] == "running"
    assert result["wait_skipped"] is True
    assert "19s" in result["note"]
    poll.assert_called_once_with("proc_test")
    wait.assert_not_called()
