"""Regression coverage for delegation.force_background dispatch policy."""

import json
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from hermes_cli.config import DEFAULT_CONFIG, load_config_readonly
from run_agent import AIAgent
from tools.delegate_tool import _load_config, _model_background_value


def _write_force_background_config(config_path, enabled: bool) -> None:
    """Write an equal-size config representation for an mtime-only reload test."""
    value = "true " if enabled else "false"
    config_path.write_text(
        f"delegation:\n  force_background: {value}\n  max_spawn_depth: 2\n"
    )


@pytest.mark.parametrize(
    ("force_background", "delegate_depth", "expected_background"),
    [
        pytest.param(False, 0, True, id="default-top-level-remains-background"),
        pytest.param(False, 1, False, id="default-subagent-remains-synchronous"),
        pytest.param(True, 0, True, id="forced-top-level-remains-background"),
        pytest.param(True, 1, True, id="forced-subagent-runs-background"),
    ],
)
def test_dispatch_delegate_task_honors_force_background(
    force_background: bool, delegate_depth: int, expected_background: bool
) -> None:
    """The live model dispatch path passes the config-derived background flag."""
    agent = cast(AIAgent, SimpleNamespace(_delegate_depth=delegate_depth))

    with (
        patch(
            "tools.delegate_tool._load_config",
            return_value={"force_background": force_background},
        ),
        patch("tools.delegate_tool.delegate_task", return_value="dispatched") as delegate_task,
    ):
        result = AIAgent._dispatch_delegate_task(agent, {"goal": "test task"})

    assert result == "dispatched"
    delegate_task.assert_called_once()
    assert delegate_task.call_args.kwargs["background"] is expected_background
    assert delegate_task.call_args.kwargs["force_background"] is (
        force_background and delegate_depth > 0
    )


def test_force_background_defaults_to_false() -> None:
    assert DEFAULT_CONFIG["delegation"]["force_background"] is False


@pytest.mark.parametrize(
    ("force_background", "delegate_depth", "expected_background"),
    [
        pytest.param(False, 1, False, id="subagent-default-remains-synchronous"),
        pytest.param(True, 1, True, id="forced-subagent-runs-background"),
    ],
)
def test_model_background_value_honors_force_background(
    force_background: bool, delegate_depth: int, expected_background: bool
) -> None:
    """The registry fallback mirrors the live dispatch config policy."""
    parent_agent = SimpleNamespace(_delegate_depth=delegate_depth)

    with patch(
        "tools.delegate_tool._load_config",
        return_value={"force_background": force_background},
    ):
        background = _model_background_value({"goal": "test task"}, parent_agent)

    assert background is expected_background


def test_forced_orchestrator_rejects_when_async_delivery_is_unsupported(
    tmp_path, monkeypatch
) -> None:
    """A force guarantee never executes a nested child in the caller's turn."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    _write_force_background_config(config_path, enabled=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Exercise the real config loader path used by both model dispatch surfaces.
    assert load_config_readonly()["delegation"]["force_background"] is True
    assert _load_config()["force_background"] is True

    agent = cast(Any, object.__new__(AIAgent))
    agent._delegate_depth = 1
    agent.session_id = "orchestrator-session"
    child = SimpleNamespace(_delegate_role="leaf", tool_progress_callback=None)
    credentials = {
        "model": "test-model",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }
    executions = []

    with (
        patch("tools.delegate_tool._build_child_agent", return_value=child),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value=credentials),
        patch("tools.delegate_tool._run_single_child") as run_child,
        patch(
            "gateway.session_context.async_delivery_supported", return_value=False
        ) as async_delivery_supported,
        patch("tools.async_delegation.dispatch_async_delegation_batch") as dispatch_async,
    ):
        result = json.loads(agent._dispatch_delegate_task({"goal": "complete this"}))

    # force_background is a guarantee.  If no durable completion route exists,
    # reject before work begins rather than taking the historical inline fallback
    # that blocks the orchestrator/main-agent turn.
    assert executions == []
    assert result["status"] == "rejected"
    assert result["mode"] == "background"
    assert "delegation_id" not in result
    assert "force_background" in result["error"]
    assert "not started" in result["note"]
    async_delivery_supported.assert_called_once()
    dispatch_async.assert_not_called()
    run_child.assert_not_called()


def test_forced_orchestrator_rejects_when_async_pool_is_full(tmp_path, monkeypatch) -> None:
    """Capacity rejection is non-blocking under the explicit force guarantee."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    _write_force_background_config(config_path, enabled=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    agent = cast(Any, object.__new__(AIAgent))
    agent._delegate_depth = 1
    agent.session_id = "orchestrator-session"
    child = SimpleNamespace(_delegate_role="leaf", tool_progress_callback=None)
    credentials = {
        "model": "test-model",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }

    with (
        patch("tools.delegate_tool._build_child_agent", return_value=child),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value=credentials),
        patch("tools.delegate_tool._run_single_child") as run_child,
        patch("gateway.session_context.async_delivery_supported", return_value=True),
        patch(
            "tools.async_delegation.dispatch_async_delegation_batch",
            return_value={"status": "rejected", "error": "pool full"},
        ) as dispatch_async,
    ):
        result = json.loads(agent._dispatch_delegate_task({"goal": "complete this"}))

    assert result["status"] == "rejected"
    assert result["mode"] == "background"
    assert "force_background" in result["error"]
    assert "pool full" in result["error"]
    assert "not started" in result["note"]
    dispatch_async.assert_called_once()
    run_child.assert_not_called()


def test_forced_nested_completion_targets_durable_parent_session(tmp_path, monkeypatch) -> None:
    """A detached leaf completion is owned by the top-level durable session."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    _write_force_background_config(config_path, enabled=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    agent = cast(Any, object.__new__(AIAgent))
    agent._delegate_depth = 1
    agent.session_id = "ephemeral-orchestrator-session"
    agent._parent_session_id = "durable-top-level-session"
    child = SimpleNamespace(_delegate_role="leaf", tool_progress_callback=None)
    credentials = {
        "model": "test-model",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }

    with (
        patch("tools.delegate_tool._build_child_agent", return_value=child),
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value=credentials),
        patch("gateway.session_context.async_delivery_supported", return_value=True),
        patch(
            "tools.async_delegation.dispatch_async_delegation_batch",
            return_value={"status": "dispatched", "delegation_id": "delegation-1"},
        ) as dispatch_async,
    ):
        result = json.loads(agent._dispatch_delegate_task({"goal": "complete this"}))

    assert result["status"] == "dispatched"
    dispatch_async.assert_called_once()
    assert (
        dispatch_async.call_args.kwargs["parent_session_id"]
        == "durable-top-level-session"
    )


def test_force_background_config_hot_reloads_for_next_dispatch(tmp_path, monkeypatch) -> None:
    """Each model dispatch sees the current force_background config generation."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    agent = cast(Any, object.__new__(AIAgent))
    agent._delegate_depth = 1

    _write_force_background_config(config_path, enabled=True)
    first_stat = config_path.stat()
    assert load_config_readonly()["delegation"]["force_background"] is True
    assert _load_config()["force_background"] is True

    with patch("tools.delegate_tool.delegate_task", return_value="dispatched") as delegate_task:
        assert AIAgent._dispatch_delegate_task(agent, {"goal": "first"}) == "dispatched"

        _write_force_background_config(config_path, enabled=False)
        # "true " and "false" have identical byte lengths, so this reload proves
        # the cache watches mtime as well as size. Give the temp file a distinct
        # nanosecond timestamp without a timing-dependent sleep.
        os.utime(
            config_path,
            ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns + 1_000_000_000),
        )
        second_stat = config_path.stat()
        assert second_stat.st_size == first_stat.st_size
        assert second_stat.st_mtime_ns != first_stat.st_mtime_ns
        assert load_config_readonly()["delegation"]["force_background"] is False
        assert _load_config()["force_background"] is False

        # The next dispatch reads the rewritten config rather than a stale cache.
        assert AIAgent._dispatch_delegate_task(agent, {"goal": "second"}) == "dispatched"

    assert [call.kwargs["background"] for call in delegate_task.call_args_list] == [True, False]
