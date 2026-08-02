"""Foreground-release regressions for long terminal calls."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _config(**overrides):
    config = {
        "env_type": "local",
        "timeout": 21,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "auto_background_timeout_threshold": 20,
    }
    config.update(overrides)
    return config


def _session():
    return SimpleNamespace(
        id="proc_auto_bg",
        pid=4242,
        notify_on_complete=False,
        watch_patterns=[],
        watcher_platform="",
        watcher_chat_id="",
        watcher_user_id="",
        watcher_user_name="",
        watcher_thread_id="",
        watcher_message_id="",
        watcher_interval=0,
    )


def _run(
    *, command="make build", config=None, async_delivery=True,
    fresh_environment=False, **kwargs,
):
    from tools.terminal_tool import terminal_tool

    env = MagicMock(env={})
    env.execute.return_value = {"output": "foreground", "returncode": 0}
    proc = _session()
    registry = MagicMock(pending_watchers=[])
    registry.spawn_local.return_value = proc
    create_environment = MagicMock(return_value=env)
    active_environments = {} if fresh_environment else {"default": env}

    async_probe = MagicMock()
    if isinstance(async_delivery, Exception):
        async_probe.side_effect = async_delivery
    else:
        async_probe.return_value = async_delivery

    with (
        patch("tools.terminal_tool._get_env_config", return_value=config or _config()),
        patch("tools.terminal_tool._start_cleanup_thread"),
        patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}),
        patch("tools.terminal_tool._active_environments", active_environments),
        patch("tools.terminal_tool._last_activity", {"default": 0}),
        patch("tools.terminal_tool._create_environment", create_environment),
        patch("tools.process_registry.process_registry", registry),
        patch("tools.approval.get_current_session_key", return_value=""),
        patch("gateway.session_context.async_delivery_supported", async_probe),
        patch("gateway.session_context.get_session_env", return_value=""),
        patch("tools.runtime_heartbeat.preflight_current_heartbeat", return_value=None),
        patch("tools.runtime_heartbeat.runtime_heartbeat.arm"),
    ):
        result = json.loads(terminal_tool(command=command, **kwargs))

    return result, env, proc, registry, create_environment


@pytest.mark.parametrize(
    ("config", "kwargs"),
    [
        pytest.param(_config(timeout=21), {}, id="omitted-timeout-and-flags"),
        pytest.param(_config(timeout=5), {"timeout": 21}, id="explicit-timeout"),
        pytest.param(
            _config(timeout=5),
            {"timeout": 21, "background": False, "notify_on_complete": False},
            id="explicit-unsafe-false",
        ),
    ],
)
def test_long_foreground_calls_promote_to_managed_background(config, kwargs):
    result, env, proc, registry, _ = _run(config=config, **kwargs)

    assert result["session_id"] == proc.id
    assert result["notify_on_complete"] is True
    assert proc.notify_on_complete is True
    registry.spawn_local.assert_called_once()
    registry.wait.assert_not_called()
    env.execute.assert_not_called()


def test_auto_promotion_rewrites_long_execution_budget_to_7200():
    result, _, _, registry, create_environment = _run(
        timeout=21,
        background=False,
        notify_on_complete=False,
        fresh_environment=True,
    )

    assert result["session_id"] == "proc_auto_bg"
    assert create_environment.call_args.kwargs["timeout"] == 7200
    registry.spawn_local.assert_called_once()


def test_timeout_at_threshold_stays_foreground():
    result, env, _, registry, _ = _run(timeout=20)

    assert result["output"] == "foreground"
    env.execute.assert_called_once()
    registry.spawn_local.assert_not_called()


def test_explicit_background_call_is_not_rewritten_or_forced_to_notify():
    result, env, proc, registry, _ = _run(
        timeout=21,
        background=True,
        notify_on_complete=False,
    )

    assert result["session_id"] == proc.id
    assert "notify_on_complete" not in result
    assert proc.notify_on_complete is False
    registry.spawn_local.assert_called_once()
    env.execute.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 21, "command": "python -m http.server"},
        {"timeout": 21, "watch_patterns": ["ready"]},
    ],
)
def test_server_and_watch_requests_keep_explicit_background_guidance(kwargs):
    command = kwargs.pop("command", "make build")
    result, env, _, registry, _ = _run(command=command, **kwargs)

    assert "background=true" in result["error"]
    env.execute.assert_not_called()
    registry.spawn_local.assert_not_called()


@pytest.mark.parametrize(
    "async_delivery",
    [False, RuntimeError("probe failed")],
    ids=["unsupported", "probe-error"],
)
def test_auto_promotion_rejects_before_start_without_completion_route(async_delivery):
    result, env, _, registry, _ = _run(
        timeout=21,
        background=False,
        async_delivery=async_delivery,
    )

    assert "not started" in result["error"].lower()
    env.execute.assert_not_called()
    registry.spawn_local.assert_not_called()


def test_threshold_is_read_from_active_config_without_an_env_bridge(tmp_path, monkeypatch):
    import tools.terminal_tool as terminal_tool

    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "terminal:\n  auto_background_timeout_threshold: 7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(terminal_tool, "_ensure_terminal_env_bridged", lambda: None)

    assert terminal_tool._get_env_config()["auto_background_timeout_threshold"] == 7


def test_terminal_schema_describes_auto_promotion():
    from hermes_cli.config import DEFAULT_CONFIG
    from tools.terminal_tool import (
        AUTO_BACKGROUND_TIMEOUT,
        TERMINAL_SCHEMA,
        TERMINAL_TOOL_DESCRIPTION,
    )

    description = TERMINAL_SCHEMA["parameters"]["properties"]["timeout"]["description"]
    assert DEFAULT_CONFIG["terminal"]["auto_background_timeout_threshold"] == 200
    assert "auto_background_timeout_threshold" in description
    assert "background=true" in description
    assert str(AUTO_BACKGROUND_TIMEOUT) in description
    assert "auto_background_timeout_threshold" in TERMINAL_TOOL_DESCRIPTION
