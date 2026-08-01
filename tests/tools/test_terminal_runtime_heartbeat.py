"""Terminal heartbeat preflight and lifecycle behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _config():
    return {
        "env_type": "local",
        "timeout": 30,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "auto_background_long_timeout": False,
        "auto_background_timeout_threshold": 19,
        "auto_background_timeout": 3300,
        "default_notify_on_background": False,
    }


def _run(monkeypatch, *, preflight):
    from tools.terminal_tool import terminal_tool

    proc = SimpleNamespace(
        id="proc-heartbeat",
        pid=123,
        notify_on_complete=False,
        watch_patterns=None,
        watcher_platform="",
        watcher_chat_id="",
        watcher_user_id="",
        watcher_user_name="",
        watcher_thread_id="",
        watcher_message_id="",
        watcher_interval=0,
    )
    registry = MagicMock()
    registry.spawn_local.return_value = proc
    env = MagicMock(env={})
    monkeypatch.setattr("tools.terminal_tool._get_env_config", _config)
    monkeypatch.setattr("tools.terminal_tool._start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        "tools.terminal_tool._check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setattr("tools.terminal_tool._active_environments", {"default": env})
    monkeypatch.setattr("tools.terminal_tool._last_activity", {"default": 0})
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr("tools.approval.get_current_session_key", lambda default="": "owner")
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "tools.runtime_heartbeat.preflight_current_heartbeat", preflight
    )
    return json.loads(
        terminal_tool(
            command="sleep 30",
            background=True,
            notify_on_complete=True,
            timeout=30,
            task_id="owner",
        )
    ), registry, proc


def test_invalid_exact_mapping_cannot_leave_spawned_orphan(monkeypatch):
    from tools.runtime_heartbeat import HeartbeatConfigError

    def _invalid():
        raise HeartbeatConfigError("missing exact mapping for custom:pm")

    result, registry, _proc = _run(monkeypatch, preflight=_invalid)

    assert "custom:pm" in result["error"]
    registry.spawn_local.assert_not_called()


def test_terminal_arms_only_after_successful_spawn(monkeypatch):
    from tools.runtime_heartbeat import runtime_heartbeat

    arm = MagicMock(return_value=True)
    monkeypatch.setattr(runtime_heartbeat, "arm", arm)
    result, registry, proc = _run(monkeypatch, preflight=lambda: 1700)

    assert result["session_id"] == proc.id
    registry.spawn_local.assert_called_once()
    arm.assert_called_once()
    assert arm.call_args.kwargs["interval"] == 1700
    assert arm.call_args.kwargs["caller_id"] == "owner"
