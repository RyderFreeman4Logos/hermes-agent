"""Terminal heartbeat preflight and lifecycle behavior."""

from __future__ import annotations

import concurrent.futures
import contextvars
import json
import queue
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


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
    if preflight is not None:
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


@pytest.mark.parametrize(
    ("providers_yaml", "expected_interval"),
    [("      custom:pm: 1700\n", 1700), ("      custom: 1700\n", None)],
)
def test_real_config_binding_and_worker_preflight_before_spawn(
    monkeypatch, tmp_path, providers_yaml, expected_interval
):
    from tools.runtime_heartbeat import (
        bind_agent_provider,
        reset_current_provider,
        runtime_heartbeat,
    )
    from tools.thread_context import propagate_context_to_thread

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "runtime:\n"
        "  heartbeat:\n"
        "    enabled: true\n"
        "    mode: per_target\n"
        "  warm_kv_timeout:\n"
        "    providers:\n"
        f"{providers_yaml}",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    arm = MagicMock(return_value=True)
    monkeypatch.setattr(runtime_heartbeat, "arm", arm)
    agent = SimpleNamespace(
        provider="custom",
        requested_provider="custom:pm",
        base_url="https://pm.invalid/v1",
        model="gpt-5.4-mini",
    )
    token = bind_agent_provider(agent)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result, registry, proc = executor.submit(
                propagate_context_to_thread(
                    lambda: _run(monkeypatch, preflight=None)
                )
            ).result(timeout=5)
    finally:
        reset_current_provider(token)

    if expected_interval is None:
        assert "custom:pm" in result["error"]
        registry.spawn_local.assert_not_called()
        arm.assert_not_called()
    else:
        assert result["session_id"] == proc.id
        registry.spawn_local.assert_called_once()
        assert arm.call_args.kwargs["interval"] == expected_interval


@pytest.mark.parametrize(
    ("provider", "requested_provider", "base_url"),
    [
        ("custom", "custom:pm", "https://pm.invalid/v1"),
        ("openai-codex", "openai-codex", "https://chatgpt.com/backend-api/codex"),
    ],
)
def test_agent_tool_boundary_binds_exact_provider_for_running_process(
    monkeypatch, tmp_path, provider, requested_provider, base_url
):
    from run_agent import AIAgent
    from tools.process_registry import process_registry
    from tools.runtime_heartbeat import (
        RuntimeHeartbeat,
        canonical_runtime_cache_context_identity,
        get_current_cache_context,
        get_current_provider,
    )

    class FakeTimer:
        def __init__(self, interval, callback):
            self.interval = interval
            self.callback = callback
            self.daemon = False
            self.cancelled = False

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "runtime:\n"
        "  heartbeat:\n"
        "    enabled: true\n"
        "    mode: per_target\n"
        "  warm_kv_timeout:\n"
        "    providers:\n"
        "      custom:pm: 1700\n"
        "      openai-codex: 1700\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.terminal_tool._get_env_config", _config)
    monkeypatch.setattr("tools.terminal_tool._start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        "tools.terminal_tool._check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setattr("tools.terminal_tool._active_environments", {})
    monkeypatch.setattr("tools.terminal_tool._last_activity", {})
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: True
    )

    heartbeat_events = queue.Queue()
    manager = RuntimeHeartbeat(
        event_queue=heartbeat_events,
        timer_factory=FakeTimer,
    )
    arm = MagicMock(wraps=manager.arm)
    monkeypatch.setattr(manager, "arm", arm)
    monkeypatch.setattr("tools.runtime_heartbeat.runtime_heartbeat", manager)

    tool_defs = [{
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "terminal",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI") as openai,
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url=base_url,
            provider=provider,
            requested_provider=requested_provider,
            model="gpt-5.4-mini",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="owner",
        )
    openai.return_value.chat.completions.create.side_effect = AssertionError(
        "tool execution must not call the model provider"
    )
    command = (
        f"{shlex.quote(sys.executable)} -c "
        f"{shlex.quote('import time; time.sleep(0.5)')}"
    )
    tool_call = SimpleNamespace(
        id="terminal-heartbeat",
        function=SimpleNamespace(
            name="terminal",
            arguments=json.dumps({
                "command": command,
                "background": True,
                "notify_on_complete": True,
                "timeout": 5,
            }),
        ),
    )
    messages = []
    session_id = ""
    try:
        execution_context = contextvars.Context()
        execution_context.run(
            agent._execute_tool_calls,
            SimpleNamespace(tool_calls=[tool_call]),
            messages,
            "owner",
        )
        assert execution_context.run(get_current_provider) == ""
        assert execution_context.run(get_current_cache_context) == ""
        result = json.loads(messages[-1]["content"])
        session_id = result["session_id"]

        arm.assert_called_once()
        assert arm.call_args.kwargs["interval"] == 1700
        target = manager._targets[session_id]
        assert target.provider == requested_provider
        assert target.cache_context == canonical_runtime_cache_context_identity(agent)

        target.timer.callback()
        check_in = heartbeat_events.get_nowait()
        assert check_in["heartbeat_interval"] == 1700
        assert check_in["provider"] == requested_provider
        assert heartbeat_events.empty()

        session = process_registry.get(session_id)
        assert session is not None and session._completion_event.wait(timeout=5)
        assert session_id not in manager._targets
    finally:
        if session_id:
            process_registry.kill_process(session_id)
        manager.cancel_all()
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_idle_1900_without_managed_target_stays_unarmed(monkeypatch):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    timer_factory = MagicMock()
    manager = RuntimeHeartbeat(
        event_queue=queue.Queue(),
        timer_factory=timer_factory,
    )
    monkeypatch.setattr("tools.runtime_heartbeat.runtime_heartbeat", manager)

    idle_seconds = 1900

    assert idle_seconds > 1700
    assert manager.outstanding_for_caller("idle-owner") == []
    timer_factory.assert_not_called()
