"""Foreground-release regressions for long terminal calls."""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
import threading
import time
from pathlib import Path
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


def _session(**overrides):
    session = SimpleNamespace(
        id="proc_auto_bg",
        pid=4242,
        command="make build",
        cwd="/tmp",
        output_buffer="",
        exit_code=0,
        _lock=threading.Lock(),
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
    for key, value in overrides.items():
        setattr(session, key, value)
    return session


def _run(
    *, command="make build", config=None, async_delivery=True,
    fresh_environment=False, proc=None, **kwargs,
):
    from tools.terminal_tool import terminal_tool

    env = MagicMock(env={})
    env.cwd = (config or _config())["cwd"]
    env.execute.return_value = {"output": "foreground", "returncode": 0}
    proc = proc or _session()
    registry = MagicMock(pending_watchers=[])
    def fake_spawn(**spawn_kwargs):
        for key, value in spawn_kwargs.get("notification_metadata", {}).items():
            setattr(proc, key, value)
        return proc

    registry.spawn_local.side_effect = fake_spawn
    registry.spawn_via_env.side_effect = fake_spawn

    def fake_promote(session, notification_metadata=None):
        for key, value in (notification_metadata or {}).items():
            setattr(session, key, value)
        return True

    registry.promote.side_effect = fake_promote
    registry.discard.return_value = {"status": "killed"}
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


@pytest.fixture
def terminal_runtime(monkeypatch):
    import tools.async_delegation as async_delegation
    import tools.process_registry as process_module
    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        async_delegation, "restore_undelivered_completions", lambda _queue: 0
    )
    registry = process_module.ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(process_module, "process_registry", registry)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: _config(timeout=2, auto_background_timeout_threshold=1),
    )
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {
            "default": SimpleNamespace(
                env={},
                cwd="/tmp",
                execute=lambda *_args, **_kwargs: {
                    "output": "short",
                    "returncode": 0,
                },
            )
        },
    )
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0})
    monkeypatch.setattr(
        terminal_tool,
        "_create_environment",
        lambda **_kwargs: pytest.fail("cached local environment should be reused"),
    )
    monkeypatch.setattr("tools.approval.get_current_session_key", lambda default="": "")
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: True
    )
    monkeypatch.setattr(
        "gateway.session_context.get_session_env", lambda *_args, **_kwargs: ""
    )
    monkeypatch.setattr(
        "tools.runtime_heartbeat.preflight_current_heartbeat", lambda: None
    )
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.arm",
        lambda *_args, **_kwargs: False,
    )
    return terminal_tool, registry


def _wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _pid_is_running(pid):
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or stat.read_text(encoding="utf-8").split()[2] != "Z"
    except OSError:
        return False


def test_omitted_long_budget_is_promoted_before_spawn_without_threshold_wait():
    result, env, proc, registry, create_environment = _run(
        config=_config(timeout=21),
        fresh_environment=True,
    )

    assert result["session_id"] == proc.id
    assert result["notify_on_complete"] is True
    assert create_environment.call_args.kwargs["timeout"] == 7200
    assert registry.spawn_local.call_args.kwargs["execution_timeout"] == 7200
    assert registry.spawn_local.call_args.kwargs["notification_metadata"][
        "notify_on_complete"
    ] is True
    registry.wait_for_promotion.assert_not_called()
    registry.promote.assert_not_called()
    env.execute.assert_not_called()


@pytest.mark.parametrize("task_id", ["main", "nested-subagent"])
def test_model_handler_preserves_omitted_flags_for_every_caller(monkeypatch, task_id):
    import tools.terminal_tool as terminal_module

    call = MagicMock(return_value="{}")
    monkeypatch.setattr(terminal_module, "terminal_tool", call)

    terminal_module._handle_terminal(
        {"command": "make build", "timeout": 21},
        task_id=task_id,
    )

    assert call.call_args.kwargs["task_id"] == task_id
    assert call.call_args.kwargs["background"] is None
    assert call.call_args.kwargs["notify_on_complete"] is None


@pytest.mark.parametrize("requested_timeout", [2, None], ids=["requested", "configured"])
def test_explicit_notify_without_background_stays_inline(
    terminal_runtime, requested_timeout
):
    terminal_tool, registry = terminal_runtime
    kwargs = {"timeout": requested_timeout} if requested_timeout is not None else {}

    result = json.loads(
        terminal_tool.terminal_tool(
            command="printf short",
            notify_on_complete=True,
            **kwargs,
        )
    )

    assert result["output"].endswith("short")
    assert result["exit_code"] == 0
    assert "session_id" not in result
    assert registry.list_sessions() == []
    assert registry.completion_queue.empty()


@pytest.mark.parametrize(
    ("config", "kwargs"),
    [
        pytest.param(_config(timeout=21), {}, id="omitted-timeout-and-flags"),
        pytest.param(_config(timeout=5), {"timeout": 21}, id="explicit-timeout"),
    ],
)
def test_long_foreground_calls_promote_to_managed_background(config, kwargs):
    result, env, proc, registry, _ = _run(config=config, **kwargs)

    assert result["session_id"] == proc.id
    assert result["notify_on_complete"] is True
    assert proc.notify_on_complete is True
    registry.spawn_local.assert_called_once()
    assert registry.spawn_local.call_args.kwargs["execution_timeout"] == 7200
    registry.wait_for_promotion.assert_not_called()
    registry.promote.assert_not_called()
    env.execute.assert_not_called()


def test_auto_promotion_rewrites_execution_deadline_to_7200():
    result, _, _, registry, create_environment = _run(
        timeout=21,
        fresh_environment=True,
    )

    assert result["session_id"] == "proc_auto_bg"
    assert create_environment.call_args.kwargs["timeout"] == 7200
    assert registry.spawn_local.call_args.kwargs["execution_timeout"] == 7200
    registry.spawn_local.assert_called_once()


def test_auto_promotion_carries_deadline_to_remote_spawn():
    result, _, _, registry, _ = _run(
        config=_config(env_type="docker", timeout=21),
        timeout=21,
    )

    assert result["session_id"] == "proc_auto_bg"
    assert registry.spawn_via_env.call_args.kwargs["execution_timeout"] == 7200
    assert "defer_registration" not in registry.spawn_via_env.call_args.kwargs


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-tree regression")
def test_promoted_execution_deadline_kills_process_tree_once(
    terminal_runtime, monkeypatch, tmp_path
):
    terminal_tool, registry = terminal_runtime
    monkeypatch.setattr(terminal_tool, "AUTO_BACKGROUND_TIMEOUT", 1)
    child_pid_path = tmp_path / "child.pid"
    child_code = "import time; time.sleep(30)"
    parent_code = (
        "import pathlib,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid)); "
        "print('parent-ready', flush=True); time.sleep(30)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_code)}"

    result = json.loads(
        terminal_tool.terminal_tool(command=command, timeout=2)
    )
    session_id = result["session_id"]
    child_pid = None
    try:
        assert _wait_until(child_pid_path.exists)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert _pid_is_running(child_pid)
        session = registry._running.get(session_id) or registry._finished[session_id]
        assert session._completion_event.wait(5)
        assert _wait_until(lambda: not _pid_is_running(child_pid))
        assert registry.poll(session_id)["status"] == "exited"

        events = []
        while not registry.completion_queue.empty():
            events.append(registry.completion_queue.get_nowait())
        matching = [event for event in events if event.get("session_id") == session_id]
        assert len(matching) == 1
        assert matching[0]["completion_reason"] == "killed"
        assert matching[0]["termination_source"] == "execution_timeout"
        assert "parent-ready" in matching[0]["output"]
    finally:
        if session_id in registry._running:
            registry.kill_process(session_id)
        if child_pid is not None and _pid_is_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_completion_before_spawn_returns_keeps_background_notification(
    terminal_runtime, monkeypatch
):
    terminal_tool, registry = terminal_runtime
    original_spawn = registry.spawn_local

    def complete_before_return(**kwargs):
        session = original_spawn(**kwargs)
        assert session._completion_event.wait(3)
        return session

    monkeypatch.setattr(registry, "spawn_local", complete_before_return)
    result = json.loads(terminal_tool.terminal_tool(command="true", timeout=2))

    assert result["exit_code"] == 0
    assert result["session_id"]
    assert result["notify_on_complete"] is True
    events = []
    while not registry.completion_queue.empty():
        events.append(registry.completion_queue.get_nowait())
    assert len(
        [event for event in events if event.get("session_id") == result["session_id"]]
    ) == 1


@pytest.mark.parametrize(
    "explicit_flags",
    [
        pytest.param({"background": False}, id="background-false"),
        pytest.param({"notify_on_complete": False}, id="notify-false"),
        pytest.param(
            {"background": False, "notify_on_complete": False},
            id="both-false",
        ),
    ],
)
def test_explicit_false_flags_are_not_auto_promoted(explicit_flags):
    result, env, _, registry, _ = _run(timeout=21, **explicit_flags)

    assert result["output"] == "foreground"
    env.execute.assert_called_once()
    registry.spawn_local.assert_not_called()


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
    assert "execution_timeout" not in registry.spawn_local.call_args.kwargs
    assert "defer_registration" not in registry.spawn_local.call_args.kwargs
    env.execute.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 21, "command": "python -m http.server", "background": False},
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
def test_auto_promotion_stops_without_completion_route(async_delivery):
    result, env, _, registry, _ = _run(
        timeout=21,
        async_delivery=async_delivery,
    )

    assert "not started" in result["error"].lower()
    env.execute.assert_not_called()
    registry.spawn_local.assert_not_called()
    registry.wait_for_promotion.assert_not_called()
    registry.discard.assert_not_called()
    registry.promote.assert_not_called()


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


def test_hot_reload_does_not_change_inflight_promotion_threshold():
    class ReloadingConfig(dict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if key == "auto_background_timeout_threshold":
                self[key] = 1
            return value

    config = ReloadingConfig(_config())
    _, _, _, registry, _ = _run(config=config, timeout=21)

    registry.wait_for_promotion.assert_not_called()
    assert config["auto_background_timeout_threshold"] == 1


def test_terminal_schema_describes_auto_promotion():
    from hermes_cli.config import DEFAULT_CONFIG
    from tools.terminal_tool import TERMINAL_SCHEMA, TERMINAL_TOOL_DESCRIPTION

    description = TERMINAL_SCHEMA["parameters"]["properties"]["timeout"]["description"]
    assert DEFAULT_CONFIG["terminal"]["auto_background_timeout_threshold"] == 200
    assert "auto_background_timeout_threshold" in description
    assert "before execution" in description
    assert "7200" in description
    assert "auto_background_timeout_threshold" in TERMINAL_TOOL_DESCRIPTION
