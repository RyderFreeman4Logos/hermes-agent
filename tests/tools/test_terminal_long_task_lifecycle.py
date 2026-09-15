"""Long-task terminal launch/completion lifecycle (#182).

Moved out of tests/tools/test_notify_on_complete.py so unique-on-pin #182
no longer edits the shared file that #149 already owns on replay.
Fixtures and assertions are preserved from the published #182 tests.
"""

import json

import pytest
from unittest.mock import MagicMock

from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


class TestSpawnLocalNotifyArming:
    def test_spawn_local_queues_notify_when_reader_finishes_immediately(
        self, registry, monkeypatch
    ):
        """A spawn-advertised notification survives an immediate reader exit."""
        import tools.process_registry as pr

        class ImmediateThread:
            def __init__(self, *, target, args, **_kwargs):
                self.target = target
                self.args = args

            def start(self):
                self.target(*self.args)

        def finish(session):
            session.exited = True
            session.exit_code = 0
            session.completion_reason = "exited"
            registry._move_to_finished(session)

        monkeypatch.setattr(pr.subprocess, "Popen", lambda *_args, **_kwargs: MagicMock(pid=123))
        monkeypatch.setattr(pr.threading, "Thread", ImmediateThread)
        monkeypatch.setattr(registry, "_reader_loop", finish)
        monkeypatch.setattr(registry, "_safe_host_start_time", lambda _pid: None)
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

        session = registry.spawn_local("true", cwd="/tmp", notify_on_complete=True)

        assert registry.completion_queue.get_nowait()["session_id"] == session.id


def _silent_bg_base_config(tmp_path, extra=None):
    cfg = {
        "env_type": "local",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "cwd": str(tmp_path),
        "timeout": 180,
        "auto_background_timeout_threshold": 200,
    }
    if extra:
        cfg.update(extra)
    return cfg


def _silent_bg_harness(monkeypatch, tmp_path, extra=None):
    """Common test fixture: patch enough of terminal_tool to spawn a fake
    background process and capture the JSON result the agent sees."""
    import tools.terminal_tool as terminal_tool_module
    from tools import process_registry as process_registry_module
    from types import SimpleNamespace

    config = _silent_bg_base_config(tmp_path, extra)
    spawn_kwargs = []
    dummy_env = SimpleNamespace(
        env={},
        execute=MagicMock(
            return_value={"output": "done", "exit_code": 0, "error": None}
        ),
    )

    def fake_spawn_local(**kwargs):
        spawn_kwargs.append(kwargs)
        return SimpleNamespace(
            id="proc_silent_test",
            pid=4242,
            notify_on_complete=False,
            watcher_platform="",
            watcher_chat_id="",
            watcher_user_id="",
            watcher_user_name="",
            watcher_thread_id="",
            watcher_message_id="",
            watcher_interval=0,
        )

    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool_module, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setattr(process_registry_module.process_registry, "spawn_local", fake_spawn_local)
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)
    terminal_tool_module._test_env = dummy_env
    setattr(terminal_tool_module, "_test_spawn_kwargs", spawn_kwargs)
    return terminal_tool_module


def _call_terminal_handler(monkeypatch, tmp_path, extra=None, **args):
    from gateway import session_context

    tt = _silent_bg_harness(monkeypatch, tmp_path, extra)
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: True)
    try:
        result = json.loads(tt._handle_terminal({"command": "printf done", **args}))
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)
    return tt, result


def test_omitted_background_and_timeout_stay_foreground(monkeypatch, tmp_path):
    tt, result = _call_terminal_handler(monkeypatch, tmp_path)

    assert result["error"] is None
    tt._test_env.execute.assert_called_once()


@pytest.mark.parametrize(
    "args",
    [{"background": True, "notify_on_complete": True}],
)
def test_notify_is_armed_at_spawn(monkeypatch, tmp_path, args):
    tt, _result = _call_terminal_handler(monkeypatch, tmp_path, **args)

    assert getattr(tt, "_test_spawn_kwargs")[-1]["notify_on_complete"] is True


def test_unsupported_immediate_exit_does_not_enqueue_completion(monkeypatch, tmp_path):
    """A refused async promise must not leak an already-completed event."""
    from gateway import session_context
    from tools import process_registry as process_registry_module

    tt = _silent_bg_harness(monkeypatch, tmp_path)
    registry = ProcessRegistry()
    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: False)

    def finish_immediately(**kwargs):
        session = ProcessSession(
            id="proc_unsupported_immediate",
            command=kwargs["command"],
            notify_on_complete=kwargs["notify_on_complete"],
        )
        registry._running[session.id] = session
        session.exited = True
        session.exit_code = 0
        session.completion_reason = "exited"
        registry._move_to_finished(session)
        return session

    monkeypatch.setattr(registry, "spawn_local", finish_immediately)
    try:
        result = json.loads(
            tt.terminal_tool(
                command="printf done", background=True, notify_on_complete=True
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert result["notify_on_complete"] is False
    assert result["notify_unsupported"]
    assert registry.completion_queue.empty()


def test_omitted_background_long_timeout_auto_promotes_before_cap(monkeypatch, tmp_path):
    _tt, result = _call_terminal_handler(monkeypatch, tmp_path, timeout=7200)

    assert result["session_id"] == "proc_silent_test"
    assert result["notify_on_complete"] is True
    assert "Foreground timeout" not in str(result.get("error"))


def test_auto_promotion_forces_explicit_notify_false_to_true(monkeypatch, tmp_path):
    _tt, result = _call_terminal_handler(
        monkeypatch, tmp_path, timeout=7200, notify_on_complete=False
    )

    assert result["session_id"] == "proc_silent_test"
    assert result["notify_on_complete"] is True


def test_explicit_background_false_long_timeout_still_rejected(monkeypatch, tmp_path):
    _tt, result = _call_terminal_handler(
        monkeypatch, tmp_path, timeout=7200, background=False
    )

    assert "Foreground timeout 7200s exceeds the maximum" in result["error"]


def test_omitted_background_short_explicit_timeout_stays_foreground(monkeypatch, tmp_path):
    tt, result = _call_terminal_handler(monkeypatch, tmp_path, timeout=30)

    assert result["error"] is None
    tt._test_env.execute.assert_called_once()


def test_config_default_auto_background_timeout_threshold_is_200():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["terminal"]["auto_background_timeout_threshold"] == 200


def test_auto_background_timeout_threshold_loader_defaults_to_200(monkeypatch):
    import tools.terminal_tool as terminal_tool_module

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"terminal": {}},
    )
    assert terminal_tool_module._auto_background_timeout_threshold() == 200


def test_omitted_background_timeout_exactly_at_threshold_stays_foreground(
    monkeypatch, tmp_path
):
    tt, result = _call_terminal_handler(monkeypatch, tmp_path, timeout=200)

    assert result["error"] is None
    assert result.get("session_id") is None
    tt._test_env.execute.assert_called_once()


@pytest.mark.parametrize("timeout", [201, 300, 600])
def test_omitted_background_timeout_above_threshold_auto_promotes(
    monkeypatch, tmp_path, timeout
):
    tt, result = _call_terminal_handler(monkeypatch, tmp_path, timeout=timeout)

    assert result.get("session_id") == "proc_silent_test"
    assert result.get("notify_on_complete") is True
    assert not tt._test_env.execute.called


def test_custom_threshold_just_over_auto_promotes(monkeypatch, tmp_path):
    tt, result = _call_terminal_handler(
        monkeypatch,
        tmp_path,
        extra={"auto_background_timeout_threshold": 100},
        timeout=150,
    )

    assert result.get("session_id") == "proc_silent_test"
    assert result.get("notify_on_complete") is True
    assert not tt._test_env.execute.called


def test_custom_threshold_exact_stays_foreground(monkeypatch, tmp_path):
    tt, result = _call_terminal_handler(
        monkeypatch,
        tmp_path,
        extra={"auto_background_timeout_threshold": 100},
        timeout=100,
    )

    assert result["error"] is None
    assert result.get("session_id") is None
    tt._test_env.execute.assert_called_once()


def test_custom_threshold_at_official_cap_keeps_below_foreground(monkeypatch, tmp_path):
    tt, result = _call_terminal_handler(
        monkeypatch,
        tmp_path,
        extra={"auto_background_timeout_threshold": 600},
        timeout=300,
    )

    assert result["error"] is None
    assert result.get("session_id") is None
    tt._test_env.execute.assert_called_once()
