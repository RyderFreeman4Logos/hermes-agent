"""TDD: omitted long timeouts run background but sync-wait by default.

Authoritative contract (/tmp/hermes-terminal-sync-wait-on-default-notify-contract.md):
- Explicit background=false is always respected (even timeout > 200).
- Explicit notify_on_complete=false is always respected.
- Defaults apply only when params are omitted (None / absent).
- Omitted timeout → config timeout (3300) → if effective > 200 and master on
  and background omitted → background=true; if notify omitted and no
  watch_patterns → notify_on_complete=true and the tool waits for the final
  background-session result.
- timeout == 200 stays foreground (not strictly greater).
- FOREGROUND_MAX hard-reject only when final background is false and the
  model explicitly requested timeout > FOREGROUND_MAX.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _base_config(**overrides):
    config = {
        "env_type": "local",
        "timeout": 3300,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "auto_background_long_timeout": True,
        "auto_background_timeout_threshold": 200,
        "default_notify_on_background": True,
    }
    config.update(overrides)
    return config


def _fake_proc_session(sid: str = "proc_auto_bg"):
    return SimpleNamespace(
        id=sid,
        pid=4242,
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


def _run_promoted(command: str, **tool_kwargs):
    """Run terminal_tool with local spawn mocked; return (result, mock_proc, mock_env, registry)."""
    from tools.terminal_tool import terminal_tool

    mock_proc = _fake_proc_session()
    mock_registry = MagicMock()
    mock_registry.spawn_local.return_value = mock_proc
    wait_result = tool_kwargs.pop("wait_result", None)
    mock_registry.wait.return_value = wait_result or {
        "status": "exited",
        "command": command,
        "exit_code": 0,
        "completion_reason": "exited",
        "termination_source": "",
        "output": "completed output",
    }
    mock_env = MagicMock()
    mock_env.env = {}

    cfg = tool_kwargs.pop("config", None) or _base_config()
    async_delivery_supported = tool_kwargs.pop("async_delivery_supported", True)
    with (
        patch("tools.terminal_tool._get_env_config", return_value=cfg),
        patch("tools.terminal_tool._start_cleanup_thread"),
        patch(
            "tools.terminal_tool._check_all_guards",
            return_value={"approved": True},
        ),
        patch("tools.terminal_tool._active_environments", {"default": mock_env}),
        patch("tools.terminal_tool._last_activity", {"default": 0}),
        patch("tools.process_registry.process_registry", mock_registry),
        patch("tools.approval.get_current_session_key", return_value=""),
        patch(
            "gateway.session_context.async_delivery_supported",
            return_value=async_delivery_supported,
        ),
    ):
        result = json.loads(terminal_tool(command=command, **tool_kwargs))

    return result, mock_proc, mock_env, mock_registry


def _run_foreground(command: str, **tool_kwargs):
    """Run terminal_tool with local execute mocked; return (result, mock_env)."""
    from tools.terminal_tool import terminal_tool

    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "done", "returncode": 0}

    cfg = tool_kwargs.pop("config", None) or _base_config()
    with (
        patch("tools.terminal_tool._get_env_config", return_value=cfg),
        patch("tools.terminal_tool._start_cleanup_thread"),
        patch(
            "tools.terminal_tool._check_all_guards",
            return_value={"approved": True},
        ),
        patch("tools.terminal_tool._active_environments", {"default": mock_env}),
        patch("tools.terminal_tool._last_activity", {"default": 0}),
    ):
        result = json.loads(terminal_tool(command=command, **tool_kwargs))

    return result, mock_env


class TestConfigDefaults:
    def test_default_config_timeout_is_3300(self):
        from hermes_cli.config import DEFAULT_CONFIG

        term = DEFAULT_CONFIG["terminal"]
        assert term["timeout"] == 3300
        assert term["auto_background_timeout_threshold"] == 200
        assert term["auto_background_long_timeout"] is True
        assert term["default_notify_on_background"] is True

    def test_get_env_config_defaults(self, monkeypatch):
        import tools.terminal_tool as tt

        # Skip bridge so bare env defaults are exercised without config.yaml I/O.
        monkeypatch.setattr(tt, "_ensure_terminal_env_bridged", lambda: None)
        for key in (
            "TERMINAL_TIMEOUT",
            "TERMINAL_AUTO_BACKGROUND_LONG_TIMEOUT",
            "TERMINAL_AUTO_BACKGROUND_TIMEOUT_THRESHOLD",
            "TERMINAL_DEFAULT_NOTIFY_ON_BACKGROUND",
            "TERMINAL_ENV",
        ):
            monkeypatch.delenv(key, raising=False)

        cfg = tt._get_env_config()
        assert cfg["timeout"] == 3300
        assert cfg["auto_background_timeout_threshold"] == 200
        assert cfg["auto_background_long_timeout"] is True
        assert cfg["default_notify_on_background"] is True


class TestAutoBackgroundLongTimeout:
    def test_omitted_all_promotes_with_notify_and_sync_waits(self):
        """Default-notify auto-promotion waits for the background result."""
        result, mock_proc, mock_env, mock_registry = _run_promoted(
            "make build",
            config=_base_config(timeout=3300, auto_background_long_timeout=True),
            # timeout / background / notify all omitted
        )

        assert "error" not in result or result["error"] is None
        assert result.get("session_id") == mock_proc.id
        assert result.get("notify_on_complete") is True
        assert result["status"] == "exited"
        assert result["output"] == "completed output"
        assert result["output"] != "Background process started"
        assert mock_proc.notify_on_complete is True
        mock_registry.spawn_local.assert_called_once()
        mock_registry.wait.assert_called_once_with(mock_proc.id, timeout=3300)
        mock_env.execute.assert_not_called()

    def test_default_notify_auto_promotion_waits_when_async_delivery_is_unavailable(self):
        """Inline waiting does not depend on a later async completion route."""
        result, mock_proc, _, mock_registry = _run_promoted(
            "make build",
            timeout=201,
            async_delivery_supported=False,
        )

        assert result["status"] == "exited"
        assert result["session_id"] == mock_proc.id
        mock_registry.wait.assert_called_once_with(mock_proc.id, timeout=3300)

    def test_timeout_201_background_omitted_rewrites_to_default_and_waits(self):
        """Auto-promotion rewrites a model's 201s timeout to the 3300s default."""
        result, mock_proc, mock_env, mock_registry = _run_promoted(
            "echo hello",
            timeout=201,
            # background omitted (None)
        )

        assert "error" not in result or result["error"] is None
        assert result.get("session_id") == mock_proc.id
        assert result.get("notify_on_complete") is True
        assert result["status"] == "exited"
        assert mock_proc.notify_on_complete is True
        mock_registry.spawn_local.assert_called_once()
        mock_registry.wait.assert_called_once_with(mock_proc.id, timeout=3300)
        mock_env.execute.assert_not_called()

    def test_promoted_explicit_notify_true_rewrites_7200_and_waits(self):
        """Explicit notify=True has the same sync-wait semantics after promotion."""
        result, mock_proc, mock_env, mock_registry = _run_promoted(
            "make build",
            timeout=7200,
            notify_on_complete=True,
        )

        assert result.get("session_id") == mock_proc.id
        assert result.get("notify_on_complete") is True
        assert result["status"] == "exited"
        assert result["output"] == "completed output"
        mock_registry.spawn_local.assert_called_once()
        mock_registry.wait.assert_called_once_with(mock_proc.id, timeout=3300)
        mock_env.execute.assert_not_called()

    def test_sync_wait_interrupt_keeps_auto_promoted_process_background(self):
        """Steering stops the wait, never the managed background child."""
        result, mock_proc, mock_env, mock_registry = _run_promoted(
            "make build",
            timeout=201,
            wait_result={
                "status": "interrupted",
                "command": "make build",
                "output": "partial output",
                "note": "User sent a new message -- wait interrupted",
            },
        )

        assert result["status"] == "interrupted"
        assert result["session_id"] == mock_proc.id
        assert result["output"] == "partial output"
        mock_registry.wait.assert_called_once_with(mock_proc.id, timeout=3300)
        mock_registry.kill_process.assert_not_called()
        mock_env.execute.assert_not_called()

    def test_timeout_201_explicit_background_false_stays_foreground(self):
        """Explicit background=False is respected even when timeout > 200."""
        result, mock_env = _run_foreground(
            "echo hello",
            timeout=201,
            background=False,
        )

        call_kwargs = mock_env.execute.call_args
        assert call_kwargs is not None
        assert call_kwargs[1]["timeout"] == 201
        assert "error" not in result or result["error"] is None
        assert "session_id" not in result

    def test_timeout_200_stays_foreground(self):
        """Boundary: timeout == threshold stays foreground (not strictly greater)."""
        result, mock_env = _run_foreground(
            "echo hello",
            timeout=200,
            # background omitted — still fg because not > 200
        )

        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[1]["timeout"] == 200
        assert "error" not in result or result["error"] is None
        assert "session_id" not in result

    def test_timeout_180_stays_foreground(self):
        result, mock_env = _run_foreground(
            "echo hello",
            timeout=180,
            config=_base_config(timeout=180),
        )

        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[1]["timeout"] == 180
        assert "error" not in result or result["error"] is None
        assert "session_id" not in result

    def test_auto_false_explicit_timeout_above_max_still_errors(self):
        """Legacy reject path when auto_background_long_timeout is false."""
        from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT, terminal_tool

        with (
            patch(
                "tools.terminal_tool._get_env_config",
                return_value=_base_config(auto_background_long_timeout=False),
            ),
            patch("tools.terminal_tool._start_cleanup_thread"),
        ):
            result = json.loads(
                terminal_tool(
                    command="echo hello",
                    timeout=9999,
                    # background omitted → false after defaults
                )
            )

        assert "error" in result
        assert "9999" in result["error"]
        assert str(FOREGROUND_MAX_TIMEOUT) in result["error"]
        assert "background=true" in result["error"]

    def test_explicit_background_false_above_foreground_max_still_errors(self):
        """Explicit fg + timeout > FOREGROUND_MAX rejects even when auto is on."""
        from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT, terminal_tool

        with (
            patch(
                "tools.terminal_tool._get_env_config",
                return_value=_base_config(auto_background_long_timeout=True),
            ),
            patch("tools.terminal_tool._start_cleanup_thread"),
        ):
            result = json.loads(
                terminal_tool(
                    command="echo hello",
                    timeout=FOREGROUND_MAX_TIMEOUT + 1,
                    background=False,
                )
            )

        assert "error" in result
        assert str(FOREGROUND_MAX_TIMEOUT + 1) in result["error"]
        assert str(FOREGROUND_MAX_TIMEOUT) in result["error"]

    def test_auto_true_timeout_above_foreground_max_omitted_bg_promotes(self):
        """When auto is on and background is omitted, timeout>FOREGROUND_MAX promotes."""
        from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT

        result, mock_proc, mock_env, mock_registry = _run_promoted(
            "echo hello",
            timeout=FOREGROUND_MAX_TIMEOUT + 1,
            # background omitted
        )

        assert "error" not in result or result["error"] is None
        assert result.get("session_id") == mock_proc.id
        assert result.get("notify_on_complete") is True
        mock_registry.spawn_local.assert_called_once()
        mock_env.execute.assert_not_called()


class TestDefaultNotifyOnBackground:
    def test_background_without_notify_defaults_to_true(self):
        result, mock_proc, _, _ = _run_promoted(
            "pytest tests/",
            background=True,
            # notify_on_complete omitted
        )

        assert result.get("notify_on_complete") is True
        assert mock_proc.notify_on_complete is True
        assert "hint" not in result

    def test_background_explicit_notify_false_is_respected(self):
        result, mock_proc, _, _ = _run_promoted(
            "python server.py",
            background=True,
            notify_on_complete=False,
        )

        assert result.get("notify_on_complete") is not True
        assert mock_proc.notify_on_complete is False
        # Explicit silent bg still gets the silent-process hint.
        assert "hint" in result

    def test_promoted_timeout_with_explicit_notify_false_preserves_false(self):
        """Explicit notify=False is true detach: handle now, no sync wait."""
        result, mock_proc, _, mock_registry = _run_promoted(
            "long task",
            timeout=201,
            # background omitted → promote
            notify_on_complete=False,
        )

        assert result.get("session_id") == mock_proc.id
        assert result["output"] == "Background process started"
        assert result.get("notify_on_complete") is not True
        assert mock_proc.notify_on_complete is False
        mock_registry.wait.assert_not_called()

    def test_background_with_watch_patterns_does_not_force_notify(self):
        result, mock_proc, _, _ = _run_promoted(
            "uvicorn app:server",
            background=True,
            watch_patterns=["Application startup complete"],
        )

        assert result.get("notify_on_complete") is not True
        assert mock_proc.notify_on_complete is False
        assert result.get("watch_patterns") == ["Application startup complete"]


class TestHandlerOmittedSentinels:
    def test_handle_terminal_omitted_notify_is_none_sentinel(self):
        """Omitted notify_on_complete must not collapse to False before defaults."""
        from tools.terminal_tool import _handle_terminal

        with patch(
            "tools.terminal_tool.terminal_tool", return_value='{"ok":true}'
        ) as mock_tt:
            _handle_terminal(
                {"command": "echo hi", "background": True},
                task_id="t1",
            )
            _, kwargs = mock_tt.call_args
            assert kwargs["notify_on_complete"] is None

    def test_handle_terminal_explicit_false_is_false(self):
        from tools.terminal_tool import _handle_terminal

        with patch(
            "tools.terminal_tool.terminal_tool", return_value='{"ok":true}'
        ) as mock_tt:
            _handle_terminal(
                {
                    "command": "echo hi",
                    "background": True,
                    "notify_on_complete": False,
                },
                task_id="t1",
            )
            _, kwargs = mock_tt.call_args
            assert kwargs["notify_on_complete"] is False

    def test_handle_terminal_omitted_background_is_none_sentinel(self):
        """Omitted background must be None so auto-promote can distinguish it."""
        from tools.terminal_tool import _handle_terminal

        with patch(
            "tools.terminal_tool.terminal_tool", return_value='{"ok":true}'
        ) as mock_tt:
            _handle_terminal({"command": "echo hi"}, task_id="t1")
            _, kwargs = mock_tt.call_args
            assert kwargs["background"] is None

    def test_handle_terminal_explicit_background_false_is_false(self):
        from tools.terminal_tool import _handle_terminal

        with patch(
            "tools.terminal_tool.terminal_tool", return_value='{"ok":true}'
        ) as mock_tt:
            _handle_terminal(
                {"command": "echo hi", "background": False},
                task_id="t1",
            )
            _, kwargs = mock_tt.call_args
            assert kwargs["background"] is False


class TestSchemaDescriptions:
    def test_schema_mentions_auto_background_and_defaults(self):
        from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT, TERMINAL_SCHEMA

        props = TERMINAL_SCHEMA["parameters"]["properties"]
        timeout_desc = props["timeout"]["description"]
        assert "3300" in timeout_desc
        assert "200" in timeout_desc
        assert str(FOREGROUND_MAX_TIMEOUT) in timeout_desc
        assert "auto" in timeout_desc.lower() or "promote" in timeout_desc.lower()
        assert "inline" in timeout_desc.lower()

        notify_desc = props["notify_on_complete"]["description"]
        assert "default" in notify_desc.lower()
        assert "background" in notify_desc.lower()
        assert "synchronously" in notify_desc.lower()
        assert "not also notified" in notify_desc.lower()

        bg_desc = props["background"]["description"]
        assert "auto" in bg_desc.lower() or "timeout" in bg_desc.lower()
        # Must not claim explicit false is force-promoted
        assert "force" not in bg_desc.lower() or "omit" in bg_desc.lower()
