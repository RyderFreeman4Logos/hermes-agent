"""Tests for foreground timeout cap in terminal_tool.

Ensures that:
- with auto_background_long_timeout=false, timeout > FOREGROUND_MAX is rejected
- with auto true, long timeouts promote to background (covered in
  test_terminal_auto_background_long_timeout.py)
- short timeouts still execute in foreground
"""
import json
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Shared test config dict — mirrors _get_env_config() return shape.
# ---------------------------------------------------------------------------
def _make_env_config(**overrides):
    """Return a minimal _get_env_config()-shaped dict with optional overrides."""
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        # Legacy reject path tests opt into auto=false; promotion covered elsewhere.
        "auto_background_long_timeout": False,
        "auto_background_timeout_threshold": 200,
        "default_notify_on_background": True,
    }
    config.update(overrides)
    return config


class TestForegroundTimeoutCap:
    """FOREGROUND_MAX_TIMEOUT rejects foreground commands that exceed it (auto off)."""

    def test_foreground_timeout_rejected_above_max(self):
        """When auto is false and model requests timeout > max, return error."""
        from tools.terminal_tool import terminal_tool, FOREGROUND_MAX_TIMEOUT

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            result = json.loads(terminal_tool(
                command="echo hello",
                timeout=9999,  # Way above max
            ))

        assert "error" in result
        assert "9999" in result["error"]
        assert str(FOREGROUND_MAX_TIMEOUT) in result["error"]
        assert "background=true" in result["error"]


    def test_foreground_allows_help_variant_for_server_command(self):
        """Informational variants like '--help' should not be blocked."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "usage", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(terminal_tool(command="pnpm dev --help"))

        assert result["error"] is None
        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[0][0] == "pnpm dev --help"


    def test_config_default_above_cap_is_rejected_when_auto_false(self):
        """The cap applies to default-derived effective foreground timeouts too."""
        from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT, terminal_tool

        # User configured TERMINAL_TIMEOUT=900 in their env; auto promotion off.
        with patch("tools.terminal_tool._get_env_config",
                   return_value=_make_env_config(
                       timeout=900,
                       auto_background_long_timeout=False,
                   )), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "done", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(
                    terminal_tool(command="make build", background=False)
                )

        mock_env.execute.assert_not_called()
        assert "error" in result
        assert "900" in result["error"]
        assert str(FOREGROUND_MAX_TIMEOUT) in result["error"]

    def test_background_not_rejected(self):
        """Background commands should NOT be subject to foreground timeout cap."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.env = {}
            mock_proc_session = MagicMock()
            mock_proc_session.id = "test-123"
            mock_proc_session.pid = 1234

            mock_registry = MagicMock()
            mock_registry.spawn_local.return_value = mock_proc_session

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}), \
                 patch("tools.process_registry.process_registry", mock_registry), \
                 patch("tools.approval.get_current_session_key", return_value=""):
                result = json.loads(terminal_tool(
                    command="python server.py",
                    background=True,
                    timeout=9999,
                    notify_on_complete=False,
                ))

        # Background should NOT be rejected
        assert "error" not in result or result["error"] is None

    def test_short_default_timeout_not_rejected(self):
        """Short config default timeout should not trigger rejection."""
        from tools.terminal_tool import terminal_tool, FOREGROUND_MAX_TIMEOUT

        # 180 < 600, so no rejection
        assert 180 < FOREGROUND_MAX_TIMEOUT

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "done", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(terminal_tool(command="echo hello"))

        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[1]["timeout"] == 180
        assert "error" not in result or result["error"] is None

    def test_exactly_at_max_not_rejected(self):
        """Timeout exactly at FOREGROUND_MAX_TIMEOUT should execute normally."""
        from tools.terminal_tool import terminal_tool, FOREGROUND_MAX_TIMEOUT

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "done", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(terminal_tool(
                    command="echo hello",
                    timeout=FOREGROUND_MAX_TIMEOUT,  # Exactly at limit
                ))

        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[1]["timeout"] == FOREGROUND_MAX_TIMEOUT
        assert "error" not in result or result["error"] is None


class TestForegroundMaxTimeoutConstant:
    """Verify the FOREGROUND_MAX_TIMEOUT constant and schema."""

    def test_default_value_is_600(self):
        """Default FOREGROUND_MAX_TIMEOUT is 600 when env var is not set."""
        from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT
        assert FOREGROUND_MAX_TIMEOUT == 600

    def test_schema_mentions_max(self):
        """Tool schema description should mention the max timeout / promote path."""
        from tools.terminal_tool import TERMINAL_SCHEMA, FOREGROUND_MAX_TIMEOUT
        timeout_desc = TERMINAL_SCHEMA["parameters"]["properties"]["timeout"]["description"]
        assert str(FOREGROUND_MAX_TIMEOUT) in timeout_desc
        assert "background" in timeout_desc.lower()
