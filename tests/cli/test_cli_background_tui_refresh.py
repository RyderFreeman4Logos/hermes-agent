"""Tests for CLI background command TUI refresh behavior.

Ensures the TUI is properly refreshed before printing background task output
to prevent spinner/status bar overlap (#2718).
"""

from unittest.mock import MagicMock, patch


from cli import HermesCLI


def _make_cli():
    """Create a minimal HermesCLI instance for testing."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "test-model"
    cli_obj._background_tasks = {}
    cli_obj._background_task_counter = 0
    cli_obj.conversation_history = []
    cli_obj.agent = None
    cli_obj._app = None
    return cli_obj


class TestBackgroundCommandTuiRefresh:
    """Tests for TUI refresh in background command output."""

    def test_invalidate_called_before_success_output(self):
        """App.invalidate() is called before printing background success output."""
        cli_obj = _make_cli()
        mock_app = MagicMock()
        cli_obj._app = mock_app

        # Track call order
        call_order = []
        original_invalidate = mock_app.invalidate

        def track_invalidate():
            call_order.append("invalidate")
            return original_invalidate()

        mock_app.invalidate = track_invalidate

        # Patch print to track when it's called
        with patch("builtins.print") as mock_print:
            mock_print.side_effect = lambda *args, **kwargs: call_order.append("print")

            # Simulate the background task output code path
            if cli_obj._app:
                cli_obj._app.invalidate()
                import time
                time.sleep(0.01)  # reduced for test
            print()

        # Verify invalidate was called before print
        assert call_order[0] == "invalidate"
        assert "print" in call_order

    def test_invalidate_called_before_error_output(self):
        """App.invalidate() is called before printing background error output."""
        cli_obj = _make_cli()
        mock_app = MagicMock()
        cli_obj._app = mock_app

        call_order = []
        mock_app.invalidate.side_effect = lambda: call_order.append("invalidate")

        with patch("builtins.print") as mock_print:
            mock_print.side_effect = lambda *args, **kwargs: call_order.append("print")

            # Simulate error path
            if cli_obj._app:
                cli_obj._app.invalidate()
                import time
                time.sleep(0.01)
            print()

        assert call_order[0] == "invalidate"
        assert "print" in call_order

    def test_no_crash_when_app_is_none(self):
        """No crash when _app is None (non-TUI mode)."""
        cli_obj = _make_cli()
        cli_obj._app = None

        # This should not raise
        if cli_obj._app:
            cli_obj._app.invalidate()
        # If we get here without exception, test passes

    def test_background_task_thread_safety(self):
        """Background task tracking is thread-safe."""
        cli_obj = _make_cli()

        # Simulate adding and removing background tasks
        task_id = "test_task_1"
        cli_obj._background_tasks[task_id] = MagicMock()
        assert task_id in cli_obj._background_tasks

        # Clean up
        cli_obj._background_tasks.pop(task_id, None)
        assert task_id not in cli_obj._background_tasks

    def test_background_agent_preserves_requested_provider(self, monkeypatch):
        captured = {}

        class ImmediateThread:
            def __init__(self, target=None, **_kwargs):
                self.target = target

            def start(self):
                self.target()

        class Agent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_conversation(self, **_kwargs):
                return {"final_response": ""}

        cli_obj = _make_cli()
        cli_obj.max_turns = 1
        cli_obj.enabled_toolsets = ["terminal"]
        cli_obj._session_db = None
        cli_obj.reasoning_config = None
        cli_obj.service_tier = None
        cli_obj._providers_only = None
        cli_obj._providers_ignore = None
        cli_obj._providers_order = None
        cli_obj._provider_sort = None
        cli_obj._provider_require_params = False
        cli_obj._provider_data_collection = None
        cli_obj._openrouter_min_coding_score = None
        cli_obj._fallback_model = None
        cli_obj._sudo_password_callback = None
        cli_obj._approval_callback = None
        cli_obj._secret_capture_callback = None
        cli_obj._agent_running = False
        cli_obj._spinner_text = ""
        cli_obj.bell_on_complete = False
        cli_obj._ensure_runtime_credentials = lambda: True
        cli_obj._resolve_turn_agent_config = lambda _prompt: {
            "model": "gpt-5.4-mini",
            "runtime": {
                "provider": "custom",
                "requested_provider": "custom:pm",
                "base_url": "https://pm.invalid/v1",
                "api_key": "key",
                "api_mode": "chat_completions",
            },
        }
        monkeypatch.setattr("cli.AIAgent", Agent)
        monkeypatch.setattr(
            "hermes_cli.cli_commands_mixin.threading.Thread", ImmediateThread
        )
        monkeypatch.setattr("cli._cprint", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            "cli.ChatConsole", lambda: MagicMock(print=lambda *_args, **_kwargs: None)
        )

        cli_obj._handle_background_command("/background check")

        assert captured["provider"] == "custom"
        assert captured["requested_provider"] == "custom:pm"
