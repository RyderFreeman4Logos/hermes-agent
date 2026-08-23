"""Tests for /resume status lines going to stderr in quiet mode (#11793).

The fix in cli._init_agent routes three messages to stderr when
``tool_progress_mode == "off"`` (set by ``hermes chat --quiet``):

  * "Session not found: ..."
  * "↻ Resumed session ... (N user messages, M total messages)"
  * "Session ... found but has no messages. Starting fresh."

Interactive mode (tool_progress_mode == "full") still uses ChatConsole.
"""

from datetime import datetime
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


from cli import HermesCLI


def _make_cli(quiet=False, session_id="20260524_111111_xyz", db=None):
    """Build a minimal HermesCLI bound to only what _init_agent needs for
    the resume code path: _resumed, _session_db, conversation_history,
    session_id, and tool_progress_mode."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = session_id
    cli._resumed = True
    cli.conversation_history = []
    cli._session_db = db
    cli.tool_progress_mode = "off" if quiet else "full"
    cli.session_start = datetime.now()
    cli.agent = None
    # We need _init_agent to reach the resume block (line ~4757) but not
    # proceed into actual AIAgent construction. _ensure_runtime_credentials
    # must return True (False returns early at line 4743). _install_tool_callbacks,
    # _ensure_tirith_security are stubbed; the resume block will either return
    # False (session-not-found) or reach the eventual AIAgent() call which
    # we'll let raise — we only check stdout/stderr printed BEFORE that.
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    return cli


class TestResumeQuietStderr:
    @pytest.mark.parametrize("target_mode", ["authoritative", "hybrid"])
    def test_startup_resume_passes_frozen_memory_mode_to_agent(self, target_mode):
        db = MagicMock()
        db.get_session.return_value = {
            "id": "20260524_111111_xyz",
            "model_config": json.dumps({"memory_provider_mode": target_mode}),
        }
        cli = _make_cli(db=db)
        cli.conversation_history = [{"role": "user", "content": "preloaded"}]
        cli.finalize_preloaded_skills = lambda: None
        cli.model = "test-model"
        cli.api_key = "test-key"
        cli.base_url = "https://example.invalid/v1"
        cli.provider = "synthetic"
        cli.requested_provider = "synthetic"
        cli.api_mode = "chat_completions"
        cli.acp_command = None
        cli.acp_args = []
        cli.max_tokens = None
        cli.max_turns = 2
        cli.enabled_toolsets = []
        cli.disabled_toolsets = []
        cli.verbose = False
        cli.system_prompt = ""
        cli.prefill_messages = []
        cli.reasoning_config = None
        cli.service_tier = None
        cli._providers_only = None
        cli._providers_ignore = None
        cli._providers_order = None
        cli._provider_sort = None
        cli._provider_require_params = False
        cli._provider_data_collection = None
        cli._openrouter_min_coding_score = None
        cli._fallback_model = []
        cli.checkpoints_enabled = False
        cli.checkpoint_max_snapshots = 0
        cli.checkpoint_max_total_size_mb = 0
        cli.checkpoint_max_file_size_mb = 0
        cli.pass_session_id = False
        cli.ignore_rules = True
        cli.streaming_enabled = False
        cli._inline_diffs_enabled = False
        cli._pending_title = None
        cli._credential_pool = None
        for name in (
            "_clarify_callback", "_current_reasoning_callback", "_on_thinking",
            "_on_tool_progress", "_on_tool_start", "_on_tool_complete", "_stream_delta",
            "_on_tool_gen_start", "_on_notice", "_on_notice_clear", "_on_reaction",
        ):
            setattr(cli, name, lambda *args, **kwargs: None)

        built = SimpleNamespace()
        with (
            patch("cli._prepare_deferred_agent_startup"),
            patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build"),
            patch("cli.AIAgent", return_value=built) as constructor,
            patch("agent.credits_tracker.seed_credits_at_session_start"),
        ):
            assert cli._init_agent() is True

        assert constructor.call_args.kwargs["memory_provider_mode_override"] == target_mode

    def test_session_not_found_goes_to_stderr_in_quiet_mode(self, capsys):
        db = MagicMock()
        db.get_session.return_value = None
        cli = _make_cli(quiet=True, db=db)

        with patch("cli._prepare_deferred_agent_startup"):
            result = cli._init_agent()

        captured = capsys.readouterr()
        assert result is False
        # stdout must stay clean
        assert "Session not found" not in captured.out
        # the resume status goes to stderr
        assert "Session not found" in captured.err
        assert "hermes sessions list" in captured.err

    def test_session_not_found_goes_to_stdout_in_full_mode(self, capsys):
        db = MagicMock()
        db.get_session.return_value = None
        cli = _make_cli(quiet=False, db=db)

        with patch("cli._prepare_deferred_agent_startup"):
            result = cli._init_agent()

        captured = capsys.readouterr()
        assert result is False
        # Interactive mode keeps the existing _cprint path → stdout.
        assert "Session not found" in captured.out

    def test_resumed_banner_goes_to_stderr_in_quiet_mode(self, capsys):
        db = MagicMock()
        db.get_session.return_value = {"id": "20260524_111111_xyz", "title": "demo"}
        db.resolve_resume_session_id.return_value = "20260524_111111_xyz"
        db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        db._conn = MagicMock()  # for the reopen execute() call

        cli = _make_cli(quiet=True, db=db)
        # Stop _init_agent right after the resume banner: prevent it from
        # constructing a real AIAgent (the next code path).
        with patch("cli._prepare_deferred_agent_startup"):
            try:
                cli._init_agent()
            except Exception:
                # The post-resume agent-init machinery may fail in this
                # stubbed context (no API key, no real config) — we only
                # care about the printed banner that comes earlier.
                pass

        captured = capsys.readouterr()
        # Banner on stderr — stdout stays clean for automation.
        assert "↻ Resumed session" not in captured.out
        assert "↻ Resumed session" in captured.err
        assert "20260524_111111_xyz" in captured.err
        assert "demo" in captured.err

    def test_no_messages_goes_to_stderr_in_quiet_mode(self, capsys):
        db = MagicMock()
        db.get_session.return_value = {"id": "20260524_111111_xyz"}
        db.resolve_resume_session_id.return_value = "20260524_111111_xyz"
        db.get_messages_as_conversation.return_value = []
        db._conn = MagicMock()

        cli = _make_cli(quiet=True, db=db)
        with patch("cli._prepare_deferred_agent_startup"):
            try:
                cli._init_agent()
            except Exception:
                pass

        captured = capsys.readouterr()
        assert "has no messages" not in captured.out
        assert "has no messages" in captured.err
        assert "Starting fresh" in captured.err
