"""Regression test for #11884: _make_agent must resolve runtime provider.

Without resolve_runtime_provider(), bare-slug models in config
(e.g. ``claude-opus-4-6`` with ``model.provider: anthropic``) leave
provider/base_url/api_key empty in AIAgent, causing HTTP 404.
"""

import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_make_agent_passes_resolved_provider():
    """_make_agent forwards provider/base_url/api_key/api_mode from
    resolve_runtime_provider to AIAgent."""

    fake_runtime = {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-test-key",
        "api_mode": "anthropic_messages",
        "command": None,
        "args": None,
        "credential_pool": None,
    }

    fake_cfg = {
        "model": {"default": "claude-opus-4-6", "provider": "anthropic"},
        "agent": {"system_prompt": "test"},
    }

    with (
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_tool_progress_mode", return_value="compact"),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ) as mock_resolve,
        patch("run_agent.AIAgent") as mock_agent,
    ):

        from tui_gateway.server import _make_agent

        _make_agent("sid-1", "key-1")

        # target_model comes from _resolve_startup_runtime() which reads
        # _load_cfg().  Due to module-level caching in tui_gateway.server,
        # the patched config may not take effect when the module was already
        # imported by an earlier test.  Assert the stable part of the call.
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs.get("requested") is None

        call_kwargs = mock_agent.call_args
        assert call_kwargs.kwargs["provider"] == "anthropic"
        assert call_kwargs.kwargs["base_url"] == "https://api.anthropic.com"
        assert call_kwargs.kwargs["api_key"] == "sk-test-key"
        assert call_kwargs.kwargs["api_mode"] == "anthropic_messages"


def test_probe_config_health_flags_null_sections():
    """Bare YAML keys (`agent:` with no value) parse as None and silently
    drop nested settings; probe must surface them so users can fix."""
    from tui_gateway.server import _probe_config_health

    assert _probe_config_health({"agent": {"x": 1}}) == ""
    assert _probe_config_health({}) == ""

    msg = _probe_config_health({"agent": None, "display": None, "model": {}})
    assert "agent" in msg and "display" in msg
    assert "model" not in msg


def test_apply_model_switch_does_not_leak_process_env():
    """Core fix for cross-session contamination: an in-session /model switch
    must mutate only the target session (record a per-session override + switch
    that session's agent in place) and must NOT write process-global env vars,
    which the single-process desktop backend shares across every live session.
    """
    from tui_gateway import server

    class _FakeResult:
        success = True
        error_message = ""
        warning_message = ""
        new_model = "zai/glm-5.1"
        target_provider = "zai"
        base_url = "https://api.z.ai/v1"
        api_key = "sk-glm"
        api_mode = "chat_completions"

    class _FakeAgent:
        def __init__(self):
            self.model = "minimax/m3"
            self.provider = "minimax"
            self.base_url = ""
            self.api_key = ""

        def switch_model(self, **kw):
            self.model = kw["new_model"]
            self.provider = kw["new_provider"]

    env_keys = (
        "HERMES_MODEL",
        "HERMES_INFERENCE_MODEL",
        "HERMES_TUI_PROVIDER",
        "HERMES_INFERENCE_PROVIDER",
    )

    sess_b = {"agent": _FakeAgent(), "session_key": "k-B", "model_override": None}
    sess_a = {"agent": _FakeAgent(), "session_key": "k-A", "model_override": None}

    with (
        patch("hermes_cli.model_switch.parse_model_flags",
              return_value=("glm-5.1", None, False, False, True)),
        patch("hermes_cli.model_switch.resolve_persist_behavior",
              return_value=False),
        patch("hermes_cli.model_switch.switch_model", return_value=_FakeResult()),
        patch("tui_gateway.server._emit"),
        patch("tui_gateway.server._restart_slash_worker"),
        patch("tui_gateway.server._session_info", return_value={}),
        patch("tui_gateway.server._persist_model_switch") as mock_persist,
    ):
        before = {k: os.environ.get(k) for k in env_keys}
        result = server._apply_model_switch("sidB", sess_b, "glm-5.1")
        after = {k: os.environ.get(k) for k in env_keys}

    assert result["value"] == "zai/glm-5.1"
    # No process-global env mutation (the contamination vector).
    assert before == after
    # persist_global was False → config untouched.
    mock_persist.assert_not_called()
    # Target session recorded a per-session override.
    assert sess_b["model_override"]["model"] == "zai/glm-5.1"
    assert sess_b["model_override"]["provider"] == "zai"
    # The switched agent mutated in place.
    assert sess_b["agent"].model == "zai/glm-5.1"
    # Sibling session is completely untouched.
    assert sess_a["model_override"] is None
    assert sess_a["agent"].model == "minimax/m3"


def test_public_model_switch_to_moa_disables_bodyless_warm_dispatch(monkeypatch, tmp_path):
    from hermes_cli import heartbeat
    from run_agent import AIAgent
    from tui_gateway import server

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    agent = AIAgent(
        api_key="synthetic-key",
        base_url="http://127.0.0.1:9/v1",
        model="local-before-switch",
        provider="custom:synthetic",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["file"],
        max_iterations=1,
    )
    generation = agent._openai_transport_generation
    physical_calls = []
    agent._interruptible_api_call = lambda *args, **kwargs: physical_calls.append((args, kwargs))
    result = SimpleNamespace(
        success=True,
        error_message="",
        warning_message="",
        new_model="review",
        target_provider="moa",
        base_url="moa://local",
        api_key="moa-virtual-provider",
        api_mode="chat_completions",
        model_info=None,
        runtime_capabilities=None,
    )
    session = {
        "agent": agent,
        "session_key": "synthetic-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
    }

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: result)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda *_args: None)
    monkeypatch.setattr(server, "_persist_live_session_system_prompt", lambda *_args: None)
    monkeypatch.setattr(server, "_append_model_switch_marker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_emit_session_info", lambda *_args: None)

    response = server._apply_model_switch(
        "sid", session, "review --provider moa", confirm_expensive_model=True
    )

    assert response["value"] == "review"
    assert agent._openai_transport_kind == "moa"
    assert agent._openai_transport_generation == generation + 1

    class _Due:
        def due_prompt(self):
            raise AssertionError("MoA must not consume or dispatch a bodyless warm request")

        def clear(self):
            pass

    route = "moa:review"
    identity = server._tui_cache_warm_identity(agent)
    session.update(_cache_warm_route=route, _cache_warm_identity=identity)
    monkeypatch.setattr(heartbeat, "HeartbeatManager", lambda *_args, **_kwargs: _Due())

    server._tui_cache_warm_due("sid", session, agent, route, identity)

    assert physical_calls == []
    assert session["running"] is False
