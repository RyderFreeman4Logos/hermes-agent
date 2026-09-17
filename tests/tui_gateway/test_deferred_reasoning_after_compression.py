"""TUI deferred --after-compression --reasoning must keep the live route.

Moved out of tests/test_tui_gateway_server.py so unique-on-pin #179 no longer
appends the shared EOF that #62/#180 already own on replay.
"""

from unittest.mock import patch

import pytest

from tui_gateway import server


def _setup_make_agent_mocks(monkeypatch, cfg):
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(
        server, "_resolve_startup_runtime", lambda: ("test-model", None)
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: {
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "off")
    monkeypatch.setattr(server, "_load_reasoning_config", lambda model="": None)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})


def test_tui_reasoning_only_after_compression_keeps_route(monkeypatch):
    from hermes_cli.model_switch import (
        ModelSwitchResult,
        get_model_switch_after_compression,
    )

    class Agent:
        model = "old/model"
        provider = "openrouter"
        api_key = "old-key"
        base_url = "https://openrouter.ai/api/v1"
        api_mode = "chat_completions"

        def __init__(self):
            self.calls = []

        def switch_model(self, **kwargs):
            self.calls.append(kwargs)

    def fake_switch(**kwargs):
        return ModelSwitchResult(
            success=True,
            new_model=kwargs["raw_input"],
            target_provider=kwargs["explicit_provider"],
            api_key="old-key",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            reasoning_config={"enabled": True, "effort": "medium"},
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch)
    monkeypatch.setattr(
        server,
        "_persist_model_switch",
        lambda _result: pytest.fail("deferred switch must stay session-scoped"),
    )
    agent = Agent()
    session = {"agent": agent}

    out = server._apply_model_switch(
        "sid",
        session,
        "--after-compression --reasoning low",
        confirm_expensive_model=True,
    )

    assert out["pending"] is True
    pending = get_model_switch_after_compression(agent)
    assert pending is not None
    assert (pending.new_model, pending.target_provider) == ("old/model", "openrouter")
    assert pending.reasoning_config == {"enabled": True, "effort": "low"}
    assert agent.calls == []


def test_tui_deferred_reasoning_survives_compression_rebuild(monkeypatch):
    """The first rebuilt TUI agent must keep explicit ``low`` reasoning."""
    from hermes_cli.model_switch import (
        ModelSwitchResult,
        apply_model_switch_after_compression,
    )

    class Agent:
        model = "old/model"
        provider = "openrouter"
        api_key = "old-key"
        base_url = "https://openrouter.ai/api/v1"
        api_mode = "chat_completions"

        def switch_model(self, new_model, new_provider, _api_key, _base_url, _api_mode, capabilities=None):
            self.model = new_model
            self.provider = new_provider

    def fake_switch(**kwargs):
        return ModelSwitchResult(
            success=True,
            new_model=kwargs["raw_input"],
            target_provider=kwargs["explicit_provider"],
            api_key="old-key",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            reasoning_config={"enabled": True, "effort": "medium"},
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch)
    agent = Agent()
    session = {"agent": agent}

    server._apply_model_switch(
        "sid",
        session,
        "--after-compression --reasoning low",
        confirm_expensive_model=True,
    )
    pending = session["after_compression_model_switch"]
    assert pending.reasoning_config == {"enabled": True, "effort": "low"}
    assert apply_model_switch_after_compression(agent) == "applied"

    _setup_make_agent_mocks(monkeypatch, {})
    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent(
            "sid",
            "key",
            model_override=session["model_override"],
        )

    assert mock_agent.call_args.kwargs["reasoning_config"] == {
        "enabled": True,
        "effort": "low",
    }
