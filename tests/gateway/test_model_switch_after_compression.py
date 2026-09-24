"""Production-boundary regression coverage for deferred reasoning switches."""

from hermes_cli.model_switch import ModelSwitchResult, apply_model_switch_after_compression
from gateway.run import GatewayRunner


class _Agent:
    def __init__(self):
        self.model = "old/model"
        self.provider = "openrouter"

    def switch_model(self, new_model, new_provider, _api_key, _base_url, _api_mode):
        self.model = new_model
        self.provider = new_provider


def test_gateway_deferred_reasoning_survives_compression_rebuild(monkeypatch):
    """The first request after compression must keep explicit ``low``."""
    runner = object.__new__(GatewayRunner)
    runner._sessions = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    monkeypatch.setattr(
        runner,
        "_load_reasoning_config",
        lambda _model="": {"enabled": True, "effort": "medium"},
    )

    session_key = "telegram:chat-1"
    agent = _Agent()
    result = ModelSwitchResult(
        success=True,
        new_model="old/model",
        target_provider="openrouter",
        api_key="old-key",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        reasoning_config={"enabled": True, "effort": "low"},
    )
    state = runner._session_state(session_key)
    state.conversation.after_compression_model_switch = result
    runner._attach_model_switch_after_compression(session_key, agent)
    runner._agent_cache[session_key] = (agent, "sig", 0, "session-1")

    assert apply_model_switch_after_compression(agent) == "applied"
    assert state.conversation.model_override["reasoning_config"] == {
        "enabled": True,
        "effort": "low",
    }

    monkeypatch.setattr(runner, "_release_evicted_agent_soft", lambda _agent: None)
    runner._evict_cached_agent(session_key)

    assert runner._resolve_session_reasoning_config(
        session_key=session_key,
        model="old/model",
    ) == {"enabled": True, "effort": "low"}
