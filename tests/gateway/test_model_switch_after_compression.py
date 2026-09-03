"""Production-boundary regression coverage for deferred reasoning switches."""

import asyncio

import pytest

from hermes_cli.model_switch import (
    ModelSwitchResult,
    apply_model_switch_after_compression,
)
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore


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
        api_key="[REDACTED]",
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


@pytest.mark.asyncio
async def test_gateway_deferred_reasoning_survives_fresh_session_reconstruction(
    tmp_path, monkeypatch
):
    """A deferred switch persists explicit reasoning through a fresh runner."""
    import hermes_state

    def _no_sqlite(*_args, **_kwargs):
        raise RuntimeError("SQLite disabled in test")

    monkeypatch.setattr(hermes_state, "SessionDB", _no_sqlite)
    config = GatewayConfig()
    store = SessionStore(tmp_path, config)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_id="chat-1",
        user_name="tester",
        chat_type="dm",
    )
    entry = store.get_or_create_session(source)
    session_key = entry.session_key

    runner = object.__new__(GatewayRunner)
    runner._sessions = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._pending_model_notes = {}
    runner._gateway_loop = asyncio.get_running_loop()
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    persisted = asyncio.Event()
    original_set_model_override = runner._async_session_store.set_model_override

    async def _persist_and_signal(key, override):
        await original_set_model_override(key, override)
        persisted.set()

    runner._async_session_store.set_model_override = _persist_and_signal
    agent = _Agent()
    result = ModelSwitchResult(
        success=True,
        new_model="new/model",
        target_provider="openrouter",
        api_key="[REDACTED]",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        reasoning_config={"enabled": True, "effort": "low"},
    )
    runner._session_state(session_key).conversation.after_compression_model_switch = result
    runner._attach_model_switch_after_compression(session_key, agent)

    assert apply_model_switch_after_compression(agent) == "applied"
    await asyncio.wait_for(persisted.wait(), timeout=1)

    store2 = SessionStore(tmp_path, config)
    assert store2.get_model_override(session_key) == {
        "model": "new/model",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "reasoning_config": {"enabled": True, "effort": "low"},
    }
    fresh_runner = object.__new__(GatewayRunner)
    fresh_runner._sessions = {}
    fresh_runner.session_store = store2
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda _provider: {
            "api_key": "[REDACTED]",
            "api_mode": "chat_completions",
        },
    )
    fresh_runner._rehydrate_session_model_override(session_key)

    assert fresh_runner._resolve_session_reasoning_config(
        session_key=session_key,
        model="new/model",
    ) == {"enabled": True, "effort": "low"}
