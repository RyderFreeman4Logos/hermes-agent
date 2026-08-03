"""Regression test for #33175: switch_model() must roll back to the pre-swap
state if the client rebuild raises.

Before the fix, ``agent.model`` and ``agent.provider`` were assigned BEFORE
the client rebuild was attempted, with no try/except to restore them on
failure.  An exception during ``build_anthropic_client`` / OpenAI client
construction left the agent with the new model+provider name but the OLD
client — producing HTTP 400s like "claude-sonnet-4-6 is not supported on
openai-codex" on the next turn.

These tests exercise both branches (openai_chat_completions and
anthropic_messages) and assert that every mutated field returns to its
pre-swap value when the rebuild raises.
"""

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.model_switch import (
    ModelSwitchResult,
    get_model_switch_after_compression,
    schedule_model_switch_after_compression,
)
from run_agent import AIAgent


def _make_agent_openrouter():
    """Agent on openrouter (openai-compatible) with sentinel client + kwargs."""
    agent = AIAgent.__new__(AIAgent)

    agent.provider = "openrouter"
    agent.model = "x-ai/grok-4"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "or-key-original"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="OriginalOpenRouterClient")
    agent._client_kwargs = {
        "api_key": "or-key-original",
        "base_url": "https://openrouter.ai/api/v1",
    }
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None

    return agent


def _make_agent_anthropic():
    """Agent on native anthropic with a sentinel anthropic client."""
    agent = AIAgent.__new__(AIAgent)

    agent.provider = "anthropic"
    agent.model = "claude-sonnet-4-5"
    agent.base_url = "https://api.anthropic.com"
    agent.api_key = "sk-ant-original"
    agent.api_mode = "anthropic_messages"
    agent.client = None
    agent._client_kwargs = {}
    agent.context_compressor = None
    agent._anthropic_api_key = "sk-ant-original"
    agent._anthropic_base_url = "https://api.anthropic.com"
    agent._anthropic_client = MagicMock(name="OriginalAnthropicClient")
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None

    return agent


def test_openai_client_rebuild_failure_rolls_back_to_original_state():
    """When OpenAI client construction fails, every mutated field must restore."""
    agent = _make_agent_openrouter()

    original_client = agent.client
    original_kwargs = dict(agent._client_kwargs)

    # _create_openai_client raises mid-swap (simulates bad key / network error)
    def boom(*_a, **_kw):
        raise RuntimeError("simulated client build failure")

    agent._create_openai_client = boom

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        with pytest.raises(RuntimeError, match="simulated client build failure"):
            agent.switch_model(
                new_model="openai/gpt-5",
                new_provider="openai-codex",
                api_key="codex-key-new",
                base_url="https://chatgpt.com/backend-api/codex/responses",
                api_mode="chat_completions",
            )

    # Core invariant: agent state is unchanged from before the call
    assert agent.model == "x-ai/grok-4"
    assert agent.provider == "openrouter"
    assert agent.base_url == "https://openrouter.ai/api/v1"
    assert agent.api_mode == "chat_completions"
    assert agent.api_key == "or-key-original"
    assert agent.client is original_client
    assert agent._client_kwargs == original_kwargs


def test_anthropic_client_rebuild_failure_rolls_back_to_original_state():
    """When build_anthropic_client raises, every mutated field must restore."""
    agent = _make_agent_anthropic()

    original_anthropic_client = agent._anthropic_client
    original_anthropic_key = agent._anthropic_api_key
    original_anthropic_base = agent._anthropic_base_url

    with (
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            side_effect=RuntimeError("simulated anthropic build failure"),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            return_value="sk-ant-resolved",
        ),
        patch("agent.anthropic_adapter._is_oauth_token", return_value=False),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        with pytest.raises(RuntimeError, match="simulated anthropic build failure"):
            agent.switch_model(
                new_model="claude-opus-4-6",
                new_provider="opencode-zen",
                api_key="zen-key-new",
                base_url="https://opencode.example/v1",
                api_mode="anthropic_messages",
            )

    # Anthropic-specific state restored
    assert agent._anthropic_client is original_anthropic_client
    assert agent._anthropic_api_key == original_anthropic_key
    assert agent._anthropic_base_url == original_anthropic_base

    # Core state also restored
    assert agent.model == "claude-sonnet-4-5"
    assert agent.provider == "anthropic"
    assert agent.base_url == "https://api.anthropic.com"
    assert agent.api_mode == "anthropic_messages"
    assert agent.api_key == "sk-ant-original"


def test_cross_branch_anthropic_to_openai_rebuild_failure_rolls_back():
    """Switching from anthropic_messages to chat_completions: failure must
    restore the anthropic state, not leave the agent half-converted."""
    agent = _make_agent_anthropic()

    original_anthropic_client = agent._anthropic_client

    def boom(*_a, **_kw):
        raise RuntimeError("openai client failed")

    agent._create_openai_client = boom

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        with pytest.raises(RuntimeError, match="openai client failed"):
            agent.switch_model(
                new_model="x-ai/grok-4",
                new_provider="openrouter",
                api_key="or-key-new",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )

    # Anthropic client preserved (not nulled by the openai branch)
    assert agent._anthropic_client is original_anthropic_client
    assert agent.model == "claude-sonnet-4-5"
    assert agent.provider == "anthropic"
    assert agent.api_mode == "anthropic_messages"
    assert agent.base_url == "https://api.anthropic.com"


def test_successful_switch_still_works_after_rollback_refactor():
    """Sanity check: the try/except wrapper hasn't broken the happy path."""
    agent = _make_agent_openrouter()

    new_client = MagicMock(name="NewClient")
    agent._create_openai_client = lambda *_a, **_kw: new_client

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        agent.switch_model(
            new_model="openai/gpt-5",
            new_provider="openrouter",
            api_key="or-key-new",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent.model == "openai/gpt-5"
    assert agent.provider == "openrouter"
    assert agent.api_key == "or-key-new"
    assert agent.client is new_client


def _schedule_deferred(agent):
    result = ModelSwitchResult(
        success=True,
        new_model="later-model",
        target_provider="later-provider",
        api_key="later-key",
        base_url="https://later.example/v1",
        api_mode="chat_completions",
    )
    schedule_model_switch_after_compression(agent, result)
    return result


def test_successful_immediate_switch_cancels_deferred_intent():
    agent = _make_agent_openrouter()
    _schedule_deferred(agent)
    agent._create_openai_client = lambda *_a, **_kw: MagicMock()

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        agent.switch_model(
            new_model="immediate-model",
            new_provider="openrouter",
            api_key="immediate-key",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )

    assert get_model_switch_after_compression(agent) is None


def test_failed_immediate_switch_keeps_deferred_intent():
    agent = _make_agent_openrouter()
    pending = _schedule_deferred(agent)
    agent._create_openai_client = MagicMock(
        side_effect=RuntimeError("simulated immediate failure")
    )

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        with pytest.raises(RuntimeError, match="simulated immediate failure"):
            agent.switch_model(
                new_model="broken-model",
                new_provider="openrouter",
                api_key="broken-key",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )

    assert get_model_switch_after_compression(agent) is pending


def test_deferred_apply_uses_scheduled_runtime_metadata_only():
    from types import SimpleNamespace

    from hermes_cli.model_switch import apply_model_switch_after_compression

    agent = _make_agent_openrouter()
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda system_message=None: "prompt"
    agent.context_compressor = SimpleNamespace(
        context_length=32_000,
        threshold_tokens=24_000,
        update_model=lambda **values: agent.context_compressor.__dict__.update(values),
    )
    pending = ModelSwitchResult(
        success=True,
        new_model="openai/gpt-4o-mini",
        target_provider="openrouter",
        api_key="new-key",
        base_url="https://openrouter.ai/api/v1",
        context_length=None,
        reasoning_config={"effort": "low"},
    )
    with patch(
        "hermes_cli.config.get_custom_provider_context_length",
        return_value=128_000,
    ):
        schedule_model_switch_after_compression(agent, pending)

    with (
        patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=AssertionError("boundary context resolver"),
        ),
        patch(
            "hermes_constants.resolve_reasoning_config",
            side_effect=AssertionError("boundary reasoning resolver"),
        ),
        patch.object(
            agent,
            "_ensure_lmstudio_runtime_loaded",
            side_effect=AssertionError("boundary provider prewarm"),
            create=True,
        ),
    ):
        status = apply_model_switch_after_compression(agent)

    assert status == "applied"
    assert agent.model == "openai/gpt-4o-mini"
    assert agent.context_compressor.context_length == 128_000
    assert agent.reasoning_config == {"effort": "low"}


def test_deferred_apply_without_destination_context_fails_closed():
    from types import SimpleNamespace

    from hermes_cli.model_switch import apply_model_switch_after_compression

    agent = _make_agent_openrouter()
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda system_message=None: "prompt"
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()
    agent.context_compressor = SimpleNamespace(
        context_length=200_000,
        threshold_tokens=160_000,
        update_model=lambda **values: agent.context_compressor.__dict__.update(values),
    )
    pending = ModelSwitchResult(
        success=True,
        new_model="custom/model",
        target_provider="custom-provider",
        api_key="new-key",
        base_url="https://custom.example/v1",
    )

    with (
        patch(
            "hermes_cli.config.get_custom_provider_context_length",
            return_value=None,
        ),
        patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=AssertionError("network-capable resolver called"),
        ),
        patch.object(
            agent,
            "_ensure_lmstudio_runtime_loaded",
            side_effect=AssertionError("boundary provider prewarm"),
            create=True,
        ),
    ):
        schedule_model_switch_after_compression(agent, pending)
        status = apply_model_switch_after_compression(agent)

    assert status == "failed"
    assert (agent.model, agent.provider) == ("x-ai/grok-4", "openrouter")
    assert agent.context_compressor.context_length == 200_000
    assert get_model_switch_after_compression(agent) is pending


def test_deferred_authoritative_callback_failure_restores_plugin_compressor():
    """Plugin compressor fields mutated by switch_model must restore exactly."""
    from hermes_cli.model_switch import apply_model_switch_after_compression

    class PluginCompressor:
        def __init__(self):
            self.model = "old-model"
            self.context_length = 32_000
            self.threshold_tokens = 24_000
            self.extra_marker = "plugin-old"

        def update_model(self, *, model, context_length=None, **_kwargs):
            self.model = model
            if context_length is not None:
                self.context_length = context_length
                self.threshold_tokens = int(context_length * 0.75)
            self.extra_marker = "plugin-new"

    agent = _make_agent_openrouter()
    agent.context_compressor = PluginCompressor()
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda system_message=None: f"prompt:{agent.model}"
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()
    agent._session_init_model_config = {
        "model": agent.model,
        "provider": agent.provider,
    }
    pending = ModelSwitchResult(
        success=True,
        new_model="new-model",
        target_provider="openrouter",
        api_key="new-key",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        context_length=128_000,
        reasoning_config={},
    )

    def fail_callback(*_args):
        raise RuntimeError("injected authoritative callback failure")

    schedule_model_switch_after_compression(
        agent, pending, on_applied=fail_callback
    )
    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        status = apply_model_switch_after_compression(agent)

    assert status == "failed"
    assert agent.model == "x-ai/grok-4"
    assert get_model_switch_after_compression(agent) is pending
    assert agent.context_compressor.model == "old-model"
    assert agent.context_compressor.context_length == 32_000
    assert agent.context_compressor.threshold_tokens == 24_000
    assert agent.context_compressor.extra_marker == "plugin-old"
