"""Focused contract tests for runtime request-overrides rebuilds."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.chat_completion_helpers import try_activate_fallback


_PRIMARY_URL = "https://primary.example/v1"
_FALLBACK_URL = "https://fallback.example/v1"


def _make_agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "agent.context_compressor.get_model_context_length", return_value=200_000
        ),
    ):
        agent = AIAgent(
            model="primary-model",
            provider="custom:primary",
            api_key="primary-key",
            base_url=_PRIMARY_URL,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock(name="primary-client")
    return agent


def _config() -> dict:
    return {
        "custom_providers": [
            {
                "provider_key": "primary",
                "name": "primary",
                "base_url": _PRIMARY_URL,
                "model": "primary-model",
                "extra_body": {"primary_only": True},
            },
            {
                "provider_key": "fallback",
                "name": "fallback",
                "base_url": _FALLBACK_URL,
                "model": "fallback-model",
                "extra_body": {"fallback_only": True, "nested": {"x": 99}},
            },
        ]
    }


def test_switch_rollback_preserves_request_overrides_deeply():
    agent = _make_agent()
    original = {"extra_body": {"nested": {"value": "primary"}}}
    agent.request_overrides = original

    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch.object(
            agent, "_create_openai_client", side_effect=RuntimeError("reject")
        ),
    ):
        with pytest.raises(RuntimeError, match="reject"):
            agent.switch_model(
                "switched-model",
                "custom:switched",
                api_key="switched-key",
                base_url="https://switched.example/v1",
            )

    assert agent.request_overrides == original
    agent.request_overrides["extra_body"]["nested"]["value"] = "mutated"
    assert original["extra_body"]["nested"]["value"] == "primary"


def test_fallback_rebuilds_overrides_and_restore_uses_deep_primary_snapshot():
    agent = _make_agent()
    agent.request_overrides = {"extra_body": {"primary_only": True, "nested": {"x": 1}}}
    agent._fallback_chain = [
        {
            "provider": "custom:fallback",
            "model": "fallback-model",
            "base_url": _FALLBACK_URL,
        }
    ]
    agent._fallback_model = agent._fallback_chain[0]

    fallback_client = SimpleNamespace(
        api_key="fallback-key",
        base_url=_FALLBACK_URL,
        _custom_headers={},
    )
    with (
        patch("hermes_cli.config.load_config", return_value=_config()),
        patch("agent.model_metadata.get_model_context_length", return_value=200_000),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "fallback-model"),
        ),
        patch("agent.credential_pool.load_pool", return_value=None),
    ):
        assert try_activate_fallback(agent) is True

    assert agent.request_overrides == {
        "extra_body": {"fallback_only": True, "nested": {"x": 99}}
    }
    agent.request_overrides["extra_body"]["nested"]["x"] = 2

    with patch("run_agent.OpenAI", return_value=MagicMock()):
        assert agent._restore_primary_runtime() is True

    assert agent.request_overrides == {
        "extra_body": {"primary_only": True, "nested": {"x": 1}}
    }


def test_failed_fallback_candidate_is_rolled_back_and_client_closed():
    agent = _make_agent()
    agent.request_overrides = {"extra_body": {"primary_only": True}}
    agent._fallback_chain = [
        {
            "provider": "custom:first",
            "model": "first-model",
            "base_url": "https://first.example/v1",
        },
        {
            "provider": "custom:second",
            "model": "second-model",
            "base_url": "https://second.example/v1",
        },
    ]
    agent._fallback_model = agent._fallback_chain[0]
    first_client = MagicMock(name="first-client")
    first_client.api_key = "first-key"
    first_client.base_url = "https://first.example/v1"
    first_client._custom_headers = {}
    second_client = SimpleNamespace(
        api_key="second-key",
        base_url="https://second.example/v1",
        _custom_headers={},
    )

    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[
                (first_client, "first-model"),
                (second_client, "second-model"),
            ],
        ),
        patch("agent.credential_pool.load_pool", return_value=None),
        patch.object(
            agent,
            "_anthropic_prompt_cache_policy",
            side_effect=[RuntimeError("reject"), (False, False)],
        ),
        patch("agent.model_metadata.get_model_context_length", return_value=200_000),
    ):
        assert try_activate_fallback(agent) is True

    first_client.close.assert_called_once()
    assert agent.provider == "custom:second"
    assert agent.model == "second-model"
    assert agent._fallback_index == 2
    assert agent._primary_runtime["provider"] == "custom:primary"
    assert agent.request_overrides == {}
