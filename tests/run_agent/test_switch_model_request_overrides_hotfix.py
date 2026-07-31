"""Regression coverage for request overrides on live model switches."""

import copy
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


GLM_URL = "https://api.z.ai/api/coding/paas/v4"
CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
PM_URL = "https://pm.example/v1"
CALLER_OVERRIDES = {
    "extra_headers": {"X-Caller": "keep"},
    "extra_body": {"caller_flag": True},
}
GLM_THINKING = {"thinking": {"type": "enabled"}}


def _agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.model = "glm-5.2"
    agent.provider = "custom:glm"
    agent.requested_provider = agent.provider
    agent.base_url = GLM_URL
    agent.api_key = "glm-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="glm-client")
    agent._client_kwargs = {"api_key": agent.api_key, "base_url": agent.base_url}
    agent._caller_request_overrides = copy.deepcopy(CALLER_OVERRIDES)
    agent.request_overrides = copy.deepcopy(CALLER_OVERRIDES)
    agent.request_overrides["extra_body"].update(GLM_THINKING)
    agent.context_compressor = None
    agent.service_tier = None
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    agent._transport_cache = {}
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
    agent._consecutive_stale_streams = 0
    agent._create_openai_client = MagicMock(return_value=MagicMock())
    agent._apply_client_headers_for_base_url = MagicMock()
    agent._ensure_lmstudio_runtime_loaded = MagicMock(return_value=None)
    agent._lmstudio_load_was_unverified = MagicMock(return_value=False)
    agent._effective_lmstudio_context_length = MagicMock(return_value=None)
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    return agent


def _configs():
    return [
        {
            "provider_key": "glm",
            "base_url": GLM_URL,
            "model": "glm-5.2",
            "extra_body": GLM_THINKING,
        },
        {
            "provider_key": "pm",
            "base_url": PM_URL,
            "model": "gpt-5.4-mini",
            "extra_body": {"pm_target": True},
        },
    ]


def _switch(agent, *, provider, base_url):
    agent.switch_model(
        new_model="gpt-5.4-mini",
        new_provider=provider,
        api_key="gpt-key",
        base_url=base_url,
        api_mode="codex_responses",
    )


@pytest.mark.parametrize(
    ("provider", "base_url", "target_extra"),
    [
        ("openai-codex", CODEX_URL, {"caller_flag": True}),
        ("custom:pm", PM_URL, {"caller_flag": True, "pm_target": True}),
    ],
)
def test_real_switch_drops_glm_thinking_and_reverse_restores_target_extras(
    provider,
    base_url,
    target_extra,
):
    agent = _agent()
    config = {
        "agent": {
            "reasoning_effort": "medium",
            "reasoning_overrides": {"gpt-5.4-mini": "low"},
        }
    }

    with (
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_configs(),
        ),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        _switch(agent, provider=provider, base_url=base_url)

        assert agent.request_overrides == {
            "extra_headers": {"X-Caller": "keep"},
            "extra_body": target_extra,
        }
        assert "thinking" not in agent.request_overrides["extra_body"]
        assert agent.reasoning_config == {"enabled": True, "effort": "low"}
        assert agent._primary_runtime["request_overrides"] == agent.request_overrides

        agent.switch_model(
            new_model="glm-5.2",
            new_provider="custom:glm",
            api_key="glm-key",
            base_url=GLM_URL,
            api_mode="chat_completions",
        )

    assert agent.request_overrides == {
        "extra_headers": {"X-Caller": "keep"},
        "extra_body": {"caller_flag": True, **GLM_THINKING},
    }


def test_failed_switch_restores_nested_request_overrides():
    agent = _agent()
    original = copy.deepcopy(agent.request_overrides)
    agent._create_openai_client.side_effect = RuntimeError("client failed")

    with (
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        pytest.raises(RuntimeError, match="client failed"),
    ):
        _switch(agent, provider="openai-codex", base_url=CODEX_URL)

    assert agent.request_overrides == original
    agent.request_overrides["extra_body"]["thinking"]["type"] = "mutated"
    assert original["extra_body"]["thinking"]["type"] == "enabled"


def test_primary_restore_uses_deep_copied_switched_overrides():
    agent = _agent()
    config = {
        "agent": {"reasoning_overrides": {"gpt-5.4-mini": "low"}}
    }
    with (
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_configs(),
        ),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        _switch(agent, provider="openai-codex", base_url=CODEX_URL)

    primary_overrides = copy.deepcopy(agent.request_overrides)
    agent.request_overrides = {"extra_body": copy.deepcopy(GLM_THINKING)}
    agent._fallback_activated = True
    agent._rate_limited_until = 0
    agent.context_compressor = MagicMock()

    assert agent._restore_primary_runtime() is True
    assert agent.request_overrides == primary_overrides
    agent.request_overrides["extra_body"]["caller_flag"] = False
    assert agent._primary_runtime["request_overrides"] == primary_overrides


def test_real_fallback_activation_rebuilds_and_restores_request_overrides():
    config = {"custom_providers": _configs()}
    fallbacks = [
        {
            "provider": "openai-codex",
            "model": "gpt-5.4-mini",
            "base_url": CODEX_URL,
            "api_key": "codex-key",
        },
        {
            "provider": "custom:pm",
            "model": "gpt-5.4-mini",
            "base_url": PM_URL,
            "api_key": "pm-key",
        },
    ]
    clients = []
    for base_url, api_key in ((CODEX_URL, "codex-key"), (PM_URL, "pm-key")):
        client = MagicMock(base_url=base_url, api_key=api_key)
        client._custom_headers = {}
        client.default_headers = {}
        clients.append(client)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_configs(),
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[(clients[0], "gpt-5.4-mini"), (clients[1], "gpt-5.4-mini")],
        ),
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent = AIAgent(
            api_key="glm-key",
            base_url=GLM_URL,
            provider="custom:glm",
            model="glm-5.2",
            request_overrides=CALLER_OVERRIDES,
            fallback_model=fallbacks,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        primary = copy.deepcopy(agent._primary_runtime["request_overrides"])
        assert primary["extra_body"] == {"thinking": {"type": "enabled"}, "caller_flag": True}

        assert agent._try_activate_fallback() is True
        assert agent.request_overrides == CALLER_OVERRIDES

        assert agent._try_activate_fallback() is True
        assert agent.request_overrides == {
            "extra_headers": {"X-Caller": "keep"},
            "extra_body": {"pm_target": True, "caller_flag": True},
        }
        agent.request_overrides["extra_body"]["caller_flag"] = False
        assert agent._caller_request_overrides == CALLER_OVERRIDES
        assert agent._primary_runtime["request_overrides"] == primary

        assert agent._restore_primary_runtime() is True

    assert agent.request_overrides == primary


def test_constructor_snapshots_explicit_caller_overrides():
    explicit = {"extra_body": {"caller_flag": {"nested": True}}}
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url=GLM_URL,
            provider="custom:glm",
            model="glm-5.2",
            request_overrides=explicit,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    explicit["extra_body"]["caller_flag"]["nested"] = False
    assert agent._caller_request_overrides["extra_body"]["caller_flag"] == {
        "nested": True
    }
    assert agent._primary_runtime["request_overrides"]["extra_body"][
        "caller_flag"
    ] == {"nested": True}
