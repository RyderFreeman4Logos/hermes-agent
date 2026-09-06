"""Regression tests for the in-place /model switch (CLI/TUI) carrying a custom
provider's request_overrides (extra_body) — _apply_switched_provider_request_overrides.

Before the fix, agent_runtime_helpers.switch_model() swapped model/provider/
base_url/api_key in place but never touched request_overrides, so a /model
switch to a thinking-enabled custom provider in the TUI/CLI kept the old
provider's extra_body.

The switched-to entry is matched by provider key + base_url + model (the same
condition agent_init._merge_custom_provider_extra_body uses at build time), so a
*different* model selected at the same named endpoint does not inherit an
extra_body configured for another model.
"""

import agent.agent_runtime_helpers as arh


class _Agent:
    pass


# Two entries share the same named endpoint / base_url but pin different models —
# the exact case a name-only match got wrong.
CUSTOM_PROVIDERS = [
    {
        "name": "main-think",
        "base_url": "http://10.0.0.1:8000/v1",
        "model": "think-model",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
    {
        "name": "main-plain",
        "base_url": "http://10.0.0.1:8000/v1",
        "model": "plain-model",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
]


def _agent(*, model, base_url, request_overrides, custom_providers=CUSTOM_PROVIDERS):
    a = _Agent()
    # switch_model() sets these on the live agent before calling the helper.
    a.model = model
    a.base_url = base_url
    a.provider = "custom"
    a.request_overrides = request_overrides
    a._custom_providers = custom_providers  # init-time cache the helper reads
    return a


def test_switch_applies_matched_provider_extra_body():
    """Switching to the matching provider+model applies its extra_body and
    preserves non-provider overrides (service_tier/speed from /fast)."""
    a = _agent(
        model="think-model",
        base_url="http://10.0.0.1:8000/v1",
        request_overrides={"service_tier": "priority"},
    )
    arh._apply_switched_provider_request_overrides(a, "custom:main-think")
    assert a.request_overrides["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert a.request_overrides["service_tier"] == "priority"  # preserved


def test_switch_to_noncustom_clears_stale_extra_body():
    """Switching to a built-in provider clears the previous provider's extra_body."""
    a = _agent(
        model="claude-x",
        base_url="https://api.anthropic.com",
        request_overrides={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "service_tier": "priority",
        },
    )
    arh._apply_switched_provider_request_overrides(a, "anthropic")
    assert "extra_body" not in a.request_overrides  # stale extra_body cleared
    assert a.request_overrides["service_tier"] == "priority"  # preserved


def test_switch_from_none_overrides():
    """A None request_overrides is handled and gets the matched extra_body."""
    a = _agent(
        model="plain-model",
        base_url="http://10.0.0.1:8000/v1",
        request_overrides=None,
    )
    arh._apply_switched_provider_request_overrides(a, "custom:main-plain")
    assert a.request_overrides == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_switch_to_different_model_same_endpoint_does_not_inherit():
    """Review regression: selecting a *different* model while naming a custom
    provider must NOT inherit that provider's extra_body when the models differ.

    'main-think' pins 'think-model'. Selecting 'plain-model' under
    custom:main-think must not carry enable_thinking=True — the model-aware
    matcher rejects the mismatch and the stale extra_body is cleared. (A
    name-only match would have wrongly carried it over.)
    """
    a = _agent(
        model="plain-model",  # differs from main-think's pinned 'think-model'
        base_url="http://10.0.0.1:8000/v1",
        request_overrides={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    arh._apply_switched_provider_request_overrides(a, "custom:main-think")
    assert "extra_body" not in a.request_overrides  # not inherited; stale cleared


def test_switch_endpoint_mismatch_does_not_inherit():
    """A matching provider *name* but a different base_url must not match either
    (endpoint identity is part of the condition)."""
    a = _agent(
        model="think-model",
        base_url="http://10.9.9.9:8000/v1",  # different endpoint than the entry
        request_overrides={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    arh._apply_switched_provider_request_overrides(a, "custom:main-think")
    assert "extra_body" not in a.request_overrides  # base_url mismatch -> cleared


def test_failed_switch_restores_original_request_overrides_deeply():
    """Failed client rebuild must restore the nested override graph, not an alias."""
    from unittest.mock import MagicMock, patch

    import pytest
    from run_agent import AIAgent

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
    agent.runtime_capabilities = {"native_compaction": False}
    original = {"extra_body": {"nested": {"value": "primary"}}}
    agent.request_overrides = original

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

    assert agent.request_overrides == original
    agent.request_overrides["extra_body"]["nested"]["value"] = "mutated"
    assert original["extra_body"]["nested"]["value"] == "primary"


def test_copy_request_overrides_hostile_deepcopy_does_not_poison_snapshot():
    """Hostile __deepcopy__ must not leave rollback without a usable override dict."""
    class HostileCopy:
        def __deepcopy__(self, _memo):
            raise ValueError("hostile")

        def __repr__(self):
            return "PAYLOAD-SENTINEL-41"

    nested = {"owned": True, "hostile": HostileCopy()}
    copied = arh._copy_request_overrides(nested)
    assert copied is not nested
    assert copied["owned"] is True
    copied["owned"] = False
    assert nested["owned"] is True


def test_primary_runtime_snapshot_deep_copies_request_overrides():
    class _Agent:
        pass

    agent = _Agent()
    agent.model = "m"
    agent.provider = "p"
    agent.requested_provider = "p"
    agent.base_url = "https://example.invalid/v1"
    agent.api_mode = "chat_completions"
    agent.api_key = "k"
    agent._client_kwargs = {"api_key": "k"}
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent.reasoning_config = None
    agent._reasoning_echo_flag = False
    original = {"extra_body": {"nested": {"value": 1}}}
    agent.request_overrides = original
    agent.runtime_capabilities = {}
    agent.context_compressor = None
    rt = arh._build_primary_runtime_snapshot(agent, "chat_completions")
    assert rt["request_overrides"] == original
    rt["request_overrides"]["extra_body"]["nested"]["value"] = 2
    assert original["extra_body"]["nested"]["value"] == 1
