"""#159 auxiliary fallback routing and candidate-owned controls."""

import asyncio
import copy
import dataclasses
import json
import pickle
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import auxiliary_client as aux
from agent.context_compressor import ContextCompressor


@pytest.fixture(autouse=True)
def _admitted_runtime_fixture():
    """Keep focused routing tests on the new one-shot admission seam."""
    def resolve(**kwargs):
        provider = kwargs.get("requested") or "custom"
        codex = provider == "openai-codex"
        return {
            "provider": provider,
            "base_url": kwargs.get("explicit_base_url") or (
                "https://chatgpt.com/backend-api" if codex
                else f"https://{provider.replace(':', '-')}.invalid/v1"
            ),
            "api_mode": "codex_responses" if codex else "chat_completions",
            "api_key": kwargs.get("explicit_api_key") or "focused-bound-key",
        }

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve):
        yield


class _SyncCompletions:
    def __init__(self, *, fail=False, seen=None, text="codex"):
        self.fail = fail
        self.seen = seen if seen is not None else []
        self.text = text

    def create(self, **kwargs):
        self.seen.append(kwargs)
        if self.fail:
            error = RuntimeError("primary rate limited")
            error.status_code = 429
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))]
        )


class _AsyncCompletions(_SyncCompletions):
    async def create(self, **kwargs):
        return super().create(**kwargs)


def _client(completions, base_url):
    return SimpleNamespace(
        base_url=base_url,
        chat=SimpleNamespace(completions=completions),
    )


def _run_fallback_call(async_mode, *, task, route_info, patches, raises=False):
    """Run one sync/async fallback probe through the same patched seam."""
    with ExitStack() as stack:
        for candidate_patch in patches:
            stack.enter_context(candidate_patch)
        if async_mode:
            call = lambda: asyncio.run(
                aux._async_call_llm_impl(
                    task=task,
                    messages=[{"role": "user", "content": "hello"}],
                    route_info=route_info,
                )
            )
        else:
            call = lambda: aux._call_llm_impl(
                task=task,
                messages=[{"role": "user", "content": "hello"}],
                route_info=route_info,
            )
        if raises:
            with pytest.raises(RuntimeError, match="primary rate limited"):
                call()
            return None
        return call()


@pytest.mark.parametrize("async_mode", [False, True])
def test_all_auxiliary_fallback_paths_prefer_codex_and_use_entry_controls(async_mode):
    """A primary failure must not settle on pm/grok before configured Codex."""
    chain = [
        {"provider": "pm", "model": "pm-model"},
        {
            "provider": "openai-codex",
            "model": "codex-model",
            "reasoning_effort": "xhigh",
        },
        {"provider": "custom:localrouter", "model": "grok-model"},
    ]
    task_config = {
        "fallback_chain": chain,
        "reasoning_effort": "low",
        "extra_body": {
            "enable_thinking": True,
            "thinking": {"type": "enabled"},
        },
    }
    primary_seen = []
    codex_seen = []
    primary_completions = (
        _AsyncCompletions(fail=True, seen=primary_seen, text="primary")
        if async_mode
        else _SyncCompletions(fail=True, seen=primary_seen, text="primary")
    )
    codex_completions = (
        _AsyncCompletions(seen=codex_seen)
        if async_mode
        else _SyncCompletions(seen=codex_seen)
    )
    primary = _client(primary_completions, "https://qwen.invalid/v1")
    codex = _client(codex_completions, "https://chatgpt.com/backend-api")
    resolved = []

    def resolve_entry(entry):
        provider = entry["provider"]
        resolved.append(provider)
        return {"pm": _client(_SyncCompletions(text="pm"), "https://pm.invalid/v1"),
                "openai-codex": codex,
                "custom:localrouter": _client(
                    _SyncCompletions(text="grok"), "https://local.invalid/v1"
                )}[provider], entry["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value=task_config),
        patch.object(
            aux,
            "_resolve_task_provider_model",
            return_value=("auto", "qwen-primary", None, None, None),
        ),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(
            patch.object(
                aux,
                "_to_async_client",
                side_effect=lambda client, model, is_vision=False: (client, model),
            )
        )

    response = _run_fallback_call(
        async_mode, task="title_generation", route_info=route_info, patches=patches,
    )

    assert response.choices[0].message.content == "codex"
    assert resolved == ["openai-codex"]
    assert route_info == {
        "provider": "openai-codex",
        "model": "codex-model",
        "fallback_label": "fallback_chain[1](openai-codex)",
    }
    assert len(primary_seen) >= 1
    assert len(codex_seen) == 1
    codex_body = codex_seen[0].get("extra_body", {})
    assert "enable_thinking" not in codex_body
    assert "thinking" not in codex_body
    assert codex_body["reasoning"] == {"enabled": True, "effort": "xhigh"}


@pytest.mark.parametrize("async_mode", [False, True])
def test_rejected_codex_continues_to_remaining_codex_only(async_mode):
    """A rejected Codex tries the next Codex, never a lower-quality provider."""
    chain = [
        {"provider": "pm", "model": "pm-model"},
        {"provider": "openai-codex", "model": "codex-model-1"},
        {"provider": "openai-codex", "model": "codex-model-2"},
        {"provider": "custom:localrouter", "model": "grok-model"},
    ]
    task_config = {"fallback_chain": chain}
    primary_seen = []
    codex_one_seen = []
    codex_two_seen = []
    pm_seen = []
    localrouter_seen = []
    completions = _AsyncCompletions if async_mode else _SyncCompletions
    primary = _client(
        completions(fail=True, seen=primary_seen, text="primary"),
        "https://qwen.invalid/v1",
    )
    codex_one = _client(
        completions(fail=True, seen=codex_one_seen, text="codex-one"),
        "https://chatgpt.com/backend-api",
    )
    codex_two = _client(
        completions(seen=codex_two_seen, text="codex-two"),
        "https://chatgpt.com/backend-api",
    )
    pm = _client(
        completions(seen=pm_seen, text="pm"),
        "https://pm.invalid/v1",
    )
    localrouter = _client(
        completions(seen=localrouter_seen, text="grok"),
        "https://local.invalid/v1",
    )
    resolved = []

    def resolve_entry(entry):
        provider = entry["provider"]
        resolved.append(provider + ":" + entry["model"])
        return {
            "pm": pm,
            "openai-codex:codex-model-1": codex_one,
            "openai-codex:codex-model-2": codex_two,
            "custom:localrouter": localrouter,
        }[provider + ":" + entry["model"]], entry["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value=task_config),
        patch.object(
            aux,
            "_resolve_task_provider_model",
            return_value=("auto", "qwen-primary", None, None, None),
        ),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(
            patch.object(
                aux,
                "_to_async_client",
                side_effect=lambda client, model, is_vision=False: (client, model),
            )
        )

    response = _run_fallback_call(
        async_mode, task="title_generation", route_info=route_info, patches=patches,
    )

    assert response.choices[0].message.content == "codex-two"
    assert resolved == ["openai-codex:codex-model-1", "openai-codex:codex-model-2"]
    assert len(codex_one_seen) == 1
    assert len(codex_two_seen) == 1
    assert pm_seen == []
    assert localrouter_seen == []
    assert route_info == {
        "provider": "openai-codex",
        "model": "codex-model-2",
        "fallback_label": "fallback_chain[2](openai-codex)",
        "codex_skip_reason": "rejected",
    }


@pytest.mark.parametrize("async_mode", [False, True])
def test_rejected_codex_continues_to_remaining_configured_fallback(async_mode):
    """A rejected Codex must continue through remaining configured entries."""
    chain = [
        {"provider": "openai-codex", "model": "codex-model"},
        {"provider": "pm", "model": "pm-model"},
        {"provider": "custom:localrouter", "model": "grok-model"},
    ]
    task_config = {"fallback_chain": chain}
    primary_seen = []
    codex_seen = []
    pm_seen = []
    localrouter_seen = []
    completions = _AsyncCompletions if async_mode else _SyncCompletions
    primary = _client(
        completions(fail=True, seen=primary_seen, text="primary"),
        "https://qwen.invalid/v1",
    )
    codex = _client(
        completions(fail=True, seen=codex_seen, text="codex"),
        "https://chatgpt.com/backend-api",
    )
    pm = _client(
        completions(seen=pm_seen, text="pm"),
        "https://pm.invalid/v1",
    )
    localrouter = _client(
        completions(seen=localrouter_seen, text="grok"),
        "https://local.invalid/v1",
    )
    resolved = []

    def resolve_entry(entry):
        provider = entry["provider"]
        resolved.append(provider)
        return {
            "pm": pm,
            "openai-codex": codex,
            "custom:localrouter": localrouter,
        }[provider], entry["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value=task_config),
        patch.object(
            aux,
            "_resolve_task_provider_model",
            return_value=("auto", "qwen-primary", None, None, None),
        ),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(
            patch.object(
                aux,
                "_to_async_client",
                side_effect=lambda client, model, is_vision=False: (client, model),
            )
        )

    response = _run_fallback_call(
        async_mode, task="moa_reference", route_info=route_info, patches=patches,
    )

    assert response is not None
    assert response.choices[0].message.content == "pm"
    assert resolved == ["openai-codex", "pm"]
    assert len(codex_seen) == 1
    assert len(pm_seen) == 1
    assert localrouter_seen == []
    assert route_info == {
        "provider": "pm",
        "model": "pm-model",
        "fallback_label": "fallback_chain[1](pm)",
        "codex_skip_reason": "rejected",
    }


@pytest.mark.parametrize("async_mode", [False, True])
def test_all_codex_rejected_fails_closed_without_non_codex(async_mode):
    """Exhausted Codex candidates must not fall through to pm or Grok."""
    chain = [
        {"provider": "pm", "model": "pm-model"},
        {"provider": "openai-codex", "model": "codex-model-1"},
        {"provider": "openai-codex", "model": "codex-model-2"},
        {"provider": "custom:localrouter", "model": "grok-model"},
    ]
    task_config = {"fallback_chain": chain}
    primary_seen = []
    codex_seen = []
    pm_seen = []
    localrouter_seen = []
    completions = _AsyncCompletions if async_mode else _SyncCompletions
    primary = _client(
        completions(fail=True, seen=primary_seen, text="primary"),
        "https://qwen.invalid/v1",
    )
    codex = _client(
        completions(fail=True, seen=codex_seen, text="codex"),
        "https://chatgpt.com/backend-api",
    )
    pm = _client(
        completions(seen=pm_seen, text="pm"),
        "https://pm.invalid/v1",
    )
    localrouter = _client(
        completions(seen=localrouter_seen, text="grok"),
        "https://local.invalid/v1",
    )
    resolved = []

    def resolve_entry(entry):
        provider = entry["provider"]
        resolved.append(provider + ":" + entry["model"])
        if provider == "openai-codex":
            return codex, entry["model"]
        return {"pm": pm, "custom:localrouter": localrouter}[provider], entry["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value=task_config),
        patch.object(
            aux,
            "_resolve_task_provider_model",
            return_value=("auto", "qwen-primary", None, None, None),
        ),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(
            patch.object(
                aux,
                "_to_async_client",
                side_effect=lambda client, model, is_vision=False: (client, model),
            )
        )

    _run_fallback_call(
        async_mode,
        task="title_generation",
        route_info=route_info,
        patches=patches,
        raises=True,
    )

    assert resolved == ["openai-codex:codex-model-1", "openai-codex:codex-model-2"]
    assert len(codex_seen) == 2
    assert pm_seen == []
    assert localrouter_seen == []
    assert route_info["codex_skip_reason"] == "rejected"


def test_codex_skip_reason_is_scalar_when_unavailable():
    """A configured but unavailable Codex entry leaves auditable route metadata."""
    chain = [
        {"provider": "openai-codex", "model": "codex-model"},
        {"provider": "pm", "model": "pm-model"},
    ]
    pm_client = _client(_SyncCompletions(text="pm"), "https://pm.invalid/v1")
    route_info = {}
    resolved = []

    def resolve_entry(entry):
        resolved.append(entry["provider"])
        if entry["provider"] == "openai-codex":
            return None, None
        return pm_client, entry["model"]

    with (
        patch.object(
            aux,
            "_get_auxiliary_task_config",
            return_value={"fallback_chain": chain},
        ),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
    ):
        client, model, label = aux._try_configured_fallback_chain(
            "title_generation",
            "qwen",
            reason="rate limit",
            route_info=route_info,
            codex_only=True,
        )

    assert client is None
    assert model is None
    assert label == ""
    assert resolved == ["openai-codex"]
    assert route_info["codex_skip_reason"] == "unavailable"


@pytest.mark.parametrize("async_mode", [False, True])
def test_auto_route_never_returns_non_codex_discovery(async_mode):
    """Auto provider resolution must fail closed when only pm is available."""
    pm_client = _client(_SyncCompletions(text="pm"), "https://pm.invalid/v1")
    runtime = {"provider": "qwen", "model": "qwen-model"}
    real_resolve = aux.resolve_provider_client

    def resolve_dispatch(provider, *args, **kwargs):
        if provider == "auto":
            return real_resolve(provider, *args, **kwargs)
        return None, None

    with (
        patch.object(aux, "resolve_provider_client", side_effect=resolve_dispatch),
        patch.object(aux, "_try_configured_fallback_chain", return_value=(None, None, "")),
        patch.object(aux, "_try_main_fallback_chain", return_value=(None, None, "")),
        patch.object(
            aux,
            "_get_provider_chain",
            return_value=[("pm", lambda: (pm_client, "pm-model"))],
        ),
        patch.object(aux, "_is_provider_unhealthy", return_value=False),
    ):
        client, model = real_resolve(
            "auto", async_mode=async_mode, main_runtime=runtime,
        )

    assert client is None
    assert model is None
    assert pm_client.chat.completions.seen == []


@pytest.mark.parametrize("async_mode", [False, True])
def test_vision_auto_never_returns_non_codex_fallback(async_mode):
    """Vision auto must fail closed rather than return OpenRouter or Grok."""
    non_codex_client = _client(
        _SyncCompletions(text="non-codex"), "https://openrouter.invalid/v1"
    )
    runtime = {"provider": "qwen", "model": "qwen-vision-model"}
    seen = []

    def fake_strict(provider, model=None):
        seen.append(provider)
        if provider == "openai-codex":
            return None, None
        return non_codex_client, f"{provider}-vision-model"

    with (
        patch.object(
            aux,
            "_resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ),
        patch.object(aux, "_read_main_provider", return_value="qwen"),
        patch.object(aux, "_read_main_model", return_value="qwen-vision-model"),
        patch.object(aux, "_resolve_provider_vision_default", return_value=None),
        patch.object(aux, "_main_model_supports_vision", return_value=False),
        patch.object(aux, "_resolve_strict_vision_backend", side_effect=fake_strict),
        patch.object(
            aux,
            "_to_async_client",
            side_effect=lambda client, model, is_vision=False: (client, model),
        ),
    ):
        provider, client, model = aux.resolve_vision_provider_client(
            provider="auto", async_mode=async_mode, main_runtime=runtime,
        )

    assert (provider, client, model) == (None, None, None)
    assert "openai-codex" in seen
    assert not set(seen) & {
        "openrouter", "nous", "deepinfra", "pm", "localrouter",
    }


@pytest.mark.parametrize("async_mode", [False, True])
def test_moa_configured_chain_preserves_declared_order(async_mode):
    """MoA keeps its configured non-Codex order instead of partitioning Codex."""
    chain = [
        {"provider": "anthropic", "model": "claude-model"},
        {"provider": "openai-codex", "model": "codex-model"},
    ]
    completions = _AsyncCompletions if async_mode else _SyncCompletions
    primary_seen, anthropic_seen, codex_seen = [], [], []
    primary = _client(completions(fail=True, seen=primary_seen), "https://qwen.invalid/v1")
    anthropic = _client(completions(seen=anthropic_seen, text="anthropic"), "https://api.anthropic.com")
    codex = _client(completions(seen=codex_seen, text="codex"), "https://chatgpt.com/backend-api")
    resolved = []

    def resolve_entry(entry):
        resolved.append(entry["provider"])
        return {"anthropic": anthropic, "openai-codex": codex}[entry["provider"]], entry["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
        patch.object(aux, "_resolve_task_provider_model", return_value=("auto", "qwen-primary", None, None, None)),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(patch.object(aux, "_to_async_client", side_effect=lambda client, model, is_vision=False: (client, model)))

    response = _run_fallback_call(
        async_mode, task="moa_reference", route_info=route_info, patches=patches,
    )

    assert response.choices[0].message.content == "anthropic"
    assert resolved == ["anthropic"]
    assert len(anthropic_seen) == 1
    assert codex_seen == []


@pytest.mark.parametrize("async_mode", [False, True])
def test_moa_main_fallback_allows_non_codex_destination(async_mode):
    """MoA also retains a valid non-Codex top-level fallback in both callers."""
    chain = [{"provider": "anthropic", "model": "claude-model"}]
    task_config = {"fallback_chain": []}
    completions = _AsyncCompletions if async_mode else _SyncCompletions
    primary = _client(completions(fail=True), "https://qwen.invalid/v1")
    anthropic = _client(completions(text="anthropic"), "https://api.anthropic.com")
    resolved = []

    def resolve_entry(entry):
        resolved.append(entry["provider"])
        return anthropic, entry["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value=task_config),
        patch.object(aux, "_resolve_task_provider_model", return_value=("auto", "qwen-primary", None, None, None)),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
        patch.object(aux, "_read_main_provider", return_value="qwen"),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.fallback_config.get_fallback_chain", return_value=chain),
    ]
    if async_mode:
        patches.append(patch.object(aux, "_to_async_client", side_effect=lambda client, model, is_vision=False: (client, model)))

    response = _run_fallback_call(
        async_mode, task="moa_aggregator", route_info=route_info, patches=patches,
    )

    assert response.choices[0].message.content == "anthropic"
    assert resolved == ["anthropic"]


def test_ordinary_payment_fallback_skips_discovery_before_resolution():
    """Ordinary auxiliary exhaustion must not resolve a non-Codex discovery rung."""
    discovery_calls = []

    def discover():
        discovery_calls.append(True)
        return _client(_SyncCompletions(text="openrouter"), "https://openrouter.invalid/v1"), "model"

    with (
        patch.object(aux, "_get_provider_chain", return_value=[("openrouter", discover)]),
        patch.object(aux, "_is_provider_unhealthy", return_value=False),
    ):
        result = aux._try_payment_fallback(
            "qwen", "title_generation", codex_only=True,
        )

    assert result == (None, None, "")
    assert discovery_calls == []


def test_explicit_payment_fallback_keeps_non_codex_contract():
    """The Codex-only gate belongs to ordinary auto exhaustion, not this API."""
    client = _client(_SyncCompletions(text="anthropic"), "https://api.anthropic.com")
    with (
        patch.object(
            aux, "_get_provider_chain", return_value=[
                ("anthropic", lambda: (client, "claude-model")),
            ],
        ),
        patch.object(aux, "_is_provider_unhealthy", return_value=False),
    ):
        result = aux._try_payment_fallback("qwen", "compression")

    assert result[0] is client
    assert result[1:] == ("claude-model", "anthropic")


def test_ordinary_main_fallback_filters_before_resolution():
    """Ordinary auto main fallback rejects non-Codex entries pre-resolution."""
    resolved = []
    with (
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch(
            "hermes_cli.fallback_config.get_fallback_chain",
            return_value=[{"provider": "openrouter", "model": "router-model"}],
        ),
        patch.object(
            aux, "_resolve_fallback_entry",
            side_effect=lambda entry: resolved.append(entry),
        ),
    ):
        result = aux._try_main_fallback_chain(
            "compression", "qwen", codex_only=True,
        )

    assert result == (None, None, "")
    assert resolved == []


@pytest.mark.parametrize("async_mode", [False, True])
def test_exact_duplicate_codex_is_resolved_and_attempted_once(async_mode):
    """Duplicate configured destinations do not repeat physical resolution/request."""
    entry = {"provider": "openai-codex", "model": "codex-model", "base_url": "https://chatgpt.com/backend-api/"}
    task_config = {"fallback_chain": [dict(entry), dict(entry)]}
    completions = _AsyncCompletions if async_mode else _SyncCompletions
    primary = _client(completions(fail=True), "https://qwen.invalid/v1")
    codex_seen = []
    codex = _client(completions(fail=True, seen=codex_seen), "https://chatgpt.com/backend-api")
    resolved = []

    def resolve_entry(candidate):
        resolved.append(candidate)
        return codex, candidate["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value=task_config),
        patch.object(aux, "_resolve_task_provider_model", return_value=("auto", "qwen-primary", None, None, None)),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(patch.object(aux, "_to_async_client", side_effect=lambda client, model, is_vision=False: (client, model)))

    _run_fallback_call(
        async_mode, task="title_generation", route_info=route_info, patches=patches,
        raises=True,
    )

    assert len(resolved) == 1
    assert len(codex_seen) == 1


def test_effective_candidate_identity_keeps_distinct_controls_and_scopes():
    """Reasoning, body/cache, endpoint, mode, and credential scope stay distinct."""
    chain = [
        {"provider": "openai-codex", "model": "m", "base_url": "https://same.invalid/v1", "reasoning_effort": "low"},
        {"provider": "openai-codex", "model": "m", "base_url": "https://same.invalid/v1", "reasoning_effort": "high"},
        {"provider": "openai-codex", "model": "m", "base_url": "https://same.invalid/v1", "extra_body": {"cache_destination": "a"}},
        {"provider": "openai-codex", "model": "m", "base_url": "https://same.invalid/v1", "extra_body": {"cache_destination": "b"}},
        {"provider": "openai-codex", "model": "m", "base_url": "https://other.invalid/v1"},
        {"provider": "openai-codex", "model": "m", "base_url": "https://same.invalid/v1", "api_mode": "codex_responses"},
        {"provider": "openai-codex", "model": "m", "base_url": "https://same.invalid/v1", "credential_scope": "one"},
        {"provider": "openai-codex", "model": "m", "base_url": "https://same.invalid/v1", "credential_scope": "two"},
    ]
    resolved = []

    def resolve_entry(entry):
        resolved.append(entry)
        return None, None

    with (
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
    ):
        result = aux._try_configured_fallback_chain(
            "title_generation", "qwen", attempted_indices=set(), attempted_identities=set(),
        )

    assert result == (None, None, "")
    assert len(resolved) == len(chain)


@pytest.mark.parametrize("async_mode", [False, True])
def test_inline_credentials_are_secret_safe_distinct_candidates(async_mode, caplog):
    """Different inline keys are attempted; an exact duplicate is not."""
    keys = ("inline-secret-alpha", "inline-secret-beta")
    chain = [
        {
            "provider": "openai-codex",
            "model": "codex-model",
            "base_url": "https://chatgpt.com/backend-api",
            "api_key": key,
        }
        for key in (*keys, keys[1])
    ]
    completions = _AsyncCompletions if async_mode else _SyncCompletions
    primary = _client(completions(fail=True), "https://qwen.invalid/v1")
    candidate_seen = []
    candidate = _client(
        completions(fail=True, seen=candidate_seen),
        "https://chatgpt.com/backend-api",
    )
    resolved = []

    def resolve_entry(entry):
        resolved.append(entry["api_key"])
        return candidate, entry["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
        patch.object(aux, "_resolve_task_provider_model", return_value=("auto", "qwen-primary", None, None, None)),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(patch.object(aux, "_to_async_client", side_effect=lambda client, model, is_vision=False: (client, model)))

    _run_fallback_call(
        async_mode, task="title_generation", route_info=route_info, patches=patches,
        raises=True,
    )

    assert resolved == list(keys)
    assert len(candidate_seen) == 2
    identities = [aux._fallback_destination_from_entry(entry).identity for entry in chain]
    assert identities[0] != identities[1]
    assert identities[1] == identities[2]
    exposed = repr((identities, route_info, vars(candidate), aux._client_cache, caplog.text))
    assert all(key not in exposed for key in keys)


@pytest.mark.parametrize("async_mode", [False, True])
def test_selected_fallback_controls_survive_config_mutation(async_mode):
    """Dispatch uses the admitted entry after mutation, reorder, and removal."""
    selected = {
        "provider": "openai-codex",
        "model": "codex-model",
        "base_url": "https://chatgpt.com/backend-api",
        "timeout": 10,
        "reasoning_effort": "low",
        "extra_body": {"metadata": {"selected": True}},
    }
    chain = [selected, {"provider": "openai-codex", "model": "other"}]
    task_config = {"fallback_chain": chain}
    completions = _AsyncCompletions if async_mode else _SyncCompletions
    primary = _client(completions(fail=True), "https://qwen.invalid/v1")
    selected_seen = []
    fallback = _client(
        completions(seen=selected_seen), "https://chatgpt.com/backend-api",
    )
    real_record = aux._record_route_info
    mutated = False

    def mutate_after_selection(route_info, provider, model, fallback_label=None):
        nonlocal mutated
        real_record(route_info, provider, model, fallback_label)
        if fallback_label and not mutated:
            mutated = True
            selected["timeout"] = 99
            selected["reasoning_effort"] = "high"
            selected["extra_body"]["metadata"]["selected"] = False
            task_config["fallback_chain"] = [
                {"provider": "openai-codex", "model": "replacement", "timeout": 77}
            ]

    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value=task_config),
        patch.object(aux, "_resolve_task_provider_model", return_value=("auto", "qwen-primary", None, None, None)),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_resolve_fallback_entry", return_value=(fallback, "codex-model")),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
        patch.object(aux, "_record_route_info", side_effect=mutate_after_selection),
    ]
    if async_mode:
        patches.append(patch.object(aux, "_to_async_client", side_effect=lambda client, model, is_vision=False: (client, model)))

    response = _run_fallback_call(
        async_mode, task="title_generation", route_info={}, patches=patches,
    )

    assert response.choices[0].message.content == "codex"
    assert mutated is True
    assert len(selected_seen) == 1
    request = selected_seen[0]
    assert request["timeout"] == 10.0
    assert request["extra_body"]["reasoning"] == {"enabled": True, "effort": "low"}
    assert request["extra_body"]["metadata"] == {"selected": True}


@pytest.mark.parametrize("async_mode", [False, True])
def test_effective_credential_is_bound_once_at_candidate_admission(async_mode):
    """Identity and physical client use one immutable effective credential."""
    primary = _client(
        (_AsyncCompletions if async_mode else _SyncCompletions)(fail=True),
        "https://qwen.invalid/v1",
    )
    candidate = _client(
        (_AsyncCompletions if async_mode else _SyncCompletions)(),
        "https://chatgpt.com/backend-api",
    )
    entry = {
        "provider": "openai-codex",
        "model": "codex-model",
        "key_env": "ROTATING_KEY",
    }
    resolved_keys = []
    key_reads = []

    def resolve_key(_entry):
        key_reads.append(True)
        return ("rotating-alpha", "rotating-beta")[len(key_reads) - 1]

    def resolve_provider(provider, **kwargs):
        resolved_keys.append(kwargs["explicit_api_key"])
        return candidate, kwargs["model"]

    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": [entry]}),
        patch.object(aux, "_resolve_task_provider_model", return_value=("auto", "qwen-primary", None, None, None)),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_fallback_entry_api_key", side_effect=resolve_key),
        patch.object(aux, "resolve_provider_client", side_effect=resolve_provider),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(patch.object(aux, "_to_async_client", side_effect=lambda client, model, is_vision=False: (client, model)))

    response = _run_fallback_call(
        async_mode, task="title_generation", route_info={}, patches=patches,
    )

    destination = candidate._hermes_fallback_destination
    expected = aux._runtime_cache_discriminator("api_key", "rotating-alpha")
    assert response.choices[0].message.content == "codex"
    assert len(key_reads) == 1
    assert resolved_keys == ["rotating-alpha"]
    assert repr(expected) in repr(destination.identity)
    assert "rotating-alpha" not in repr(destination)


@pytest.mark.parametrize("async_mode", [False, True])
def test_fallback_identity_and_metadata_never_expose_request_secrets(async_mode, caplog):
    """Endpoint userinfo and nested controls stay private but affect dedupe."""
    sentinels = ("url-user-secret", "url-password-secret", "nested-control-secret")
    raw_url = "https://url-user-secret:url-password-secret@example.invalid:8443/v1/private?mode=one"
    entry = {
        "provider": "openai-codex",
        "model": "codex-model",
        "base_url": raw_url,
        "extra_body": {"metadata": {"client_secret": sentinels[2]}},
    }
    variants = [
        entry,
        {**entry, "base_url": raw_url.replace("mode=one", "mode=two")},
        {**entry, "extra_body": {"metadata": {"client_secret": "other"}}},
    ]
    primary = _client(
        (_AsyncCompletions if async_mode else _SyncCompletions)(fail=True),
        "https://qwen.invalid/v1",
    )
    candidate = _client(
        (_AsyncCompletions if async_mode else _SyncCompletions)(),
        "https://example.invalid:8443/v1/private",
    )
    physical_urls = []

    def resolve_provider(provider, **kwargs):
        physical_urls.append(kwargs["explicit_base_url"])
        return candidate, kwargs["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": [entry]}),
        patch.object(aux, "_resolve_task_provider_model", return_value=("auto", "qwen-primary", None, None, None)),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "resolve_provider_client", side_effect=resolve_provider),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(patch.object(aux, "_to_async_client", side_effect=lambda client, model, is_vision=False: (client, model)))

    _run_fallback_call(
        async_mode, task="title_generation", route_info=route_info, patches=patches,
    )

    identities = [aux._fallback_destination_from_entry(value).identity for value in variants]
    destination = candidate._hermes_fallback_destination
    exposed = repr((identities, destination, route_info, candidate._hermes_fallback_label, aux._client_cache, caplog.text))
    assert physical_urls == [raw_url]
    assert len(set(identities)) == len(identities)
    assert aux._fallback_destination_from_entry(dict(entry)).identity == identities[0]
    assert "example.invalid" in repr(identities[0])
    assert "8443" in repr(identities[0])
    assert all(secret not in exposed for secret in sentinels)


@pytest.mark.parametrize("async_mode", [False, True])
def test_recursive_yaml_candidate_is_skipped_without_aborting_chain(async_mode, caplog):
    """A recursive safe-loaded body rejects only its malformed candidate."""
    import yaml

    chain = yaml.safe_load("""
- provider: openai-codex
  model: cyclic-model
  extra_body: &cyclic
    payload: cyclic-payload-secret
    self: *cyclic
- provider: openai-codex
  model: valid-model
""")
    assert chain[0]["extra_body"]["self"] is chain[0]["extra_body"]
    primary = _client(
        (_AsyncCompletions if async_mode else _SyncCompletions)(fail=True),
        "https://qwen.invalid/v1",
    )
    valid = _client(
        (_AsyncCompletions if async_mode else _SyncCompletions)(text="valid"),
        "https://chatgpt.com/backend-api",
    )
    resolved = []

    def resolve_entry(entry):
        resolved.append(entry["model"])
        return valid, entry["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
        patch.object(aux, "_resolve_task_provider_model", return_value=("auto", "qwen-primary", None, None, None)),
        patch.object(aux, "_get_cached_client", return_value=(primary, "qwen-primary")),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(patch.object(aux, "_to_async_client", side_effect=lambda client, model, is_vision=False: (client, model)))

    response = _run_fallback_call(
        async_mode, task="title_generation", route_info=route_info, patches=patches,
    )

    assert response.choices[0].message.content == "valid"
    assert resolved == ["valid-model"]
    assert route_info["model"] == "valid-model"
    assert "cyclic-payload-secret" not in repr((route_info, caplog.text))


@pytest.mark.parametrize(
    "provider,api_mode",
    [
        ("openai-codex", "codex_responses"),
        ("xai-oauth", "codex_responses"),
        ("nous", "chat_completions"),
        ("qwen-oauth", "chat_completions"),
        ("custom:command", "chat_completions"),
        ("azure-foundry", "chat_completions"),
    ],
)
def test_implicit_runtime_is_bound_once_for_probe_and_client(provider, api_mode):
    """Admission owns the exact runtime key/callable through construction."""
    credential = (lambda: "token") if provider in {"custom:command", "azure-foundry"} else object()
    runtime = {
        "provider": provider,
        "base_url": "https://bound.example/v1",
        "api_mode": api_mode,
        "api_key": credential,
    }
    resolutions = []
    builds = []

    def resolve_runtime(**kwargs):
        resolutions.append(kwargs)
        return runtime

    def build(provider_name=None, **kwargs):
        builds.append((provider_name, kwargs))
        return _client(_SyncCompletions(), runtime["base_url"]), kwargs["model"]

    with (
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve_runtime),
        patch.object(aux, "resolve_provider_client", side_effect=build),
        patch.object(aux, "_candidate_context_window", return_value=None) as probe,
    ):
        entry = {"provider": provider, "model": "bound-model"}
        destination = aux._fallback_destination_from_entry(entry)
        token = aux._fallback_resolution_destination.set(destination)
        try:
            client, model = aux._resolve_fallback_entry(entry)
        finally:
            aux._fallback_resolution_destination.reset(token)
        aux._candidate_context_window(
            destination.provider,
            destination.model,
            base_url=destination._base_url,
            api_key=destination._api_key,
        )

    assert (client is not None, model) == (True, "bound-model")
    assert len(resolutions) == 1
    assert builds[0][1]["_bound_runtime"] is runtime
    assert destination._bound_runtime is runtime
    assert destination._api_key is credential
    assert probe.call_args.kwargs["api_key"] is credential


def test_main_agent_fallback_admits_one_runtime_for_metadata_and_client():
    runtime = {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_mode": "anthropic_messages",
        "api_key": object(),
    }
    runtime_calls = []
    physical_calls = []
    destinations = {}
    client = _client(_SyncCompletions(), runtime["base_url"])

    def resolve_runtime(**kwargs):
        runtime_calls.append(kwargs)
        return runtime

    def resolve_client(*args, **kwargs):
        physical_calls.append((args, kwargs))
        return client, "main-model"

    with (
        patch.object(aux, "_read_main_provider", return_value="anthropic"),
        patch.object(aux, "_read_main_model", return_value="main-model"),
        patch.object(aux, "_is_provider_unhealthy", return_value=False),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve_runtime),
        patch.object(aux, "resolve_provider_client", side_effect=resolve_client),
    ):
        result = aux._try_main_agent_model_fallback(
            "qwen",
            failed_model="qwen-model",
            candidate_destinations=destinations,
            codex_only=False,
        )

    assert result == (client, "main-model", "main-agent(anthropic)")
    assert len(runtime_calls) == 1
    assert physical_calls[0][1]["_bound_runtime"] is runtime
    assert destinations["main-agent(anthropic)"]._bound_runtime is runtime


@pytest.mark.parametrize("async_mode", [False, True])
def test_auth_recovery_retires_plan_and_admits_one_new_runtime(async_mode):
    runtime_a = {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_mode": "anthropic_messages",
        "api_key": object(),
    }
    runtime_b = {**runtime_a, "api_key": object()}
    runtimes = iter((runtime_a, runtime_b))
    runtime_calls = []
    physical_bound = []

    class AuthCompletions:
        def create(self, **kwargs):
            error = RuntimeError("expired")
            setattr(error, "status_code", 401)
            raise error

    class AsyncAuthCompletions:
        async def create(self, **kwargs):
            error = RuntimeError("expired")
            setattr(error, "status_code", 401)
            raise error

    def resolve_runtime(**kwargs):
        runtime_calls.append(kwargs)
        return next(runtimes)

    retry_completions = _AsyncCompletions(text="recovered") if async_mode else _SyncCompletions(text="recovered")
    retry_client = _client(retry_completions, runtime_b["base_url"])

    def resolve_client(*args, **kwargs):
        physical_bound.append(kwargs["_bound_runtime"])
        return retry_client, "fallback-model"

    initial_completions = AsyncAuthCompletions() if async_mode else AuthCompletions()
    initial_client = _client(initial_completions, runtime_a["base_url"])
    with (
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve_runtime),
        patch.object(aux, "resolve_provider_client", side_effect=resolve_client),
        patch.object(aux, "_refresh_provider_credentials", return_value=True),
        patch.object(aux, "_get_cached_client", return_value=(retry_client, "fallback-model")),
        patch.object(aux, "_replan_synchronous_cache_sections", side_effect=lambda messages, tools, **kwargs: (messages, tools or [])),
    ):
        destination = aux._fallback_destination_from_entry({
            "provider": "anthropic",
            "model": "fallback-model",
            "base_url": runtime_a["base_url"],
        })
        if async_mode:
            response = asyncio.run(aux._call_fallback_candidate_async(
                initial_client, "fallback-model", "fallback_chain[0]",
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
                temperature=None,
                max_tokens=64,
                tools=None,
                effective_timeout=10.0,
                effective_extra_body={},
                reasoning_config=None,
                destination=destination,
            ))
        else:
            response = aux._call_fallback_candidate_sync(
                initial_client, "fallback-model", "fallback_chain[0]",
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
                temperature=None,
                max_tokens=64,
                tools=None,
                effective_timeout=10.0,
                effective_extra_body={},
                reasoning_config=None,
                destination=destination,
            )

    assert aux.extract_content_or_reasoning(response) == "recovered"
    assert len(runtime_calls) == 2
    assert physical_bound == [runtime_b]


def test_selected_fallback_plan_fails_closed_on_every_serialization_surface():
    """The attached private plan remains useful but cannot be copied or encoded."""
    secret = "plan-secret-sentinel"
    endpoint = "https://user:password@example.invalid:8443/private?q=secret#fragment"
    runtime = {
        "provider": "custom",
        "base_url": endpoint,
        "api_mode": "chat_completions",
        "api_key": secret,
    }
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime):
        plan = aux._fallback_destination_from_entry({
            "provider": "custom",
            "model": "safe-model",
            "base_url": endpoint,
            "api_key": secret,
            "extra_body": {"metadata": {"secret": secret}},
        })

    client = SimpleNamespace(_hermes_fallback_destination=plan)

    @dataclasses.dataclass
    class Envelope:
        value: object

    assert not dataclasses.is_dataclass(plan)
    assert not hasattr(plan, "__dict__")
    assert "example.invalid:8443" in repr(plan)
    assert all(value not in repr(plan) for value in (secret, "user", "password", "private", "q=secret"))
    with pytest.raises(TypeError):
        vars(plan)
    with pytest.raises(TypeError):
        dataclasses.asdict(plan)
    for operation in (
        lambda: pickle.dumps(plan),
        lambda: copy.copy(plan),
        lambda: copy.deepcopy(plan),
        lambda: pickle.dumps(client),
        lambda: dataclasses.asdict(Envelope(plan)),
        lambda: pickle.dumps({"client": client, "plan": plan}),
    ):
        with pytest.raises(TypeError, match="fallback plan is not serializable"):
            operation()
    for operation in (lambda: json.dumps(plan), lambda: json.dumps(plan, default=vars)):
        with pytest.raises(TypeError):
            operation()
    assert secret not in json.dumps(plan, default=str)
    async_client = SimpleNamespace()
    aux._tag_fallback_client(client, "fallback_chain[0](custom)", destination=plan)
    aux._copy_fallback_selection(client, async_client)
    assert async_client._hermes_fallback_destination is plan


_MALFORMED_ENDPOINTS = [
    "https://[::1",
    "https://[not-ip]/v1",
    "https://example.invalid：443/v1",
    "https://example.invalid:abc/v1",
    "https://example.invalid:65536/v1",
    "https://2001:db8::1/v1",
    "https://" + "a" * 64 + ".invalid/v1",
    "https://\ud800.invalid/v1",
    "https://example.invalid/\ud800",
    "https://example.invalid/%",
    "https://example.invalid/%2",
    "https://example.invalid/%GG",
    "https:///missing-host",
    "HTTPS:///missing-host",
    r"HTTP:\missing-host",
    r"HtTpS:\missing-host",
    r"HtTpS:\\missing-host",
]


@pytest.mark.parametrize("endpoint", _MALFORMED_ENDPOINTS)
def test_malformed_endpoint_is_rejected_before_runtime_or_control_hooks(endpoint):
    class Hook:
        calls = 0

        def __repr__(self):
            type(self).calls += 1
            raise AssertionError("repr hook")

        __str__ = __repr__
        __deepcopy__ = __repr__
        __reduce__ = __repr__
        __reduce_ex__ = __repr__

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as resolver:
        with pytest.raises(aux._FallbackCandidateRejected, match="^invalid endpoint$"):
            aux._fallback_destination_from_entry({
                "provider": "openai-codex",
                "model": "bad-model",
                "base_url": endpoint,
                "extra_body": {"value": Hook()},
            })
    resolver.assert_not_called()
    assert Hook.calls == 0


@pytest.mark.parametrize("endpoint", _MALFORMED_ENDPOINTS)
@pytest.mark.parametrize("async_mode", [False, True])
def test_malformed_endpoint_continues_in_codex_chain_without_payload_egress(endpoint, async_mode, caplog):
    chain = [
        {"provider": "openai-codex", "model": "bad", "base_url": endpoint},
        {"provider": "openai-codex", "model": "good", "base_url": "https://good.example/v1"},
    ]
    primary = _client(
        (_AsyncCompletions if async_mode else _SyncCompletions)(fail=True),
        "https://primary.example/v1",
    )
    good = _client(
        (_AsyncCompletions if async_mode else _SyncCompletions)(text="good"),
        "https://good.example/v1",
    )
    resolved = []
    runtimes = []

    def runtime(**kwargs):
        runtimes.append(kwargs["target_model"])
        return {
            "provider": "openai-codex",
            "base_url": kwargs.get("explicit_base_url") or "https://good.example/v1",
            "api_mode": "codex_responses",
            "api_key": "bound-key",
        }

    def resolve(entry):
        resolved.append(entry["model"])
        return good, entry["model"]

    route_info = {}
    patches = [
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
        patch.object(aux, "_resolve_task_provider_model", return_value=("auto", "primary", None, None, None)),
        patch.object(aux, "_get_cached_client", return_value=(primary, "primary")),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=runtime),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_provider_requires_stream", return_value=False),
    ]
    if async_mode:
        patches.append(patch.object(aux, "_to_async_client", side_effect=lambda client, model, is_vision=False: (client, model)))

    response = _run_fallback_call(
        async_mode, task="title_generation", route_info=route_info, patches=patches,
    )

    assert response.choices[0].message.content == "good"
    assert resolved == ["good"]
    assert runtimes == ["good"]
    assert route_info["model"] == "good"
    assert endpoint not in caplog.text


def test_malformed_endpoint_continues_for_moa_non_codex_candidate(caplog):
    endpoint = "https://[moa-private-endpoint"
    chain = [
        {"provider": "anthropic", "model": "bad", "base_url": endpoint},
        {"provider": "anthropic", "model": "good", "base_url": "https://api.anthropic.com"},
    ]
    good = _client(_SyncCompletions(text="good"), "https://api.anthropic.com")
    resolved = []

    def resolve(entry):
        resolved.append(entry["model"])
        return good, entry["model"]

    with (
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve),
        patch.object(aux, "_is_provider_unhealthy", return_value=False),
    ):
        client, model, label = aux._try_configured_fallback_chain(
            "moa_reference", "qwen", codex_only=False,
        )

    assert (client, model, label) == (good, "good", "fallback_chain[1](anthropic)")
    assert resolved == ["good"]
    assert endpoint not in caplog.text


def test_malformed_endpoint_continues_in_main_fallback_chain(caplog):
    endpoint = "https://[main-private-endpoint"
    chain = [
        {"provider": "openai-codex", "model": "bad", "base_url": endpoint},
        {"provider": "openai-codex", "model": "good", "base_url": "https://good.example/v1"},
    ]
    good = _client(_SyncCompletions(text="good"), "https://good.example/v1")
    resolved = []
    destinations = {}

    def resolve(entry):
        resolved.append(entry["model"])
        return good, entry["model"]

    with (
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.fallback_config.get_fallback_chain", return_value=chain),
        patch.object(aux, "_read_main_provider", return_value="qwen"),
        patch.object(aux, "_is_provider_unhealthy", return_value=False),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve),
    ):
        client, model, provider = aux._try_main_fallback_chain(
            "title_generation",
            failed_provider="qwen",
            candidate_destinations=destinations,
            codex_only=True,
        )

    assert (client, model, provider) == (good, "good", "openai-codex")
    assert resolved == ["good"]
    assert list(destinations) == ["openai-codex"]
    assert endpoint not in caplog.text
