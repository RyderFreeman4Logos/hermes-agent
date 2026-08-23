"""#159 auxiliary fallback routing and candidate-owned controls."""

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import auxiliary_client as aux
from agent.context_compressor import ContextCompressor


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
