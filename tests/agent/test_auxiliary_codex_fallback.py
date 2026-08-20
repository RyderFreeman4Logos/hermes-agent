"""#159 auxiliary fallback routing and candidate-owned controls."""

import asyncio
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
            return_value=("qwen", "qwen-primary", None, None, None),
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

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        if async_mode:
            with patches[6]:
                response = asyncio.run(
                    aux._async_call_llm_impl_unscoped(
                        task="title_generation",
                        messages=[{"role": "user", "content": "hello"}],
                        route_info=route_info,
                    )
                )
        else:
            response = aux._call_llm_impl_unscoped(
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
                route_info=route_info,
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
            return_value=("qwen", "qwen-primary", None, None, None),
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

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        if async_mode:
            with patches[6]:
                response = asyncio.run(
                    aux._async_call_llm_impl_unscoped(
                        task="title_generation",
                        messages=[{"role": "user", "content": "hello"}],
                        route_info=route_info,
                    )
                )
        else:
            response = aux._call_llm_impl_unscoped(
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
                route_info=route_info,
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
            return_value=("qwen", "qwen-primary", None, None, None),
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

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with pytest.raises(RuntimeError, match="primary rate limited"):
            if async_mode:
                with patches[6]:
                    asyncio.run(
                        aux._async_call_llm_impl_unscoped(
                            task="title_generation",
                            messages=[{"role": "user", "content": "hello"}],
                            route_info=route_info,
                        )
                    )
            else:
                aux._call_llm_impl_unscoped(
                    task="title_generation",
                    messages=[{"role": "user", "content": "hello"}],
                    route_info=route_info,
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
        )

    assert client is None
    assert model is None
    assert label == ""
    assert resolved == ["openai-codex"]
    assert route_info["codex_skip_reason"] == "unavailable"


def test_lean_digest_workers_reuse_selected_codex_route_settings():
    """Parallel lean workers carry the selected entry-owned route controls."""
    calls = []

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        assert task == "compression"
        calls.append(kwargs)
        route_info = kwargs.get("route_info")
        if route_info is not None and not route_info:
            route_info.update(
                provider="openai-codex",
                model="codex-model",
                fallback_label="fallback_chain[1](openai-codex)",
            )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="DIGEST"))]
        )
        return response

    turns = [
        {"role": "user", "content": "MARKER-A " + ("a" * 70)},
        {"role": "user", "content": "MARKER-B " + ("b" * 70)},
        {"role": "user", "content": "MARKER-C " + ("c" * 70)},
    ]
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 88),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch("agent.auxiliary_client._get_task_max_concurrency", return_value=3),
    ):
        compressor._build_chunk_digests(turns)

    assert len(calls) == 3
    assert calls[0]["route_info"]["fallback_label"] == "fallback_chain[1](openai-codex)"
    for call in calls[1:]:
        assert call["provider"] == "openai-codex"
        assert call["model"] == "codex-model"
        assert call["route_info"] == {
            "fallback_label": "fallback_chain[1](openai-codex)"
        }
