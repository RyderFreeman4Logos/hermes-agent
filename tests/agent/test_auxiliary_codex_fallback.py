"""#160 Codex-only auxiliary fallback destinations on official resolver APIs."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux
from agent.context_compressor import ContextCompressor

_NON_CODEX_CHAIN_PROVIDERS = (
    "custom",
    "custom:",
    "custom:localrouter",
    "local/custom",
    "pm",
    "nous",
    "openrouter",
)


def _client(text, base_url):
    return SimpleNamespace(
        base_url=base_url,
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
                )
            )
        ),
    )


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _configure_classifier_fallbacks(monkeypatch, tmp_path, count=1):
    from hermes_cli import config as config_mod

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    fallback_chain = "\n".join(
        f"      - provider: openai-codex\n        model: codex-{index}"
        for index in range(count)
    )
    (hermes_home / "config.yaml").write_text(
        "\n".join((
            "auxiliary:", "  classifier:", "    provider: primary-provider",
            "    model: primary-model", "    fallback_chain:", fallback_chain,
        )),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(config_mod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})


def _chat_client(create, base_url):
    return SimpleNamespace(
        base_url=base_url,
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


def test_configured_chain_returns_codex_not_cheaper_prefix():
    """Configured fallback must not settle on pm/grok before Codex."""
    chain = [
        {"provider": "pm", "model": "pm-model"},
        {"provider": "openai-codex", "model": "codex-model"},
        {"provider": "custom:localrouter", "model": "grok-model"},
    ]
    resolved = []
    route_info = {}
    clients = {
        "pm": _client("pm", "https://pm.invalid/v1"),
        "openai-codex": _client("codex", "https://chatgpt.com/backend-api"),
        "custom:localrouter": _client("grok", "https://local.invalid/v1"),
    }

    def resolve_entry(entry):
        resolved.append(entry["provider"])
        return clients[entry["provider"]], entry["model"]

    with (
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
    ):
        client, model, label = aux._try_configured_fallback_chain(
            "title_generation",
            "qwen",
            reason="rate limit",
            route_info=route_info,
        )

    assert model == "codex-model"
    assert "openai-codex" in label
    assert client is clients["openai-codex"]
    assert resolved == ["openai-codex"]
    assert route_info.get("fallback_label", label).startswith("fallback_chain[")


def test_configured_chain_fails_closed_without_codex():
    """Exhausted / absent Codex must not fall through to pm."""
    chain = [{"provider": "pm", "model": "pm-model"}]
    resolved = []
    route_info = {}

    def resolve_entry(entry):
        resolved.append(entry["provider"])
        return _client("pm", "https://pm.invalid/v1"), entry["model"]

    with (
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
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
    assert resolved == []
    assert route_info.get("codex_skip_reason") == "unavailable"


@pytest.mark.parametrize("provider", _NON_CODEX_CHAIN_PROVIDERS)
def test_configured_chain_rejects_non_codex_entry(provider):
    """Every non-Codex fallback_chain label fails closed (#160)."""
    chain = [{"provider": provider, "model": "forbidden-model"}]
    resolved = []

    def resolve_entry(entry):
        resolved.append(entry["provider"])
        return _client("leak", "https://forbidden.invalid/v1"), entry["model"]

    with (
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
    ):
        client, model, label = aux._try_configured_fallback_chain(
            "title_generation",
            "openai-codex",
            reason="payment error",
        )

    assert client is None
    assert model is None
    assert label == ""
    assert resolved == []


def test_configured_chain_keeps_valid_codex_after_rejected_prefix():
    """Rejected custom/aggregator prefixes must not block a later Codex entry."""
    chain = [
        {"provider": "custom", "model": "custom-model"},
        {"provider": "custom:localrouter", "model": "grok-model"},
        {"provider": "openai-codex", "model": "codex-model"},
        {"provider": "openrouter", "model": "or-model"},
    ]
    resolved = []
    clients = {
        "openai-codex": _client("codex", "https://chatgpt.com/backend-api"),
    }

    def resolve_entry(entry):
        resolved.append(entry["provider"])
        return clients[entry["provider"]], entry["model"]

    with (
        patch.object(aux, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
        patch.object(aux, "_resolve_fallback_entry", side_effect=resolve_entry),
    ):
        client, model, label = aux._try_configured_fallback_chain(
            "title_generation",
            "qwen",
            reason="rate limit",
        )

    assert model == "codex-model"
    assert "openai-codex" in label
    assert client is clients["openai-codex"]
    assert resolved == ["openai-codex"]


def test_distinct_custom_endpoint_cache_isolation_at_shared_seam():
    """Hosted custom billing state must not share the local custom cache key."""
    hosted_url = "https://hosted.example/v1"
    local_url = "http://127.0.0.1:8080/v1"
    aux._reset_aux_unhealthy_cache()
    try:
        aux._mark_provider_unhealthy("custom", base_url=hosted_url)
        assert aux._is_provider_unhealthy("custom", hosted_url) is True
        assert aux._is_provider_unhealthy("custom", local_url) is False
        assert aux._is_provider_unhealthy("custom:") is False
        assert aux._is_provider_unhealthy("custom:localrouter", local_url) is False
        assert aux._is_provider_unhealthy("local/custom", local_url) is False
    finally:
        aux._reset_aux_unhealthy_cache()


def test_payment_fallback_skips_non_codex_discovery():
    """Payment/discovery fallback must not return Nous/OpenRouter."""
    nous = _client("nous", "https://nous.invalid/v1")
    with (
        patch.object(aux, "_try_openrouter", return_value=(None, None)),
        patch.object(aux, "_try_nous", return_value=(nous, "nous-model")),
        patch.object(aux, "_try_custom_endpoint", return_value=(None, None)),
        patch.object(aux, "_resolve_api_key_provider", return_value=(None, None)),
        patch.object(aux, "_read_main_provider", return_value="openrouter"),
        patch.object(aux, "_is_provider_unhealthy", return_value=False),
    ):
        client, model, label = aux._try_payment_fallback("openrouter", task="compression")

    assert client is None
    assert model is None
    assert label == ""


def test_main_agent_fallback_fails_closed_for_non_codex():
    """Non-Codex main-agent safety nets fail closed."""
    with (
        patch.object(aux, "_read_main_provider", return_value="qwen"),
        patch.object(aux, "_read_main_model", return_value="qwen-model"),
        patch.object(aux, "resolve_provider_client") as resolve,
    ):
        client, model, label = aux._try_main_agent_model_fallback(
            "openrouter", task="compression", reason="rate limit"
        )

    assert client is None
    assert model is None
    assert label == ""
    resolve.assert_not_called()


def test_discovery_skips_non_codex_destinations():
    """Auto discovery must not return OpenRouter/Nous."""
    nous = _client("nous", "https://nous.invalid/v1")
    with (
        patch.object(aux, "_try_openrouter", return_value=(MagicMock(), "or-model")),
        patch.object(aux, "_try_nous", return_value=(nous, "nous-model")),
        patch.object(aux, "_try_custom_endpoint", return_value=(None, None)),
        patch.object(aux, "_resolve_api_key_provider", return_value=(None, None)),
        patch.object(aux, "_is_provider_unhealthy", return_value=False),
    ):
        client, model, label = aux._try_discovery_chain()

    assert client is None
    assert model is None
    assert label == ""


def test_vision_auto_order_is_codex_only():
    assert aux._VISION_AUTO_PROVIDER_ORDER == ("openai-codex",)


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
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="DIGEST"))]
        )

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


def test_public_sync_generic_error_advances_failed_candidate_to_later_codex_hop(monkeypatch, tmp_path):
    _configure_classifier_fallbacks(monkeypatch, tmp_path, count=2)
    calls = []
    tool = {"type": "function", "function": {"name": "once", "description": "once", "parameters": {"type": "object"}}}

    def primary(**kwargs):
        calls.append(("primary", kwargs))
        raise ValueError("upstream transport: safety filter cache unavailable")

    def failed_candidate(**kwargs):
        calls.append(("codex-0", kwargs))
        raise ValueError("approval denied cache unavailable")

    def successful_candidate(**kwargs):
        calls.append(("codex-1", kwargs))
        return _response("later hop")

    primary_client = _chat_client(primary, "https://primary.invalid/v1")
    fallback_clients = iter((
        _chat_client(failed_candidate, "https://codex-0.invalid/v1"),
        _chat_client(successful_candidate, "https://codex-1.invalid/v1"),
    ))
    monkeypatch.setattr(aux, "_get_cached_client", lambda _provider, model, **_kwargs: (primary_client, model))
    monkeypatch.setattr(aux, "resolve_provider_client", lambda _provider, model, **_kwargs: (next(fallback_clients), model))
    monkeypatch.setattr(aux, "_try_main_agent_model_fallback", lambda *_args, **_kwargs: (None, None, ""))

    route_info = {}
    result = aux.call_llm(task="classifier", messages=[{"role": "user", "content": "hi"}], tools=[tool], route_info=route_info)

    assert result.choices[0].message.content == "later hop"
    assert [name for name, _ in calls] == ["primary", "codex-0", "codex-1"]
    assert all(kwargs["tools"] == [tool] for _, kwargs in calls)
    assert route_info["fallback_label"].startswith("fallback_chain[1]")


def test_public_async_generic_error_advances_to_codex(monkeypatch, tmp_path):
    _configure_classifier_fallbacks(monkeypatch, tmp_path)
    calls = []

    async def primary(**_kwargs):
        calls.append("primary")
        raise ValueError("approval denied cache unavailable")

    async def fallback(**_kwargs):
        calls.append("codex")
        return _response("async fallback")

    primary_client = _chat_client(primary, "https://primary.invalid/v1")
    fallback_client = _chat_client(fallback, "https://codex.invalid/v1")
    monkeypatch.setattr(aux, "_get_cached_client", lambda _provider, model, **_kwargs: (primary_client, model))
    monkeypatch.setattr(aux, "resolve_provider_client", lambda _provider, model, **_kwargs: (fallback_client, model))
    monkeypatch.setattr(aux, "_to_async_client", lambda client, model, **_kwargs: (client, model))

    result = asyncio.run(aux.async_call_llm(task="classifier", messages=[{"role": "user", "content": "hi"}]))

    assert result.choices[0].message.content == "async fallback"
    assert calls == ["primary", "codex"]


def test_public_forced_stream_generic_error_advances_to_codex(monkeypatch, tmp_path):
    _configure_classifier_fallbacks(monkeypatch, tmp_path)
    calls = []

    def primary(**kwargs):
        calls.append(("primary", kwargs))
        assert kwargs["stream"] is True
        raise ValueError("content policy cache unavailable")

    def fallback(**kwargs):
        calls.append(("codex", kwargs))
        assert kwargs["stream"] is True
        chunk = SimpleNamespace(
            id="stream-1", model="codex-0", usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="stream fallback", reasoning=None, reasoning_content=None, reasoning_details=None, tool_calls=None),
                finish_reason="stop",
            )],
        )
        return iter((chunk,))

    primary_client = _chat_client(primary, "https://primary.invalid/v1")
    fallback_client = _chat_client(fallback, "https://codex.invalid/v1")
    monkeypatch.setattr(aux, "_get_cached_client", lambda _provider, model, **_kwargs: (primary_client, model))
    monkeypatch.setattr(aux, "resolve_provider_client", lambda _provider, model, **_kwargs: (fallback_client, model))
    monkeypatch.setattr(aux, "_provider_requires_stream", lambda *_args: True)

    result = aux.call_llm(task="classifier", messages=[{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == "stream fallback"
    assert [name for name, _ in calls] == ["primary", "codex"]


def test_public_exhaustion_reraises_original_after_admitted_candidate(monkeypatch, tmp_path):
    _configure_classifier_fallbacks(monkeypatch, tmp_path)
    attempts = []
    original = ValueError("safety filter cache unavailable")

    def primary(**_kwargs):
        attempts.append("primary")
        raise original

    def fallback(**_kwargs):
        attempts.append("codex")
        raise ValueError("candidate unavailable")

    primary_client = _chat_client(primary, "https://primary.invalid/v1")
    fallback_client = _chat_client(fallback, "https://codex.invalid/v1")
    monkeypatch.setattr(aux, "_get_cached_client", lambda _provider, model, **_kwargs: (primary_client, model))
    monkeypatch.setattr(aux, "resolve_provider_client", lambda _provider, model, **_kwargs: (fallback_client, model))
    monkeypatch.setattr(aux, "_try_main_agent_model_fallback", lambda *_args, **_kwargs: (None, None, ""))

    with pytest.raises(ValueError) as raised:
        aux.call_llm(task="classifier", messages=[{"role": "user", "content": "hi"}])

    assert raised.value is original
    assert attempts == ["primary", "codex"]


def test_public_explicit_cancellation_does_not_activate_fallback(monkeypatch, tmp_path):
    _configure_classifier_fallbacks(monkeypatch, tmp_path)
    primary_client = _chat_client(
        lambda **_kwargs: (_ for _ in ()).throw(aux.AuxiliaryExplicitCancellation()),
        "https://primary.invalid/v1",
    )
    resolve = MagicMock()
    monkeypatch.setattr(aux, "_get_cached_client", lambda _provider, model, **_kwargs: (primary_client, model))
    monkeypatch.setattr(aux, "resolve_provider_client", resolve)

    with pytest.raises(aux.AuxiliaryExplicitCancellation):
        aux.call_llm(task="classifier", messages=[{"role": "user", "content": "hi"}])

    resolve.assert_not_called()


def test_public_structured_policy_denial_does_not_fallback(monkeypatch, tmp_path):
    _configure_classifier_fallbacks(monkeypatch, tmp_path)

    class ExplicitPolicyError(ValueError):
        body = {"error": {"code": "content_policy_violation"}}

    original = ExplicitPolicyError("provider rejected request")
    primary_client = _chat_client(lambda **_kwargs: (_ for _ in ()).throw(original), "https://primary.invalid/v1")
    resolve = MagicMock()
    monkeypatch.setattr(aux, "_get_cached_client", lambda _provider, model, **_kwargs: (primary_client, model))
    monkeypatch.setattr(aux, "resolve_provider_client", resolve)

    with pytest.raises(ExplicitPolicyError) as raised:
        aux.call_llm(task="classifier", messages=[{"role": "user", "content": "hi"}])

    assert raised.value is original
    resolve.assert_not_called()
