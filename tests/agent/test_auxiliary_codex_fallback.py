"""#160 Codex-only auxiliary fallback destinations on official resolver APIs."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import auxiliary_client as aux
from agent.context_compressor import ContextCompressor


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
