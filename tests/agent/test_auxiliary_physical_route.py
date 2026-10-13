"""A physical auxiliary retry publishes the destination it actually sends."""

import agent.auxiliary_client as aux


class _Client:
    base_url = "https://recovered.example/v1"
    api_key = "recovered-key"


def test_record_physical_route_keeps_recovered_destination(monkeypatch):
    seen = {}

    def _set(provider, model, api_mode):
        seen["relay"] = (provider, model, api_mode)

    monkeypatch.setattr(aux, "_set_relay_auxiliary_route", _set)
    monkeypatch.setattr(aux, "_fallback_provider_from_label", lambda label: label)
    monkeypatch.setattr(aux, "_effective_provider_for_client", lambda client, provider: "custom")

    route = {"provider": "stale", "model": "old"}
    aux._record_physical_route(
        route,
        "custom",
        _Client(),
        {"model": "recovered-model", "timeout": 12},
        "chat_completions",
    )

    assert route["provider"] == "custom"
    assert route["model"] == "recovered-model"
    assert route["base_url"] == "https://recovered.example/v1"
    assert route["api_key"] == "recovered-key"
    assert route["api_mode"] == "chat_completions"
    assert route["timeout"] == 12
    assert "stale" not in route.values()
    assert seen["relay"] == ("custom", "recovered-model", "chat_completions")
    assert "fallback_label" not in route


def test_record_physical_route_keeps_configured_chain_label(monkeypatch):
    monkeypatch.setattr(aux, "_set_relay_auxiliary_route", lambda *_args: None)
    monkeypatch.setattr(aux, "_fallback_provider_from_label", lambda label: "openai-codex")
    monkeypatch.setattr(aux, "_effective_provider_for_client", lambda client, provider: provider)

    route = {}
    aux._record_physical_route(
        route,
        "openai-codex",
        _Client(),
        {"model": "codex-model", "timeout": 37},
        "codex_responses",
        fallback_label="fallback_chain[0](openai-codex)",
    )

    assert route["fallback_label"] == "fallback_chain[0](openai-codex)"
    assert route["provider"] == "openai-codex"
    assert route["base_url"] == "https://recovered.example/v1"
