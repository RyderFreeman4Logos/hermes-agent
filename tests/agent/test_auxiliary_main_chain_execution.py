"""Public main-chain recovery uses only configured Codex destinations."""
import asyncio
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
def test_public_main_chain_advances_once_and_preserves_primary(monkeypatch, tmp_path, async_mode, exhausted):
    from hermes_cli import config

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "auxiliary:\n  classifier:\n    provider: auto\n"
        "fallback_providers:\n"
        "  - {provider: nous, model: forbidden}\n"
        "  - {provider: openai-codex, model: first, base_url: 'https://first.invalid/v1'}\n"
        "  - {provider: openai-codex, model: second, base_url: 'https://second.invalid/v1'}\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(config, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(config, "_RAW_CONFIG_CACHE", {})
    aux._reset_aux_unhealthy_cache()
    attempts = []
    original = RuntimeError("primary execution failed")

    def send(model):
        attempts.append(model)
        if model == "primary":
            raise original
        if model == "first" or exhausted:
            raise RuntimeError("candidate execution failed")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK", tool_calls=None))])

    def client(model, base):
        def create(**kwargs):
            return send(model)
        async def acreate(**kwargs):
            return send(model)
        return SimpleNamespace(base_url=base, chat=SimpleNamespace(completions=SimpleNamespace(create=acreate if async_mode else create)))

    primary = client("primary", "https://primary.invalid/v1")
    monkeypatch.setattr(aux, "_get_cached_client", lambda *args, **kwargs: (primary, "primary"))
    def resolve(provider, model, **kwargs):
        assert provider == "openai-codex"
        return client(model, kwargs.get("base_url") or f"https://{model}.invalid/v1"), model
    monkeypatch.setattr(aux, "resolve_provider_client", resolve)
    monkeypatch.setattr(aux, "_to_async_client", lambda c, m, **kwargs: (c, m))
    monkeypatch.setattr(aux, "_try_payment_fallback", lambda *args, **kwargs: (None, None, ""))
    def invoke():
        kwargs = dict(task="classifier", messages=[{"role": "user", "content": "hi"}])
        return asyncio.run(aux.async_call_llm(**kwargs)) if async_mode else aux.call_llm(**kwargs)
    try:
        if exhausted:
            with pytest.raises(RuntimeError) as raised:
                invoke()
            assert raised.value is original
        else:
            assert invoke().choices[0].message.content == "OK"
        assert attempts == ["primary", "first", "second"]
    finally:
        aux._reset_aux_unhealthy_cache()


def test_public_unselected_accounts_are_not_discovery_destinations(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    from hermes_cli import config
    from agent.auxiliary_unavailable import AuxiliaryClientUnavailable

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model: {provider: auto}\nauxiliary:\n  classifier: {provider: auto}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "unselected-account-canary")
    monkeypatch.setattr(config, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(config, "_RAW_CONFIG_CACHE", {})
    monkeypatch.setattr(aux, "_read_nous_auth", lambda: {"access_token": "unselected-account-canary"})
    construct = MagicMock()
    monkeypatch.setattr(aux, "OpenAI", construct)
    with pytest.raises(AuxiliaryClientUnavailable):
        aux.call_llm(task="classifier", messages=[{"role": "user", "content": "hi"}], main_runtime={"provider": "auto"})
    construct.assert_not_called()
