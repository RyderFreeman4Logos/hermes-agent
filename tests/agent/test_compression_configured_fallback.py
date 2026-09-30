"""Compression must honor explicitly configured non-Codex rescue routes."""

import copy
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import auxiliary_client as aux
from agent.context_compressor import ContextCompressor


def _setup(monkeypatch, tmp_path, codex_fails, grok_fails=False):
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    provider: deepinfra.man\n"
        "    model: deepseek-ai/DeepSeek-V4-Flash-0731\n"
        "    fallback_chain:\n      - provider: openai-codex\n        model: gpt-6-luna\n"
        "      - provider: localrouter\n        model: grok-4.7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    aux._reset_aux_unhealthy_cache()
    monkeypatch.setattr(aux, "_client_cache", {})
    calls = []
    primary_error = RuntimeError("You need positive balance to do inference")
    primary_error.status_code = 402

    def client(provider, failure=None):
        def create(**kwargs):
            calls.append((provider, kwargs["model"]))
            if failure is not None:
                raise failure
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="Verified summary", tool_calls=None),
                finish_reason="stop",
            )])
        return SimpleNamespace(
            base_url=f"https://{provider}.invalid/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )

    primary = client("deepinfra.man", primary_error)
    codex = client("openai-codex", TimeoutError("Codex stream stalled") if codex_fails else None)
    grok = client("localrouter", primary_error if grok_fails else None)
    clients = {"openai-codex": codex, "localrouter": grok}

    def resolve(provider, model=None, **kwargs):
        # A primary/main credential must never be forwarded to another destination.
        assert kwargs.get("explicit_api_key") is None
        return clients[provider], model

    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (primary, "deepseek-ai/DeepSeek-V4-Flash-0731"))
    monkeypatch.setattr(aux, "resolve_provider_client", resolve)
    monkeypatch.setattr(aux, "_candidate_context_window", lambda *a, **kw: 256000)
    monkeypatch.setattr(aux, "_try_main_agent_model_fallback", lambda *a, **kw: (None, None, ""))
    return calls, primary_error


@pytest.mark.parametrize("codex_fails", [False, True])
def test_summary_402_uses_gpt_then_configured_grok(monkeypatch, tmp_path, codex_fails):
    calls, _ = _setup(monkeypatch, tmp_path, codex_fails)
    compressor = ContextCompressor(model="gpt-6.1-sol", quiet_mode=True)
    compressor._record_aux_compression_call = Mock()
    assert compressor._call_summary_llm("Sanitized test summary", time.monotonic()) == "Verified summary"
    assert calls == [
        ("deepinfra.man", "deepseek-ai/DeepSeek-V4-Flash-0731"),
        ("openai-codex", "gpt-6-luna"),
    ] + ([("localrouter", "grok-4.7")] if codex_fails else [])
    aux._reset_aux_unhealthy_cache()


def test_all_configured_summary_routes_exhausted_preserves_messages(monkeypatch, tmp_path):
    calls, primary_error = _setup(monkeypatch, tmp_path, True, True)
    messages = [
        {"role": role, "content": f"Sanitized turn {i}"}
        for i in range(12) for role in ("user", "assistant")
    ]
    before = copy.deepcopy(messages)
    compressor = ContextCompressor(
        model="gpt-6.1-sol", quiet_mode=True, protect_first_n=2,
        protect_last_n=2, abort_on_summary_failure=True,
    )
    compressor._record_aux_compression_call = Mock()
    result = compressor.compress(messages, current_tokens=999999, force=True)
    assert result == before == messages
    assert compressor._last_compress_aborted is True
    assert compressor._last_summary_error == str(primary_error)
    providers = [provider for provider, _ in calls]
    assert providers[:3] == ["deepinfra.man", "openai-codex", "localrouter"]
    # The compressor's legacy main-model retry may repeat the primary call;
    # each exhausted configured fallback remains quarantined.
    assert providers.count("openai-codex") == providers.count("localrouter") == 1
    aux._reset_aux_unhealthy_cache()
