"""Credential refusal must survive compression's pinned summary path (#359)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import auxiliary_client as aux
from agent.context_compressor import ContextCompressor, pin_summary_route
from agent.conversation_compression import resolve_compression_fallback_route
from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.config import atomic_config_write
from hermes_constants import get_hermes_home


@pytest.mark.parametrize("field", ["key_env", "api_key_env"])
@pytest.mark.parametrize(
    "case,reference",
    [("list", []), ("dict", {}), ("false", False), ("null", None),
     ("blank", ""), ("nested", ["credential-canary"]), ("missing", "MISSING"),
     ("valid", "KEY"), ("absent", None), ("inline", [])],
)
def test_pinned_summary_sends_only_the_accepted_hop(field, case, reference, monkeypatch, caplog):
    first = {"provider": "first-owner", "model": "first-model"}
    if case != "absent":
        first[field] = reference
    if case == "inline":
        first["api_key"] = "fixture-inline"
    second = {"provider": "second-owner", "model": "second-model"}
    chain = [first, second]
    atomic_config_write(get_hermes_home() / "config.yaml", {
        "providers": {
            "first-owner": {"base_url": "https://first.invalid/v1", "api_key": "fixture-first-default"},
            "second-owner": {"base_url": "https://second.invalid/v1", "api_key": "fixture-second-default"},
        },
        "auxiliary": {"compression": {"fallback_chain": chain}},
    })
    monkeypatch.setattr(aux, "_create_openai_client", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(aux, "get_model_context_length", lambda *a, **kw: 100000)
    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *a, **kw: 100000)
    sends = []

    def send(client, request, **kwargs):
        sends.append((request["model"], str(client.base_url).rstrip("/"), client.api_key))
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Synthetic summary body"), finish_reason="stop",
        )])

    monkeypatch.setattr(aux, "_relay_sync_completion", send)
    caplog.set_level("DEBUG", logger="agent.conversation_compression")
    token = set_secret_scope({"KEY": "fixture-scoped"}, profile_home=str(get_hermes_home()))
    try:
        route = resolve_compression_fallback_route()
        assert route is not None
        compressor = ContextCompressor(model="main-model", quiet_mode=True)
        with pin_summary_route(route):
            summary = compressor._generate_summary([
                {"role": "user", "content": "Summarize this synthetic conversation."},
                {"role": "assistant", "content": "A synthetic reply."},
            ])
    finally:
        reset_secret_scope(token)

    accepted_first = case in {"valid", "absent", "inline"}
    owner = "first" if accepted_first else "second"
    key = {"valid": "fixture-scoped", "inline": "fixture-inline"}.get(case, f"fixture-{owner}-default")
    assert summary and "Synthetic summary body" in summary
    assert route["provider"] == f"{owner}-owner"
    assert sends == [(f"{owner}-model", f"https://{owner}.invalid/v1", key)]
    assert "credential-canary" not in caplog.text


@pytest.mark.parametrize("field", ["key_env", "api_key_env"])
@pytest.mark.parametrize("reference", [[], {}, False, None, "", ["credential-canary"], "MISSING", 0, {"credential-canary": 1}])
def test_refused_only_hop_never_starts_a_retry(field, reference, monkeypatch, caplog):
    from agent.conversation_compression import _retry_compression_on_fallback_chain

    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {
        "fallback_chain": [{"provider": "first-owner", "model": "first-model", field: reference}],
    })
    worker = Mock(side_effect=AssertionError("Rejected route must not start a worker"))
    client = Mock(side_effect=AssertionError("Rejected route must not resolve provider defaults"))
    monkeypatch.setattr(aux, "resolve_provider_client", client)
    caplog.set_level("DEBUG", logger="agent.conversation_compression")
    token = set_secret_scope({}, profile_home=str(get_hermes_home()))
    try:
        assert resolve_compression_fallback_route() is None
        assert _retry_compression_on_fallback_chain(
            worker=worker, messages=[], system_prompt_fallback="unchanged",
            idle_timeout_seconds=2, total_ceiling_seconds=5,
        ) is None
    finally:
        reset_secret_scope(token)
    worker.assert_not_called()
    client.assert_not_called()
    assert "credential-canary" not in caplog.text
