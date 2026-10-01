"""Failed runtime identities must not be retried through a later alias."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason as R

_PRIMARY_URL = "http://127.0.0.1:8101/v1"
_SHARED_URL = "http://127.0.0.1:8102/v1"


def _make_agent(fallback_model):
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url=_PRIMARY_URL,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
    agent.client = MagicMock()
    return agent


def _client(base_url, api_key="fallback-key"):
    client = MagicMock()
    client.base_url = base_url
    client.api_key = api_key
    return client


def test_failed_backend_is_not_retried_through_a_later_alias():
    from agent.error_classifier import FailoverReason

    agent = _make_agent(
        [
            {"provider": "route-b", "model": "model-b", "base_url": _SHARED_URL},
            {"provider": "route-a-alias", "model": "model-a", "base_url": _PRIMARY_URL},
        ]
    )
    agent.provider = "route-a"
    agent.model = "model-a"
    agent.base_url = _PRIMARY_URL

    with (
        patch("agent.chat_completion_helpers._fallback_entry_unavailable_without_network", return_value=None),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[(_client(_SHARED_URL), "model-b"), (_client(_PRIMARY_URL, "test-key"), "model-a")],
        ) as resolve,
    ):
        assert agent._try_activate_fallback(FailoverReason.server_error) is True
        assert agent.provider == "route-b"
        assert agent._try_activate_fallback(FailoverReason.server_error) is False

    assert [call.args[0] for call in resolve.call_args_list] == ["route-b", "route-a-alias"]

    with patch("agent.agent_runtime_helpers.time.monotonic", return_value=0):
        agent._restore_primary_runtime()
    assert agent._runtime_failed_backend_identities == set()


def test_billing_keeps_distinct_provider_eligible_with_shared_resolved_key():
    """A billing failure must not skip a different provider that reuses the key string."""
    from agent.error_classifier import FailoverReason

    agent = _make_agent([{"provider": "openai", "model": "gpt-4o"}])
    agent.provider = "xai-oauth"
    agent.model = "grok-4.6"
    agent.base_url = "https://fallback.invalid/v1"
    agent.api_key = "fallback-key"

    with (
        patch("agent.chat_completion_helpers._fallback_entry_unavailable_without_network", return_value=None),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_client("https://fallback.invalid/v1", "fallback-key"), "gpt-4o"),
        ) as resolve,
    ):
        assert agent._try_activate_fallback(FailoverReason.billing) is True

    assert resolve.call_count == 1
    assert agent.provider == "openai"


def test_same_label_different_key_stays_eligible_after_auth_failure():
    from agent.error_classifier import FailoverReason

    agent = _make_agent(
        [{"provider": "openrouter", "model": "model-b", "base_url": "https://openrouter.ai/api/v1", "api_key": "fixture-key-b"}]
    )
    agent.provider = "openrouter"
    agent.model = "model-a"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "fixture-key-a"

    with (
        patch("agent.chat_completion_helpers._fallback_entry_unavailable_without_network", return_value=None),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_client("https://openrouter.ai/api/v1", "fixture-key-b"), "model-b"),
        ) as resolve,
    ):
        assert agent._try_activate_fallback(FailoverReason.auth) is True

    assert resolve.call_count == 1
    assert resolve.call_args.kwargs["explicit_api_key"] == "fixture-key-b"


def test_post_resolution_duplicate_closes_only_discarded_client_once():
    from agent.error_classifier import FailoverReason

    agent = _make_agent([{"provider": "zai", "model": "zai/glm-4.7"}])
    active_client = agent.client
    shared_client = MagicMock()
    discarded = _client(_PRIMARY_URL, "test-key")
    agent.provider = "zai"
    agent.model = "glm-4.7"
    agent.base_url = _PRIMARY_URL
    agent._anthropic_client = shared_client

    with (
        patch("agent.chat_completion_helpers._fallback_entry_unavailable_without_network", return_value=None),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(discarded, "glm-4.7"),
        ),
    ):
        assert agent._try_activate_fallback(FailoverReason.server_error) is False

    discarded.close.assert_called_once_with()
    active_client.close.assert_not_called()
    shared_client.close.assert_not_called()


def test_failed_identity_warning_never_logs_private_url(caplog):
    import logging

    from agent import chat_completion_helpers as helpers
    from agent.backend_identity import BackendIdentity, FailureScope

    url = "http://127.0.0.1:65534/private-route"
    agent = SimpleNamespace(
        provider="other",
        model="same-model",
        base_url="http://127.0.0.1:65533/v1",
        api_key=None,
        _runtime_failed_backend_identities={(
            BackendIdentity.build(provider="", model="same-model", base_url=url),
            FailureScope.MODEL,
        )},
    )
    with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
        skipped = helpers._should_skip_fallback_candidate(
            agent,
            {"provider": "alias", "model": "same-model", "base_url": url},
            ("alias", "same-model", url),
            "alias",
            "same-model",
            set(),
        )
    assert skipped is True
    assert url not in caplog.text


@pytest.mark.parametrize("reason,provider,key_source,expected", [
    (R.server_error, "route-a", "explicit-new", True),
    (R.rate_limit, "route-a", "explicit-new", True),
    (R.rate_limit, "custom:route-a", "env-new", True),
    (R.rate_limit, "route-c", "configured-new", True),
    (R.rate_limit, "route-a", "configured-new", True),
    (R.server_error, "custom:route-a", "same", False),
    (R.auth, "custom:route-a", "same", False),
    (R.billing, "Route A", "configured-same", False),
    (R.auth, "route-a", "explicit-new", True),
    (R.auth, "route-c", "configured-same", True),
    (R.server_error, "route-c", "configured-same", True),
    (R.ssl_cert_verification, "route-a", "explicit-new", False),
])
def test_named_fallback_history_uses_owned_credentials(
    tmp_path, monkeypatch, reason, provider, key_source, expected,
):
    """Real named resolver and SDK: A fails, B fails, then an owned A candidate."""
    from agent import auxiliary_client as aux

    url_a, url_b = "https://a.invalid/v1", "https://b.invalid/v1"
    key_a, key_b = "fixture-key-a", "fixture-key-b"
    config = {"providers": {
        "route-a": {"name": "Route A", "base_url": url_a, "api_key": key_a},
        "route-b": {"base_url": url_b, "api_key": key_a},
        "route-c": {"base_url": url_a, "api_key": key_b if key_source == "configured-new" else key_a},
    }}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    if provider == "route-a" and key_source == "configured-new":
        config["providers"]["route-a"]["api_key"] = key_b
        (tmp_path / "config.yaml").write_text(json.dumps(config))
    final = {"provider": provider, "model": "model-a", "base_url": url_a}
    if key_source in {"explicit-new", "same"}:
        final["api_key"] = key_b if key_source == "explicit-new" else key_a
    elif key_source == "env-new":
        monkeypatch.setenv("FALLBACK_FIXTURE_KEY", key_b)
        final["key_env"] = "FALLBACK_FIXTURE_KEY"
    sends, clients = [], []
    status = 503

    def create_client(**kwargs):
        def reply(request):
            sends.append((request.url.host, request.headers["authorization"]))
            return httpx.Response(status, json={"error": {"message": "fixture outage"}})

        allowed = {k: v for k, v in kwargs.items()
                   if k in {"api_key", "base_url", "default_headers", "default_query"}}
        client = openai.OpenAI(**allowed, max_retries=0,
                              http_client=httpx.Client(transport=httpx.MockTransport(reply)))
        clients.append(client)
        return client

    def dispatch_failure(agent, code):
        nonlocal status
        status = code
        before = len(sends)
        with pytest.raises(openai.APIStatusError) as error:
            agent.client.chat.completions.create(
                model=agent.model, messages=[{"role": "user", "content": "fixture"}])
        assert error.value.status_code == code
        assert len(sends) == before + 1

    monkeypatch.setattr(aux, "_create_openai_client", create_client)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *args, **kwargs: 128000)
    agent = _make_agent([{"provider": "route-b", "model": "model-b"}, final])
    agent.provider, agent.model = "route-a", "model-a"
    agent.base_url, agent.api_key = url_a, key_a
    agent.client = create_client(api_key=key_a, base_url=url_a)
    try:
        dispatch_failure(agent, {R.auth: 401, R.billing: 402, R.rate_limit: 429}.get(reason, 503))
        assert agent._try_activate_fallback(reason)
        assert agent.provider == "route-b"
        dispatch_failure(agent, 503)
        assert agent._try_activate_fallback(R.server_error) is expected
        if expected:
            owned_key = key_b if key_source.endswith("new") else key_a
            assert agent.api_key == owned_key
            dispatch_failure(agent, 503)
            assert sends[-1] == ("a.invalid", "Bearer " + owned_key)
        assert all(key_a not in repr(identity) and key_b not in repr(identity)
                   for identity, _ in agent._runtime_failed_backend_identities)
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("control", ["construction", "unconfigured", "fingerprints"])
def test_fallback_identity_boundary_controls(tmp_path, monkeypatch, control):
    """Constructor failure is not a send; missing auth cannot inherit runtime auth."""
    from agent import auxiliary_client as aux
    from agent.backend_identity import BackendIdentity, same_credential_surface

    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *args, **kwargs: 128000)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({"providers": {
        "route-b": {"base_url": "https://b.invalid/v1", "api_key": "fixture-key-a"},
    }}))
    if control == "fingerprints":
        a = BackendIdentity.build(provider="p", api_key="fixture-key-a")
        b = BackendIdentity.build(provider="p", api_key="fixture-key-b")
        absent = BackendIdentity.build(provider="p")
        assert a.credential_fingerprint and not absent.credential_fingerprint
        assert not same_credential_surface(a, b)
        assert not same_credential_surface(a, absent)
        assert same_credential_surface(absent, absent)
        assert "fixture-key-a" not in repr(a)
        return
    agent = _make_agent([
        {"provider": "route-b", "model": "model-b"},
        {"provider": "custom:route-b", "model": "model-b"},
    ] if control == "construction" else [
        {"provider": "unknown-unconfigured", "model": "model-c", "base_url": "https://unapproved.invalid/v1"},
    ])
    client = openai.OpenAI(api_key="fixture-key-a", base_url="https://b.invalid/v1",
                          http_client=httpx.Client(transport=httpx.MockTransport(
                              lambda _: pytest.fail("constructor-only control must not dispatch"))))
    try:
        with patch.object(aux, "_create_openai_client", side_effect=[ValueError("fixture pre-dispatch setup"), client]) as create:
            assert agent._try_activate_fallback(R.server_error) is (control == "construction")
            if control == "construction":
                assert agent.provider == "custom:route-b"
                assert create.call_count == 2
            else:
                create.assert_not_called()
                assert agent._runtime_failed_backend_identities
                agent._restore_primary_runtime()
                assert agent._runtime_failed_backend_identities == set()
    finally:
        client.close()
