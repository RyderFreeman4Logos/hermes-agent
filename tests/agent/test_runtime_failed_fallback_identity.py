"""Failed runtime identities must not be retried through a later alias."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
            side_effect=[(_client(_SHARED_URL), "model-b"), (_client(_PRIMARY_URL), "model-a")],
        ) as resolve,
    ):
        assert agent._try_activate_fallback(FailoverReason.server_error) is True
        assert agent.provider == "route-b"
        assert agent._try_activate_fallback(FailoverReason.server_error) is False

    assert [call.args[0] for call in resolve.call_args_list] == ["route-b"]

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
    discarded = _client(_PRIMARY_URL)
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
