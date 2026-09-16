"""Regression coverage for issue #327 runtime fallback identity retention."""

from unittest.mock import MagicMock, patch

import pytest


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


def _client(base_url):
    client = MagicMock()
    client.base_url = base_url
    client.api_key = "fallback-key"
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
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_client(_SHARED_URL), "model-b")) as resolve,
    ):
        assert agent._try_activate_fallback(FailoverReason.server_error) is True
        assert agent.provider == "route-b"
        assert agent._try_activate_fallback(FailoverReason.server_error) is False

    assert [call.args[0] for call in resolve.call_args_list] == ["route-b"]

    with patch("agent.agent_runtime_helpers.time.monotonic", return_value=0):
        agent._restore_primary_runtime()
    assert agent._runtime_failed_backend_identities == set()


@pytest.mark.parametrize(
    ("reason", "current_provider", "candidate_provider", "current_url", "candidate_url"),
    [
        ("server_error", "route", "route", _PRIMARY_URL, _SHARED_URL),
        ("auth", "credential-a", "credential-b", _SHARED_URL, _SHARED_URL),
    ],
)
def test_failure_scope_keeps_distinct_route_or_credential_eligible(
    reason, current_provider, candidate_provider, current_url, candidate_url,
):
    from agent.error_classifier import FailoverReason

    reason = FailoverReason(reason)
    agent = _make_agent(
        [{"provider": candidate_provider, "model": "same-model", "base_url": candidate_url}]
    )
    agent.provider = current_provider
    agent.model = "same-model"
    agent.base_url = current_url

    with (
        patch("agent.chat_completion_helpers._fallback_entry_unavailable_without_network", return_value=None),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_client(candidate_url), "same-model"),
        ) as resolve,
    ):
        assert agent._try_activate_fallback(reason) is True

    assert resolve.call_count == 1
    assert agent.provider == candidate_provider
