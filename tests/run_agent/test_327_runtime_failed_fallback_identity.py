"""Regression coverage for issue #327 runtime fallback identity retention."""

import logging
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


_PRIMARY_URL = "http://127.0.0.1:8101/v1"
_SHARED_URL = "http://127.0.0.1:8102/v1"


class _ServerError(Exception):
    def __init__(self):
        super().__init__("fixture backend unavailable")
        self.status_code = 503
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "fixture backend unavailable"}}


def _response(text):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                    reasoning_details=None,
                    model_extra={},
                ),
                finish_reason="stop",
            )
        ],
        model="model-a",
        usage=None,
    )


def test_run_conversation_skips_omitted_url_alias_after_real_resolution(tmp_path, monkeypatch):
    """An omitted-url alias must not reissue a failed deployment after resolution."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(
        """model:
  default: model-a
custom_providers:
  - name: route-b
    base_url: http://127.0.0.1:8102/v1
  - name: route-a-alias
    base_url: http://127.0.0.1:8101/v1
""",
        encoding="utf-8",
    )
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="no-key-required",
        base_url=_PRIMARY_URL,
        provider="route-a",
        model="model-a",
        fallback_model=[
            {"provider": "route-b", "model": "model-b"},
            {"provider": "route-a-alias", "model": "model-a"},
        ],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._delegate_model_profile = "standard"
    agent._disable_streaming = True
    agent._delegate_successful_llm_route = None
    agent._api_max_retries = 3
    calls = []

    def dispatch(_kwargs):
        calls.append(agent.provider)
        if len(calls) < 3:
            raise _ServerError()
        return _response("alias was incorrectly retried")

    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=dispatch))
        stack.enter_context(patch.object(agent, "_persist_session"))
        stack.enter_context(patch.object(agent, "_save_trajectory"))
        stack.enter_context(patch.object(agent, "_cleanup_task_resources"))
        stack.enter_context(patch.object(agent, "_try_recover_primary_transport", return_value=False))
        stack.enter_context(patch("agent.turn_recovery.time.sleep"))
        stack.enter_context(patch("agent.retry_utils.jittered_backoff", return_value=0))
        stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=200_000))
        result = agent.run_conversation("fixture request")

    assert calls[:2] == ["route-a", "route-b"]
    assert "route-a-alias" not in calls
    assert result["completed"] is True


def test_failed_identity_warning_never_logs_empty_provider_url(caplog):
    """An unnamed failed route may be identified without logging its private URL."""
    from agent import chat_completion_helpers as helpers
    from agent.backend_identity import BackendIdentity, FailureScope

    url_canary = "http://127.0.0.1:65534/private-route"
    agent = SimpleNamespace(
        provider="other",
        model="same-model",
        base_url="http://127.0.0.1:65533/v1",
        _runtime_failed_backend_identities={(
            BackendIdentity.build(provider="", model="same-model", base_url=url_canary),
            FailureScope.MODEL,
        )},
        _unavailable_fallback_keys=set(),
    )

    with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
        skipped = helpers._should_skip_fallback_candidate(
            agent,
            {"provider": "alias", "model": "same-model", "base_url": url_canary},
            ("alias", "same-model", url_canary),
            "alias",
            "same-model",
            set(),
        )

    assert skipped is True
    assert url_canary not in caplog.text


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
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[(_client(_SHARED_URL), "model-b"), (_client(_PRIMARY_URL), "model-a")],
        ) as resolve,
    ):
        assert agent._try_activate_fallback(FailoverReason.server_error) is True
        assert agent.provider == "route-b"
        assert agent._try_activate_fallback(FailoverReason.server_error) is False

    # The explicit same endpoint is sufficient to reject the alias before it selects credentials.
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
