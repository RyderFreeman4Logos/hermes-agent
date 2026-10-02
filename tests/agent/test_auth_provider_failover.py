"""Auth-failure provider failover (conversation loop).

A 401/403 that survives the per-provider credential-refresh attempt
(revoked OAuth, blocked/expired key, an account pinned to a dead/staging
endpoint) must escalate to the configured fallback chain instead of
thrashing on the same dead credential every turn.

Before the fix, the conversation loop's generic failover dispatch only
fired for ``{rate_limit, billing}`` reasons; ``auth`` / ``auth_permanent``
fell through to "switch providers manually" advice and never called
``_try_activate_fallback()``. These tests pin:

  1. 401/403 classify as auth (``classified.is_auth`` True).
  2. ``_try_activate_fallback`` advances the chain on an auth reason.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai import APIStatusError

from run_agent import AIAgent
from agent.error_classifier import classify_api_error, FailoverReason
from agent.turn_api_error import handle_api_error
from agent.turn_retry_state import TurnRetryState

def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent

def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock

def _auth_error(status=401, msg="Your API key is invalid, blocked or out of funds."):
    err = Exception(f"Error code: {status} - {msg}")
    err.status_code = status
    return err

def _sdk_status(status, message, body=None):
    response = MagicMock(status_code=status, headers={})
    response.json.return_value = body if body is not None else {"error": {"message": message}}
    err = APIStatusError(message, response=response, body=response.json.return_value)
    err.status_code = status
    return err

def _handler(agent, api_error):
    agent._recover_with_credential_pool = MagicMock(return_value=(False, False))
    retry = TurnRetryState()
    verdict = handle_api_error(
        agent, api_error=api_error, _retry=retry, thinking_spinner=None,
        messages=[{"role": "user", "content": "hello"}],
        api_messages=[{"role": "user", "content": "hello"}], api_kwargs={},
        system_message=None, active_system_prompt=None, conversation_history=[],
        approx_tokens=10, retry_count=0, max_retries=10, compression_attempts=0,
        max_compression_attempts=1, api_call_count=1, api_request_id="req",
        api_start_time=time.time(), effective_task_id=None, turn_id="turn",
    )
    return verdict, retry

class TestAuthErrorClassification:
    def test_401_is_auth(self):
        c = classify_api_error(_auth_error(401))
        assert c.reason in {FailoverReason.auth, FailoverReason.auth_permanent}
        assert c.is_auth is True

    def test_500_is_not_auth(self):
        err = Exception("Error code: 500 - internal server error")
        err.status_code = 500
        c = classify_api_error(err)
        assert c.is_auth is False

class TestAuthFailoverActivation:
    """The decision the loop makes on a persistent auth failure: when a
    fallback chain exists and the guard hasn't fired, escalate to it."""

    def test_auth_failover_fires_when_chain_present(self):
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        classified = classify_api_error(_auth_error(401))
        # The activation primitive advances the chain on an auth reason.
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "gpt-4o"),
        ):
            advanced = agent._try_activate_fallback(reason=classified.reason)
        assert advanced is True
        assert agent._fallback_index == 1

    def test_503_auth_unavailable_real_error_handler_switches_without_retries(self):
        """Empty auth pool: next eligible route immediately, zero same-route retries."""
        overload = classify_api_error(_sdk_status(503, "service overloaded"), provider="xai")
        assert overload.reason == FailoverReason.overloaded and overload.is_auth is False
        overflow = classify_api_error(_sdk_status(503, "maximum context length exceeded"), provider="xai")
        assert overflow.reason == FailoverReason.context_overflow and overflow.is_auth is False
        other = classify_api_error(_sdk_status(529, "auth_unavailable: no auth available"), provider="xai")
        assert other.reason == FailoverReason.overloaded and other.is_auth is False
        spend = _sdk_status(403, "spending limit", {"error": {
            "code": "personal-team-blocked:spending-limit", "message": "spending limit",
        }})
        billing = classify_api_error(spend, provider="xai-oauth")
        assert billing.reason == FailoverReason.billing and billing.is_auth is False

        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        agent.provider, agent.model = "xai", "grok-4.6"
        client = _mock_client("https://api.openai.com/v1")
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "gpt-4o")) as resolve:
            verdict, retry = _handler(agent, _sdk_status(
                503, "auth_unavailable: no auth available (providers=xai, model=grok-4.6)",
            ))
        assert verdict.action == "break" and retry.restart_with_rebuilt_messages is True
        assert verdict.retry_count == 0 and resolve.call_count == 1
        assert agent.provider == "openai" and agent.model == "gpt-4o" and agent.api_key == "fb-key"
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="accepted"))],
        )
        reply = client.chat.completions.create(model=agent.model, messages=verdict.messages)
        assert reply.choices[0].message.content == "accepted"

        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        agent.provider, agent.model = "xai", "grok-4.6"
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)) as resolve:
            verdict, retry = _handler(agent, _sdk_status(503, "no auth available"))
        assert resolve.call_count == 1 and agent.provider == "xai" and agent.model == "grok-4.6"
        assert agent.api_key == "test-key" and verdict.retry_count == 1
        assert verdict.action == "return" and retry.restart_with_rebuilt_messages is False

    def test_no_failover_without_chain(self):
        """A user with no fallback configured (the common case for the
        original incident) does NOT failover — falls through to the
        terminal auth handling."""
        agent = _make_agent()
        classified = classify_api_error(_auth_error(401))
        assert agent._try_activate_fallback(reason=classified.reason) is False
