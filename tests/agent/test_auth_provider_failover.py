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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai import APIStatusError

from run_agent import AIAgent
from agent.error_classifier import classify_api_error, FailoverReason

def _make_agent(fallback_model=None, base_url="https://openrouter.ai/api/v1", **runtime):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url=base_url,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
            **runtime,
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

def _conversation(agent):
    agent._cached_system_prompt = "You are helpful."
    agent.compression_enabled = False
    agent.save_trajectories = False
    # model.streaming=false: isolate recovery from the separate streaming-5xx probe.
    agent._disable_streaming = True
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.turn_api_error.interruptible_backoff_sleep", side_effect=AssertionError("same-route retry")),
    ):
        return agent.run_conversation("hello")

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
        mixed = classify_api_error(_sdk_status(
            503, "maximum context length exceeded; auth_unavailable: no auth available",
        ), provider="xai")
        assert mixed.reason == FailoverReason.context_overflow and mixed.should_compress
        assert mixed.is_auth is False and mixed.should_rotate_credential is False
        spend = _sdk_status(403, "spending limit", {"error": {
            "code": "personal-team-blocked:spending-limit", "message": "spending limit",
        }})
        billing = classify_api_error(spend, provider="xai-oauth")
        assert billing.reason == FailoverReason.billing and billing.is_auth is False
        policy = classify_api_error(_sdk_status(503, "content_filter: no auth available"), provider="xai")
        assert policy.reason == FailoverReason.content_policy_blocked and policy.is_auth is False

        for marker in ("auth_unavailable", "no auth available"):
            for available in (True, False):
                agent = _make_agent(
                    provider="xai", model="grok-4.6", base_url="https://api.x.ai/v1", api_mode="chat_completions",
                    fallback_model=[{"provider": "openai", "model": "gpt-4o", "api_mode": "chat_completions"}],
                )
                primary = agent.client
                fallback = _mock_client("https://api.openai.com/v1")
                fallback._custom_headers = fallback.default_headers = None
                calls = []

                def send(**kwargs):
                    calls.append((agent.provider, kwargs["model"], agent.api_mode))
                    if agent.client is primary:
                        raise _sdk_status(503, marker)
                    assert agent.client is fallback
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="accepted", tool_calls=None), finish_reason="stop")],
                        model="gpt-4o", usage=None,
                    )

                primary.chat.completions.create.side_effect = send
                fallback.chat.completions.create.side_effect = send
                with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback, "gpt-4o") if available else (None, None)) as resolve:
                    result = _conversation(agent)
                assert resolve.call_count == 1
                assert resolve.call_args.kwargs["explicit_api_key"] is None
                assert primary.chat.completions.create.call_count == 1
                assert calls[0] == ("xai", "grok-4.6", "chat_completions")
                if available:
                    assert result["completed"] is True and result["final_response"] == "accepted"
                    assert calls == [("xai", "grok-4.6", "chat_completions"), ("openai", "gpt-4o", "chat_completions")]
                    assert agent.client is fallback and agent.api_key == "fb-key"
                    assert fallback.chat.completions.create.call_count == 1
                else:
                    assert result["failed"] is True and result["completed"] is False
                    assert len(calls) == 1 and fallback.chat.completions.create.call_count == 0
                    assert (agent.provider, agent.model, agent.api_key) == ("xai", "grok-4.6", "test-key")

    def test_no_failover_without_chain(self):
        """A user with no fallback configured (the common case for the
        original incident) does NOT failover — falls through to the
        terminal auth handling."""
        agent = _make_agent()
        classified = classify_api_error(_auth_error(401))
        assert agent._try_activate_fallback(reason=classified.reason) is False
