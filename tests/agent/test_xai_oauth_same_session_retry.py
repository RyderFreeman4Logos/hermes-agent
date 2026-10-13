"""An unverified xAI OAuth spending-limit 403 retries the same session once.

A second hit still rotates. The session is not exhausted on the first one.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import FAILURE_REASON_BILLING_UNVERIFIED
from agent.error_classifier import FailoverReason


def _agent():
    agent = SimpleNamespace(
        provider="xai-oauth",
        model="grok-4.6",
        base_url="https://api.x.ai/v1",
        api_key="test-key",
        _credential_pool_entry_id="sess",
        _credential_pool=None,
        _swap_credential=MagicMock(),
    )
    return agent


def _classified():
    return SimpleNamespace(
        reason=FailoverReason.billing,
        billing_unverified=True,
        error_context={"reason": "personal-team-blocked:spending-limit"},
    )


def test_first_unverified_xai_403_retries_same_session_without_exhausting():
    agent = _agent()
    pool = MagicMock()
    pool.provider = "xai-oauth"
    agent._credential_pool = pool
    classified = _classified()

    recovered, retried = recover_with_credential_pool(
        agent,
        status_code=403,
        has_retried_429=False,
        classified_reason=classified.reason,
        error_context=classified.error_context,
        billing_unverified=classified.billing_unverified,
    )

    assert recovered is False
    assert retried is True
    pool.mark_exhausted_and_rotate.assert_not_called()
    agent._swap_credential.assert_not_called()


def test_second_unverified_xai_403_rotates_with_short_cooldown_reason():
    agent = _agent()
    pool = MagicMock()
    pool.provider = "xai-oauth"
    pool.mark_exhausted_and_rotate.return_value = SimpleNamespace(
        id="other", base_url="https://api.x.ai/v1", priority=1,
    )
    agent._credential_pool = pool
    classified = _classified()

    recovered, retried = recover_with_credential_pool(
        agent,
        status_code=403,
        has_retried_429=True,
        classified_reason=classified.reason,
        error_context=classified.error_context,
        billing_unverified=classified.billing_unverified,
    )

    assert recovered is True
    assert retried is False
    kwargs = pool.mark_exhausted_and_rotate.call_args.kwargs
    assert kwargs["failure_reason"] == FAILURE_REASON_BILLING_UNVERIFIED
    agent._swap_credential.assert_called_once()
