"""A standard child repairs the accepted route before abandoning it.

Provider failures may still switch later. A failed switch that leaves the
runtime untouched must re-enter same-route recovery first. Content-policy
denials stay terminal, and a real identity change does not recover first.
"""

import unittest

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from agent.error_classifier import FailoverReason
from agent.turn_recovery import route_classified_error
from agent.turn_retry_state import TurnRetryState


def _child(profile="standard"):
    agent = SimpleNamespace(
        model="std-model",
        provider="deepseek",
        base_url="http://std/v1",
        api_mode="chat_completions",
        requested_provider="deepseek",
        client=object(),
        _credential_pool=object(),
        _credential_pool_entry_id="accepted",
        _delegate_model_profile=profile,
        _fallback_chain=[{"provider": "groq", "model": "backup"}],
        _fallback_index=0,
        log_prefix="",
        compression_enabled=True,
    )
    agent._buffer_diagnostic_status = lambda *_a, **_k: None
    agent._has_pending_fallback = lambda: True
    agent._try_activate_fallback = lambda **_k: False
    return agent


def _route(agent, reason, *, is_auth=False, retry_count=2):
    return route_classified_error(
        agent, RuntimeError("provider failure"),
        SimpleNamespace(reason=reason, is_auth=is_auth, billing_unverified=False, error_context={}),
        TurnRetryState(), error_msg="provider failure", error_context={},
        recovered_with_pool=False, base_url=agent.base_url, model=agent.model,
        messages=[], api_messages=[], system_message="", active_system_prompt="",
        conversation_history=[], retry_count=retry_count, max_retries=3,
        compression_attempts=0, max_compression_attempts=2, api_call_count=1,
        effective_task_id="t",
    )


class TestStandardChildRecoveryOrder(unittest.TestCase):
    def test_intact_failed_switch_repairs_current_route_before_abandoning(self):
        agent = _child()
        seen = []

        def recover(*_a, **_k):
            seen.append("same-route")
            return True, False

        with patch("agent.turn_recovery.recover_after_classification", side_effect=recover):
            verdict = _route(agent, FailoverReason.server_error)
        self.assertEqual(seen, ["same-route"])
        self.assertEqual(verdict.action, "continue")
        self.assertEqual(agent.model, "std-model")
        self.assertEqual(agent.provider, "deepseek")

    def test_changed_identity_does_not_recover_the_old_route(self):
        agent = _child()
        seen = []

        def switched(**_k):
            agent.model, agent.provider = "backup", "groq"
            return True

        def recover(*_a, **_k):
            seen.append("same-route")
            return True, False

        agent._try_activate_fallback = switched
        with patch("agent.turn_recovery.recover_after_classification", side_effect=recover), patch(
            "agent.conversation_loop._arm_fallback_restart", return_value="armed",
        ):
            verdict = _route(agent, FailoverReason.server_error)
        self.assertEqual(seen, [])
        self.assertEqual(verdict.action, "break")
        self.assertEqual((agent.model, agent.provider), ("backup", "groq"))

    def test_content_policy_does_not_fallback_or_recover(self):
        agent = _child()
        calls = []
        agent._try_activate_fallback = lambda **_k: calls.append("fallback") or True

        def recover(*_a, **_k):
            calls.append("same-route")
            return True, False

        with patch("agent.turn_recovery.recover_after_classification", side_effect=recover):
            verdict = _route(agent, FailoverReason.content_policy_blocked)
        self.assertEqual(calls, [])
        self.assertEqual(verdict.action, "fallthrough")

    def test_non_standard_child_keeps_pool_deferral(self):
        agent = _child(profile="review")
        calls = []
        agent._try_activate_fallback = lambda **_k: calls.append("fallback") or False

        def recover(*_a, **_k):
            calls.append("same-route")
            return True, False

        with patch("agent.turn_recovery.recover_after_classification", side_effect=recover), patch(
            "run_agent._pool_may_recover_from_rate_limit", return_value=True,
        ) as pool:
            verdict = _route(agent, FailoverReason.rate_limit, retry_count=0)
        self.assertTrue(pool.called)
        self.assertEqual(calls, [])
        self.assertEqual(verdict.action, "fallthrough")

    def test_exhausted_same_route_recovery_still_falls_through(self):
        agent = _child()
        seen = []

        def recover(*_a, **_k):
            seen.append("same-route")
            return False, False

        with patch("agent.turn_recovery.recover_after_classification", side_effect=recover):
            verdict = _route(agent, FailoverReason.server_error)
        self.assertEqual(seen, ["same-route"])
        self.assertEqual(verdict.action, "fallthrough")
        self.assertEqual(agent.model, "std-model")


@pytest.mark.parametrize("reason,expected", [
    (FailoverReason.server_error, "Provider error"),
    (FailoverReason.rate_limit, "Rate limited"),
])
def test_eager_recovery_status_names_failure(reason, expected, capsys):
    agent = _child()
    agent._buffer_diagnostic_status = print
    with patch("agent.turn_recovery.recover_after_classification", return_value=(False, False)):
        _route(agent, reason)
    output = capsys.readouterr().out
    assert expected in output
    if reason == FailoverReason.server_error:
        assert "Rate limited" not in output


if __name__ == "__main__":
    unittest.main()
