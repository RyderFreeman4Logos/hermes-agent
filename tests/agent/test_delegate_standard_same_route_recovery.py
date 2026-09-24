"""A standard child with a pending fallback repairs the current route first.

Provider failures may still switch later. A failed switch that leaves the runtime
untouched must re-enter same-route recovery instead of abandoning it.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.error_classifier import FailoverReason
from agent.turn_recovery import route_classified_error
from agent.turn_retry_state import TurnRetryState


def _child():
    agent = SimpleNamespace(
        model="std-model",
        provider="deepseek",
        base_url="http://std/v1",
        api_mode="chat_completions",
        requested_provider="deepseek",
        client=object(),
        _credential_pool=None,
        _credential_pool_entry_id=None,
        _delegate_model_profile="standard",
        _fallback_chain=[{"provider": "groq", "model": "backup"}],
        _fallback_index=0,
        log_prefix="",
        compression_enabled=True,
    )
    agent._buffer_diagnostic_status = lambda *_a, **_k: None
    agent._has_pending_fallback = lambda: True
    agent._try_activate_fallback = lambda **_k: False
    return agent


class TestStandardChildRecoveryOrder(unittest.TestCase):
    def test_intact_failed_switch_repairs_current_route_before_abandoning(self):
        agent = _child()
        seen = []

        def recover(*_a, **_k):
            seen.append("same-route")
            return True, False

        with patch("agent.turn_recovery.recover_after_classification", side_effect=recover):
            verdict = route_classified_error(
                agent, RuntimeError("upstream 503"),
                SimpleNamespace(reason=FailoverReason.server_error, is_auth=False),
                TurnRetryState(), error_msg="upstream 503", error_context={},
                recovered_with_pool=False, base_url=agent.base_url, model=agent.model,
                messages=[], api_messages=[], system_message="", active_system_prompt="",
                conversation_history=[], retry_count=2, max_retries=3,
                compression_attempts=0, max_compression_attempts=2, api_call_count=1,
                effective_task_id="t",
            )
        self.assertEqual(seen, ["same-route"])
        self.assertEqual(verdict.action, "continue")


if __name__ == "__main__":
    unittest.main()
