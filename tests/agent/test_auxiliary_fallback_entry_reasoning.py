"""#130 — per-entry reasoning_effort on auxiliary fallback_chain."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.auxiliary_client import _call_fallback_candidate_sync


def test_fallback_entries_apply_own_reasoning_effort():
    """Entry A high and entry B low are applied; omitted inherits task default."""
    chain = [
        {"provider": "custom", "model": "a", "reasoning_effort": "high"},
        {"provider": "custom", "model": "b", "reasoning_effort": "low"},
        {"provider": "custom", "model": "c"},
    ]
    task_body = {"reasoning": {"enabled": True, "effort": "medium"}}

    def _effort_for(label):
        seen = {}

        class _FakeCompletions:
            def create(self, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

        client = SimpleNamespace(
            base_url="https://example.invalid/v1",
            chat=SimpleNamespace(completions=_FakeCompletions()),
        )
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"fallback_chain": chain, "reasoning_effort": "medium"},
        ):
            resp = _call_fallback_candidate_sync(
                client,
                "model",
                label,
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
                temperature=None,
                max_tokens=None,
                tools=None,
                effective_timeout=30.0,
                effective_extra_body=task_body,
                reasoning_config=None,
            )
        assert resp is not None
        return (seen.get("extra_body") or {}).get("reasoning")

    assert _effort_for("fallback_chain[0](custom)") == {
        "enabled": True,
        "effort": "high",
    }
    assert _effort_for("fallback_chain[1](custom)") == {
        "enabled": True,
        "effort": "low",
    }
    assert _effort_for("fallback_chain[2](custom)") == {
        "enabled": True,
        "effort": "medium",
    }
    assert task_body == {"reasoning": {"enabled": True, "effort": "medium"}}
