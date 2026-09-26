import pytest
from agent.checkpoint_engine.replay import ReplayAdapter, ReplayUnavailable


def test_replay_requires_real_tool_adapter_and_never_uses_old_assistant_text():
    messages = [{"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "write"}}]}]
    with pytest.raises(ReplayUnavailable):
        ReplayAdapter({}).replay(messages)


def test_replay_uses_registered_tool_runner():
    adapter = ReplayAdapter({"write": lambda call: {"status": "succeeded", "content": "receipt"}})
    result = adapter.replay([{"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "write"}}]}])
    assert result == [{"role": "tool", "tool_call_id": "c", "status": "succeeded", "content": "receipt"}]
