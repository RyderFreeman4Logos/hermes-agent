from agent.checkpoint_engine import CheckpointContextEngine


def test_planner_keeps_tool_call_and_result_together():
    messages = [
        {"role": "user", "content": "do it", "_row_id": 1},
        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function"}], "content": None, "_row_id": 2},
        {"role": "tool", "tool_call_id": "call-1", "content": "done", "_row_id": 3},
    ]
    groups = CheckpointContextEngine()._plan_causal_groups(messages)
    assert any(group.event_indices == (1, 2) for group in groups)


def test_live_projection_keeps_latest_user_verbatim_and_not_assistant_claim():
    messages = [
        {"role": "system", "content": "rules", "_row_id": 1},
        {"role": "assistant", "content": "I wrote the file", "_row_id": 2},
        {"role": "user", "content": "exact correction: use UTF-8", "_row_id": 3},
    ]
    result = CheckpointContextEngine({"mode": "live"}, session_id="s").compress(messages)
    assert {m.get("content") for m in result if m.get("role") == "user"} >= {messages[-1]["content"]}
    assert "succeeded" not in " ".join(str(m.get("content")) for m in result)


def test_inflight_tool_is_a_noop_without_success_count():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
    ]
    engine = CheckpointContextEngine({"mode": "live"})
    assert engine.compress(messages) is messages
    assert engine.compression_count == 0
