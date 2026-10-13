from agent.checkpoint_engine import CheckpointContextEngine, final_request_exceeds_hard_wire_budget


def test_hard_budget_counts_final_wire_not_only_checkpoint_body():
    messages = [{"role": "user", "content": "x" * 400}]
    engine = CheckpointContextEngine({"mode": "live", "hard_max_wire_tokens": 10, "target_wire_tokens": 5})
    assert final_request_exceeds_hard_wire_budget(engine, messages)
    assert engine.compress(messages) is messages


def test_projection_does_not_emit_orphan_tool_results():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "tool_calls": [{"id": "old"}], "content": None},
        {"role": "tool", "tool_call_id": "old", "content": "old"},
    ]
    candidate = CheckpointContextEngine({"mode": "live"}).compress(messages)
    for index, message in enumerate(candidate):
        if message.get("role") == "tool":
            assert any(call.get("id") == message.get("tool_call_id") for prior in candidate[:index] if prior.get("role") == "assistant" for call in prior.get("tool_calls", ()))
