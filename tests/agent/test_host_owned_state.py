from agent.checkpoint_engine import CheckpointContextEngine


def test_only_tool_receipt_can_mark_effect_succeeded():
    messages = [
        {"role": "user", "content": "write file"},
        {"role": "assistant", "content": "I wrote it"},
        {"role": "tool", "tool_call_id": "c1", "status": "success", "content": "receipt: file.txt"},
    ]
    result = CheckpointContextEngine({"mode": "live"}).compress(messages)
    assert any("Observed effect c1: succeeded" in str(m.get("content")) for m in result)
