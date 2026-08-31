import json

from agent.checkpoint_engine import CheckpointContextEngine


def test_only_tool_receipt_can_mark_effect_succeeded():
    messages = [
        {"role": "user", "content": "write file"},
        {"role": "assistant", "content": "I wrote it"},
        {"role": "tool", "tool_call_id": "c1", "status": "success", "content": "receipt: file.txt"},
    ]
    result = CheckpointContextEngine(
        {"mode": "live"},
        map_caller=lambda request: {
            "schema_version": 1,
            "source_event_ids": json.loads(request["messages"][0]["content"])["source_event_ids"],
            "facts": [],
        },
    ).compress(messages)
    rendered = "\n".join(str(m.get("content")) for m in result)
    assert "Observed effect c1: observed" in rendered
    assert "Observed effect c1: succeeded" not in rendered
