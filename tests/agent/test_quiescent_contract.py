import pytest
from agent.checkpoint_engine import CheckpointContextEngine


def test_live_mode_requires_retained_raw_history():
    with pytest.raises(ValueError, match="raw history"):
        CheckpointContextEngine({"mode": "live", "raw_history": False})


def test_invalid_map_response_is_a_noop_without_digest_placeholder():
    engine = CheckpointContextEngine({"mode": "live", "map_routes": [{"model": "m", "structured_output": True}]}, map_caller=lambda _: "not-json")
    messages = [{"role": "user", "content": "keep me"}]
    result = engine.compress(messages)
    assert result is messages
    assert "digest unavailable" not in str(result)
