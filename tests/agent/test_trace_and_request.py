from agent.checkpoint_engine import CheckpointContextEngine


def test_trace_contains_reproducible_input_and_output_hashes():
    engine = CheckpointContextEngine({"mode": "shadow", "trace": True}, session_id="trace")
    engine.compress([{"role": "user", "content": "hello", "_row_id": 1}])
    assert engine.last_trace is not None
    assert engine.last_trace.prompt_hash
    assert engine.last_trace.schema_hash
    assert not engine.last_trace.response_hash
    assert engine.last_trace.benchmark_admissible is False


def test_final_request_uses_one_provider_preparation_boundary():
    engine = CheckpointContextEngine({"mode": "live"})
    messages = [{"role": "user", "content": "hello"}]
    projected = engine.compress(messages)
    request = engine.prepare_provider_request(projected, model="m")
    assert request["model"] == "m"
    assert request["messages"] == projected
    assert "estimated_input_tokens" in request
