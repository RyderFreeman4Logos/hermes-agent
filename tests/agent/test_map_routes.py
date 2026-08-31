from agent.checkpoint_engine import CheckpointContextEngine


def test_required_map_uses_only_configured_structured_fallback():
    calls = []

    def caller(request):
        calls.append(request["model"])
        if request["model"] == "bad":
            raise RuntimeError("route unavailable")
        return {"schema_version": 1, "source_event_ids": [0], "facts": []}

    engine = CheckpointContextEngine({
        "mode": "shadow", "structured_output": "required",
        "map_routes": [
            {"model": "bad", "structured_output": True},
            {"model": "good", "structured_output": True},
        ],
    }, map_caller=caller)
    engine.compress([{"role": "user", "content": "hi"}])
    assert calls == ["bad", "good"]


def test_preferred_route_visibly_downgrades_wire_request():
    request = []

    def caller(payload):
        request.append(payload)
        return {"schema_version": 1, "source_event_ids": [0], "facts": []}

    engine = CheckpointContextEngine({
        "structured_output": "preferred", "map_routes": [{"model": "plain", "structured_output": False}],
    }, map_caller=caller)
    engine.compress([{"role": "user", "content": "hi"}])
    assert "response_format" not in request[0]


def test_required_map_length_is_inadmissible_before_json_parse():
    engine = CheckpointContextEngine(
        {"mode": "live", "trace": True},
        map_caller=lambda _request: {
            "content": '{"schema_version": 1,',
            "_checkpoint_identity": {
                "actual_wire_mode": "structured",
                "finish_reason": "length",
                "output_tokens": 16_384,
            },
        },
    )
    messages = [{"role": "user", "content": "source", "_row_id": 1}]

    assert engine.compress(messages) is messages
    assert "finish_reason=length" in (engine.last_rejection or "")
    assert [(record.actual_wire_mode, record.finish_reason, record.output_tokens)
            for record in engine._map_attempt_records] == [("structured", "length", 16_384)]


def test_required_map_rejects_a_causal_group_over_effective_output_cap_before_send():
    calls = []
    engine = CheckpointContextEngine(
        {"mode": "live", "map": {"max_output_tokens": 32_768}},
        map_caller=lambda request: calls.append(request),
    )
    messages = [{"role": "user", "content": "x" * 70_000, "_row_id": 1}]

    assert engine.compress(messages) is messages
    assert "effective Map output cap (16384)" in (engine.last_rejection or "")
    assert calls == []


def test_required_map_caps_configured_output_at_effective_route_limit():
    requests = []

    def caller(request):
        requests.append(request)
        return {"schema_version": 1, "source_event_ids": [1], "facts": []}

    engine = CheckpointContextEngine(
        {"mode": "live", "map": {"max_output_tokens": 32_768}}, map_caller=caller,
    )

    engine.compress([{"role": "user", "content": "source", "_row_id": 1}])

    assert requests[0]["max_tokens"] == 16_384
