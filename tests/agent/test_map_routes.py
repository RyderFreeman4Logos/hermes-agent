import json

import pytest

from agent.checkpoint_engine import CheckpointContextEngine, MapResponse, parse_map_response


def test_required_map_schema_keeps_identity_host_owned_and_binds_single_evidence():
    requests = []

    def caller(request):
        requests.append(request)
        payload = json.loads(request["messages"][0]["content"])
        assert "source_event_ids" not in payload
        schema = request["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"]) == {"facts"}
        fact = schema["properties"]["facts"]["items"]
        assert "source_event_ids" not in fact["properties"]
        evidence = fact["properties"]["evidence"]["items"]
        assert "event_index" not in evidence["properties"]
        assert "event_index" not in evidence["required"]
        assert "event_id" not in evidence["properties"]
        return {
            "facts": [{
                "kind": "observation",
                "evidence": [{"start_char": 0, "end_char": 6}],
            }],
        }

    engine = CheckpointContextEngine(
        {"mode": "live", "protect_last_n": 0}, map_caller=caller,
    )
    result = engine.compress([{"role": "user", "content": "source text", "_row_id": 1}])

    assert requests
    checkpoint = next(message["content"] for message in result if message.get("checkpoint_projection"))
    assert "observed observation: source" in checkpoint


def test_map_schema_and_parser_reject_model_owned_fact_text():
    excerpt = "x" * 7_576
    fact_schema = MapResponse.schema((1,), (excerpt,))["properties"]["facts"]["items"]

    assert "text" not in fact_schema["properties"]

    with pytest.raises(ValueError, match="^invalid map fact$"):
        parse_map_response(
            {"facts": [{
                "kind": "tool_result", "text": excerpt,
                "evidence": [{"start_char": 0, "end_char": len(excerpt)}],
            }]},
            expected_source_event_ids=(1,), source_events={"1": excerpt},
        )


def test_map_summary_is_bounded_in_schema_and_parser():
    fact_schema = MapResponse.schema((1,), ("host text",))["properties"]["facts"]["items"]

    assert fact_schema["properties"]["summary"]["maxLength"] == 512

    with pytest.raises(ValueError, match="^summary exceeds maximum length$"):
        parse_map_response(
            {"facts": [{
                "kind": "observation", "summary": "x" * 513,
                "evidence": [{"start_char": 0, "end_char": 4}],
            }]},
            expected_source_event_ids=(1,), source_events={"1": "host text"},
        )


def test_map_requests_one_textual_host_source_and_host_disposes_empty_assistant():
    requests = []

    def caller(request):
        payload = json.loads(request["messages"][0]["content"])
        requests.append(payload)
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "tool"
        assert "source_event_ids" not in payload
        evidence_schema = request["response_format"]["json_schema"]["schema"]["properties"]["facts"]["items"]["properties"]["evidence"]["items"]
        assert "event_index" not in evidence_schema["properties"]
        assert "event_index" not in evidence_schema["required"]
        text = payload["messages"][0]["content"]["evidence"][0]["text"]
        return {"facts": [{
            "kind": "tool_result",
            "evidence": [{"start_char": 0, "end_char": len(text)}],
        }]}

    engine = CheckpointContextEngine(
        {"mode": "live", "protect_first_n": 0, "protect_last_n": 0}, map_caller=caller,
    )
    messages = [
        {"role": "assistant", "content": None, "_row_id": 2516801, "tool_calls": [
            {"id": "call-1", "function": {"name": "read_file"}},
            {"id": "call-2", "function": {"name": "read_file"}},
            {"id": "call-3", "function": {"name": "read_file"}},
            {"id": "call-4", "function": {"name": "read_file"}},
        ]},
        *[
            {"role": "tool", "tool_call_id": f"call-{n}", "content": f"tool {n} text", "_row_id": 2516801 + n}
            for n in range(1, 5)
        ],
    ]

    result = engine.compress(messages)

    assert result is not messages
    assert len(requests) == 4
    assert engine.last_rejection is None
    checkpoint = next(message["content"] for message in result if message.get("checkpoint_projection"))
    assert all(f"observed tool_result: tool {n}" in checkpoint for n in range(1, 5))


def test_required_map_default_shard_cap_allows_40_singleton_sources():
    requests = []

    def caller(request):
        requests.append(request)
        payload = json.loads(request["messages"][0]["content"])
        assert len(payload["messages"]) == 1
        return {"facts": []}

    engine = CheckpointContextEngine({"mode": "live"}, map_caller=caller)
    messages = [
        {"role": "user", "content": f"source {index}", "_row_id": index}
        for index in range(40)
    ]

    assert engine.compress(messages) is not messages
    assert engine.last_rejection is None
    assert len(requests) == 40


@pytest.mark.parametrize(("span", "error"), [
    ({"end_char": 1}, "missing evidence start_char"),
    ({"start_char": 0}, "missing evidence end_char"),
    ({"start_char": "0", "end_char": 1}, "missing evidence start_char"),
    ({"start_char": 0, "end_char": "1"}, "missing evidence end_char"),
])
def test_map_parser_rejects_missing_or_non_integer_evidence_chars_without_salvage(span, error):
    with pytest.raises(ValueError, match=f"^{error}$"):
        parse_map_response(
            {"facts": [{"kind": "text", "evidence": [span]}]},
            expected_source_event_ids=(25,), source_events={"25": "host text"},
        )


def test_map_parser_binds_omitted_event_index_to_the_single_host_source():
    parsed = parse_map_response(
        {"facts": [{"kind": "observation", "evidence": [{"start_char": 0, "end_char": 4}]}]},
        expected_source_event_ids=(2516802,), source_events={"2516802": "host text"},
    )

    assert parsed.facts[0].evidence[0].event_id == "2516802"


def test_map_parser_rejects_model_owned_identity_fields():
    with pytest.raises(ValueError, match="^invalid map schema$"):
        parse_map_response(
            {"schema_version": 1, "facts": []},
            expected_source_event_ids=(1,), source_events={"1": "host text"},
        )


@pytest.mark.parametrize("policy", ("preferred", "disabled"))
def test_nonrequired_map_rejects_shard_limit_before_building_multi_source_request(policy):
    calls = []
    messages = [
        {"role": "user", "content": "first source", "_row_id": 1},
        {"role": "assistant", "content": "second source", "_row_id": 2},
    ]
    engine = CheckpointContextEngine(
        {"mode": "live", "structured_output": policy, "max_map_shards": 1},
        map_caller=calls.append,
    )

    assert engine.compress(messages) is messages
    assert engine.last_rejection == "Map plan exceeds configured shard limit"
    assert calls == []


@pytest.mark.parametrize("policy", ("preferred", "disabled"))
def test_nonrequired_map_uses_singleton_requests_when_shard_limit_allows(policy):
    requests = []

    def caller(request):
        requests.append(request)
        payload = json.loads(request["messages"][0]["content"])
        assert len(payload["messages"]) == 1
        assert "event_index" not in json.dumps(request)
        return {"facts": []}

    engine = CheckpointContextEngine(
        {"mode": "live", "structured_output": policy, "max_map_shards": 2}, map_caller=caller,
    )

    messages = [
        {"role": "user", "content": "first source", "_row_id": 1},
        {"role": "assistant", "content": "second source", "_row_id": 2},
    ]
    result = engine.compress(messages)

    assert result is not messages
    assert engine.last_rejection is None
    assert len(requests) == 2


def test_required_map_hides_externalized_artifact_identity_from_model():
    requests = []

    def caller(request):
        requests.append(request)
        payload = json.loads(request["messages"][0]["content"])
        tool_content = payload["messages"][0]["content"]
        assert "artifact_id" not in tool_content
        assert "source_event_id" not in tool_content
        assert tool_content["evidence"][0]["text"].startswith("host excerpt")
        schema = request["response_format"]["json_schema"]["schema"]
        evidence = schema["properties"]["facts"]["items"]["properties"]["evidence"]["items"]
        assert "event_index" not in evidence["properties"]
        return {"facts": [{
            "kind": "tool_result",
            "evidence": [{"start_char": 0, "end_char": 12}],
        }]}

    engine = CheckpointContextEngine(
        {"mode": "live", "protect_last_n": 0}, map_caller=caller,
    )
    result = engine.compress([
        {"role": "assistant", "content": None, "_row_id": 1,
         "tool_calls": [{"id": "call-1", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "host excerpt plus hidden artifact", "_row_id": 2},
    ])

    assert requests
    checkpoint = next(message["content"] for message in result if message.get("checkpoint_projection"))
    assert "observed tool_result: host excerpt" in checkpoint


def test_required_map_uses_only_configured_structured_fallback():
    calls = []

    def caller(request):
        calls.append(request["model"])
        if request["model"] == "bad":
            raise RuntimeError("route unavailable")
        return {"facts": []}

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
        return {"facts": []}

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


def test_required_map_externalizes_an_oversized_tool_result_before_planning():
    requests = []

    def caller(request):
        requests.append(request)
        return {"facts": []}

    engine = CheckpointContextEngine(
        {"mode": "live", "map": {"max_output_tokens": 32_768}}, map_caller=caller,
    )
    tool_body = "x" * 70_000
    messages = [
        {"role": "assistant", "content": None, "_row_id": 1,
         "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": tool_body, "_row_id": 2},
    ]

    result = engine.compress(messages)

    assert result is not messages
    assert engine.last_rejection is None
    assert requests and all(request["max_tokens"] == 16_384 for request in requests)
    payload = json.loads(requests[0]["messages"][0]["content"])
    tool_pointer = payload["messages"][0]["content"]
    assert tool_body not in requests[0]["messages"][0]["content"]
    assert "artifact_id" not in tool_pointer
    assert len(engine._map_artifact_ids) == 1
    assert engine.checkpoint_artifact_read(next(iter(engine._map_artifact_ids))) == tool_body


def test_required_map_binds_evidence_bounds_and_parser_to_wire_excerpt():
    excerpt = "x" * 5_339
    map_body = {
        "facts": [{
            "kind": "tool_result",
            "evidence": [{"start_char": 0, "end_char": 168}],
        }],
    }

    def caller(request):
        payload = json.loads(request["messages"][0]["content"])
        wire_message = payload["messages"][0]
        assert "api_content" not in wire_message
        wire_excerpt = wire_message["content"]
        span_schema = request["response_format"]["json_schema"]["schema"]["properties"]["facts"]["items"]["properties"]["evidence"]["items"]
        assert len(wire_excerpt) == len(excerpt)
        assert span_schema["properties"]["start_char"]["maximum"] == len(wire_excerpt)
        assert span_schema["properties"]["end_char"]["maximum"] == len(wire_excerpt)
        return map_body

    engine = CheckpointContextEngine(
        {"mode": "live", "protect_last_n": 0}, map_caller=caller,
    )
    result = engine.compress([{
        "role": "user", "content": "clean", "api_content": excerpt, "_row_id": 1,
    }])

    checkpoint = next(message["content"] for message in result if message.get("checkpoint_projection"))
    assert "observed tool_result: " + excerpt[:168] in checkpoint


def test_required_map_rejects_externalized_evidence_past_host_excerpt():
    tool_body = "host-visible evidence\n" + "x" * 70_000

    def caller(request):
        return {
            "facts": [{
                "kind": "tool_result",
                "evidence": [{"start_char": 0, "end_char": len(tool_body)}],
            }],
        }

    engine = CheckpointContextEngine(
        {"mode": "live", "protect_last_n": 0}, map_caller=caller,
    )
    messages = [
        {"role": "assistant", "content": None, "_row_id": 1,
         "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": tool_body, "_row_id": 2},
    ]

    assert engine.compress(messages) is messages
    assert "evidence span exceeds source bounds" in (engine.last_rejection or "")


def test_required_map_extracts_only_host_excerpt_from_externalized_evidence():
    tool_body = "host-visible evidence\n" + "x" * 70_000

    def caller(request):
        return {
            "facts": [{
                "kind": "tool_result",
                "evidence": [{"start_char": 0, "end_char": 21}],
            }],
        }

    engine = CheckpointContextEngine(
        {"mode": "live", "protect_last_n": 0}, map_caller=caller,
    )
    result = engine.compress([
        {"role": "assistant", "content": None, "_row_id": 1,
         "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": tool_body, "_row_id": 2},
    ])

    checkpoint = next(message["content"] for message in result if message.get("checkpoint_projection"))
    assert "observed tool_result: host-visible evidence" in checkpoint
    assert tool_body not in checkpoint


def test_checkpoint_keeps_externalized_tool_artifact_after_tail_eviction():
    def caller(request):
        return {"facts": []}

    tool_body = "durable artifact result\n" + "x" * 70_000
    engine = CheckpointContextEngine(
        {"mode": "live", "protect_last_n": 2}, map_caller=caller,
    )
    result = engine.compress([
        {"role": "assistant", "content": None, "_row_id": 1,
         "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": tool_body, "_row_id": 2},
        {"role": "user", "content": "later one", "_row_id": 3},
        {"role": "assistant", "content": "later two", "_row_id": 4},
        {"role": "user", "content": "later three", "_row_id": 5},
    ])

    checkpoint = next(message["content"] for message in result if message.get("checkpoint_projection"))
    artifact_id = next(
        line.removeprefix("Artifact available via checkpoint_artifact_read: ")
        for line in checkpoint.splitlines()
        if line.startswith("Artifact available via checkpoint_artifact_read: ")
    )
    assert all(message.get("tool_call_id") != "call-1" for message in result)
    assert engine.checkpoint_artifact_read(artifact_id) == tool_body


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
        return {"facts": []}

    engine = CheckpointContextEngine(
        {"mode": "live", "map": {"max_output_tokens": 32_768}}, map_caller=caller,
    )

    engine.compress([{"role": "user", "content": "source", "_row_id": 1}])

    assert requests[0]["max_tokens"] == 16_384
