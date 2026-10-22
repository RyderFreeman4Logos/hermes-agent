import json
import pytest
from agent.checkpoint_engine import (
    StructuredOutputPolicy, StructuredOutputUnavailable, prepare_provider_request,
    MapResponse, EvidenceSpan, parse_map_response,
)


def test_required_structured_request_is_real_wire_contract():
    request = prepare_provider_request([{"role": "user", "content": "x"}],
                                      policy=StructuredOutputPolicy.REQUIRED,
                                      schema=MapResponse.schema())
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "extra_body" not in request


def test_required_rejects_routes_without_structured_support():
    with pytest.raises(StructuredOutputUnavailable):
        prepare_provider_request([], policy=StructuredOutputPolicy.REQUIRED,
                                 route_capabilities={"structured_output": False})


def test_disabled_sends_no_structured_request():
    request = prepare_provider_request([], policy=StructuredOutputPolicy.DISABLED,
                                       schema=MapResponse.schema())
    assert "response_format" not in request


def test_map_parser_rejects_repair_and_evidence_is_typed():
    payload = {"facts": []}
    with pytest.raises(ValueError):
        parse_map_response("```json\\n" + json.dumps(payload) + "\\n```", expected_source_event_ids=(1,))
    assert EvidenceSpan("e1", 0, 2).text("abcd") == "ab"
