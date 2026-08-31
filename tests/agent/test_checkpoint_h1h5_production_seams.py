import json
from types import SimpleNamespace

import pytest

from agent.checkpoint_engine import CheckpointContextEngine, DurableCheckpointStore


def test_returned_tool_error_emits_failed_checkpoint_receipt(monkeypatch):
    import model_tools

    monkeypatch.setattr(
        model_tools.registry, "dispatch", lambda *_args, **_kwargs: '{"error":"denied"}'
    )
    receipts = []
    model_tools.handle_function_call(
        "write_file", {}, tool_call_id="call-error", receipt_callback=receipts.append,
        skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )
    assert receipts[0].status == "failed"
    assert receipts[0].error_type == "tool_error"


def test_production_auxiliary_map_adapter_reaches_required_structured_route(monkeypatch):
    from agent.agent_init import _build_checkpoint_map_caller

    calls = []

    def call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "schema_version": 1, "source_event_ids": [1], "facts": [],
            })))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    monkeypatch.setattr("agent.auxiliary_client.call_llm", call_llm)
    agent = SimpleNamespace(
        provider="configured-provider", model="configured-model",
        base_url="https://configured.example/v1", api_key="key", api_mode="chat_completions",
    )
    engine = CheckpointContextEngine(
        {"mode": "live", "map_routes": [{"model": "map-model", "structured_output": True}]},
        map_caller=_build_checkpoint_map_caller(agent),
    )
    engine.compress([{"role": "user", "content": "source", "_row_id": 1}])

    assert calls and calls[0]["task"] == "checkpoint"
    assert calls[0]["model"] == "map-model"
    assert calls[0]["extra_body"]["response_format"]["type"] == "json_schema"
    assert engine._last_route_attempts == ["map-model"]


def test_parser_rejects_model_text_without_evidence_and_uses_host_span_text():
    from agent.checkpoint_engine import parse_map_response

    payload = {
        "schema_version": 1,
        "source_event_ids": [1],
        "facts": [{
            "kind": "instruction", "text": "model paraphrase", "source_event_ids": [1],
            "evidence": [{"event_id": "1", "start_char": 0, "end_char": 9}],
        }],
    }
    parsed = parse_map_response(payload, expected_source_event_ids=(1,), source_events={"1": "host text only"})
    assert parsed.facts[0].text == "host text"

    payload["facts"][0].pop("evidence")
    with pytest.raises(ValueError, match="evidence"):
        parse_map_response(payload, expected_source_event_ids=(1,), source_events={"1": "host text only"})


def test_compress_kwargs_cannot_make_trace_benchmark_admissible():
    engine = CheckpointContextEngine({"mode": "shadow", "trace": True}, session_id="trace")
    engine.compress(
        [{"role": "user", "content": "hello", "_row_id": 1}],
        code_snapshot={"head": "forged", "tree": "forged", "dirty": False, "dirty_diff_hash": "forged"},
        physical_model="forged", configured_route="forged", wire_mode="structured",
    )
    assert engine.last_trace is not None
    assert engine.last_trace.execution_identity_complete is False
    assert engine.last_trace.benchmark_admissible is False


def test_artifact_recovery_is_advertised_dispatched_and_reloaded(tmp_path):
    store = DurableCheckpointStore(tmp_path)
    engine = CheckpointContextEngine({"mode": "live"}, store=store, session_id="s")
    artifact = engine.externalize_artifact("durable artifact")

    assert [schema["name"] for schema in engine.get_tool_schemas()] == ["checkpoint_artifact_read"]
    assert json.loads(engine.handle_tool_call("checkpoint_artifact_read", {"artifact_id": artifact.artifact_id})) == {
        "artifact_id": artifact.artifact_id, "content": "durable artifact"
    }

    reloaded = CheckpointContextEngine({"mode": "live"}, store=DurableCheckpointStore(tmp_path), session_id="s")
    assert reloaded.checkpoint_artifact_read(artifact.artifact_id) == "durable artifact"
