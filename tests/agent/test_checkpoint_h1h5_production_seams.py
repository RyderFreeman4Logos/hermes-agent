import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.checkpoint_engine import CheckpointContextEngine, CheckpointMapCallRejected, DurableCheckpointStore
from hermes_state import SessionDB


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
                "facts": [],
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


def test_production_map_externalization_sends_bounded_host_evidence(monkeypatch):
    from agent.agent_init import _build_checkpoint_map_caller

    calls = []

    def call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "facts": [],
            })))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    monkeypatch.setattr("agent.auxiliary_client.call_llm", call_llm)
    agent = SimpleNamespace(
        provider="configured-provider", model="configured-model",
        base_url="https://configured.example/v1", api_key="key", api_mode="chat_completions",
    )
    tool_body = "host evidence: keep this exact result\n" + "x" * 70_000
    engine = CheckpointContextEngine(
        {"mode": "live", "map_routes": [{"model": "map-model", "structured_output": True}]},
        map_caller=_build_checkpoint_map_caller(agent),
    )

    assert engine.compress([
        {"role": "assistant", "content": None, "_row_id": 1,
         "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": tool_body, "_row_id": 2},
    ])

    outbound = calls[0]["messages"][0]["content"]
    tool_payload = json.loads(outbound)["messages"][0]["content"]
    assert tool_body not in outbound
    assert "artifact_id" not in tool_payload
    assert "source_event_id" not in tool_payload
    assert tool_payload["evidence"][0]["text"] == tool_body[:tool_payload["evidence"][0]["end_char"]]
    assert len(tool_payload["evidence"][0]["text"]) < len(tool_body)


def test_production_map_forwards_the_effective_output_cap(monkeypatch):
    from agent.agent_init import _build_checkpoint_map_caller

    calls = []
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "facts": [],
            })))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        ),
    )
    agent = SimpleNamespace(
        provider="configured-provider", model="configured-model",
        base_url="https://configured.example/v1", api_key="key", api_mode="chat_completions",
    )
    engine = CheckpointContextEngine(
        {
            "mode": "live", "map": {"max_output_tokens": 32_768},
            "map_routes": [{"model": "map-model", "structured_output": True}],
        },
        map_caller=_build_checkpoint_map_caller(agent),
    )

    engine.compress([{"role": "user", "content": "source", "_row_id": 1}])

    assert calls[0]["max_tokens"] == 16_384


def test_checkpoint_transport_keeps_its_configured_output_cap_on_the_wire():
    from agent.auxiliary_client import _build_call_kwargs

    request = _build_call_kwargs(
        "custom", "map-model", [{"role": "user", "content": "source"}],
        max_tokens=16_384, task="checkpoint",
    )

    assert request["max_tokens"] == 16_384


def test_production_codex_responses_map_forwards_cap_and_rejects_truncated_json(monkeypatch):
    """The real auxiliary Responses adapter must preserve both Map guards."""
    from agent.agent_init import _build_checkpoint_map_caller
    from agent.auxiliary_client import CodexAuxiliaryClient

    valid_map_json = json.dumps({"facts": []})

    class FakeResponses:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                output=[SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text=valid_map_json)],
                )],
                usage=SimpleNamespace(input_tokens=3, output_tokens=16_384, total_tokens=16_387),
            )

    real_client = SimpleNamespace(
        api_key="key", base_url="https://chatgpt.com/backend-api/codex", responses=FakeResponses(),
    )
    client = CodexAuxiliaryClient(real_client, "map-model")
    agent = SimpleNamespace(
        provider="configured-provider", model="configured-model",
        base_url="https://configured.example/v1", api_key="key", api_mode="codex_responses",
    )
    engine = CheckpointContextEngine(
        {"mode": "live", "trace": True, "map_routes": [{"model": "map-model", "structured_output": True}]},
        map_caller=_build_checkpoint_map_caller(agent),
    )

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openai-codex", "map-model", None, None, "codex_responses")),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "map-model")),
    ):
        source = [{"role": "user", "content": "source", "_row_id": 1}]
        assert engine.compress(source) is source

    wire_kwargs = dict(real_client.responses.kwargs)
    wire_kwargs.update(wire_kwargs.get("extra_body") or {})
    assert wire_kwargs["max_output_tokens"] == 16_384
    assert engine._map_attempt_records[-1].finish_reason == "length"
    assert "finish_reason=length" in (engine.last_rejection or "")


def test_production_map_does_not_replay_structured_rejection_prompt_only():
    from agent.agent_init import _build_checkpoint_map_caller

    client = MagicMock()
    client.base_url = "https://api.openai.com/v1"
    client.chat.completions.create.side_effect = [
        RuntimeError("HTTP 400: This response_format type is unavailable now"),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "facts": [],
            })))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        ),
    ]
    agent = SimpleNamespace(
        provider="configured-provider", model="configured-model",
        base_url="https://configured.example/v1", api_key="key", api_mode="chat_completions",
    )
    engine = CheckpointContextEngine(
        {"mode": "live", "map_routes": [{"model": "map-model", "structured_output": True}]},
        map_caller=_build_checkpoint_map_caller(agent),
    )

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openai-codex", "map-model", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "map-model")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda response, *_a, **_k: response),
    ):
        result = engine.compress([{"role": "user", "content": "source", "_row_id": 1}])

    assert client.chat.completions.create.call_count == 1, client.chat.completions.create.call_args_list
    assert result == [{"role": "user", "content": "source", "_row_id": 1}]
    assert "response_format" in client.chat.completions.create.call_args.kwargs["extra_body"]


def test_production_map_records_each_transient_physical_send(monkeypatch):
    from agent.agent_init import _build_checkpoint_map_caller

    client = MagicMock()
    client.base_url = "https://api.openai.com/v1"
    client.chat.completions.create.side_effect = [
        RuntimeError("connection reset"),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "facts": [],
            })), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        ),
    ]
    agent = SimpleNamespace(
        provider="configured-provider", model="configured-model",
        base_url="https://configured.example/v1", api_key="key", api_mode="chat_completions",
    )
    engine = CheckpointContextEngine(
        {"mode": "live", "trace": True, "map_routes": [{"model": "map-model", "structured_output": True}]},
        map_caller=_build_checkpoint_map_caller(agent),
    )

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openai-codex", "map-model", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "map-model")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda response, *_a, **_k: response),
        patch("agent.auxiliary_client._is_transient_transport_error", return_value=True),
        patch("agent.auxiliary_client._transient_retry_count", return_value=1),
        patch("agent.auxiliary_client.time.sleep"),
    ):
        engine.compress([{"role": "user", "content": "source", "_row_id": 1}])

    records = engine.last_trace.map_attempt_records
    assert client.chat.completions.create.call_count == 2
    assert [(record.configured_route, record.physical_model, record.actual_wire_mode)
            for record in records] == [
        ("openai-codex", "map-model", "structured"),
        ("openai-codex", "map-model", "structured"),
    ]
    assert records[0].fallback_rejection and "transient" in records[0].fallback_rejection
    assert records[1].input_tokens == 3
    assert records[1].output_tokens == 2
    assert records[1].finish_reason == "stop"
    assert records[1].response_hash
    assert engine.last_trace.benchmark_admissible is False


def test_production_map_records_primary_and_fallback_physical_sends(monkeypatch):
    from agent.agent_init import _build_checkpoint_map_caller
    from agent.auxiliary_client import _FallbackDestination

    primary = MagicMock()
    primary.base_url = "https://primary.example/v1"
    primary.chat.completions.create.side_effect = RuntimeError("connection down")
    fallback = MagicMock()
    fallback.base_url = "https://fallback.example/v1"
    fallback.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "facts": [],
        })), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=4),
    )
    agent = SimpleNamespace(
        provider="configured-provider", model="configured-model",
        base_url="https://configured.example/v1", api_key="key", api_mode="chat_completions",
    )
    engine = CheckpointContextEngine(
        {"mode": "live", "trace": True, "map_routes": [{"model": "primary-model", "structured_output": True}]},
        map_caller=_build_checkpoint_map_caller(agent),
    )

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("auto", "primary-model", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda response, *_a, **_k: response),
        patch("agent.auxiliary_client._is_transient_transport_error", return_value=False),
        patch("agent.auxiliary_client._is_connection_error", return_value=True),
        patch("agent.auxiliary_client._try_configured_fallback_chain",
              return_value=(fallback, "fallback-model", "fallback_chain[0](fallback-provider)")),
        patch("agent.auxiliary_client._fallback_destination", return_value=_FallbackDestination(
            "fallback-provider", "https://fallback.example/v1", None, "fallback-model",
        )),
        patch("agent.auxiliary_client._replan_synchronous_cache_sections", side_effect=lambda messages, tools, **_k: (messages, tools)),
    ):
        engine.compress([{"role": "user", "content": "source", "_row_id": 1}])

    records = engine.last_trace.map_attempt_records
    assert primary.chat.completions.create.call_count == 1
    assert fallback.chat.completions.create.call_count == 1
    assert [(record.configured_route, record.physical_model, record.actual_wire_mode)
            for record in records] == [
        ("auto", "primary-model", "structured"),
        ("fallback-provider", "fallback-model", "structured"),
    ]
    assert records[0].fallback_rejection and "rejected" in records[0].fallback_rejection
    assert records[1].input_tokens == 5
    assert records[1].output_tokens == 4
    assert records[1].finish_reason == "stop"
    assert records[1].response_hash
    assert engine.last_trace.benchmark_admissible is False


def test_production_map_amends_invalid_physical_primary_before_fallback():
    from agent.agent_init import _build_checkpoint_map_caller
    from agent.auxiliary_client import _FallbackDestination

    primary = MagicMock()
    primary.base_url = "https://primary.example/v1"
    primary.chat.completions.create.return_value = SimpleNamespace()
    fallback = MagicMock()
    fallback.base_url = "https://fallback.example/v1"
    fallback.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "facts": [],
        })), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=4),
    )
    agent = SimpleNamespace(
        provider="configured-provider", model="configured-model",
        base_url="https://configured.example/v1", api_key="key", api_mode="chat_completions",
    )
    engine = CheckpointContextEngine(
        {"mode": "live", "trace": True, "map_routes": [{"model": "primary-model", "structured_output": True}]},
        map_caller=_build_checkpoint_map_caller(agent),
    )

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("auto", "primary-model", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")),
        patch("agent.auxiliary_client._try_configured_fallback_chain",
              return_value=(fallback, "fallback-model", "fallback_chain[0](fallback-provider)")),
        patch("agent.auxiliary_client._fallback_destination", return_value=_FallbackDestination(
            "fallback-provider", "https://fallback.example/v1", None, "fallback-model",
        )),
        patch("agent.auxiliary_client._replan_synchronous_cache_sections", side_effect=lambda messages, tools, **_k: (messages, tools)),
    ):
        engine.compress([{"role": "user", "content": "source", "_row_id": 1}])

    records = engine.last_trace.map_attempt_records
    assert primary.chat.completions.create.call_count == 1
    assert fallback.chat.completions.create.call_count == 1
    assert [(record.configured_route, record.physical_model)
            for record in records] == [
        ("auto", "primary-model"),
        ("fallback-provider", "fallback-model"),
    ]
    assert records[0].fallback_rejection and "rejected" in records[0].fallback_rejection
    assert engine.last_trace.benchmark_admissible is False


def test_parser_rejects_model_text_without_evidence_and_uses_host_span_text():
    from agent.checkpoint_engine import parse_map_response

    payload = {
        "facts": [{
            "kind": "instruction", "text": "model paraphrase",
            "evidence": [{"start_char": 0, "end_char": 9}],
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


def test_trace_requires_every_map_attempt_to_be_complete_and_structured():
    attempts = iter([
        {
            "content": {"facts": []},
            "_checkpoint_identity": {
                "configured_route": "first", "physical_model": "m1",
                "actual_wire_mode": "prompt_only", "fallback_rejection": "response_format rejected",
                "input_tokens": 3, "output_tokens": 2, "latency_ms": 4,
                "finish_reason": "stop", "response_hash": "first",
                "code_snapshot": {"head": "h", "tree": "t", "dirty": False, "dirty_diff_hash": "d"},
            },
        },
        {
            "content": {"facts": []},
            "_checkpoint_identity": {
                "configured_route": "last", "physical_model": "m2",
                "actual_wire_mode": "structured", "fallback_rejection": "",
                "input_tokens": None, "output_tokens": 2, "latency_ms": None,
                "finish_reason": "stop", "response_hash": "last",
                "code_snapshot": {"head": "h", "tree": "t", "dirty": False, "dirty_diff_hash": "d"},
            },
        },
    ])
    engine = CheckpointContextEngine(
        {"mode": "live", "trace": True, "max_map_shards": 2}, map_caller=lambda _request: next(attempts),
    )
    engine.compress([
        {"role": "user", "content": "first", "_row_id": 1},
        {"role": "assistant", "content": "second", "_row_id": 2},
    ])

    assert engine.last_trace is not None
    assert len(engine.last_trace.map_attempt_records) == 2
    assert engine.last_trace.execution_identity_complete is False
    assert engine.last_trace.benchmark_admissible is False


def test_rejected_physical_route_is_retained_when_next_structured_route_succeeds():
    identity = {
        "configured_route": "rejected", "physical_model": "m1", "actual_wire_mode": "structured",
        "fallback_rejection": "response_format rejected", "input_tokens": None, "output_tokens": None,
        "latency_ms": 1, "finish_reason": None, "response_hash": None,
        "code_snapshot": {"head": "h", "tree": "t", "dirty": False, "dirty_diff_hash": "d"},
    }
    calls = 0

    def caller(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CheckpointMapCallRejected(identity, RuntimeError("response_format rejected"))
        return {"content": {"facts": []}, "_checkpoint_identity": {
            **identity, "configured_route": "second", "physical_model": "m2",
            "fallback_rejection": None, "input_tokens": 3, "output_tokens": 2,
            "finish_reason": "stop", "response_hash": "ok",
        }}

    engine = CheckpointContextEngine({"mode": "live", "trace": True, "map_routes": [
        {"model": "m1", "structured_output": True}, {"model": "m2", "structured_output": True},
    ]}, map_caller=caller)
    engine.compress([{"role": "user", "content": "source", "_row_id": 1}])

    assert engine.last_trace is not None
    assert engine.last_trace.route_attempts == ("m1", "m2")
    assert len(engine.last_trace.map_attempt_records) == 2
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


def test_production_bound_store_persists_compaction_artifact_across_restore(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    map_caller = lambda request: {"facts": []}
    first = CheckpointContextEngine({"mode": "live"}, session_id="s", map_caller=map_caller)
    first.bind_session_state(db, "s")
    first.compress([{"role": "user", "content": "durable source", "_row_id": 1}])
    generation = first._store.generation("s")

    assert generation is not None and generation.artifact_dependencies
    artifact_id = generation.artifact_dependencies[0]
    restored = CheckpointContextEngine({"mode": "live"}, session_id="s")
    restored.bind_session_state(SessionDB(db_path=tmp_path / "state.db"), "s")
    assert json.loads(restored.handle_tool_call("checkpoint_artifact_read", {"artifact_id": artifact_id}))["content"].startswith("CHECKPOINT")
