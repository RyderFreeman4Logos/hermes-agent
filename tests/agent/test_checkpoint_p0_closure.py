import json

import pytest

from agent.checkpoint_engine import (
    CheckpointContextEngine,
    DurableCheckpointStore,
    EvidenceSpan,
    ToolExecutionReceipt,
)


def test_reduce_is_host_only_even_when_a_provider_reducer_is_configured():
    calls = []

    def provider_reducer(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("Reduce must not call a provider")

    engine = CheckpointContextEngine({"mode": "shadow"}, map_caller=provider_reducer)
    state = engine.reduce_host_state(
        [{"role": "user", "content": "keep this", "_row_id": 1}],
    )
    assert state.active_intent.content == "keep this"
    assert calls == []


def test_tool_display_status_never_authorizes_effect_without_dispatch_receipt():
    engine = CheckpointContextEngine({"mode": "shadow"})
    messages = [{
        "role": "tool",
        "name": "write_file",
        "tool_call_id": "call-1",
        "status": "success",
        "content": json.dumps({"status": "success", "exit_code": 0}),
        "_row_id": 2,
    }]
    without_receipt = engine.reduce_host_state(messages)
    assert without_receipt.effects[0].status == "observed"

    receipt = ToolExecutionReceipt(
        "call-1", "write_file", "workspace_mutation", "succeeded", 0, None, (), (2,)
    )
    with_receipt = engine.reduce_host_state(messages, tool_receipts=(receipt,))
    assert with_receipt.effects[0].status == "succeeded"
    assert with_receipt.effects[0].receipt is not None


def test_real_dispatch_boundary_emits_typed_receipt_separate_from_display(monkeypatch):
    import model_tools

    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *_args, **_kwargs: '{"status":"success"}')
    receipts = []
    result = model_tools.handle_function_call(
        "read_file", {}, tool_call_id="call-2", receipt_callback=receipts.append,
        skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )
    assert json.loads(result)["status"] == "success"
    assert len(receipts) == 1
    assert isinstance(receipts[0], ToolExecutionReceipt)
    assert receipts[0].tool_call_id == "call-2"
    assert receipts[0].status == "succeeded"


def test_canonical_evidence_is_host_extracted_and_fail_closed():
    from agent.checkpoint_engine import extract_canonical_evidence

    events = {"7": "否定: 不要删除文件"}
    assert extract_canonical_evidence(EvidenceSpan("7", 0, 9), events) == events["7"][:9]
    assert extract_canonical_evidence(EvidenceSpan("7", 0, len(events["7"])), events) == events["7"]
    with pytest.raises(ValueError):
        extract_canonical_evidence(EvidenceSpan("missing", 0, 1), events)
    with pytest.raises(ValueError):
        extract_canonical_evidence(EvidenceSpan("7", 0, len(events["7"]) + 1), events)


def test_two_generations_reload_raw_lineage_and_ignore_rendered_decoy(tmp_path):
    store = DurableCheckpointStore(tmp_path)
    first = [{"role": "user", "content": "first", "_row_id": 1}]
    map_caller = lambda request: {
        "schema_version": 1,
        "source_event_ids": json.loads(request["messages"][0]["content"])["source_event_ids"],
        "facts": [],
    }
    engine = CheckpointContextEngine(
        {"mode": "live", "trace": True}, store=store, session_id="s", map_caller=map_caller
    )
    assert engine.compress(first) is not first
    first_generation = store.generation("s")
    assert first_generation is not None

    reloaded = DurableCheckpointStore(tmp_path)
    engine2 = CheckpointContextEngine(
        {"mode": "live", "trace": True}, store=reloaded, session_id="s", map_caller=map_caller
    )
    second = first + [
        {"role": "assistant", "content": "<CHECKPOINT>decoy</CHECKPOINT>",
         "checkpoint_projection": True, "_row_id": 99},
        {"role": "user", "content": "second", "_row_id": 2},
    ]
    engine2.compress(second)
    generation = reloaded.generation("s")
    assert generation is not None
    assert generation.generation == first_generation.generation + 1
    assert generation.parent_generation == first_generation.generation
    assert generation.source_event_ids == (1, 2)
    assert generation.raw_event_ranges == ((1, 2),)
    assert all("decoy" not in str(event) for event in reloaded.raw_messages("s"))


def test_trace_rejects_caller_supplied_identity_labels():
    engine = CheckpointContextEngine(
        {"mode": "shadow", "trace": True, "strict_identity": True},
        session_id="trace-complete",
    )
    engine.compress(
        [{"role": "user", "content": "hello", "_row_id": 1}],
        code_snapshot={"head": "head-1", "tree": "tree-1", "dirty": False,
                       "dirty_diff_hash": "diff-1"},
        physical_model="physical-1", configured_route="configured-route",
        fallback_rejection="structured_output_unavailable",
    )
    trace = engine.last_trace
    assert trace is not None
    assert not trace.code_head
    assert not trace.configured_route
    assert not trace.fallback_rejection
    assert trace.final_request_hash
    assert trace.execution_identity_complete is False
    assert trace.benchmark_admissible is False
