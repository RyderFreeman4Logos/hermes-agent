import pytest

from agent.checkpoint_engine import (
    ArtifactReference,
    CheckpointContextEngine,
    HostLifecycleEvent,
    ToolExecutionReceipt,
)
from agent.checkpoint_engine.replay import (
    CheckpointReplayAdapter,
    ContinuationReplayAdapter,
    LeanReplayAdapter,
    RecordedOriginalAdapter,
    ReplayUnavailable,
)


def test_task_epochs_and_effects_use_typed_host_events_and_receipts():
    engine = CheckpointContextEngine({"mode": "live"})
    event = HostLifecycleEvent(7, "task_started", "task-7", 3)
    receipt = ToolExecutionReceipt(
        tool_call_id="c1", tool_name="write", effect_class="workspace_mutation",
        status="succeeded", exit_code=0, error_type=None, artifact_refs=(),
        source_event_ids=(8,),
    )
    state = engine.reduce_host_state(
        [{"role": "user", "content": "write it", "_row_id": 7},
         {"role": "tool", "tool_call_id": "c1", "name": "write", "content": "ok", "_row_id": 8}],
        host_events=(event,), tool_receipts=(receipt,),
    )
    assert state.epochs == (engine.task_epochs((event,))[0],)
    assert state.effects[0].status == "succeeded"


def test_artifact_reference_has_exact_read_path(tmp_path):
    engine = CheckpointContextEngine({"mode": "live"}, artifact_root=tmp_path)
    ref = engine.externalize_artifact("exact receipt")
    assert isinstance(ref, ArtifactReference)
    assert engine.checkpoint_artifact_read(ref.artifact_id) == "exact receipt"


def test_named_replay_adapters_do_not_fallback_to_post_context():
    messages = [{"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "write"}}]}]
    for adapter in (
        CheckpointReplayAdapter({}), LeanReplayAdapter({}),
        RecordedOriginalAdapter({}),
    ):
        with pytest.raises(ReplayUnavailable):
            adapter.replay(messages, post_context=[{"role": "assistant", "content": "old"}])


def test_continuation_replay_executes_frozen_continuation():
    calls = []
    adapter = ContinuationReplayAdapter(
        {"read": lambda call: {"status": "succeeded", "content": "receipt"}},
        continuation=lambda context: calls.append(context) or [{"role": "assistant", "content": "continued"}],
    )
    result = adapter.replay(
        [{"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "read"}}]}],
        continuation_context=[{"role": "user", "content": "continue"}],
    )
    assert result[-1]["content"] == "continued"
    assert calls == [[{"role": "user", "content": "continue"}]]
