"""Checkpoint ContextEngine: opt-in shadow no-op (DESIGN.md §10 item 1)."""

from copy import deepcopy
import json
from types import SimpleNamespace

from hermes_cli.config_defaults import DEFAULT_CONFIG
from plugins.context_engine import discover_context_engines, load_context_engine


class _FakeAuxiliaryClient:
    def __init__(self, *responses):
        self.calls = []
        self._responses = list(responses)

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeMainModel:
    def __init__(self):
        self.calls = 0

    def complete(self, **_kwargs):
        self.calls += 1
        raise AssertionError("checkpoint Map must not call the main model")


def _map_response(payload, *, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(payload), tool_calls=tool_calls or []
                )
            )
        ]
    )


def test_default_context_engine_remains_compressor():
    assert DEFAULT_CONFIG["context"]["engine"] == "compressor"


def test_checkpoint_engine_name_is_checkpoint():
    engine = load_context_engine("checkpoint")
    assert engine is not None
    assert engine.name == "checkpoint"


def test_discover_includes_checkpoint():
    names = [name for name, _desc, _available in discover_context_engines()]
    assert "checkpoint" in names


def test_shadow_compress_is_noop():
    engine = load_context_engine("checkpoint")
    assert engine is not None
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    original = deepcopy(messages)

    result = engine.compress(messages)

    assert result == original
    assert result is messages
    assert engine.compression_count == 0


def test_inflight_tool_call_refuses_checkpoint():
    engine = load_context_engine("checkpoint")
    assert engine is not None
    messages = [
        {"role": "user", "content": "change the file"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                }
            ],
        },
    ]
    original = deepcopy(messages)

    assert engine._has_inflight_tools(messages)
    result = engine.compress(messages)

    assert result == original
    assert result is messages
    assert engine.compression_count == 0


def test_stale_snapshot_refuses_checkpoint(monkeypatch):
    engine = load_context_engine("checkpoint")
    assert engine is not None
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "original request"},
    ]
    snapshot = engine._capture_snapshot(messages)

    assert not engine._snapshot_is_current(deepcopy(messages), snapshot)
    messages[-1]["content"] = "newer request"
    assert not engine._snapshot_is_current(messages, snapshot)

    revision_checked = False

    def stale_revision(_messages, _snapshot):
        nonlocal revision_checked
        revision_checked = True
        return False

    monkeypatch.setattr(engine, "_snapshot_is_current", stale_revision)

    result = engine.compress(messages)

    assert revision_checked
    assert result is messages
    assert engine.compression_count == 0


def test_causal_groups_keep_tool_call_and_results_together():
    engine = load_context_engine("checkpoint")
    assert engine is not None
    messages = [
        {"role": "user", "content": "change the file"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "run_terminal", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "written"},
        {"role": "tool", "tool_call_id": "call_2", "content": "passed"},
        {"role": "assistant", "content": "The change is ready."},
    ]

    groups = engine._plan_causal_groups(messages)

    assert [group.event_indices for group in groups] == [(0,), (1, 2, 3), (4,)]


def test_deterministic_lanes_keep_latest_user_turn_as_active_intent():
    engine = load_context_engine("checkpoint")
    assert engine is not None
    messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "Summary: old request was handled."},
        {"role": "user", "content": "rename the generated file"},
        {
            "role": "assistant",
            "content": "Summary: the user asked to rename the generated file.",
        },
    ]

    lanes = engine._extract_deterministic_lanes(messages)

    assert lanes.active_intent is not None
    assert lanes.active_intent.content == "rename the generated file"
    assert lanes.active_intent.event_indices == (2,)


def test_assistant_prose_does_not_mark_effect_succeeded():
    engine = load_context_engine("checkpoint")
    assert engine is not None
    messages = [
        {
            "role": "assistant",
            "content": "I will write the file.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file.txt"},
        {"role": "assistant", "content": "I wrote the file."},
    ]

    lanes = engine._extract_deterministic_lanes(messages)

    assert len(lanes.effects) == 1
    assert lanes.effects[0].operation == "write_file"
    assert lanes.effects[0].status == "unknown"


def test_typed_map_keeps_source_event_ids_and_uses_no_tools():
    from agent.checkpoint_engine import CheckpointContextEngine

    auxiliary = _FakeAuxiliaryClient(
        _map_response({"source_event_ids": [0], "facts": []}),
        _map_response(
            {
                "source_event_ids": [1],
                "facts": [
                    {
                        "kind": "request",
                        "text": "The user said hello.",
                        "source_event_ids": [1],
                    }
                ],
            }
        )
    )
    engine = CheckpointContextEngine(auxiliary_client=auxiliary)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]

    result = engine.compress(messages)

    assert result is messages
    assert all(call["tools"] == [] for call in auxiliary.calls)
    assert [shard.source_event_ids for shard in engine.last_map_shards] == [(0,), (1,)]
    assert engine.last_map_shards[1].facts[0].source_event_ids == (1,)


def test_invalid_or_truncated_map_json_rejects_candidate():
    from agent.checkpoint_engine import CheckpointContextEngine

    auxiliary = _FakeAuxiliaryClient(
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{", tool_calls=[]))]
        )
    )
    engine = CheckpointContextEngine(auxiliary_client=auxiliary)
    messages = [{"role": "user", "content": "hello"}]

    result = engine.compress(messages)

    assert result is messages
    assert auxiliary.calls
    assert engine.last_map_shards == ()
    assert "[digest unavailable]" not in str(result)
    assert engine.compression_count == 0


def test_map_tool_call_rejects_candidate():
    from agent.checkpoint_engine import CheckpointContextEngine

    auxiliary = _FakeAuxiliaryClient(
        _map_response(
            {"source_event_ids": [0], "facts": []},
            tool_calls=[{"id": "call_1"}],
        )
    )
    engine = CheckpointContextEngine(auxiliary_client=auxiliary)
    messages = [{"role": "user", "content": "hello"}]

    assert engine.compress(messages) is messages
    assert engine.last_map_shards == ()
    assert engine.compression_count == 0


def test_missing_map_coverage_rejects_candidate():
    from agent.checkpoint_engine import CheckpointContextEngine

    auxiliary = _FakeAuxiliaryClient(
        _map_response({"source_event_ids": [0], "facts": []}),
        _map_response({"source_event_ids": [0], "facts": []}),
    )
    engine = CheckpointContextEngine(auxiliary_client=auxiliary)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]

    assert engine.compress(messages) is messages
    assert engine.last_map_shards == ()
    assert engine.compression_count == 0


def test_exhausted_auxiliary_map_never_calls_main_model():
    from agent.checkpoint_engine import CheckpointContextEngine

    main_model = _FakeMainModel()
    auxiliary = _FakeAuxiliaryClient(RuntimeError("configured chain exhausted"))
    engine = CheckpointContextEngine(auxiliary_client=auxiliary)
    messages = [{"role": "user", "content": "hello"}]

    assert engine.compress(messages) is messages
    assert auxiliary.calls
    assert main_model.calls == 0
    assert engine.compression_count == 0


def test_checkpoint_map_concurrency_is_capped_at_two_by_default():
    from agent.checkpoint_engine import CheckpointContextEngine

    assert CheckpointContextEngine().map_concurrency == 2
