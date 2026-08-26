"""Checkpoint ContextEngine: opt-in shadow no-op (DESIGN.md §10 item 1)."""

from copy import deepcopy
import json
from types import SimpleNamespace

from agent.context_engine import sanitize_memory_context
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


class _EchoMapClient:
    """Return a valid Map record for whichever causal group is requested."""

    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["messages"][-1]["content"])
        return _map_response({"source_event_ids": payload["source_event_ids"], "facts": []})


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


def test_trigger_tracks_host_thresholds_and_provider_usage():
    from agent.checkpoint_engine import CheckpointContextEngine

    engine = CheckpointContextEngine()
    engine.threshold_percent = 0.50
    engine.model_thresholds = {"narrow": 0.60}

    engine.update_model(model="narrow-model", context_length=1_000)
    engine.update_from_response(
        {
            "prompt_tokens": 600,
            "completion_tokens": 25,
            "total_tokens": 625,
            "input_tokens": 400,
            "output_tokens": 25,
            "cache_read_tokens": 200,
            "cache_write_tokens": 10,
            "reasoning_tokens": 5,
        }
    )

    assert engine.context_length == 1_000
    assert engine.threshold_percent == 0.60
    assert engine.threshold_tokens == 600
    assert engine.last_prompt_tokens == 600
    assert engine.last_input_tokens == 400
    assert engine.last_cache_read_tokens == 200
    assert engine.should_compress() is True
    assert engine.should_compress(599) is False
    assert engine.last_trigger_reason == "below_threshold"


def test_auto_trigger_cooldown_and_force_retry_after_failed_checkpoint():
    from agent.checkpoint_engine import CheckpointContextEngine

    auxiliary = _FakeAuxiliaryClient(RuntimeError("temporary failure"), RuntimeError("retry"))
    engine = CheckpointContextEngine(auxiliary_client=auxiliary)
    engine.update_model(model="test", context_length=100)
    messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active request"},
    ]

    assert engine.should_compress(engine.threshold_tokens) is True
    assert engine.compress(messages, current_tokens=engine.threshold_tokens) is messages
    should_compress, reason = engine.should_compress_info(engine.threshold_tokens)
    assert should_compress is False
    assert reason is not None and reason.startswith("cooldown:")

    attempts_before_force = len(auxiliary.calls)
    assert engine.compress(messages, current_tokens=engine.threshold_tokens, force=True) is messages
    assert len(auxiliary.calls) > attempts_before_force


def test_checkpoint_skips_unreclaimable_or_inflight_history_with_block_reason():
    from agent.checkpoint_engine import CheckpointContextEngine

    engine = CheckpointContextEngine(auxiliary_client=_EchoMapClient())
    engine.update_model(model="test", context_length=100)
    unreclaimable = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "active request"},
    ]

    assert engine.has_content_to_compress(unreclaimable) is False
    assert engine.compress(unreclaimable, current_tokens=engine.threshold_tokens) is unreclaimable
    assert engine.should_compress_info(engine.threshold_tokens) == (
        False,
        "cooldown:nothing_reclaimable",
    )

    inflight = [
        {"role": "user", "content": "active request"},
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
    assert engine.compress(inflight, current_tokens=engine.threshold_tokens, force=True) is inflight
    assert engine.last_trigger_reason == "in_flight_tools"


def test_checkpoint_uses_focus_memory_and_complete_request_estimate():
    from agent.checkpoint_engine import CheckpointContextEngine

    memory_context = "api_key=sk-test-secret\nRemember the release checklist."
    auxiliary = _FakeAuxiliaryClient(
        _map_response({"source_event_ids": [0, 1, 2, 3], "facts": []}),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="Continue from the verified state.", tool_calls=[]
            ))]
        ),
    )
    engine = CheckpointContextEngine(auxiliary_client=auxiliary)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "finish the release"},
    ]

    assert engine.compress(
        messages,
        current_tokens=900,
        focus_topic="release validation",
        memory_context=memory_context,
    ) is messages

    semantic_payload = json.loads(auxiliary.calls[-1]["messages"][-1]["content"])
    assert engine.last_request_tokens == 900
    assert semantic_payload["focus_topic"] == "release validation"
    assert semantic_payload["memory_context"] == sanitize_memory_context(memory_context)


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
        _map_response(
            {
                "source_event_ids": [0, 1],
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
    assert [shard.source_event_ids for shard in engine.last_map_shards] == [(0, 1)]
    assert engine.last_map_shards[0].facts[0].source_event_ids == (1,)


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
    engine = CheckpointContextEngine(
        auxiliary_client=auxiliary,
        main_model=main_model,
    )
    messages = [{"role": "user", "content": "hello"}]

    result = engine.compress(messages)

    assert result is messages
    assert engine._main_model is main_model
    assert auxiliary.calls
    assert main_model.calls == 0
    assert engine.last_map_shards == ()
    assert engine.compression_count == 0
    assert "[digest unavailable]" not in str(result)


def test_checkpoint_map_concurrency_is_capped_at_two_by_default():
    from agent.checkpoint_engine import CheckpointContextEngine

    assert CheckpointContextEngine().map_concurrency == 2


def test_map_planner_packs_consecutive_causal_units_without_splitting_tool_receipts(
    monkeypatch,
):
    import agent.checkpoint_engine as checkpoint_engine
    from agent.checkpoint_engine import CheckpointContextEngine

    monkeypatch.setattr(checkpoint_engine, "_MAP_SHARD_TARGET_INPUT_TOKENS", 12)
    monkeypatch.setattr(checkpoint_engine, "_MAP_SHARD_MAX_INPUT_TOKENS", 16)

    def token_counter(prompt):
        payload = json.loads(prompt[-1]["content"])
        return 6 * len(payload["source_event_ids"])

    engine = CheckpointContextEngine(token_counter=token_counter)
    messages = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "first response"},
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
        {"role": "tool", "tool_call_id": "call_1", "content": "written"},
        {"role": "assistant", "content": "verified"},
    ]

    shards = engine._plan_map_shards(messages, engine._plan_causal_groups(messages))

    assert [shard.event_indices for shard in shards] == [(0, 1), (2,), (3, 4), (5,)]


def test_map_planner_externalizes_an_oversized_causal_unit_without_tearing_it(
    monkeypatch,
):
    import agent.checkpoint_engine as checkpoint_engine
    from agent.checkpoint_engine import CheckpointContextEngine

    monkeypatch.setattr(checkpoint_engine, "_MAP_SHARD_TARGET_INPUT_TOKENS", 12)
    monkeypatch.setattr(checkpoint_engine, "_MAP_SHARD_MAX_INPUT_TOKENS", 16)

    def token_counter(prompt):
        payload = json.loads(prompt[-1]["content"])
        return 20 if payload["source_event_ids"] == [0, 1] else 6

    engine = CheckpointContextEngine(token_counter=token_counter)
    messages = [
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
        {"role": "tool", "tool_call_id": "call_1", "content": "receipt"},
        {"role": "user", "content": "continue"},
    ]

    shards = engine._plan_map_shards(messages, engine._plan_causal_groups(messages))

    assert [shard.event_indices for shard in shards] == [(2,)]
    assert [group.event_indices for group in engine.last_map_externalized_groups] == [(0, 1)]


def test_map_planner_fails_closed_when_its_total_input_budget_is_exceeded(monkeypatch):
    import agent.checkpoint_engine as checkpoint_engine
    from agent.checkpoint_engine import CheckpointContextEngine

    monkeypatch.setattr(checkpoint_engine, "_MAP_SHARD_TARGET_INPUT_TOKENS", 6)
    monkeypatch.setattr(checkpoint_engine, "_MAP_TOTAL_INPUT_TOKENS", 11)

    def token_counter(prompt):
        return 6 * len(json.loads(prompt[-1]["content"])["source_event_ids"])

    engine = CheckpointContextEngine(token_counter=token_counter)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first response"},
    ]

    assert engine._plan_map_shards(messages, engine._plan_causal_groups(messages)) is None


def test_map_planner_fails_closed_when_its_total_output_budget_is_exceeded(monkeypatch):
    import agent.checkpoint_engine as checkpoint_engine
    from agent.checkpoint_engine import CheckpointContextEngine

    monkeypatch.setattr(checkpoint_engine, "_MAP_SHARD_TARGET_INPUT_TOKENS", 6)
    monkeypatch.setattr(checkpoint_engine, "_MAP_TOTAL_OUTPUT_TOKENS", 1_500)

    def token_counter(prompt):
        return 6 * len(json.loads(prompt[-1]["content"])["source_event_ids"])

    engine = CheckpointContextEngine(token_counter=token_counter)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first response"},
    ]

    assert engine._plan_map_shards(messages, engine._plan_causal_groups(messages)) is None


def test_map_planner_has_a_bounded_default_shard_cap():
    from agent.checkpoint_engine import CheckpointContextEngine

    assert DEFAULT_CONFIG["checkpoint"]["max_map_shards"] == 32
    assert CheckpointContextEngine().max_map_shards == 32
    assert CheckpointContextEngine(max_map_shards=33).max_map_shards == 32


def test_map_token_estimate_uses_host_request_estimator_and_safe_fallback(monkeypatch):
    import agent.model_metadata as model_metadata
    from agent.checkpoint_engine import CheckpointContextEngine

    monkeypatch.setattr(model_metadata, "estimate_request_tokens_rough", lambda *_args, **_kwargs: 91)
    engine = CheckpointContextEngine()
    messages = [{"role": "user", "content": "中文" * 40}]
    group = engine._plan_causal_groups(messages)[0]

    assert engine._map_input_tokens(messages, group) >= 91


def test_token_fallback_is_conservative_for_cjk_and_json(monkeypatch):
    import agent.model_metadata as model_metadata
    from agent.checkpoint_engine import CheckpointContextEngine

    monkeypatch.setattr(model_metadata, "estimate_tokens_rough", lambda _value: 1)
    engine = CheckpointContextEngine()
    cjk = "中文" * 40
    structured = {"content": '{"code":"def f(): return {\"ok\": true}"}'}

    assert engine._rough_token_count(cjk) >= len(cjk)
    assert engine._rough_token_count(structured) >= len(json.dumps(structured)) // 2

def test_deterministic_reduce_merges_identity_supersession_and_action_state():
    from agent.checkpoint_engine import (
        ActiveIntent,
        CheckpointContextEngine,
        DeterministicLanes,
        Effect,
        MapFact,
        MapShard,
    )

    lanes = DeterministicLanes(
        ActiveIntent("finish the migration", (5,)),
        (Effect("call_1", "write_file", "unknown", (3, 4)),),
    )
    shards = (
        MapShard(
            (0, 1, 2),
            (
                MapFact("identity", "old project name", (0,), identity="project:old"),
                MapFact(
                    "identity",
                    "current project name",
                    (1,),
                    identity="project:current",
                    supersedes=("project:old",),
                ),
                MapFact(
                    "action",
                    "review migration",
                    (2,),
                    identity="review:migration",
                    action_state="planned",
                ),
            ),
        ),
        MapShard(
            (3, 4),
            (
                MapFact(
                    "action",
                    "write migration",
                    (3,),
                    identity="write:migration",
                    action_state="issued",
                ),
                MapFact(
                    "action",
                    "migration written",
                    (4,),
                    identity="write:migration",
                    action_state="succeeded",
                ),
            ),
        ),
    )

    reduced = CheckpointContextEngine()._reduce(lanes, shards)

    facts = {fact.identity: fact for fact in reduced.facts}
    assert set(facts) == {
        "project:current",
        "review:migration",
        "write:migration",
    }
    assert facts["write:migration"].action_state == "succeeded"
    assert [fact.identity for fact in reduced.plans] == ["review:migration"]
    assert reduced.effects == lanes.effects


def test_semantic_reduce_builds_a_shadow_candidate_without_a_new_user_turn():
    from agent.checkpoint_engine import CheckpointContextEngine

    reduced_states = []
    engine = CheckpointContextEngine(
        auxiliary_client=_FakeAuxiliaryClient(_map_response({"source_event_ids": [0], "facts": []})),
        semantic_reducer=lambda state: reduced_states.append(state) or "Continue from the verified state.",
        mode="shadow",
        token_counter=lambda _value: 1,
        output_reserve_tokens=0,
    )
    messages = [{"role": "user", "content": "finish the migration"}]

    result = engine.compress(messages)

    assert result is messages
    assert reduced_states
    assert engine.last_candidate is not None
    assert all(
        message["role"] != "user" or message["content"] != "Continue from the verified state."
        for message in engine.last_candidate
    )
    assert engine.compression_count == 0


def test_invalid_semantic_reduce_output_rejects_candidate():
    from agent.checkpoint_engine import CheckpointContextEngine

    engine = CheckpointContextEngine(
        auxiliary_client=_EchoMapClient(),
        semantic_reducer=lambda _state: {"not": "checkpoint text"},
    )
    messages = [{"role": "user", "content": "finish the migration"}]

    assert engine.compress(messages) is messages
    assert engine.last_candidate is None
    assert engine.compression_count == 0


def test_full_wire_hard_cap_rejects_before_live_commit():
    from agent.checkpoint_engine import CheckpointContextEngine

    counted = []
    tool_schema_stub = {"name": "write_file", "parameters": {}}

    def token_counter(value):
        counted.append(value)
        return 1

    engine = CheckpointContextEngine(
        auxiliary_client=_FakeAuxiliaryClient(_map_response({"source_event_ids": [0], "facts": []})),
        semantic_reducer=lambda _state: "Continue from the verified state.",
        mode="live",
        token_counter=token_counter,
        tool_schemas=tool_schema_stub,
        output_reserve_tokens=1,
        target_wire_tokens=3,
        hard_max_wire_tokens=3,
    )
    messages = [{"role": "user", "content": "finish the migration"}]

    assert engine.compress(messages) is messages
    assert tool_schema_stub in counted
    assert engine.compression_count == 0


def test_live_mode_commits_only_a_valid_under_cap_projection():
    from agent.checkpoint_engine import CheckpointContextEngine

    engine = CheckpointContextEngine(
        auxiliary_client=_EchoMapClient(),
        semantic_reducer=lambda _state: "Continue from the verified state.",
        mode="live",
        token_counter=lambda _value: 1,
        tool_schemas={"name": "write_file", "parameters": {}},
        output_reserve_tokens=1,
        target_wire_tokens=6,
        hard_max_wire_tokens=6,
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "finish the migration"},
    ]

    result = engine.compress(messages)

    assert result is not messages
    assert result[-1] == messages[-1]
    assert any(message["content"] == "Continue from the verified state." for message in result)
    assert engine.compression_count == 1


def test_renderer_shrinks_only_complete_old_causal_groups_before_rejecting():
    from agent.checkpoint_engine import CheckpointContextEngine

    engine = CheckpointContextEngine(
        auxiliary_client=_EchoMapClient(),
        semantic_reducer=lambda _state: "Continue from the verified state.",
        mode="live",
        token_counter=lambda _value: 1,
        output_reserve_tokens=0,
        target_wire_tokens=4,
        hard_max_wire_tokens=4,
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old request"},
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
        {"role": "tool", "tool_call_id": "call_1", "content": "written"},
        {"role": "user", "content": "active request"},
    ]

    result = engine.compress(messages)

    assert result is not messages
    assert result[-1] == messages[-1]
    assert all(message.get("content") not in {"old request", "written"} for message in result)
    assert not any(message.get("tool_calls") for message in result)
    assert engine.last_degradation_steps == (
        "completed",
        "tool_bodies",
        "decisions",
        "tail",
        "tail",
    )
