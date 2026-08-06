"""Restart-safety regressions for proactive tool-result pruning."""

from __future__ import annotations

import ast
import copy
import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_compressor import (
    _capture_durable_compaction_baseline,
    _estimate_msg_budget_tokens,
)
from hermes_state import (
    CompressionSessionBusyError,
    CompressionSessionClosedError,
    SessionDB,
)


_REARM_KEY = "_proactive_prune_rearm_tokens"


def test_all_runtime_archive_and_compact_calls_are_fenced() -> None:
    """Every production whole-session rewrite carries lease and snapshot guards."""
    repo_root = Path(__file__).resolve().parents[2]
    excluded_dirs = {
        ".git", ".venv", "venv", "build", "dist", "node_modules", "tests",
    }
    archive_calls: list[tuple[Path, ast.Call]] = []
    for path in repo_root.rglob("*.py"):
        if any(part in excluded_dirs for part in path.relative_to(repo_root).parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            is_archive_call = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "archive_and_compact"
            ) or (
                isinstance(node.func, ast.Name)
                and node.func.id == "archive_and_compact"
            )
            if is_archive_call:
                archive_calls.append((path.relative_to(repo_root), node))

    assert archive_calls, "production archive_and_compact caller audit found no calls"
    required = {
        "compression_lock_holder",
        "require_compression_lease",
        "expected_active_messages",
        "expected_active_row_ids",
        "expected_active_row_fingerprint",
    }
    for path, call in archive_calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert required <= keywords, (
            f"unfenced archive_and_compact call at {path}:{call.lineno}; "
            f"missing {sorted(required - keywords)}"
        )


def _assistant_call(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'},
        }],
    }


def _tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _history(*, large_chars: int = 24_000) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": "start"}]
    for index in range(8):
        call_id = f"call_{index}"
        messages.append(_assistant_call(call_id))
        content = chr(65 + index) * large_chars if index < 3 else "ok"
        messages.append(_tool_result(call_id, content))
    return messages


def _build_agent(db: SessionDB, session_id: str, *, platform: str = "telegram"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            platform=platform,
            skip_context_files=True,
            skip_memory=True,
        )


def _configure_pruning(agent) -> None:
    compressor = agent.context_compressor
    compressor.proactive_prune_tokens = 48_000
    compressor.proactive_prune_min_result_chars = 8_000
    compressor.proactive_prune_min_reclaim_tokens = 4_096
    compressor.protect_first_n = 2
    compressor.protect_last_n = 4


def _model_config(db: SessionDB, session_id: str) -> dict:
    raw = db.get_session(session_id)["model_config"]
    return json.loads(raw) if raw else {}


def test_gateway_eviction_reload_keeps_prune_and_durable_runway(tmp_path: Path) -> None:
    """A fresh gateway agent must reload both the pruned body and its runway."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "GATEWAY_PRUNE_RESTART"
    db.create_session(
        session_id, source="telegram", model_config={"keep": "value"},
    )
    db.append_messages_batch(session_id, _history())

    first_agent = _build_agent(db, session_id)
    _configure_pruning(first_agent)
    before = db.get_messages_as_conversation(session_id)
    with patch.object(
        db, "archive_and_compact", wraps=db.archive_and_compact,
    ) as archive:
        pruned, count = first_agent.context_compressor.prune_tool_results_only(
            before, current_tokens=120_000,
        )

    assert count >= 1
    archive.assert_called_once()
    assert archive.call_args.kwargs["require_compression_lease"] is True
    assert archive.call_args.kwargs["compression_lock_holder"]
    assert archive.call_args.kwargs["expected_active_messages"] == before
    assert db.get_compression_lock_holder(session_id) is None
    durable = db.get_messages_as_conversation(session_id)
    assert [message["content"] for message in durable] == [
        message["content"] for message in pruned
    ]
    assert len(durable[2]["content"]) < 24_000
    stored_runway = _model_config(db, session_id)[_REARM_KEY]
    assert _model_config(db, session_id)["keep"] == "value"
    assert stored_runway > sum(map(_estimate_msg_budget_tokens, durable))

    # Simulate gateway cache eviction / process restart: construct a wholly
    # new AIAgent and load the active transcript from SQLite.
    resumed_agent = _build_agent(db, session_id)
    _configure_pruning(resumed_agent)
    assert resumed_agent.context_compressor._proactive_prune_rearm_tokens == stored_runway
    reloaded = db.get_messages_as_conversation(session_id)
    archived_before = len(db.get_messages(session_id, include_inactive=True))
    result, second_count = resumed_agent.context_compressor.prune_tool_results_only(
        reloaded, current_tokens=1_000_000,
    )

    assert result is reloaded
    assert second_count == 0
    assert len(db.get_messages(session_id, include_inactive=True)) == archived_before

    db.archive_and_compact(
        session_id,
        reloaded,
        model_config_patch={_REARM_KEY: None},
    )
    cleared = _build_agent(db, session_id)
    _configure_pruning(cleared)
    assert cleared.context_compressor._proactive_prune_rearm_tokens == 0
    assert cleared.context_compressor._proactive_prune_runway_authoritative is True


def test_published_child_fences_stale_proactive_prune(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_id = "PRUNE_STALE_PARENT"
    child_id = "PRUNE_LIVE_CHILD"
    db.create_session(
        parent_id, source="telegram", model_config={"keep": "parent"},
    )
    db.append_messages_batch(parent_id, _history())
    stale = _build_agent(db, parent_id)
    _configure_pruning(stale)
    stale_messages = db.get_messages_as_conversation(parent_id)

    winner = "pid=1:tid=1:agent=winner:nonce=winner"
    assert db.try_acquire_compression_lock(parent_id, winner, ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=parent_id,
        child_session_id=child_id,
        source="telegram",
        messages=[{"role": "user", "content": "compressed handoff"}],
        model_config={"keep": "child"},
        compression_lock_holder=winner,
    )
    db.release_compression_lock(parent_id, winner)
    parent_contents = [
        message["content"] for message in db.get_messages_as_conversation(parent_id)
    ]
    parent_config = db.get_session(parent_id)["model_config"]
    rejected: list[Exception] = []
    real_archive = db.archive_and_compact

    def _record_rejection(*args, **kwargs):
        try:
            return real_archive(*args, **kwargs)
        except Exception as exc:
            rejected.append(exc)
            raise

    with patch.object(db, "archive_and_compact", side_effect=_record_rejection):
        result, count = stale.context_compressor.prune_tool_results_only(
            stale_messages, current_tokens=120_000,
        )

    assert result is stale_messages
    assert count == 0
    assert len(rejected) == 1
    assert isinstance(rejected[0], CompressionSessionClosedError)
    assert db.get_session(parent_id)["model_config"] == parent_config
    assert [
        message["content"] for message in db.get_messages_as_conversation(parent_id)
    ] == parent_contents
    assert [
        message["content"] for message in db.get_messages_as_conversation(child_id)
    ] == ["compressed handoff"]


def test_in_place_winner_fences_stale_proactive_prune(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_STALE_IN_PLACE"
    db.create_session(
        session_id, source="telegram", model_config={"keep": "original"},
    )
    db.append_messages_batch(session_id, _history())
    stale = _build_agent(db, session_id)
    _configure_pruning(stale)
    stale_messages = db.get_messages_as_conversation(session_id)

    winner_messages = [
        {"role": "user", "content": "winner summary"},
        {"role": "assistant", "content": "winner reply"},
    ]
    winner = "pid=1:tid=1:agent=winner:nonce=winner"
    assert db.try_acquire_compression_lock(session_id, winner, ttl_seconds=60)
    db.archive_and_compact(
        session_id,
        winner_messages,
        model_config_patch={_REARM_KEY: None},
        compression_lock_holder=winner,
        require_compression_lease=True,
        expected_active_messages=stale_messages,
    )
    db.release_compression_lock(session_id, winner)
    durable_before = db.get_messages_as_conversation(session_id)
    config_before = db.get_session(session_id)["model_config"]
    rows_before = len(db.get_messages(session_id, include_inactive=True))

    result, count = stale.context_compressor.prune_tool_results_only(
        stale_messages, current_tokens=120_000,
    )

    assert result is stale_messages
    assert count == 0
    assert db.get_messages_as_conversation(session_id) == durable_before
    assert db.get_session(session_id)["model_config"] == config_before
    assert len(db.get_messages(session_id, include_inactive=True)) == rows_before
    assert db.get_compression_lock_holder(session_id) is None


def test_stale_snapshot_rejection_rolls_back_archive_and_config(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_STALE_ROLLBACK"
    db.create_session(
        session_id, source="telegram", model_config={"keep": "original"},
    )
    db.append_messages_batch(session_id, _history())
    stale_messages = db.get_messages_as_conversation(session_id)
    db.archive_and_compact(
        session_id,
        [{"role": "user", "content": "winner summary"}],
        model_config_patch={"winner": True},
        expected_active_messages=stale_messages,
    )
    durable_before = db.get_messages_as_conversation(session_id)
    config_before = db.get_session(session_id)["model_config"]
    rows_before = db.get_messages(session_id, include_inactive=True)

    stale_holder = "pid=1:tid=1:agent=stale:nonce=stale"
    assert db.try_acquire_compression_lock(session_id, stale_holder, ttl_seconds=60)
    try:
        with pytest.raises(
            CompressionSessionBusyError,
            match="transcript changed before compaction",
        ):
            db.archive_and_compact(
                session_id,
                stale_messages,
                model_config_patch={"winner": False, "stale": True},
                compression_lock_holder=stale_holder,
                require_compression_lease=True,
                expected_active_messages=stale_messages,
            )
    finally:
        db.release_compression_lock(session_id, stale_holder)

    assert db.get_messages_as_conversation(session_id) == durable_before
    assert db.get_session(session_id)["model_config"] == config_before
    assert db.get_messages(session_id, include_inactive=True) == rows_before


def _raw_compaction_baseline(
    db: SessionDB,
    session_id: str,
    live_messages: list[dict],
) -> tuple[list[dict], list[int], list[tuple]]:
    baseline, row_ids, row_fingerprint = _capture_durable_compaction_baseline(
        db, session_id, live_messages,
    )
    assert baseline is not None
    assert row_ids is not None
    assert row_fingerprint is not None
    return baseline, row_ids, row_fingerprint


def test_repaired_restore_uses_raw_rows_for_exact_compaction_fence(
    tmp_path: Path,
) -> None:
    """A legal restore repair must not weaken the raw transactional fence."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "REPAIRED_RESTORE_BASELINE"
    db.create_session(session_id, source="tui")
    db.append_messages_batch(session_id, [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply"},
    ])

    live = db.get_messages_as_conversation(session_id)
    repair_message_sequence(None, live)
    assert [message["role"] for message in live] == ["user", "assistant"]
    baseline, row_ids, row_fingerprint = _raw_compaction_baseline(
        db, session_id, live,
    )
    assert len(baseline) == 3

    replacement = [{"role": "user", "content": "summary"}]
    db.archive_and_compact(
        session_id,
        replacement,
        expected_active_messages=baseline,
        expected_active_row_ids=row_ids,
        expected_active_row_fingerprint=row_fingerprint,
    )
    assert [
        (message["role"], message["content"])
        for message in db.get_messages_as_conversation(session_id)
    ] == [("user", "summary")]


def test_persisted_text_and_live_multimodal_projection_share_raw_fence(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "MULTIMODAL_RESTORE_BASELINE"
    db.create_session(session_id, source="tui")
    db.append_messages_batch(session_id, [{
        "role": "user",
        "content": "describe\n[screenshot]",
    }])
    live = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
        ],
        "_db_persisted": True,
    }]

    baseline, row_ids, row_fingerprint = _raw_compaction_baseline(
        db, session_id, live,
    )
    replacement = [{"role": "user", "content": "summary"}]
    db.archive_and_compact(
        session_id,
        replacement,
        expected_active_messages=baseline,
        expected_active_row_ids=row_ids,
        expected_active_row_fingerprint=row_fingerprint,
    )
    assert [
        (message["role"], message["content"])
        for message in db.get_messages_as_conversation(session_id)
    ] == [("user", "summary")]


def test_compaction_baseline_accepts_only_clean_or_hot_api_content_view(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "API_CONTENT_RESTORE_VIEWS"
    db.create_session(session_id, source="tui")
    db.append_messages_batch(session_id, [{
        "role": "user",
        "content": "clean",
        "api_content": "wire",
    }])
    cold = db.get_messages_as_conversation(session_id)

    assert _capture_durable_compaction_baseline(db, session_id, cold)[0] is not None

    hot = copy.deepcopy(cold)
    hot[0]["content"] = hot[0].pop("api_content")
    assert _capture_durable_compaction_baseline(db, session_id, hot)[0] is not None

    mutated = copy.deepcopy(cold)
    mutated[0]["content"] = "MUTATED"
    with pytest.raises(
        CompressionSessionBusyError,
        match="not a legal restore",
    ):
        _capture_durable_compaction_baseline(db, session_id, mutated)


def test_compaction_baseline_rejects_tool_content_whitespace_mutation(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "TOOL_CONTENT_BYTES"
    db.create_session(session_id, source="tui")
    db.append_messages_batch(session_id, [{
        "role": "tool",
        "content": "result",
        "tool_call_id": "call-1",
    }])
    live = db.get_messages_as_conversation(session_id)
    live[0]["content"] = " result "

    with pytest.raises(
        CompressionSessionBusyError,
        match="not a legal restore",
    ):
        _capture_durable_compaction_baseline(db, session_id, live)


def test_compaction_baseline_never_substitutes_tool_api_content(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "TOOL_API_CONTENT_MASK"
    db.create_session(session_id, source="tui")
    db.append_messages_batch(session_id, [{
        "role": "tool",
        "content": "clean",
        "api_content": "mask",
        "tool_call_id": "call-1",
    }])
    live = db.get_messages_as_conversation(session_id)
    live[0]["content"] = "mask"

    with pytest.raises(
        CompressionSessionBusyError,
        match="not a legal restore",
    ):
        _capture_durable_compaction_baseline(db, session_id, live)


def test_raw_row_fence_rejects_append_without_mutating_caller(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "APPEND_RACE_REJECTED"
    db.create_session(session_id, source="tui")
    db.append_messages_batch(session_id, [{"role": "user", "content": "before"}])
    live = db.get_messages_as_conversation(session_id)
    baseline, row_ids, row_fingerprint = _raw_compaction_baseline(
        db, session_id, live,
    )
    db.append_messages_batch(session_id, [{"role": "assistant", "content": "later"}])
    replacement = [{"role": "user", "content": "summary"}]
    before = copy.deepcopy(replacement)

    with pytest.raises(
        CompressionSessionBusyError,
        match="transcript changed before compaction",
    ):
        db.archive_and_compact(
            session_id,
            replacement,
            expected_active_messages=baseline,
            expected_active_row_ids=row_ids,
            expected_active_row_fingerprint=row_fingerprint,
        )

    assert replacement == before
    assert [message["content"] for message in db.get_messages_as_conversation(session_id)] == [
        "before", "later",
    ]


def test_raw_row_fence_rejects_same_row_sanitize_equivalent_update(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "RAW_FENCE_WHITESPACE_UPDATE"
    db.create_session(session_id, source="tui")
    db.append_messages_batch(
        session_id, [{"role": "user", "content": "original"}],
    )
    live = db.get_messages_as_conversation(session_id)
    baseline, row_ids, row_fingerprint = _raw_compaction_baseline(
        db, session_id, live,
    )
    db._conn.execute(
        "UPDATE messages SET content = ? WHERE id = ?",
        (" original ", row_ids[0]),
    )
    db._conn.commit()
    durable_before = db.get_messages(session_id, include_inactive=True)

    with pytest.raises(
        CompressionSessionBusyError,
        match="transcript changed before compaction",
    ):
        db.archive_and_compact(
            session_id,
            [{"role": "user", "content": "summary"}],
            expected_active_messages=baseline,
            expected_active_row_ids=row_ids,
            expected_active_row_fingerprint=row_fingerprint,
        )

    assert db.get_messages(session_id, include_inactive=True) == durable_before


@pytest.mark.parametrize("mutation", ["rewrite", "delete", "replace"])
def test_raw_row_fence_rejects_durable_rewrite_delete_or_reorder(
    tmp_path: Path,
    mutation: str,
) -> None:
    db = SessionDB(db_path=tmp_path / f"{mutation}.db")
    session_id = f"RAW_FENCE_{mutation.upper()}"
    db.create_session(session_id, source="tui")
    original = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    db.append_messages_batch(session_id, original)
    live = db.get_messages_as_conversation(session_id)
    baseline, row_ids, row_fingerprint = _raw_compaction_baseline(
        db, session_id, live,
    )

    if mutation == "rewrite":
        db._conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("changed", row_ids[0]),
        )
        db._conn.commit()
    elif mutation == "delete":
        db._conn.execute(
            "UPDATE messages SET active = 0 WHERE id = ?",
            (row_ids[0],),
        )
        db._conn.commit()
    else:
        db.archive_and_compact(
            session_id,
            list(reversed(original)),
            expected_active_messages=baseline,
        )

    durable_before = db.get_messages(session_id, include_inactive=True)
    replacement = [{"role": "user", "content": "summary"}]
    with pytest.raises(
        CompressionSessionBusyError,
        match="transcript changed before compaction",
    ):
        db.archive_and_compact(
            session_id,
            replacement,
            expected_active_messages=baseline,
            expected_active_row_ids=row_ids,
            expected_active_row_fingerprint=row_fingerprint,
        )

    assert db.get_messages(session_id, include_inactive=True) == durable_before


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("api_content", "changed wire content"),
        ("display_kind", "changed-kind"),
        ("display_metadata", {"changed": True}),
        ("tool_call_id", "changed-call"),
        ("tool_name", "changed-tool"),
        ("effect_disposition", "changed-effect"),
        ("tool_calls", [{"id": "changed", "type": "function"}]),
        ("finish_reason", "length"),
        ("reasoning", "changed reasoning"),
        ("reasoning_content", "changed reasoning content"),
        ("reasoning_details", [{"changed": True}]),
        ("codex_reasoning_items", [{"changed": True}]),
        ("codex_message_items", [{"changed": True}]),
        ("observed", False),
        ("message_id", "changed-message-id"),
    ],
)
def test_snapshot_fence_compares_every_durable_replay_field(
    tmp_path: Path, field: str, changed_value,
) -> None:
    db = SessionDB(db_path=tmp_path / f"{field}.db")
    session_id = f"REPLAY_FIELD_{field}"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, [{
        "role": "assistant",
        "content": "durable content",
        "api_content": "wire content",
        "display_kind": "status",
        "display_metadata": {"phase": "done"},
        "tool_call_id": "call-1",
        "tool_name": "terminal",
        "effect_disposition": "committed",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "terminal", "arguments": "{}"},
        }],
        "finish_reason": "tool_calls",
        "reasoning": "reasoning",
        "reasoning_content": "reasoning content",
        "reasoning_details": [{"type": "summary_text", "text": "detail"}],
        "codex_reasoning_items": [{"type": "reasoning", "id": "r1"}],
        "codex_message_items": [{"type": "message", "id": "m1"}],
        "observed": True,
        "message_id": "platform-message-id",
    }])
    durable = db.get_messages_as_conversation(session_id)
    stale = copy.deepcopy(durable)
    stale[0][field] = changed_value
    rows_before = db.get_messages(session_id, include_inactive=True)

    with pytest.raises(
        CompressionSessionBusyError,
        match="transcript changed before compaction",
    ):
        db.archive_and_compact(
            session_id,
            [{"role": "assistant", "content": "replacement"}],
            expected_active_messages=stale,
        )

    assert db.get_messages(session_id, include_inactive=True) == rows_before


def test_fresh_agent_rearms_after_durable_history_regrowth_once(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_DURABLE_REGROWTH"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    first_agent = _build_agent(db, session_id)
    _configure_pruning(first_agent)
    first, first_count = first_agent.context_compressor.prune_tool_results_only(
        db.get_messages_as_conversation(session_id), current_tokens=120_000,
    )
    assert first_count >= 1
    first_runway = _model_config(db, session_id)[_REARM_KEY]

    growth = [
        _assistant_call("regrown_large"),
        _tool_result("regrown_large", "z" * 240_000),
        _assistant_call("tail_1"),
        _tool_result("tail_1", "ok"),
        _assistant_call("tail_2"),
        _tool_result("tail_2", "ok"),
    ]
    db.append_messages_batch(session_id, growth)

    resumed = _build_agent(db, session_id)
    _configure_pruning(resumed)
    grown = db.get_messages_as_conversation(session_id)
    assert sum(map(_estimate_msg_budget_tokens, grown)) >= first_runway
    second, second_count = resumed.context_compressor.prune_tool_results_only(
        grown, current_tokens=1_000_000,
    )

    assert second_count >= 1
    second_runway = _model_config(db, session_id)[_REARM_KEY]
    assert second_runway > first_runway

    restarted = _build_agent(db, session_id)
    _configure_pruning(restarted)
    durable = db.get_messages_as_conversation(session_id)
    result, third_count = restarted.context_compressor.prune_tool_results_only(
        durable, current_tokens=1_000_000,
    )
    assert result is durable
    assert third_count == 0
    assert restarted.context_compressor._proactive_prune_rearm_tokens == second_runway


def test_prune_persistence_failure_is_a_noop(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_PERSISTENCE_FAILURE"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    agent = _build_agent(db, session_id)
    _configure_pruning(agent)
    messages = db.get_messages_as_conversation(session_id)
    original_contents = [message["content"] for message in messages]

    with patch.object(
        db, "archive_and_compact", side_effect=RuntimeError("disk full"),
    ):
        result, count = agent.context_compressor.prune_tool_results_only(
            messages, current_tokens=120_000,
        )

    assert result is messages
    assert count == 0
    assert agent.context_compressor._proactive_prune_rearm_tokens == 0
    assert [message["content"] for message in messages] == original_contents
    assert [message["content"] for message in db.get_messages_as_conversation(session_id)] == original_contents
    assert _REARM_KEY not in _model_config(db, session_id)


@pytest.mark.parametrize("failure", ["malformed", "read_error"])
def test_unknown_durable_runway_blocks_until_authoritative_reload(
    tmp_path: Path, failure: str,
) -> None:
    db = SessionDB(db_path=tmp_path / f"{failure}.db")
    session_id = f"PRUNE_UNKNOWN_{failure.upper()}"
    db.create_session(
        session_id, source="telegram", model_config={"keep": "value"},
    )
    db.append_messages_batch(session_id, _history())
    agent = _build_agent(db, session_id)
    _configure_pruning(agent)

    if failure == "malformed":
        db._conn.execute(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            ("{malformed", session_id),
        )
        db._conn.commit()
        agent.context_compressor.bind_session_state(db, session_id)
    else:
        with patch.object(
            db, "get_session", side_effect=sqlite3.OperationalError("read failed"),
        ):
            agent.context_compressor.bind_session_state(db, session_id)

    assert agent.context_compressor._proactive_prune_runway_authoritative is False
    messages = db.get_messages_as_conversation(session_id)
    original_contents = [message["content"] for message in messages]
    original_config = db.get_session(session_id)["model_config"]
    with patch.object(
        db, "archive_and_compact", wraps=db.archive_and_compact,
    ) as archive:
        result, count = agent.context_compressor.prune_tool_results_only(
            messages, current_tokens=120_000,
        )

    assert result is messages
    assert count == 0
    archive.assert_not_called()
    assert db.get_session(session_id)["model_config"] == original_config
    assert [
        message["content"] for message in db.get_messages_as_conversation(session_id)
    ] == original_contents

    db._conn.execute(
        "UPDATE sessions SET model_config = ? WHERE id = ?",
        (json.dumps({"keep": "value"}), session_id),
    )
    db._conn.commit()
    agent.context_compressor.bind_session_state(db, session_id)
    assert agent.context_compressor._proactive_prune_runway_authoritative is True
    recovered, recovered_count = agent.context_compressor.prune_tool_results_only(
        messages, current_tokens=120_000,
    )
    assert recovered_count >= 1
    assert recovered is not messages
    assert _model_config(db, session_id)[_REARM_KEY] > 0


def test_archive_model_config_patch_rolls_back_with_transcript(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_ATOMIC_ARCHIVE_FAILURE"
    db.create_session(
        session_id,
        source="telegram",
        model_config={"keep": "value", _REARM_KEY: 120_000},
    )
    original = [{"role": "user", "content": "original"}]
    db.append_messages_batch(session_id, original)

    with patch.object(
        db, "_insert_message_rows", side_effect=RuntimeError("insert failed"),
    ):
        with pytest.raises(RuntimeError, match="insert failed"):
            db.archive_and_compact(
                session_id,
                [{"role": "user", "content": "replacement"}],
                model_config_patch={_REARM_KEY: None},
            )

    assert db.get_messages_as_conversation(session_id)[0]["content"] == "original"
    assert _model_config(db, session_id) == {"keep": "value", _REARM_KEY: 120_000}
