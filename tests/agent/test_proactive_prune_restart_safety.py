"""Restart-safety regressions for proactive tool-result pruning."""

from __future__ import annotations

import copy
import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_compressor import (
    _estimate_msg_budget_tokens,
)
from hermes_state import (
    CompressionSessionBusyError,
    CompressionSessionClosedError,
    SessionDB,
)


_REARM_KEY = "_proactive_prune_rearm_tokens"


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


def test_lazy_session_create_reloads_authoritative_runway(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_LAZY_CREATE"
    agent = _build_agent(db, session_id)

    assert db.get_session(session_id) is None
    assert agent.context_compressor._proactive_prune_runway_authoritative is False

    agent._ensure_db_session()

    assert db.get_session(session_id) is not None
    assert agent.context_compressor._proactive_prune_runway_authoritative is True
    assert agent.context_compressor._proactive_prune_rearm_tokens == 0


def test_lazy_session_create_read_error_stays_unknown_and_does_not_prune(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_LAZY_CREATE_READ_ERROR"
    agent = _build_agent(db, session_id)
    _configure_pruning(agent)

    with patch.object(
        db, "get_session", side_effect=sqlite3.OperationalError("read failed"),
    ):
        agent._ensure_db_session()

    assert db.get_session(session_id) is not None
    assert agent.context_compressor._proactive_prune_runway_authoritative is False
    messages = _history()
    with patch.object(
        db, "archive_and_compact", wraps=db.archive_and_compact,
    ) as archive:
        result, count = agent.context_compressor.prune_tool_results_only(
            messages, current_tokens=120_000,
        )

    assert result is messages
    assert count == 0
    archive.assert_not_called()


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
    before_fingerprint = db.get_compaction_fingerprint(session_id)
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
    assert archive.call_args.kwargs["expected_active_fingerprint"] == (
        before_fingerprint
    )
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

    assert result is not reloaded
    assert [message["content"] for message in result] == [
        message["content"] for message in reloaded
    ]
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


def test_proactive_prune_does_not_publish_pending_completion_suffix(
    tmp_path: Path,
) -> None:
    """A whole-session prune keeps pending tool work live-only until finalization."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    session_id = "PRUNE_PENDING_COMPLETION"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    agent = _build_agent(db, session_id)
    _configure_pruning(agent)
    settled = db.get_messages_as_conversation(session_id)
    pending = [
        *settled,
        {
            "role": "user",
            "content": "completion payload",
            "_completion_delivery_synthetic": True,
        },
        _assistant_call("pending_completion"),
        _tool_result("pending_completion", "pending completion tool result"),
    ]

    pruned, count = agent.context_compressor.prune_tool_results_only(
        pending, current_tokens=120_000,
    )

    assert count >= 1
    assert pruned[-1]["content"] == "pending completion tool result"
    assert not pruned[-1].get("_db_persisted")
    db.close()

    resumed_db = SessionDB(db_path=db_path)
    try:
        resumed = resumed_db.get_messages_as_conversation(session_id)
        assert "completion payload" not in [row.get("content") for row in resumed]
        assert "pending completion tool result" not in [
            row.get("content") for row in resumed
        ]
        assert [row.get("content") for row in resumed] == [
            row.get("content") for row in pruned[: len(settled)]
        ]
    finally:
        resumed_db.close()


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

    assert result is not stale_messages
    assert [message["content"] for message in result] == parent_contents
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


def test_stale_snapshot_rejection_rolls_back_archive_and_config(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_STALE_ROLLBACK"
    db.create_session(
        session_id, source="telegram", model_config={"keep": "original"},
    )
    db.append_messages_batch(session_id, _history())
    stale_messages = db.get_messages_as_conversation(session_id)
    stale_fingerprint = db.get_compaction_fingerprint(session_id)
    db.archive_and_compact(
        session_id,
        [{"role": "user", "content": "winner summary"}],
        model_config_patch={"winner": True},
        expected_active_fingerprint=stale_fingerprint,
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
                expected_active_fingerprint=stale_fingerprint,
            )
    finally:
        db.release_compression_lock(session_id, stale_holder)

    assert db.get_messages_as_conversation(session_id) == durable_before
    assert db.get_session(session_id)["model_config"] == config_before
    assert db.get_messages(session_id, include_inactive=True) == rows_before


def test_in_place_winner_fences_stale_proactive_prune_input(tmp_path: Path) -> None:
    """A stale caller must adopt the durable winner before pruning.

    The lease only serializes writers.  A loser can acquire it after the
    winner releases it, so capturing the winner's fingerprint and then
    pruning the loser's stale in-memory list is still an overwrite unless the
    computation input is reloaded from the durable active view.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_STALE_IN_PLACE_WINNER"
    db.create_session(session_id, source="tui")
    db.append_messages_batch(session_id, _history())
    stale_agent = _build_agent(db, session_id, platform="tui")
    _configure_pruning(stale_agent)
    stale_messages = db.get_messages_as_conversation(session_id)
    pending_suffix = [
        {
            "role": "user",
            "content": "pending completion payload",
            "_completion_delivery_synthetic": True,
        },
        _assistant_call("pending_after_winner"),
        _tool_result("pending_after_winner", "pending completion result"),
    ]

    winner_messages = [
        {"role": "user", "content": "winner summary"},
        {"role": "assistant", "content": "winner reply"},
    ]
    winner_fingerprint = db.get_compaction_fingerprint(session_id)
    db.archive_and_compact(
        session_id,
        winner_messages,
        expected_active_fingerprint=winner_fingerprint,
    )

    result, count = stale_agent.context_compressor.prune_tool_results_only(
        [*stale_messages, *pending_suffix], current_tokens=120_000,
    )

    assert count == 0
    assert [message["content"] for message in result] == [
        "winner summary",
        "winner reply",
        "pending completion payload",
        "",
        "pending completion result",
    ]
    assert [message["content"] for message in db.get_messages_as_conversation(session_id)] == [
        "winner summary",
        "winner reply",
    ]


def _raw_compaction_fingerprint(
    db: SessionDB,
    session_id: str,
) -> tuple[list[int], list[tuple]]:
    fingerprint = db.get_compaction_fingerprint(session_id)
    return [int(row[0]) for row in fingerprint], fingerprint


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
    row_ids, fingerprint = _raw_compaction_fingerprint(
        db, session_id,
    )
    assert len(row_ids) == 3

    replacement = [{"role": "user", "content": "summary"}]
    db.archive_and_compact(
        session_id,
        replacement,
        expected_active_fingerprint=fingerprint,
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

    _row_ids, fingerprint = _raw_compaction_fingerprint(
        db, session_id,
    )
    replacement = [{"role": "user", "content": "summary"}]
    db.archive_and_compact(
        session_id,
        replacement,
        expected_active_fingerprint=fingerprint,
    )
    assert [
        (message["role"], message["content"])
        for message in db.get_messages_as_conversation(session_id)
    ] == [("user", "summary")]


def test_raw_row_fence_rejects_append_without_mutating_caller(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "APPEND_RACE_REJECTED"
    db.create_session(session_id, source="tui")
    db.append_messages_batch(session_id, [{"role": "user", "content": "before"}])
    row_ids, fingerprint = _raw_compaction_fingerprint(
        db, session_id,
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
            expected_active_fingerprint=fingerprint,
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
    row_ids, fingerprint = _raw_compaction_fingerprint(
        db, session_id,
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
            expected_active_fingerprint=fingerprint,
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
    row_ids, fingerprint = _raw_compaction_fingerprint(
        db, session_id,
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
            expected_active_fingerprint=fingerprint,
        )

    assert db.get_messages(session_id, include_inactive=True) == durable_before


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
    assert result is not durable
    assert [message["content"] for message in result] == [
        message["content"] for message in durable
    ]
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

    assert result is not messages
    assert count == 0
    assert agent.context_compressor._proactive_prune_rearm_tokens == 0
    assert [message["content"] for message in result] == original_contents
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


def test_model_switch_clears_durable_runway(tmp_path: Path) -> None:
    """update_model must clear BOTH the in-memory and the durable runway."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "MODEL_SWITCH_CLEARS_RUNWAY"
    db.create_session(
        session_id,
        source="telegram",
        model_config={"keep": "value", _REARM_KEY: 120_000},
    )
    agent = _build_agent(db, session_id)
    compressor = agent.context_compressor
    assert compressor._proactive_prune_rearm_tokens == 120_000

    compressor.update_model("other/model", 200_000)

    assert compressor._proactive_prune_rearm_tokens == 0
    assert _REARM_KEY not in _model_config(db, session_id)
    assert _model_config(db, session_id)["keep"] == "value"


def test_patch_session_model_config_merge_and_delete(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PATCH_MODEL_CONFIG"
    db.create_session(
        session_id, source="cli", model_config={"keep": "value", "drop": 1},
    )

    db.patch_session_model_config(session_id, {"drop": None, "added": 7})
    assert _model_config(db, session_id) == {"keep": "value", "added": 7}

    # Missing rows and empty patches are no-ops, never errors.
    db.patch_session_model_config("NO_SUCH_SESSION", {"x": 1})
    db.patch_session_model_config(session_id, {})


def test_incapable_store_short_circuits_before_prune_scan(tmp_path: Path) -> None:
    """A bound store without archive_and_compact must not pay the prune scan."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "INCAPABLE_STORE_FAST_NOOP"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())
    agent = _build_agent(db, session_id)
    _configure_pruning(agent)
    compressor = agent.context_compressor

    class _NoArchiveStore:
        pass

    compressor.bind_session_state(_NoArchiveStore(), session_id)
    messages = db.get_messages_as_conversation(session_id)
    with patch.object(
        type(compressor), "_prune_old_tool_results",
        side_effect=AssertionError("scan must not run for incapable stores"),
    ):
        result, count = compressor.prune_tool_results_only(
            messages, current_tokens=120_000,
        )

    assert result is messages
    assert count == 0
