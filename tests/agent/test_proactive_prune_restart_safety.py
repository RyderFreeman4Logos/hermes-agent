"""Restart-safety regressions for proactive tool-result pruning."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_compressor import _estimate_msg_budget_tokens
from hermes_state import CompressionSessionClosedError, SessionDB


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
