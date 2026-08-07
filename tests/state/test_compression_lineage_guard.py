"""Regression tests for stale writes after a compression session split."""

from __future__ import annotations

import pytest

from hermes_state import CompressionSessionBusyError, SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _compression_parent(db: SessionDB, session_id: str = "parent") -> None:
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "before split")
    db.end_session(session_id, "compression")


def test_find_live_compression_child_returns_unique_direct_child(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child", source="webui", parent_session_id="parent")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "child"
    assert child["parent_session_id"] == "parent"
    assert child["ended_at"] is None


def test_find_live_compression_child_fails_closed_when_ambiguous(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child-a", source="webui", parent_session_id="parent")
    db.create_session("child-b", source="webui", parent_session_id="parent")

    assert db.find_live_compression_child("parent") is None




def test_find_live_compression_child_ignores_non_continuation_children(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("canonical", source="webui", parent_session_id="parent")
    db.create_session(
        "branch",
        source="webui",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    db.create_session(
        "delegate",
        source="webui",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )
    db.create_session("tool-child", source="tool", parent_session_id="parent")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "canonical"








def test_publish_compression_child_is_atomic_on_handoff_failure(
    db: SessionDB, monkeypatch
) -> None:
    db.create_session("atomic-parent", source="webui")
    db.append_message("atomic-parent", "user", "original")
    assert db.try_acquire_compression_lock("atomic-parent", "winner", ttl_seconds=60)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("handoff insert failed")

    monkeypatch.setattr(db, "_insert_message_rows", _boom)
    with pytest.raises(RuntimeError, match="handoff insert failed"):
        db.publish_compression_child(
            parent_session_id="atomic-parent",
            child_session_id="atomic-child",
            source="webui",
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder="winner",
        )

    parent = db.get_session("atomic-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert db.get_session("atomic-child") is None


def test_publish_compression_child_exposes_complete_child(db: SessionDB) -> None:
    db.create_session("atomic-parent", source="webui")
    db.append_message("atomic-parent", "user", "original")
    assert db.try_acquire_compression_lock("atomic-parent", "winner", ttl_seconds=60)

    db.publish_compression_child(
        parent_session_id="atomic-parent",
        child_session_id="atomic-child",
        source="webui",
        system_prompt="compressed system",
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="winner",
    )

    assert db.get_session("atomic-parent")["end_reason"] == "compression"
    child = db.find_live_compression_child("atomic-parent")
    assert child is not None
    assert child["id"] == "atomic-child"
    assert child["system_prompt"] == "compressed system"
    assert [m["content"] for m in db.get_messages("atomic-child")] == ["summary"]


def test_publish_compression_child_rejects_lost_or_expired_lease(db: SessionDB) -> None:
    db.create_session("lease-parent", source="webui")
    db.append_message("lease-parent", "user", "new durable turn")
    assert db.try_acquire_compression_lock("lease-parent", "new-winner", ttl_seconds=60)

    with pytest.raises(RuntimeError, match="lease lost"):
        db.publish_compression_child(
            parent_session_id="lease-parent",
            child_session_id="stale-child",
            source="webui",
            messages=[{"role": "user", "content": "stale summary"}],
            compression_lock_holder="old-loser",
        )

    parent = db.get_session("lease-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert db.get_session("stale-child") is None
    assert [m["content"] for m in db.get_messages("lease-parent")] == [
        "new durable turn"
    ]


def test_publish_compression_child_rejects_changed_parent_fingerprint(
    db: SessionDB,
) -> None:
    """Rotation must fence a transcript rewrite after summary input capture."""
    parent_id = "rotation-fingerprint-parent"
    db.create_session(parent_id, source="webui")
    db.append_message(parent_id, "user", "summary input")
    expected = db.get_compaction_fingerprint(parent_id)
    assert db.try_acquire_compression_lock(parent_id, "winner", ttl_seconds=60)

    db.archive_and_compact(
        parent_id,
        [{"role": "user", "content": "newer active view"}],
        expected_active_fingerprint=expected,
    )

    with pytest.raises(
        CompressionSessionBusyError,
        match="changed before compression child publication",
    ):
        db.publish_compression_child(
            parent_session_id=parent_id,
            child_session_id="stale-rotation-child",
            source="webui",
            messages=[{"role": "user", "content": "stale summary"}],
            compression_lock_holder="winner",
            expected_active_fingerprint=expected,
        )

    assert db.get_session(parent_id)["ended_at"] is None
    assert db.get_session("stale-rotation-child") is None
    assert [message["content"] for message in db.get_messages(parent_id)] == [
        "newer active view"
    ]


def test_compaction_snapshot_preserves_exact_five_row_payload(db: SessionDB) -> None:
    """Compaction sees raw durable rows, not the repaired replay projection."""
    session_id = "raw-compaction-snapshot"
    db.create_session(session_id, source="webui")
    expected = [
        {
            "role": "user",
            "content": "  synthetic spaced content  ",
            "api_content": "  exact wire bytes  ",
            "timestamp": 101.25,
            "token_count": 7,
        },
        {
            "role": "user",
            "content": (
                "Review the conversation above and update the skill library "
                "without changing these bytes"
            ),
            "timestamp": 102.25,
        },
        {
            "role": "assistant",
            "content": "  curator reply stays present  ",
            "timestamp": 103.25,
        },
        {
            "role": "assistant",
            "content": "[memory]",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "memory", "arguments": "{}"},
                }
            ],
            "finish_reason": "tool_calls",
            "reasoning": "  exact reasoning  ",
            "reasoning_content": "  exact reasoning content  ",
            "reasoning_details": [{"type": "summary", "text": "detail"}],
            "codex_reasoning_items": [{"type": "reasoning", "id": "r1"}],
            "codex_message_items": [{"type": "message", "id": "m1"}],
            "message_id": "platform-1",
            "observed": True,
            "timestamp": 104.25,
            "display_kind": "tool_call",
            "display_metadata": {"raw": "metadata"},
        },
        {
            "role": "tool",
            "content": "result",
            "tool_call_id": "call-1",
            "tool_name": "memory",
            "name": "memory",
            "effect_disposition": "observed",
            "timestamp": 105.25,
        },
    ]
    db.append_messages_batch(session_id, expected)

    fingerprint, snapshot = db.get_compaction_snapshot(session_id)

    assert len(fingerprint) == 5
    assert len(snapshot) == 5
    assert snapshot == expected
    assert all("_db_persisted" not in message for message in snapshot)


def test_compression_lease_blocks_non_owner_but_allows_owner_flush(
    db: SessionDB,
) -> None:
    db.create_session("leased", source="webui")
    assert db.try_acquire_compression_lock("leased", "winner", ttl_seconds=60)

    with pytest.raises(RuntimeError, match="being compressed"):
        db.append_message("leased", "user", "late stale turn")

    db.append_message(
        "leased",
        "assistant",
        "winner flush",
        compression_lock_holder="winner",
    )
    assert [m["content"] for m in db.get_messages("leased")] == ["winner flush"]


def test_archive_and_compact_rejects_raw_transcript_change_after_capture(
    db: SessionDB,
) -> None:
    session_id = "raw-fingerprint"
    holder = "compressor"
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "same visible content")
    assert db.try_acquire_compression_lock(session_id, holder, ttl_seconds=60)
    fingerprint = db.get_compaction_fingerprint(session_id)

    # This sidecar backfill mutates the durable request payload without changing
    # the projected role/content transcript and is not blocked by the lease.
    assert db.set_latest_user_api_content(
        session_id,
        "same visible content",
        "different wire payload",
    ) == 1

    with pytest.raises(CompressionSessionBusyError, match="transcript changed"):
        db.archive_and_compact(
            session_id,
            [{"role": "user", "content": "stale summary"}],
            compression_lock_holder=holder,
            require_compression_lease=True,
            expected_active_fingerprint=fingerprint,
        )

    messages = db.get_messages(session_id, include_inactive=True)
    assert [message["content"] for message in messages] == ["same visible content"]
    assert messages[0]["active"] == 1
