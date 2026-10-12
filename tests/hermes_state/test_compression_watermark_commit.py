"""Watermark commit: concurrent appends survive in-place compaction (#75316).

The provider summary call is external and slow. Messages that arrive while it
runs must (a) persist immediately — appends are not fenced by the compression
lock — and (b) survive the commit: ``archive_and_compact(watermark=...)``
re-sequences every active row above the watermark after the compacted set
instead of archiving it. The commit is holder-fenced: a compression whose
lease was reclaimed cannot publish a stale compaction.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import types
from pathlib import Path

import pytest

from agent import conversation_compression as cc
from hermes_state import SessionCompressionInProgressError, SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    d = SessionDB(tmp_path / "state.db")
    d.create_session("sess1", source="test")
    return d


def _seed(db: SessionDB, n: int = 6) -> None:
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        db.append_message("sess1", role=role, content=f"turn {i}")


SUMMARY = [
    {"role": "user", "content": "[CONTEXT COMPACTION] summary of turns 0-5"},
    {"role": "assistant", "content": "Continuing from the summary."},
]


class TestWatermarkCommit:
    def test_concurrent_tail_survives_compaction(self, db: SessionDB) -> None:
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        # Simulate the slow summary window: two messages land after capture.
        db.append_message("sess1", role="user", content="mid-compression steer")
        db.append_message("sess1", role="assistant", content="mid-compression reply")

        count = db.archive_and_compact("sess1", SUMMARY, watermark=watermark)

        live = db.get_messages("sess1")
        contents = [r["content"] for r in live]
        assert contents == [
            "[CONTEXT COMPACTION] summary of turns 0-5",
            "Continuing from the summary.",
            "mid-compression steer",
            "mid-compression reply",
        ], "tail must follow the summary, in arrival order"
        assert count == 4

    def test_source_guard_allows_append_only_foreign_tail(self, db: SessionDB) -> None:
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        source_ids = [int(row["id"]) for row in db.get_messages("sess1")]
        db.append_message("sess1", role="user", content="mid-compression steer")

        db.archive_and_compact(
            "sess1", SUMMARY, watermark=watermark, source_ids=source_ids
        )

        assert [row["content"] for row in db.get_messages("sess1")] == [
            SUMMARY[0]["content"],
            SUMMARY[1]["content"],
            "mid-compression steer",
        ]

    def test_source_guard_rejects_pre_watermark_source_mutation(
        self, db: SessionDB
    ) -> None:
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        source_ids = [int(row["id"]) for row in db.get_messages("sess1")]
        db._conn.execute("UPDATE messages SET active = 0 WHERE id = ?", (source_ids[0],))
        db._conn.commit()

        with pytest.raises(SessionCompressionInProgressError, match="source changed"):
            db.archive_and_compact(
                "sess1", SUMMARY, watermark=watermark, source_ids=source_ids
            )

        assert [row["content"] for row in db.get_messages("sess1")] == [
            f"turn {i}" for i in range(1, 6)
        ]

    def test_source_guard_rejects_in_place_pre_watermark_sidecar_mutation(
        self, db: SessionDB
    ) -> None:
        _seed(db)
        watermark, source_signature = db.get_active_message_source_snapshot("sess1")
        assert db.set_latest_user_api_content("sess1", "turn 4", "stamped sidecar") == 1

        with pytest.raises(SessionCompressionInProgressError, match="source changed"):
            db.archive_and_compact(
                "sess1",
                SUMMARY,
                watermark=watermark,
                source_signature=source_signature,
            )

        assert [row["content"] for row in db.get_messages("sess1")] == [
            f"turn {i}" for i in range(6)
        ]

    def test_tail_clone_preserves_every_column(self, db: SessionDB) -> None:
        """The pure-SQL clone must carry sidecar fields byte-exact."""
        _seed(db, 2)
        watermark = db.get_active_message_watermark("sess1")
        db.append_message(
            "sess1",
            role="assistant",
            content="tool caller",
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "terminal", "arguments": "{}"}}],
        )
        db.append_message(
            "sess1", role="tool", content="tool output",
            tool_call_id="c1", tool_name="terminal",
        )

        db.archive_and_compact("sess1", SUMMARY, watermark=watermark)

        live = db.get_messages("sess1")
        by_content = {r["content"]: r for r in live}
        caller = by_content["tool caller"]
        result = by_content["tool output"]
        parsed = caller["tool_calls"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        assert parsed and parsed[0]["id"] == "c1"
        assert result["tool_call_id"] == "c1"
        assert result["tool_name"] == "terminal"

    def test_conversation_load_is_correct_after_commit(self, db: SessionDB) -> None:
        """The live conversation projection sees summary + tail, in order."""
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        db.append_message("sess1", role="user", content="late arrival")

        db.archive_and_compact("sess1", SUMMARY, watermark=watermark)

        convo = db.get_messages_as_conversation("sess1")
        assert [m["content"] for m in convo] == [
            "[CONTEXT COMPACTION] summary of turns 0-5",
            "Continuing from the summary.",
            "late arrival",
        ]

    def test_no_tail_behaves_identically_to_legacy(self, db: SessionDB) -> None:
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        count = db.archive_and_compact("sess1", SUMMARY, watermark=watermark)
        assert count == 2
        assert [r["content"] for r in db.get_messages("sess1")] == [
            SUMMARY[0]["content"], SUMMARY[1]["content"],
        ]

    def test_none_watermark_preserves_historical_behavior(self, db: SessionDB) -> None:
        """watermark=None archives everything — the pre-#75316 contract."""
        _seed(db)
        db.append_message("sess1", role="user", content="gets archived")
        count = db.archive_and_compact("sess1", SUMMARY, watermark=None)
        assert count == 2
        contents = [r["content"] for r in db.get_messages("sess1")]
        assert "gets archived" not in contents

    def test_archived_rows_stay_recoverable(self, db: SessionDB) -> None:
        """Originals (snapshot AND tail source rows) survive as archived."""
        _seed(db, 4)
        watermark = db.get_active_message_watermark("sess1")
        db.append_message("sess1", role="user", content="tail row")
        db.archive_and_compact("sess1", SUMMARY, watermark=watermark)

        everything = db.get_messages("sess1", include_inactive=True)
        archived = [r for r in everything if not r["active"]]
        assert sum(1 for r in archived if r["content"] == "turn 0") == 1
        # The tail original is archived; its clone is the live copy.
        tail_rows = [r for r in everything if r["content"] == "tail row"]
        assert sorted(bool(r["active"]) for r in tail_rows) == [False, True]

    def test_session_counters_include_tail(self, db: SessionDB) -> None:
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        db.append_message(
            "sess1", role="assistant", content="tail with tools",
            tool_calls=[{"id": "t1", "type": "function",
                         "function": {"name": "x", "arguments": "{}"}}],
        )
        db.archive_and_compact("sess1", SUMMARY, watermark=watermark)
        info = db.get_session("sess1")
        assert info["message_count"] == 3
        assert info["tool_call_count"] == 1


class TestCommitFence:
    def test_commit_refused_when_lease_lost(self, db: SessionDB) -> None:
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        assert db.try_acquire_compression_lock("sess1", "worker-A") is True
        # Lease reclaimed by another writer while worker-A's summary ran.
        db.release_compression_lock("sess1", "worker-A")
        assert db.try_acquire_compression_lock("sess1", "worker-B") is True

        with pytest.raises(SessionCompressionInProgressError):
            db.archive_and_compact(
                "sess1", SUMMARY, watermark=watermark, lock_holder="worker-A"
            )
        # Nothing committed: original transcript intact.
        assert [r["content"] for r in db.get_messages("sess1")] == [
            f"turn {i}" for i in range(6)
        ]

    def test_commit_refused_when_lease_expired(self, db: SessionDB) -> None:
        _seed(db)
        assert db.try_acquire_compression_lock(
            "sess1", "worker-A", ttl_seconds=0.05
        ) is True
        time.sleep(0.1)
        with pytest.raises(SessionCompressionInProgressError):
            db.archive_and_compact("sess1", SUMMARY, lock_holder="worker-A")

    def test_commit_allowed_for_live_holder(self, db: SessionDB) -> None:
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        assert db.try_acquire_compression_lock("sess1", "worker-A") is True
        count = db.archive_and_compact(
            "sess1", SUMMARY, watermark=watermark, lock_holder="worker-A"
        )
        assert count == 2

    def test_refused_commit_rolls_back_atomically(self, db: SessionDB) -> None:
        """Failure injection: the fence raise must leave zero partial writes."""
        _seed(db)
        before = db.get_messages("sess1", include_inactive=True)
        with pytest.raises(SessionCompressionInProgressError):
            db.archive_and_compact("sess1", SUMMARY, lock_holder="never-held")
        after = db.get_messages("sess1", include_inactive=True)
        assert len(before) == len(after)
        assert all(r["active"] for r in after)


class TestConcurrentAppendDuringCompaction:
    def test_append_racing_the_commit_transaction(self, db: SessionDB) -> None:
        """An append serialized behind the commit lands AFTER it — never lost.

        SQLite's write lock serializes the two transactions; whichever side
        wins, the append must end up in the live transcript.
        """
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")

        barrier = threading.Barrier(2, timeout=10)
        append_err: list = []

        def _racer():
            barrier.wait()
            try:
                db.append_message("sess1", role="user", content="racer")
            except Exception as exc:  # pragma: no cover
                append_err.append(exc)

        t = threading.Thread(target=_racer, daemon=True)
        t.start()
        barrier.wait()
        db.archive_and_compact("sess1", SUMMARY, watermark=watermark)
        t.join(timeout=10)

        assert not append_err, f"append died during commit race: {append_err}"
        contents = [r["content"] for r in db.get_messages("sess1")]
        assert "racer" in contents, "racing append was lost"


class TestRotationPathWatermark:
    """Legacy (non-in-place) compression rotates to a child session —
    the concurrent tail must follow the rotation instead of stranding in
    the closed parent."""

    @pytest.mark.parametrize(
        "tail_content",
        [
            "mid-rotation steer",
            [
                {"type": "text", "text": "inspect this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        ],
        ids=["text", "multimodal"],
    )
    def test_tail_clones_into_the_child(self, db: SessionDB, tail_content) -> None:
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        assert db.try_acquire_compression_lock("sess1", "rotator") is True
        db.append_message("sess1", role="user", content=tail_content)
        # Ceiling captured AFTER the foreign append, BEFORE the rotation
        # path's own pre-publish flush (which this test has none of).
        ceiling = db.get_active_message_watermark("sess1")

        db.publish_compression_child(
            parent_session_id="sess1",
            child_session_id="child1",
            source="test",
            messages=SUMMARY,
            compression_lock_holder="rotator",
            require_compression_lease=True,
            watermark=watermark,
            watermark_ceiling=ceiling,
        )

        child = db.get_messages_as_conversation("child1")
        assert [m["content"] for m in child] == [
            SUMMARY[0]["content"],
            SUMMARY[1]["content"],
            tail_content,
        ]
        model_history, display_history = db.get_resume_conversations("child1")
        visible_steers = [
            message
            for message in display_history
            if message.get("content") == tail_content
        ]
        assert len(visible_steers) == 1
        assert visible_steers[0]["_row_id"] == model_history[-1]["_row_id"]
        assert all(
            message.get("content") != tail_content
            for message in db.get_ancestor_display_prefix("child1")
        )
        info = db.get_session("child1")
        assert info["message_count"] == 3
        # Parent keeps its copy for lineage recovery; parent is closed.
        parent_info = db.get_session("sess1")
        assert parent_info["end_reason"] == "compression"

    def test_rotation_source_guard_rejects_pre_watermark_mutation(
        self, db: SessionDB
    ) -> None:
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        source_ids = [int(row["id"]) for row in db.get_messages("sess1")]
        assert db.try_acquire_compression_lock("sess1", "rotator") is True
        db._conn.execute("DELETE FROM messages WHERE id = ?", (source_ids[0],))
        db._conn.commit()

        with pytest.raises(SessionCompressionInProgressError, match="source changed"):
            db.publish_compression_child(
                parent_session_id="sess1",
                child_session_id="child1",
                source="test",
                messages=SUMMARY,
                compression_lock_holder="rotator",
                require_compression_lease=True,
                watermark=watermark,
                source_ids=source_ids,
            )

        assert db.get_session("child1") is None
        assert db.get_session("sess1")["ended_at"] is None

    def test_rotation_adoption_keeps_signed_cutoff(
        self, db: SessionDB
    ) -> None:
        """Adoption advances the clone cutoff and keeps the signed digest.

        A later append or preflush is legal. Rewriting a signed prefix still
        aborts, and the child is not published.
        """
        _seed(db, 2)
        signed, digest = db.get_active_message_source_snapshot("sess1")
        assert db.try_acquire_compression_lock("sess1", "rotator") is True
        db.append_message(
            "sess1", role="user", content="adopted preflush",
            compression_lock_holder="rotator",
        )
        adopted = db.get_active_message_watermark("sess1")
        assert adopted > signed
        db.append_message("sess1", role="user", content="foreign after adoption")
        ceiling = db.get_active_message_watermark("sess1")

        db.publish_compression_child(
            parent_session_id="sess1",
            child_session_id="child1",
            source="test",
            messages=SUMMARY,
            compression_lock_holder="rotator",
            require_compression_lease=True,
            watermark=adopted,
            watermark_ceiling=ceiling,
            source_signature=digest,
            source_watermark=signed,
        )
        child = [m["content"] for m in db.get_messages_as_conversation("child1")]
        assert child == [
            SUMMARY[0]["content"],
            SUMMARY[1]["content"],
            "foreign after adoption",
        ]
        assert "adopted preflush" not in child

        db.create_session("sess2", source="test")
        for i in range(2):
            db.append_message("sess2", role="user" if i % 2 == 0 else "assistant", content=f"turn {i}")
        signed2, digest2 = db.get_active_message_source_snapshot("sess2")
        assert db.try_acquire_compression_lock("sess2", "rotator") is True
        prefix = next(int(row["id"]) for row in db.get_messages("sess2"))
        db.append_message("sess2", role="user", content="adopted tail")
        adopted2 = db.get_active_message_watermark("sess2")
        db._conn.execute("UPDATE messages SET content = ? WHERE id = ?", ("rewritten", prefix))
        db._conn.commit()
        with pytest.raises(SessionCompressionInProgressError, match="source changed"):
            db.publish_compression_child(
                parent_session_id="sess2",
                child_session_id="child2",
                source="test",
                messages=SUMMARY,
                compression_lock_holder="rotator",
                require_compression_lease=True,
                watermark=adopted2,
                source_signature=digest2,
                source_watermark=signed2,
            )
        assert db.get_session("child2") is None
        assert db.get_session("sess2")["ended_at"] is None

    def _lease(self, db: SessionDB, sid: str):
        signed, digest = db.get_active_message_source_snapshot(sid)
        agent = types.SimpleNamespace(
            _session_db=db, session_id=sid, platform="cli", model="m",
            _session_init_model_config=None, working_directory=None,
            _memory_manager=None, context_compressor=types.SimpleNamespace(),
            _flush_messages_to_session_db=lambda *a, **k: True,
            _persist_user_message_idx=None, _session_messages=None,
            _gateway_session_key=None, _cached_system_prompt="sys",
            commit_memory_session=lambda *_a, **_k: None,
        )
        lease = cc._CompressionLease(
            agent, db=db, sid=sid, ttl=60.0, refresh_interval=None,
            commit_fence=None, lifecycle=cc._CompactionLifecycle(agent, False),
        )
        lease.holder = "rotator"
        lease.watermark = signed
        lease.signed_watermark = signed
        lease.source_signature = digest
        return agent, lease

    def _adopt(self, db: SessionDB, sid: str, before: list):
        agent, lease = self._lease(db, sid)
        signed = lease.signed_watermark
        db.append_message(sid, role="user", content="adopted preflush")
        adopted = cc._adopt_grown_durable_parent(agent, lease, before)
        assert adopted is not None
        assert lease.watermark > signed and lease.signed_watermark == signed
        return agent, lease, adopted

    @pytest.mark.parametrize("mutate", [False, True])
    def test_rotated_publication_keeps_signed_cutoff_after_adoption(
        self, db: SessionDB, mutate: bool
    ) -> None:
        """Rotation publishes through the real caller after adoption.

        The clone cutoff advances. The signed cutoff does not. A later append
        is legal; a signed-prefix rewrite still aborts.
        """
        _seed(db, 2)
        before = db.get_messages_as_conversation("sess1")
        assert db.try_acquire_compression_lock("sess1", "rotator") is True
        agent, lease = self._lease(db, "sess1")
        db.append_message("sess1", role="user", content="adopted preflush")
        adopted = cc._adopt_grown_durable_parent(agent, lease, before)
        assert adopted is not None
        assert lease.watermark > lease.signed_watermark
        db.append_message("sess1", role="user", content="foreign after adoption")
        if mutate:
            prefix = next(int(row["id"]) for row in db.get_messages("sess1"))
            db._conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?", ("rewritten", prefix),
            )
            db._conn.commit()
            with pytest.raises(SessionCompressionInProgressError, match="source changed"):
                cc._publish_rotated_compaction(
                    agent, before, SUMMARY, new_system_prompt="sys", lease=lease,
                    old_session_id="sess1", compressed_user_turn_outcome="none",
                )
            assert agent.session_id == "sess1"
            assert db.get_session("sess1")["ended_at"] is None
            return
        cc._publish_rotated_compaction(
            agent, before, SUMMARY, new_system_prompt="sys", lease=lease,
            old_session_id="sess1", compressed_user_turn_outcome="none",
        )
        child = [m["content"] for m in db.get_messages_as_conversation(agent.session_id)]
        assert child == [
            SUMMARY[0]["content"], SUMMARY[1]["content"], "foreign after adoption",
        ]
        assert "adopted preflush" not in child

    def test_inplace_publication_keeps_signed_cutoff_after_adoption(
        self, db: SessionDB
    ) -> None:
        """The in-place caller uses the same signed cutoff after adoption."""
        _seed(db, 2)
        before = db.get_messages_as_conversation("sess1")
        assert db.try_acquire_compression_lock("sess1", "rotator") is True
        agent, lease, adopted = self._adopt(db, "sess1", before)
        db.append_message("sess1", role="user", content="foreign after adoption")
        from agent.context_compressor import _DB_PERSISTED_MARKER
        held = [{**m, _DB_PERSISTED_MARKER: True} for m in adopted]
        outcome = cc._commit_compaction(
            agent, held, SUMMARY, in_place=True, lease=lease,
            new_system_prompt="sys", system_message="sys",
            compressed_user_turn_outcome="none", messages_before_compression=held,
            made_progress=True, attempt=cc._Attempt({}, 0, 0.0, {}),
        )
        assert outcome.session_commit_succeeded
        live = [m["content"] for m in db.get_messages_as_conversation("sess1")]
        assert live == [
            SUMMARY[0]["content"], SUMMARY[1]["content"], "foreign after adoption",
        ]
        assert "adopted preflush" not in live

        db.create_session("sess2", source="test")
        for i in range(2):
            db.append_message(
                "sess2", role="user" if i % 2 == 0 else "assistant", content=f"turn {i}",
            )
        before2 = db.get_messages_as_conversation("sess2")
        assert db.try_acquire_compression_lock("sess2", "rotator") is True
        prefix = next(int(row["id"]) for row in db.get_messages("sess2"))
        _agent2, lease2, adopted2 = self._adopt(db, "sess2", before2)
        db._conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?", ("rewritten", prefix),
        )
        db._conn.commit()
        held2 = [{**m, _DB_PERSISTED_MARKER: True} for m in adopted2]
        before_rows = [m["content"] for m in db.get_messages_as_conversation("sess2")]
        outcome2 = cc._commit_compaction(
            _agent2, held2, SUMMARY, in_place=True, lease=lease2,
            new_system_prompt="sys", system_message="sys",
            compressed_user_turn_outcome="none",
            messages_before_compression=held2, made_progress=True,
            attempt=cc._Attempt({}, 0, 0.0, {}),
        )
        assert outcome2.session_commit_succeeded is False
        assert outcome2.split_status == "failed_not_indexed"
        assert [m["content"] for m in db.get_messages_as_conversation("sess2")] == before_rows
        assert db.get_session("sess2")["ended_at"] is None

    def test_ceiling_excludes_the_rotators_own_flush(self, db: SessionDB) -> None:
        """Rows the rotation path flushes AFTER the ceiling (its own input
        transcript, already inside the handoff) must NOT be cloned."""
        _seed(db)
        watermark = db.get_active_message_watermark("sess1")
        assert db.try_acquire_compression_lock("sess1", "rotator") is True
        db.append_message("sess1", role="user", content="foreign steer")
        ceiling = db.get_active_message_watermark("sess1")
        # Simulates the #47202 pre-publish flush of the rotator's own input.
        db.append_message(
            "sess1", role="user", content="rotator's own flush",
            compression_lock_holder="rotator",
        )

        db.publish_compression_child(
            parent_session_id="sess1",
            child_session_id="child1",
            source="test",
            messages=SUMMARY,
            compression_lock_holder="rotator",
            require_compression_lease=True,
            watermark=watermark,
            watermark_ceiling=ceiling,
        )

        child_contents = [
            m["content"] for m in db.get_messages_as_conversation("child1")
        ]
        assert "foreign steer" in child_contents
        assert "rotator's own flush" not in child_contents

    def test_no_watermark_keeps_historical_rotation(self, db: SessionDB) -> None:
        _seed(db)
        assert db.try_acquire_compression_lock("sess1", "rotator") is True
        db.append_message("sess1", role="user", content="stranded either way")
        db.publish_compression_child(
            parent_session_id="sess1",
            child_session_id="child1",
            source="test",
            messages=SUMMARY,
            compression_lock_holder="rotator",
            require_compression_lease=True,
        )
        child = db.get_messages_as_conversation("child1")
        assert [m["content"] for m in child] == [
            SUMMARY[0]["content"], SUMMARY[1]["content"],
        ]
