"""Every display projection of a compacted session must agree.

In-place compaction archives earlier turns as ``active=0, compacted=1`` rows.
They are durable display history — the user's own conversation, still on disk.
#80680 taught the REST transcript read to include them, but three GATEWAY
display projections kept filtering ``active = 1``:

- ``get_resume_conversations()``   — what ``session.resume`` ships
- ``get_ancestor_display_prefix()`` — the ancestor lineage prefix
- ``get_messages_as_conversation()`` — the warm-session payload on tab switch

So the same conversation read four ways gave two different answers: REST showed
everything, the gateway cut the transcript off at the compaction boundary. The
user sees their chat "vanish" down to a summary plus a couple of carried-forward
turns, and a resumed agent that cannot see its own completed work starts it over
(#92080, #93618, #68321).

These tests assert the INVARIANT — all display reads of one session return the
same transcript — rather than any particular row count, and pin the two things
that must NOT grow with it: the model-fed projection stays compressed, and
soft-deleted Undo/Rewind rows stay hidden.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _compact_in_place(db, sid, *, epochs=3, turns=4, tail_count=2):
    """Drive *sid* through repeated in-place compaction, like a long chat."""
    db.create_session(sid, source="desktop")
    for epoch in range(epochs):
        for i in range(turns):
            db.append_message(sid, "user", f"e{epoch} user {i}")
            db.append_message(sid, "assistant", f"e{epoch} assistant {i}")
        live = db.get_messages_as_conversation(sid)
        db.archive_and_compact(
            sid,
            [{"role": "user", "content": f"[summary {epoch}]"}] + live[-tail_count:],
            tail_count=tail_count,
        )
    return sid


def _texts(messages):
    return [(m["role"], m["content"]) for m in messages]


def _rest_display(db, sid):
    """The read that was already correct — the parity reference."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in db.get_messages(sid, include_compacted=True)
    ]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("limit", [1, 2, 3])
def test_bounded_resume_keeps_final_replay_representative_and_fills(db, legacy, limit):
    db.create_session("replay", source="cli")
    db.append_messages_batch("replay", [
        {"role": "assistant", "content": "earlier", "timestamp": 1.0},
        {"role": "user", "content": "ask", "timestamp": 2.0},
        {"role": "user", "content": "ask", "timestamp": 3.0},
        {"role": "assistant", "content": "answer", "timestamp": 4.0},
    ])
    if legacy:
        db._conn.execute("UPDATE messages SET display_identity=NULL, display_order=NULL")
        db._conn.commit()
    _, full = db.get_resume_conversations("replay")
    _, bounded = db.get_resume_conversations("replay", max_display_messages=limit)
    assert bounded == full[-limit:]


@pytest.mark.parametrize("legacy", [False, True])
def test_bounded_resume_does_not_expand_across_answer_barrier(db, legacy):
    db.create_session("replay", source="cli")
    db.append_messages_batch("replay", [
        {"role": "user", "content": "same", "timestamp": 1.0},
        {"role": "assistant", "content": "first answer", "timestamp": 2.0},
        {"role": "user", "content": "same", "timestamp": 3.0},
        {"role": "assistant", "content": "second answer", "timestamp": 4.0},
    ])
    if legacy:
        db._conn.execute("UPDATE messages SET display_identity=NULL, display_order=NULL")
        db._conn.commit()
    pages = []
    reader = db._read_all

    def traced(*args, **kwargs):
        result = reader(*args, **kwargs)
        if "FROM messages WHERE id IN (" in args[0]:
            pages.append(len(result))
        return result

    db._read_all = traced
    _, display = db.get_resume_conversations("replay", max_display_messages=2)
    assert [(m["content"], m["timestamp"]) for m in display] == [
        ("same", 3.0), ("second answer", 4.0),
    ]
    assert pages == [2]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("boundary", ["adjacent", "hidden", "answer", "branch"])
def test_dense_replay_selects_before_bounded_payload_transfer(tmp_path, legacy, boundary):
    path = tmp_path / "dense.db"
    db = SessionDB(path)
    db.create_session("root", source="cli")
    db.append_messages_batch("root", [
        {"role": "user", "content": "same" + "x" * 4096, "timestamp": float(i)}
        for i in range(100)
    ])
    target = "root"
    if boundary == "branch":
        db.create_session("branch", source="cli", parent_session_id="root",
                          model_config={"_branched_from": "root"})
        target = "branch"
    elif boundary == "hidden":
        db.append_message("root", "assistant", "invisible", display_metadata={"model_only": True})
    elif boundary == "answer":
        db.append_message("root", "assistant", "first answer")
    db.append_messages_batch(target, [
        {"role": "user", "content": "same" + "x" * 4096, "timestamp": 200.0},
        {"role": "assistant", "content": "end", "timestamp": 201.0},
    ])
    if legacy:
        db._conn.execute("UPDATE messages SET display_identity=NULL, display_order=NULL")
        db._conn.commit()
    db.close()
    before = path.read_bytes()
    ro = SessionDB(path, read_only=True)
    try:
        _, full = ro.get_resume_conversations(target)
        payload_pages = []
        reader = ro._read_all

        def observed(sql, params=()):
            rows = reader(sql, params)
            if "FROM messages WHERE id IN (" in sql:
                payload_pages.append(len(rows))
            return rows

        ro._read_all = observed
        _, bounded = ro.get_resume_conversations(target, max_display_messages=2)
        assert bounded == full[-2:]
        assert payload_pages and max(payload_pages) <= 2
    finally:
        ro.close()
        assert path.read_bytes() == before


class TestDisplayProjectionParity:
    def test_resume_display_matches_the_rest_transcript(self, db):
        sid = _compact_in_place(db, "chat")

        _, display = db.get_resume_conversations(sid)

        assert _texts(display) == _texts(_rest_display(db, sid))

    def test_warm_session_display_matches_the_rest_transcript(self, db):
        """The read behind ``_live_visible_history`` (switching back to a tab)."""
        sid = _compact_in_place(db, "chat")

        warm = db.get_messages_as_conversation(
            sid, include_ancestors=True, include_row_ids=True, include_compacted=True
        )

        assert _texts(warm) == _texts(_rest_display(db, sid))

    def test_pre_compaction_turns_survive_in_the_resume_transcript(self, db):
        """The user's own first turn is still there after several compactions."""
        sid = _compact_in_place(db, "chat")

        _, display = db.get_resume_conversations(sid)

        assert ("user", "e0 user 0") in _texts(display)
        assert ("assistant", "e0 assistant 0") in _texts(display)

    def test_display_read_dedupes_carried_forward_tail(self, db):
        """Each logical message appears once, not once per compaction epoch."""
        sid = _compact_in_place(db, "chat", epochs=4, tail_count=2)

        _, display = db.get_resume_conversations(sid)
        seen = _texts(display)

        assert len(seen) == len(set(seen))


class TestModelProjectionStaysCompressed:
    def test_model_history_excludes_archived_rows(self, db):
        """Compaction must still do its job: the model gets the compressed set."""
        sid = _compact_in_place(db, "chat")

        model, display = db.get_resume_conversations(sid)

        assert len(model) < len(display)
        assert ("user", "e0 user 0") not in _texts(model)

    def test_model_history_matches_the_active_only_read(self, db):
        sid = _compact_in_place(db, "chat")

        model, _ = db.get_resume_conversations(sid)
        active_only = db.get_messages_as_conversation(sid, repair_alternation=True)

        assert _texts(model) == _texts(active_only)


class TestSoftDeletedRowsStayHidden:
    def test_rewound_rows_are_excluded_from_the_display_projections(self, db):
        """Undo/Rewind rows (active=0, compacted=0) are NOT display history."""
        sid = "chat"
        db.create_session(sid, source="desktop")
        db.append_message(sid, "user", "kept")
        db.append_message(sid, "assistant", "kept reply")
        db.append_message(sid, "user", "taken back")
        db.append_message(sid, "assistant", "taken back reply")

        rewind_target = next(
            m for m in reversed(db.get_messages(sid)) if m["role"] == "user"
        )
        db.rewind_to_message(sid, rewind_target["id"])

        _, display = db.get_resume_conversations(sid)
        warm = db.get_messages_as_conversation(
            sid, include_ancestors=True, include_compacted=True
        )

        for projection in (display, warm):
            contents = [c for _, c in _texts(projection)]
            assert "taken back" not in contents
            assert "kept" in contents


class TestAncestorPrefix:
    def test_prefix_includes_a_compacted_ancestor_s_archived_rows(self, db):
        """A compression ROTATION's parent still shows its pre-compaction turns."""
        parent, child = "parent", "child"
        db.create_session(parent, source="desktop")
        for i in range(3):
            db.append_message(parent, "user", f"P user {i}")
            db.append_message(parent, "assistant", f"P assistant {i}")
        db.archive_and_compact(parent, [{"role": "user", "content": "[parent summary]"}])

        db.create_session(child, source="desktop", parent_session_id=parent)
        db.append_message(child, "user", "C user 0")
        db.append_message(child, "assistant", "C assistant 0")

        prefix = db.get_ancestor_display_prefix(child)
        _, display = db.get_resume_conversations(child)

        assert ("user", "P user 0") in _texts(prefix)
        assert ("user", "P user 0") in _texts(display)
        # The child's own turns belong to the tip, never the ancestor prefix.
        assert ("user", "C user 0") not in _texts(prefix)

    def test_explicit_branch_has_no_ancestor_prefix(self, db):
        """A /branch copy owns its transcript; the live parent must not leak in."""
        sid = _compact_in_place(db, "chat")
        db.create_session(
            "branch",
            source="desktop",
            parent_session_id=sid,
            model_config={"_branched_from": sid},
        )
        db.append_message("branch", "user", "branch turn")

        assert db.get_ancestor_display_prefix("branch") == []

        _, display = db.get_resume_conversations("branch")
        assert _texts(display) == [("user", "branch turn")]


class TestResumeGuardBoundsWhatResumeLoads:
    def test_guard_counts_the_rows_the_display_read_materializes(self, db):
        """The guard must not undercount: it bounds an in-memory materialization."""
        sid = _compact_in_place(db, "chat", epochs=4)

        _, display = db.get_resume_conversations(sid)

        assert db.get_resume_message_count(sid) >= len(display)

    def test_guard_rejects_a_lineage_over_the_limit(self, db):
        from hermes_state import SessionResumeTooLargeError

        sid = _compact_in_place(db, "chat", epochs=4)

        with pytest.raises(SessionResumeTooLargeError):
            db.assert_resume_safe(sid, max_messages=2)

    def test_tip_only_guard_still_bounds_only_the_live_tip(self, db):
        """The #4130 carve-out: a healthy compacted chat must stay resumable.

        A well-compressed conversation is exactly the shape compression is
        meant to produce. Counting its archive against a tip-sized budget is
        what stranded Bot Chats on "Waking up…"; ``tip_only`` callers never
        materialize the archive, so they keep the active-only bound.
        """
        sid = _compact_in_place(db, "chat", epochs=4)

        tip_count = db.get_resume_message_count(sid, tip_only=True)

        assert tip_count < db.get_resume_message_count(sid)
        assert db.assert_resume_safe(sid, max_messages=tip_count, tip_only=True)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("scenario", ["interleaved", "hidden", "new_time", "compacted", "edited", "branch", "pagination"])
def test_bounded_resume_preserves_display_projection(tmp_path, legacy, scenario):
    """Selection must preserve logical order, not re-sort the chosen generation ids."""
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        db.create_session("root", source="cli")
        db.append_messages_batch("root", [
            {"role": "user", "content": "question", "timestamp": 1.0},
            {"role": "assistant", "content": "between", "timestamp": 2.0},
        ])
        if scenario == "new_time":
            db.append_messages_batch("root", [{"role": "user", "content": "question", "timestamp": 4.0}])
        if scenario in ("compacted", "pagination"):
            db._conn.execute("UPDATE messages SET active=0, compacted=1 WHERE content='question'")
            db._conn.commit()
        content = "question"
        if scenario == "edited":
            row_id = db.get_messages("root")[0]["id"]
            assert db.set_user_message_content("root", row_id, "edited") == 1
            content = "edited"
        target = "root" if scenario == "pagination" else "tip"
        if target == "tip":
            db.create_session("tip", source="compression", parent_session_id="root",
                              model_config={"_branched_from": "root"} if scenario == "branch" else None)
        if scenario not in ("new_time", "branch"):
            db.append_messages_batch(target, [{"role": "user", "content": content, "timestamp": 1.0}])
        if scenario in ("hidden", "pagination"):
            db.append_messages_batch(target, [{
                "role": "assistant", "content": "hidden", "timestamp": 5.0,
                "display_metadata": {"model_only": True},
            }])
        db.append_messages_batch(target, [{"role": "assistant", "content": "end", "timestamp": 6.0}])
        if legacy:
            db._conn.execute("UPDATE messages SET display_identity=NULL, display_order=NULL")
            db._conn.commit()
    finally:
        db.close()

    before = path.read_bytes()
    ro = SessionDB(path, read_only=True)
    try:
        model, full = ro.get_resume_conversations(target)
        for limit in range(len(full) + 3):
            bounded_model, display = ro.get_resume_conversations(target, max_display_messages=limit)
            assert bounded_model == model
            assert display == (full[-limit:] if limit else [])
        if scenario == "pagination":
            full_page = ro.get_messages(target, include_compacted=True)
            for latest in (False, True):
                for offset in range(len(full_page) + 2):
                    for limit in (0, 1, 2, len(full_page) + 2):
                        expected = (full_page[::-1][offset:offset + limit][::-1] if latest
                                    else full_page[offset:offset + limit])
                        assert ro.get_messages(target, include_compacted=True, latest=latest,
                                               offset=offset, limit=limit) == expected
    finally:
        ro.close()
        assert path.read_bytes() == before


@pytest.mark.parametrize("shape", ["current", "null", "no-identity", "no-order", "no-both"])
@pytest.mark.parametrize("session_index", [False, True])
def test_bounded_resume_readonly_schema_capabilities(tmp_path, monkeypatch, shape, session_index):
    """Legacy schema capability precedes index probing; no migration or full payload fetch."""
    seed = tmp_path / "seed.db"
    db = SessionDB(seed)
    try:
        db.create_session("root", source="cli")
        db.append_messages_batch("root", [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": f"{i}:" + "x" * 4096, "timestamp": float(i + 1)}
            for i in range(200)
        ])
        db.create_session("tip", source="compression", parent_session_id="root")
        db.append_messages_batch("tip", [
            {"role": "user", "content": "tip-user", "timestamp": 201.0},
            {"role": "assistant", "content": "tip-answer", "timestamp": 202.0},
        ])
    finally:
        db.close()
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("ATTACH DATABASE ? AS src", (str(seed),))
        excluded = {"no-identity": {"display_identity"}, "no-order": {"display_order"},
                    "no-both": {"display_identity", "display_order"}}.get(shape, set())
        columns = [row[1] for row in conn.execute("PRAGMA src.table_info(messages)") if row[1] not in excluded]
        quoted = ",".join(f'"{col}"' for col in columns)
        conn.execute(f"CREATE TABLE messages AS SELECT {quoted} FROM src.messages")
        conn.execute("CREATE TABLE sessions AS SELECT * FROM src.sessions")
        if session_index:
            conn.execute("CREATE INDEX idx_messages_session_id ON messages(session_id, id)")
        if shape == "null":
            conn.execute("UPDATE messages SET display_identity=NULL, display_order=NULL")
    before = path.read_bytes()
    ro = SessionDB(path, read_only=True)
    try:
        model, full = ro.get_resume_conversations("tip")
        assert [m["content"] for m in model] == ["tip-user", "tip-answer"]
        root = ro.get_messages("root", include_compacted=True)
        assert ro.get_messages("root", include_compacted=True, limit=4, latest=True) == root[-4:]
        assert ro.get_resume_conversations("tip", max_display_messages=0) == (model, [])
        sizes = []
        original = ro._read_all

        def observed(sql, params=()):
            rows = original(sql, params)
            sizes.append(len(rows))
            return rows

        monkeypatch.setattr(ro, "_read_all", observed)
        assert ro.get_resume_conversations("tip", max_display_messages=4) == (model, full[-4:])
        assert sizes and max(sizes) <= 4
    finally:
        ro.close()
        assert path.read_bytes() == before
