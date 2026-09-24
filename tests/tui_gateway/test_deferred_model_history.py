"""Bounded Desktop hydration salvaged from Benjamin Brumbaugh's PR #106838."""

import threading

import pytest

from agent.replay_cleanup import canonicalize_replay_history
from hermes_state import SessionDB
from tui_gateway import server


@pytest.mark.parametrize("source,omit_messages", [("desktop", True), ("desktop", False), ("tui", True)])
@pytest.mark.parametrize("profile", [None, "work"])
def test_deferred_resume_preserves_model_history_and_db_ownership(tmp_path, monkeypatch, source, omit_messages, profile):
    from hermes_state import resolved_max_resume_messages

    home = tmp_path / "work"
    home.mkdir()
    db = SessionDB(home / "state.db")
    db.create_session("parent", source=source)
    db.append_message("parent", "user", "ancestor-only display", timestamp=100.0)
    db.end_session("parent", "compression")
    db.create_session("tip", source=source, parent_session_id="parent")
    db.append_message("tip", "user", "archived display", timestamp=101.0)
    db.archive_and_compact("tip", [
        {"role": "assistant", "content": "summary", "_compressed_summary": True},
        {"role": "user", "content": "current ask"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "dangling", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
    ])
    expected = canonicalize_replay_history(db.get_messages_as_conversation(
        "tip", repair_alternation=True, include_row_ids=True))
    stored = db.get_session("tip")
    assert stored is not None
    stored_count = stored["message_count"]
    _, display = db.get_resume_conversations("tip", max_display_messages=resolved_max_resume_messages() or None)
    prefix = [m for m in display if m.get("_row_id") not in {row.get("_row_id") for row in expected}]
    display_reads = []
    original_display = db.get_resume_conversations
    original_prefix = db.get_ancestor_display_prefix
    closed = threading.Event()
    built = threading.Event()
    events = []
    original_close = db.close

    def close():
        original_close()
        closed.set()

    def read_display(sid):
        display_reads.append("display")
        return original_display(sid)

    def read_prefix(sid):
        display_reads.append("prefix")
        return original_prefix(sid)

    def acquire(db_path=None, **kwargs):
        assert db_path == home / "state.db"
        return db

    monkeypatch.setattr(db, "close", close)
    monkeypatch.setattr(db, "get_resume_conversations", read_display)
    monkeypatch.setattr(db, "get_ancestor_display_prefix", read_prefix)
    monkeypatch.setattr("hermes_state_registry.acquire", acquire)
    monkeypatch.setattr(server, "_profile_home", lambda p: home if p else None)
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _: str(tmp_path))
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *args: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: built.set())
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload: events.append((kind, payload)))
    sid = None
    try:
        response = server.handle_request({"id": "resume", "method": "session.resume", "params": {
            "session_id": "tip", "source": source, "defer_history": True,
            "omit_messages": omit_messages, **({"profile": profile} if profile else {}),
        }})
        assert response is not None and "error" not in response, response
        sid = response["result"]["session_id"]
        session = server._sessions[sid]
        assert session["resume_history_ready"].wait(5)
        assert built.wait(5)
        model_only = source == "desktop" and omit_messages
        assert display_reads == ([] if model_only else ["display"])
        assert session["history"] == expected
        assert session["display_history_prefix"] == ([] if model_only else prefix)
        count = stored_count if model_only else len(display)
        assert session["resume_message_count"] == count
        assert ("session.resume_progress", {"message_count": count, "phase": "history", "status": "complete"}) in events
        if profile:
            assert closed.wait(5)
            assert db._conn is None
        else:
            assert not closed.is_set()
            assert db.get_session("tip") is not None
    finally:
        if sid is not None:
            server._sessions.pop(sid, None)
        db.close()


@pytest.mark.parametrize("outcome", ["replaced", "failed"])
def test_model_hydration_discards_stale_results_and_closes_owned_db(tmp_path, monkeypatch, outcome):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("stored", source="desktop")
    db.append_message("stored", "user", "model context")
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    original_read, original_close = db.get_messages_as_conversation, db.close
    events, builds = [], []
    old = {"history": [], "history_lock": threading.RLock(), "resume_hydrating": True,
           "resume_history_ready": threading.Event(), "agent_ready": threading.Event(),
           "resume_message_count": 1}
    replacement = {"history": [{"role": "user", "content": "replacement"}]}
    monkeypatch.setitem(server._sessions, "hydrating", old)

    def read(*args, **kwargs):
        started.set()
        assert release.wait(5)
        if outcome == "failed":
            db._read_all("SELECT * FROM deliberately_missing_table")
        return original_read(*args, **kwargs)

    def close():
        original_close()
        closed.set()

    monkeypatch.setattr(db, "get_messages_as_conversation", read)
    monkeypatch.setattr(db, "close", close)
    monkeypatch.setattr(server, "_emit", lambda *args: events.append(args))
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: builds.append(args))
    try:
        server._schedule_resume_hydration("hydrating", "stored", db, close_db=True, model_history_only=True)
        assert started.wait(5)
        assert not closed.is_set()
        if outcome == "replaced":
            server._sessions["hydrating"] = replacement
        release.set()
        assert closed.wait(5)
        assert db._conn is None
        assert builds == []
        if outcome == "replaced":
            assert server._sessions["hydrating"] is replacement
            assert replacement == {"history": [{"role": "user", "content": "replacement"}]}
            assert old["history"] == []
            assert not any(payload.get("status") == "complete" for _, _, payload in events)
        else:
            assert "hydrating" not in server._sessions
            assert old["resume_history_ready"].is_set()
            assert "deliberately_missing_table" in old["resume_history_error"]
    finally:
        release.set()
        assert closed.wait(5)
        server._sessions.pop("hydrating", None)
        original_close()


def test_deferred_hydration_uses_the_bounded_display_window(tmp_path, monkeypatch):
    """A lineage under the resume guard but over the display window must not be fully materialized.

    Deferred resume without ``omit_messages`` hydrates through ``_load_resume_transcript``.
    The tip-only guard lets that path through; the display read must still use the SQL window
    and keep ancestor identity. A 20k-row store is unnecessary: the bound is the configured limit.
    """
    from hermes_state import SessionDB

    limit = 4
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", source="tui")
    db.append_messages_batch("parent", [
        {"role": "user", "content": f"ancestor-{i}", "timestamp": float(i)} for i in range(6)
    ])
    db.end_session("parent", "compression")
    db.create_session("tip", source="tui", parent_session_id="parent")
    db.append_messages_batch("tip", [
        {"role": "user", "content": "ancestor-5", "timestamp": 5.0},
        {"role": "assistant", "content": "tip-answer", "timestamp": 6.0},
    ])
    monkeypatch.setattr("hermes_state.resolved_max_resume_messages", lambda: limit)
    lineage_reads = []
    original_read_all = db._read_all

    def record_read(sql, params=()):
        rows = original_read_all(sql, params)
        if "FROM messages" in sql and "COUNT" not in sql.upper():
            lineage_reads.append(len(rows))
        return rows

    db._read_all = record_read
    session = server._deferred_session_record("tip", cols=80, cwd=str(tmp_path), history=[], lease=None)
    session.update(resume_history_ready=threading.Event(), resume_hydrating=True, resume_message_count=99)
    built = threading.Event()
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: built.set())
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *_a, **_k: None)
    monkeypatch.setitem(server._sessions, "hyd-window", session)
    try:
        server._schedule_resume_hydration("hyd-window", "tip", db)
        assert session["resume_history_ready"].wait(5)
        assert built.wait(5)
        assert session.get("resume_history_error") is None
        assert [m["content"] for m in session["history"]] == ["ancestor-5", "tip-answer"]
        prefix = [m["content"] for m in session["display_history_prefix"]]
        assert prefix == ["ancestor-3", "ancestor-4"]
        assert "ancestor-0" not in prefix
        assert lineage_reads
        assert max(lineage_reads) <= limit
    finally:
        server._sessions.pop("hyd-window", None)
        db.close()
