"""Real SQLite regressions for shared FTS health probes (#371/#373)."""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_repair import _db_opens_cleanly


def test_healthy_write_probe_rolls_back_every_probe_row(tmp_path):
    path = tmp_path / "state.db"
    with SessionDB(db_path=path) as db:
        db.create_session("healthy", source="cli")
        db.append_message("healthy", role="user", content="original searchable text")
    with sqlite3.connect(path) as conn:
        before = list(conn.iterdump())
    assert _db_opens_cleanly(path) is None
    assert _db_opens_cleanly(path) is None
    with sqlite3.connect(path) as conn:
        assert list(conn.iterdump()) == before
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)


@pytest.mark.parametrize("table", ["messages_fts", "messages_fts_trigram"])
def test_constructor_detects_corrupt_index_without_rebuilding_under_holder(
    tmp_path, monkeypatch, table
):
    path = tmp_path / "state.db"
    with SessionDB(db_path=path) as db:
        db.create_session("corrupt", source="cli")
        db.append_message("corrupt", role="user", content="original searchable text")
    with sqlite3.connect(path) as conn:
        # Damage a real index segment, not the vtable's configuration/header.
        conn.execute(
            f"UPDATE {table}_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF' WHERE id > 10"
        )
        conn.execute(f"SELECT * FROM {table} LIMIT 0")
        # Establish scoped corruption independently of the health helper.
        with pytest.raises(sqlite3.DatabaseError) as corrupt:
            conn.execute(
                f"SELECT rowid FROM {table} WHERE {table} MATCH 'hermes' LIMIT 1"
            ).fetchone()
        assert corrupt.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT_VTAB
        print(
            f"CORRUPTION_WITNESS {table}: MATCH raised {corrupt.value.sqlite_errorname} "
            f"({corrupt.value.sqlite_errorcode}); LIMIT 0 succeeded"
        )
    monkeypatch.setattr(
        SessionDB, "_foreign_state_db_holders", lambda self: [(222, str(path))]
    )
    with SessionDB(db_path=path) as db:
        assert not db._fts_enabled
        assert db._fts_stale
        assert db.get_meta("fts_stale") == "1"
        assert not db._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name LIKE 'messages_fts%'"
        ).fetchone()
        db.append_message("corrupt", role="user", content="canonical write survives")
        assert len(db.get_messages("corrupt")) == 2
    monkeypatch.setattr(SessionDB, "_foreign_state_db_holders", lambda self: [])
    with SessionDB(db_path=path) as db:
        assert db._fts_enabled
        assert not db._fts_stale
        assert (
            db._conn.execute(
                "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'survives'"
            ).fetchone()[0]
            == 1
        )
