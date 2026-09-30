"""Canonical exemptions must prove ownership, not just match reserved names."""
import contextlib

import pytest

from hermes_state import SessionDB
import hermes_state_repair as repair


def _stale_owner(tmp_path, monkeypatch, *, reserved_table=False):
    path = tmp_path / "state.db"
    monkeypatch.setattr("hermes_state_wal.is_sqlite_wal_reset_vulnerable", lambda **kwargs: False)
    with SessionDB(db_path=path) as seed:
        seed.create_session("s", source="cli")
        seed.append_message("s", role="user", content="seed")
        if reserved_table:
            seed._conn.execute("DROP VIEW messages_fts_src")
            seed._conn.execute("CREATE TABLE messages_fts_src(id INTEGER PRIMARY KEY, content, tool_name, tool_calls)")
            seed._conn.execute("INSERT INTO messages_fts_src VALUES(900,'canonical payload','plugin','{}')")
        seed._conn.execute("UPDATE messages_fts_data SET block=X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
    with monkeypatch.context() as opening:
        opening.setattr(SessionDB, "_foreign_state_db_holders", lambda self: [(222, str(path))])
        return SessionDB(db_path=path)


@pytest.mark.parametrize("mutation", ["change", "delete", "insert", "unchanged"])
def test_null_meta_keys_are_canonical(tmp_path, monkeypatch, mutation):
    with _stale_owner(tmp_path, monkeypatch) as db:
        db._conn.execute("INSERT INTO state_meta(rowid,key,value) VALUES(900,NULL,'canonical')")
        original = repair._strategy_drop_fts_vacuum
        reached = []
        def mutate(conn):
            original(conn)
            sql = {"change": "UPDATE state_meta SET value='changed' WHERE key IS NULL",
                   "delete": "DELETE FROM state_meta WHERE key IS NULL",
                   "insert": "INSERT INTO state_meta(rowid,key,value) VALUES(901,NULL,'extra')",
                   "unchanged": "SELECT 1"}[mutation]
            conn.execute(sql)
            reached.append(True)
        monkeypatch.setattr(repair, "_strategy_drop_fts_vacuum", mutate)
        assert db.retry_deferred_fts_recovery() is (mutation == "unchanged")
        assert reached == [True]
        assert [tuple(r) for r in db._conn.execute("SELECT rowid,key,value FROM state_meta WHERE key IS NULL")] == [(900, None, "canonical")]


def test_reserved_name_does_not_authorize_canonical_table_deletion(tmp_path, monkeypatch):
    with _stale_owner(tmp_path, monkeypatch, reserved_table=True) as db:
        assert not db.retry_deferred_fts_recovery()
        assert db._conn.execute("SELECT type FROM sqlite_master WHERE name='messages_fts_src'").fetchone()[0] == "table"
        assert tuple(db._conn.execute("SELECT * FROM messages_fts_src").fetchone()) == (900, "canonical payload", "plugin", "{}")
        assert db._fts_stale and not db._fts_enabled


@pytest.mark.parametrize("ddl", [
    "CREATE VIEW messages_fts_src AS SELECT id, content, tool_name, tool_calls FROM messages",
    "CREATE TABLE messages_fts_data(id INTEGER PRIMARY KEY, canonical TEXT)",
    "CREATE VIRTUAL TABLE plugin USING fts5(payload)",
])
def test_ambiguous_owned_or_extension_objects_refuse_snapshot(tmp_path, ddl):
    # A normal canonical virtual table may have hidden/external state: do not reconstruct it.
    import sqlite3
    with contextlib.closing(sqlite3.connect(tmp_path / "probe.db")) as conn:
        conn.execute("CREATE TABLE messages(id,content,tool_name,tool_calls)")
        conn.execute(ddl)
        with pytest.raises(ValueError):
            repair._validate_fts_snapshot(conn, conn)
