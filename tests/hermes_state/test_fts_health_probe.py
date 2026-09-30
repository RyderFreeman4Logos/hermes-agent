"""Real SQLite regressions for shared FTS health probes (#371/#373)."""

import contextlib
import sqlite3
import threading

import pytest

from hermes_state import SessionDB
from hermes_state_repair import _db_opens_cleanly


@pytest.mark.parametrize("pooled", [False, True])
def test_header_recovery_preserves_complete_canonical_store(tmp_path, monkeypatch, pooled):
    if pooled:
        monkeypatch.setattr("hermes_state_wal.is_sqlite_wal_reset_vulnerable", lambda **kwargs: False)
    path = tmp_path / "state.db"
    with SessionDB(db_path=path) as db:
        db.create_session("corrupt", source="cli")
        db.append_message("corrupt", role="user", content="searchable seed")
        db._conn.executescript('''
            CREATE TABLE messages_fts_archive(id INTEGER PRIMARY KEY AUTOINCREMENT, payload BLOB);
            INSERT INTO messages_fts_archive VALUES(37, X'00FF');
            DELETE FROM messages_fts_archive;
            INSERT INTO messages_fts_archive VALUES(3, X'00FF');
            CREATE INDEX plugin_index ON messages_fts_archive(payload);
            CREATE VIEW plugin_view AS SELECT * FROM messages_fts_archive;
        ''')
        db._conn.execute("UPDATE messages_fts_data SET block=X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
        db.append_message("corrupt", role="user", content="canonical survives")
    with monkeypatch.context() as opening:
        opening.setattr(SessionDB, "_foreign_state_db_holders", lambda self: [(222, str(path))])
        reopened = SessionDB(db_path=path)
    with reopened as db:
        if pooled:
            with db._read_ctx() as conn:
                assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 2
            assert db._read_pool.qsize() == 1
        before = {
            name: [tuple(row) for row in db._conn.execute(f'SELECT * FROM "{name}"')]
            for (name,) in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if not name.startswith("messages_fts") and name != "state_meta"
        }
        inode = path.stat().st_ino
        assert db.retry_deferred_fts_recovery()
        assert path.stat().st_ino == inode
        assert db.get_meta("fts_stale") is None
        for name, rows in before.items():
            assert [tuple(row) for row in db._conn.execute(f'SELECT * FROM "{name}"')] == rows
        assert tuple(db._conn.execute("SELECT * FROM messages_fts_archive").fetchone()) == (3, b'\x00\xff')
        assert db._conn.execute("SELECT seq FROM sqlite_sequence WHERE name='messages_fts_archive'").fetchone()[0] == 37
        assert db._conn.execute("SELECT payload FROM plugin_view").fetchone()[0] == b'\x00\xff'
        assert db.search_messages("survives")
        db.append_message("corrupt", role="user", content="post repair indexed")
        assert db.search_messages("indexed")


@pytest.mark.parametrize("failure", [
    "promotion", "canonical_row", "canonical_schema", "implicit_rowids", "peer", "reader",
    "lifecycle_lock", "repair_lock", "foreign", "deleted_wal", "generation", "generation_handoff",
])
def test_header_recovery_refuses_or_rolls_back_without_losing_store(tmp_path, monkeypatch, caplog, failure):
    import hermes_state_repair as repair
    from hermes_cli import sqlite_safe_read as safe
    from hermes_state_common import _FTS_TRIGGERS

    path = tmp_path / "state.db"
    monkeypatch.setattr("hermes_state_wal.is_sqlite_wal_reset_vulnerable", lambda **kwargs: False)
    with SessionDB(db_path=path) as seed:
        seed.create_session("s", source="cli")
        seed.append_message("s", role="user", content="seed")
        seed._conn.execute("CREATE TABLE plugin(id INTEGER PRIMARY KEY, payload BLOB)")
        seed._conn.execute("INSERT INTO plugin VALUES(3, X'00FF')")
        seed._conn.execute("UPDATE messages_fts_data SET block=X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
        seed.append_message("s", role="user", content="canonical survives")
    with monkeypatch.context() as opening:
        opening.setattr(SessionDB, "_foreign_state_db_holders", lambda self: [(222, str(path))])
        db = SessionDB(db_path=path)
    with db, contextlib.ExitStack() as cleanup:
        if failure == "implicit_rowids":
            db._conn.execute("CREATE TABLE unindexed(payload TEXT DEFAULT 'WITHOUT ROWID')")
            db._conn.execute("INSERT INTO unindexed(rowid, payload) VALUES(19, 'keep rowid')")
        before = {}
        queries = {}
        without_rowid = {r[1]: bool(r[4]) for r in db._conn.execute("PRAGMA table_list")}
        for (name,) in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            if name not in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"):
                quoted = '"' + name.replace('"', '""') + '"'
                fields = "*" if without_rowid[name] else "rowid,*"
                count = len(db._conn.execute(f"SELECT {fields} FROM {quoted} LIMIT 0").description)
                queries[name] = f"SELECT {fields} FROM {quoted} ORDER BY " + ",".join(str(i) for i in range(1, count + 1))
                before[name] = [tuple(r) for r in db._conn.execute(queries[name])]
        master = [tuple(r) for r in db._conn.execute("SELECT * FROM sqlite_master ORDER BY rowid")]
        inode = path.stat().st_ino
        reached = []
        if failure == "peer":
            with monkeypatch.context() as opening:
                opening.setattr(SessionDB, "_foreign_state_db_holders", lambda self: [(222, str(path))])
                peer = cleanup.enter_context(SessionDB(db_path=path))
            assert peer._conn is not db._conn
            # The peer's startup legitimately advances the shared deferral breadcrumb.
            before["state_meta"] = [tuple(r) for r in db._conn.execute(queries["state_meta"])]
        elif failure == "reader":
            reader = cleanup.enter_context(db._read_ctx())
            assert reader is not db._conn
            reader.execute("SELECT * FROM plugin").fetchall()
        elif failure == "lifecycle_lock":
            # Include an idle owned reader: its close must not wait on the lifecycle lock.
            with db._read_ctx() as reader:
                reader.execute("SELECT * FROM plugin").fetchall()
            ready, release = threading.Event(), threading.Event()
            def holder():
                with safe._live_lock:
                    ready.set()
                    assert release.wait(10)
            thread = threading.Thread(target=holder)
            thread.start()
            assert ready.wait(10)
            cleanup.callback(thread.join, 10)
            cleanup.callback(release.set)
        elif failure == "repair_lock":
            handle = cleanup.enter_context(open(str(path) + ".repair.lock", "a+b"))
            repair._try_lock_nonblocking(handle)
        elif failure in ("foreign", "deleted_wal"):
            suffix = "-wal (deleted)" if failure == "deleted_wal" else "-wal"
            monkeypatch.setattr(db, "_foreign_state_db_holders", lambda: [(4242, str(path) + suffix)])
        elif failure == "generation":
            db._conn.execute(f"PRAGMA application_id={db._db_file_application_id + 1}")
            db._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        elif failure in ("canonical_row", "canonical_schema"):
            original = repair._strategy_drop_fts_vacuum
            def bad_scratch(conn):
                original(conn)
                reached.append("scratch")
                conn.execute("DELETE FROM plugin" if failure == "canonical_row" else "DROP TABLE plugin")
            monkeypatch.setattr(repair, "_strategy_drop_fts_vacuum", bad_scratch)
        elif failure in ("promotion", "generation_handoff"):
            original_copy = repair._copy_database_snapshot
            def interrupted_copy(source, destination, **kwargs):
                if failure == "generation_handoff" and kwargs.get("source_connection") is not None:
                    original_copy(source, destination, **kwargs)
                    kwargs["source_connection"].execute("PRAGMA application_id=12345")
                    reached.append("generation")
                elif failure == "promotion" and kwargs.get("destination_connection") is not None:
                    with repair._repair_conn(source) as conn:
                        def interrupt(*_args):
                            reached.append("promotion")
                            raise OSError("injected partially copied promotion")
                        conn.backup(kwargs["destination_connection"], pages=1, progress=interrupt)
                else:
                    original_copy(source, destination, **kwargs)
            monkeypatch.setattr(repair, "_copy_database_snapshot", interrupted_copy)
        recovered = db.retry_deferred_fts_recovery()
        if failure == "implicit_rowids":
            rows = [tuple(r) for r in db._conn.execute(queries["unindexed"])]
            assert rows == before["unindexed"], {"before": before["unindexed"], "after": rows}
        assert recovered is False
        assert db._fts_stale and not db._fts_enabled
        assert path.stat().st_ino == inode
        # Refusal/backoff is still authoritative; don't silently consume schema-repair's ledger.
        assert db._fts_stale_retry_interval > 0
        assert not path.with_name(path.name + ".repair-attempts.json").exists()
        if failure == "lifecycle_lock":
            release.set()
            thread.join(10)
            assert not thread.is_alive()
        # Generation refusal can intentionally leave the owner detached; inspect read-only.
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            assert [tuple(r) for r in conn.execute("SELECT * FROM sqlite_master ORDER BY rowid")] == master
            for name, rows in before.items():
                after = [tuple(r) for r in conn.execute(queries[name])]
                if failure in ("foreign", "deleted_wal") and name == "state_meta":
                    assert dict((r[1], r[2]) for r in after)["fts_stale"] == "1"
                else:
                    assert after == rows
            assert not conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN (" + ",".join("?" for _ in _FTS_TRIGGERS) + ")",
                _FTS_TRIGGERS,
            ).fetchall()
        if failure == "promotion":
            assert reached == ["promotion"]
            assert "injected partially copied promotion" in caplog.text
        if failure.startswith("canonical_"):
            assert reached == ["scratch"] and "changed canonical" in caplog.text
        if failure == "implicit_rowids":
            assert "changed canonical rows in unindexed" in caplog.text
        if failure == "generation_handoff":
            assert reached == ["generation"] and "generation changed" in caplog.text
            assert db._db_replaced and db._conn is None


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
