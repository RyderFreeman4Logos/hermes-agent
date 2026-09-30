"""One cooperative execution budget, independent of lock entrance/backoff."""
import contextlib
import sqlite3
import time
import pytest

from hermes_state import SessionDB
import hermes_state_repair as repair


def _owner(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with SessionDB(db_path=path) as seed:
        seed.create_session("s", source="cli")
        seed.append_message("s", role="user", content="seed")
        seed._conn.execute("UPDATE messages_fts_data SET block=X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
    with monkeypatch.context() as opening:
        opening.setattr(SessionDB, "_foreign_state_db_holders", lambda self: [(222, str(path))])
        return SessionDB(db_path=path)


@pytest.mark.parametrize("stage", ["forensic", "staging", "scratch", "comparison"])
def test_one_deadline_does_not_restart_at_stage_boundaries(tmp_path, monkeypatch, stage):
    with _owner(tmp_path, monkeypatch) as db:
        before = [tuple(r) for r in db._conn.execute("SELECT * FROM messages")]
        real_clock = repair.time.monotonic
        elapsed = [0]
        monkeypatch.setattr(repair.time, "monotonic", lambda: real_clock() + elapsed[0])
        if stage == "forensic":
            seam = "_publish_backup_bundle"
        elif stage == "staging":
            seam = "_copy_database_snapshot"
        elif stage == "scratch":
            seam = "_strategy_drop_fts_vacuum"
        else:
            seam = "_validate_fts_snapshot"
        original = getattr(repair, seam)
        reached = []
        def exhaust(*args, **kwargs):
            result = original(*args, **kwargs)
            elapsed[0] += 1_000_000
            reached.append(stage)
            return result
        monkeypatch.setattr(repair, seam, exhaust)
        assert not db.retry_deferred_fts_recovery()
        assert reached == [stage]
        assert db._conn is not None and not db._read_conns_closed
        assert [tuple(r) for r in db._conn.execute("SELECT * FROM messages")] == before
        assert db._fts_stale and not db._fts_enabled
        assert not db.db_path.with_name("state.db.repair-scratch").exists()


def test_staging_baseexception_cleans_scratch_and_reopens_owner(tmp_path, monkeypatch):
    with _owner(tmp_path, monkeypatch) as db:
        original = repair._copy_database_snapshot
        before = [tuple(r) for r in db._conn.execute("SELECT * FROM messages")]
        def interrupt(source, destination, **kwargs):
            original(source, destination, **kwargs)
            if kwargs.get("source_connection") is not None:
                raise KeyboardInterrupt("staging interrupted")
        monkeypatch.setattr(repair, "_copy_database_snapshot", interrupt)
        with pytest.raises(KeyboardInterrupt, match="staging interrupted"):
            db.retry_deferred_fts_recovery()
        assert db._conn is not None and not db._read_conns_closed
        assert [tuple(r) for r in db._conn.execute("SELECT * FROM messages")] == before
        assert not db.db_path.with_name("state.db.repair-scratch").exists()


def test_sqlite_progress_callback_interrupts_and_rolls_back(tmp_path):
    path = tmp_path / "progress.db"
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE canonical(value)")
        conn.execute("INSERT INTO canonical VALUES('retained')")
        conn.commit()
    with repair._repair_io_scope(None, deadline=time.monotonic() - 1):
        with repair._repair_conn(path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            with pytest.raises(sqlite3.OperationalError, match="interrupted"):
                conn.execute("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<10000) "
                             "INSERT INTO canonical SELECT x FROM n")
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT * FROM canonical").fetchall() == [('retained',)]
        assert conn.execute("PRAGMA integrity_check").fetchone() == ('ok',)


def test_execution_budget_interrupts_partial_promotion_without_partial_commit(tmp_path):
    source, target = tmp_path / 'source.db', tmp_path / 'target.db'
    with contextlib.closing(sqlite3.connect(source)) as conn:
        conn.execute("CREATE TABLE canonical(value)")
        conn.execute("INSERT INTO canonical VALUES(zeroblob(4*1024*1024))")
        conn.commit()
    with contextlib.closing(sqlite3.connect(target)) as guard:
        guard.execute("CREATE TABLE canonical(value)")
        guard.execute("INSERT INTO canonical VALUES('retained')")
        guard.commit()
        with repair._repair_io_scope(None, deadline=time.monotonic() - 1):
            with pytest.raises(TimeoutError, match="snapshot transfer"):
                repair._copy_database_snapshot(source, target, destination_connection=guard)
        assert guard.execute("SELECT * FROM canonical").fetchall() == [('retained',)]
        assert guard.execute("PRAGMA integrity_check").fetchone() == ('ok',)
