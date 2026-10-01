"""Shared application deadline, with synchronous safety settlement after expiry."""
import contextlib
import sqlite3
import time
import threading
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


@pytest.mark.parametrize("entry", ["startup", "retry"])
def test_admission_expiry_precedes_recovery_mutation(tmp_path, monkeypatch, entry):
    with _owner(tmp_path, monkeypatch) as db:
        clock = [100.0]
        monkeypatch.setattr(repair.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(repair, "_repair_snapshot_timeout_seconds", lambda path: 1.0)
        statements = []
        db._conn.set_trace_callback(statements.append)
        def holders():
            clock[0] += 5.0
            return []
        monkeypatch.setattr(db, "_foreign_state_db_holders", holders)
        if entry == "retry":
            result = db.retry_deferred_fts_recovery()
        else:
            result = db._recover_stale_fts(db._conn.cursor(), legacy=False)
        assert not result
        assert not any(s.startswith("BEGIN IMMEDIATE") for s in statements)
        assert db._fts_stale and db._conn is not None
        db._conn.set_trace_callback(None)


def test_normal_rebuild_sql_obeys_application_deadline(tmp_path, monkeypatch):
    with SessionDB(db_path=tmp_path / "normal.db") as db:
        db.create_session("s", source="cli")
        for _ in range(100):
            db.append_message("s", role="user", content="searchable canonical seed")
        db._drop_all_fts_triggers(db._conn.cursor())
        db._conn.execute("INSERT OR REPLACE INTO state_meta VALUES('fts_stale','1')")
        db._conn.commit()
        db._fts_stale, db._fts_enabled = True, False
        rows = [tuple(r) for r in db._conn.execute("SELECT * FROM messages")]
        clock = [100.0]
        monkeypatch.setattr(repair.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(repair, "_repair_snapshot_timeout_seconds", lambda path: 1.0)
        def expire_at_mutation(sql):
            if sql == "BEGIN IMMEDIATE;":
                clock[0] = 105.0
        db._conn.set_trace_callback(expire_at_mutation)
        assert not db.retry_deferred_fts_recovery()
        db._conn.set_trace_callback(None)
        assert db._fts_stale and not db._conn.in_transaction
        assert [tuple(r) for r in db._conn.execute("SELECT * FROM messages")] == rows
        assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("lock_name", ["_lock", "_read_conns_lock"])
def test_retry_writer_lock_uses_remaining_budget(tmp_path, monkeypatch, lock_name):
    with _owner(tmp_path, monkeypatch) as db:
        clock = [100.0]
        monkeypatch.setattr(repair.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(repair, "_repair_snapshot_timeout_seconds", lambda path: 1.0)
        real_lock = getattr(db, lock_name)
        entered, release = threading.Event(), threading.Event()
        observed = []
        class Lock:
            def acquire(self, *args, **kwargs):
                observed.append(kwargs.get("timeout"))
                clock[0] = 105.0
                return False
            def release(self):
                real_lock.release()
            def __enter__(self):
                entered.set()
                assert release.wait(2)
                return self
            def __exit__(self, *args):
                pass
        setattr(db, lock_name, Lock())
        results = []
        worker = threading.Thread(target=lambda: results.append(db.retry_deferred_fts_recovery()))
        try:
            worker.start()
            worker.join(0.1)
        finally:
            release.set()
            worker.join(2)
            setattr(db, lock_name, real_lock)
        assert not worker.is_alive()
        assert observed == [4.0]
        assert results == [False]


def test_filename_collision_loop_stops_at_shared_deadline(tmp_path, monkeypatch):
    path = tmp_path / "collision.db"
    path.write_bytes(b"forensic image")
    original = type(path).exists
    clock, attempts = [100.0], []
    def collisions(p):
        if ".malformed-backup-" in p.name:
            attempts.append(p)
            clock[0] += 1.0
            if len(attempts) > 8:
                raise AssertionError("collision loop ignored expiry")
            return True
        return original(p)
    monkeypatch.setattr(type(path), "exists", collisions)
    monkeypatch.setattr(repair.time, "monotonic", lambda: clock[0])
    with repair._repair_io_scope(None, deadline=103.0):
        with pytest.raises(TimeoutError, match="collision"):
            repair._backup_db_file(path)
    assert len(attempts) == 3


def test_flock_admission_consumes_elapsed_time_without_restart(tmp_path, monkeypatch):
    import fcntl
    import hermes_state_schema as schema
    with _owner(tmp_path, monkeypatch) as db:
        clock = [100.0]
        monkeypatch.setattr(repair.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(repair.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
        monkeypatch.setattr(repair, "_repair_snapshot_timeout_seconds", lambda path: 1.0)
        original = schema.fts_rebuild_admission
        @contextlib.contextmanager
        def slow_entrance(*args, **kwargs):
            clock[0] += 3.0
            with original(*args, **kwargs) as admitted:
                yield admitted
        monkeypatch.setattr(schema, "fts_rebuild_admission", slow_entrance)
        with open(str(db.db_path) + ".fts_rebuild.lock", "a+b") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            assert not db._recover_stale_fts(db._conn.cursor(), legacy=False)
        assert clock[0] <= 104.0


def test_scratch_inherits_admission_and_normal_work_budget(tmp_path, monkeypatch):
    with _owner(tmp_path, monkeypatch) as db:
        clock, deadlines, rebuilt = [100.0], [], []
        monkeypatch.setattr(repair.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(repair, "_repair_snapshot_timeout_seconds", lambda path: 1.0)
        def holders():
            clock[0] += 2.0
            return []
        monkeypatch.setattr(db, "_foreign_state_db_holders", holders)
        original = repair._copy_database_snapshot
        def consume(*args, **kwargs):
            deadlines.append(repair._repair_deadline.get())
            result = original(*args, **kwargs)
            clock[0] += 2.1
            return result
        monkeypatch.setattr(repair, "_copy_database_snapshot", consume)
        monkeypatch.setattr(repair, "_strategy_drop_fts_vacuum", lambda conn: rebuilt.append(True))
        assert not db.retry_deferred_fts_recovery()
        assert deadlines == [104.0] and not rebuilt
        assert db._conn is not None and db._fts_stale


@pytest.mark.parametrize("committed", [False, True])
def test_over_budget_settlement_keeps_custody_and_honest_outcome(tmp_path, monkeypatch, committed):
    from hermes_cli import sqlite_safe_read as safe
    with _owner(tmp_path, monkeypatch) as db:
        before = [tuple(r) for r in db._conn.execute("SELECT * FROM messages")]
        clock, expired, results, errors, reopened, cjk = [100.0], [], [], [], [], []
        entered, release, progressed = threading.Event(), threading.Event(), threading.Event()
        monkeypatch.setattr(repair.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(repair, "_repair_snapshot_timeout_seconds", lambda path: 1.0)
        seam = "_restore_journal_mode_after_repair" if committed else "_validate_fts_snapshot"
        original = getattr(repair, seam)
        def exhaust(*args, **kwargs):
            result = original(*args, **kwargs)
            expired.append(True)
            clock[0] = 105.0
            return result
        monkeypatch.setattr(repair, seam, exhaust)
        unlink = repair._unlink_db_triple
        def settle(path):
            if expired and path.name.endswith(".repair-scratch"):
                entered.set()
                assert release.wait(10)
            return unlink(path)
        monkeypatch.setattr(repair, "_unlink_db_triple", settle)
        opening = db._open_writer_conn
        def reopen(**kwargs):
            reopened.append(repair._repair_deadline.get())
            return opening(**kwargs)
        monkeypatch.setattr(db, "_open_writer_conn", reopen)
        monkeypatch.setattr(db, "_ensure_fts_cjk_schema", lambda cursor: cjk.append(True))
        def recover():
            try:
                results.append(db.retry_deferred_fts_recovery())
            except BaseException as exc:
                errors.append(repr(exc))
        def unrelated():
            try:
                with pytest.raises(safe.LiveConnectionError):
                    safe.connect_tracked(db.db_path, timeout=0)
                with pytest.raises(safe.LiveConnectionError):
                    with safe.offline_file_access(db.db_path):
                        pytest.fail("settlement lost main custody")
                with contextlib.closing(safe.connect_tracked(tmp_path / "unrelated.db", timeout=0)) as conn:
                    conn.execute("CREATE TABLE progress(value)")
                progressed.set()
            except BaseException as exc:
                errors.append(repr(exc))
        owner, peer = threading.Thread(target=recover), threading.Thread(target=unrelated)
        owner.start()
        try:
            assert entered.wait(10)
            assert db._conn is None and db._read_conns_closed and owner.is_alive()
            peer.start()
            assert progressed.wait(2)
        finally:
            release.set()
            owner.join(10)
            if peer.ident is not None:
                peer.join(10)
        assert not owner.is_alive() and not peer.is_alive() and not errors, errors
        assert results == [committed] and reopened == [None] and not cjk
        assert db._conn is not None and not db._read_conns_closed
        assert [tuple(r) for r in db._conn.execute("SELECT * FROM messages")] == before
        assert db._fts_stale is (not committed)
        assert not db.db_path.with_name("state.db.repair-scratch").exists()
        if committed:
            assert db.search_messages("seed")


@pytest.mark.parametrize("entry", ["startup", "retry"])
def test_committed_normal_recovery_skips_expired_optional_tail(tmp_path, monkeypatch, entry):
    with SessionDB(db_path=tmp_path / "tail.db") as db:
        db.create_session("s", source="cli")
        db.append_message("s", role="user", content="indexed seed")
        db._fts_stale, db._fts_enabled = True, False
        clock, cjk = [100.0], []
        monkeypatch.setattr(repair.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(repair, "_repair_snapshot_timeout_seconds", lambda path: 1.0)
        original = db._recover_stale_fts_locked
        def committed(*args, **kwargs):
            result = original(*args, **kwargs)
            assert result
            clock[0] = 105.0
            return result
        monkeypatch.setattr(db, "_recover_stale_fts_locked", committed)
        monkeypatch.setattr(db, "_ensure_fts_cjk_schema", lambda cursor: cjk.append(True))
        if entry == "startup":
            db._init_fts(db._conn.cursor())
        else:
            assert db.retry_deferred_fts_recovery()
        assert not cjk and not db._fts_stale
        assert db.search_messages("indexed")


def test_budgeted_repair_sql_does_not_add_sequential_busy_waits(tmp_path):
    with repair._repair_io_scope(None, deadline=time.monotonic() + 10):
        with repair._repair_conn(tmp_path / "busy.db") as conn:
            assert conn.execute("PRAGMA busy_timeout").fetchone() == (0,)


@pytest.mark.parametrize("stage", ["inventory", "prune"])
def test_optional_backup_work_stops_before_expired_mutation(tmp_path, monkeypatch, stage):
    path = tmp_path / "scan.db"
    victim = tmp_path / "scan.db.malformed-backup-1"
    victim.write_bytes(b"retained backup")
    monkeypatch.setattr(repair.time, "monotonic", lambda: 101.0)
    with repair._repair_io_scope(None, deadline=100.0):
        with pytest.raises(TimeoutError):
            if stage == "inventory":
                repair._existing_malformed_backups(path)
            else:
                repair._prune_malformed_backups(path, keep=0)
    assert victim.read_bytes() == b"retained backup"


def test_post_promotion_expiry_skips_optional_journal_work(tmp_path, monkeypatch):
    import hermes_state_wal as wal
    with _owner(tmp_path, monkeypatch) as db:
        clock, optional = [100.0], []
        monkeypatch.setattr(repair.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(repair, "_repair_snapshot_timeout_seconds", lambda path: 1.0)
        copy = repair._copy_database_snapshot
        def promote(*args, **kwargs):
            result = copy(*args, **kwargs)
            if kwargs.get("destination_connection") is not None:
                clock[0] = 105.0
            return result
        monkeypatch.setattr(repair, "_copy_database_snapshot", promote)
        apply = wal.apply_wal_with_fallback
        def policy(*args, **kwargs):
            if repair._repair_deadline.get() is not None:
                optional.append(True)
            return apply(*args, **kwargs)
        monkeypatch.setattr(wal, "apply_wal_with_fallback", policy)
        assert db.retry_deferred_fts_recovery()
        assert not optional and not db._fts_stale
        assert db.search_messages("seed")


def test_normal_expiry_before_script_does_not_start_mutation(tmp_path, monkeypatch):
    with SessionDB(db_path=tmp_path / "premutation.db") as db:
        db._fts_stale, db._fts_enabled = True, False
        clock, statements = [100.0], []
        monkeypatch.setattr(repair.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(repair, "_repair_snapshot_timeout_seconds", lambda path: 1.0)
        probe = db._fts_table_probe
        def expire(*args):
            result = probe(*args)
            clock[0] = 105.0
            return result
        monkeypatch.setattr(db, "_fts_table_probe", expire)
        db._conn.set_trace_callback(statements.append)
        assert not db.retry_deferred_fts_recovery()
        db._conn.set_trace_callback(None)
        assert not any(s.startswith(("BEGIN IMMEDIATE", "DROP TRIGGER", "DROP TABLE")) for s in statements)
        assert db._fts_stale and not db._conn.in_transaction
