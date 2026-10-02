"""Real SQLite admission is bounded; already admitted healthy work may finish."""

import fcntl
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

import hermes_state
import hermes_state_fts
from hermes_state import SessionDB
from hermes_state_errors import SessionCompressionInProgressError


@pytest.fixture
def database(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db._execute_write(lambda c: c.execute("CREATE TABLE witness(value INTEGER)"))
    yield db
    conn = db._conn
    if conn is not None:
        assert not conn.in_transaction
        assert db._lock.acquire(blocking=False)
        db._lock.release()
        with Path(str(db.db_path) + ".write.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
    db.close()
    assert db._conn is None
    if conn is not None:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    clock.sleep = lambda delay: setattr(clock, "now", clock.now + delay)
    monkeypatch.setattr(hermes_state, "time", clock)
    monkeypatch.setattr(hermes_state_fts, "time", clock)
    return clock


def fts_error():
    return sqlite3.DatabaseError('fts5: corrupt structure record for table "messages_fts"')


def triggers(db):
    return db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE '%fts%' ORDER BY name"
    ).fetchall()


def invoke(db, recovery, deadline):
    if recovery:
        assert not db._enter_fts_fail_open(fts_error(), deadline=deadline)
    else:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            db._execute_write(lambda c: c.execute("INSERT INTO witness VALUES(1)"), deadline=deadline)


@pytest.mark.parametrize("recovery", [False, True])
def test_local_mutex_wait_does_not_outlive_admission(database, recovery):
    ready, release = threading.Event(), threading.Event()

    def holder():
        with database._lock:
            ready.set()
            release.wait(3)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert ready.wait(3)
        start = time.monotonic()
        invoke(database, recovery, start + 0.05)
        elapsed = time.monotonic() - start
        assert elapsed < 2, f"mutex ignored 0.05s admission budget: {elapsed}"
        assert not release.is_set()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("prior_ms", [1000, 731])
def test_sqlite_busy_wait_uses_remaining_budget_and_restores(database, monkeypatch, recovery, prior_ms):
    conn = database._conn
    conn.execute(f"PRAGMA busy_timeout={prior_ms}")
    original = conn.execute
    observed = []

    def execute(sql, *args, **kwargs):
        if sql == "BEGIN IMMEDIATE":
            observed.append(original("PRAGMA busy_timeout").fetchone()[0])
        return original(sql, *args, **kwargs)

    monkeypatch.setattr(conn, "execute", execute)
    rival = sqlite3.connect(database.db_path, isolation_level=None)
    try:
        rival.execute("BEGIN IMMEDIATE")
        start = time.monotonic()
        invoke(database, recovery, start + 0.05)
        assert time.monotonic() - start < 2
    finally:
        rival.rollback()
        rival.close()
    assert observed and all(0 <= ms <= 50 for ms in observed), observed
    assert original("PRAGMA busy_timeout").fetchone()[0] == prior_ms
    assert not conn.in_transaction
    assert original("SELECT count(*) FROM witness").fetchone()[0] == 0
    database._execute_write(lambda c: c.execute("INSERT INTO witness VALUES(2)"))
    assert original("SELECT value FROM witness").fetchone()[0] == 2


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("stage", ["expired", "reopen", "begin"])
def test_expiry_fences_every_transaction_admission(database, clock, monkeypatch, recovery, stage):
    conn = database._conn
    before = triggers(database)
    opened, begun = [], []
    original_open = database._open_writer_conn
    original_execute = conn.execute

    def late_begin(sql, *args, **kwargs):
        result = original_execute(sql, *args, **kwargs)
        if sql == "BEGIN IMMEDIATE":
            begun.append(clock.now)
            clock.now = 24
        return result

    def late_open(**kwargs):
        opened.append(clock.now)
        result = original_open(**kwargs)
        clock.now = 24
        return result

    if stage in ("expired", "reopen"):
        database.close()
        monkeypatch.setattr(database, "_open_writer_conn", late_open)
        if stage == "expired":
            clock.now = 24
    else:
        monkeypatch.setattr(conn, "execute", late_begin)
    invoke(database, recovery, 20)
    assert opened == ([] if stage != "reopen" else [0])
    assert begun == ([0] if stage == "begin" else [])
    if database._conn is not None:
        assert not database._conn.in_transaction
        assert triggers(database) == before
        assert database._conn.execute("SELECT count(*) FROM witness").fetchone()[0] == 0
        assert database._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1000


@pytest.mark.parametrize("stage", ["callback", "recovery"])
def test_expired_recovery_never_commits(database, clock, monkeypatch, stage):
    before = triggers(database)
    calls = []
    if stage == "callback":
        def callback(conn):
            calls.append(clock.now)
            conn.execute("INSERT INTO witness VALUES(1)")
            clock.now = 24
            raise fts_error()
        with pytest.raises(sqlite3.DatabaseError):
            database._execute_write(callback, deadline=20)
        assert calls == [0]
    else:
        original = database._drop_all_fts_triggers
        def late_drop(cursor):
            original(cursor)
            clock.now = 24
        monkeypatch.setattr(database, "_drop_all_fts_triggers", late_drop)
        assert not database._enter_fts_fail_open(fts_error(), deadline=20)
    assert triggers(database) == before
    assert not database._fts_stale
    assert not database._conn.in_transaction
    assert database._conn.execute("SELECT count(*) FROM witness").fetchone()[0] == 0


@pytest.mark.parametrize("phase", ["healthy", "callback", "commit", "rollback"])
@pytest.mark.parametrize("fault", ["ioerr", "cancel"])
def test_transaction_settlement_restores_connection(database, clock, monkeypatch, phase, fault):
    conn = database._conn
    conn.execute("PRAGMA busy_timeout=731")
    calls = []
    error = sqlite3.OperationalError("disk I/O error") if fault == "ioerr" else KeyboardInterrupt()
    if isinstance(error, sqlite3.Error):
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR | (1 << 8)
    original_rollback = conn.rollback
    def fail():
        raise error
    def failed_rollback():
        original_rollback()
        raise sqlite3.OperationalError("database is locked")
    def callback(c):
        calls.append(clock.now)
        c.execute("INSERT INTO witness VALUES(7)")
        clock.now = 24
        if phase in ("callback", "rollback"):
            fail()
    if phase == "commit":
        monkeypatch.setattr(conn, "commit", fail)
    if phase == "rollback":
        monkeypatch.setattr(conn, "rollback", failed_rollback)
    if phase == "healthy":
        database._execute_write(callback, deadline=20)
    else:
        expected = type(error)
        caught = None
        try:
            database._execute_write(callback, deadline=20)
        except BaseException as exc:
            caught = exc
        assert isinstance(caught, expected), repr(caught)
    assert calls == [0]
    assert conn.execute("SELECT count(*) FROM witness").fetchone()[0] == (phase == "healthy")
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 731
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert not conn.in_transaction


@pytest.mark.parametrize("kind", ["lock", "compression", "no-more-rows"])
def test_retry_admission_does_not_renew_any_owner_budget(database, clock, monkeypatch, kind):
    compression = kind == "compression"
    error = {
        "lock": sqlite3.OperationalError("database is locked"),
        "compression": SessionCompressionInProgressError("being compressed"),
        "no-more-rows": sqlite3.InterfaceError("no more rows available"),
    }[kind]
    calls, begun = [], []
    original = database._conn.execute
    def execute(sql, *args, **kwargs):
        result = original(sql, *args, **kwargs)
        if sql == "BEGIN IMMEDIATE":
            begun.append(clock.now)
            if len(begun) == 2:
                clock.now = 3 if compression else 21
        return result
    def sleep(delay):
        assert database._lock.acquire(blocking=False)
        database._lock.release()
        with Path(str(database.db_path) + ".write.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        clock.now += delay
    def callback(c):
        calls.append(clock.now)
        c.execute("INSERT INTO witness VALUES(1)")
        if len(calls) == 1:
            raise error
    clock.sleep = sleep
    database._COMPRESSION_BUSY_WAIT_S = 2
    monkeypatch.setattr(hermes_state.random, "uniform", lambda *_args: 0.25)
    monkeypatch.setattr(database._conn, "execute", execute)
    with pytest.raises(type(error)) as caught:
        database._execute_write(callback, deadline=20)
    assert caught.value is error
    assert calls == [0]
    assert begun == [0, 0.25]
    assert database._conn.execute("SELECT count(*) FROM witness").fetchone()[0] == 0


def test_zero_patience_is_one_nonwaiting_attempt(database, monkeypatch):
    seen = []
    original = database._conn.execute
    def execute(sql, *args, **kwargs):
        if sql == "BEGIN IMMEDIATE":
            seen.append(original("PRAGMA busy_timeout").fetchone()[0])
        return original(sql, *args, **kwargs)
    monkeypatch.setattr(database._conn, "execute", execute)
    database._execute_write(lambda c: c.execute("INSERT INTO witness VALUES(1)"), patience_s=0)
    assert seen == [0]
    assert original("PRAGMA busy_timeout").fetchone()[0] == 1000


def test_timeout_restoration_failure_never_replays_a_committed_write(database, monkeypatch):
    conn = database._conn
    conn.execute("PRAGMA busy_timeout=731")
    original = conn.execute
    calls, failed = [], []
    secondary = sqlite3.OperationalError("restore secondary")
    def execute(sql, *args, **kwargs):
        if sql == "PRAGMA busy_timeout=731" and calls and not conn.in_transaction and not failed:
            failed.append(True)
            raise secondary
        return original(sql, *args, **kwargs)
    def callback(c):
        calls.append(1)
        c.execute("INSERT INTO witness VALUES(7)")
    monkeypatch.setattr(conn, "execute", execute)
    with pytest.raises(RuntimeError) as caught:
        database._execute_write(callback)
    assert caught.value.__cause__ is secondary
    assert failed == [True]
    assert calls == [1]
    assert original("SELECT count(*) FROM witness").fetchone()[0] == 1
    assert not conn.in_transaction
    # The injected failure left the setting unchanged; restore for the next owner.
    original("PRAGMA busy_timeout=731")


def _sqlite(message, code):
    error = sqlite3.OperationalError(message)
    error.sqlite_errorcode = code
    error.sqlite_errorname = "SQLITE_IOERR" if code == sqlite3.SQLITE_IOERR else "SQLITE_ERROR"
    return error


@pytest.mark.parametrize("seam", ["rollback", "restore"])
@pytest.mark.parametrize("kind", ["cancel", "exit", "io", "sql"])
def test_cleanup_keeps_primary_identity_and_records_cleanup(database, monkeypatch, seam, kind):
    conn = database._conn
    conn.execute("PRAGMA busy_timeout=731")
    primary = {
        "cancel": KeyboardInterrupt("cancel-primary"),
        "exit": SystemExit("exit-primary"),
        "io": _sqlite("disk I/O error", sqlite3.SQLITE_IOERR),
        "sql": _sqlite("no such column: original", sqlite3.SQLITE_ERROR),
    }[kind]
    secondary = sqlite3.OperationalError("cleanup secondary")
    original_rollback, original_execute = conn.rollback, conn.execute
    calls, faults = [], []

    def rollback():
        original_rollback()
        faults.append("rollback")
        raise secondary

    def execute(sql, *args, **kwargs):
        if seam == "restore" and sql == "PRAGMA busy_timeout=731" and calls and not conn.in_transaction:
            faults.append("restore")
            raise secondary
        return original_execute(sql, *args, **kwargs)

    def callback(c):
        calls.append(1)
        c.execute("INSERT INTO witness VALUES(1)")
        raise primary

    monkeypatch.setattr(conn, "rollback", rollback if seam == "rollback" else original_rollback)
    monkeypatch.setattr(conn, "execute", execute)
    caught = None
    try:
        database._execute_write(callback)
    except BaseException as exc:
        caught = exc
    finally:
        original_execute("PRAGMA busy_timeout=731")
    assert calls == [1] and faults == [seam]
    assert not conn.in_transaction
    assert original_execute("SELECT count(*) FROM witness").fetchone()[0] == 0
    assert caught is primary
    notes = " ".join(getattr(note, "message", str(note)) for note in getattr(primary, "__notes__", ()))
    assert "cleanup" in notes or "restoration" in notes
    assert str(secondary) in notes


def test_uncertain_rollback_is_never_retried(database, monkeypatch):
    conn = database._conn
    original_rollback = conn.rollback
    calls = []
    primary = _sqlite("disk I/O error", sqlite3.SQLITE_IOERR)
    secondary = sqlite3.OperationalError("rollback secondary")

    def rollback():
        original_rollback()
        calls.append("rollback")
        raise secondary

    def callback(c):
        calls.append("callback")
        c.execute("INSERT INTO witness VALUES(1)")
        raise primary

    monkeypatch.setattr(conn, "rollback", rollback)
    with pytest.raises(sqlite3.OperationalError) as caught:
        database._execute_write(callback)
    assert caught.value is primary
    notes = " ".join(getattr(note, "message", str(note)) for note in primary.__notes__)
    assert str(secondary) in notes
    assert calls == ["callback", "rollback"]
    assert not conn.in_transaction
    assert conn.execute("SELECT count(*) FROM witness").fetchone()[0] == 0


def test_successful_fts_recovery_keeps_current_owner_after_expiry(database, clock, monkeypatch):
    conn = database._conn
    original_commit = conn.commit
    calls, commits = [], []
    current = sqlite3.DatabaseError('fts5: corrupt structure record for table "messages_fts"')

    def commit():
        original_commit()
        commits.append(clock.now)
        if commits:
            clock.now = 21

    def callback(c):
        calls.append(clock.now)
        c.execute("INSERT INTO witness VALUES(1)")
        raise current

    monkeypatch.setattr(conn, "commit", commit)
    with pytest.raises(sqlite3.DatabaseError) as caught:
        database._execute_write(callback, deadline=20)
    assert caught.value is current
    assert calls == [0] and commits == [0]
    assert clock.now == 21
    assert not conn.in_transaction
    assert conn.execute("SELECT count(*) FROM witness").fetchone()[0] == 0


def test_expired_fts_owner_has_no_further_side_effect(database, clock, monkeypatch):
    before = triggers(database)
    calls, commits = [], []
    original_commit = database._conn.commit

    def commit():
        commits.append(clock.now)
        original_commit()

    def callback(c):
        calls.append(clock.now)
        raise fts_error()

    clock.now = 21
    monkeypatch.setattr(database._conn, "commit", commit)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        database._execute_write(callback, deadline=20)
    assert calls == [] and commits == []
    assert triggers(database) == before
    assert not database._fts_stale
