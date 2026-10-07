"""FTS maintenance retries real contention within its shared phase patience."""

import sqlite3
import threading

import pytest

from hermes_state import SessionDB
from hermes_state_search import _FTS_WRITE_DEADLINE


def _pending(db):
    db.set_meta("fts_rebuild_high_water", "2")
    db.set_meta("fts_rebuild_progress", "0")
    db._FTS_REBUILD_CHUNK_ROWS = 1
    db._FTS_REBUILD_MIN_PAUSE = 0.03
    db._FTS_REBUILD_DUTY_FACTOR = 0.0
    db._WRITE_PATIENCE_S = 0.05


@pytest.mark.parametrize("message", ["no such column: missing", "disk i/o error"])
def test_nonlock_operational_error_stops_optimize(tmp_path, monkeypatch, message):
    """A permanent schema error stops the driver after one failed chunk."""
    db = SessionDB(db_path=tmp_path / "state.db")
    original = db._execute_write
    calls = []
    error = sqlite3.OperationalError(message)

    def _fail_once(_do, patience_s=None):
        calls.append(patience_s)
        if len(calls) == 2:
            raise error
        return original(_do, patience_s=patience_s)

    try:
        _pending(db)
        monkeypatch.setattr(db, "_execute_write", _fail_once)
        with pytest.raises(sqlite3.OperationalError) as caught:
            db.fts_rebuild_step()
            db.fts_rebuild_step()
        assert caught.value is error
        calls.clear()
        result = db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()

    assert len(calls) == 2
    assert result == {"ok": False, "reason": "fts_error", "vacuumed": None}


def _contended_optimize(tmp_path, monkeypatch, *, release=False, trash=False):
    db = SessionDB(db_path=tmp_path / "state.db")
    blocker = sqlite3.connect(db.db_path, timeout=0)
    sleeps, attempts, timeouts = [], [], []

    def clock():
        return sum(sleeps)

    def sleep(delay):
        sleeps.append(delay)
        if release and clock() >= 0.04:
            blocker.rollback()

    def trace(sql):
        if sql == "BEGIN IMMEDIATE":
            attempts.append(clock())
            timeouts.append(db._conn.execute("PRAGMA busy_timeout").fetchone()[0])

    def progress(status):
        # The first chunk makes durable progress before another writer takes over.
        if not blocker.in_transaction and not release:
            ready = status["indexed"] == 1 if not trash else (
                status["phase"] == "teardown"
                and db._conn.execute("SELECT COUNT(*) FROM fts_v22_trash_test").fetchone()[0] == 2
            )
            if ready:
                blocker.execute("BEGIN IMMEDIATE")
        elif release and status["indexed"] == 1 and not sleeps:
            blocker.execute("BEGIN IMMEDIATE")

    try:
        _pending(db)
        if trash:
            db._write_sql("CREATE TABLE fts_v22_trash_test(id INTEGER PRIMARY KEY)")
            db._write_sql("INSERT INTO fts_v22_trash_test VALUES(1),(2),(3)")
            db.set_meta("fts_rebuild_high_water", "0")
        db._conn.execute("PRAGMA busy_timeout=0")
        monkeypatch.setattr("hermes_state.time.monotonic", clock)
        monkeypatch.setattr("hermes_state.time.sleep", sleep)
        monkeypatch.setattr("hermes_state.random.uniform", lambda *_: 0.005)
        db._conn.set_trace_callback(trace)
        result = db.optimize_fts_storage(vacuum=False, progress_cb=progress)
        assert db._fts_write_patience() is None, "phase budget must not leak to standalone calls"
        remaining = (db._conn.execute("SELECT COUNT(*) FROM fts_v22_trash_test").fetchone()[0]
                     if trash else db.get_meta("fts_rebuild_progress")) if not release else None
        restored = db._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        db._conn.set_trace_callback(None)
        blocker.rollback()
        blocker.close()
        db.close()
    return result, sleeps, attempts, timeouts, remaining, restored


@pytest.mark.parametrize("trash", [False, True])
def test_repeated_lock_error_stops_at_write_patience(tmp_path, monkeypatch, trash):
    result, sleeps, attempts, timeouts, remaining, restored = _contended_optimize(
        tmp_path, monkeypatch, trash=trash,
    )
    contended = [now for now in attempts if now >= 0.03]
    assert len(contended) >= 3, "must exercise multiple real contended BEGIN attempts"
    assert len(sleeps) >= 4, "an immediate-return retry must not pass"
    assert sum(sleeps) == pytest.approx(0.05)
    assert all(now < 0.05 for now in contended), "no new attempt after phase expiry"
    assert timeouts == [0] * len(attempts)
    assert restored == 0
    assert remaining == (2 if trash else "1"), "failed chunk must not advance progress"
    assert result == {"ok": False, "reason": "fts_error", "vacuumed": None}


def test_lock_error_retries_then_succeeds(tmp_path, monkeypatch):
    result, sleeps, attempts, _, _, restored = _contended_optimize(
        tmp_path, monkeypatch, release=True,
    )
    assert len([now for now in attempts if now >= 0.03]) >= 3
    assert len(sleeps) >= 3
    assert sum(sleeps) == pytest.approx(0.04)
    assert restored == 0
    assert result["ok"] is True


def test_lock_error_busy_timeout_restored(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    blocker = sqlite3.connect(db.db_path, timeout=0)
    attempts = []
    try:
        blocker.execute("BEGIN IMMEDIATE")
        db._conn.execute("PRAGMA busy_timeout=5000")

        def trace(sql):
            if sql == "BEGIN IMMEDIATE":
                attempts.append(db._conn.execute("PRAGMA busy_timeout").fetchone()[0])

        db._conn.set_trace_callback(trace)
        with pytest.raises(sqlite3.OperationalError, match="locked") as caught:
            db._execute_write(lambda conn: conn.execute("SELECT 1"), patience_s=0.02)
        assert attempts and all(0 <= ms <= 20 for ms in attempts)
        assert caught.value.__cause__ is not None
        assert db._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        blocker.rollback()
        db._execute_write(lambda conn: conn.execute("SELECT 1"), patience_s=0.02)
        assert db._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

        def cancel(conn):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            db._execute_write(cancel, patience_s=0.02)
        assert not db._conn.in_transaction
        assert db._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        db._conn.set_trace_callback(None)
        blocker.rollback()
        blocker.close()
        db.close()


def test_lock_error_finish_uses_remaining_patience(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    sleeps, attempts = [], []
    original = db._execute_write

    def write(fn, patience_s=None):
        result = original(fn, patience_s=patience_s)
        if db.get_meta("fts_rebuild_progress") == "2":
            sleeps.append(0.02)  # the last successful chunk reaches phase expiry
        return result

    try:
        _pending(db)
        monkeypatch.setattr("hermes_state.time.monotonic", lambda: sum(sleeps))
        monkeypatch.setattr("hermes_state.time.sleep", sleeps.append)
        monkeypatch.setattr(db, "_execute_write", write)
        db._conn.set_trace_callback(lambda sql: attempts.append((sql, sum(sleeps))))
        result = db.optimize_fts_storage(vacuum=False)
        assert result == {"ok": False, "reason": "fts_error", "vacuumed": None}
        assert db.get_meta("fts_rebuild_progress") == "2"
        assert db.get_meta("fts_rebuild_high_water") == "2"
        assert all(now < 0.05 for sql, now in attempts if sql == "BEGIN IMMEDIATE")
    finally:
        db._conn.set_trace_callback(None)
        db.close()


def test_lock_error_python_lock_wait_is_bounded(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    held, release = threading.Event(), threading.Event()
    attempts = []

    def hold():
        with db._lock:
            held.set()
            release.wait(2)

    owner = threading.Thread(target=hold)
    try:
        db._conn.set_trace_callback(attempts.append)
        owner.start()
        assert held.wait(2)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            db._execute_write(lambda conn: conn.execute("SELECT 1"), patience_s=0.02)
        assert "BEGIN IMMEDIATE" not in attempts
    finally:
        release.set()
        owner.join(2)
        db._conn.set_trace_callback(None)
        db.close()
    assert not owner.is_alive()


def test_lock_error_expired_write_does_not_start(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    attempts = []
    token = _FTS_WRITE_DEADLINE.set((db, -1.0))
    try:
        db._conn.set_trace_callback(attempts.append)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            db._fts_rebuild_finish()
        assert "BEGIN IMMEDIATE" not in attempts
        # Zero patience outside the phase remains an explicit no-wait single attempt.
        assert db._execute_write(lambda conn: 1, patience_s=0) == 1
    finally:
        _FTS_WRITE_DEADLINE.reset(token)
        db._conn.set_trace_callback(None)
        db.close()
