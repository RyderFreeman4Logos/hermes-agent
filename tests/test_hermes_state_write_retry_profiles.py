"""Behavioral coverage for SessionDB's fast and latency-tolerant write retries."""

from __future__ import annotations

import multiprocessing
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator, cast

import pytest

import hermes_state
from hermes_state import SessionDB


class _TransientLockingConnection:
    """Delegate to a real connection while making selected BEGINs contend."""

    def __init__(self, connection: sqlite3.Connection, locked_begins: int) -> None:
        self._connection = connection
        self._locked_begins = locked_begins
        self.begin_attempts = 0
        self.busy_timeout_sets: list[int] = []

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        normalized = " ".join(sql.strip().upper().split())
        if normalized == "BEGIN IMMEDIATE":
            self.begin_attempts += 1
            if self.begin_attempts <= self._locked_begins:
                raise sqlite3.OperationalError("database is locked")
        if normalized.startswith("PRAGMA BUSY_TIMEOUT="):
            self.busy_timeout_sets.append(int(normalized.split("=", 1)[1]))
        return self._connection.execute(sql, *args, **kwargs)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def _hold_sqlite_write_lock(
    db_path: str, ready_queue: Any, hold_s: float
) -> None:
    """Acquire a real SQLite writer lock from a separate OS process."""
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path, timeout=0.0, isolation_level=None)
        connection.execute("BEGIN IMMEDIATE")
        ready_queue.put(("locked", ""))
        time.sleep(hold_s)
        connection.rollback()
    except Exception as exc:
        ready_queue.put(("error", repr(exc)))
        raise
    finally:
        if connection is not None:
            connection.close()


@contextmanager
def _external_write_lock(db_path: str, hold_s: float) -> Iterator[None]:
    """Hold a write lock in a child process, then prove it shut down cleanly."""
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    process = context.Process(
        target=_hold_sqlite_write_lock,
        args=(db_path, ready_queue, hold_s),
    )
    process.start()
    try:
        status, detail = ready_queue.get(timeout=5.0)
        assert status == "locked", f"lock holder failed: {detail}"
        yield
    finally:
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        assert process.exitcode == 0


@pytest.fixture()
def db(tmp_path) -> Iterator[SessionDB]:
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _record_retry_delays(monkeypatch: pytest.MonkeyPatch) -> tuple[list[tuple[float, float]], list[float]]:
    ranges: list[tuple[float, float]] = []
    sleeps: list[float] = []

    def choose_delay(lower: float, upper: float) -> float:
        ranges.append((lower, upper))
        return lower

    monkeypatch.setattr(hermes_state.random, "uniform", choose_delay)
    monkeypatch.setattr(hermes_state.time, "sleep", sleeps.append)
    return ranges, sleeps


def test_reopen_session_uses_slow_retry_profile_after_transient_lock(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit resume waits through contention with the slow jitter bounds."""
    db.create_session("resume-target", source="tui")
    db.end_session("resume-target", "client_close")
    original_connection = db._conn
    assert original_connection is not None
    connection = _TransientLockingConnection(original_connection, locked_begins=2)
    db_any = cast(Any, db)
    db_any._conn = connection
    ranges, sleeps = _record_retry_delays(monkeypatch)

    try:
        db.reopen_session("resume-target")
    finally:
        db_any._conn = original_connection

    assert connection.begin_attempts == 3
    assert ranges == [
        (db_any._SLOW_WRITE_RETRY_MIN_S, db_any._SLOW_WRITE_RETRY_MAX_S),
        (db_any._SLOW_WRITE_RETRY_MIN_S, db_any._SLOW_WRITE_RETRY_MAX_S),
    ]
    assert sleeps == [db_any._SLOW_WRITE_RETRY_MIN_S, db_any._SLOW_WRITE_RETRY_MIN_S]
    assert int(original_connection.execute("PRAGMA busy_timeout").fetchone()[0]) == int(
        db_any._WRITE_BUSY_TIMEOUT_S * 1000
    )
    assert int(db_any._SLOW_WRITE_BUSY_TIMEOUT_S * 1000) in connection.busy_timeout_sets
    session = db.get_session("resume-target")
    assert session is not None
    assert session["ended_at"] is None


def test_append_message_keeps_fast_retry_profile_under_transient_lock(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frequent transcript appends retain the existing short-jitter policy."""
    db.create_session("append-target", source="tui")
    original_connection = db._conn
    assert original_connection is not None
    connection = _TransientLockingConnection(original_connection, locked_begins=2)
    db_any = cast(Any, db)
    db_any._conn = connection
    ranges, sleeps = _record_retry_delays(monkeypatch)

    try:
        db.append_message("append-target", role="user", content="after contention")
    finally:
        db_any._conn = original_connection

    assert connection.begin_attempts == 3
    assert ranges == [
        (db._WRITE_RETRY_MIN_S, db._WRITE_RETRY_MAX_S),
        (db._WRITE_RETRY_MIN_S, db._WRITE_RETRY_MAX_S),
    ]
    assert sleeps == [db._WRITE_RETRY_MIN_S, db._WRITE_RETRY_MIN_S]
    assert connection.busy_timeout_sets == []
    assert [message["content"] for message in db.get_messages("append-target")] == [
        "after contention"
    ]


def test_explicit_zero_max_retries_keeps_one_write_attempt(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit zero preserves the legacy single-attempt override."""
    original_connection = db._conn
    assert original_connection is not None
    connection = _TransientLockingConnection(original_connection, locked_begins=10)
    db_any = cast(Any, db)
    db_any._conn = connection
    ranges, sleeps = _record_retry_delays(monkeypatch)

    try:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            db_any._execute_write(lambda _connection: None, max_retries=0)
    finally:
        db_any._conn = original_connection

    assert connection.begin_attempts == 1
    assert ranges == []
    assert sleeps == []


def test_slow_retry_profile_exhaustion_preserves_database_lock_error(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slow writes remain bounded and surface the final SQLite lock error."""
    original_connection = db._conn
    assert original_connection is not None
    connection = _TransientLockingConnection(original_connection, locked_begins=10)
    db_any = cast(Any, db)
    db_any._conn = connection
    monkeypatch.setattr(db, "_SLOW_WRITE_MAX_RETRIES", 3)
    _ranges, sleeps = _record_retry_delays(monkeypatch)

    try:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            db_any._execute_write(lambda _connection: None, retry_profile="slow")
    finally:
        db_any._conn = original_connection

    assert connection.begin_attempts == 3
    assert sleeps == [db_any._SLOW_WRITE_RETRY_MIN_S, db_any._SLOW_WRITE_RETRY_MIN_S]


def test_reopen_session_survives_real_cross_process_write_lock(db: SessionDB) -> None:
    """A resumed session succeeds when another process briefly owns WAL's writer lock."""
    db.create_session("cross-process-resume", source="tui")
    db.end_session("cross-process-resume", "client_close")

    with _external_write_lock(str(db.db_path), hold_s=0.5):
        db.reopen_session("cross-process-resume")

    session = db.get_session("cross-process-resume")
    assert session is not None
    assert session["ended_at"] is None


def test_reopen_session_slow_profile_respects_wall_clock_budget(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile deadline prevents a late resume write after contention outlives it."""
    db.create_session("budgeted-resume", source="tui")
    db.end_session("budgeted-resume", "client_close")
    monkeypatch.setattr(db, "_SLOW_WRITE_MAX_WAIT_S", 0.2)
    monkeypatch.setattr(db, "_SLOW_WRITE_BUSY_TIMEOUT_S", 0.05)

    with _external_write_lock(str(db.db_path), hold_s=0.75):
        started_at = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            db.reopen_session("budgeted-resume")
        elapsed_s = time.monotonic() - started_at

    assert elapsed_s < 0.7
    session = db.get_session("budgeted-resume")
    assert session is not None
    assert session["ended_at"] is not None
