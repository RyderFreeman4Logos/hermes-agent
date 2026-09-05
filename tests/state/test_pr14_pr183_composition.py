"""Composition of #14 lock-retry with #183 sidecar write coordination.

#14 owns busy_timeout restore, recover_fts_errors, and max_retries.
#183 owns the sidecar flock / _write_guard. Replay stacks #183 on #14, so
_execute_write must keep both contracts: flock is released before retry
sleep, and _write_guard must not nest a second self._lock.
"""

from __future__ import annotations

import inspect
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_state
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path) -> Iterator[SessionDB]:
    database = SessionDB(tmp_path / "state.db")
    yield database
    database.close()


def test_execute_write_uses_write_guard_without_nesting_lock() -> None:
    assert hasattr(SessionDB, "_write_guard")
    src = inspect.getsource(SessionDB._execute_write)
    assert "self._write_guard" in src
    assert "with self._lock" not in src
    assert "previous_busy_timeout_ms" in src
    assert "recover_fts_errors" in src
    assert "max_retries" in src
    params = inspect.signature(SessionDB._execute_write).parameters
    assert "recover_fts_errors" in params
    assert "max_retries" in params


def test_write_guard_lock_order_is_writer_then_flock_then_lock() -> None:
    src = inspect.getsource(SessionDB._write_guard)
    writer = src.index("self._writer_lock")
    flock = src.index("_session_db_advisory_write_lock")
    inner = src.index("self._lock")
    assert writer < flock < inner


def test_busy_timeout_is_restored_after_locked_begin(db: SessionDB) -> None:
    db._conn.execute("PRAGMA busy_timeout=4321")
    blocker = sqlite3.connect(
        str(db.db_path), timeout=0, isolation_level=None, check_same_thread=False
    )
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            db._execute_write(
                lambda conn: conn.execute("SELECT 1"),
                patience_s=0.0,
                recover_fts_errors=False,
                max_retries=1,
            )
    finally:
        blocker.rollback()
        blocker.close()
    restored = int(db._conn.execute("PRAGMA busy_timeout").fetchone()[0])
    assert restored == 4321


def test_retry_sleep_does_not_hold_advisory_flock(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path_fn = getattr(hermes_state, "session_db_write_lock_path", None)
    open_handle = getattr(hermes_state, "_open_session_db_advisory_lock_handle", None)
    try_lock = getattr(hermes_state, "_try_acquire_exclusive_file_lock", None)
    release_lock = getattr(hermes_state, "_release_exclusive_file_lock", None)
    assert lock_path_fn is not None
    assert open_handle is not None
    assert try_lock is not None
    assert release_lock is not None

    held_during_sleep: list[bool] = []
    original_sleep = db._sleep_before_write_retry

    def sleep_probe(deadline: float, patience_s: float) -> bool:
        handle = open_handle(lock_path_fn(db.db_path))
        try:
            acquired = try_lock(handle)
            held_during_sleep.append(not acquired)
            if acquired:
                release_lock(handle)
        finally:
            handle.close()
        return original_sleep(deadline, patience_s)

    monkeypatch.setattr(db, "_sleep_before_write_retry", sleep_probe)
    monkeypatch.setattr(
        hermes_state,
        "time",
        SimpleNamespace(
            monotonic=lambda: 1000.0,
            sleep=lambda _duration: None,
            time=time.time,
        ),
    )

    calls = {"n": 0}
    original_execute = db._conn.execute

    def execute(sql, *args, **kwargs):
        if isinstance(sql, str) and sql.strip().upper() == "BEGIN IMMEDIATE":
            calls["n"] += 1
            if calls["n"] < 3:
                raise sqlite3.OperationalError("database is locked")
        return original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(db._conn, "execute", execute)

    result = db._execute_write(
        lambda conn: 7,
        patience_s=1.0,
        recover_fts_errors=False,
        max_retries=15,
    )
    assert result == 7
    assert calls["n"] == 3
    assert held_during_sleep == [False, False]
