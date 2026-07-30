"""Cross-process advisory write-lock coordination for SessionDB (#27).

Behavioural coverage only — no source-regex assertions.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pytest

from hermes_state import (
    SessionDB,
    _try_acquire_exclusive_file_lock,
    session_db_write_lock_path,
)


def _hold_write_lock(lock_path: str, hold_s: float, ready_path: str, done_path: str) -> None:
    """Child process: hold the advisory write lock for *hold_s* seconds."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        deadline = time.monotonic() + 5.0
        while not _try_acquire_exclusive_file_lock(handle):
            if time.monotonic() >= deadline:
                raise RuntimeError("child failed to acquire write lock")
            time.sleep(0.010)
        Path(ready_path).write_text("ready", encoding="utf-8")
        time.sleep(hold_s)
    finally:
        try:
            from hermes_state import _release_exclusive_file_lock

            _release_exclusive_file_lock(handle)
        except Exception:
            pass
        handle.close()
        Path(done_path).write_text("done", encoding="utf-8")


def _wait_for_file(path: Path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.010)
    raise AssertionError(f"timed out waiting for {path}")


@pytest.fixture
def db(tmp_path: Path):
    database = SessionDB(db_path=tmp_path / "state.db")
    yield database
    database.close()


def test_write_lock_path_is_sidecar_of_db(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "state.db"
    assert session_db_write_lock_path(db_path) == Path(str(db_path) + ".write.lock")


def test_read_only_sessiondb_does_not_take_write_lock(tmp_path: Path) -> None:
    writable = SessionDB(db_path=tmp_path / "state.db")
    try:
        writable.create_session(session_id="ro-seed", source="test")
        lock_path = session_db_write_lock_path(writable.db_path)
        # Force the lock file into existence by performing a write.
        assert lock_path.exists()
        handle = lock_path.open("a+b")
        assert _try_acquire_exclusive_file_lock(handle)
        try:
            # Holding the exclusive lock must not block read-only open/reads.
            ro = SessionDB(db_path=writable.db_path, read_only=True)
            try:
                assert ro.read_only is True
                assert ro._write_lock_path is None
                assert ro._write_lock_handle is None
                assert ro.get_session("ro-seed") is not None
            finally:
                ro.close()
        finally:
            from hermes_state import _release_exclusive_file_lock

            _release_exclusive_file_lock(handle)
            handle.close()
    finally:
        writable.close()


def test_cross_process_write_waits_for_advisory_lock_then_succeeds(
    db: SessionDB, tmp_path: Path
) -> None:
    db.create_session(session_id="coord-target", source="test")
    lock_path = session_db_write_lock_path(db.db_path)
    ready = tmp_path / "holder-ready"
    done = tmp_path / "holder-done"

    # Hold the advisory lock longer than a single jitter sleep so the writer
    # must observe contention, then release so the retry succeeds.
    holder = multiprocessing.Process(
        target=_hold_write_lock,
        args=(str(lock_path), 0.35, str(ready), str(done)),
    )
    holder.start()
    try:
        _wait_for_file(ready)
        started = time.monotonic()
        db.append_message(
            session_id="coord-target",
            role="user",
            content="after-cross-process-lock",
        )
        elapsed = time.monotonic() - started
        assert elapsed >= 0.25
        messages = db.get_messages("coord-target")
        assert any(m["content"] == "after-cross-process-lock" for m in messages)
    finally:
        holder.join(timeout=5)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=2)
        assert holder.exitcode == 0
        _wait_for_file(done)


def test_advisory_lock_timeout_fails_closed_with_database_locked(
    db: SessionDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db.create_session(session_id="timeout-target", source="test")
    lock_path = session_db_write_lock_path(db.db_path)
    ready = tmp_path / "holder-ready"
    done = tmp_path / "holder-done"

    # Shrink the slow profile so the test stays quick while still exercising
    # the wall-clock budget path that reopen_session uses.
    monkeypatch.setattr(db, "_SLOW_WRITE_MAX_RETRIES", 4)
    monkeypatch.setattr(db, "_SLOW_WRITE_RETRY_MIN_S", 0.010)
    monkeypatch.setattr(db, "_SLOW_WRITE_RETRY_MAX_S", 0.020)
    monkeypatch.setattr(db, "_SLOW_WRITE_BUSY_TIMEOUT_S", 0.050)
    monkeypatch.setattr(db, "_SLOW_WRITE_MAX_WAIT_S", 0.15)

    holder = multiprocessing.Process(
        target=_hold_write_lock,
        args=(str(lock_path), 2.0, str(ready), str(done)),
    )
    holder.start()
    try:
        _wait_for_file(ready)
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            db._execute_write(lambda _conn: None, retry_profile="slow")
        elapsed = time.monotonic() - started
        assert 0.10 <= elapsed < 1.5
    finally:
        holder.terminate()
        holder.join(timeout=2)


def test_default_profile_lock_exhaustion_preserves_locked_error(
    db: SessionDB, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = session_db_write_lock_path(db.db_path)
    ready = tmp_path / "holder-ready"
    done = tmp_path / "holder-done"

    monkeypatch.setattr(db, "_WRITE_MAX_RETRIES", 3)
    monkeypatch.setattr(db, "_WRITE_RETRY_MIN_S", 0.005)
    monkeypatch.setattr(db, "_WRITE_RETRY_MAX_S", 0.010)

    holder = multiprocessing.Process(
        target=_hold_write_lock,
        args=(str(lock_path), 2.0, str(ready), str(done)),
    )
    holder.start()
    try:
        _wait_for_file(ready)
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            db._execute_write(lambda _conn: None, retry_profile="default")
    finally:
        holder.terminate()
        holder.join(timeout=2)


def test_reopen_session_compatible_with_advisory_lock_and_slow_profile(
    db: SessionDB, tmp_path: Path
) -> None:
    """reopen_session (slow profile from #26) still works under lock contention."""
    db.create_session(session_id="resume-target", source="test")
    db.end_session("resume-target", "client_close")
    session = db.get_session("resume-target")
    assert session is not None
    assert session["ended_at"] is not None

    lock_path = session_db_write_lock_path(db.db_path)
    ready = tmp_path / "holder-ready"
    done = tmp_path / "holder-done"
    holder = multiprocessing.Process(
        target=_hold_write_lock,
        args=(str(lock_path), 0.30, str(ready), str(done)),
    )
    holder.start()
    try:
        _wait_for_file(ready)
        db.reopen_session("resume-target")
        session = db.get_session("resume-target")
        assert session is not None
        assert session["ended_at"] is None
    finally:
        holder.join(timeout=5)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=2)


def test_close_releases_write_lock_fd(db: SessionDB) -> None:
    db.create_session(session_id="close-target", source="test")
    assert db._write_lock_handle is not None or session_db_write_lock_path(db.db_path).exists()
    # Ensure handle is open after a write.
    db.append_message(session_id="close-target", role="user", content="x")
    assert db._write_lock_handle is not None
    assert db._write_lock_held is False
    db.close()
    assert db._write_lock_handle is None
    assert db._write_lock_held is False
