"""Real SQLite multiprocess coverage for SessionDB writer coordination."""

import multiprocessing
import os
import platform
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_state import (
    SessionDB,
    _release_exclusive_file_lock,
    _try_acquire_exclusive_file_lock,
    session_db_write_lock_path,
)


def _append_turn(home: str, session_id: str, ready, go, result) -> None:
    os.environ["HERMES_HOME"] = home
    db = SessionDB()
    try:
        ready.put(session_id)
        go.wait(5)
        db.append_message(session_id, "user", f"{session_id}-user")
        db.append_message(session_id, "assistant", f"{session_id}-assistant")
        result.put(None)
    except BaseException as exc:
        result.put(repr(exc))
    finally:
        db.close()


def _hold_sqlite_write(db_path: str, ready, release) -> None:
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ready.set()
        release.wait(5)
        conn.execute("COMMIT")
    finally:
        conn.close()


def _hold_advisory_lock(lock_path: str, ready, release) -> None:
    handle = Path(lock_path).open("a+b")
    try:
        if not _try_acquire_exclusive_file_lock(handle):
            raise RuntimeError("could not acquire advisory lock")
        ready.set()
        release.wait(5)
    finally:
        _release_exclusive_file_lock(handle)
        handle.close()


def _crash_with_advisory_lock(lock_path: str, ready) -> None:
    handle = Path(lock_path).open("a+b")
    if not _try_acquire_exclusive_file_lock(handle):
        os._exit(2)
    ready.set()
    os._exit(23)


@pytest.fixture
def process_ctx():
    return multiprocessing.get_context("spawn")


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    database = SessionDB()
    yield database
    database.close()


def test_concurrent_writers_preserve_both_turns(db, process_ctx) -> None:
    for session_id in ("turn-a", "turn-b"):
        db.create_session(session_id, "test")

    ready, result, go = process_ctx.Queue(), process_ctx.Queue(), process_ctx.Event()
    workers = [
        process_ctx.Process(
            target=_append_turn,
            args=(str(db.db_path.parent), session_id, ready, go, result),
        )
        for session_id in ("turn-a", "turn-b")
    ]
    for worker in workers:
        worker.start()
    assert {ready.get(timeout=10), ready.get(timeout=10)} == {"turn-a", "turn-b"}
    go.set()
    assert [result.get(timeout=10), result.get(timeout=10)] == [None, None]
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0

    for session_id in ("turn-a", "turn-b"):
        assert [row["content"] for row in db.get_messages(session_id)] == [
            f"{session_id}-user",
            f"{session_id}-assistant",
        ]


def test_transient_sqlite_busy_retries(db, process_ctx) -> None:
    db.create_session("busy", "test")
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_sqlite_write, args=(str(db.db_path), ready, release)
    )
    holder.start()
    assert ready.wait(10)
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    try:
        db.append_message("busy", "user", "retried")
    finally:
        release.set()
        release_timer.join()
        holder.join(10)
    assert holder.exitcode == 0
    assert [row["content"] for row in db.get_messages("busy")] == ["retried"]


def test_non_lock_errors_fail_immediately(db) -> None:
    started = time.monotonic()
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        db._execute_write(
            lambda conn: conn.execute("SELECT * FROM definitely_missing_table"),
            patience_s=1.0,
        )
    assert time.monotonic() - started < 0.5


def test_windows_byte_lock_seeds_fresh_file_without_truncating_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(fd, mode, nbytes):
            offset = os.lseek(fd, 0, os.SEEK_CUR)
            if offset != 0 or offset + nbytes > os.fstat(fd).st_size:
                raise OSError("cannot lock past EOF")
            calls.append((mode, nbytes))

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt)

    fresh = tmp_path / "fresh.lock"
    with fresh.open("a+b") as handle:
        assert fresh.stat().st_size == 0
        assert _try_acquire_exclusive_file_lock(handle)
        _release_exclusive_file_lock(handle)
    assert fresh.read_bytes() == b"\0"

    existing = tmp_path / "existing.lock"
    existing.write_bytes(b"keep me")
    with existing.open("a+b") as handle:
        assert _try_acquire_exclusive_file_lock(handle)
        _release_exclusive_file_lock(handle)
    assert existing.read_bytes() == b"keep me"
    assert calls == [
        (FakeMsvcrt.LK_NBLCK, 1),
        (FakeMsvcrt.LK_UNLCK, 1),
        (FakeMsvcrt.LK_NBLCK, 1),
        (FakeMsvcrt.LK_UNLCK, 1),
    ]


def test_advisory_lock_timeout_is_bounded(db, process_ctx) -> None:
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(str(session_db_write_lock_path(db.db_path)), ready, release),
    )
    holder.start()
    assert ready.wait(10)
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            db._execute_write(lambda _conn: None, patience_s=0.2)
    finally:
        release.set()
        holder.join(10)
    elapsed = time.monotonic() - started
    assert holder.exitcode == 0
    assert 0.15 <= elapsed < 1.0


def test_advisory_lock_is_cleaned_up_after_crash(db, process_ctx) -> None:
    lock_path = session_db_write_lock_path(db.db_path)
    ready = process_ctx.Event()
    worker = process_ctx.Process(
        target=_crash_with_advisory_lock, args=(str(lock_path), ready)
    )
    worker.start()
    assert ready.wait(10)
    worker.join(10)
    assert worker.exitcode == 23

    db.set_meta("after-crash", "written")
    assert db.get_meta("after-crash") == "written"
