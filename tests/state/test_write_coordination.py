"""Real SQLite multiprocess coverage for SessionDB writer coordination."""

from contextlib import contextmanager
import errno
import multiprocessing
import os
import platform
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import pytest

import hermes_state
from hermes_state import (
    SessionDB,
    _open_session_db_advisory_lock_handle,
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


def _hold_advisory_lock(lock_path: str, ready, release, hold_s: float = 5.0) -> None:
    handle = _open_session_db_advisory_lock_handle(Path(lock_path))
    try:
        if not _try_acquire_exclusive_file_lock(handle):
            raise RuntimeError("could not acquire advisory lock")
        ready.set()
        release.wait(hold_s)
    finally:
        _release_exclusive_file_lock(handle)
        handle.close()


def _crash_with_advisory_lock(lock_path: str, ready) -> None:
    handle = _open_session_db_advisory_lock_handle(Path(lock_path))
    if not _try_acquire_exclusive_file_lock(handle):
        os._exit(2)
    ready.set()
    os._exit(23)


_WINDOWS_RACE_BARRIER = None
_WINDOWS_RACE_PATH = None
_WINDOWS_RACE_REAL_OPEN = os.open


def _windows_race_open(path, flags, mode=0o777):
    fd = _WINDOWS_RACE_REAL_OPEN(path, flags, mode)
    if (
        _WINDOWS_RACE_BARRIER is not None
        and os.fspath(path) == _WINDOWS_RACE_PATH
        and os.fstat(fd).st_size == 0
    ):
        _WINDOWS_RACE_BARRIER.wait(10)
    return fd


class _RaceMsvcrt:
    LK_NBLCK = 1
    LK_UNLCK = 2

    @staticmethod
    def locking(fd, mode, nbytes):
        if os.lseek(fd, 0, os.SEEK_CUR) != 0:
            raise OSError("lock offset was not byte zero")
        if os.fstat(fd).st_size < nbytes:
            raise OSError("cannot lock past EOF")


def _windows_first_use_worker(db_path: str, result) -> None:
    try:
        with hermes_state._session_db_advisory_write_lock(
            Path(db_path),
            deadline=time.monotonic() + 5.0,
            patience_s=5.0,
            allow_unsupported=False,
        ):
            result.put("ok")
    except BaseException as exc:
        result.put(repr(exc))


def _fail_advisory_lock(monkeypatch, system_name: str, error_code: int) -> None:
    if system_name == "Windows":
        class FakeMsvcrt:
            LK_NBLCK = 1
            LK_UNLCK = 2

            @staticmethod
            def locking(_fd, _mode, _nbytes):
                raise OSError(error_code, "advisory lock failure")

        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt)
        return

    import fcntl

    def fail(*_args):
        raise OSError(error_code, "advisory lock failure")

    monkeypatch.setattr(fcntl, "flock", fail)


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


@contextmanager
def _advisory_holder(
    process_ctx, lock_path: Path, *, hold_s: float = 5.0, release_after: float = 0.3
):
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(str(lock_path), ready, release, hold_s),
    )
    holder.start()
    release_timer = threading.Timer(release_after, release.set)
    try:
        assert ready.wait(10)
        release_timer.start()
        yield holder
    finally:
        release.set()
        if release_timer.is_alive():
            release_timer.join(10)
        holder.join(10)
        if holder.is_alive():
            holder.terminate()
            holder.join(10)
    assert holder.exitcode == 0


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
    with _open_session_db_advisory_lock_handle(fresh) as handle:
        assert _try_acquire_exclusive_file_lock(handle)
        _release_exclusive_file_lock(handle)
    assert fresh.read_bytes() == b"\0"

    existing = tmp_path / "existing.lock"
    existing.write_bytes(b"keep me")
    with _open_session_db_advisory_lock_handle(existing) as handle:
        assert _try_acquire_exclusive_file_lock(handle)
        _release_exclusive_file_lock(handle)
    assert existing.read_bytes() == b"keep me"
    assert calls == [
        (FakeMsvcrt.LK_NBLCK, 1),
        (FakeMsvcrt.LK_UNLCK, 1),
        (FakeMsvcrt.LK_NBLCK, 1),
        (FakeMsvcrt.LK_UNLCK, 1),
    ]


def test_windows_first_use_race_seeds_exactly_one_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global _WINDOWS_RACE_BARRIER, _WINDOWS_RACE_PATH

    ctx = multiprocessing.get_context("fork")
    db_path = tmp_path / "state.db"
    lock_path = session_db_write_lock_path(db_path)
    _WINDOWS_RACE_BARRIER = ctx.Barrier(2)
    _WINDOWS_RACE_PATH = os.fspath(lock_path)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "msvcrt", _RaceMsvcrt)
    monkeypatch.setattr(hermes_state.os, "open", _windows_race_open)
    results = ctx.Queue()
    workers = [
        ctx.Process(target=_windows_first_use_worker, args=(str(db_path), results))
        for _ in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        assert [results.get(timeout=10), results.get(timeout=10)] == ["ok", "ok"]
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
        assert lock_path.read_bytes() == b"\0"
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(10)
        _WINDOWS_RACE_BARRIER = None
        _WINDOWS_RACE_PATH = None


def test_windows_lock_violation_is_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_advisory_lock(monkeypatch, "Windows", errno.EACCES)
    lock_path = tmp_path / "busy.lock"
    lock_path.write_bytes(b"\0")
    with lock_path.open("a+b") as handle:
        assert not _try_acquire_exclusive_file_lock(handle)


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


def test_waiting_writer_does_not_block_same_instance_public_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_ctx,
) -> None:
    monkeypatch.setattr(hermes_state, "resolve_journal_mode", lambda: "delete")
    database = SessionDB(db_path=tmp_path / "reader-convoy.db")
    assert database._conn is not None
    assert database._conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    database.create_session("readable", source="test", model="test")
    database.append_message("readable", "user", "hello")
    database.set_meta("seed", "value")

    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(str(session_db_write_lock_path(database.db_path)), ready, release, 10.0),
    )
    writer_waiting = threading.Event()
    original_try_lock = hermes_state._try_acquire_exclusive_file_lock

    def observe_contention(handle) -> bool:
        acquired = original_try_lock(handle)
        if not acquired:
            writer_waiting.set()
        return acquired

    executor = ThreadPoolExecutor(max_workers=3)
    holder.start()
    try:
        assert ready.wait(10)
        monkeypatch.setattr(
            hermes_state, "_try_acquire_exclusive_file_lock", observe_contention
        )
        writer = executor.submit(database.set_meta, "writer", "done")
        assert writer_waiting.wait(2)

        meta_reader = executor.submit(database.get_meta, "seed")
        messages_reader = executor.submit(database.get_messages, "readable")
        wait((meta_reader,), timeout=2)
        wait((messages_reader,), timeout=2)
        readers_completed_before_release = meta_reader.done() and messages_reader.done()
        writer_blocked_before_release = not writer.done()
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)
        holder.join(10)
        if holder.is_alive():
            holder.terminate()
            holder.join(10)
        writer_value = database.get_meta("writer")
        database.close()

    assert holder.exitcode == 0
    assert readers_completed_before_release
    assert writer_blocked_before_release
    assert meta_reader.result() == "value"
    assert [message["content"] for message in messages_reader.result()] == ["hello"]
    assert writer_value == "done"


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


@pytest.mark.parametrize("system_name", ["POSIX", "Windows"])
def test_delete_mode_public_write_survives_unsupported_advisory_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system_name: str,
) -> None:
    monkeypatch.setattr(hermes_state, "resolve_journal_mode", lambda: "delete")
    database = SessionDB(db_path=tmp_path / f"{system_name}.db")
    assert database._conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"

    _fail_advisory_lock(monkeypatch, system_name, errno.ENOTSUP)

    try:
        database.set_meta("unsupported-lock", "written")
        assert database.get_meta("unsupported-lock") == "written"
    finally:
        database.close()


@pytest.mark.parametrize("system_name", ["POSIX", "Windows"])
def test_fatal_advisory_lock_error_propagates_without_contention_wait(
    db,
    monkeypatch: pytest.MonkeyPatch,
    system_name: str,
) -> None:
    _fail_advisory_lock(monkeypatch, system_name, errno.EIO)

    started = time.monotonic()
    with pytest.raises(OSError) as excinfo:
        db.set_meta("fatal-lock", "not-written")
    assert excinfo.value.errno == errno.EIO
    assert time.monotonic() - started < 0.5
    assert db.get_meta("fatal-lock") is None


def test_automatic_fts_merge_waits_for_advisory_lock(db, process_ctx) -> None:
    if not db._fts_enabled:
        pytest.skip("FTS5 unavailable in this build")
    started = time.monotonic()
    with _advisory_holder(process_ctx, session_db_write_lock_path(db.db_path)):
        db._try_incremental_merge_fts()
    assert time.monotonic() - started >= 0.2
    assert db._fts_usermerge_floor_applied


def test_schema_open_waits_for_advisory_lock(db, process_ctx) -> None:
    started = time.monotonic()
    with _advisory_holder(process_ctx, session_db_write_lock_path(db.db_path)):
        reopened = SessionDB(db_path=db.db_path)
    assert time.monotonic() - started >= 0.2
    reopened.set_meta("coordinated-open", "written")
    assert reopened.get_meta("coordinated-open") == "written"
    reopened.close()


def test_vacuum_waits_for_advisory_lock(
    db, process_ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "optimize_fts", lambda: 0)
    started = time.monotonic()
    with _advisory_holder(process_ctx, session_db_write_lock_path(db.db_path)):
        assert db.vacuum() == 0
    assert time.monotonic() - started >= 0.2


def test_reconnect_after_notadb_waits_for_advisory_lock(db, process_ctx) -> None:
    started = time.monotonic()
    with _advisory_holder(
        process_ctx, session_db_write_lock_path(db.db_path), hold_s=10.0
    ):
        assert db._reconnect_after_notadb()
    assert time.monotonic() - started >= 0.2
