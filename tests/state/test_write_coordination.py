"""Real SQLite multiprocess coverage for SessionDB writer coordination."""

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
    handle = Path(lock_path).open("a+b")
    try:
        if not _try_acquire_exclusive_file_lock(handle):
            raise RuntimeError("could not acquire advisory lock")
        ready.set()
        release.wait(hold_s)
    finally:
        _release_exclusive_file_lock(handle)
        handle.close()


def _crash_with_advisory_lock(lock_path: str, ready) -> None:
    handle = Path(lock_path).open("a+b")
    if not _try_acquire_exclusive_file_lock(handle):
        os._exit(2)
    ready.set()
    os._exit(23)


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


@pytest.mark.parametrize("lock_name", ["_writer_lock", "_lock"])
@pytest.mark.parametrize("patience_s", [0.0, 0.1])
def test_process_local_lock_timeout_uses_write_deadline(
    db, lock_name: str, patience_s: float
) -> None:
    ready = threading.Event()
    release = threading.Event()

    def hold_local_lock() -> None:
        with getattr(db, lock_name):
            ready.set()
            release.wait(2.0)

    holder = threading.Thread(target=hold_local_lock)
    holder.start()
    assert ready.wait(1.0)
    release_timer = threading.Timer(0.8, release.set)
    release_timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            with db._write_guard(patience_s=patience_s):
                pass
    finally:
        release.set()
        release_timer.cancel()
        release_timer.join()
        holder.join(2.0)

    assert not holder.is_alive()
    assert time.monotonic() - started < 0.5


@pytest.mark.parametrize(
    ("lock_name", "source"),
    [
        ("_writer_lock", "in-process SessionDB writer"),
        ("_lock", "SessionDB connection"),
    ],
)
def test_execute_write_preserves_process_local_timeout_source(
    db, lock_name: str, source: str
) -> None:
    ready = threading.Event()
    release = threading.Event()

    def hold_local_lock() -> None:
        with getattr(db, lock_name):
            ready.set()
            release.wait(2.0)

    holder = threading.Thread(target=hold_local_lock)
    holder.start()
    assert ready.wait(1.0)
    try:
        with pytest.raises(sqlite3.OperationalError, match=source):
            db._execute_write(lambda _conn: None, patience_s=0.0)
    finally:
        release.set()
        holder.join(2.0)
    assert not holder.is_alive()


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
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(str(session_db_write_lock_path(db.db_path)), ready, release),
    )
    holder.start()
    assert ready.wait(10)
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    started = time.monotonic()
    try:
        db._try_incremental_merge_fts()
    finally:
        release.set()
        release_timer.join()
        holder.join(10)
    assert holder.exitcode == 0
    assert time.monotonic() - started >= 0.2
    assert db._fts_usermerge_floor_applied


def test_fts_probe_and_merge_share_one_writer_interval(
    db, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not db._fts_enabled:
        pytest.skip("FTS5 unavailable in this build")

    peer = SessionDB(db_path=db.db_path)
    probe_paused = threading.Event()
    resume_probe = threading.Event()
    peer_contended = threading.Event()
    peer_finished = threading.Event()
    peer_thread_id = {}
    outcomes = {}

    original_exists = db._fts_table_exists
    original_try_lock = hermes_state._try_acquire_exclusive_file_lock

    def pause_after_positive_probe(name: str) -> bool:
        exists = original_exists(name)
        if name == "messages_fts" and exists:
            probe_paused.set()
            assert resume_probe.wait(5.0)
        return exists

    def observe_peer_contention(handle) -> bool:
        acquired = original_try_lock(handle)
        if (
            threading.get_ident() == peer_thread_id.get("value")
            and not acquired
        ):
            peer_contended.set()
        return acquired

    monkeypatch.setattr(db, "_fts_table_exists", pause_after_positive_probe)
    monkeypatch.setattr(
        hermes_state, "_try_acquire_exclusive_file_lock", observe_peer_contention
    )

    def merge() -> None:
        try:
            outcomes["merge"] = db._merge_fts_incrementally(
                max_pages=1, max_commands=1
            )
        except BaseException as exc:
            outcomes["merge_error"] = exc

    def drop_table() -> None:
        peer_thread_id["value"] = threading.get_ident()
        try:
            with peer._write_guard(patience_s=2.0):
                peer._conn.execute("DROP TABLE messages_fts")
                peer._conn.commit()
        except BaseException as exc:
            outcomes["peer_error"] = exc
        finally:
            peer_finished.set()

    merge_thread = threading.Thread(target=merge)
    peer_thread = threading.Thread(target=drop_table)
    merge_thread.start()
    serialized = False
    try:
        assert probe_paused.wait(2.0)
        peer_thread.start()
        deadline = time.monotonic() + 2.0
        while (
            not peer_contended.is_set()
            and not peer_finished.is_set()
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        serialized = peer_contended.is_set() and not peer_finished.is_set()
    finally:
        resume_probe.set()
        merge_thread.join(5.0)
        if peer_thread.ident is not None:
            peer_thread.join(5.0)
        peer.close()

    assert serialized, "peer schema mutation entered after probe but before merge"
    assert not merge_thread.is_alive()
    assert not peer_thread.is_alive()
    assert "merge_error" not in outcomes
    assert "peer_error" not in outcomes


def test_schema_open_waits_for_advisory_lock(db, process_ctx) -> None:
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(str(session_db_write_lock_path(db.db_path)), ready, release),
    )
    holder.start()
    assert ready.wait(10)
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    started = time.monotonic()
    reopened = None
    try:
        reopened = SessionDB(db_path=db.db_path)
    finally:
        release.set()
        release_timer.join()
        holder.join(10)
    assert holder.exitcode == 0
    assert time.monotonic() - started >= 0.2
    reopened.set_meta("coordinated-open", "written")
    assert reopened.get_meta("coordinated-open") == "written"
    reopened.close()


def test_schema_repair_waits_for_advisory_lock(db, process_ctx) -> None:
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(str(session_db_write_lock_path(db.db_path)), ready, release),
    )
    holder.start()
    assert ready.wait(10)
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    started = time.monotonic()
    try:
        report = hermes_state.repair_state_db_schema(db.db_path, backup=False)
    finally:
        release.set()
        release_timer.join()
        holder.join(10)
    assert holder.exitcode == 0
    assert time.monotonic() - started >= 0.2
    assert report["repaired"]
    assert report["strategy"] == "already_healthy"


def test_vacuum_waits_for_advisory_lock(
    db, process_ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "optimize_fts", lambda: 0)
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(str(session_db_write_lock_path(db.db_path)), ready, release),
    )
    holder.start()
    assert ready.wait(10)
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    started = time.monotonic()
    try:
        assert db.vacuum() == 0
    finally:
        release.set()
        release_timer.join()
        holder.join(10)
    assert holder.exitcode == 0
    assert time.monotonic() - started >= 0.2
