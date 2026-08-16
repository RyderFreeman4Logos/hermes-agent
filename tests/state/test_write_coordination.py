"""Multiprocess coverage for SessionDB advisory writer coordination."""

import fcntl
import multiprocessing
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB


def _hold_advisory_lock(lock_path: str, ready, release) -> None:
    with Path(lock_path).open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        ready.set()
        release.wait(5)


def _hold_sqlite_write(db_path: str, ready, release) -> None:
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ready.set()
        release.wait(5)
        conn.execute("COMMIT")
    finally:
        conn.close()


@pytest.fixture
def process_ctx():
    return multiprocessing.get_context("spawn")


@pytest.fixture
def db(tmp_path: Path):
    database = SessionDB(db_path=tmp_path / "state.db")
    yield database
    database.close()


def test_advisory_lock_excludes_other_process(db, process_ctx) -> None:
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(f"{db.db_path}.write.lock", ready, release),
    )
    holder.start()
    assert ready.wait(10)
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    try:
        db._execute_write(lambda _conn: assert_lock_released(release), patience_s=1.0)
    finally:
        release.set()
        release_timer.join()
        holder.join(10)
    assert holder.exitcode == 0


def test_fts_fail_open_waits_for_advisory_lock(db, process_ctx) -> None:
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(f"{db.db_path}.write.lock", ready, release),
    )
    holder.start()
    assert ready.wait(10)
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    try:
        assert db._enter_fts_fail_open(sqlite3.DatabaseError("fts5 corrupt"))
        assert release.is_set(), "FTS fail-open committed while another process held its lock"
    finally:
        release.set()
        release_timer.join()
        holder.join(10)
    assert holder.exitcode == 0


def test_reconnect_after_notadb_waits_for_advisory_lock(db, process_ctx) -> None:
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(f"{db.db_path}.write.lock", ready, release),
    )
    holder.start()
    assert ready.wait(10)
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    try:
        assert db._reconnect_after_notadb()
        assert release.is_set(), "reconnect completed while another process held its lock"
    finally:
        release.set()
        release_timer.join()
        holder.join(10)
    assert holder.exitcode == 0


def test_schema_initialization_waits_for_advisory_lock(tmp_path, process_ctx) -> None:
    db_path = tmp_path / "state.db"
    initial = SessionDB(db_path=db_path)
    initial.close()
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(f"{db_path}.write.lock", ready, release),
    )
    holder.start()
    assert ready.wait(10)
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    database = None
    try:
        database = SessionDB(db_path=db_path)
        assert release.is_set(), "schema initialization ran while another process held its lock"
    finally:
        if database is not None:
            database.close()
        release.set()
        release_timer.join()
        holder.join(10)
    assert holder.exitcode == 0


def test_advisory_lock_failure_is_bounded(db, process_ctx) -> None:
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_advisory_lock,
        args=(f"{db.db_path}.write.lock", ready, release),
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
    assert holder.exitcode == 0
    assert 0.15 <= time.monotonic() - started < 1.0


def test_transient_sqlite_busy_still_retries(db, process_ctx) -> None:
    ready, release = process_ctx.Event(), process_ctx.Event()
    holder = process_ctx.Process(
        target=_hold_sqlite_write, args=(str(db.db_path), ready, release)
    )
    holder.start()
    assert ready.wait(10)
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    try:
        assert db._execute_write(lambda _conn: "retried", patience_s=1.0) == "retried"
    finally:
        release.set()
        release_timer.join()
        holder.join(10)
    assert holder.exitcode == 0


def assert_lock_released(release) -> None:
    assert release.is_set(), "SessionDB write entered while another process held its lock"
