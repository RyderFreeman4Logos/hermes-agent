"""Tests for ``SessionDB`` compression-lock primitives.

These cover the atomic per-session lock that prevents two compression
paths from racing on the same ``session_id`` and producing orphan child
sessions (Damien's "parent → two orphan children" repro shape, see
``tests/agent/test_compression_concurrent_fork.py`` for the
behavioural regression test).

Focus here: the lock primitives themselves (acquire, release, TTL,
diagnostic accessor) — not the wiring into compression.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_state
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    return SessionDB(tmp_path / "state.db")


# ----------------------------------------------------------------------
# Single-holder semantics
# ----------------------------------------------------------------------


def test_acquire_succeeds_when_unlocked(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "holder1") is True
    assert db.get_compression_lock_holder("sess1") == "holder1"


def test_acquire_blocks_second_holder(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "holder1") is True
    assert db.try_acquire_compression_lock("sess1", "holder2") is False
    # First holder still owns it
    assert db.get_compression_lock_holder("sess1") == "holder1"








# ----------------------------------------------------------------------
# Per-session isolation
# ----------------------------------------------------------------------


def test_locks_are_per_session(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "holder1") is True
    # Different session: independent lock
    assert db.try_acquire_compression_lock("sess2", "holder2") is True
    assert db.get_compression_lock_holder("sess1") == "holder1"
    assert db.get_compression_lock_holder("sess2") == "holder2"


# ----------------------------------------------------------------------
# TTL / expiry recovery
# ----------------------------------------------------------------------


def test_expired_lock_is_reclaimable(db: SessionDB) -> None:
    """A crashed compressor must not permanently block the session."""
    # Acquire with a very short TTL
    db.try_acquire_compression_lock("sess1", "crashed_holder", ttl_seconds=0.05)
    time.sleep(0.1)
    # Holder check honours expiry
    assert db.get_compression_lock_holder("sess1") is None
    # New holder can claim it
    assert db.try_acquire_compression_lock("sess1", "fresh_holder") is True
    assert db.get_compression_lock_holder("sess1") == "fresh_holder"


def test_non_expired_lock_is_held(db: SessionDB) -> None:
    db.try_acquire_compression_lock("sess1", "holder1", ttl_seconds=60)
    # Immediately after, still held
    assert db.try_acquire_compression_lock("sess1", "holder2") is False


def test_non_expired_lock_from_dead_pid_is_reclaimed(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No ``os.name`` pin: the probe below injects a fake ``psutil``, and
    # ``_process_is_gone`` consults psutil *before* its POSIX/nt split — the
    # nt early-return is unreachable here on any host, so faking the platform
    # bought nothing.
    dead_holder = "pid=424242:tid=1:agent=abc:nonce=deadbeef"
    assert db.try_acquire_compression_lock(
        "sess1", dead_holder, ttl_seconds=300
    ) is True

    probed: list[int] = []

    def process_is_gone(pid: int) -> bool:
        probed.append(pid)
        return False

    monkeypatch.setattr(
        hermes_state, "psutil", SimpleNamespace(pid_exists=process_is_gone)
    )

    assert db.try_acquire_compression_lock(
        "sess1", "pid=525252:tid=2:agent=def:nonce=fresh", ttl_seconds=300
    ) is True
    assert probed == [424242]




def test_probe_doubt_keeps_lease_until_ttl(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that errors out is doubt, not proof of death → TTL protects."""
    holder = "pid=424242:tid=1:agent=abc:nonce=doubt"
    assert db.try_acquire_compression_lock(
        "sess1", holder, ttl_seconds=300
    ) is True

    def probe_blows_up(pid: int) -> bool:
        raise RuntimeError("transient probe failure")

    monkeypatch.setattr(
        hermes_state, "psutil", SimpleNamespace(pid_exists=probe_blows_up)
    )

    assert db.try_acquire_compression_lock(
        "sess1", "pid=525252:tid=2:agent=def:nonce=other", ttl_seconds=300
    ) is False
    assert db.get_compression_lock_holder("sess1") == holder


def test_non_expired_lock_from_live_pid_is_not_reclaimed(db: SessionDB) -> None:
    live_holder = f"pid={os.getpid()}:tid=1:agent=abc:nonce=live"
    assert db.try_acquire_compression_lock(
        "sess1", live_holder, ttl_seconds=300
    ) is True
    assert db.try_acquire_compression_lock(
        "sess1", "pid=525252:tid=2:agent=def:nonce=other", ttl_seconds=300
    ) is False




def test_unstructured_holder_waits_for_ttl(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert db.try_acquire_compression_lock(
        "sess1", "legacy_holder", ttl_seconds=300
    ) is True
    monkeypatch.setattr(
        hermes_state,
        "psutil",
        SimpleNamespace(
            pid_exists=lambda _pid: pytest.fail(
                "unstructured holder must not probe a PID"
            )
        ),
    )
    monkeypatch.setattr(
        hermes_state.os,
        "kill",
        lambda *_args: pytest.fail("unstructured holder must not probe a PID"),
    )
    assert db.try_acquire_compression_lock(
        "sess1", "pid=525252:tid=2:agent=def:nonce=other", ttl_seconds=300
    ) is False


def _hold_sqlite_writer(db: SessionDB) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(db.db_path), timeout=0, isolation_level=None, check_same_thread=False
    )
    conn.execute("BEGIN IMMEDIATE")
    return conn


def test_busy_acquire_retries_with_jitter_then_succeeds(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = _hold_sqlite_writer(db)
    db._conn.execute("PRAGMA busy_timeout=50")
    jitters: list[float] = []
    retry_observed = threading.Event()

    def jitter(_low: float, _high: float) -> float:
        jitters.append(0.02)
        retry_observed.set()
        return 0.02

    def release_writer() -> None:
        retry_observed.wait()
        blocker.rollback()
        blocker.close()

    monkeypatch.setattr(hermes_state.random, "uniform", jitter)
    releaser = threading.Thread(target=release_writer)
    releaser.start()
    try:
        assert db.try_acquire_compression_lock("busy-session", "winner") is True
    finally:
        retry_observed.set()
        releaser.join(timeout=2)

    assert not releaser.is_alive()
    assert jitters
    assert db.get_compression_lock_holder("busy-session") == "winner"


def test_busy_acquire_retries_through_exhaustion_budget(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = _hold_sqlite_writer(db)
    patience_s = 0.12
    sleeps: list[float] = []

    db._conn.execute("PRAGMA busy_timeout=0")
    monkeypatch.setattr(db, "_COMPRESSION_LOCK_ACQUIRE_PATIENCE_S", patience_s)
    monkeypatch.setattr(
        hermes_state,
        "time",
        SimpleNamespace(
            time=time.time, monotonic=lambda: sum(sleeps), sleep=sleeps.append
        ),
    )
    monkeypatch.setattr(hermes_state.random, "uniform", lambda *_args: 0.02)

    try:
        assert db.try_acquire_compression_lock("busy-session", "candidate") is False
    finally:
        blocker.rollback()
        blocker.close()

    assert len(sleeps) > 1
    assert sum(sleeps) == pytest.approx(patience_s)


def test_acquire_non_busy_sqlite_error_fails_without_retry(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fail_write(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(db, "_execute_write", fail_write)
    monkeypatch.setattr(
        hermes_state.random,
        "uniform",
        lambda *_args: pytest.fail("non-busy failures must not jitter-retry"),
    )

    with pytest.raises(sqlite3.DatabaseError, match="disk image is malformed"):
        db.try_acquire_compression_lock("broken-session", "candidate")

    assert calls == 1






# ----------------------------------------------------------------------
# Empty / invalid input
# ----------------------------------------------------------------------








# ----------------------------------------------------------------------
# Concurrency: real threads racing on the same session_id
# ----------------------------------------------------------------------


def test_concurrent_acquire_only_one_winner(db: SessionDB) -> None:
    """Damien's race shape: N threads call acquire on the same session_id;
    exactly one must win, the rest must be cleanly rejected."""
    results: list[bool] = []
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def try_acquire(idx: int) -> None:
        holder = f"thread_{idx}"
        barrier.wait()  # synchronize start
        got = db.try_acquire_compression_lock("contended_session", holder)
        with lock:
            results.append(got)

    threads = [threading.Thread(target=try_acquire, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread acquired
    assert sum(1 for r in results if r is True) == 1
    assert sum(1 for r in results if r is False) == 7
    # The single winner still owns it
    assert db.get_compression_lock_holder("contended_session") is not None
