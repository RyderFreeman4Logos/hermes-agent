"""Committed FTS progress is not contention; retries share one write budget."""

import sqlite3
from types import SimpleNamespace

import pytest

import hermes_state
import hermes_state_search
from hermes_state import SessionDB


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    time = monotonic

    def sleep(self, delay):
        assert delay >= 0
        self.now += delay
        assert self.now <= 1000, "finite retry sentinel"


@pytest.fixture
def pending_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        for i in range(5):
            db.append_message("s", role="user", content=f"needle{i}")
        def seed(conn):
            db._reset_fts_index_to_empty(conn)
            db._seed_fts_rebuild_markers(conn, force=True)
        db._execute_write(seed)
        db._FTS_REBUILD_CHUNK_ROWS = 1
        db._FTS_REBUILD_MIN_PAUSE = 1
        db._FTS_REBUILD_DUTY_FACTOR = 0
        yield db
    finally:
        db.close()
        assert db._conn is None


@pytest.mark.parametrize("phase", ["base", "cjk", "trash"])
@pytest.mark.parametrize("chunk_seconds", [8, 24])
def test_healthy_chunks_outlive_write_patience(pending_db, monkeypatch, phase, chunk_seconds):
    db = pending_db
    clock = Clock()
    calls = []
    if phase == "base":
        name = "fts_rebuild_step"
        step = db.fts_rebuild_step
    else:
        assert db.optimize_fts_storage(vacuum=False)["ok"]
        if phase == "cjk":
            # Exercise the shared CJK engine, not an unavailable tokenizer.
            db._execute_write(lambda c: c.execute("CREATE TABLE fixture_cjk(content,tool_name,tool_calls)"))
            db.set_meta("fts_cjk_rebuild_high_water", "5")
            db.set_meta("fts_cjk_rebuild_progress", "0")
            name = "fts_cjk_rebuild_step"
            def step(**kwargs):
                return db._rebuild_step(
                    "fts_cjk_rebuild", [db._CHUNK_INSERT_SQL.format(table="fixture_cjk", extra="")],
                    fail_msg="test %s", finish=lambda **kw: db._rebuild_finish("fts_cjk_rebuild", [], **kw),
                    **kwargs,
                )
        else:
            db._execute_write(lambda c: c.execute("CREATE TABLE fts_v22_trash_fixture(id INTEGER PRIMARY KEY)"))
            db._execute_write(lambda c: c.executemany("INSERT INTO fts_v22_trash_fixture VALUES(?)", [(i,) for i in range(1, 6)]))
            name = "_fts_teardown_trash_step"
            step = db._fts_teardown_trash_step
    write = db._execute_write
    def slow_write(fn, **kwargs):
        def work(conn):
            result = fn(conn)
            clock.now += chunk_seconds
            return result
        return write(work, **kwargs)
    def slow_step(**kwargs):
        assert len(calls) < 8
        with monkeypatch.context() as scoped:
            scoped.setattr(db, "_execute_write", slow_write)
            result = step(**kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(db, name, slow_step)
    monkeypatch.setattr(hermes_state_search, "time", clock)
    monkeypatch.setattr(hermes_state, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))
    assert db.optimize_fts_storage(vacuum=False)["ok"]
    assert clock.now > db._WRITE_PATIENCE_S
    assert len(db.get_messages("s")) == 5
    assert db.get_meta("fts_rebuild_high_water") is None
    assert db.get_meta("fts_cjk_rebuild_high_water") is None
    assert not db._conn.in_transaction
    db._conn.execute("INSERT INTO messages_fts(messages_fts,rank) VALUES('integrity-check',1)")


@pytest.mark.parametrize("release, pause", [(False, 1), (False, 2), (True, 1)])
def test_retry_budget_reaches_actual_sqlite_write(pending_db, monkeypatch, release, pause):
    db = pending_db
    clock = Clock()
    original = db.fts_rebuild_step
    starts = []
    begins = []
    db._FTS_REBUILD_MIN_PAUSE = pause
    blocker = sqlite3.connect(db.db_path, timeout=0)
    db._conn.execute("PRAGMA busy_timeout=0")
    db._conn.set_trace_callback(lambda sql: begins.append(clock.now) if sql == "BEGIN IMMEDIATE" else None)
    def step(**kwargs):
        starts.append(clock.now)
        assert len(starts) < 10
        if len(starts) == 1:
            # An outer retry already spent 18 seconds without committing anything.
            clock.now += 18
            blocker.execute("BEGIN IMMEDIATE")
            raise_error = sqlite3.OperationalError("database is locked")
            return db._fts_chunk_error(raise_error, "test %s")
        return original(**kwargs)
    def sleep(delay):
        clock.sleep(delay)
        if release:
            blocker.rollback()
    monkeypatch.setattr(db, "fts_rebuild_step", step)
    monkeypatch.setattr(hermes_state_search, "time", clock)
    monkeypatch.setattr(hermes_state, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=sleep))
    monkeypatch.setattr(hermes_state.random, "uniform", lambda *args: 0.5 if release else 4)
    try:
        result = db.optimize_fts_storage(vacuum=False)
        if release:
            assert result["ok"]
            assert len(db.search_messages("needle0")) == 1
        else:
            assert result == {"ok": False, "reason": "fts_error", "vacuumed": None}
            assert clock.now == db._WRITE_PATIENCE_S
            assert max(begins) < db._WRITE_PATIENCE_S
            assert starts == ([0, 19] if pause == 1 else [0])
            assert db.get_meta("fts_rebuild_progress") == "0"
        assert len(db.get_messages("s")) == 5
        assert not db._conn.in_transaction
    finally:
        blocker.rollback()
        blocker.close()
        db._conn.set_trace_callback(None)
    if not release:
        monkeypatch.setattr(db, "fts_rebuild_step", original)
        assert db.optimize_fts_storage(vacuum=False)["ok"]
        db._conn.execute("INSERT INTO messages_fts(messages_fts,rank) VALUES('integrity-check',1)")


@pytest.mark.parametrize("expired", [False, True])
def test_write_boundary_admission(pending_db, monkeypatch, expired):
    db = pending_db
    calls = []
    clock = Clock()
    monkeypatch.setattr(hermes_state, "time", clock)
    if expired:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            db._execute_write(lambda c: calls.append(1), deadline=clock.now)
        assert calls == []
    else:
        db._execute_write(lambda c: calls.append(1), patience_s=0)
        assert calls == [1]


@pytest.mark.parametrize("phase", ["base", "cjk", "trash"])
@pytest.mark.parametrize("fault", ["busy_column", "syntax", "ioerr", "cancel", "generation", "quarantine"])
def test_chunk_fault_rolls_back_actual_mutations(pending_db, monkeypatch, phase, fault):
    from hermes_state_errors import StateDbCorruptError, StateDbReplacedError

    db = pending_db
    if phase == "base":
        action = db.fts_rebuild_step
        marker = "fts_rebuild_progress"
    elif phase == "cjk":
        db._execute_write(lambda c: c.execute("CREATE TABLE fixture_cjk(content,tool_name,tool_calls)"))
        db.set_meta("fts_cjk_rebuild_high_water", "5")
        db.set_meta("fts_cjk_rebuild_progress", "0")
        marker = "fts_cjk_rebuild_progress"
        def action():
            return db._rebuild_step(
                "fts_cjk_rebuild", [db._CHUNK_INSERT_SQL.format(table="fixture_cjk", extra="")],
                fail_msg="test %s", finish=lambda: None,
            )
    else:
        db._execute_write(lambda c: c.execute("CREATE TABLE fts_v22_trash_fixture(id INTEGER PRIMARY KEY)"))
        db._execute_write(lambda c: c.executemany("INSERT INTO fts_v22_trash_fixture VALUES(?)", [(1,), (2,)]))
        action = db._fts_teardown_trash_step
        marker = "fts_teardown_fts_v22_trash_fixture_progress"
    before = db.get_meta(marker)
    messages = db.get_messages("s")
    write = db._execute_write
    calls = []
    errors = {"ioerr": sqlite3.OperationalError, "cancel": KeyboardInterrupt,
              "generation": StateDbReplacedError, "quarantine": StateDbCorruptError}
    def failing_write(fn, **kwargs):
        def work(conn):
            calls.append(1)
            assert len(calls) == 1, "a permanent failure must never replay"
            fn(conn)  # Actual INSERT/DELETE and its progress update precede the fault.
            assert conn.execute("SELECT value FROM state_meta WHERE key=?", (marker,)).fetchone()[0] != before
            if fault == "busy_column":
                conn.execute("SELECT busy")
            elif fault == "syntax":
                conn.execute("SELEC broken")
            else:
                raise errors[fault]("disk I/O error" if fault == "ioerr" else "stop")
        return write(work, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(db, "_execute_write", failing_write)
        with pytest.raises(errors.get(fault, sqlite3.OperationalError)):
            action()
    assert calls == [1]
    assert db.get_meta(marker) == before
    assert db.get_messages("s") == messages
    assert not db._conn.in_transaction
    if phase == "cjk":
        assert db._conn.execute("SELECT count(*) FROM fixture_cjk").fetchone()[0] == 0
    elif phase == "trash":
        assert db._conn.execute("SELECT count(*) FROM fts_v22_trash_fixture").fetchone()[0] == 2
    else:
        db._FTS_REBUILD_MIN_PAUSE = 0
        assert db.optimize_fts_storage(vacuum=False)["ok"]
        db._conn.execute("INSERT INTO messages_fts(messages_fts,rank) VALUES('integrity-check',1)")
