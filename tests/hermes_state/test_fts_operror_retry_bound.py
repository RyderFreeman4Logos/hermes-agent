"""FTS maintenance retries lock contention without spinning on permanent errors."""

import sqlite3

from hermes_state import SessionDB


def _pending(db, monkeypatch):
    db.set_meta("fts_rebuild_high_water", "1")
    db.set_meta("fts_rebuild_progress", "0")
    db._FTS_REBUILD_MIN_PAUSE = 0.0
    db._FTS_REBUILD_DUTY_FACTOR = 0.0
    db._WRITE_PATIENCE_S = 0.05


def test_nonlock_operational_error_stops_optimize(tmp_path, monkeypatch):
    """A permanent schema error stops the driver after one failed chunk."""
    db = SessionDB(db_path=tmp_path / "state.db")
    original = db._execute_write
    calls = {"n": 0}

    def _fail_once(_do, patience_s=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return original(_do, patience_s=patience_s)
        if calls["n"] == 2:
            raise sqlite3.OperationalError("no such column: missing")
        return original(_do, patience_s=patience_s)

    try:
        _pending(db, monkeypatch)
        monkeypatch.setattr(db, "_execute_write", _fail_once)
        result = db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()

    assert calls["n"] == 2
    assert result == {"ok": False, "reason": "fts_error", "vacuumed": None}


def test_lock_error_retries_then_succeeds(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    original = db._execute_write
    calls = {"n": 0}

    def _locked_once(_do, patience_s=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("database is locked")
        return original(_do, patience_s=patience_s)

    try:
        _pending(db, monkeypatch)
        monkeypatch.setattr(db, "_execute_write", _locked_once)
        result = db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()

    assert calls["n"] >= 3
    assert result["ok"] is True


def test_repeated_lock_error_stops_at_write_patience(tmp_path, monkeypatch):
    """Repeated transient failures stop at the shared write-patience deadline."""
    db = SessionDB(db_path=tmp_path / "state.db")
    original = db._execute_write
    calls = {"n": 0}
    clock = {"now": 0.0}

    def _monotonic():
        clock["now"] += 0.1
        return clock["now"]

    def _locked_until_bound(_do, patience_s=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return original(_do, patience_s=patience_s)
        if calls["n"] <= 8:
            raise sqlite3.OperationalError("database is locked")
        return original(_do, patience_s=patience_s)

    try:
        _pending(db, monkeypatch)
        monkeypatch.setattr(db, "_execute_write", _locked_until_bound)
        monkeypatch.setattr("hermes_state_search.time.monotonic", _monotonic)
        monkeypatch.setattr("hermes_state_search.time.sleep", lambda _delay: None)
        result = db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()

    assert calls["n"] < 9
    assert result == {"ok": False, "reason": "fts_error", "vacuumed": None}
