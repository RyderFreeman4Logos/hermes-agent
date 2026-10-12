"""Non-lock OperationalError must not spin the FTS optimize driver (#349)."""

import sqlite3

from hermes_state import SessionDB


def _pending(db, monkeypatch):
    db.set_meta("fts_rebuild_high_water", "1")
    db.set_meta("fts_rebuild_progress", "0")
    db._FTS_REBUILD_MIN_PAUSE = 0.0
    db._FTS_REBUILD_DUTY_FACTOR = 0.0
    db._WRITE_PATIENCE_S = 0.05


def test_nonlock_operational_error_stops_optimize(tmp_path, monkeypatch):
    """A deterministic schema error is not lock contention and must not be retried."""
    db = SessionDB(db_path=tmp_path / "state.db")
    orig = db._execute_write
    calls = {"n": 0}

    def _raise(_do, patience_s=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return orig(_do, patience_s=patience_s)
        raise sqlite3.OperationalError("no such column: missing")

    try:
        _pending(db, monkeypatch)
        monkeypatch.setattr(db, "_execute_write", _raise)
        result = db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()

    assert calls["n"] == 2
    assert result == {"ok": False, "reason": "fts_error", "vacuumed": None}


def test_lock_error_retries_then_succeeds(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    orig = db._execute_write
    calls = {"n": 0}

    def _locked_once(_do, patience_s=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("database is locked")
        return orig(_do, patience_s=patience_s)

    try:
        _pending(db, monkeypatch)
        monkeypatch.setattr(db, "_execute_write", _locked_once)
        result = db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()

    assert calls["n"] >= 3
    assert result["ok"] is True


def test_repeated_lock_error_stops_at_write_patience(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    orig = db._execute_write
    calls = {"n": 0}

    def _always_locked(_do, patience_s=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return orig(_do, patience_s=patience_s)
        raise sqlite3.OperationalError("database is locked")

    try:
        _pending(db, monkeypatch)
        monkeypatch.setattr(db, "_execute_write", _always_locked)
        result = db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()

    assert calls["n"] > 2
    assert result == {"ok": False, "reason": "fts_error", "vacuumed": None}
