"""Regression contracts for #183: advisory waits never renew caller patience."""

import fcntl
from pathlib import Path
import sqlite3

import pytest

import hermes_state
from hermes_state import SessionDB


@pytest.fixture
def database(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        yield db


def test_activity_postcommit_checkpoint_does_not_wait(database, monkeypatch):
    database.create_session("deadline-witness", source="cli")
    database._write_count = database._CHECKPOINT_EVERY_N_WRITES - 1
    database._WRITE_PATIENCE_S = 0.3
    database._ACTIVITY_WRITE_PATIENCE_S = 0.03
    clock = [0.0]
    sleeps = []
    checkpoint = database._try_wal_checkpoint
    observed = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    def competitor_wins_gap():
        # Only schedule the rival; the checkpoint, transaction and flock stay real.
        with Path(f"{database.db_path}.write.lock").open("a+b") as holder:
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with sqlite3.connect(database.db_path) as reader:
                observed.append(reader.execute(
                    "SELECT last_activity_at, last_activity_description FROM sessions WHERE id = ?",
                    ("deadline-witness",),
                ).fetchone())
            checkpoint()

    monkeypatch.setattr(hermes_state.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(hermes_state.time, "sleep", sleep)
    monkeypatch.setattr(database, "_try_wal_checkpoint", competitor_wins_gap)
    database.touch_session_activity("deadline-witness", ts=123.0, description="durable")
    assert observed == [(123.0, "durable")]
    assert sleeps == [], "best-effort checkpoint waited after the activity was committed"
    # The skipped checkpoint neither replays nor strands the committed write/lock.
    database.clear_session_activity_labels("deadline-witness")
    assert database.get_session("deadline-witness")["last_activity_description"] == ""


def test_schema_retry_shares_original_advisory_deadline(database, monkeypatch):
    database.close()
    database._WRITE_PATIENCE_S = 0.2
    clock = [0.0]
    opens = []
    with Path(f"{database.db_path}.write.lock").open("a+b") as holder:
        held = False

        def opener():
            opens.append(clock[0])
            clock[0] += 0.15
            raise sqlite3.OperationalError("database is locked")

        def sleep(seconds):
            nonlocal held
            clock[0] += seconds
            if not held:
                fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True

        monkeypatch.setattr(database, "_open_writer_conn", opener)
        monkeypatch.setattr(hermes_state.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(hermes_state.time, "sleep", sleep)
        monkeypatch.setattr(hermes_state.random, "uniform", lambda *_args: 0.01)
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            database._connect_and_init_with_lock_patience()
        assert opens == [0.0]
        assert clock[0] == pytest.approx(0.2), "retry renewed the aggregate sidecar budget"
