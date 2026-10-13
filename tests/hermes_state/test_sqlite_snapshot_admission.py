"""Ordinary raw snapshots exclude opens, then release only their resource."""

import contextlib
import sqlite3
import threading

import pytest

from hermes_cli import sqlite_safe_read as safe


@pytest.mark.parametrize("failure", [None, "open", "setup", "snapshot"])
def test_snapshot_waits_then_settles_failed_and_successful_opens(tmp_path, monkeypatch, failure):
    path = tmp_path / "snapshot.db"
    other = tmp_path / "other.db"
    with contextlib.closing(safe.connect_tracked(path)):
        pass
    waiting = threading.Event()
    physical = threading.Event()
    errors = []
    real_connect = sqlite3.connect
    real_query_path = safe._canonical_db_path
    real_wait = safe._admission_changed.wait

    def wait(*args, **kwargs):
        waiting.set()
        return real_wait(*args, **kwargs)

    def opener(*args, **kwargs):
        if str(args[0]) == str(path):
            physical.set()
            if failure == "open":
                raise sqlite3.OperationalError("expected open failure")
        return real_connect(*args, **kwargs)

    def query_path(conn):
        actual = real_query_path(conn)
        if failure == "setup" and actual == str(path):
            raise RuntimeError("expected setup failure")
        return actual

    def connect():
        try:
            with contextlib.closing(safe.connect_tracked(path)) as conn:
                assert conn.execute("SELECT 1").fetchone() == (1,)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=connect)
    monkeypatch.setattr(safe._admission_changed, "wait", wait)
    # Patch the native binding, not an opaque custom opener: opaque identity
    # remains an immediate refusal whenever a reservation exists.
    monkeypatch.setattr(sqlite3, "connect", opener)
    monkeypatch.setattr(safe, "_canonical_db_path", query_path)
    try:
        with safe.offline_file_access(path):
            worker.start()
            assert waiting.wait(5)
            assert not physical.wait(0.1)
            # Unrelated progress must not depend on the snapshot being released.
            with contextlib.closing(safe.connect_tracked(other)) as conn:
                assert conn.execute("SELECT 1").fetchone() == (1,)
            with pytest.raises(safe.ConnectionAdmissionError):
                safe.connect_tracked(path)
            if failure == "snapshot":
                raise ValueError("expected snapshot failure")
    except ValueError:
        assert failure == "snapshot"
    finally:
        worker.join(5)
    assert not worker.is_alive()
    assert physical.is_set()
    assert [str(exc) for exc in errors] == (
        [f"expected {failure} failure"] if failure in {"open", "setup"} else []
    )
    assert not safe.has_live_connection(path)
    with safe.offline_file_access(path):
        pass
