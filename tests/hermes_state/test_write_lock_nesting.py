"""Real flock ownership regressions for constructor/FTS composition (#183)."""

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from hermes_state import SessionDB, _session_db_advisory_write_lock


def _lock(path):
    return _session_db_advisory_write_lock(
        path, deadline=time.monotonic(), patience_s=0.0
    )


def _rival_process(path):
    result = subprocess.run(
        [sys.executable, "-c", """
import sqlite3, sys, time
from pathlib import Path
from hermes_state import _session_db_advisory_write_lock
try:
    with _session_db_advisory_write_lock(Path(sys.argv[1]), deadline=time.monotonic(), patience_s=0):
        pass
except sqlite3.OperationalError:
    sys.exit(23)
""", str(path)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode in (0, 23), result.stderr
    return result.returncode


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_nested_resource_owner_retains_custody_and_releases(tmp_path, alias, raise_inside):
    path = tmp_path / "state.db"
    peer = path if alias == "same" else tmp_path / "alias.db"
    outcomes = []

    def rival_thread():
        try:
            with _lock(peer):
                outcomes.append("acquired")
        except sqlite3.OperationalError:
            outcomes.append("refused")

    with _lock(path):
        if alias == "symlink":
            Path(f"{peer}.write.lock").symlink_to(Path(f"{path}.write.lock"))
        elif alias == "hardlink":
            os.link(f"{path}.write.lock", f"{peer}.write.lock")
        try:
            with _lock(peer):
                thread = threading.Thread(target=rival_thread)
                thread.start()
                thread.join(10)
                assert not thread.is_alive()
                assert outcomes == ["refused"]
                assert _rival_process(peer) == 23
                if raise_inside:
                    raise ValueError("inner scope failed")
        except ValueError as exc:
            assert raise_inside and str(exc) == "inner scope failed"
        assert _rival_process(path) == 23, "inner exit released outer custody"
    assert _rival_process(peer) == 0, "outer exit stranded flock"
    with _lock(peer):
        with _lock(path):
            pass


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("table", ["messages_fts", "messages_fts_trigram"])
def test_corrupt_fts_detach_can_nest_under_existing_writer(tmp_path, table):
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("corrupt", source="cli")
        db.append_message("corrupt", role="user", content="original searchable text")
        db._conn.execute(
            f"UPDATE {table}_data SET block=X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF' WHERE id>10"
        )
        with pytest.raises(sqlite3.DatabaseError) as corrupt:
            db._conn.execute(
                f"SELECT rowid FROM {table} WHERE {table} MATCH 'hermes' LIMIT 1"
            ).fetchone()
        assert corrupt.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT_VTAB
        with db._advisory_write_lock(deadline=time.monotonic()):
            assert db._enter_fts_fail_open(corrupt.value, deadline=time.monotonic())
            assert _rival_process(path) == 23
        assert _rival_process(path) == 0
        assert db.get_meta("fts_stale") == "1"
        assert not db._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name LIKE 'messages_fts%'"
        ).fetchone()
        db.append_message("corrupt", role="user", content="canonical write survives")
        assert len(db.get_messages("corrupt")) == 2


@pytest.mark.platforms("linux")
def test_expired_recovery_refuses_replaced_sidecar_inode(tmp_path):
    from hermes_state import _advisory_write_owners

    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("corrupt", source="cli")
        db.append_message("corrupt", role="user", content="original searchable text")
        db._conn.execute(
            "UPDATE messages_fts_data SET block=X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF' WHERE id>10"
        )
        with pytest.raises(sqlite3.DatabaseError) as corrupt:
            db._conn.execute(
                "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'hermes' LIMIT 1"
            ).fetchone()
        assert corrupt.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT_VTAB
        lock_path = Path(f"{path}.write.lock")
        changed = []
        transaction_code = SessionDB._write_transaction.__wrapped__.__code__
        triggers_before = db._conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='trigger' AND name LIKE 'messages_fts%'"
        ).fetchone()[0]
        assert triggers_before > 0

        def replace_before_admission(frame, event, arg):
            # Schedule the real rename between recovery entry and FD admission;
            # never replace the ownership oracle, clock, transaction or SQLite.
            if event == "call" and frame.f_code is transaction_code and not changed:
                previous = lock_path.stat()
                lock_path.rename(Path(f"{lock_path}.retired"))
                with lock_path.open("xb"):
                    pass
                current = lock_path.stat()
                old_key = (os.getpid(), previous.st_dev, previous.st_ino)
                new_key = (os.getpid(), current.st_dev, current.st_ino)
                assert old_key in _advisory_write_owners.held
                assert new_key not in _advisory_write_owners.held
                assert old_key != new_key
                changed.append((old_key, new_key))
            return replace_before_admission

        with db._advisory_write_lock(deadline=time.monotonic()):
            deadline = time.monotonic()
            previous_trace = sys.gettrace()
            sys.settrace(replace_before_admission)
            try:
                result = db._enter_fts_fail_open(corrupt.value, deadline=deadline)
            finally:
                sys.settrace(previous_trace)
            assert changed, "replacement schedule did not run"
            assert time.monotonic() > deadline
            assert result is False, "expired recovery admitted a not-preowned sidecar inode"
            old_key, new_key = changed[0]
            assert old_key in _advisory_write_owners.held
            assert new_key not in _advisory_write_owners.held
            assert db.get_meta("fts_stale") is None
            assert db._conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='trigger' AND name LIKE 'messages_fts%'"
            ).fetchone()[0] == triggers_before
            assert not db._conn.in_transaction
            assert _rival_process(path) == 0, "refused new-inode admission stranded its flock"
        assert old_key not in _advisory_write_owners.held
