"""Tests for fail-closed state.db NOTADB handling and journal-mode EIO retries.

Covers the two independently-valuable pieces salvaged from the state.db
hardening rollup:

* fail closed when a live write connection reports ``file is not a database``;
* transient ``disk i/o error`` retry in ``_on_disk_journal_mode`` so a
  one-shot EIO doesn't push callers onto the fail-closed unknown-mode branch.
"""

import os
import shutil
import sqlite3
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import MagicMock

import pytest

import hermes_state
from hermes_state import SessionDB, _on_disk_journal_mode


def _make_valid_replacement(path):
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE replacement_meta (value TEXT NOT NULL)")
        conn.execute("INSERT INTO replacement_meta VALUES ('replacement')")
        conn.commit()
    finally:
        conn.close()


class _NotADbOnce:
    """Connection proxy that raises 'file is not a database' on execute."""

    def __init__(self, real_conn):
        self._real = real_conn

    def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.mark.parametrize(
    ("path", "mode", "expected"),
    [
        (
            PurePosixPath("/tmp/state?db.db"),
            "rw",
            "file:///tmp/state%3Fdb.db?mode=rw",
        ),
        (
            PureWindowsPath(r"C:\tmp\state#db.db"),
            "ro",
            "file:///C:/tmp/state%23db.db?mode=ro",
        ),
        (
            PureWindowsPath(r"\\server\share\state%db.db"),
            "rw",
            "file://server/share/state%25db.db?mode=rw",
        ),
    ],
)
def test_sqlite_file_uri_escapes_path_component(path, mode, expected):
    assert hermes_state._sqlite_file_uri(path, mode) == expected


_SPECIAL_DB_FILENAMES = (
    "state?db.db",
    "state#db.db",
    "state%db.db",
    "state db.db",
    "state–db.db",
    "state[db].db",
    "state&=db.db",
)


class TestFailClosedAfterNotADb:
    def test_write_does_not_reopen_after_connection_identity_breaks(
        self, tmp_path, monkeypatch
    ):
        """One connection cannot safely heal a shared DB identity change."""
        db = SessionDB(db_path=tmp_path / "state.db")
        real_conn = db._conn
        try:
            db.create_session(session_id="s1", source="cli", model="test")
            reopen = MagicMock()
            monkeypatch.setattr("hermes_state._connect_tracked_db", reopen)
            db._conn = _NotADbOnce(real_conn)
            with pytest.raises(sqlite3.DatabaseError, match="not a database"):
                db.create_session(session_id="s2", source="cli", model="test")
            reopen.assert_not_called()
        finally:
            db._conn = real_conn
            db.close()

    def test_explicit_reconnect_rejects_atomic_replacement_without_mutation(
        self, tmp_path
    ):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        old_conn = db._conn
        expected_identity = db._db_identity
        replacement = tmp_path / "replacement.db"
        _make_valid_replacement(replacement)
        os.replace(replacement, db_path)
        replacement_bytes = db_path.read_bytes()
        replacement_stat = db_path.stat()
        try:
            assert db._reconnect_after_notadb() is False
            assert db._conn is None
            assert db._db_identity == expected_identity
            assert db_path.read_bytes() == replacement_bytes
            assert db_path.stat().st_ino == replacement_stat.st_ino
            assert db_path.stat().st_mtime_ns == replacement_stat.st_mtime_ns
            with sqlite3.connect(db_path) as conn:
                assert conn.execute(
                    "SELECT value FROM replacement_meta"
                ).fetchone()[0] == "replacement"
            with pytest.raises(sqlite3.ProgrammingError):
                old_conn.execute("SELECT 1")
        finally:
            db.close()

    def test_explicit_reconnect_accepts_repaired_bytes_with_same_identity(
        self, tmp_path
    ):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        old_conn = db._conn
        expected_identity = db._db_identity
        repaired = tmp_path / "repaired.db"
        _make_valid_replacement(repaired)
        shutil.copyfile(repaired, db_path)
        try:
            assert db._reconnect_after_notadb() is True
            assert db._conn is not None
            assert db._conn is not old_conn
            assert db._db_identity == expected_identity
            assert db._conn.execute(
                "SELECT value FROM replacement_meta"
            ).fetchone()[0] == "replacement"
        finally:
            db.close()

    @pytest.mark.parametrize("filename", _SPECIAL_DB_FILENAMES)
    def test_explicit_reconnect_preserves_special_path(self, tmp_path, filename):
        db_path = tmp_path / filename
        db = SessionDB(db_path=db_path)
        wrong_paths = {
            candidate
            for candidate in (
                tmp_path / "state",
                tmp_path / "state.db",
                tmp_path / "state%db.db",
            )
            if candidate != db_path
        }
        wrong_before = {
            path: path.read_bytes() if path.exists() else None
            for path in wrong_paths
        }
        try:
            assert db._reconnect_after_notadb() is True
            assert db._conn is not None
            opened_path = Path(
                db._conn.execute("PRAGMA database_list").fetchone()[2]
            ).resolve()
            assert opened_path == db_path.resolve()
            for path, before in wrong_before.items():
                assert path.exists() is (before is not None)
                if before is not None:
                    assert path.read_bytes() == before
        finally:
            db.close()

    @pytest.mark.parametrize("filename", _SPECIAL_DB_FILENAMES)
    def test_explicit_reconnect_rejects_swap_for_special_path(
        self, tmp_path, monkeypatch, filename
    ):
        db_path = tmp_path / filename
        db = SessionDB(db_path=db_path)
        original_bytes = db_path.read_bytes()
        replacement = tmp_path / "replacement.db"
        _make_valid_replacement(replacement)
        replacement_bytes = replacement.read_bytes()
        wrong_paths = {
            candidate
            for candidate in (
                tmp_path / "state",
                tmp_path / "state.db",
                tmp_path / "state%db.db",
            )
            if candidate != db_path
        }
        wrong_before = {
            path: path.read_bytes() if path.exists() else None
            for path in wrong_paths
        }
        real_connect = hermes_state._connect_tracked_db
        calls = 0

        def connect_and_swap(path, **kwargs):
            nonlocal calls
            calls += 1
            conn = real_connect(path, **kwargs)
            if calls == 2:
                os.replace(replacement, db_path)
            return conn

        monkeypatch.setattr(hermes_state, "_connect_tracked_db", connect_and_swap)
        try:
            assert db._reconnect_after_notadb() is False
            assert calls == 2
            assert db._conn is None
            assert db_path.read_bytes() == replacement_bytes
            assert original_bytes != replacement_bytes
            for path, before in wrong_before.items():
                assert path.exists() is (before is not None)
                if before is not None:
                    assert path.read_bytes() == before
        finally:
            db.close()

    def test_explicit_reconnect_rejects_swap_after_new_handle_opens(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        expected_identity = db._db_identity
        replacement = tmp_path / "replacement.db"
        _make_valid_replacement(replacement)
        replacement_bytes = replacement.read_bytes()
        real_connect = hermes_state._connect_tracked_db
        calls = 0

        def connect_and_swap(path, **kwargs):
            nonlocal calls
            calls += 1
            conn = real_connect(path, **kwargs)
            if calls == 2:
                os.replace(replacement, db_path)
            return conn

        monkeypatch.setattr(hermes_state, "_connect_tracked_db", connect_and_swap)
        try:
            assert db._reconnect_after_notadb() is False
            assert calls == 2
            assert db._conn is None
            assert db._db_identity == expected_identity
            assert db_path.read_bytes() == replacement_bytes

        finally:
            db.close()

    def test_explicit_reconnect_rejects_missing_path_after_new_handle_opens(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        expected_identity = db._db_identity
        real_connect = hermes_state._connect_tracked_db
        calls = 0

        def connect_then_unlink(path, **kwargs):
            nonlocal calls
            calls += 1
            conn = real_connect(path, **kwargs)
            if calls == 2:
                db_path.unlink()
            return conn

        monkeypatch.setattr(hermes_state, "_connect_tracked_db", connect_then_unlink)
        try:
            assert db._reconnect_after_notadb() is False
            assert calls == 2
            assert db._conn is None
            assert db._db_identity == expected_identity
            assert not db_path.exists()
        finally:
            db.close()


class TestOnDiskJournalModeEioRetry:
    def _conn_raising_then(self, failures, result_rows):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = result_rows
        conn.execute.side_effect = list(failures) + [cursor]
        return conn

    def test_transient_eio_clears_on_retry(self):
        conn = self._conn_raising_then(
            [sqlite3.OperationalError("disk i/o error")] * 2, ("wal",)
        )
        assert _on_disk_journal_mode(conn) == "wal"

    def test_persistent_eio_returns_none(self):
        conn = MagicMock()
        conn.execute.side_effect = sqlite3.OperationalError("disk i/o error")
        assert _on_disk_journal_mode(conn) is None
        # Bounded: retried a handful of times, not forever.
        assert conn.execute.call_count == 4

    def test_non_eio_operational_error_fails_fast(self):
        conn = MagicMock()
        conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        assert _on_disk_journal_mode(conn) is None
        assert conn.execute.call_count == 1
