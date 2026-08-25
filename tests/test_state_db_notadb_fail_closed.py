"""Tests for fail-closed state.db NOTADB handling and journal-mode EIO retries.

Covers the two independently-valuable pieces salvaged from the state.db
hardening rollup:

* fail closed when a live write connection reports ``file is not a database``;
* transient ``disk i/o error`` retry in ``_on_disk_journal_mode`` so a
  one-shot EIO doesn't push callers onto the fail-closed unknown-mode branch.
"""

import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import hermes_state
from hermes_state import SessionDB, _on_disk_journal_mode


class _NotADbOnce:
    """Connection proxy that raises 'file is not a database' on execute."""

    def __init__(self, real_conn):
        self._real = real_conn

    def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def __getattr__(self, name):
        return getattr(self._real, name)


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

    @pytest.mark.linux_only
    def test_sqlite_fullpath_drift_is_rejected_before_database_pragmas(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state.db"
        replacement = tmp_path / "replacement.db"
        for path, value in ((db_path, "original"), (replacement, "replacement")):
            db = SessionDB(db_path=path)
            db.set_meta("identity", value)
            db.close()

        real_connect = hermes_state.sqlite3.connect
        ordinary_called = False

        def connect_replacement(_path, *args, **kwargs):
            return real_connect(replacement, *args, **kwargs)

        def ordinary_pragma(*_args, **_kwargs):
            nonlocal ordinary_called
            ordinary_called = True
            raise AssertionError("database work ran before fullPath verification")

        monkeypatch.setattr(hermes_state.sqlite3, "connect", connect_replacement)
        monkeypatch.setattr(hermes_state, "apply_database_pragmas", ordinary_pragma)

        with pytest.raises(sqlite3.DatabaseError, match="identity verification"):
            SessionDB(db_path=db_path, read_only=True)
        assert not ordinary_called

    @pytest.mark.linux_only
    def test_parent_namespace_drift_is_rejected_before_database_pragmas(
        self, tmp_path, monkeypatch
    ):
        outer = tmp_path / "outer"
        parent = outer / "parent"
        parent.mkdir(parents=True)
        db_path = parent / "state.db"
        db = SessionDB(db_path=db_path)
        db.close()

        replacement = outer / "replacement"
        replacement.mkdir()
        (replacement / "state.db").write_bytes(db_path.read_bytes())
        parked = outer / "parked"
        real_connect = hermes_state.sqlite3.connect
        ordinary_called = False

        def connect_then_swap(path, *args, **kwargs):
            conn = real_connect(path, *args, **kwargs)
            os.replace(parent, parked)
            os.replace(replacement, parent)
            return conn

        def ordinary_pragma(*_args, **_kwargs):
            nonlocal ordinary_called
            ordinary_called = True
            raise AssertionError("database work ran before namespace verification")

        monkeypatch.setattr(hermes_state.sqlite3, "connect", connect_then_swap)
        monkeypatch.setattr(hermes_state, "apply_database_pragmas", ordinary_pragma)
        try:
            with pytest.raises(sqlite3.DatabaseError, match="identity verification"):
                SessionDB(db_path=db_path, read_only=True)
        finally:
            os.replace(parent, replacement)
            os.replace(parked, parent)
        assert not ordinary_called

    @pytest.mark.linux_only
    def test_linux_connections_do_not_claim_connection_owned_inode(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        reader = None
        try:
            assert db._conn is not None
            assert not hasattr(db._conn, "_hermes_db_identity")
            if db._wal_active:
                reader = db._get_read_conn()
                assert reader is not None
                assert not hasattr(reader, "_hermes_db_identity")
        finally:
            if reader is not None:
                db._close_read_conn(reader)
            db.close()

    @pytest.mark.linux_only
    def test_linux_writer_read_only_pool_and_wal_remain_usable(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state?db.db"
        monkeypatch.setattr(
            hermes_state,
            "is_sqlite_wal_reset_vulnerable",
            lambda version_info=None: False,
        )
        writer = SessionDB(db_path=db_path)
        pooled = read_only = None
        try:
            writer.set_meta("identity", "ordinary")
            assert writer._wal_active
            assert writer._conn.execute("PRAGMA database_list").fetchone()[2] == str(
                db_path
            )
            assert Path(f"{db_path}-wal").exists()

            pooled = writer._get_read_conn()
            assert pooled is not None
            assert (
                pooled.execute(
                    "SELECT value FROM state_meta WHERE key = 'identity'"
                ).fetchone()[0]
                == "ordinary"
            )

            read_only = SessionDB(db_path=db_path, read_only=True)
            assert read_only.get_meta("identity") == "ordinary"
        finally:
            if pooled is not None:
                writer._close_read_conn(pooled)
            if read_only is not None:
                read_only.close()
            writer.close()

    @pytest.mark.linux_only
    @pytest.mark.parametrize("symlink_parent", [False, True])
    def test_linux_open_rejects_symlinked_namespace(self, tmp_path, symlink_parent):
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        target = target_dir / "state.db"
        db = SessionDB(db_path=target)
        db.close()

        if symlink_parent:
            link_dir = tmp_path / "linked-parent"
            link_dir.symlink_to(target_dir.name, target_is_directory=True)
            requested = link_dir / target.name
        else:
            requested = tmp_path / "linked-state.db"
            requested.symlink_to(target)

        with pytest.raises(sqlite3.DatabaseError, match="identity verification"):
            SessionDB(db_path=requested, read_only=True)


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
