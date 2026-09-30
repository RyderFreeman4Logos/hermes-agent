"""Resource identity and scratch recovery admission regressions."""
import contextlib
import os
import threading

import pytest

from hermes_cli import sqlite_safe_read as safe
from hermes_state import SessionDB
import hermes_state_repair as repair
from hermes_state_registry import acquire, release as registry_release
def _stale_owner(tmp_path, monkeypatch, *, backup=False):
    path = tmp_path / "state.db"
    monkeypatch.setattr("hermes_state_wal.is_sqlite_wal_reset_vulnerable", lambda **kwargs: False)
    with SessionDB(db_path=path) as seed:
        seed.create_session("s", source="cli")
        seed.append_message("s", role="user", content="seed")
        seed._conn.execute("UPDATE messages_fts_data SET block=X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
    if backup:
        path.with_name("state.db.malformed-backup-20000101").write_bytes(path.read_bytes())
    with monkeypatch.context() as opening:
        opening.setattr(SessionDB, "_foreign_state_db_holders", lambda self: [(222, str(path))])
        return SessionDB(db_path=path)


@pytest.mark.parametrize("alias_kind", ["hardlink", "rename", "symlink"])
def test_alias_remains_live_and_refuses_offline_access(tmp_path, alias_kind):
    original, alias = tmp_path / "original.db", tmp_path / "alias.db"
    with contextlib.closing(safe.connect_tracked(original)) as conn:
        conn.execute("CREATE TABLE payload(value)")
        if alias_kind == "hardlink":
            os.link(original, alias)
        elif alias_kind == "rename":
            original.rename(alias)
        else:
            alias.symlink_to(original)
        assert safe.has_live_connection(alias)
        with pytest.raises(safe.LiveConnectionError):
            with safe.offline_file_access(alias):
                pytest.fail("raw alias admitted")


def test_hardlink_peer_refuses_handoff(tmp_path):
    original, alias = tmp_path / "state.db", tmp_path / "alias.db"
    with contextlib.ExitStack() as stack:
        owner = stack.enter_context(contextlib.closing(safe.connect_tracked(original)))
        owner.execute("CREATE TABLE payload(value)")
        os.link(original, alias)
        stack.enter_context(contextlib.closing(safe.connect_tracked(alias)))
        with safe.connection_handoff(original, owner) as admitted:
            assert not admitted


def test_old_generation_close_cannot_untrack_replacement(tmp_path):
    original, renamed = tmp_path / "state.db", tmp_path / "retired.db"
    old = safe.connect_tracked(original)
    original.rename(renamed)
    with contextlib.closing(safe.connect_tracked(original)) as current:
        old.close()
        assert safe.has_live_connection(original)
        assert not safe.has_live_connection(renamed)
        with safe.connection_handoff(original, current) as admitted:
            assert admitted


def test_reservation_is_explicit_and_blocks_original_and_renamed_inode(tmp_path):
    original, alias = tmp_path / "state.db", tmp_path / "renamed.db"
    with contextlib.closing(safe.connect_tracked(original)) as owner:
        with safe.connection_handoff(original, owner) as admitted:
            assert admitted
            owner.close()
            original.rename(alias)
            # Even the reserving thread cannot silently acquire an extra handle.
            for path in (original, alias):
                with pytest.raises(safe.LiveConnectionError):
                    safe.connect_tracked(path, timeout=0)
                with pytest.raises(safe.LiveConnectionError):
                    with safe.offline_file_access(path):
                        pytest.fail("reservation bypassed")


@pytest.mark.parametrize("stage", ["forensic", "forensic-backup", "staging", "scratch", "comparison", "promotion", "reopen"])
def test_recovery_does_not_block_unrelated_lifecycle(tmp_path, monkeypatch, stage):
    with _stale_owner(tmp_path, monkeypatch, backup=stage.startswith("forensic")) as db:
        unrelated = tmp_path / "unrelated.db"
        idle = safe.connect_tracked(unrelated, check_same_thread=False)
        entered, release, completed = threading.Event(), threading.Event(), threading.Event()
        results, errors = [], []
        def park():
            entered.set()
            assert release.wait(10)
        if stage.startswith("forensic"):
            original = repair._backup_content_identity
            def parked(path):
                # Park INSIDE sanctioned raw ownership, not before its lock entrance.
                original_reader = repair._read_offline
                def read_offline(path, what, reader):
                    def read():
                        if (path == db.db_path) == (stage == "forensic"):
                            park()
                        return reader()
                    return original_reader(path, what, read)
                with monkeypatch.context() as patch:
                    patch.setattr(repair, "_read_offline", read_offline)
                    return original(path)
            monkeypatch.setattr(repair, "_backup_content_identity", parked)
        elif stage == "scratch":
            original = repair._strategy_drop_fts_vacuum
            def parked(conn):
                park()
                return original(conn)
            monkeypatch.setattr(repair, "_strategy_drop_fts_vacuum", parked)
        elif stage == "comparison":
            original = repair._validate_fts_snapshot
            def parked(*args):
                park()
                return original(*args)
            monkeypatch.setattr(repair, "_validate_fts_snapshot", parked)
        elif stage in {"staging", "promotion"}:
            original = repair._copy_database_snapshot
            def parked(*args, **kwargs):
                if ("source_connection" in kwargs) == (stage == "staging"):
                    park()
                return original(*args, **kwargs)
            monkeypatch.setattr(repair, "_copy_database_snapshot", parked)
        else:
            original = db._open_writer_conn
            def parked(**kwargs):
                park()
                return original(**kwargs)
            monkeypatch.setattr(db, "_open_writer_conn", parked)
        def recover():
            try:
                results.append(db.retry_deferred_fts_recovery())
            except BaseException as exc:
                errors.append(repr(exc))
        def unrelated_work():
            try:
                with pytest.raises(safe.LiveConnectionError):
                    safe.connect_tracked(db.db_path, timeout=0)
                with pytest.raises(safe.LiveConnectionError):
                    with safe.offline_file_access(db.db_path):
                        pytest.fail("target reservation lost during slow work")
                idle.close()
                with contextlib.closing(safe.connect_tracked(unrelated, timeout=0)) as conn:
                    conn.execute("SELECT 1")
                shared = acquire(db_path=tmp_path / "shared.db")
                try:
                    reader = shared._checkout_read_conn()
                    assert reader is not None
                    shared._read_pool.put_nowait(reader)
                    assert shared._evict_one_idle_read_conn()
                finally:
                    assert registry_release(shared)
                completed.set()
            except BaseException as exc:
                errors.append(repr(exc))
        owner, peer = threading.Thread(target=recover), threading.Thread(target=unrelated_work)
        owner.start()
        try:
            assert entered.wait(10)
            peer.start()
            progressed = completed.wait(2)
        finally:
            release.set()
            owner.join(10)
            if peer.ident is not None:
                peer.join(10)
            idle.close()
        assert not owner.is_alive() and not peer.is_alive()
        assert not errors, errors
        assert progressed, f"{stage} blocked unrelated open/close"
        assert results == [True]
