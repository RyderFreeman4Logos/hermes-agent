"""Conservative identity custody and admission-edge regressions."""
import contextlib
import os
import sqlite3
import threading

import pytest

from hermes_cli import sqlite_safe_read as safe


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_late_sidecar_alias_is_fenced(tmp_path, suffix):
    path, alias = tmp_path / "state.db", tmp_path / "aliased-sidecar"
    with contextlib.closing(safe.connect_tracked(path)) as owner:
        owner.execute("PRAGMA journal_mode=WAL")
        owner.execute("CREATE TABLE payload(value)")
        os.link(str(path) + suffix, alias)
        assert safe.has_live_connection(alias)
        with pytest.raises(safe.LiveConnectionError):
            with safe.offline_file_access(alias):
                pytest.fail("live sidecar admitted")
        with safe.connection_handoff(path, owner) as claim:
            assert claim
            with pytest.raises(safe.LiveConnectionError):
                safe.connect_tracked(alias)


def test_tracking_path_is_not_actual_resource_authority(tmp_path):
    actual, claimed = tmp_path / "actual.db", tmp_path / "claimed.db"
    with contextlib.closing(safe.connect_tracked(claimed)):
        pass
    with contextlib.closing(safe.connect_tracked(actual, tracking_path=claimed)) as conn:
        assert safe.has_live_connection(actual) and safe.has_live_connection(claimed)
        for path in (actual, claimed):
            with safe.connection_handoff(path, conn) as claim:
                assert not claim
    assert not safe.has_live_connection(actual) and not safe.has_live_connection(claimed)


def test_retarget_during_open_retains_both_candidates_but_grants_no_authority(tmp_path):
    old, new, alias = tmp_path / "old.db", tmp_path / "new.db", tmp_path / "alias.db"
    with contextlib.closing(safe.connect_tracked(old)), contextlib.closing(safe.connect_tracked(new)):
        pass
    alias.symlink_to(old)
    def retarget(path, **kwargs):
        conn = sqlite3.connect(path, **kwargs)
        alias.unlink()
        alias.symlink_to(new)
        return conn
    with contextlib.closing(safe.connect_tracked(alias, connect_fn=retarget)) as conn:
        assert safe.has_live_connection(old) and safe.has_live_connection(new)
        for path in (old, new, alias):
            with safe.connection_handoff(path, conn) as claim:
                assert not claim
    assert not safe.has_live_connection(old) and not safe.has_live_connection(new)


@pytest.mark.parametrize("seam", ["before-open", "after-open"])
def test_pending_physical_open_cannot_enter_sole_owner_handoff(tmp_path, seam):
    path = tmp_path / "state.db"
    entered, release = threading.Event(), threading.Event()
    peers, errors = [], []
    with contextlib.closing(safe.connect_tracked(path)) as owner:
        def opener(path, **kwargs):
            if seam == "before-open":
                entered.set(); assert release.wait(10)
            conn = sqlite3.connect(path, **kwargs)
            if seam == "after-open":
                entered.set(); assert release.wait(10)
            return conn
        def pending():
            try:
                peers.append(safe.connect_tracked(path, connect_fn=opener, check_same_thread=False))
            except BaseException as exc:
                errors.append(repr(exc))
        worker = threading.Thread(target=pending)
        worker.start()
        try:
            assert entered.wait(10)
            with safe.connection_handoff(path, owner) as claim:
                assert not claim
        finally:
            release.set(); worker.join(10)
        assert not worker.is_alive() and not errors
        for conn in peers:
            conn.close()
        with safe.connection_handoff(path, owner) as claim:
            assert claim


def test_failed_explicit_owner_open_settles_pending_admission(tmp_path):
    path = tmp_path / "state.db"
    with contextlib.closing(safe.connect_tracked(path)) as owner:
        with safe.connection_handoff(path, owner) as claim:
            assert claim
            owner.close()
            def cancelled(*args, **kwargs):
                raise KeyboardInterrupt("opener cancelled")
            with pytest.raises(KeyboardInterrupt):
                safe.connect_tracked(path, handoff=claim, connect_fn=cancelled)
            assert claim.pending == 0
            with contextlib.closing(safe.connect_tracked(path, handoff=claim)) as reopened:
                assert reopened.execute("SELECT 1").fetchone() == (1,)
    assert not safe.has_live_connection(path)
    assert not safe._reservations


def test_unknown_actual_disk_resource_is_tracked_but_never_permission(tmp_path, monkeypatch):
    path, other = tmp_path / 'unknown.db', tmp_path / 'other.db'
    monkeypatch.setattr(safe, '_canonical_db_path', lambda conn: None)
    with contextlib.closing(safe.connect_tracked(path)) as conn:
        assert safe.has_live_connection(path)
        with safe.connection_handoff(path, conn) as claim:
            assert not claim
        with pytest.raises(safe.LiveConnectionError):
            with safe.offline_file_access(other):
                pytest.fail('unknown disk descriptor cannot be localized safely')
    assert not safe.has_live_connection(path)
