"""Physical-open admission cannot be delegated to an advisory tracking label."""
import contextlib
import os
import sqlite3

import pytest

from hermes_cli import sqlite_safe_read as safe


@pytest.mark.parametrize("spelling", ["path", "uri", "symlink", "hardlink", "custom", "factory"])
def test_reserved_actual_resource_refuses_before_opener(tmp_path, spelling):
    target, label, alias = (tmp_path / name for name in ("target.db", "label.db", "alias.db"))
    with contextlib.closing(safe.connect_tracked(target)):
        pass
    calls = []
    path, kwargs = target, {}
    if spelling == "uri":
        path, kwargs = target.as_uri() + "?mode=ro", {"uri": True}
    elif spelling == "symlink":
        alias.symlink_to(target)
        path = alias
    elif spelling == "hardlink":
        os.link(target, alias)
        path = alias
    elif spelling == "custom":
        def opener(ignored, **kw):
            calls.append(True)
            return sqlite3.connect(target, **kw)
        path, kwargs = label, {"connect_fn": opener}
    elif spelling == "factory":
        class Retarget(sqlite3.Connection):
            def __init__(self, ignored, **kw):
                calls.append(True)
                super().__init__(str(target), **kw)
        path, kwargs = label, {"factory": Retarget}
    with safe.offline_file_access(target):
        with pytest.raises(safe.LiveConnectionError):
            with contextlib.closing(safe.connect_tracked(path, tracking_path=label, **kwargs)):
                pytest.fail("actual reserved database physically opened")
    assert not calls
    assert not safe.has_live_connection(target)


def test_custom_opener_can_open_after_raw_custody_settles(tmp_path):
    target = tmp_path / "target.db"
    with contextlib.closing(safe.connect_tracked(target)):
        pass
    with safe.offline_file_access(target):
        pass
    def opener(path, **kw):
        return sqlite3.connect(path, **kw)
    with contextlib.closing(safe.connect_tracked(target, connect_fn=opener)) as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)


def test_native_alias_retarget_does_not_change_admitted_target(tmp_path, monkeypatch):
    original, reserved, alias = (tmp_path / name for name in ("old.db", "reserved.db", "alias.db"))
    for path in (original, reserved):
        with contextlib.closing(safe.connect_tracked(path)):
            pass
    alias.symlink_to(original)
    native, opened = sqlite3.connect, []
    def retarget(path, **kw):
        alias.unlink()
        alias.symlink_to(reserved)
        conn = native(path, **kw)
        opened.append(conn.execute("PRAGMA database_list").fetchone()[2])
        return conn
    with safe.offline_file_access(reserved):
        with monkeypatch.context() as patch:
            patch.setattr(sqlite3, "connect", retarget)
            try:
                with contextlib.closing(safe.connect_tracked(alias)):
                    pass
            except safe.LiveConnectionError:
                pass
    assert opened == [str(original)]
    assert not safe.has_live_connection(original)


def test_capability_does_not_authorize_sqlite_inside_raw_access(tmp_path):
    path = tmp_path / "state.db"
    with contextlib.closing(safe.connect_tracked(path)) as owner:
        with safe.connection_handoff(path, owner) as claim:
            assert claim
            owner.close()
            with safe.offline_file_access(path, handoff=claim):
                with pytest.raises(safe.LiveConnectionError):
                    with contextlib.closing(safe.connect_tracked(path, handoff=claim)):
                        pytest.fail("capability mixed SQLite and raw descriptors")
            with contextlib.closing(safe.connect_tracked(path, handoff=claim)) as reopened:
                assert reopened.execute("SELECT 1").fetchone() == (1,)
