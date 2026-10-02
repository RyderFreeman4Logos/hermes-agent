"""SQLite result codes outrank misleading identifiers in error messages."""

import sqlite3

import pytest

from hermes_state_errors import is_sqlite_lock_error, is_transient_sqlite_error


@pytest.mark.parametrize("identifier", ["busy", "locked", "disk I/O error"])
def test_known_nonlock_error_is_not_transient(identifier):
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(sqlite3.OperationalError) as caught:
            conn.execute(f"SELECT [{identifier}]")
        assert caught.value.sqlite_errorcode == sqlite3.SQLITE_ERROR
        assert not is_sqlite_lock_error(caught.value)
        assert not is_transient_sqlite_error(caught.value)
    finally:
        conn.close()


@pytest.mark.parametrize("code, text, transient, lock", [
    (sqlite3.SQLITE_BUSY | (2 << 8), "vtable constructor failed", True, True),
    (sqlite3.SQLITE_LOCKED | (1 << 8), "not a lock phrase", True, True),
    (sqlite3.SQLITE_IOERR | (1 << 8), "disk I/O error", True, False),
    (sqlite3.SQLITE_ERROR, "database is busy", False, False),
    (None, "database is locked", True, True),
    (None, "database table is locked: messages", True, True),
    (None, "database schema is locked: main", True, True),
    (None, "database is busy", True, True),
    (None, "disk I/O error", True, False),
    (None, "no such column: busy", False, False),
    (None, "no such column: locked", False, False),
])
def test_known_codes_and_narrow_codeless_fallback(code, text, transient, lock):
    exc = sqlite3.OperationalError(text)
    if code is not None:
        exc.sqlite_errorcode = code
    assert is_transient_sqlite_error(exc) is transient
    assert is_sqlite_lock_error(exc) is lock


@pytest.mark.parametrize("code", [None, sqlite3.SQLITE_ERROR])
def test_wrapped_journal_mode_lock_keeps_narrow_fallback(code):
    from hermes_state_wal import _CANNOT_VERIFY_DELETE_MSG

    exc = sqlite3.OperationalError(_CANNOT_VERIFY_DELETE_MSG)
    if code is not None:
        exc.sqlite_errorcode = code
    assert is_sqlite_lock_error(exc) is (code is None)
    assert is_transient_sqlite_error(exc) is (code is None)
