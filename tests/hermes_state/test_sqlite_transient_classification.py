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


@pytest.mark.parametrize("code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR, sqlite3.SQLITE_ERROR, None])
def test_fts_retry_respects_codes_over_io_prose(code):
    from hermes_state_search import SessionSearchMixin

    exc = sqlite3.OperationalError("disk I/O error: database is locked")
    if code is not None:
        exc.sqlite_errorcode = code
    if code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        assert SessionSearchMixin._fts_chunk_error(None, exc, "%s") == "retry"
    else:
        with pytest.raises(sqlite3.OperationalError) as caught:
            SessionSearchMixin._fts_chunk_error(None, exc, "%s")
        assert caught.value is exc


@pytest.mark.parametrize("code", [None, sqlite3.SQLITE_ERROR])
def test_wrapped_journal_mode_lock_keeps_narrow_fallback(code):
    from hermes_state_wal import _CANNOT_VERIFY_DELETE_MSG

    exc = sqlite3.OperationalError(_CANNOT_VERIFY_DELETE_MSG)
    if code is not None:
        exc.sqlite_errorcode = code
    assert is_sqlite_lock_error(exc) is (code is None)
    assert is_transient_sqlite_error(exc) is (code is None)


@pytest.mark.parametrize("case", [
    "codeless-io", "io-extended", "busy", "locked-extended", "native-busy", "native-locked",
    "database is locked", "database table is locked", "database schema is locked",
    "database is busy", "disk I/O error",
])
def test_reopen_preserves_sqlite_provenance_through_http_and_write(tmp_path, monkeypatch, case):
    from hermes_state import SessionDB
    from hermes_cli.web_routers import sessions

    db = SessionDB(tmp_path / "state.db")
    native = sqlite3.connect(tmp_path / "native.db", timeout=0, isolation_level=None)
    rival = sqlite3.connect(tmp_path / "native.db", timeout=0, isolation_level=None)
    cursor = None
    try:
        native.execute("CREATE TABLE held(x)")
        native.executemany("INSERT INTO held VALUES(?)", [(1,), (2,)])
        if case == "native-busy":
            rival.execute("BEGIN IMMEDIATE")
            with pytest.raises(sqlite3.OperationalError) as caught:
                native.execute("BEGIN IMMEDIATE")
            cause = caught.value
        elif case == "native-locked":
            cursor = native.execute("SELECT * FROM held")
            cursor.fetchone()
            with pytest.raises(sqlite3.OperationalError) as caught:
                native.execute("DROP TABLE held")
            cause = caught.value
        elif case in ("codeless-io", "io-extended", "busy", "locked-extended"):
            cause = sqlite3.OperationalError("disk I/O error" if "io" in case else "opaque failure")
            code = {"codeless-io": None, "io-extended": 266, "busy": 517, "locked-extended": 262}[case]
            if code is not None:
                cause.sqlite_errorcode = code
                cause.sqlite_errorname = {266: "SQLITE_IOERR_READ", 517: "SQLITE_BUSY_SNAPSHOT", 262: "SQLITE_LOCKED_SHAREDCACHE"}[code]
        else:
            with pytest.raises(sqlite3.OperationalError) as caught:
                native.execute(f"SELECT [{case}]")
            cause = caught.value
            assert cause.sqlite_errorcode == sqlite3.SQLITE_ERROR
        expected_lock = case in ("busy", "locked-extended", "native-busy", "native-locked")
        expected_transient = expected_lock or case in ("codeless-io", "io-extended")
        db.close()
        opens, retries, callbacks = [], [], []
        def fail_open(**kwargs):
            opens.append(kwargs)
            raise cause
        monkeypatch.setattr(db, "_open_writer_conn", fail_open)
        with pytest.raises(sqlite3.OperationalError) as caught:
            db.get_messages("missing")
        wrapped = caught.value
        assert wrapped.__cause__ is cause
        assert is_sqlite_lock_error(wrapped) is expected_lock
        assert is_transient_sqlite_error(wrapped) is expected_transient
        if hasattr(cause, "sqlite_errorcode"):
            assert wrapped.sqlite_errorcode == cause.sqlite_errorcode
            assert wrapped.sqlite_errorname == cause.sqlite_errorname
        monkeypatch.setattr(sessions, "_maybe_auto_archive_for_profile", lambda *_a, **_k: None)
        def fail_http(*_args, **_kwargs):
            raise wrapped
        monkeypatch.setattr(sessions, "_open_session_db_for_profile", fail_http)
        with pytest.raises(sessions.HTTPException) as http:
            sessions.get_sessions(limit=20, offset=0)
        assert http.value.status_code == (503 if expected_transient else 500)
        # One bounded retry decision; no mutation callback may be replayed by IO availability.
        opens.clear()
        monkeypatch.setattr(db, "_sleep_before_write_retry", lambda *_args: retries.append(1) or False)
        with pytest.raises(sqlite3.OperationalError):
            db._execute_write(lambda c: callbacks.append(c), patience_s=1)
        assert len(opens) == 1 and not callbacks
        assert len(retries) == int(expected_transient)
    finally:
        if cursor is not None:
            cursor.close()
        rival.rollback()
        rival.close()
        native.close()
        db.close()


@pytest.mark.parametrize("kind", ["cause", "context", "suppressed-context"])
@pytest.mark.parametrize("outer_code", [None, sqlite3.SQLITE_ERROR])
def test_arbitrary_exception_chains_do_not_grant_retry(kind, outer_code):
    outer = sqlite3.OperationalError("unrelated failure")
    if outer_code is not None:
        outer.sqlite_errorcode = outer_code
    inner = sqlite3.OperationalError("disk I/O error")
    inner.sqlite_errorcode = sqlite3.SQLITE_IOERR
    setattr(outer, "__cause__" if kind == "cause" else "__context__", inner)
    if kind == "suppressed-context":
        outer.__suppress_context__ = True
    assert not is_transient_sqlite_error(outer)
    assert not is_sqlite_lock_error(outer)


@pytest.mark.parametrize("boundary", ["read", "read-open", "begin"])
@pytest.mark.parametrize("genuine_io", [False, True])
def test_io_retry_owners_respect_result_code(tmp_path, monkeypatch, boundary, genuine_io):
    import hermes_state
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    try:
        with pytest.raises(sqlite3.OperationalError) as caught:
            db._conn.execute("SELECT [disk I/O error]")
        error = caught.value
        if genuine_io:
            error = sqlite3.OperationalError("opaque IO failure")
            error.sqlite_errorcode = sqlite3.SQLITE_IOERR | (1 << 8)
        calls, callbacks = [], []
        def fail(*_args, **_kwargs):
            calls.append(1)
            raise error
        monkeypatch.setattr(hermes_state.time, "sleep", lambda _seconds: None)
        if boundary == "read":
            run = lambda: db._read_retrying_ioerr(fail)
        elif boundary == "read-open":
            db.close()
            monkeypatch.setattr(db, "_connect_read_only", fail)
            run = db._open_read_only
        else:
            original = db._conn.execute
            def execute(sql, *args, **kwargs):
                if sql == "BEGIN IMMEDIATE":
                    fail()
                return original(sql, *args, **kwargs)
            monkeypatch.setattr(db._conn, "execute", execute)
            run = lambda: db._execute_write(lambda c: callbacks.append(c))
        with pytest.raises(sqlite3.OperationalError) as caught:
            run()
        assert caught.value is error
        expected = (2 if boundary == "begin" else hermes_state._READ_ONLY_IOERR_RETRY_ATTEMPTS + 1) if genuine_io else 1
        assert len(calls) == expected
        assert callbacks == []
        if db._conn is not None:
            assert not db._conn.in_transaction
    finally:
        db.close()
