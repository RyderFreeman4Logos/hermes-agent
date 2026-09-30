"""Pending readers degrade to the writer while recovery owns admission."""
import threading

from hermes_state import SessionDB
import hermes_state_repair as repair


def _owner(tmp_path, monkeypatch):
    path = tmp_path / 'state.db'
    monkeypatch.setattr('hermes_state_wal.is_sqlite_wal_reset_vulnerable', lambda **kw: False)
    with SessionDB(db_path=path) as db:
        db.create_session('s', source='cli')
        db.append_message('s', role='user', content='seed')
        db._conn.execute("UPDATE messages_fts_data SET block=X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
    with monkeypatch.context() as opening:
        opening.setattr(SessionDB, '_foreign_state_db_holders', lambda self: [(222, str(path))])
        return SessionDB(db_path=path)



def test_reader_between_flag_check_and_open_falls_back(tmp_path, monkeypatch):
    with _owner(tmp_path, monkeypatch) as db:
        permits = db._read_budget.permits._value
        entered, resume, done = threading.Event(), threading.Event(), threading.Event()
        original_read = db._connect_read_only
        original_strategy = repair._strategy_drop_fts_vacuum
        errors, rows = [], []
        def delayed_read(**kwargs):
            entered.set()
            assert resume.wait(8)
            return original_read(**kwargs)
        monkeypatch.setattr(db, '_connect_read_only', delayed_read)
        def read():
            try:
                with db._read_ctx() as conn:
                    rows.extend(tuple(r) for r in conn.execute('SELECT content FROM messages'))
            except BaseException as exc:
                errors.append((type(exc).__name__, str(exc)))
            finally:
                done.set()
        reader = threading.Thread(target=read)
        def scratch(conn):
            resume.set()
            done.wait(0.5)
            return original_strategy(conn)
        monkeypatch.setattr(repair, '_strategy_drop_fts_vacuum', scratch)
        reader.start()
        try:
            assert entered.wait(8)
            recovered = db.retry_deferred_fts_recovery()
        finally:
            resume.set()
            reader.join(8)
        print('PENDING_READER', {'recovered': recovered, 'errors': errors, 'rows': rows})
        assert not reader.is_alive()
        assert recovered and not errors and rows == [('seed',)]
        assert db._read_budget.permits._value == permits
