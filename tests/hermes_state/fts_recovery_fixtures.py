"""Exercise the scratch fallback independently of SQLite's direct DDL recovery."""
import sqlite3

from hermes_state import SessionDB


def force_scratch_recovery(monkeypatch):
    """Inject only the old-engine constructor failure; scratch SQLite work stays real."""
    original = SessionDB._recover_stale_fts_locked

    class ConstructorFailureCursor:
        def __init__(self, cursor):
            self.cursor = cursor

        def __getattr__(self, name):
            return getattr(self.cursor, name)

        def executescript(self, sql):
            raise sqlite3.DatabaseError("vtable constructor failed: messages_fts")

    def recover(self, cursor, *, legacy):
        return original(self, ConstructorFailureCursor(cursor), legacy=legacy)

    monkeypatch.setattr(SessionDB, "_recover_stale_fts_locked", recover)
