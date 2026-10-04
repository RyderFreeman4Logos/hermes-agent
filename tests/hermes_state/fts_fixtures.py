"""Real constructor corruption for tests of the snapshot-only recovery boundary."""
import sqlite3

import pytest

from hermes_state import SessionDB


def require_fts_snapshot(db: SessionDB) -> None:
    """Keep the detached canonical owner, but make normal DROP/rebuild fail for real.

    Corrupt structure blocks alone remain droppable on newer SQLite builds. Losing
    the derived config table fails xConnect; reopen only this fixture-owned writer
    so SQLite cannot reuse the valid vtable it cached before the damage.
    """
    assert db._fts_stale and not db._fts_enabled
    assert db._read_pool.empty()
    db._conn.execute("DROP TABLE messages_fts_config")
    db._conn.close()
    db._conn = db._open_writer_conn()
    with pytest.raises(sqlite3.OperationalError, match="vtable constructor failed: messages_fts"):
        db._conn.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'seed'").fetchone()
