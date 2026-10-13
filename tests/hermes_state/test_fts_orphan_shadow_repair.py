"""An orphan name/layout does not prove derived ownership.

The previous #103840 automatic cleanup contract is deliberately fail-closed: a
`.recover` residue is indistinguishable from a canonical extension table without
its owning virtual-table declaration. Proven families still recover normally.
"""

import contextlib
import sqlite3
import pytest

from hermes_state import SessionDB


def _orphan_family(db_path, family: str) -> None:
    """Emulate the ``.recover`` residue for one family: vtable row gone, shadows kept."""
    raw = sqlite3.connect(db_path)
    raw.isolation_level = None
    raw.execute("PRAGMA writable_schema=ON")
    raw.execute(
        "DELETE FROM sqlite_master WHERE name = ? AND sql LIKE 'CREATE VIRTUAL TABLE%'", (family,),
    )
    version = raw.execute("PRAGMA schema_version").fetchone()[0]
    raw.execute(f"PRAGMA schema_version={version + 1}")
    raw.execute("PRAGMA writable_schema=OFF")
    raw.close()


def _fts_master_rows(db_path, prefix: str) -> list:
    raw = sqlite3.connect(db_path)
    try:
        return raw.execute(
            "SELECT rowid, type, name FROM sqlite_master WHERE name LIKE ? ESCAPE '\\' ORDER BY rowid",
            (prefix.replace("_", "\\_") + "%",),
        ).fetchall()
    finally:
        raw.close()


def _orphan_contents(db_path):
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('messages_fts_data','messages_fts_idx','messages_fts_docsize','messages_fts_config')"
        )]
        schema = conn.execute("SELECT rowid,type,name,tbl_name,sql,rootpage FROM sqlite_master ORDER BY rowid").fetchall()
        # sqlite rows preserve BLOB bytes and storage classes; sorting preserves multiplicity.
        contents = {name: conn.execute(f'SELECT * FROM "{name}" ORDER BY 1').fetchall() for name in names}
        for name in ('messages_fts_data', 'messages_fts_docsize'):
            if name in names:
                contents[name + ':rowids'] = conn.execute(f'SELECT rowid,* FROM "{name}" ORDER BY rowid').fetchall()
        contents['messages'] = conn.execute('SELECT rowid,* FROM messages ORDER BY rowid').fetchall()
        contents['sequence'] = conn.execute('SELECT rowid,* FROM sqlite_sequence ORDER BY rowid').fetchall()
        return schema, contents


def test_orphaned_base_family_refuses_without_ownership_proof(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s1", source="cli", model="m")
    for i in range(3):
        db.append_message("s1", "user", f"recovered orphan {i}")
    db.close()

    _orphan_family(db_path, "messages_fts")
    trigram_before = _fts_master_rows(db_path, "messages_fts_trigram")
    assert trigram_before, "fixture needs a live trigram family"
    raw = sqlite3.connect(db_path)
    orphan_shadows = raw.execute(
        "SELECT count(*) FROM sqlite_master WHERE name IN ('messages_fts_data', 'messages_fts_config')"
    ).fetchone()[0]
    raw.close()
    assert orphan_shadows == 2, "fixture must leave the base shadows behind"

    base_before = _fts_master_rows(db_path, "messages_fts")
    canonical_before = _orphan_contents(db_path)
    with pytest.raises(sqlite3.OperationalError, match="ownership"):
        SessionDB(db_path=db_path)
    assert _fts_master_rows(db_path, "messages_fts") == base_before
    assert _orphan_contents(db_path) == canonical_before
    with contextlib.closing(sqlite3.connect(db_path)) as raw:
        assert raw.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
    # Same rowids: the healthy family was neither dropped nor recreated.
    assert _fts_master_rows(db_path, "messages_fts_trigram") == trigram_before


def test_normal_shadow_name_table_is_not_disposable(tmp_path):
    from hermes_state_fts import _drop_orphan_fts_shadow_tables
    with contextlib.closing(sqlite3.connect(tmp_path / "canonical.db")) as conn:
        conn.execute("CREATE TABLE messages_fts_data(id INTEGER PRIMARY KEY, block BLOB)")
        conn.execute("INSERT INTO messages_fts_data VALUES(100,X'CAFE')")
        before = conn.execute("SELECT rowid,type,name,tbl_name,sql,rootpage FROM sqlite_master").fetchall()
        with pytest.raises(sqlite3.OperationalError, match="ownership"):
            _drop_orphan_fts_shadow_tables(conn.cursor(), ("messages_fts",))
        assert conn.execute("SELECT * FROM messages_fts_data").fetchall() == [(100, b'\xca\xfe')]
        assert conn.execute("SELECT rowid,type,name,tbl_name,sql,rootpage FROM sqlite_master").fetchall() == before
