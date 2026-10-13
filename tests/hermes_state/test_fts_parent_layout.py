"""An FTS parent owns only its selected SQLite-generated shadow family."""
import contextlib
import sqlite3

import pytest

from hermes_state_common import FTS_SQL, FTS_TRIGRAM_SQL, LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL
import hermes_state_repair as repair


@pytest.mark.parametrize("external,inline,family", [
    (FTS_SQL, LEGACY_FTS_SQL, "messages_fts"),
    (FTS_TRIGRAM_SQL, LEGACY_FTS_TRIGRAM_SQL, "messages_fts_trigram"),
])
def test_external_parent_refuses_inline_only_shadow(external, inline, family):
    with contextlib.closing(sqlite3.connect(":memory:")) as ref:
        ref.execute("CREATE TABLE messages(id)")
        ref.executescript(inline)
        ddl = ref.execute("SELECT sql FROM sqlite_master WHERE name=?", (family + "_content",)).fetchone()[0]
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE messages(id,content,tool_name,tool_calls)")
        conn.executescript(external)
        conn.execute(ddl)
        with pytest.raises(ValueError, match="ownership"):
            repair._owned_fts_objects(conn)


@pytest.mark.parametrize("main,trigram", [
    (FTS_SQL, FTS_TRIGRAM_SQL),
    (LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL),
    (FTS_SQL, LEGACY_FTS_TRIGRAM_SQL),
    (LEGACY_FTS_SQL, FTS_TRIGRAM_SQL),
])
def test_parent_layout_selection_is_per_family(main, trigram):
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE messages(id,content,tool_name,tool_calls)")
        conn.executescript(main + trigram)
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master") if r[0].startswith("messages_fts")}
        assert repair._owned_fts_objects(conn) == names
