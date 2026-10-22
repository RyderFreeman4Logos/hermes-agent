"""Worker and local sidecar backfills preserve structured model input."""
import sqlite3
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_worker_lifecycle import worker_api_content, worker_message_api_content
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.parametrize("writer", [worker_api_content, worker_message_api_content])
def test_worker_backfill_preserves_structured_wire_and_rejects_invalid_shape(writer):
    db = SimpleNamespace(_encode_content=SessionDB._encode_content)
    wire = [{"type": "text", "text": "exact timed input"}]
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE messages(id INTEGER, session_id TEXT, role TEXT, active INTEGER, content TEXT, api_content TEXT)")
        conn.execute("INSERT INTO messages VALUES(1, 'owner', 'user', 1, 'clean', NULL)")
        payload = {"content": "clean", "api_content": wire}
        if writer is worker_message_api_content:
            payload["row_id"] = 1
        assert writer(db, conn, "owner", payload) == {"value": 1}
        stored = conn.execute("SELECT api_content FROM messages").fetchone()[0]
        assert SessionDB._decode_content(stored) == wire
        with pytest.raises(RuntimeStoreError):
            writer(db, conn, "owner", {**payload, "api_content": 42})
