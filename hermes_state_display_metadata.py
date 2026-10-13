"""Display-only metadata updates for the SessionDB message mixin."""
from __future__ import annotations

from typing import Any, Optional

_SET_DISPLAY_META_SQL = "UPDATE messages SET display_metadata = ? WHERE id = ?"


class SessionDisplayMetadataMixin:
    """Display metadata helpers layered onto SessionMessagesMixin's DB codecs/writer."""

    def _reaction_list(self: Any, meta: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
        """Well-formed (dict) reactions stored under ``REACTIONS_METADATA_KEY``."""
        reactions = (meta or {}).get(self.REACTIONS_METADATA_KEY)
        return [r for r in reactions if isinstance(r, dict)] if isinstance(reactions, list) else []

    def _merge_assistant_display_metadata(self: Any, message_row_id: int, patch: dict[str, Any]) -> bool:
        """Merge display-only keys into one persisted assistant row by its physical row id."""
        if (
            not isinstance(message_row_id, int)
            or isinstance(message_row_id, bool)
            or message_row_id <= 0
            or not isinstance(patch, dict)
            or not patch
        ):
            return False

        def _do(conn):
            row = conn.execute(
                "SELECT display_metadata FROM messages WHERE id = ? AND role = 'assistant'",
                (message_row_id,),
            ).fetchone()
            if row is None:
                return False
            meta = self._decode_display_metadata(row[0]) or {}
            meta.update(patch)
            conn.execute(_SET_DISPLAY_META_SQL, (self._encode_display_metadata(meta), message_row_id))
            return True

        return self._execute_write(_do)
