"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from typing import Any, MutableMapping, Optional, TypeVar


# These fields describe Hermes' durable record, not provider-visible message
# content. They must not influence context-pressure decisions.
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset({"timestamp"})
LOOP_TIMING_TURN_ID = "loop_timing_turn_id"

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time.

    Gateway adapters can supply the platform event time; all other callers use
    the local wall clock. Returns the same mapping for use at append sites.
    """
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Stamp and append one live transcript message."""
    messages.append(stamp_message_timestamp(message, timestamp=timestamp))
    return message


def is_hidden_loop_timing(message: Any) -> bool:
    """Whether a durable hidden system row is the loop-timing projection."""
    if not isinstance(message, dict) or message.get("role") != "system":
        return False
    content = message.get("content", "")
    if not isinstance(content, str):
        return False
    if content.startswith("[Agent loop timing] Current loop start:"):
        return True
    lines = content.splitlines()
    return (
        bool(lines)
        and lines[0] == "[Agent loop timing]"
        and any(line.startswith("Current loop start:") for line in lines[1:])
    )
