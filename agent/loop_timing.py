"""Append-only timing metadata for agent-loop input rows."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Mapping, Optional

from utils import is_truthy_value


logger = logging.getLogger(__name__)

LOOP_STOP_KEY = "_latest_successful_loop_stop"
_HEADER = "[Agent loop timing]"


def _now() -> datetime:
    return datetime.now().astimezone()


def _enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        agent_config = config.get("agent", {}) if isinstance(config, dict) else {}
        return is_truthy_value(
            agent_config.get("loop_timing_context")
            if isinstance(agent_config, dict)
            else None,
            default=True,
        )
    except Exception:
        return True


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _latest_stop(agent: Any) -> Optional[str]:
    db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    getter = getattr(db, "get_session_model_config_value", None)
    if not session_id or not callable(getter):
        return None
    try:
        value = getter(session_id, LOOP_STOP_KEY)
    except Exception:
        logger.warning(
            "Could not read loop timing state for session=%s",
            session_id,
            exc_info=True,
        )
        return None
    return value if isinstance(value, str) and value else None


def _timing_block(agent: Any, *, now: Optional[datetime] = None) -> str:
    lines = [
        _HEADER,
        f"Current loop start: {_iso(now or _now())}",
    ]
    latest_stop = _latest_stop(agent)
    if latest_stop:
        lines.append(f"Latest successful completed loop stop: {latest_stop}")
    return "\n".join(lines)


def _append_block(content: Any, block: str) -> Any:
    if isinstance(content, str):
        return f"{content}\n\n{block}" if content else block
    if isinstance(content, list):
        return [*content, {"type": "text", "text": block}]
    return content


def decorate_loop_start_input(
    agent: Any,
    user_message: Any,
    persist_user_message: Any,
    *,
    now: Optional[datetime] = None,
) -> tuple[Any, Any]:
    """Append one identical timing block to this loop's API and durable input.

    ``run_conversation`` calls this once. Provider retries and synchronous tool/LLM
    continuations reuse the staged row, so they cannot acquire a second block.
    Caller-owned structured lists are copied before the text part is appended.
    """
    if not _enabled() or not isinstance(user_message, (str, list)):
        agent._loop_timing_start_decorated = False
        return user_message, persist_user_message

    block = _timing_block(agent, now=now)
    decorated_user = _append_block(user_message, block)
    # Keep the display transcript clean. The persistence override makes the exact
    # decorated wire input land in api_content, including structured messages.
    decorated_persist = (
        persist_user_message if persist_user_message is not None else user_message
    )
    agent._loop_timing_start_decorated = True
    return decorated_user, decorated_persist


def record_completed_loop_stop(
    agent: Any,
    result: Any,
    *,
    now: Optional[datetime] = None,
) -> bool:
    """Persist a stop only for a successfully completed decorated loop."""
    if getattr(agent, "_loop_timing_start_decorated", None) is not True:
        return False
    if not isinstance(result, Mapping) or not result.get("completed"):
        return False
    if result.get("failed") or result.get("interrupted"):
        return False
    db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    patcher = getattr(db, "patch_session_model_config", None)
    if not session_id or not callable(patcher):
        return False
    try:
        patcher(session_id, {LOOP_STOP_KEY: _iso(now or _now())})
    except Exception:
        logger.warning(
            "Could not persist completed loop stop for session=%s",
            session_id,
            exc_info=True,
        )
        return False
    return True


def model_config_for_compression_child(agent: Any, base: Any) -> dict[str, Any]:
    """Copy the current stop scalar into a compression continuation's config."""
    child = dict(base) if isinstance(base, dict) else {}
    latest_stop = _latest_stop(agent)
    if latest_stop:
        child[LOOP_STOP_KEY] = latest_stop
    else:
        child.pop(LOOP_STOP_KEY, None)
    return child
