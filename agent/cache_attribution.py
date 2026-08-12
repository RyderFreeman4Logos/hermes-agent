from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

POST_COMPRESSION_CACHE_PENDING_KEY = "_awaiting_cache_usage_after_compression"


def set_post_compression_cache_pending(agent: Any, pending: bool) -> None:
    """Mirror the one-shot cache-attribution marker in memory and SessionDB."""
    agent._awaiting_cache_usage_after_compression = bool(pending)
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    ensure_session = getattr(agent, "_ensure_db_session", None)
    patch = getattr(session_db, "patch_session_model_config", None)
    try:
        if pending and callable(ensure_session):
            ensure_session()
        if not session_id or not callable(patch):
            return
        patch(
            session_id,
            {POST_COMPRESSION_CACHE_PENDING_KEY: True if pending else None},
        )
    except Exception:
        logger.debug(
            "post-compression cache marker update failed (session=%s)",
            session_id,
            exc_info=True,
        )


def consume_post_compression_cache_pending(agent: Any) -> bool:
    """Consume the one-shot marker and return whether it was armed."""
    pending = bool(
        getattr(agent, "_awaiting_cache_usage_after_compression", False)
    )
    if pending:
        set_post_compression_cache_pending(agent, False)
    return pending


def clear_post_compression_cache_pending_after_empty_usage(agent: Any) -> bool:
    """Consume a pending marker when a terminal turn ingested no usage."""
    if getattr(agent, "_turn_received_provider_response", False) is not True:
        return False
    if getattr(agent, "_last_turn_usage", None) is not None:
        return False
    return consume_post_compression_cache_pending(agent)


def load_post_compression_cache_pending(agent: Any) -> bool:
    """Restore the durable marker for a resumed session."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    getter = getattr(session_db, "get_session_model_config_value", None)
    pending = False
    if session_id and callable(getter):
        try:
            pending = getter(
                session_id,
                POST_COMPRESSION_CACHE_PENDING_KEY,
                False,
            ) is True
        except Exception:
            logger.debug(
                "post-compression cache marker load failed (session=%s)",
                session_id,
                exc_info=True,
            )
    agent._awaiting_cache_usage_after_compression = pending
    return pending
