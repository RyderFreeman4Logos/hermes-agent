"""Content-free cache attribution for one provider response per turn."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

POST_COMPRESSION_CACHE_PENDING_KEY = "_awaiting_cache_usage_after_compression"
logger = logging.getLogger(__name__)


def _token_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float, str)):
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
    return 0


def _field_is_present(value: object, field: str) -> bool:
    if isinstance(value, Mapping):
        return field in value and value.get(field) is not None
    fields_set = getattr(value, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(value, "__fields_set__", None)
    if isinstance(fields_set, (set, frozenset)):
        return field in fields_set
    return hasattr(value, field) and getattr(value, field, None) is not None


def _field_value(value: object, field: str) -> object:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def cache_telemetry_present(raw_usage: object) -> bool:
    """Return whether the provider supplied a cache-specific usage field."""
    if raw_usage is None:
        return False
    direct_fields = (
        "cachedInputTokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "cached_tokens",
        "cache_write_tokens",
    )
    if any(_field_is_present(raw_usage, field) for field in direct_fields):
        return True
    nested_fields = (
        "cached_tokens",
        "cache_write_tokens",
        "cache_creation_tokens",
        "cache_creation_input_tokens",
    )
    for container in ("prompt_tokens_details", "input_tokens_details"):
        details = _field_value(raw_usage, container)
        if details is not None and any(
            _field_is_present(details, field) for field in nested_fields
        ):
            return True
    return False


def format_cache_hit_percent(cached_tokens: int, prompt_tokens: int) -> str:
    """Format a positive sub-percent cache hit without displaying zero."""
    if cached_tokens > 0 and prompt_tokens > 0 and cached_tokens * 100 < prompt_tokens:
        return "<1%"
    if prompt_tokens <= 0:
        return "0%"
    return f"{round(100 * cached_tokens / prompt_tokens)}%"


def cache_info_from_usage(
    usage: Mapping[str, object], *, telemetry_present: bool
) -> dict[str, int | str]:
    cached = _token_count(usage.get("cache_read_tokens", 0))
    written = _token_count(usage.get("cache_write_tokens", 0))
    prompt = _token_count(usage.get("prompt_tokens", 0))
    if not telemetry_present:
        return {
            "cached_tokens": cached,
            "prompt_tokens": prompt,
            "state": "no_field",
            "text": "Cache: unavailable",
            "write_tokens": written,
        }
    state = "hit" if cached else "cold_write" if written else "miss"
    return {
        "cached_tokens": cached,
        "prompt_tokens": prompt,
        "state": state,
        "text": (
            f"Cache: {cached:,}/{prompt:,} tokens "
            f"({format_cache_hit_percent(cached, prompt)} hit, {written:,} written)"
        ),
        "write_tokens": written,
    }


def cache_log_suffix(info: Mapping[str, object]) -> str:
    """Build the fixed-field, content-free cache suffix for an API-call log."""
    state = str(info.get("state") or "no_field")
    prompt = _token_count(info.get("prompt_tokens", 0))
    suffix = f"cache_state={state}"
    if state != "no_field":
        suffix += (
            f" cache_read={_token_count(info.get('cached_tokens', 0))}"
            f" cache_write={_token_count(info.get('write_tokens', 0))}"
        )
    return f"{suffix} cache_prompt={prompt}"


def set_post_compression_cache_pending(agent: Any, pending: bool) -> None:
    """Mirror the one-shot post-compression marker in memory and SessionDB."""
    agent._awaiting_cache_usage_after_compression = bool(pending)
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    patch = getattr(session_db, "patch_session_model_config", None)
    ensure_session = getattr(agent, "_ensure_db_session", None)
    try:
        if pending and callable(ensure_session):
            ensure_session()
        if session_id and callable(patch):
            patch(
                session_id,
                {POST_COMPRESSION_CACHE_PENDING_KEY: True if pending else None},
            )
    except Exception:
        logger.debug("post-compression cache marker update failed", exc_info=True)


def _post_compression_cache_pending(agent: Any) -> bool:
    marker_loaded = hasattr(agent, "_awaiting_cache_usage_after_compression")
    if getattr(agent, "_awaiting_cache_usage_after_compression", False) is True:
        return True
    compressor = getattr(agent, "context_compressor", None)
    if getattr(compressor, "awaiting_real_usage_after_compression", False) is True:
        return True
    if marker_loaded:
        return False
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    getter = getattr(session_db, "get_session_model_config_value", None)
    if not session_id or not callable(getter):
        return False
    try:
        pending = getter(
            session_id,
            POST_COMPRESSION_CACHE_PENDING_KEY,
            False,
        ) is True
        agent._awaiting_cache_usage_after_compression = pending
        return pending
    except Exception:
        logger.debug("post-compression cache marker lookup failed", exc_info=True)
        return False


def clear_post_compression_cache_pending(agent: Any) -> bool:
    pending = _post_compression_cache_pending(agent)
    if pending:
        set_post_compression_cache_pending(agent, False)
    return pending


def record_first_turn_cache_info(
    agent: Any,
    usage: Mapping[str, object],
    *,
    telemetry_present: bool,
) -> dict[str, int | str] | None:
    """Retain the first sample, or the first sample after a cache boundary."""
    post_compression = clear_post_compression_cache_pending(agent)
    if (
        isinstance(getattr(agent, "_first_turn_cache_info", None), dict)
        and not post_compression
    ):
        return None
    info = cache_info_from_usage(usage, telemetry_present=telemetry_present)
    if post_compression:
        info["attribution"] = "post_compression"
    agent._first_turn_cache_info = info
    return info


def consume_turn_cache_info(agent: Any) -> dict[str, int | str] | None:
    """Take the current turn's first sample so it cannot leak to later turns."""
    info = getattr(agent, "_first_turn_cache_info", None)
    agent._first_turn_cache_info = None
    return dict(info) if isinstance(info, dict) else None
