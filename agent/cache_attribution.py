"""Content-free cache attribution for one provider response per turn."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

POST_COMPRESSION_CACHE_PENDING_KEY = "_awaiting_cache_usage_after_compression"
CACHE_HIT_ERROR_THRESHOLD = 95
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
            "level": "info",
            "prompt_tokens": prompt,
            "state": "no_field",
            "text": "Cache: unavailable",
            "write_tokens": written,
        }
    state = "hit" if cached else "cold_write" if written else "miss"
    hit_percent = round(100 * cached / prompt) if prompt > 0 else 0
    return {
        "cached_tokens": cached,
        "level": "info" if hit_percent >= CACHE_HIT_ERROR_THRESHOLD else "error",
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
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    agent._awaiting_cache_usage_after_compression = bool(pending)
    agent._awaiting_cache_usage_after_compression_session_id = session_id
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


def _compressor_cache_pending(agent: Any, session_id: object) -> bool:
    compressor = getattr(agent, "context_compressor", None)
    bound_session_id = getattr(compressor, "_session_id", None)
    if bound_session_id not in (None, "", session_id):
        return False
    return (
        getattr(compressor, "awaiting_real_usage_after_compression", False) is True
    )


def _post_compression_cache_pending(agent: Any) -> bool:
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    marker_loaded = (
        hasattr(agent, "_awaiting_cache_usage_after_compression")
        and getattr(
            agent,
            "_awaiting_cache_usage_after_compression_session_id",
            None,
        )
        == session_id
    )
    if marker_loaded:
        if getattr(agent, "_awaiting_cache_usage_after_compression", False) is True:
            return True
        return _compressor_cache_pending(agent, session_id)

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
        agent._awaiting_cache_usage_after_compression_session_id = session_id
        if pending:
            return True
        return _compressor_cache_pending(agent, session_id)
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
        info["level"] = "info"
        if info["state"] == "no_field":
            note = "post-compression cache unavailable"
        elif (
            info["state"] == "hit"
            and round(
                100
                * _token_count(info["cached_tokens"])
                / max(1, _token_count(info["prompt_tokens"]))
            )
            >= CACHE_HIT_ERROR_THRESHOLD
        ):
            note = "post-compression cache warm"
        else:
            note = "post-compression warmup (expected)"
        info["note"] = note
        info["text"] = f"{info['text']} · {note}"
    agent._first_turn_cache_info = info
    callback = getattr(agent, "_tui_cache_callback", None)
    if callable(callback):
        try:
            callback(dict(info))
        except Exception:
            logger.debug("TUI first-response cache callback failed", exc_info=True)
    return info


def consume_turn_cache_info(agent: Any) -> dict[str, int | str] | None:
    """Take the current turn's first sample so it cannot leak to later turns."""
    info = getattr(agent, "_first_turn_cache_info", None)
    agent._first_turn_cache_info = None
    return dict(info) if isinstance(info, dict) else None


def settle_returned_turn_cache_info(agent: Any, result: Any) -> Any:
    """Attach and consume cache evidence at the public returned-turn boundary.

    ``finalize_turn`` normally owns this settlement.  Some terminal error
    branches return before reaching it, so the outer AIAgent boundary must
    provide the same one-shot guarantee without replacing an already-finalized
    result.  Raised exceptions deliberately remain unsettled here: their
    caller-specific terminal backstop consumes the sample instead.
    """
    retained = consume_turn_cache_info(agent)
    if (
        isinstance(result, dict)
        and not isinstance(result.get("cache_info"), dict)
        and retained is not None
    ):
        result["cache_info"] = retained
    return result
