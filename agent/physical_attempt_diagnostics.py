"""Opt-in, content-free diagnostics for physical provider attempts.

The v1 stream stages stop at Python's first wire event, callback execution, and
the synchronous gateway ``transport.write`` handoff.  They do not yet model
wire-event type/cadence/character counts, detailed terminal/backpressure/drop
counters, delayed websocket flush/socket send or Tee legs, Node receive/parse/
dispatch, bounded retention, or full failure-path privacy and overhead evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


_SCHEMA = "hermes.physical_attempt.v1"
_ROOT_PARTS = ("observability",)
_KEY_FILE = "physical_attempt_digests.key"
_RECORDS_FILE = "physical_attempt_digests.jsonl"
_ROLES = {"main", "subagent", "reviewer", "fallback", "auxiliary", "unknown"}
_OUTCOMES = {"completed", "incomplete", "failed", "error", "cancelled", "interrupted"}
_PHASES = {"start", "progress", "ambiguity", "reconciliation", "terminal"}
_PROGRESS = {"semantic"}
_AMBIGUITY_CLASSES = {
    "transport",
    "missing_terminal",
    "ttfb_timeout",
    "sse_idle_timeout",
    "no_progress_timeout",
    "total_request_timeout",
}
_RECONCILIATION_ACTIONS = {"halt", "retain", "wait", "reaped", "resume"}
_INTERNAL_SCOPE_KEY = "_hermes_physical_attempt_scope"
_VISIBLE_CATEGORIES = {"text", "reasoning", "thinking", "tool", "terminal"}
_TRANSPORT_KINDS = {"stdio", "websocket", "tee", "other"}
_INTERNAL_LIFECYCLE_KEY = "_hermes_physical_attempt_lifecycle"


@dataclass
class Attempt:
    digest: str
    streamed: bool = False
    finished: bool = False
    progress_recorded: bool = False
    ambiguity_recorded: bool = False
    reconciliation_actions: set[str] = field(default_factory=set)
    disposition: str | None = None
    started_ns: int | None = None
    dispatch_ns: int | None = None
    first_wire_ns: int | None = None
    first_visible_ns: int | None = None
    first_visible_category: str | None = None
    wire_event_count: int = 0
    visible_event_count: int = 0
    callback_depth: int = 0
    callback_stats: dict[str, list[int]] = field(default_factory=dict)
    transport_stats: dict[str, list[int]] = field(default_factory=dict)


_ACTIVE_ATTEMPT: ContextVar[Attempt | None] = ContextVar(
    "physical_attempt_stage_latency", default=None
)


def activate_attempt(attempt: Attempt | None) -> None:
    """Bind a streamed attempt, or clear attribution when provenance is unknown."""
    if attempt is None:
        _ACTIVE_ATTEMPT.set(None)
        return
    if not isinstance(attempt, Attempt) or not attempt.streamed or attempt.finished:
        return
    if _ACTIVE_ATTEMPT.get() is not attempt:
        _ACTIVE_ATTEMPT.set(attempt)


def current_attempt() -> Attempt | None:
    attempt = _ACTIVE_ATTEMPT.get()
    return (
        attempt
        if isinstance(attempt, Attempt) and attempt.streamed and not attempt.finished
        else None
    )


def mark_dispatch(attempt: Attempt | None) -> None:
    if not isinstance(attempt, Attempt) or not attempt.streamed or attempt.finished:
        return
    try:
        if attempt.dispatch_ns is None:
            attempt.dispatch_ns = time.monotonic_ns()
    except Exception:
        return


def mark_wire_event(attempt: Attempt | None) -> None:
    if not isinstance(attempt, Attempt) or not attempt.streamed or attempt.finished:
        return
    try:
        now = time.monotonic_ns()
        if attempt.dispatch_ns is None:
            attempt.dispatch_ns = now
        if attempt.first_wire_ns is None:
            attempt.first_wire_ns = now
        attempt.wire_event_count += 1
        activate_attempt(attempt)
    except Exception:
        return


def begin_callback(category: str) -> tuple[Attempt, str, int] | None:
    attempt = current_attempt()
    if attempt is None or category not in _VISIBLE_CATEGORIES:
        return None
    try:
        now = time.monotonic_ns()
        if attempt.first_visible_ns is None:
            attempt.first_visible_ns = now
            attempt.first_visible_category = category
        attempt.visible_event_count += 1
        started_ns = time.monotonic_ns()
        attempt.callback_depth += 1
        return attempt, category, started_ns
    except Exception:
        return None


def end_callback(marker: tuple[Attempt, str, int] | None) -> None:
    _end_stage(marker, transport=False)


def begin_transport(kind: str) -> tuple[Attempt, str, int] | None:
    attempt = current_attempt()
    if attempt is None or attempt.callback_depth <= 0:
        return None
    safe_kind = kind if kind in _TRANSPORT_KINDS else "other"
    try:
        return attempt, safe_kind, time.monotonic_ns()
    except Exception:
        return None


def end_transport(marker: tuple[Attempt, str, int] | None) -> None:
    _end_stage(marker, transport=True)


def _end_stage(
    marker: tuple[Attempt, str, int] | None, *, transport: bool
) -> None:
    if marker is None:
        return
    attempt, category, started_ns = marker
    if not transport:
        attempt.callback_depth = max(0, attempt.callback_depth - 1)
    if attempt.finished:
        return
    try:
        elapsed = max(0, time.monotonic_ns() - started_ns)
        target = attempt.transport_stats if transport else attempt.callback_stats
        stats = target.setdefault(category, [0, 0, 0])
        stats[0] += 1
        stats[1] += elapsed
        stats[2] = max(stats[2], elapsed)
    except Exception:
        return


def _stage_stats(values: dict[str, list[int]]) -> dict[str, dict[str, int]]:
    return {
        category: {"count": stats[0], "total_ns": stats[1], "max_ns": stats[2]}
        for category, stats in sorted(values.items())
    }


def _stage_latency(attempt: Attempt, terminal_ns: int) -> dict[str, Any]:
    dispatch = attempt.dispatch_ns
    first_wire = attempt.first_wire_ns
    first_visible = attempt.first_visible_ns
    return {
        "dispatch_monotonic_ns": dispatch,
        "dispatch_ns": (
            max(0, dispatch - attempt.started_ns)
            if dispatch is not None and attempt.started_ns is not None
            else None
        ),
        "duration_ns": max(0, terminal_ns - dispatch) if dispatch is not None else None,
        "ttfb_ns": (
            max(0, first_wire - dispatch)
            if dispatch is not None and first_wire is not None
            else None
        ),
        "first_visible_ns": (
            max(0, first_visible - dispatch)
            if dispatch is not None and first_visible is not None
            else None
        ),
        "wire_to_visible_ns": (
            max(0, first_visible - first_wire)
            if first_wire is not None and first_visible is not None
            else None
        ),
        "first_visible_category": attempt.first_visible_category,
        "wire_event_count": attempt.wire_event_count,
        "visible_event_count": attempt.visible_event_count,
        "callbacks": _stage_stats(attempt.callback_stats),
        "transports": _stage_stats(attempt.transport_stats),
    }


def enabled() -> bool:
    try:
        from hermes_cli.config import read_raw_config_readonly

        config = read_raw_config_readonly() or {}
        observability = (
            config.get("observability") if isinstance(config, dict) else None
        )
        physical = (
            observability.get("physical_attempt_digests")
            if isinstance(observability, dict)
            else None
        )
        return isinstance(physical, dict) and physical.get("enabled") is True
    except Exception:
        return False


def prepare_cache_scope(value: Any) -> dict[str, Any] | None:
    """Digest a logical scope before it leaves ``build_kwargs``."""
    if not enabled():
        return None
    try:
        key = _digest_key()
        digest, length = _digest_value(key, "scope", value)
        return {"scope_digest": digest, "scope_bytes": length}
    except Exception:
        return None


def start_responses_attempt(
    request: dict[str, Any],
    *,
    scope: dict[str, Any] | None,
    route: str,
    provider: str,
    model: str,
    role: str,
    retry: int,
    continuation: int,
) -> Attempt | None:
    return start_attempt(
        request,
        api_mode="codex_responses",
        scope=scope,
        route=route,
        provider=provider,
        model=model,
        role=role,
        retry=retry,
        continuation=continuation,
        streamed=True,
    )


def start_attempt(
    request: dict[str, Any],
    *,
    api_mode: str,
    route: str,
    provider: str,
    model: str,
    role: str,
    retry: int,
    continuation: int,
    scope: dict[str, Any] | None = None,
    streamed: bool = False,
) -> Attempt | None:
    """Persist one allowlisted start record at a physical dispatch seam."""
    if not enabled():
        return None
    try:
        key = _digest_key()
        key_value = _effective_cache_key(request)
        instructions, provider_input, tools = _request_identity(request, api_mode)
        prefix = {"instructions": instructions, "input": provider_input}
        equivalent = {
            "scope_digest": (scope or {}).get("scope_digest"),
            "cache_key": key_value,
            "instructions": instructions,
            "tools": tools,
        }
        key_digest, key_bytes = _digest_value(key, "key", key_value)
        prefix_digest, prefix_bytes = _digest_value(key, "prefix", prefix)
        tool_digest, tool_bytes = _digest_value(key, "tool", tools)
        equivalent_digest, equivalent_bytes = _digest_value(
            key, "equivalent", equivalent
        )
        attempt_digest = hmac.new(
            key, b"attempt\0" + secrets.token_bytes(32), hashlib.sha256
        ).hexdigest()
        started_ns = time.monotonic_ns()
        attempt = Attempt(attempt_digest, streamed=streamed, started_ns=started_ns)
        record = {
            "schema": _SCHEMA,
            "phase": "start",
            "attempt_digest": attempt_digest,
            "monotonic_ns": started_ns,
            "route": _safe_label(route, 128),
            "provider": _safe_label(provider, 128),
            "model": _safe_label(model, 128),
            "role": role if role in _ROLES else "unknown",
            "retry": max(0, int(retry)),
            "continuation": max(0, int(continuation)),
            "scope_digest": (scope or {}).get("scope_digest"),
            "scope_bytes": (scope or {}).get("scope_bytes"),
            "key_digest": key_digest,
            "key_bytes": key_bytes,
            "prefix_digest": prefix_digest,
            "prefix_bytes": prefix_bytes,
            "tool_digest": tool_digest,
            "tool_bytes": tool_bytes,
            "equivalent_digest": equivalent_digest,
            "equivalent_bytes": equivalent_bytes,
        }
        _append(record)
        return attempt
    except Exception:
        return None


def finish_responses_attempt(
    attempt: Attempt | None, *, usage: Any, outcome: str
) -> None:
    finish_attempt(
        attempt,
        usage=usage,
        outcome=outcome,
        api_mode="codex_responses",
        provider="",
    )


def finish_attempt(
    attempt: Attempt | None,
    *,
    usage: Any,
    outcome: str,
    api_mode: str,
    provider: str,
) -> None:
    """Pair a start with terminal token/cache metadata when it is available."""
    if attempt is None or attempt.finished:
        return
    attempt.finished = True
    try:
        terminal_ns = time.monotonic_ns()
        if _ACTIVE_ATTEMPT.get() is attempt:
            _ACTIVE_ATTEMPT.set(None)
        from agent.usage_pricing import normalize_usage

        canonical = normalize_usage(usage, api_mode=api_mode, provider=provider)
        cache_present = bool(usage) and canonical.cache_telemetry_present
        cache_state = (
            "unknown"
            if not cache_present
            else "hit"
            if canonical.cache_read_tokens > 0
            else "miss"
        )
        record: dict[str, Any] = {
            "schema": _SCHEMA,
            "phase": "terminal",
            "attempt_digest": attempt.digest,
            "monotonic_ns": terminal_ns,
            "outcome": outcome if outcome in _OUTCOMES else "unknown",
            "disposition": attempt.disposition
            or (outcome if outcome in _OUTCOMES else "unknown"),
            "cache_state": cache_state,
            "input_tokens": canonical.input_tokens if usage else None,
            "output_tokens": canonical.output_tokens if usage else None,
            "cache_read_tokens": canonical.cache_read_tokens if cache_present else None,
            "cache_write_tokens": canonical.cache_write_tokens
            if cache_present
            else None,
        }
        if attempt.streamed:
            record["stage_latency"] = _stage_latency(attempt, terminal_ns)
        _append(record)
    except Exception:
        return


def _append_phase(record: dict[str, Any]) -> None:
    try:
        _append(record)
    except Exception:
        pass


def record_progress(attempt: Attempt | None, *, progress: str) -> None:
    """Record the first content-free semantic-progress phase for an attempt."""
    if (
        not isinstance(attempt, Attempt)
        or attempt.finished
        or attempt.progress_recorded
    ):
        return
    attempt.progress_recorded = True
    _append_phase(
        {
            "schema": _SCHEMA,
            "phase": "progress",
            "attempt_digest": attempt.digest,
            "monotonic_ns": time.monotonic_ns(),
            "progress": progress if progress in _PROGRESS else "semantic",
        }
    )


def record_ambiguity(attempt: Attempt | None, *, failure_class: str) -> None:
    """Record one acceptance-ambiguity phase without exception text."""
    if (
        not isinstance(attempt, Attempt)
        or attempt.finished
        or attempt.ambiguity_recorded
    ):
        return
    attempt.ambiguity_recorded = True
    attempt.disposition = "ambiguous_halt"
    _append_phase(
        {
            "schema": _SCHEMA,
            "phase": "ambiguity",
            "attempt_digest": attempt.digest,
            "monotonic_ns": time.monotonic_ns(),
            "failure_class": failure_class
            if failure_class in _AMBIGUITY_CLASSES
            else "transport",
            "acceptance": "unknown",
        }
    )


def record_reconciliation(attempt: Attempt | None, *, action: str) -> None:
    """Record each bounded reconciliation action once per attempt."""
    if not isinstance(attempt, Attempt):
        return
    action_value = action if action in _RECONCILIATION_ACTIONS else "halt"
    if action_value in attempt.reconciliation_actions:
        return
    attempt.reconciliation_actions.add(action_value)
    _append_phase(
        {
            "schema": _SCHEMA,
            "phase": "reconciliation",
            "attempt_digest": attempt.digest,
            "monotonic_ns": time.monotonic_ns(),
            "action": action_value,
        }
    )


def _request_identity(
    request: dict[str, Any], api_mode: str
) -> tuple[Any, Any, Any]:
    if api_mode == "codex_responses":
        return (
            request.get("instructions"),
            request.get("input"),
            request.get("tools") or [],
        )

    messages = request.get("messages")
    instructions = request.get("system")
    if instructions is None and isinstance(messages, list):
        static_messages = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system",
                "developer",
            }:
                break
            static_messages.append(message)
        instructions = static_messages
    tools = request.get("tools")
    if tools is None:
        tools = request.get("toolConfig")
    return instructions, messages, tools or []


def _effective_cache_key(request: dict[str, Any]) -> Any:
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict) and extra_body.get("prompt_cache_key") is not None:
        return extra_body["prompt_cache_key"]
    if request.get("prompt_cache_key") is not None:
        return request["prompt_cache_key"]
    headers = request.get("extra_headers")
    if isinstance(headers, dict):
        return headers.get("x-client-request-id")
    return None


def _safe_label(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > limit or "://" in text:
        return "unknown"
    try:
        from agent.redact import redact_sensitive_text

        if redact_sensitive_text(
            text, force=True, redact_url_credentials=True
        ) != text:
            return "unknown"
    except Exception:
        return "unknown"
    if not all(ch.isascii() and (ch.isalnum() or ch in "._/:-") for ch in text):
        return "unknown"
    return text


def _digest_value(key: bytes, label: str, value: Any) -> tuple[str, int]:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8", errors="replace")
    return hmac.new(
        key, label.encode() + b"\0" + encoded, hashlib.sha256
    ).hexdigest(), len(encoded)


def _root() -> Path:
    return get_hermes_home().joinpath(*_ROOT_PARTS)


def _ensure_root() -> Path:
    root = _root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise PermissionError("unsafe physical-attempt diagnostics directory")
    if stat.S_IMODE(info.st_mode) != 0o700:
        root.chmod(0o700)
    return root


def _private_fd(path: Path, flags: int, *, create: bool = False) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = (
        os.open(path, flags | nofollow, 0o600)
        if create
        else os.open(path, flags | nofollow)
    )
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        os.close(fd)
        raise PermissionError("unsafe physical-attempt diagnostics file")
    return fd


def _digest_key() -> bytes:
    path = _ensure_root() / _KEY_FILE
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = _private_fd(path, flags, create=True)
    except FileExistsError:
        fd = _private_fd(path, os.O_RDONLY)
        try:
            value = os.read(fd, 33)
        finally:
            os.close(fd)
        if len(value) != 32:
            raise ValueError("invalid physical-attempt digest key")
        return value
    value = secrets.token_bytes(32)
    try:
        os.write(fd, value)
        os.fsync(fd)
    finally:
        os.close(fd)
    return value


def _append(record: dict[str, Any]) -> None:
    if record.get("phase") not in _PHASES:
        return
    path = _ensure_root() / _RECORDS_FILE
    fd = _private_fd(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, create=True)
    try:
        payload = (
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
        os.write(fd, payload)
    finally:
        os.close(fd)
