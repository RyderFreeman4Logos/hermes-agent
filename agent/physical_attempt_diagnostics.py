"""Opt-in, content-free digests for adjacent physical provider attempts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

__all__ = [
    "Attempt",
    "activate_attempt",
    "begin_callback",
    "begin_transport",
    "current_attempt",
    "enabled",
    "end_callback",
    "end_transport",
    "finish_attempt",
    "mark_dispatch",
    "mark_wire_event",
    "prepare_cache_scope",
    "record_ambiguity",
    "record_checkpoint",
    "record_reconciliation",
    "start_attempt",
]

_KEY_FILE = "physical_attempt_digests.key"
_RECORDS_FILE = "physical_attempt_digests.jsonl"
_LOCK = threading.Lock()
_LAST_ATTEMPT: dict[tuple[str, str], "Attempt"] = {}
_MAX_PAIR_IDENTITIES = 1024
_CACHE_SCOPE_ENVELOPE = "_hermes_physical_attempt_cache_scope"
_VISIBLE_CATEGORIES = {"text", "reasoning", "thinking", "tool", "terminal"}
_TRANSPORT_KINDS = {"stdio", "websocket", "tee", "other"}
_AMBIGUITY_CLASSES = {
    "transport",
    "missing_terminal",
    "ttfb_timeout",
    "sse_idle_timeout",
    "no_progress_timeout",
    "total_request_timeout",
}
_RECONCILIATION_ACTIONS = {"halt", "retain", "wait", "reaped", "resume"}


@dataclass
class Attempt:
    """Private in-memory identity for one provider request."""

    digest: str
    route: str
    loop: int
    retry: int
    correlation: str
    components: dict[str, str]
    byte_lengths: dict[str, int]
    streamed: bool = False
    finished: bool = False
    started_ns: int | None = None
    dispatch_ns: int | None = None
    first_wire_ns: int | None = None
    first_visible_ns: int | None = None
    first_visible_category: str | None = None
    wire_event_count: int = 0
    visible_event_count: int = 0
    callback_depth: int = 0
    callback_stats: dict[str, list[int]] | None = None
    transport_stats: dict[str, list[int]] | None = None
    ambiguity_recorded: bool = False
    reconciliation_actions: set[str] | None = None
    checkpoints: set[str] | None = None
    disposition: str | None = None

    def __post_init__(self) -> None:
        if self.callback_stats is None:
            self.callback_stats = {}
        if self.transport_stats is None:
            self.transport_stats = {}
        if self.reconciliation_actions is None:
            self.reconciliation_actions = set()
        if self.checkpoints is None:
            self.checkpoints = set()


_ACTIVE_ATTEMPT: ContextVar[Attempt | None] = ContextVar(
    "physical_attempt_stage_latency", default=None
)


def activate_attempt(attempt: Attempt | None) -> None:
    """Bind a live streamed attempt, or clear attribution when provenance is unknown."""
    if attempt is None:
        _ACTIVE_ATTEMPT.set(None)
    elif isinstance(attempt, Attempt) and attempt.streamed and not attempt.finished:
        _ACTIVE_ATTEMPT.set(attempt)


def current_attempt() -> Attempt | None:
    attempt = _ACTIVE_ATTEMPT.get()
    if isinstance(attempt, Attempt) and attempt.streamed and not attempt.finished:
        return attempt
    return None


def mark_dispatch(attempt: Attempt | None) -> None:
    if not isinstance(attempt, Attempt) or not attempt.streamed or attempt.finished:
        return
    try:
        if attempt.dispatch_ns is None:
            attempt.dispatch_ns = time.monotonic_ns()
        activate_attempt(attempt)
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
        attempt.callback_depth += 1
        return attempt, category, now
    except Exception:
        return None


def end_callback(marker: tuple[Attempt, str, int] | None) -> None:
    _end_stage(marker, transport=False)


def begin_transport(kind: str) -> tuple[Attempt, str, int] | None:
    attempt = current_attempt()
    if attempt is None or attempt.callback_depth <= 0:
        return None
    try:
        safe_kind = kind if kind in _TRANSPORT_KINDS else "other"
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
        assert target is not None
        stats = target.setdefault(category, [0, 0, 0])
        stats[0] += 1
        stats[1] += elapsed
        stats[2] = max(stats[2], elapsed)
    except Exception:
        return


def enabled() -> bool:
    """Return whether local paired-attempt diagnostics were explicitly enabled."""
    try:
        from hermes_cli.config import read_raw_config_readonly

        config = read_raw_config_readonly() or {}
        value = config.get("observability", {}).get("physical_attempt_digests", {})
        return isinstance(value, dict) and value.get("enabled") is True
    except Exception:
        return False


def prepare_cache_scope(value: Any) -> dict[str, str] | None:
    """Digest cache-scope input without retaining it."""
    if not enabled():
        return None
    try:
        return {"digest": _digest(_key(), "cache_scope", value)}
    except Exception:
        return None


def take_cache_scope(request: dict[str, Any]) -> dict[str, str] | None:
    """Remove and return the private, short-lived cache-scope envelope."""
    scope = request.pop(_CACHE_SCOPE_ENVELOPE, None)
    return scope if isinstance(scope, dict) else None


def start_attempt(
    request: dict[str, Any],
    *,
    api_mode: str,
    route: str,
    provider: str,
    model: str,
    retry: int,
    loop: int | None,
    correlation: str,
    scope: dict[str, str] | None = None,
    streamed: bool = False,
) -> Attempt | None:
    """Record an opt-in HMAC-only request identity and adjacent-loop pair."""
    del api_mode, provider, model
    if not enabled():
        return None
    try:
        key = _key()
        component_values = {
            "prefix": _prefix(request),
            "tools": request.get("tools") or request.get("toolConfig") or [],
            "cache_scope": {"scope": (scope or {}).get("digest"), "key": _cache_key(request)},
            "later_history": _later_history(request),
        }
        component_bytes = {
            name: _serialized(value) for name, value in component_values.items()
        }
        components = {
            name: hmac.new(key, name.encode() + b"\0" + value, hashlib.sha256).hexdigest()
            for name, value in component_bytes.items()
        }
        byte_lengths = {name: len(value) for name, value in component_bytes.items()}
        attempt = Attempt(
            digest=hmac.new(key, b"attempt\0" + secrets.token_bytes(32), hashlib.sha256).hexdigest(),
            route=_label(route),
            loop=loop if loop is not None and loop >= 0 else -1,
            retry=max(0, int(retry)),
            correlation=correlation,
            components=components,
            byte_lengths=byte_lengths,
            streamed=streamed,
            started_ns=time.monotonic_ns(),
        )
        _append({
            "schema": "hermes.physical_attempt.v1",
            "phase": "attempt",
            "attempt_digest": attempt.digest,
            "route": attempt.route,
            "loop": attempt.loop,
            "retry": attempt.retry,
            "digests": components,
            "byte_lengths": byte_lengths,
            "monotonic_ns": attempt.started_ns,
            "streamed": streamed,
        })
        _pair(attempt)
        return attempt
    except Exception:
        return None


def finish_attempt(
    attempt: Attempt | None,
    *,
    usage: Any,
    outcome: str,
    api_mode: str,
    provider: str,
) -> None:
    """Append one content-free terminal record for a physical dispatch."""
    if not isinstance(attempt, Attempt) or attempt.finished:
        return
    attempt.finished = True
    try:
        terminal_ns = time.monotonic_ns()
        if _ACTIVE_ATTEMPT.get() is attempt:
            _ACTIVE_ATTEMPT.set(None)
        from agent.usage_pricing import normalize_usage

        canonical = normalize_usage(usage, api_mode=api_mode, provider=provider)
        cache_present = bool(usage) and canonical.cache_telemetry_present
        record = {
            "schema": "hermes.physical_attempt.v1",
            "phase": "terminal",
            "attempt_digest": attempt.digest,
            "monotonic_ns": terminal_ns,
            "outcome": outcome
            if outcome in {"completed", "incomplete", "failed", "error", "cancelled", "interrupted"}
            else "unknown",
            "disposition": attempt.disposition or outcome,
            "cache_state": (
                "unknown"
                if not cache_present
                else "hit"
                if canonical.cache_read_tokens > 0
                else "miss"
            ),
            "input_tokens": canonical.input_tokens if usage else None,
            "output_tokens": canonical.output_tokens if usage else None,
            "cache_read_tokens": canonical.cache_read_tokens if cache_present else None,
            "cache_write_tokens": canonical.cache_write_tokens if cache_present else None,
        }
        if attempt.streamed:
            record["stage_latency"] = _stage_latency(attempt, terminal_ns)
        _append(record)
    except Exception:
        return


def record_checkpoint(attempt: Attempt | None, *, checkpoint: str) -> None:
    """Append a bounded lifecycle checkpoint without provider content."""
    if not isinstance(attempt, Attempt) or attempt.finished:
        return
    value = checkpoint if checkpoint in {"dispatch", "wire", "semantic"} else "semantic"
    checkpoints = attempt.checkpoints
    if checkpoints is None or value in checkpoints:
        return
    checkpoints.add(value)
    try:
        _append({
            "schema": "hermes.physical_attempt.v1",
            "phase": "checkpoint",
            "attempt_digest": attempt.digest,
            "monotonic_ns": time.monotonic_ns(),
            "checkpoint": value,
        })
    except Exception:
        return


def record_ambiguity(attempt: Attempt | None, *, failure_class: str) -> None:
    """Record acceptance ambiguity without exception text or response content."""
    if (
        not isinstance(attempt, Attempt)
        or attempt.finished
        or attempt.ambiguity_recorded
    ):
        return
    attempt.ambiguity_recorded = True
    attempt.disposition = "ambiguous_halt"
    try:
        _append({
            "schema": "hermes.physical_attempt.v1",
            "phase": "ambiguity",
            "attempt_digest": attempt.digest,
            "monotonic_ns": time.monotonic_ns(),
            "failure_class": failure_class
            if failure_class in _AMBIGUITY_CLASSES
            else "transport",
            "acceptance": "unknown",
        })
    except Exception:
        return


def record_reconciliation(attempt: Attempt | None, *, action: str) -> None:
    """Record a bounded reconciliation action once per attempt."""
    if not isinstance(attempt, Attempt):
        return
    value = action if action in _RECONCILIATION_ACTIONS else "halt"
    actions = attempt.reconciliation_actions
    if actions is None or value in actions:
        return
    actions.add(value)
    try:
        _append({
            "schema": "hermes.physical_attempt.v1",
            "phase": "reconciliation",
            "attempt_digest": attempt.digest,
            "monotonic_ns": time.monotonic_ns(),
            "action": value,
        })
    except Exception:
        return


def _stage_latency(attempt: Attempt, terminal_ns: int) -> dict[str, Any]:
    dispatch = attempt.dispatch_ns
    first_wire = attempt.first_wire_ns
    first_visible = attempt.first_visible_ns

    def stats(values: dict[str, list[int]] | None) -> dict[str, dict[str, int]]:
        return {
            category: {"count": row[0], "total_ns": row[1], "max_ns": row[2]}
            for category, row in sorted((values or {}).items())
        }

    return {
        "dispatch_monotonic_ns": dispatch,
        "dispatch_ns": max(0, dispatch - attempt.started_ns)
        if dispatch is not None and attempt.started_ns is not None
        else None,
        "duration_ns": max(0, terminal_ns - dispatch)
        if dispatch is not None
        else None,
        "ttfb_ns": max(0, first_wire - dispatch)
        if dispatch is not None and first_wire is not None
        else None,
        "first_visible_ns": max(0, first_visible - dispatch)
        if dispatch is not None and first_visible is not None
        else None,
        "wire_to_visible_ns": max(0, first_visible - first_wire)
        if first_wire is not None and first_visible is not None
        else None,
        "first_visible_category": attempt.first_visible_category,
        "wire_event_count": attempt.wire_event_count,
        "visible_event_count": attempt.visible_event_count,
        "callbacks": stats(attempt.callback_stats),
        "transports": stats(attempt.transport_stats),
    }


def _pair(current: Attempt) -> None:
    if current.loop < 0 or not current.correlation:
        return
    identity = (current.correlation, current.route)
    with _LOCK:
        previous = _LAST_ATTEMPT.pop(identity, None)
        _LAST_ATTEMPT[identity] = current
        while len(_LAST_ATTEMPT) > _MAX_PAIR_IDENTITIES:
            _LAST_ATTEMPT.pop(next(iter(_LAST_ATTEMPT)))
    if previous is None or previous.loop != current.loop - 1:
        return
    names = ("prefix", "tools", "cache_scope", "later_history")
    equal = {name: previous.components[name] == current.components[name] for name in names}
    _append({
        "schema": "hermes.physical_attempt.v1",
        "phase": "pair",
        "previous_attempt_digest": previous.digest,
        "current_attempt_digest": current.digest,
        "previous_loop": previous.loop,
        "current_loop": current.loop,
        "previous_attempt_retry": previous.retry,
        "digests": {
            name: current.components[name]
            for name in ("cache_scope", "later_history", "prefix", "tools")
        },
        "byte_lengths": current.byte_lengths,
        "equal": equal,
        "first_differing_segment": next(
            (name for name in names if not equal[name]), "none"
        ),
    })


def _prefix(request: dict[str, Any]) -> Any:
    if request.get("instructions") is not None:
        return request["instructions"]
    if request.get("system") is not None:
        return request["system"]
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    prefix = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
            break
        prefix.append(message)
    return prefix


def _later_history(request: dict[str, Any]) -> Any:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    if request.get("instructions") is not None or request.get("system") is not None:
        return messages
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
            return messages[index:]
    return []


def _cache_key(request: dict[str, Any]) -> Any:
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict) and extra_body.get("prompt_cache_key") is not None:
        return extra_body["prompt_cache_key"]
    return request.get("prompt_cache_key")


def _digest(key: bytes, label: str, value: Any) -> str:
    encoded = _serialized(value)
    return hmac.new(key, label.encode() + b"\0" + encoded, hashlib.sha256).hexdigest()


def _serialized(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")


def _label(value: Any) -> str:
    text = str(value or "")
    return text if text and len(text) <= 128 and "://" not in text else "unknown"


def _root() -> Path:
    return get_hermes_home() / "observability"


def _key() -> bytes:
    root = _root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise PermissionError("unsafe physical attempt diagnostics directory")
    if stat.S_IMODE(info.st_mode) != 0o700:
        root.chmod(0o700)
    path = root / _KEY_FILE
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        fd = _private_fd(path, os.O_RDONLY)
        try:
            value = os.read(fd, 33)
        finally:
            os.close(fd)
        if len(value) != 32:
            raise ValueError("invalid physical attempt digest key")
        return value
    value = secrets.token_bytes(32)
    try:
        os.write(fd, value)
    finally:
        os.close(fd)
    return value


def _private_fd(path: Path, flags: int) -> int:
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        os.close(fd)
        raise PermissionError("unsafe physical attempt diagnostics file")
    return fd


def _append(record: dict[str, Any]) -> None:
    path = _root() / _RECORDS_FILE
    fd = _private_fd(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    try:
        os.write(fd, (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    finally:
        os.close(fd)
