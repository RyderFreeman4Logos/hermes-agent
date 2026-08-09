"""Opt-in, content-free diagnostics for physical provider attempts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


_SCHEMA = "hermes.physical_attempt.v1"
_ROOT_PARTS = ("observability",)
_KEY_FILE = "physical_attempt_digests.key"
_RECORDS_FILE = "physical_attempt_digests.jsonl"
_ROLES = {"main", "subagent", "reviewer", "fallback", "auxiliary", "unknown"}
_OUTCOMES = {"completed", "incomplete", "failed", "error", "cancelled", "interrupted"}
_INTERNAL_SCOPE_KEY = "_hermes_physical_attempt_scope"


@dataclass
class Attempt:
    digest: str
    finished: bool = False


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
        record = {
            "schema": _SCHEMA,
            "phase": "start",
            "attempt_digest": attempt_digest,
            "monotonic_ns": time.monotonic_ns(),
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
        return Attempt(attempt_digest)
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
        _append({
            "schema": _SCHEMA,
            "phase": "terminal",
            "attempt_digest": attempt.digest,
            "monotonic_ns": time.monotonic_ns(),
            "outcome": outcome if outcome in _OUTCOMES else "unknown",
            "cache_state": cache_state,
            "input_tokens": canonical.input_tokens if usage else None,
            "output_tokens": canonical.output_tokens if usage else None,
            "cache_read_tokens": canonical.cache_read_tokens if cache_present else None,
            "cache_write_tokens": canonical.cache_write_tokens
            if cache_present
            else None,
        })
    except Exception:
        return


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
    text = str(value or "unknown")[:limit]
    return "".join(char if char.isalnum() or char in "._:/-" else "_" for char in text)


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
