"""Default-off, local-only physical provider-attempt digest records."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any


_UNKNOWN = "unknown"


class PhysicalAttemptDigestSink:
    """Append content-free, HMAC-keyed attempt start/terminal records to JSONL."""

    def __init__(self, home: Path, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._dir = Path(home) / "observability"
        self._key_path = self._dir / "physical_attempt_digests.key"
        self._records_path = self._dir / "physical_attempt_digests.jsonl"
        self._key: bytes | None = None

    def _secret(self) -> bytes | None:
        if not self.enabled:
            return None
        try:
            self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self._dir, 0o700)
            try:
                fd = os.open(
                    self._key_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                fd = None
            if fd is not None:
                with os.fdopen(fd, "wb") as key_file:
                    key_file.write(secrets.token_bytes(32))
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self._key_path, flags)
            try:
                os.fchmod(fd, 0o600)
                key = os.read(fd, 64)
            finally:
                os.close(fd)
            return key if len(key) == 32 else None
        except OSError:
            return None

    @staticmethod
    def _canonical(value: Any) -> bytes | None:
        try:
            return json.dumps(
                value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError):
            return None

    def _digest(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {"value": _UNKNOWN, "bytes": _UNKNOWN}
        data = self._canonical(value)
        if data is None or self._key is None:
            return {"value": _UNKNOWN, "bytes": _UNKNOWN}
        return {
            "value": "hmac-sha256:" + hmac.new(self._key, data, hashlib.sha256).hexdigest(),
            "bytes": len(data),
        }

    def _append(self, record: dict[str, Any]) -> None:
        try:
            fd = os.open(self._records_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.fchmod(fd, 0o600)
                os.write(fd, (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            pass

    def start(
        self,
        *,
        route: Any,
        provider: Any,
        model: Any,
        role: Any,
        retry: Any,
        continuation: Any,
        cache_scope: Any,
        cache_key: Any,
        tools: Any,
        instructions: Any,
        wire_prefix: Any,
    ) -> str | None:
        if not self.enabled:
            return None
        self._key = self._secret()
        if self._key is None:
            return None
        attempt = "hmac-sha256:" + hmac.new(
            self._key, secrets.token_bytes(32), hashlib.sha256
        ).hexdigest()
        self._append(
            {
                "event": "start",
                "attempt": attempt,
                "monotonic_ns": time.monotonic_ns(),
                "route": str(route or _UNKNOWN),
                "provider": str(provider or _UNKNOWN),
                "model": str(model or _UNKNOWN),
                "role": str(role or _UNKNOWN),
                "retry": retry if isinstance(retry, int) else _UNKNOWN,
                "continuation": continuation if isinstance(continuation, int) else _UNKNOWN,
                "digests": {
                    "cache_scope": self._digest(cache_scope),
                    "cache_key": self._digest(cache_key),
                    "tools": self._digest(tools),
                    "static_instructions": self._digest(instructions),
                    "wire_prefix": self._digest(wire_prefix),
                },
            }
        )
        return attempt

    def finish(self, attempt: str | None, usage: Any) -> None:
        if not self.enabled or not attempt:
            return
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", _UNKNOWN))
        self._append(
            {
                "event": "terminal",
                "attempt": attempt,
                "monotonic_ns": time.monotonic_ns(),
                "usage": {
                    "input_tokens": input_tokens if isinstance(input_tokens, int) else _UNKNOWN,
                    "cache_read_tokens": usage.get("cache_read_tokens")
                    if isinstance(usage.get("cache_read_tokens"), int)
                    else _UNKNOWN,
                    "cache_write_tokens": usage.get("cache_write_tokens")
                    if isinstance(usage.get("cache_write_tokens"), int)
                    else _UNKNOWN,
                },
            }
        )


def _enabled_sink() -> PhysicalAttemptDigestSink | None:
    """Construct a sink only when the persisted default-off setting enables it."""
    try:
        from hermes_cli.config import load_config
        from hermes_constants import get_hermes_home

        enabled = (load_config() or {}).get("observability", {}).get(
            "physical_attempt_digests", {}
        ).get("enabled") is True
        return PhysicalAttemptDigestSink(get_hermes_home(), enabled=enabled) if enabled else None
    except Exception:
        return None


def start_codex_attempt(agent: Any, api_kwargs: dict[str, Any], *, retry: int, role: str, identity: Any = None) -> tuple[PhysicalAttemptDigestSink | None, str | None]:
    """Append a start record from internal identity material removed before dispatch."""
    if identity is None:
        identity = api_kwargs.pop("_hermes_physical_attempt_identity", {})
    if not isinstance(identity, dict):
        identity = {}
    sink = _enabled_sink()
    if sink is None:
        return None, None
    return sink, sink.start(
        route="codex_responses",
        provider=getattr(agent, "provider", None),
        model=api_kwargs.get("model"),
        role=role,
        retry=retry,
        continuation=getattr(agent, "_continuation_epoch", _UNKNOWN),
        cache_scope=identity.get("cache_scope"),
        cache_key=identity.get("cache_key", api_kwargs.get("prompt_cache_key")),
        tools=identity.get("tools", api_kwargs.get("tools")),
        instructions=identity.get("instructions", api_kwargs.get("instructions")),
        wire_prefix=identity.get("wire_prefix", api_kwargs.get("input")),
    )


def response_usage(response: Any) -> dict[str, Any]:
    """Select only numeric cache-usage fields from a terminal response."""
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        return usage
    cached = getattr(usage, "cache_read_tokens", None)
    details = getattr(usage, "input_tokens_details", None)
    if not isinstance(cached, int):
        cached = getattr(details, "cached_tokens", None)
    values = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "cache_read_tokens": cached,
        "cache_write_tokens": getattr(usage, "cache_write_tokens", None),
    }
    return {key: value for key, value in values.items() if type(value) is int}
