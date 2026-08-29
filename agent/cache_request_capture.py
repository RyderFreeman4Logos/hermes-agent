"""Opt-in exact provider-bound request capture for cache debugging."""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

__all__ = ["capture_provider_request", "enabled", "strict_write_enabled"]

_SCHEMA = "hermes.cache_request.v1"
_REDACTED = "[REDACTED]"
_SECRET_KEY = re.compile(r"[^a-z0-9]+")
_SECRET_KEYS = {
    "authorization",
    "apikey",
    "apikeys",
    "cookie",
    "cookies",
    "refreshtoken",
    "refreshtokens",
    "setcookie",
}
_PRESERVE_KEYS = {
    "body",
    "cachecontrol",
    "input",
    "message",
    "messages",
    "prompt",
    "prompts",
    "tool",
    "toolconfig",
    "tools",
}


def _settings() -> dict[str, Any]:
    try:
        from hermes_cli.config import read_raw_config_readonly

        config = read_raw_config_readonly() or {}
        debug = config.get("debug", {})
        settings = debug.get("cache_requests", {}) if isinstance(debug, dict) else {}
        return settings if isinstance(settings, dict) else {}
    except Exception:
        return {}


def enabled() -> bool:
    """Return whether exact request capture was explicitly enabled."""
    return _settings().get("enabled") is True


def strict_write_enabled() -> bool:
    """Return whether capture write failures should fail provider calls."""
    return _settings().get("strict_write") is True


def capture_provider_request(
    request: dict[str, Any],
    *,
    api_mode: str = "unknown",
    route: str = "unknown",
    provider: str = "unknown",
    model: str = "unknown",
    correlation: str | None = None,
    attempt_id: str | None = None,
    retry: int = 0,
) -> None:
    """Persist one exact physical provider request immediately before transport.

    The input is copied while redacting transport credentials only; the
    provider's request is untouched. Capture is best-effort unless strict write
    mode is enabled.
    """
    if not enabled():
        return
    try:
        retry_value = max(0, int(retry))
    except (TypeError, ValueError):
        retry_value = 0
    try:
        _persist(
            {
                "schema": _SCHEMA,
                "timestamp_ns": time.time_ns(),
                "route": {
                    "route": route,
                    "provider": provider,
                    "model": model,
                    "api_mode": api_mode,
                },
                "physical_attempt": {
                    "correlation": correlation,
                    "attempt_id": attempt_id,
                    "retry": retry_value,
                },
                "request": _redact(request),
            }
        )
    except Exception:
        if strict_write_enabled():
            raise


def _redact(value: Any, *, preserve: bool = False) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            normalized = _SECRET_KEY.sub("", str(key).lower())
            if preserve or normalized in _PRESERVE_KEYS:
                result[str(key)] = _copy_exact(child)
            elif _is_secret_key(key):
                result[str(key)] = _REDACTED
            else:
                result[str(key)] = _redact(child)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(child, preserve=preserve) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _copy_exact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy_exact(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_exact(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_secret_key(key: Any) -> bool:
    normalized = _SECRET_KEY.sub("", str(key).lower())
    return (
        normalized in _SECRET_KEYS
        or normalized.startswith("xapikey")
        or normalized.endswith(
            (
                "authorization",
                "apikey",
                "apikeys",
                "cookie",
                "cookies",
                "refreshtoken",
                "refreshtokens",
            )
        )
    )


def _open_private_dir_chain(path: Path) -> int:
    """Open/create capture directories with no-follow dirfd traversal."""
    absolute = path.is_absolute()
    parts = path.parts[1:] if absolute else path.parts
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path.anchor if absolute else ".", flags)
    try:
        for index, part in enumerate(parts):
            created = False
            try:
                os.mkdir(part, 0o700, dir_fd=fd)
                created = True
            except FileExistsError:
                pass
            child = os.open(part, flags, dir_fd=fd)
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise PermissionError("unsafe cache request capture directory")
            if created or index == len(parts) - 1:
                if info.st_uid != os.geteuid():
                    os.close(child)
                    raise PermissionError("foreign cache request capture directory")
                if stat.S_IMODE(info.st_mode) != 0o700:
                    os.fchmod(child, 0o700)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _persist(payload: dict[str, Any]) -> None:
    rootfd = _open_private_dir_chain(get_hermes_home() / "debug" / "cache-requests")
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    temp_name: str | None = None
    fd: int | None = None
    try:
        for _ in range(8):
            final_name = f"request-{time.time_ns()}-{secrets.token_hex(6)}.json"
            temp_name = f".tmp-{secrets.token_hex(12)}"
            try:
                fd = os.open(
                    temp_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=rootfd,
                )
            except FileExistsError:
                temp_name = None
                continue
            os.fchmod(fd, 0o600)
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
            os.fsync(fd)
            os.close(fd)
            fd = None
            try:
                os.link(temp_name, final_name, src_dir_fd=rootfd, dst_dir_fd=rootfd)
            except FileExistsError:
                os.unlink(temp_name, dir_fd=rootfd)
                temp_name = None
                continue
            os.unlink(temp_name, dir_fd=rootfd)
            temp_name = None
            os.fsync(rootfd)
            return
        raise FileExistsError("could not allocate cache request capture name")
    finally:
        if fd is not None:
            os.close(fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=rootfd)
            except OSError:
                pass
        os.close(rootfd)
