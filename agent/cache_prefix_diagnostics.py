"""Opt-in, local-only evidence for Codex Responses cache-prefix changes."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import secrets
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable

_MAX_BYTES = 4 * 1024 * 1024
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_key_lock = threading.Lock()


def _enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly

        value = (load_config_readonly().get("agent") or {}).get("codex_cache_diagnostics") or {}
        return bool(value.get("enabled", False))
    except Exception:
        return False


def _paths() -> tuple[Path, Path, Path]:
    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    directory = home / "cache"
    return directory, directory / "codex-cache-prefix.jsonl", directory / "codex-cache-prefix.lock"


def _safe_dir(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
        mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise OSError("diagnostic directory is not a real directory")
    os.chmod(path, 0o700)


def _key(directory: Path) -> bytes:
    path = directory / "codex-cache-prefix.key"
    with _key_lock:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
            try:
                value = os.read(fd, 128)
            finally:
                os.close(fd)
            if len(value) != 32:
                raise OSError("invalid diagnostics key")
            return value
        try:
            value = secrets.token_bytes(32)
            os.write(fd, value)
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
            return value
        finally:
            os.close(fd)


def _digest(key: bytes, value: Any) -> dict[str, Any]:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()
    return {"hmac": hmac.new(key, encoded, hashlib.sha256).hexdigest(), "bytes": len(encoded)}


def _segments(request: dict[str, Any], key: bytes) -> dict[str, Any]:
    items = request.get("input")
    messages = items if isinstance(items, list) else []
    tools = request.get("tools") if isinstance(request.get("tools"), list) else []
    scope = {"prompt_cache_key": request.get("prompt_cache_key"), "model": request.get("model")}
    return {
        "static": _digest(key, request.get("instructions")),
        "messages": _digest(key, messages),
        "tools": _digest(key, tools),
        "scope": _digest(key, scope),
    }


def _usage(response: Any) -> int | None:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None) if usage is not None else None
    value = getattr(details, "cached_tokens", None) if details is not None else None
    if value is None and isinstance(usage, dict):
        details = usage.get("input_tokens_details") or {}
        value = details.get("cached_tokens") if isinstance(details, dict) else None
    return value if isinstance(value, int) and value >= 0 else None


def _append(row: dict[str, Any]) -> None:
    directory, output, lock_path = _paths()
    _safe_dir(directory)
    key = _key(directory)
    del key
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | _O_NOFOLLOW, 0o600)
    try:
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        try:
            current = output.stat().st_size
        except FileNotFoundError:
            current = 0
        if current + len(payload) <= _MAX_BYTES:
            fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _O_NOFOLLOW, 0o600)
            try:
                os.write(fd, payload)
                os.fchmod(fd, 0o600)
            finally:
                os.close(fd)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def record_attempt_start(request: dict[str, Any], *, correlation: str = "", attempt: int | None = None) -> tuple[str, dict[str, Any]] | None:
    if not _enabled():
        return None
    directory, _output, _lock = _paths()
    try:
        _safe_dir(directory)
        key = _key(directory)
    except OSError:
        return None
    correlation_digest = _digest(key, correlation or "anonymous")["hmac"]
    token = (correlation_digest, _segments(request, key))
    _append({"kind": "start", "at": time.time_ns(), "correlation": correlation_digest, "attempt": attempt, "segments": token[1]})
    return token


def record_attempt_terminal(token: tuple[str, dict[str, Any]] | None, response: Any = None) -> None:
    if token is None:
        return
    _append({"kind": "terminal", "at": time.time_ns(), "correlation": token[0], "cache_tokens": _usage(response)})


def call_final_codex_create(create: Callable[..., Any], request: dict[str, Any], *, correlation: str = "", attempt: int | None = None) -> Any:
    token = record_attempt_start(request, correlation=correlation, attempt=attempt)
    try:
        response = create(**request)
    except BaseException:
        record_attempt_terminal(token)
        raise
    record_attempt_terminal(token, response)
    return response


__all__ = ["call_final_codex_create", "record_attempt_start", "record_attempt_terminal"]
