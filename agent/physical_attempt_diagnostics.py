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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no POSIX flock.
    fcntl = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home

__all__ = ["Attempt", "enabled", "prepare_cache_scope", "start_attempt"]

_KEY_FILE = "physical_attempt_digests.key"
_RECORDS_FILE = "physical_attempt_digests.jsonl"
_LOCK = threading.Lock()
_LAST_ATTEMPT: dict[tuple[str, str, str, str], "Attempt"] = {}
_CACHE_SCOPE_ENVELOPE = "_hermes_physical_attempt_cache_scope"
# ponytail: fixed local caps; add configurable rotation only if diagnostics volume needs it.
_MAX_TRACKED_CORRELATIONS = 256
_MAX_RECORDS_BYTES = 8 * 1024 * 1024
_MAX_COMPONENT_BYTES = 1024 * 1024
_LABEL_ALLOWLIST = frozenset(
    {
        "anthropic_messages",
        "chat_completions",
        "codex_responses",
        "openai_chat",
        "responses",
        "unknown",
        "system",
        "developer",
        "user",
        "assistant",
        "tool",
        "function",
        "text",
        "image_url",
    }
)
_SAFE_STRING_VALUES = _LABEL_ALLOWLIST | {"ephemeral", "persistent", "object", "array"}


def _opaque_digest(value: Any, key: bytes) -> str:
    """Return a content-free stable token for one scalar value."""
    if isinstance(value, str):
        raw = value.encode("utf-8", "replace")
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raw = type(value).__name__.encode("utf-8")
    return hmac.new(key, b"opaque\0" + raw, hashlib.sha256).hexdigest()


def _sanitize_value(value: Any, key: bytes, *, depth: int = 0) -> Any:
    """Keep only bounded structure; never pass request values downstream."""
    if depth > 8:
        return {"type": type(value).__name__, "depth": "limit"}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if value in _SAFE_STRING_VALUES:
            return value
        return {
            "type": "string",
            "length": len(value.encode("utf-8", "replace")),
            "digest": _opaque_digest(value, key),
        }
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "length": len(value), "digest": _opaque_digest(value, key)}
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_name, child in list(value.items())[:256]:
            name = str(raw_name)
            if not _safe_key(name) or len(name) > 128:
                continue
            result[name] = _sanitize_value(child, key, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(child, key, depth=depth + 1) for child in value[:256]]
    return {"type": type(value).__name__, "digest": _opaque_digest(value, key)}


def _raw_json_size(value: Any, *, depth: int = 0) -> int:
    """Measure JSON-shaped input without serializing or retaining its values."""
    if depth > 8:
        return 0
    if value is None:
        return 4
    if value is True:
        return 4
    if value is False:
        return 5
    if isinstance(value, int):
        return len(str(value))
    if isinstance(value, float):
        if value != value:
            return 3
        if value == float("inf"):
            return 8
        if value == float("-inf"):
            return 9
        return len(repr(value))
    if isinstance(value, str):
        size = 2
        for char in value:
            code = ord(char)
            if char in {'"', "\\"} or char in "\b\f\n\r\t":
                size += 2
            elif code < 0x20:
                size += 6
            else:
                size += len(char.encode("utf-8", "surrogatepass"))
        return size
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, dict):
        items = list(value.items())[:256]
        return 2 + sum(
            _raw_json_size(str(name), depth=depth + 1)
            + 1
            + _raw_json_size(child, depth=depth + 1)
            + (1 if index else 0)
            for index, (name, child) in enumerate(items)
        )
    if isinstance(value, (list, tuple)):
        items = value[:256]
        return 2 + sum(
            _raw_json_size(child, depth=depth + 1) + (1 if index else 0)
            for index, child in enumerate(items)
        )
    return len(type(value).__name__)


@dataclass
class Attempt:
    """Private in-memory identity for one provider request."""

    digest: str
    route: str
    provider: str
    model: str
    loop: int
    retry: int
    correlation: str
    components: dict[str, str]
    byte_lengths: dict[str, int]


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
        key = _key()
        return {"digest": _digest(key, "cache_scope", _sanitize_value(value, key))}
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
) -> Attempt | None:
    """Record an opt-in HMAC-only request identity and adjacent-loop pair."""
    if not enabled() or loop is None or loop < 0 or not correlation:
        return None
    try:
        key = _key()
        raw_component_values = {
            "prefix": _prefix(request),
            "tools": request.get("tools") or request.get("toolConfig") or [],
            "cache_scope": {"scope": (scope or {}).get("digest"), "key": _cache_key(request)},
            "later_history": _later_history(request, api_mode),
        }
        component_lengths = {
            name: min(_raw_json_size(value), _MAX_COMPONENT_BYTES)
            for name, value in raw_component_values.items()
        }
        component_values = {
            name: _sanitize_value(value, key)
            for name, value in raw_component_values.items()
        }
        component_bytes = {
            name: _serialized(value) for name, value in component_values.items()
        }
        components = {
            name: hmac.new(key, name.encode() + b"\0" + value, hashlib.sha256).hexdigest()
            for name, value in component_bytes.items()
        }
        byte_lengths = component_lengths
        attempt = Attempt(
            digest=hmac.new(key, b"attempt\0" + secrets.token_bytes(32), hashlib.sha256).hexdigest(),
            route=_label(route, key),
            provider=_label(provider, key),
            model=_label(model, key),
            loop=loop,
            retry=max(0, int(retry)),
            correlation=correlation,
            components=components,
            byte_lengths=byte_lengths,
        )
        _append({
            "schema": "hermes.physical_attempt.v1",
            "phase": "attempt",
            "timestamp_ns": time.time_ns(),
            "attempt_digest": attempt.digest,
            "route": attempt.route,
            "provider": attempt.provider,
            "model": attempt.model,
            "loop": attempt.loop,
            "retry": attempt.retry,
            "digests": components,
            "byte_lengths": byte_lengths,
        })
        _pair(attempt)
        return attempt
    except Exception:
        return None


def _pair(current: Attempt) -> None:
    identity = (current.correlation, current.route, current.provider, current.model)
    with _LOCK:
        previous = _LAST_ATTEMPT.pop(identity, None)
        _LAST_ATTEMPT[identity] = current
        while len(_LAST_ATTEMPT) > _MAX_TRACKED_CORRELATIONS:
            _LAST_ATTEMPT.pop(next(iter(_LAST_ATTEMPT)))
    if previous is None or previous.loop != current.loop - 1:
        return
    names = ("prefix", "tools", "cache_scope", "later_history")
    equal = {name: previous.components[name] == current.components[name] for name in names}
    _append({
        "schema": "hermes.physical_attempt.v1",
        "phase": "pair",
        "timestamp_ns": time.time_ns(),
        "route": current.route,
        "provider": current.provider,
        "model": current.model,
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


def _later_history(request: dict[str, Any], api_mode: str) -> Any:
    if api_mode == "codex_responses":
        return request.get("input") or []
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


def _label(value: Any, key: bytes) -> str:
    try:
        safe = _sanitize_value(value, key)
        if isinstance(safe, str) and safe in _LABEL_ALLOWLIST:
            return safe
        encoded = json.dumps(safe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(key, b"label\0" + encoded, hashlib.sha256).hexdigest()
    except Exception:
        return "unknown"


def _safe_key(key: Any) -> bool:
    text = str(key).lower()
    return (
        not text.startswith("_hermes")
        and text not in {"messages", "input", "tools", "toolconfig", "instructions", "system"}
        and not any(
            part in text
            for part in (
                "authorization",
                "cookie",
                "credential",
                "header",
                "password",
                "secret",
                "token",
                "url",
                "uri",
                "connection",
                "prompt_cache_key",
            )
        )
    )


def _root() -> Path:
    return get_hermes_home() / "observability"


def _posix_euid() -> int | None:
    getter = getattr(os, "geteuid", None)
    return getter() if getter is not None else None


def _open_private_dir_chain(path: Path) -> int:
    """Open/create every ancestor with no-follow dirfd traversal."""
    absolute = path.is_absolute()
    parts = path.parts[1:] if absolute else path.parts
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path.anchor if absolute else ".", flags)
    for index, part in enumerate(parts):
        created = False
        try:
            os.mkdir(part, 0o700, dir_fd=fd)
            created = True
        except FileExistsError:
            pass
        child = os.open(part, flags, dir_fd=fd)
        info = os.fstat(child)
        euid = _posix_euid()
        if not stat.S_ISDIR(info.st_mode):
            os.close(child)
            raise PermissionError("unsafe physical attempt diagnostics directory")
        if euid is not None and (created or index == len(parts) - 1):
            if info.st_uid != euid:
                os.close(child)
                raise PermissionError("foreign physical attempt diagnostics directory")
            if stat.S_IMODE(info.st_mode) != 0o700:
                os.fchmod(child, 0o700)
        os.close(fd)
        fd = child
    return fd


def _open_private_leaf(dirfd: int, name: str, flags: int, mode: int) -> int:
    fd = os.open(name, flags | getattr(os, "O_NOFOLLOW", 0), mode, dir_fd=dirfd)
    info = os.fstat(fd)
    euid = _posix_euid()
    if not stat.S_ISREG(info.st_mode) or (
        euid is not None and (info.st_uid != euid or stat.S_IMODE(info.st_mode) & 0o077)
    ):
        os.close(fd)
        raise PermissionError("unsafe physical attempt diagnostics file")
    return fd


def _key() -> bytes:
    root = _root()
    rootfd = _open_private_dir_chain(root)
    try:
        try:
            fd = _open_private_leaf(rootfd, _KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            fd = _open_private_leaf(rootfd, _KEY_FILE, os.O_RDONLY, 0o600)
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
            os.fsync(fd)
        finally:
            os.close(fd)
        return value
    finally:
        os.close(rootfd)


def _private_fd(path: Path, flags: int) -> int:
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    info = os.fstat(fd)
    euid = _posix_euid()
    if not stat.S_ISREG(info.st_mode) or (
        euid is not None
        and (info.st_uid != euid or stat.S_IMODE(info.st_mode) & 0o077)
    ):
        os.close(fd)
        raise PermissionError("unsafe physical attempt diagnostics file")
    return fd


def _append(record: dict[str, Any]) -> None:
    root = _root()
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with _LOCK:
        rootfd = _open_private_dir_chain(root)
        lockfd = recordfd = None
        try:
            lockfd = _open_private_leaf(
                rootfd, ".physical_attempt_digests.lock", os.O_RDWR | os.O_CREAT, 0o600
            )
            if fcntl is not None:
                fcntl.flock(lockfd, fcntl.LOCK_EX)
            recordfd = _open_private_leaf(
                rootfd, _RECORDS_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
            )
            if os.fstat(recordfd).st_size + len(payload) > _MAX_RECORDS_BYTES:
                os.ftruncate(recordfd, 0)
            offset = 0
            while offset < len(payload):
                offset += os.write(recordfd, payload[offset:])
            os.fsync(recordfd)
        finally:
            if recordfd is not None:
                os.close(recordfd)
            if lockfd is not None:
                if fcntl is not None:
                    fcntl.flock(lockfd, fcntl.LOCK_UN)
                os.close(lockfd)
            os.close(rootfd)
