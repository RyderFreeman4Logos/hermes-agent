"""Opt-in exact provider-bound request capture for cache debugging."""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import stat
import time
from pathlib import Path
from typing import Any

from agent.redact import _SENSITIVE_BODY_KEYS, _redact_url_userinfo, redact_sensitive_text
from hermes_constants import get_hermes_home

__all__ = [
    "capture_provider_request",
    "compare_captures",
    "enabled",
    "strict_write_enabled",
]

_SCHEMA = "hermes.cache_request.v1"
_REDACTED = "[REDACTED]"
_NORMALIZE_KEY = re.compile(r"[^a-z0-9]+")
_SECRET_KEY_TOKEN = re.compile(
    r"(?:^|[_-])(?:secrets?|tokens?|passwords?|credentials?|authorization|cookies?)(?:[_-]|$)",
    re.IGNORECASE,
)
_API_OR_ACCESS_KEY_TOKEN = re.compile(
    r"(?:^|[_-])(?:api|access)[_-]key(?:[_-]|$)", re.IGNORECASE
)
_URI_USERINFO = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/@]*@", re.IGNORECASE)
_PRESERVE_KEYS = {
    "body",
    "bodybytes",
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
    body: bytes | None = None,
) -> None:
    """Persist one exact physical provider request immediately before transport.

    The input is copied while redacting secrets; the
    provider's request is untouched. Capture is best-effort unless strict write
    mode is enabled. ``body`` is already-buffered HTTP bytes when available.
    Privacy wins over exact-wire identity: sensitive bytes are sanitized or
    omitted and labeled; unchanged secret-free bytes may be ``exact_wire``.
    """
    if not enabled():
        return
    identity = tuple(_identity_value(value) for value in (route, provider, model, api_mode))
    request_model = _identity_value(request.get("model"))
    if not all(identity) or (request_model is not None and request_model != identity[2]):
        return
    try:
        retry_value = max(0, int(retry))
    except (TypeError, ValueError):
        retry_value = 0
    try:
        request_payload = _redact(request)
        if isinstance(body, (bytes, bytearray)):
            body_bytes, status = _sanitize_transport_body(bytes(body))
        else:
            body_bytes = _serialize_body(request_payload)
            status = "kwargs_fallback"
        _persist(
            {
                "schema": _SCHEMA,
                "timestamp_ns": time.time_ns(),
                "route": {
                    "route": identity[0],
                    "provider": identity[1],
                    "model": identity[2],
                    "api_mode": identity[3],
                },
                "physical_attempt": {
                    "correlation": correlation,
                    "attempt_id": attempt_id,
                    "retry": retry_value,
                },
                "request": request_payload,
                "body_bytes": {
                    "encoding": "base64",
                    "data": base64.b64encode(body_bytes).decode("ascii"),
                    "status": status,
                },
            }
        )
    except Exception:
        if strict_write_enabled():
            raise


def _serialize_body(request: dict[str, Any]) -> bytes:
    """Serialize the sanitized provider payload exactly once for persistence."""
    return json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sanitize_transport_body(body: bytes) -> tuple[bytes, str]:
    """Return persistable body bytes and an honest capture status.

    ``exact_wire`` is only for unchanged secret-free buffered bytes.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return b"", "omitted"
    redacted_text = _redact_scalar(text)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        if redacted_text != text:
            return redacted_text.encode("utf-8"), "sanitized"
        return body, "exact_wire"
    redacted_obj = _redact(parsed)
    if redacted_obj != parsed or redacted_text != text:
        return json.dumps(redacted_obj, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        ), "sanitized"
    return body, "exact_wire"


def _identity_value(value: Any) -> str | None:
    identity = str(value or "").strip()
    return identity if identity and identity.lower() != "unknown" else None


def _redact(value: Any, *, preserve: bool = False) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            normalized = _NORMALIZE_KEY.sub("", str(key).lower())
            if _is_secret_key(key):
                result[str(key)] = _REDACTED
            elif normalized == "bodybytes":
                result[str(key)] = _copy_exact(child)
            elif preserve or normalized in _PRESERVE_KEYS:
                result[str(key)] = _redact(child, preserve=True)
            else:
                result[str(key)] = _redact(child)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(child, preserve=preserve) for child in value]
    if isinstance(value, str):
        return _redact_scalar(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _redact_scalar(str(value))


def _copy_exact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy_exact(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_exact(child) for child in value]
    return value


def _redact_scalar(value: str) -> str:
    redacted = _redact_url_userinfo(
        redact_sensitive_text(value, force=True, redact_url_credentials=True)
    )
    return _REDACTED if _URI_USERINFO.search(redacted) else redacted


def _is_secret_key(key: Any) -> bool:
    name = str(key)
    normalized = name.lower().replace("-", "_")
    return (
        normalized in _SENSITIVE_BODY_KEYS
        or bool(_SECRET_KEY_TOKEN.search(name))
        or bool(_API_OR_ACCESS_KEY_TOKEN.search(name))
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
    data = (
        json.dumps(_redact(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
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


def _body_bytes(payload: dict[str, Any]) -> tuple[bytes, str]:
    record = payload.get("body_bytes") if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        return b"", "kwargs_fallback"
    status = str(record.get("status") or "kwargs_fallback")
    data = record.get("data")
    if not isinstance(data, str) or not data:
        return b"", status
    try:
        return base64.b64decode(data), status
    except (TypeError, ValueError):
        return b"", status


def _first_byte(left: bytes, right: bytes) -> int | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit if len(left) != len(right) else None


def _first_difference(left: Any, right: Any, path: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
    if type(left) is not type(right) or not isinstance(left, (dict, list)):
        return path if left != right else None
    if isinstance(left, list):
        if len(left) != len(right):
            return path + ("length",)
        for index, (left_child, right_child) in enumerate(zip(left, right)):
            difference = _first_difference(left_child, right_child, path + (index,))
            if difference is not None:
                return difference
        return None
    preferred = (
        "messages",
        "input",
        "tools",
        "toolConfig",
        "system",
        "instructions",
        "prompt_cache_key",
        "cache_control",
    )
    keys = [key for key in preferred if key in left or key in right]
    keys.extend(sorted((left.keys() | right.keys()) - set(keys), key=str))
    for key in keys:
        if key not in left or key not in right:
            return path + (key,)
        difference = _first_difference(left[key], right[key], path + (key,))
        if difference is not None:
            return difference
    return None


def _json_pointer(path: tuple[Any, ...] | None) -> str | None:
    if path is None:
        return None
    parts = []
    for item in path:
        if item == "length":
            parts.append("-")
            continue
        text = str(item).replace("~", "~0").replace("/", "~1")
        parts.append(text)
    return "/" + "/".join(parts)


def _message_location(path: tuple[Any, ...] | None) -> dict[str, Any]:
    if not path or path[0] not in {"messages", "input"}:
        return {"index": None, "field": None}
    index = path[1] if len(path) > 1 and isinstance(path[1], int) else None
    field = path[-1] if len(path) > 1 and path[-1] not in {"messages", "input", "length"} else None
    return {"index": index, "field": field}


def _tools_changed(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (left.get("tools") or left.get("toolConfig")) != (
        right.get("tools") or right.get("toolConfig")
    )


def _cache_scope_changed(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("prompt_cache_key", "cache_control")
    return any(left.get(key) != right.get(key) for key in keys)


def compare_captures(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare two persisted captures. Never claims exact-wire on sanitized/omitted bytes."""
    left_body, left_status = _body_bytes(left)
    right_body, right_status = _body_bytes(right)
    wire_comparable = left_status == "exact_wire" and right_status == "exact_wire"
    left_request = left.get("request") if isinstance(left.get("request"), dict) else {}
    right_request = right.get("request") if isinstance(right.get("request"), dict) else {}
    if not isinstance(left_request, dict):
        left_request = {}
    if not isinstance(right_request, dict):
        right_request = {}
    path = _first_difference(left_request, right_request)
    equal_bodies = left_body == right_body
    return {
        "equal": equal_bodies and path is None,
        "wire_comparable": wire_comparable,
        "claim": "exact_wire" if wire_comparable else "unavailable_wire",
        "first_differing_byte": _first_byte(left_body, right_body) if not equal_bodies else None,
        "left_bytes": len(left_body),
        "right_bytes": len(right_body),
        "json_pointer": _json_pointer(path),
        "message": _message_location(path),
        "tools": {"changed": _tools_changed(left_request, right_request)},
        "cache_scope": {"changed": _cache_scope_changed(left_request, right_request)},
        "left": {"status": left_status},
        "right": {"status": right_status},
    }
