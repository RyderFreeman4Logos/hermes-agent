"""Sanitized same-route request-pair dumps for low cache-hit responses."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from agent.physical_attempt_diagnostics import (
    _cache_key,
    _digest,
    _later_history,
    _prefix,
    _serialized,
    _label,
    _sanitize_value,
    _open_private_dir_chain,
    _posix_euid,
)
from agent.usage_pricing import CanonicalUsage

__all__ = [
    "MAX_DUMPS",
    "maybe_dump_on_usage",
    "remember_sent_request",
    "reset_for_tests",
]

MAX_DUMPS = 8
_PAIR_SCHEMA = "hermes.cache_lowhit_pair.v1"
_PAIR_MARKER = ".hermes_cache_lowhit_pair.v1"
_COMPONENTS = ("prefix", "tools", "cache_key_or_scope", "later_history")
_SEGMENTS = {
    "prefix": "system_or_static_prefix",
    "tools": "tools",
    "cache_key_or_scope": "cache_key_or_scope",
    "later_history": "later_history",
}
_SENSITIVE_KEY_PARTS = (
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
_LOCK = threading.Lock()
_BUFFERS: OrderedDict[tuple[str, str, str, str], deque[dict[str, Any]]] = OrderedDict()
_MAX_BUFFERED_PACKETS = 256
_SANITIZE_KEY = secrets.token_bytes(32)


def reset_for_tests() -> None:
    """Clear the in-memory sanitized request buffers. Test-only."""
    with _LOCK:
        _BUFFERS.clear()


def remember_sent_request(
    request: dict[str, Any],
    *,
    api_mode: str = "chat_completions",
    route: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    correlation: str | None = None,
    attempt_id: str | None = None,
) -> None:
    """Keep one sanitized packet per route and logical owner."""
    try:
        api_mode = _identity(api_mode)
        route_key = (
            _identity(route or api_mode),
            _identity(provider),
            _identity(model or request.get("model")),
            api_mode,
        )
        packet = _sanitize_packet(
            request,
            api_mode=api_mode,
            route=route_key[0],
            provider=route_key[1],
            model=route_key[2],
        )
        owner = correlation.strip() if isinstance(correlation, str) and correlation.strip() else None
        physical = attempt_id.strip() if isinstance(attempt_id, str) and attempt_id.strip() else None
        snapshot = {
            "_terminal": False,
            "_abandoned": False,
            "_route_key": route_key,
            "_correlation": owner,
            "_attempt_id": physical,
            **packet,
        }
        with _LOCK:
            history = _BUFFERS.setdefault(route_key, deque())
            if owner is not None:
                for prior in history:
                    if prior.get("_correlation") == owner and not prior["_terminal"]:
                        prior["_abandoned"] = True
            history.append(snapshot)
            while len(history) > _MAX_BUFFERED_PACKETS:
                history.popleft()
            _BUFFERS.move_to_end(route_key)
    except Exception:
        # Request capture is diagnostic-only and must not affect the provider call.
        return


def _sanitize_packet(
    request: dict[str, Any], *, api_mode: str, route: str, provider: str, model: str
) -> dict[str, Any]:
    components = {
        "prefix": _sanitize_value(_prefix(request), _SANITIZE_KEY),
        "tools": _sanitize_value(request.get("tools") or request.get("toolConfig") or [], _SANITIZE_KEY),
        "cache_key_or_scope": _sanitize_value(_cache_key(request), _SANITIZE_KEY),
        "later_history": _sanitize_value(_later_history(request, api_mode), _SANITIZE_KEY),
    }
    return {
        "route": {
            "route": _label(route, _SANITIZE_KEY),
            "provider": _label(provider, _SANITIZE_KEY),
            "model": _label(model, _SANITIZE_KEY),
            "api_mode": _label(api_mode, _SANITIZE_KEY),
        },
        "structure": _structure(request, components, api_mode),
        "components": {
            name: {
                "digest": _digest(_SANITIZE_KEY, f"cache_lowhit.{name}", value),
                "bytes": len(_serialized(value)),
            }
            for name, value in components.items()
        },
        "usage": _empty_usage(),
        "log_lines": [],
    }


def _structure(request: dict[str, Any], components: dict[str, Any], api_mode: str) -> dict[str, Any]:
    prefix = components["prefix"]
    tools = components["tools"]
    return {
        "keys": sorted(
            str(key)
            for key in request
            if _safe_key(key)
        ),
        "prefix_keys": sorted(_shape_keys(prefix)),
        "roles": sorted(_roles(prefix)),
        "cache_control": _cache_controls(request),
        "tool_names": _tool_names(tools),
        "parameter_names": _parameter_names(tools),
        "has_cache_key_or_scope": components["cache_key_or_scope"] is not None,
        "api_mode": _label(api_mode, _SANITIZE_KEY),
    }


def _safe_key(key: Any) -> bool:
    text = str(key).lower()
    return (
        not text.startswith("_hermes")
        and text not in {"messages", "input", "tools", "toolconfig", "instructions", "system"}
        and not any(part in text for part in _SENSITIVE_KEY_PARTS)
    )


def _identity(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _shape_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value if _safe_key(key)}
    if isinstance(value, list):
        found: set[str] = set()
        for child in value:
            found.update(_shape_keys(child))
        return found
    return set()


def _roles(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {str(value["role"]) for _ in (value,) if value.get("role") in {"system", "developer"}}
        for child in value.values():
            found.update(_roles(child))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for child in value:
            found.update(_roles(child))
        return found
    return set()


def _cache_controls(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() == "cache_control" and isinstance(child, dict):
                found.append(
                    {
                        str(name): _scalar(child[name])
                        for name in ("type", "ttl", "ephemeral")
                        if name in child
                    }
                )
            else:
                found.extend(_cache_controls(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_cache_controls(child))
    return found[:32]


def _tool_names(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        return []
    names: list[str] = []
    for tool in tools[:32]:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = tool.get("name")
        if name is None and isinstance(function, dict):
            name = function.get("name")
        if name is not None:
            names.append(_label(name, _SANITIZE_KEY))
    return names


def _parameter_names(tools: Any) -> list[str]:
    names: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                names.update(_label(name, _SANITIZE_KEY) for name in list(properties)[:64])
                for child in properties.values():
                    visit(child)
            for key, child in value.items():
                if key != "properties":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(tools)
    return sorted(names)[:64]


def _scalar(value: Any) -> Any:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return _label(value, _SANITIZE_KEY)


def _empty_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "request_count": 0,
        "cache_telemetry": "unavailable",
    }


def _usage_payload(usage: CanonicalUsage) -> dict[str, Any]:
    return {
        "prompt_tokens": max(0, int(usage.prompt_tokens)),
        "input_tokens": max(0, int(usage.input_tokens)),
        "output_tokens": max(0, int(usage.output_tokens)),
        "cache_read_tokens": max(0, int(usage.cache_read_tokens)),
        "cache_write_tokens": max(0, int(usage.cache_write_tokens)),
        "reasoning_tokens": max(0, int(usage.reasoning_tokens)),
        "request_count": max(0, int(usage.request_count)),
        "cache_telemetry": (
            usage.cache_telemetry if usage.cache_telemetry in {"reported", "unavailable"} else "unavailable"
        ),
    }


def _log_line(packet: dict[str, Any]) -> str:
    usage = packet["usage"]
    route = packet["route"]
    return (
        "cache_lowhit route={route} provider={provider} model={model} api_mode={api_mode} "
        "prompt_tokens={prompt} input_tokens={input} output_tokens={output} "
        "cache_read_tokens={read} cache_write_tokens={write} reasoning_tokens={reasoning} "
        "request_count={count} cache_telemetry={telemetry}"
    ).format(
        route=route["route"],
        provider=route["provider"],
        model=route["model"],
        api_mode=route["api_mode"],
        prompt=usage["prompt_tokens"],
        input=usage["input_tokens"],
        output=usage["output_tokens"],
        read=usage["cache_read_tokens"],
        write=usage["cache_write_tokens"],
        reasoning=usage["reasoning_tokens"],
        count=usage["request_count"],
        telemetry=usage["cache_telemetry"],
    )


def _is_near_zero(usage: CanonicalUsage) -> bool:
    if usage.cache_telemetry != "reported" or usage.prompt_tokens <= 0:
        return False
    return 100 * max(0, usage.cache_read_tokens) < 95 * usage.prompt_tokens


def maybe_dump_on_usage(
    usage: CanonicalUsage,
    *,
    route: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_mode: str | None = None,
    correlation: str | None = None,
    attempt_id: str | None = None,
) -> None:
    """Attach terminal scalar usage and persist a sanitized same-owner pair."""
    try:
        if not _complete_identity(route, provider, model, api_mode):
            return
        with _LOCK:
            key = _select_key(route=route, provider=provider, model=model, api_mode=api_mode)
            if key is None:
                return
            history = _BUFFERS[key]
            current = _select_packet(history, correlation=correlation, attempt_id=attempt_id)
            if current is None or current["_terminal"] or current.get("_abandoned"):
                return
            current["usage"] = _usage_payload(usage)
            current["log_lines"] = [_log_line(current)]
            current["_terminal"] = True
            previous = next(
                (
                    packet
                    for packet in reversed(history)
                    if packet is not current
                    and packet.get("_correlation") == current.get("_correlation")
                    and packet.get("_route_key") == key
                    and packet["_terminal"]
                    and not packet.get("_abandoned")
                ),
                None,
            )
            pair = (
                _public_packet(previous),
                _public_packet(current),
            ) if previous is not None else None
            history_copy = [
                packet for packet in history if not packet["_terminal"] or packet is current
            ]
            history.clear()
            history.extend(history_copy)
        if pair is not None and _is_near_zero(usage):
            _persist_pair(pair)
    except Exception:
        # Persistence and diagnostic bookkeeping are best-effort by contract.
        return


def _select_packet(
    history: deque[dict[str, Any]], *, correlation: str | None, attempt_id: str | None
) -> dict[str, Any] | None:
    owner = correlation.strip() if isinstance(correlation, str) and correlation.strip() else None
    physical = attempt_id.strip() if isinstance(attempt_id, str) and attempt_id.strip() else None
    for packet in reversed(history):
        if packet["_terminal"] or packet.get("_abandoned"):
            continue
        if owner is not None and packet.get("_correlation") != owner:
            continue
        if physical is not None and packet.get("_attempt_id") != physical:
            continue
        return packet
    return None


def _complete_identity(*values: Any) -> bool:
    return all(
        isinstance(value, str)
        and bool(value.strip())
        and _identity(value) != "unknown"
        for value in values
    )


def _select_key(
    *, route: str | None, provider: str | None, model: str | None, api_mode: str | None
) -> tuple[str, str, str, str] | None:
    candidates = reversed(_BUFFERS)
    for key in candidates:
        if route is not None and key[0] != _identity(route):
            continue
        if provider is not None and key[1] != _identity(provider):
            continue
        if model is not None and key[2] != _identity(model):
            continue
        if api_mode is not None and key[3] != _identity(api_mode):
            continue
        return key
    return None


def _public_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": packet["route"],
        "structure": packet["structure"],
        "components": packet["components"],
        "usage": packet["usage"],
        "log_lines": packet["log_lines"] or [_log_line(packet)],
    }


def _persist_pair(pair: tuple[dict[str, Any], dict[str, Any]]) -> None:
    previous, current = pair
    equal = {
        name: previous.get("components", {}).get(name, {}).get("digest")
        == current.get("components", {}).get(name, {}).get("digest")
        for name in _COMPONENTS
    }
    payload = {
        "schema": _PAIR_SCHEMA,
        "timestamp_ns": time.time_ns(),
        "route": current.get("route", {}),
        "requests": [previous, current],
        "comparison": {
            "equal": equal,
            "first_differing_segment": next(
                (_SEGMENTS[name] for name in _COMPONENTS if not equal[name]), "none"
            ),
        },
        "log_lines": [*previous.get("log_lines", []), *current.get("log_lines", [])],
    }
    root = get_hermes_home() / "observability" / "cache_lowhit"
    _ensure_private_dir(root.parent)
    _ensure_private_dir(root)
    staging = _new_pair_dir(root, prefix=".staging-")
    try:
        _exclusive_write(staging / "pair.json", payload)
        _exclusive_write(
            staging / "log_lines.jsonl",
            "".join(
                json.dumps({"line": line}, separators=(",", ":")) + "\n"
                for line in payload["log_lines"]
            ),
        )
        _exclusive_write(staging / _PAIR_MARKER, _PAIR_SCHEMA + "\n")
        final = root / f"pair-{time.time_ns()}-{secrets.token_hex(4)}"
        os.rename(staging, final)
        _fsync_dir(root)
        staging = None
        _retain(root)
    finally:
        if staging is not None:
            _remove_dir(staging)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_dir(path: Path) -> None:
    try:
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
        path.rmdir()
    except OSError:
        return


def _ensure_private_dir(path: Path) -> None:
    fd = _open_private_dir_chain(path)
    os.close(fd)


def _new_pair_dir(root: Path, *, prefix: str = "pair-") -> Path:
    for _ in range(8):
        path = root / f"{prefix}{time.time_ns()}-{secrets.token_hex(4)}"
        try:
            path.mkdir(mode=0o700)
            return path
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate unique cache low-hit pair directory")


def _exclusive_write(path: Path, value: Any) -> None:
    if isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _owned_sealed_pair(path: Path) -> bool:
    try:
        info = path.stat(follow_symlinks=False)
        euid = _posix_euid()
        if not stat.S_ISDIR(info.st_mode) or (euid is not None and info.st_uid != euid):
            return False
        if stat.S_IMODE(info.st_mode) != 0o700:
            return False
        names = {child.name for child in path.iterdir()}
        if not names.issubset({_PAIR_MARKER, "pair.json", "log_lines.jsonl"}):
            return False
        marker = path / _PAIR_MARKER
        pair = path / "pair.json"
        lines = path / "log_lines.jsonl"
        for child in (marker, pair, lines):
            child_info = child.stat(follow_symlinks=False)
            if not stat.S_ISREG(child_info.st_mode) or (euid is not None and child_info.st_uid != euid):
                return False
        if marker.read_text(encoding="utf-8").strip() != _PAIR_SCHEMA:
            return False
        return json.loads(pair.read_text(encoding="utf-8")).get("schema") == _PAIR_SCHEMA
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _retain(root: Path) -> None:
    try:
        pairs = sorted(
            (
                path
                for path in root.iterdir()
                if path.name.startswith("pair-") and _owned_sealed_pair(path)
            ),
            key=lambda path: path.stat(follow_symlinks=False).st_mtime_ns,
            reverse=True,
        )
        for stale in pairs[MAX_DUMPS:]:
            _remove_dir(stale)
    except OSError:
        return
