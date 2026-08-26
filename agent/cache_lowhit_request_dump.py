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
) -> None:
    """Keep one sanitized packet per route; never retain request bodies."""
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
        snapshot = {"_terminal": False, "_route_key": route_key, **packet}
        with _LOCK:
            history = _BUFFERS.setdefault(route_key, deque(maxlen=2))
            history.append(snapshot)
            _BUFFERS.move_to_end(route_key)
    except Exception:
        # Request capture is diagnostic-only and must not affect the provider call.
        return


def _sanitize_packet(
    request: dict[str, Any], *, api_mode: str, route: str, provider: str, model: str
) -> dict[str, Any]:
    components = {
        "prefix": _prefix(request),
        "tools": request.get("tools") or request.get("toolConfig") or [],
        "cache_key_or_scope": _cache_key(request),
        "later_history": _later_history(request, api_mode),
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
) -> None:
    """Attach terminal scalar usage and persist a sanitized same-route pair."""
    try:
        if not _complete_identity(route, provider, model, api_mode):
            return
        with _LOCK:
            key = _select_key(route=route, provider=provider, model=model, api_mode=api_mode)
            if key is None:
                return
            history = _BUFFERS[key]
            current = history[-1]
            if current["_terminal"]:
                return
            current["usage"] = _usage_payload(usage)
            current["log_lines"] = [_log_line(current)]
            current["_terminal"] = True
            previous = history[-2] if len(history) == 2 else None
            pair = (
                _public_packet(previous),
                _public_packet(current),
            ) if previous is not None and previous["_terminal"] else None
            history.clear()
            history.append(current)
        if pair is not None and _is_near_zero(usage):
            _persist_pair(pair)
    except Exception:
        # Persistence and diagnostic bookkeeping are best-effort by contract.
        return


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
        name: previous["components"][name]["digest"] == current["components"][name]["digest"]
        for name in _COMPONENTS
    }
    payload = {
        "schema": _PAIR_SCHEMA,
        "timestamp_ns": time.time_ns(),
        "route": current["route"],
        "requests": [previous, current],
        "comparison": {
            "equal": equal,
            "first_differing_segment": next(
                (_SEGMENTS[name] for name in _COMPONENTS if not equal[name]), "none"
            ),
        },
        "log_lines": [*previous["log_lines"], *current["log_lines"]],
    }
    root = get_hermes_home() / "observability" / "cache_lowhit"
    _ensure_private_dir(root.parent)
    _ensure_private_dir(root)
    pair_dir = _new_pair_dir(root)
    _exclusive_write(pair_dir / "pair.json", payload)
    _exclusive_write(
        pair_dir / "log_lines.jsonl",
        "".join(json.dumps({"line": line}, separators=(",", ":")) + "\n" for line in payload["log_lines"]),
    )
    _retain(root)


def _ensure_private_dir(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
    except FileExistsError:
        pass
    info = path.stat(follow_symlinks=False)
    euid = getattr(os, "geteuid", lambda: None)()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PermissionError(f"unsafe cache low-hit directory: {path}")
    if euid is not None and info.st_uid != euid:
        raise PermissionError(f"foreign cache low-hit directory: {path}")
    if stat.S_IMODE(info.st_mode) != 0o700:
        path.chmod(0o700)


def _new_pair_dir(root: Path) -> Path:
    for _ in range(8):
        path = root / f"{time.time_ns()}-{secrets.token_hex(4)}"
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


def _retain(root: Path) -> None:
    pairs = []
    for path in root.iterdir():
        try:
            info = path.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                children = list(path.iterdir())
                if children and all(
                    child.name in {"pair.json", "log_lines.jsonl"}
                    and stat.S_ISREG(child.stat(follow_symlinks=False).st_mode)
                    for child in children
                ):
                    pairs.append((info.st_mtime_ns, path))
        except OSError:
            continue
    for _, stale in sorted(pairs)[:-MAX_DUMPS]:
        for child in stale.iterdir():
            if child.is_file() and not child.is_symlink():
                child.unlink(missing_ok=True)
        stale.rmdir()
