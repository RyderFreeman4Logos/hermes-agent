"""Opt-in, local-only, opaque evidence for final Codex Responses attempts."""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import secrets
import stat
import threading
from pathlib import Path
from typing import Any

_MAX_BYTES = 4 * 1024 * 1024
_MAX_HISTORY = 64
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_key_lock = threading.Lock()


def _enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly
        return bool(((load_config_readonly().get("agent") or {}).get("codex_cache_diagnostics") or {}).get("enabled", False))
    except Exception:
        return False


def _paths() -> tuple[Path, Path, Path, Path]:
    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    directory = home / "cache"
    return directory, directory / "codex-cache-prefix.jsonl", directory / "codex-cache-prefix.jsonl.1", directory / "codex-cache-prefix.lock"


def _private_dir(path: Path) -> None:
    for ancestor in (*reversed(path.parents), path):
        if ancestor.is_symlink():
            raise OSError("diagnostic ancestor is unsafe")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError:
            pass
        mode = path.lstat().st_mode
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode) or stat.S_IMODE(mode) != 0o700:
        raise OSError("diagnostic directory is unsafe")


def _key(directory: Path) -> bytes:
    path = directory / "codex-cache-prefix.key"
    with _key_lock:
        lock_fd = os.open(directory / "codex-cache-prefix.lock", os.O_RDWR | os.O_CREAT | _NOFOLLOW, 0o600)
        try:
            if stat.S_IMODE(os.fstat(lock_fd).st_mode) != 0o600:
                raise OSError("diagnostic lock is unsafe")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            return _key_locked(path)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_fd)


def _key_locked(path: Path) -> bytes:
        try:
            fd = os.open(path, os.O_RDONLY | _NOFOLLOW)
        except FileNotFoundError:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o600)
            try:
                value = secrets.token_bytes(32)
                os.write(fd, value)
                os.fchmod(fd, 0o600)
                return value
            finally:
                os.close(fd)
        try:
            mode = os.fstat(fd)
            value = os.read(fd, 64)
            if stat.S_IMODE(mode.st_mode) != 0o600 or len(value) != 32:
                raise OSError("diagnostic key is unsafe")
            return value
        finally:
            os.close(fd)


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


def _component(key: bytes, name: str, value: Any) -> dict[str, Any]:
    encoded = _encoded(value)
    return {"key": name, "hmac": hmac.new(key, encoded, hashlib.sha256).hexdigest(), "bytes": len(encoded)}


def _components(request: dict[str, Any], key: bytes) -> list[dict[str, Any]]:
    body = request.get("extra_body") if isinstance(request.get("extra_body"), dict) else {}
    if not isinstance(body, dict):
        body = {}
    history = body.get("input", request.get("input"))
    if not isinstance(history, list):
        history = []
    tools = body.get("tools", request.get("tools"))
    result = [_component(key, "system", body.get("instructions", request.get("instructions"))), _component(key, "tools", tools if isinstance(tools, list) else []), _component(key, "scope", {"model": request.get("model"), "prompt_cache_key": body.get("prompt_cache_key", request.get("prompt_cache_key"))})]
    for index, item in enumerate(history[:_MAX_HISTORY]):
        result.append(_component(key, f"history:{index}", item))
    if len(history) > _MAX_HISTORY:
        result.append(_component(key, "history:overflow", len(history) - _MAX_HISTORY))
    return result


def _attr(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _usage(response: Any) -> dict[str, int | None]:
    usage = _attr(response, "usage")
    total, details = _attr(usage, "input_tokens"), _attr(usage, "input_tokens_details")
    cached = _attr(details, "cached_tokens")
    total = total if isinstance(total, int) and total >= 0 else None
    cached = cached if isinstance(cached, int) and cached >= 0 else None
    return {"cache_read": cached, "uncached_input": total - cached if total is not None and cached is not None and cached <= total else None}


def _append(row: dict[str, Any]) -> None:
    directory, output, rotated, lock_path = _paths()
    _private_dir(directory)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | _NOFOLLOW, 0o600)
    try:
        if stat.S_IMODE(os.fstat(lock_fd).st_mode) != 0o600:
            raise OSError("diagnostic lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        last_sequence = -1
        for path in (output, rotated):
            try:
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    raise OSError("diagnostic sequence file is unsafe")
                with path.open("rb") as stream:
                    data = stream.read(_MAX_BYTES + 8192)
                    if data:
                        last_sequence = int(json.loads(data.splitlines()[-1])["sequence"])
                        break
            except FileNotFoundError:
                continue
        row["sequence"] = last_sequence + 1
        payload = _encoded(row) + b"\n"
        try:
            info = output.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise OSError("diagnostic output is unsafe")
            size = info.st_size
        except FileNotFoundError:
            size = 0
        if size and size + len(payload) > _MAX_BYTES:
            if rotated.exists() or rotated.is_symlink():
                info = rotated.lstat()
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                    raise OSError("diagnostic rotated output is unsafe")
                rotated.unlink()
            os.replace(output, rotated)
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _NOFOLLOW, 0o600)
        try:
            if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                raise OSError("diagnostic output permissions are unsafe")
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def begin_attempt(request: dict[str, Any], *, session_id: str, turn_id: str, api_id: str, ordinal: int, retry: int, role: str = "unknown", route: str = "codex_responses") -> tuple[bytes, dict[str, Any]] | None:
    if not _enabled():
        return None
    try:
        directory, *_ = _paths()
        _private_dir(directory)
        key = _key(directory)
        correlation = {name: hmac.new(key, str(value).encode(), hashlib.sha256).hexdigest() for name, value in (("session", session_id), ("turn", turn_id), ("api", api_id))}
        metadata = {"correlation": correlation, "ordinal": int(ordinal), "retry": int(retry), "role": role if role in {"primary", "fallback", "delegated"} else "unknown", "route": route if route == "codex_responses" else "unknown", "components": _components(request, key)}
        return key, metadata
    except Exception:
        return None


def finish_attempt(token: tuple[bytes, dict[str, Any]] | None, response: Any = None, error: BaseException | None = None) -> None:
    if token is None:
        return
    try:
        _key_bytes, metadata = token
        row = dict(metadata)
        row["kind"] = "error" if error is not None else "terminal"
        row["usage"] = {"cache_read": None, "uncached_input": None} if error is not None else _usage(response)
        _append(row)
    except Exception:
        return


__all__ = ["begin_attempt", "finish_attempt"]
