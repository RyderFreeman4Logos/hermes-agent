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

from hermes_constants import get_hermes_home

_MAX_BYTES = 4 * 1024 * 1024
_MAX_HISTORY = 64
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_key_lock = threading.Lock()


def _enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly
        return bool(((load_config_readonly().get("agent") or {}).get("codex_cache_diagnostics") or {}).get("enabled", False))
    except Exception:
        return False


def _paths(home: Path | None = None) -> tuple[Path, Path, Path, Path]:
    root = home if home is not None else get_hermes_home()
    directory = root / "cache"
    return directory, directory / "codex-cache-prefix.jsonl", directory / "codex-cache-prefix.jsonl.1", directory / "codex-cache-prefix.lock"


def _private_dir(path: Path) -> None:
    if not _NOFOLLOW:
        raise OSError("diagnostic directory requires nofollow support")

    home = Path(os.path.abspath(path.parent))
    flags = os.O_RDONLY | _DIRECTORY | _NOFOLLOW
    uid_getter = getattr(os, "getuid", None)

    def validate(fd: int, *, private: bool) -> None:
        info = os.fstat(fd)
        mode = stat.S_IMODE(info.st_mode)
        if not stat.S_ISDIR(info.st_mode):
            raise OSError("diagnostic ancestor is not a directory")
        if private:
            if uid_getter is not None and info.st_uid != uid_getter():
                raise OSError("diagnostic directory has the wrong owner")
            if mode != 0o700:
                raise OSError("diagnostic directory is not private")
        elif mode & 0o022 and not (mode & stat.S_ISVTX):
            raise OSError("diagnostic ancestor is writable")

    def open_child(parent_fd: int, name: str) -> int:
        try:
            return os.open(name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            return os.open(name, flags, dir_fd=parent_fd)

    # ponytail: ancestors above HERMES_HOME are only checked for symlink traversal and writeability;
    # descriptor ownership hardening starts at the configured private root.
    fd = os.open(os.sep, flags)
    try:
        parts = home.parts[1:]
        for index, name in enumerate(parts):
            child_fd = open_child(fd, name)
            os.close(fd)
            fd = child_fd
            validate(fd, private=index == len(parts) - 1)
        if not parts:
            validate(fd, private=True)
        child_fd = open_child(fd, path.name)
        try:
            validate(child_fd, private=True)
        finally:
            os.close(child_fd)
    finally:
        os.close(fd)


def _flock_now(fd: int) -> None:
    # ponytail: one nonblocking try. A busy diagnostic lock is skipped, never waited.
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise OSError("diagnostic lock is busy") from None


def _key(directory: Path) -> bytes:
    if not _key_lock.acquire(blocking=False):
        raise OSError("diagnostic lock is busy")
    try:
        return _key_locked(directory / "codex-cache-prefix.key")
    finally:
        _key_lock.release()


def _key_locked(path: Path) -> bytes:
    try:
        value = _read_private(path, 64)
    except FileNotFoundError:
        fd = _open_private(path, os.O_WRONLY | os.O_CREAT)
        try:
            value = secrets.token_bytes(32)
            os.write(fd, value)
            return value
        finally:
            os.close(fd)
    if len(value) != 32:
        raise OSError("diagnostic key is unsafe")
    return value


def _regular(fd: int) -> os.stat_result:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise OSError("diagnostic file is unsafe")
    uid_getter = getattr(os, "getuid", None)
    if uid_getter is not None and info.st_uid != uid_getter():
        raise OSError("diagnostic file has the wrong owner")
    return info


def _open_private(path: Path, flags: int) -> int:
    """Open a private regular file. Callers must not pass a blocking read of an existing FIFO."""
    if flags & os.O_CREAT:
        try:
            fd = os.open(path, flags | os.O_EXCL | _NOFOLLOW, 0o600)
        except FileExistsError:
            fd = os.open(path, (flags & ~os.O_CREAT) | os.O_NONBLOCK | _NOFOLLOW)
    else:
        fd = os.open(path, flags | os.O_NONBLOCK | _NOFOLLOW)
    try:
        _regular(fd)
        if flags & os.O_NONBLOCK == 0:
            os.set_blocking(fd, True)
        return fd
    except Exception:
        os.close(fd)
        raise


def _read_private(path: Path, limit: int) -> bytes:
    fd = _open_private(path, os.O_RDONLY)
    try:
        return os.read(fd, limit)
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
        result.append(_component(key, "history:overflow", history[_MAX_HISTORY:]))
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


def _append(row: dict[str, Any], *, home: Path | None = None) -> None:
    directory, output, rotated, lock_path = _paths(home)
    _private_dir(directory)
    if not _key_lock.acquire(blocking=False):
        return
    lock_fd = _open_private(lock_path, os.O_RDWR | os.O_CREAT)
    try:
        _flock_now(lock_fd)
        last_sequence = -1
        for path in (output, rotated):
            try:
                data = _read_private(path, _MAX_BYTES + 8192)
            except FileNotFoundError:
                continue
            if data:
                last_sequence = int(json.loads(data.splitlines()[-1])["sequence"])
                break
        row["sequence"] = last_sequence + 1
        payload = _encoded(row) + b"\n"
        try:
            size_fd = _open_private(output, os.O_RDONLY)
        except FileNotFoundError:
            size = 0
        else:
            try:
                size = _regular(size_fd).st_size
            finally:
                os.close(size_fd)
        if size and size + len(payload) > _MAX_BYTES:
            try:
                _read_private(rotated, 0)
                rotated.unlink()
            except FileNotFoundError:
                pass
            os.replace(output, rotated)
        fd = _open_private(output, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)
        _key_lock.release()


def begin_attempt(request: dict[str, Any], *, session_id: str, turn_id: str, api_id: str, ordinal: int, retry: int, role: str = "unknown", route: str = "codex_responses") -> tuple[bytes, dict[str, Any]] | None:
    if not _enabled():
        return None
    try:
        directory, *_ = _paths()
        _private_dir(directory)
        key = _key(directory)
        correlation = {name: hmac.new(key, str(value).encode(), hashlib.sha256).hexdigest() for name, value in (("session", session_id), ("turn", turn_id), ("api", api_id))}
        metadata = {"correlation": correlation, "ordinal": int(ordinal), "retry": int(retry), "role": role if role in {"primary", "fallback", "delegated"} else "unknown", "route": route if route == "codex_responses" else "unknown", "components": _components(request, key), "home": str(directory.parent)}
        return key, metadata
    except Exception:
        return None


def finish_attempt(token: tuple[bytes, dict[str, Any]] | None, response: Any = None, error: BaseException | None = None) -> None:
    if token is None:
        return
    try:
        _key_bytes, metadata = token
        row = dict(metadata)
        home = Path(str(row.pop("home")))
        row["kind"] = "error" if error is not None else "terminal"
        row["usage"] = {"cache_read": None, "uncached_input": None} if error is not None else _usage(response)
        _append(row, home=home)
    except Exception:
        return


__all__ = ["begin_attempt", "finish_attempt"]
