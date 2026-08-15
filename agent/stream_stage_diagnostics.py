"""Opt-in, local-only stage timings for streamed provider attempts."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from hermes_constants import get_hermes_home

__all__ = ["Attempt", "enabled", "start_attempt"]

_RECORDS_FILE = "stream_stage_latency.jsonl"
_STAGE_ORDER = ("first_byte", "later_chunk", "callback", "aggregate")
_VALID_STAGES = frozenset(_STAGE_ORDER)
_LOCK = threading.Lock()


@dataclass
class _Stage:
    duration_ms: float = 0.0
    events: int = 0


class Attempt:
    """Content-free timings for one streamed attempt."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started = clock()
        self._previous_chunk: float | None = None
        self._chunks = 0
        self._finished = False
        self._stages = {name: _Stage() for name in _STAGE_ORDER}

    def observe_chunk(self) -> None:
        """Record first-byte or later-chunk timing without accepting payload data."""
        if self._finished:
            return
        try:
            now = self._clock()
            self._chunks += 1
            if self._previous_chunk is None:
                self._record("first_byte", now - self._started)
            else:
                self._record("later_chunk", now - self._previous_chunk)
            self._previous_chunk = now
        except Exception:
            pass

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a local stage; only the fixed stage name is persisted."""
        if name not in _VALID_STAGES or self._finished:
            yield
            return
        try:
            started = self._clock()
        except Exception:
            yield
            return
        try:
            yield
        finally:
            try:
                self._record(name, self._clock() - started)
            except Exception:
                pass

    def finish(self) -> None:
        """Persist the bounded stage summary once, best-effort and locally."""
        if self._finished:
            return
        self._finished = True
        try:
            self._record("aggregate", self._clock() - self._started, self._chunks)
        except Exception:
            pass
        record = {
            "schema": "hermes.stream_stage_latency.v1",
            "stages": [
                {
                    "name": name,
                    "duration_ms": round(max(0.0, stage.duration_ms), 3),
                    "events": stage.events,
                }
                for name in _STAGE_ORDER
                if (stage := self._stages[name]).events
            ],
        }
        try:
            _append(record)
        except Exception:
            pass

    def _record(self, name: str, duration: float, events: int = 1) -> None:
        if name not in _VALID_STAGES:
            return
        stage = self._stages[name]
        stage.duration_ms += max(0.0, duration) * 1000.0
        stage.events += max(0, events)


def enabled() -> bool:
    """Return whether local streamed stage diagnostics were explicitly enabled."""
    try:
        from hermes_cli.config import read_raw_config_readonly

        config = read_raw_config_readonly() or {}
        value = config.get("observability", {}).get("stream_stage_latency", {})
        return isinstance(value, dict) and value.get("enabled") is True
    except Exception:
        return False


def start_attempt() -> Attempt | None:
    """Start an enabled attempt, or return ``None`` with no side effects."""
    if not enabled():
        return None
    try:
        return Attempt()
    except Exception:
        return None


def _root() -> Path:
    return get_hermes_home() / "observability"


def _append(record: dict[str, object]) -> None:
    root = _root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.stat(follow_symlinks=False)
    euid = os.geteuid() if hasattr(os, "geteuid") else None
    if not stat.S_ISDIR(info.st_mode) or (euid is not None and info.st_uid != euid):
        raise PermissionError("unsafe stream stage diagnostics directory")
    if stat.S_IMODE(info.st_mode) != 0o700:
        root.chmod(0o700)

    path = root / _RECORDS_FILE
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        info = os.fstat(fd)
        euid = os.geteuid() if hasattr(os, "geteuid") else None
        if (
            not stat.S_ISREG(info.st_mode)
            or (euid is not None and info.st_uid != euid)
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise PermissionError("unsafe stream stage diagnostics file")
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with _LOCK:
            os.write(fd, payload)
    finally:
        os.close(fd)
