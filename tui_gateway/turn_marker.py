"""Durable interrupted-turn markers for the desktop/TUI auto-continue path.

A running turn's progress lives only in process memory (the agent flushes to
SQLite at turn end, not mid-turn), so an app/backend/machine death mid-turn
leaves no durable trace of the interrupted prompt. This sidecar is that
trace: a marker is written when a turn starts running and cleared when the
turn concludes — success, handled error, or interrupt all clear it, so only
a process death leaves one behind. ``session.resume`` reads the marker to
decide whether to auto-continue the interrupted turn (see
``_maybe_schedule_auto_continue`` in ``tui_gateway/server.py``).

Markers are stored per ``HERMES_HOME`` (callers pass the session's home so
profile sessions keep their state in their own profile directory). Ordinary
turn markers are age/count bounded. Completion-delivery markers are durable
backlog receipts and are never pruned until their exact turn retires them.

Marker I/O stays exception-safe. Ordinary turns may ignore a failed write;
completion consumers use the boolean result to withhold acknowledgement rather
than lose an event.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MARKER_DIR = "desktop"
_MARKER_FILE = "interrupted_turns.json"
_MAX_AGE_SECS = 24 * 3600
_MAX_ENTRIES = 32
# Enough to re-submit any realistic prompt; guards the sidecar against a
# pathological multi-megabyte paste being journaled on every turn.
_MAX_PROMPT_CHARS = 64_000

_lock = threading.Lock()


def _marker_path(home: Path | str) -> Path:
    return Path(home) / _MARKER_DIR / _MARKER_FILE


def _load(path: Path) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("unreadable turn-marker file %s; starting fresh", path, exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _prune(entries: dict[str, dict], now: float) -> dict[str, dict]:
    protected = {
        key: entry
        for key, entry in entries.items()
        if isinstance(entry.get("completion_delivery_id"), str)
        and entry["completion_delivery_id"]
    }
    fresh = {
        key: entry
        for key, entry in entries.items()
        if key not in protected
        if now - float(entry.get("started_at") or 0) <= _MAX_AGE_SECS
    }
    if len(fresh) > _MAX_ENTRIES:
        fresh = dict(
            sorted(
                fresh.items(),
                key=lambda item: float(item[1].get("started_at") or 0),
                reverse=True,
            )[:_MAX_ENTRIES]
        )
    return {**fresh, **protected}


def _store(path: Path, entries: dict[str, dict]) -> None:
    if not entries:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".turn-marker-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_turn_start(
    home: Path | str,
    session_key: str,
    prompt: str,
    *,
    attempts: int = 0,
    completion_delivery_id: str | None = None,
) -> bool:
    """Persist the marker for a turn that is about to run.

    ``attempts`` counts how many auto-continues led to this run: 0 for a
    user-initiated turn, N for the Nth automatic re-run — the crash-loop
    breaker reads it back on the next resume.
    """
    if not session_key or not prompt:
        return False
    now = time.time()
    entry = {
        "attempts": max(0, int(attempts)),
        "prompt": prompt[:_MAX_PROMPT_CHARS],
        "started_at": now,
    }
    if completion_delivery_id:
        entry["completion_delivery_id"] = completion_delivery_id
    try:
        with _lock:
            path = _marker_path(home)
            entries = _prune(_load(path), now)
            existing_delivery_id = str(
                (entries.get(session_key) or {}).get("completion_delivery_id") or ""
            )
            if existing_delivery_id and existing_delivery_id != completion_delivery_id:
                return False
            entries[session_key] = entry
            _store(path, entries)
        return True
    except Exception:
        logger.debug("failed to record turn marker for %s", session_key, exc_info=True)
        return False


def clear_turn_marker(home: Path | str, session_key: str) -> None:
    """Remove the marker once its turn concluded (any outcome the client saw)."""
    if not session_key:
        return
    try:
        with _lock:
            path = _marker_path(home)
            entries = _load(path)
            if session_key not in entries:
                return
            del entries[session_key]
            _store(path, entries)
    except Exception:
        logger.debug("failed to clear turn marker for %s", session_key, exc_info=True)


def read_turn_marker(home: Path | str, session_key: str) -> dict[str, Any] | None:
    """The marker left by a turn that never concluded, or None."""
    if not session_key:
        return None
    try:
        with _lock:
            entry = _load(_marker_path(home)).get(session_key)
    except Exception:
        return None
    if not isinstance(entry, dict):
        return None
    prompt = str(entry.get("prompt") or "")
    if not prompt.strip():
        return None
    try:
        started_at = float(entry.get("started_at") or 0)
        attempts = max(0, int(entry.get("attempts") or 0))
    except (TypeError, ValueError):
        return None
    result = {"attempts": attempts, "prompt": prompt, "started_at": started_at}
    completion_delivery_id = entry.get("completion_delivery_id")
    if isinstance(completion_delivery_id, str) and completion_delivery_id:
        result["completion_delivery_id"] = completion_delivery_id
    return result
