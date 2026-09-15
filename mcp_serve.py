"""
Hermes MCP Server — expose messaging conversations as MCP tools (`hermes mcp serve`).

A stdio MCP server letting any MCP client (Claude Code, Cursor, Codex, ...) list
conversations, read history, send messages, poll live events, and manage approvals.
Matches OpenClaw's 9-tool channel bridge surface plus the Hermes-specific
channels_list. Client config: {"mcpServers": {"hermes": {"command": "hermes", "args": ["mcp", "serve"]}}}
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("hermes.mcp_serve")

# mcp 2.0 removed `mcp.server.fastmcp`; `mcp.server.MCPServer` keeps the same
# `@server.tool()` / `run_stdio_async()` surface (docstring -> description,
# signature -> input schema).
_MCP_SERVER_AVAILABLE = False
try:
    from mcp.server import MCPServer

    _MCP_SERVER_AVAILABLE = True
except ImportError:
    MCPServer = None  # type: ignore[assignment,misc]


# --- Helpers -----------------------------------------------------------------

def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _get_sessions_dir() -> Path:
    return _hermes_home() / "sessions"


def _read_json(path: Path):
    """Parsed JSON file, or {} when missing/unreadable."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to load %s: %s", path.name, e)
        return {}


def _close_quietly(db, what: str) -> None:
    try:
        db.close()
    except Exception:
        logger.debug("Failed to close MCP %s SessionDB", what, exc_info=True)


def _get_session_db():
    """SessionDB instance for reading message transcripts, or None."""
    try:
        from hermes_state_registry import acquire
        return acquire()
    except Exception as e:
        logger.debug("SessionDB unavailable: %s", e)
        return None


def _load_session_messages(session_id: str):
    """(messages, error) for one session; closes the temporary database handle."""
    db = _get_session_db()
    if db is None:
        return None, "Session database unavailable"
    try:
        return db.get_messages(session_id), None
    except Exception as e:
        return None, f"Failed to read messages: {e}"
    finally:
        try:
            from hermes_state_registry import release_or_close
            release_or_close(db)
        except Exception:
            logger.debug("Failed to close MCP SessionDB", exc_info=True)


def _load_sessions_index() -> dict:
    """Gateway routing index: session_key -> entry dict.

    state.db is primary (gateway session rows carry session_key/origin metadata);
    sessions.json is the fallback for pre-migration databases without session_keys.

    state.db is the primary source (#9006): gateway sessions persist their routing metadata (session_key,
    chat/thread ids, display_name, origin) on the durable session row, so a single database read replaces
    the old dual-file sessions.json dependency.
    """
    return _load_sessions_index_from_db() or _load_sessions_index_from_json()


def _load_sessions_index_strict() -> dict:
    """Read the EventBridge index without converting a failed read to empty.

    The ordinary MCP listing tools keep their tolerant loaders. Event delivery
    needs to retain a pending database change when either authoritative index
    read fails, so only this poller path exposes an error to its caller.
    """
    db = _get_session_db()
    if db is None:
        raise RuntimeError("Session database unavailable for EventBridge index")
    try:
        lister = getattr(db, "list_gateway_sessions", None)
        if not callable(lister):
            # Legacy/test databases without routing rows retain the JSON
            # fallback; a callable loader that fails is never treated as empty.
            return _load_sessions_index_from_json_strict()
        entries = {
            row["session_key"]: _row_to_index_entry(row)
            for row in lister(active_only=True)
            if row.get("session_key")
        }
    finally:
        _close_quietly(db, "EventBridge index")
    return entries or _load_sessions_index_from_json_strict()


def _load_sessions_index_from_json_strict() -> dict:
    """Read the legacy index, preserving malformed/read-error outcomes."""
    path = _get_sessions_dir() / "sessions.json"
    try:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("Failed to read EventBridge legacy index") from exc
    if not isinstance(data, dict):
        raise RuntimeError("EventBridge legacy index is not an object")
    return {key: value for key, value in data.items() if not str(key).startswith("_")}


def _iso(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).isoformat() if ts else ""
    except (TypeError, ValueError, OSError):
        return ""


def _row_to_index_entry(row: dict) -> dict:
    """Convert a state.db gateway session row to the sessions.json entry shape."""
    origin = {}
    if row.get("origin_json"):
        try:
            parsed = json.loads(row["origin_json"])
            if isinstance(parsed, dict):
                origin = parsed
        except (TypeError, ValueError):
            pass
    if not origin:  # pre-origin_json rows: synthesize the minimal origin from columns
        origin = {"platform": row.get("source", ""), **{k: row.get(k) for k in ("chat_id", "chat_type", "thread_id", "user_id")}}

    input_tokens = int(row.get("input_tokens") or 0)
    output_tokens = int(row.get("output_tokens") or 0)
    return {
        "session_id": str(row.get("id", "")), "session_key": row.get("session_key", ""),
        "platform": row.get("source", ""),
        "chat_type": row.get("chat_type") or origin.get("chat_type", ""),
        "display_name": row.get("display_name") or origin.get("chat_name") or "",
        "origin": origin,
        "created_at": _iso(row.get("started_at")),
        "updated_at": _iso(row.get("last_active") or row.get("started_at")),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _load_sessions_index_from_db() -> dict:
    """Build the routing index from state.db gateway session rows."""
    db = _get_session_db()
    if db is None:
        return {}
    try:
        lister = getattr(db, "list_gateway_sessions", None)
        if not callable(lister):
            return {}
        return {row["session_key"]: _row_to_index_entry(row) for row in lister(active_only=True) if row.get("session_key")}
    except Exception as e:
        logger.debug("Failed to load gateway sessions from state.db: %s", e)
        return {}
    finally:
        try:
            db.close()
        except Exception:
            pass


def _load_sessions_index_from_json() -> dict:
    """Legacy fallback: read sessions.json directly (avoids importing SessionStore,
    which needs GatewayConfig). Keys starting with "_" are metadata sentinels
    (e.g. "_README"), not session entries."""
    data = _read_json(_get_sessions_dir() / "sessions.json")
    return {k: v for k, v in data.items() if not str(k).startswith("_")} if isinstance(data, dict) else {}


def _load_channel_directory() -> dict:
    """Load the cached channel directory for available targets."""
    return _read_json(_hermes_home() / "channel_directory.json")


def _coerce_int(value, *, default: int, minimum: int, maximum: int) -> int:
    """Clamped int for MCP tool boundaries; *default* when the client sent an unconvertible value."""
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(coerced, maximum))


def _extract_message_content(msg: dict) -> str:
    """Extract text content from a message, handling multi-part content."""
    content = msg.get("content", "")
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content) if content else ""


def _extract_attachments(msg: dict) -> List[dict]:
    """Non-text attachments: image/file content blocks plus MEDIA: tags in the text."""
    attachments = []
    content = msg.get("content", "")
    for part in content if isinstance(content, list) else ():
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype == "image_url":
            url = part.get("image_url", {}).get("url", "") if isinstance(part.get("image_url"), dict) else ""
        elif ptype == "image":
            url = part.get("url", part.get("source", {}).get("url", ""))
        else:
            if ptype != "text":
                attachments.append({"type": ptype, "data": part})
            continue
        if url:
            attachments.append({"type": "image", "url": url})
    for match in re.finditer(r'MEDIA:\s*(\S+)', _extract_message_content(msg)):
        attachments.append({"type": "media", "path": match.group(1)})
    return attachments


# --- Event Bridge — polls SessionDB for new messages, maintains event queue ---

QUEUE_LIMIT = 1000
POLL_INTERVAL = 0.2  # seconds between DB polls (200ms)


@dataclass
class QueueEvent:
    """An event in the bridge's in-memory queue."""
    cursor: int
    type: str  # "message", "approval_requested", "approval_resolved"
    session_key: str = ""
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"cursor": self.cursor, "type": self.type, "session_key": self.session_key, **self.data}


def _ts_float(ts) -> float:
    """Normalize a message timestamp (epoch int/float or ISO string) to float."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if not (isinstance(ts, str) and ts):
        return 0.0
    try:
        return float(ts)
    except ValueError:
        try:
            return datetime.fromisoformat(ts).timestamp()
        except Exception:
            return 0.0


def _latest_ts(messages) -> float:
    """Newest normalized timestamp among *messages* (0.0 when none)."""
    return max((_ts_float(m.get("timestamp", 0)) for m in (messages or ())), default=0.0)


class EventBridge:
    """Background poller watching SessionDB for new messages, feeding an in-memory
    event queue with waiter support (the Hermes analogue of OpenClaw's WebSocket
    gateway bridge, polling SQLite instead)."""

    def __init__(self):
        self._queue: List[QueueEvent] = []
        self._cursor = 0
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._new_event = threading.Event()
        self._running = False
        self._starting = False
        self._stop_requested = False
        self._thread: Optional[threading.Thread] = None
        self._last_poll_timestamps: Dict[str, float] = {}  # session_key -> unix timestamp
        # A body-read failure after baseline has observed a session must not
        # turn its existing history into a later event on recovery.
        self._baseline_cutoffs: Dict[str, int] = {}
        self._pending_approvals: Dict[str, dict] = {}  # populated from events
        self._state_db_mtime: float = 0.0  # skip polling work when state.db is unchanged
        # PRAGMA data_version watermark, sampled on the long-lived read-only
        # connection below. Only comparable across samples from that ONE
        # connection, so it is reset whenever the connection is reopened.
        self._state_db_version: Optional[int] = None
        self._state_watch_conn: Optional[sqlite3.Connection] = None
        self._state_watch_identity: Optional[tuple[int, int]] = None
        self._cached_sessions_index: dict = {}

    def start(self):
        """Start one polling owner after a successful baseline reservation."""
        with self._lifecycle_lock:
            if self._starting or self._running or (
                self._thread is not None and self._thread.is_alive()
            ):
                return False
            self._starting = True
            self._stop_requested = False
        try:
            if not self._establish_baseline():
                self._close_state_watch_conn()
                return False
            with self._lifecycle_lock:
                if self._stop_requested:
                    self._close_state_watch_conn()
                    return False
                thread = threading.Thread(target=self._poll_loop, daemon=True)
                self._thread = thread
                self._running = True
                try:
                    thread.start()
                except Exception:
                    self._running = False
                    self._thread = None
                    self._close_state_watch_conn()
                    raise
            logger.debug("EventBridge started")
            return True
        finally:
            with self._lifecycle_lock:
                self._starting = False

    def stop(self):
        """Request the current owner stop without replacing a live owner."""
        idle_watcher = None
        with self._lifecycle_lock:
            self._stop_requested = True
            self._running = False
            thread = self._thread
            starting = self._starting
            if thread is None and not starting:
                # Detach the exact residue owned by this idle decision while
                # admission is still excluded.  A later start may install its
                # own watcher as soon as this lock is released.
                idle_watcher, self._state_watch_conn = self._state_watch_conn, None
                self._state_watch_identity = None
            self._new_event.set()
        if thread is not None:
            thread.join(timeout=5)
        elif idle_watcher is not None:
            try:
                idle_watcher.close()
            except (sqlite3.Error, OSError) as exc:
                logger.debug("EventBridge: closing idle state.db watcher failed: %s", exc)
        if thread is not None and thread.is_alive():
            logger.warning(
                "EventBridge: poll thread still running after stop(); "
                "retaining its state.db watcher until worker cleanup"
            )
        logger.debug("EventBridge stopped")


    def _matching(self, after_cursor: int, session_key: Optional[str], limit: int) -> List[dict]:
        with self._lock:
            return [e.as_dict() for e in self._queue
                    if e.cursor > after_cursor and (not session_key or e.session_key == session_key)][:limit]

    def poll_events(self, after_cursor: int = 0, session_key: Optional[str] = None, limit: int = 20) -> dict:
        """Return events since after_cursor, optionally filtered by session_key."""
        events = self._matching(after_cursor, session_key, limit)
        return {"events": events, "next_cursor": events[-1]["cursor"] if events else after_cursor}

    def wait_for_event(self, after_cursor: int = 0, session_key: Optional[str] = None, timeout_ms: int = 30000) -> Optional[dict]:
        """Block until a matching event arrives or timeout expires."""
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        while time.monotonic() < deadline:
            found = self._matching(after_cursor, session_key, 1)
            if found:
                return found[0]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._new_event.clear()
            self._new_event.wait(timeout=min(remaining, POLL_INTERVAL))
        return None

    def list_pending_approvals(self) -> List[dict]:
        """List approval requests observed during this bridge session."""
        with self._lock:
            return sorted(self._pending_approvals.values(), key=lambda a: a.get("created_at", ""))

    def respond_to_approval(self, approval_id: str, decision: str) -> dict:
        """Resolve a pending approval (best-effort without gateway IPC)."""
        with self._lock:
            approval = self._pending_approvals.pop(approval_id, None)
        if not approval:
            return {"error": f"Approval not found: {approval_id}"}
        self._enqueue(QueueEvent(0, "approval_resolved", approval.get("session_key", ""),  # cursor set by _enqueue
                                 {"approval_id": approval_id, "decision": decision}))
        return {"resolved": True, "approval_id": approval_id, "decision": decision}

    def _enqueue(self, event: QueueEvent) -> None:
        """Add one event through the shared queue/progress commit boundary."""
        self._commit_session_events((event,))

    def _commit_session_events(
        self, events: tuple[QueueEvent, ...] | list[QueueEvent], *,
        session_key: Optional[str] = None, latest: Optional[float] = None,
    ) -> None:
        """Commit one prepared session's events and progress before waking waiters."""
        with self._lock:
            for event in events:
                self._cursor += 1
                event.cursor = self._cursor
                self._queue.append(event)
                while len(self._queue) > QUEUE_LIMIT:
                    self._queue.pop(0)
            if session_key is not None and latest is not None:
                last_seen = self._last_poll_timestamps.get(session_key, 0.0)
                if latest > last_seen:
                    self._last_poll_timestamps[session_key] = latest
        if events:
            # A wake failure occurs after the queue/progress commit. A later
            # changed scan therefore cannot republish this session's events.
            self._new_event.set()

    def _prepare_session_events(
        self, session_key: str, messages: list[dict], last_seen: float,
        baseline_cutoff: Optional[int] = None,
    ) -> tuple[list[QueueEvent], float]:
        """Build a session's events before making any of them visible."""
        events = []
        for msg in messages:
            try:
                message_id = int(msg.get("id", 0) or 0)
            except (TypeError, ValueError):
                message_id = 0
            if (
                msg.get("role", "") not in {"user", "assistant"}
                or _ts_float(msg.get("timestamp", 0)) <= last_seen
                or (baseline_cutoff is not None and message_id <= baseline_cutoff)
            ):
                continue
            content = _extract_message_content(msg)
            if not content:
                continue
            events.append(QueueEvent(0, "message", session_key, {
                "role": msg.get("role", ""), "content": content[:500],
                "timestamp": str(msg.get("timestamp", "")), "message_id": str(msg.get("id", "")),
            }))
        return events, _latest_ts(messages)

    def _establish_baseline(self) -> bool:
        """Record startup history without replaying an incomplete baseline later."""
        # start() admits this baseline only after excluding an older worker.
        # Row IDs belong to that worker's database generation, not this one.
        self._baseline_cutoffs = {}
        db = _get_session_db()
        if not db:
            return False
        db_file = _hermes_home() / "state.db"
        try:
            try:
                entries, watermark = self._refresh_index_and_watermark(db_file)
            except Exception:
                # A failed index read is not an empty index and cannot define a
                # safe startup cohort.
                logger.debug("EventBridge: startup index read failed", exc_info=True)
                return False
            cutoffs = {}
            try:
                for session_key, entry in entries.items():
                    session_id = entry.get("session_id", "")
                    if not session_id:
                        continue
                    getter = getattr(db, "get_active_message_baseline", None)
                    if callable(getter):
                        cutoff, latest = getter(session_id)
                        cutoffs[session_key] = cutoff
                        if latest > 0.0:
                            self._last_poll_timestamps[session_key] = latest
                    else:
                        getter = getattr(db, "get_active_message_watermark", None)
                        cutoffs[session_key] = getter(session_id) if callable(getter) else None
            except Exception:
                # Never invent a zero cutoff when the real active-row query
                # cannot establish the startup boundary.
                logger.debug("EventBridge: startup message watermark read failed", exc_info=True)
                return False

            reads_succeeded = True
            for session_key, entry in entries.items():
                session_id = entry.get("session_id", "")
                if not session_id:
                    continue
                try:
                    messages = db.get_messages(session_id)
                except Exception:
                    reads_succeeded = False
                    cutoff = cutoffs.get(session_key)
                    if cutoff is not None:
                        self._baseline_cutoffs[session_key] = cutoff
                    continue
                snapshot_ids = [
                    int(message["id"])
                    for message in messages
                    if isinstance(message.get("id"), (int, float))
                ]
                cutoff = cutoffs.get(session_key)
                if cutoff is not None and cutoff > 0 and snapshot_ids and min(snapshot_ids) > cutoff:
                    # A transcript rewrite retired every row from the active
                    # watermark and inserted a carried replacement.  The one
                    # successful body read is its coherent history snapshot.
                    self._baseline_cutoffs[session_key] = max(snapshot_ids)
                elif cutoff is not None:
                    # A normal append after the active-row snapshot remains
                    # pending for the first poll, as before.
                    self._baseline_cutoffs[session_key] = cutoff
                    messages = [
                        message for message in messages
                        if not isinstance(message.get("id"), (int, float))
                        or int(message["id"]) <= cutoff
                    ]
                latest = _latest_ts(messages)
                if latest > self._last_poll_timestamps.get(session_key, 0.0):
                    self._last_poll_timestamps[session_key] = latest
            if reads_succeeded:
                self._state_db_mtime, self._state_db_version = watermark
            return True
        finally:
            _close_quietly(db, "baseline")

    def _refresh_index_and_watermark(self, db_file: Path) -> tuple[dict, tuple[float, Optional[int]]]:
        """Refresh the routing index and return its pending watermark.

        The stored watermark is sampled AFTER an index refresh, because that
        refresh opens SessionDB, whose schema initialisation commits the first
        time it runs against a database. A watermark taken before it would
        record the bridge's own write as a peer's change and force a redundant
        scan on the next tick.

        Only one skew is dangerous: a watermark ahead of the index. A session
        registered after the index query but before the sample would be absent
        from the entries below AND already counted in the watermark, so every
        later poll would take the confirmed-quiet skip and its first message
        would wait for an unrelated commit — the dropped-new-conversation
        shape (#8925) this poller exists to avoid. So when anything at all
        committed while the index was being read, the index is read once more,
        now that the watermark is fixed: that read sees everything committed
        through it, and a commit landing afterwards moves the version again
        and is picked up on the next poll. The opposite skew — index ahead of
        the watermark — only ever costs one redundant scan.
        """
        before = self._sample_state_watermark(db_file)
        entries = _load_sessions_index_strict()
        watermark = self._sample_state_watermark(db_file)
        if watermark != before:
            entries = _load_sessions_index_strict()
        self._cached_sessions_index = entries
        return entries, watermark

    def _close_state_watch_conn(self) -> None:
        """Drop the watcher connection; the next sample reopens it."""
        conn, self._state_watch_conn = self._state_watch_conn, None
        self._state_watch_identity = None
        if conn is None:
            return
        try:
            conn.close()
        except (sqlite3.Error, OSError) as exc:
            logger.debug("EventBridge: closing state.db watcher failed: %s", exc)

    def _sample_state_watermark(self, db_file: Path) -> tuple[float, Optional[int]]:
        """Return the (mtime, data_version) pair state.db shows right now.

        Both halves are taken at the same instant so the gate in _poll_once
        compares like with like on the next tick; a missing file yields the
        (0.0, None) "nothing to watch" pair.
        """
        try:
            db_stat = db_file.stat()
        except OSError:
            self._close_state_watch_conn()
            return 0.0, None
        return db_stat.st_mtime, self._sample_state_db_version(db_file, db_stat)

    def _sample_state_db_version(self, db_file: Path, db_stat) -> Optional[int]:
        """Return state.db's PRAGMA data_version, or None when it can't be read.

        data_version changes whenever ANOTHER connection commits, including WAL
        commits that never touch the main file's mtime, so it answers the
        question mtime cannot: has anything landed since the last sample?

        The counter is per-connection, so samples are only comparable while the
        same connection stays open. A replaced file (different st_dev/st_ino)
        therefore closes the old connection and clears the watermark, which
        makes the caller treat the database as changed.

        None means "unknown" — the file is not a readable SQLite database, the
        open failed, or the read raced a writer. Callers must scan on None
        rather than assume the database is quiet.
        """
        identity = (db_stat.st_dev, db_stat.st_ino)
        if self._state_watch_conn is not None and identity != self._state_watch_identity:
            self._close_state_watch_conn()

        if self._state_watch_conn is None:
            try:
                from hermes_cli.sqlite_safe_read import (
                    UntrackableConnectionError,
                    connect_tracked,
                )
            except ImportError as exc:
                logger.debug("EventBridge: sqlite_safe_read unavailable: %s", exc)
                return None
            try:
                # Tracked, so the byte-probe guard in sqlite_safe_read still
                # holds; read-only and isolation_level=None, so watching takes
                # no lock the gateway's writers could contend with.
                conn = connect_tracked(
                    f"{db_file.resolve().as_uri()}?mode=ro",
                    tracking_path=db_file,
                    uri=True,
                    check_same_thread=False,
                    isolation_level=None,
                    timeout=1.0,
                )
            except (UntrackableConnectionError, sqlite3.Error, OSError) as exc:
                logger.debug("EventBridge: state.db watcher open failed: %s", exc)
                return None
            self._state_watch_conn = conn
            self._state_watch_identity = identity
            self._state_db_version = None

        try:
            row = self._state_watch_conn.execute("PRAGMA data_version").fetchone()
        except (sqlite3.Error, OSError) as exc:
            logger.debug("EventBridge: state.db data_version unreadable: %s", exc)
            self._close_state_watch_conn()
            return None
        return row[0] if row else None

    def _poll_loop(self):
        """Background loop; this worker owns release and watcher cleanup."""
        db = None
        try:
            db = _get_session_db()
            if not db:
                logger.warning("EventBridge: SessionDB unavailable, event polling disabled")
                return
            while self._running:
                try:
                    self._poll_once(db)
                except Exception as e:
                    logger.debug("EventBridge poll error: %s", e)
                time.sleep(POLL_INTERVAL)
        finally:
            if db is not None:
                _close_quietly(db, "polling")
            self._close_state_watch_conn()
            with self._lifecycle_lock:
                if self._thread is threading.current_thread():
                    self._running = False


    def _poll_once(self, db):
        """Check for new messages across all sessions.

        A cheap mtime check on state.db carries the common case — it makes
        200ms polling essentially free. The routing index lives in the same
        file as the messages, so a new conversation and its first message land
        under a single check (no dual-file race that could drop brand-new
        conversations). See #8925, #9006.

        An unchanged mtime is not proof of an unchanged database: filesystem
        timestamps tick on the coarse clock, so a commit landing in the same
        tick as the previous stat leaves the mtime identical, and under WAL a
        commit does not touch the main file at all until checkpoint. So the
        database itself is asked, via PRAGMA data_version, before any poll is
        skipped.
        """
        db_file = _hermes_home() / "state.db"
        try:
            db_stat = db_file.stat()
        except OSError:
            db_stat = None

        db_mtime = db_stat.st_mtime if db_stat is not None else 0.0

        if db_stat is None:
            # Nothing to watch: drop the connection and the watermark so a
            # recreated state.db is sampled from scratch.
            self._close_state_watch_conn()
            self._state_db_version = None
            if db_mtime == self._state_db_mtime:
                return  # Still absent since last poll — skip entirely
        elif db_mtime == self._state_db_mtime:
            version = self._sample_state_db_version(db_file, db_stat)
            # An unreadable version (None) leaves quiescence unconfirmed, so
            # the scan below runs rather than risk dropping events.
            if version is not None and version == self._state_db_version:
                return  # Confirmed quiet since last poll — skip entirely

        # Refresh the index on every change tick: one indexed query, never lags
        # messages. Commit the pending watermark only after every required
        # message read succeeds, so a transient read failure cannot turn an
        # unread commit into confirmed quietness.
        try:
            entries, watermark = self._refresh_index_and_watermark(db_file)
        except Exception:
            # Keep the prior watermark so this change is retried rather than
            # confusing a failed index load with a real empty index.
            logger.debug("EventBridge: poll index read failed", exc_info=True)
            return

        reads_succeeded = True
        for session_key, entry in entries.items():
            session_id = entry.get("session_id", "")
            if not session_id:
                continue
            last_seen = self._last_poll_timestamps.get(session_key, 0.0)
            try:
                messages = db.get_messages(session_id)
                events, latest = self._prepare_session_events(
                    session_key, messages, last_seen,
                    self._baseline_cutoffs.get(session_key),
                )
            except Exception:
                # A failed required read or conversion leaves the complete
                # session pending; an already committed sibling stays intact.
                logger.debug("EventBridge: poll session read failed", exc_info=True)
                reads_succeeded = False
                continue
            self._commit_session_events(
                events, session_key=session_key, latest=latest
            )
            self._baseline_cutoffs.pop(session_key, None)
        if reads_succeeded:
            self._state_db_mtime, self._state_db_version = watermark


# --- MCP Server ---------------------------------------------------------------

def _conversation_messages(session_key: str):
    """(messages, error_json) for a conversation; exactly one is None."""
    entry = _load_sessions_index().get(session_key)
    if not entry:
        return None, json.dumps({"error": f"Conversation not found: {session_key}"})
    session_id = entry.get("session_id", "")
    if not session_id:
        return None, json.dumps({"error": "No session ID for this conversation"})
    messages, error = _load_session_messages(session_id)
    if error:
        return None, json.dumps({"error": error})
    return messages, None


def _platform_matches(wanted: Optional[str], actual: str) -> bool:
    return not wanted or actual.lower() == wanted.lower()


class _ToolHandlers:
    """The MCP tool handlers; each method named in _TOOL_NAMES is registered as one tool.

    Method docstrings are the wire-format tool descriptions and signatures the
    input schemas — do not reword or reflow them.
    """

    def __init__(self, bridge: EventBridge):
        self.bridge = bridge

    def conversations_list(self, platform: Optional[str] = None, limit: int = 50, search: Optional[str] = None) -> str:
        """List active messaging conversations across connected platforms.

        Returns conversations with their session keys (needed for messages_read),
        platform, chat type, display name, and last activity time.

        Args:
            platform: Filter by platform name (telegram, discord, slack, etc.)
            limit: Maximum number of conversations to return (default 50)
            search: Optional text to filter conversations by name
        """
        limit = _coerce_int(limit, default=50, minimum=1, maximum=200)
        conversations = []
        for key, entry in _load_sessions_index().items():
            origin = entry.get("origin", {})
            entry_platform = entry.get("platform") or origin.get("platform", "")
            if not _platform_matches(platform, entry_platform):
                continue
            display_name = entry.get("display_name", "")
            chat_name = origin.get("chat_name", "")
            if search and not any(search.lower() in s.lower() for s in (display_name, chat_name, key)):
                continue
            conversations.append({
                "session_key": key, "session_id": entry.get("session_id", ""), "platform": entry_platform,
                "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
                "display_name": display_name, "chat_name": chat_name,
                "user_name": origin.get("user_name", ""), "updated_at": entry.get("updated_at", ""),
            })

        conversations = sorted(conversations, key=lambda c: c.get("updated_at", ""), reverse=True)[:limit]
        return json.dumps({"count": len(conversations), "conversations": conversations}, indent=2)

    def conversation_get(self, session_key: str) -> str:
        """Get detailed info about one conversation by its session key.

        Args:
            session_key: The session key from conversations_list
        """
        entry = _load_sessions_index().get(session_key)
        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})
        origin = entry.get("origin", {})
        return json.dumps({
            "session_key": session_key, "session_id": entry.get("session_id", ""),
            "platform": entry.get("platform") or origin.get("platform", ""),
            "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
            "display_name": entry.get("display_name", ""),
            "user_name": origin.get("user_name", ""), "chat_name": origin.get("chat_name", ""),
            "chat_id": origin.get("chat_id", ""), "thread_id": origin.get("thread_id"),
            "updated_at": entry.get("updated_at", ""), "created_at": entry.get("created_at", ""),
            "input_tokens": entry.get("input_tokens", 0), "output_tokens": entry.get("output_tokens", 0),
            "total_tokens": entry.get("total_tokens", 0),
        }, indent=2)

    def messages_read(self, session_key: str, limit: int = 50) -> str:
        """Read recent messages from a conversation.

        Returns the message history in chronological order with role, content,
        and timestamp for each message.

        Args:
            session_key: The session key from conversations_list
            limit: Maximum number of messages to return (default 50, most recent)
        """
        limit = _coerce_int(limit, default=50, minimum=1, maximum=200)
        all_messages, error = _conversation_messages(session_key)
        if error:
            return error
        filtered = []
        for msg in all_messages:
            role = msg.get("role", "")
            content = _extract_message_content(msg) if role in {"user", "assistant"} else ""
            if content:
                filtered.append({"id": str(msg.get("id", "")), "role": role,
                                 "content": content[:2000], "timestamp": msg.get("timestamp", "")})
        messages = filtered[-limit:]
        return json.dumps({"session_key": session_key, "count": len(messages),
                           "total_in_session": len(filtered), "messages": messages}, indent=2)

    def attachments_fetch(self, session_key: str, message_id: str) -> str:
        """List non-text attachments for a message in a conversation.

        Extracts images, media files, and other non-text content blocks
        from the specified message.

        Args:
            session_key: The session key from conversations_list
            message_id: The message ID from messages_read
        """
        all_messages, error = _conversation_messages(session_key)
        if error:
            return error
        target_msg = next((m for m in all_messages if str(m.get("id", "")) == message_id), None)
        if not target_msg:
            return json.dumps({"error": f"Message not found: {message_id}"})
        attachments = _extract_attachments(target_msg)
        return json.dumps({"message_id": message_id, "count": len(attachments), "attachments": attachments}, indent=2)

    def events_poll(self, after_cursor: int = 0, session_key: Optional[str] = None, limit: int = 20) -> str:
        """Poll for new conversation events since a cursor position.

        Returns events that have occurred since the given cursor. Use the
        returned next_cursor value for subsequent polls.

        Event types: message, approval_requested, approval_resolved

        Args:
            after_cursor: Return events after this cursor (0 for all)
            session_key: Optional filter to one conversation
            limit: Maximum events to return (default 20)
        """
        after_cursor = _coerce_int(after_cursor, default=0, minimum=0, maximum=10**18)
        limit = _coerce_int(limit, default=20, minimum=1, maximum=200)
        result = self.bridge.poll_events(after_cursor=after_cursor, session_key=session_key, limit=limit)
        return json.dumps(result, indent=2)

    def events_wait(self, after_cursor: int = 0, session_key: Optional[str] = None, timeout_ms: int = 30000) -> str:
        """Wait for the next conversation event (long-poll).

        Blocks until a matching event arrives or the timeout expires.
        Use this for near-real-time event delivery without polling.

        Args:
            after_cursor: Wait for events after this cursor
            session_key: Optional filter to one conversation
            timeout_ms: Maximum wait time in milliseconds (default 30000)
        """
        after_cursor = _coerce_int(after_cursor, default=0, minimum=0, maximum=10**18)
        timeout_ms = _coerce_int(timeout_ms, default=30000, minimum=0, maximum=300000)  # cap 5 min
        event = self.bridge.wait_for_event(after_cursor=after_cursor, session_key=session_key, timeout_ms=timeout_ms)
        return json.dumps({"event": event} if event else {"event": None, "reason": "timeout"}, indent=2)

    def messages_send(self, target: str, message: str) -> str:
        """Send a message to a platform conversation.

        The target format is "platform:chat_id" — same format used by the
        channels_list tool. You can also use human-friendly channel names
        that will be resolved automatically.

        Examples:
            target="telegram:6308981865"
            target="discord:#general"
            target="slack:#engineering"

        Args:
            target: Platform target in "platform:identifier" format
            message: The message text to send
        """
        if not target or not message:
            return json.dumps({"error": "Both target and message are required"})
        try:
            from tools.send_message_tool import send_message_tool
            return send_message_tool({"action": "send", "target": target, "message": message})
        except ImportError:
            return json.dumps({"error": "Send message tool not available"})
        except Exception as e:
            return json.dumps({"error": f"Send failed: {e}"})

    def channels_list(self, platform: Optional[str] = None) -> str:
        """List available messaging channels and targets across platforms.

        Returns channels that you can send messages to. The target strings
        returned here can be used directly with the messages_send tool.

        Args:
            platform: Filter by platform name (telegram, discord, slack, etc.)
        """
        directory = _load_channel_directory()
        if not directory:
            # No cached directory: derive send targets from the routing index.
            targets, seen = [], set()
            for key, entry in _load_sessions_index().items():
                origin = entry.get("origin", {})
                p = entry.get("platform") or origin.get("platform", "")
                chat_id = origin.get("chat_id", "")
                target_str = f"{p}:{chat_id}"
                if not p or not chat_id or not _platform_matches(platform, p) or target_str in seen:
                    continue
                seen.add(target_str)
                targets.append({"target": target_str, "platform": p,
                                "name": entry.get("display_name") or origin.get("chat_name", ""),
                                "chat_type": entry.get("chat_type", origin.get("chat_type", ""))})
            return json.dumps({"count": len(targets), "channels": targets}, indent=2)
        channels = []
        for plat, entries_list in directory.get("platforms", {}).items():
            if not _platform_matches(platform, plat) or not isinstance(entries_list, list):
                continue
            for ch in entries_list:
                if isinstance(ch, dict):
                    chat_id = ch.get("id", ch.get("chat_id", ""))
                    channels.append({"target": f"{plat}:{chat_id}" if chat_id else plat, "platform": plat,
                                     "name": ch.get("name", ch.get("display_name", "")), "chat_type": ch.get("type", "")})
        return json.dumps({"count": len(channels), "channels": channels}, indent=2)

    def permissions_list_open(self) -> str:
        """List pending approval requests observed during this bridge session.

        Returns exec and plugin approval requests that the bridge has seen
        since it started. Approvals are live-session only — older approvals
        from before the bridge connected are not included.
        """
        approvals = self.bridge.list_pending_approvals()
        return json.dumps({"count": len(approvals), "approvals": approvals}, indent=2)

    def permissions_respond(self, id: str, decision: str) -> str:
        """Respond to a pending approval request.

        Args:
            id: The approval ID from permissions_list_open
            decision: One of "allow-once", "allow-always", or "deny"
        """
        if decision not in {"allow-once", "allow-always", "deny"}:
            return json.dumps({"error": f"Invalid decision: {decision}. Must be allow-once, allow-always, or deny"})
        return json.dumps(self.bridge.respond_to_approval(id, decision), indent=2)


# Registration order == list_tools order (wire format).
_TOOL_NAMES = (
    "conversations_list", "conversation_get", "messages_read", "attachments_fetch",
    "events_poll", "events_wait", "messages_send", "channels_list",
    "permissions_list_open", "permissions_respond",
)


def create_mcp_server(event_bridge: Optional[EventBridge] = None) -> "MCPServer":
    """Create and return the Hermes MCP server with all tools registered."""
    if not _MCP_SERVER_AVAILABLE:
        raise ImportError(f"MCP server requires the 'mcp' package. Install with: {sys.executable} -m pip install 'mcp'")
    mcp = MCPServer("hermes", instructions=(
        "Hermes Agent messaging bridge. Use these tools to interact with "
        "conversations across Telegram, Discord, Slack, WhatsApp, Signal, "
        "Matrix, and other connected platforms."
    ))
    handlers = _ToolHandlers(event_bridge or EventBridge())
    for name in _TOOL_NAMES:
        mcp.tool()(getattr(handlers, name))
    return mcp


def run_mcp_server(verbose: bool = False) -> None:
    """Start the Hermes MCP server on stdio."""
    if not _MCP_SERVER_AVAILABLE:
        print("Error: MCP server requires the 'mcp' package.\n"
              f"Install with: {sys.executable} -m pip install 'mcp'", file=sys.stderr)
        sys.exit(1)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, stream=sys.stderr)
    bridge = EventBridge()
    try:
        if not bridge.start():
            raise RuntimeError("EventBridge could not establish its startup baseline")
        server = create_mcp_server(event_bridge=bridge)
        import asyncio
        asyncio.run(server.run_stdio_async())
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
