#!/usr/bin/env python3
"""
Async (background) delegation registry.

Backs ``delegate_task(background=true)``: the parent agent dispatches a
subagent that runs on a module-level daemon executor and returns a handle
immediately, so the user and the model can keep working while the child runs.

When the child finishes, a completion event is pushed onto the SHARED
``process_registry.completion_queue`` with ``type="async_delegation"``. The
CLI (``cli.py`` process_loop) and gateway (``_run_process_watcher`` /
``completion_queue`` drain) already poll that queue while the agent is idle
and forge a fresh user/internal turn from each event. We deliberately reuse
that rail rather than reaching into a running agent loop:

  - completions surface as a NEW turn when the agent is idle, never spliced
    between a tool result and an assistant message. That keeps strict
    message-role alternation legal and the prompt cache intact (hard
    invariant: never mutate past context).
  - we inherit the queue's de-dup, crash-recovery checkpoint, and the
    existing CLI + gateway drain wiring for free — no new drain loops in the
    two largest files in the repo.

The completion payload carries a RICH, self-contained task-source block (the
original goal, the context the parent supplied, toolsets, model, dispatch
time, status, and the full result summary). When the result re-enters the
conversation the parent may be deep in unrelated context and won't remember
why the subagent existed; the block lets it either use the result or
re-dispatch if the world has moved on.

This module owns ONLY the async lifecycle. The actual child build + run is
delegated back to ``delegate_tool._run_single_child`` via an injected
runner, so all the credential leasing, heartbeat, timeout, and result-shaping
logic stays in one place.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.thread_context import propagate_context_to_thread

logger = logging.getLogger(__name__)

# Back-compat alias — the daemon executor now lives in tools.daemon_pool so
# other subsystems (tool_executor, memory_manager, delegate_tool, skills_hub)
# can share it. Existing imports of ``_DaemonThreadPoolExecutor`` keep working.
_DaemonThreadPoolExecutor = DaemonThreadPoolExecutor


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
# A persistent daemon executor (NOT a `with ThreadPoolExecutor()` block, which
# would join on exit and defeat the whole point of async). Workers are daemon
# threads so a hard process exit doesn't hang on an in-flight child.
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_executor_max_workers: int = 0

_records_lock = threading.RLock()
_admission_condition = threading.Condition(_records_lock)
_admission_cap = 0
# Accepted records that have not yet left the admission backlog. Keep this
# separate from record status so a cancelled queued future still occupies its
# bounded slot until the executor drains it.
_pending_admission_ids: set[str] = set()
# delegation_id -> record dict. Kept for the lifetime of the run plus a short
# tail after completion so `list_async_delegations()` can show recent results.
_records: Dict[str, Dict[str, Any]] = {}

_ACTIVE_STATUSES = frozenset(
    {"queued", "running", "stalling", "cancelling", "finalizing"}
)

_DEFAULT_MAX_ASYNC_CHILDREN = 3
_BACKLOG_FULL_ERROR = (
    "Async delegation capacity reached: the process-wide background backlog is full. "
    "Wait for queued work to start before dispatching more."
)
# How many completed records to retain for status queries before pruning.
_MAX_RETAINED_COMPLETED = 50
_DURABLE_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_DURABLE_PENDING = 1000
# A pending completion whose delivery keeps failing is retried across claim
# cycles (and across restarts via restore_undelivered_completions). Cap the
# attempts so an unroutable row converges to a terminal 'dropped' state
# instead of replaying on every restart forever.
_MAX_DELIVERY_ATTEMPTS = 8
_DB_LOCK = threading.Lock()


def _current_process_identity() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _delivery_owner_live(
    owner_pid,
    owner_started_at,
    pid_exists,
    get_process_start_time,
) -> Optional[bool]:
    """Return verified liveness, or None when the incarnation is unverifiable."""
    if owner_pid is None:
        return False
    if type(owner_pid) is not int or owner_pid <= 0:
        raise ValueError("owner_pid must be a positive integer or null")
    if owner_started_at is not None and (
        type(owner_started_at) is not int or owner_started_at <= 0
    ):
        raise ValueError("owner_started_at must be a positive integer or null")
    if pid_exists is None:
        return None
    try:
        if not pid_exists(owner_pid):
            return False
    except Exception:
        return None
    if owner_started_at is None or get_process_start_time is None:
        return None
    try:
        current_started_at = get_process_start_time(owner_pid)
    except Exception:
        return None
    if current_started_at is None:
        return None
    return current_started_at == owner_started_at


def _delivery_attempt_count(value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("delivery_attempts must be a non-negative integer")
    return value


# ---------------------------------------------------------------------------
# Stale-delegation detection (progress-based, on by default)
# ---------------------------------------------------------------------------
# A detached runner that wedges before returning (e.g. stuck inside its first
# model API call — #60203) never reaches its ``finally`` finalizer, so no
# completion event is ever published: the delegation shows "dispatched"
# forever and the owning session looks silent until a process restart. We do
# NOT fix this with a wall-clock timeout — legitimate heavy subagent work
# (deep reviews, research fan-outs, slow reasoning models) must never be
# killed for taking long (see delegate_tool.DEFAULT_CHILD_TIMEOUT rationale).
# Instead a single monitor thread watches per-dispatch PROGRESS (api-call
# count + current tool, via an injected ``progress_fn``): a child that is
# advancing is left alone forever; a child with NO progress past the stale
# threshold is interrupted, given a grace window to unwind and deliver its
# partial results through the normal finalize path, and only force-finalized
# with a terminal ``stalled`` event if it never returns.
#
# Thresholds mirror the sync-path heartbeat staleness monitor in
# delegate_tool: idle (not inside a tool) stays tight so a wedged first API
# call is caught quickly; in-tool is much higher so legitimately slow tools
# (long terminal commands, big fetches) get time to finish.
_STALE_CHECK_INTERVAL = 30.0  # seconds between monitor sweeps
_STALE_IDLE_SECONDS = 450.0  # no progress, no current tool → stalled
_STALE_IN_TOOL_SECONDS = 1200.0  # no progress while inside a tool → stalled
_STALL_GRACE_SECONDS = 120.0  # after interrupt, time for the runner to return

_monitor_lock = threading.Lock()
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop = threading.Event()


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (async_delegation)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS async_delegations (
            delegation_id TEXT PRIMARY KEY,
            origin_session TEXT NOT NULL,
            origin_ui_session_id TEXT NOT NULL DEFAULT '',
            parent_session_id TEXT,
            state TEXT NOT NULL,
            dispatched_at REAL NOT NULL,
            completed_at REAL,
            updated_at REAL NOT NULL,
            event_json TEXT,
            result_json TEXT,
            delivery_state TEXT NOT NULL DEFAULT 'pending',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            delivered_at REAL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            task_json TEXT,
            delivery_claim TEXT,
            delivery_claimed_at REAL,
            origin_session_id TEXT NOT NULL DEFAULT '',
            delivery_tombstoned_at REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS ordinary_completion_deliveries (
            delivery_id TEXT PRIMARY KEY,
            event_json TEXT NOT NULL,
            delivery_state TEXT NOT NULL DEFAULT 'pending',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL,
            delivered_at REAL,
            delivery_claim TEXT,
            delivery_claimed_at REAL,
            delivery_owner_pid INTEGER,
            delivery_owner_started_at INTEGER,
            delivery_tombstoned_at REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS completion_delivery_recoveries (
            session_id TEXT NOT NULL,
            event_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            recovered_at REAL NOT NULL,
            PRIMARY KEY (session_id, event_json, reason)
        )"""
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(async_delegations)")}
    for name, sql_type in (
        ("owner_pid", "INTEGER"),
        ("owner_started_at", "INTEGER"),
        ("task_json", "TEXT"),
        ("delivery_claim", "TEXT"),
        ("delivery_claimed_at", "REAL"),
        # Raw api_server session id (X-Hermes-Session-Id) of the ORIGINATING
        # request — the wake self-post target. Without persisting it,
        # completions recovered after a process restart are unroutable on
        # api_server (the in-memory record that carried it is gone).
        ("origin_session_id", "TEXT"),
        ("delivery_tombstoned_at", "REAL"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE async_delegations ADD COLUMN {name} {sql_type}")
    ordinary_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(ordinary_completion_deliveries)")
    }
    if "delivery_tombstoned_at" not in ordinary_columns:
        conn.execute(
            "ALTER TABLE ordinary_completion_deliveries "
            "ADD COLUMN delivery_tombstoned_at REAL"
        )
    for table, id_column in (
        ("async_delegations", "delegation_id"),
        ("ordinary_completion_deliveries", "delivery_id"),
    ):
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS {table}_tombstone_insert
                AFTER INSERT ON {table}
                WHEN NEW.delivery_state NOT IN ('pending', 'effect_started')
                 AND NEW.delivery_tombstoned_at IS NULL
                BEGIN
                  UPDATE {table}
                     SET delivery_tombstoned_at=CAST(strftime('%s','now') AS REAL)
                   WHERE {id_column}=NEW.{id_column};
                END"""
        )
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS {table}_tombstone_update
                AFTER UPDATE OF delivery_state ON {table}
                WHEN NEW.delivery_state NOT IN ('pending', 'effect_started')
                 AND NEW.delivery_tombstoned_at IS NULL
                BEGIN
                  UPDATE {table}
                     SET delivery_tombstoned_at=CAST(strftime('%s','now') AS REAL)
                   WHERE {id_column}=NEW.{id_column};
                END"""
        )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the
    transaction; they do not close the connection. Using ``with _connect()``
    alone therefore leaks a connection — and its WAL/SHM file descriptors — on
    every durable dispatch, completion, and delivery-claim, deferring the close
    to the garbage collector. On a long-running gateway that exhausts
    ``RLIMIT_NOFILE`` (the cron-ledger sibling of this bug was #69567 / PR #69594).
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _persist_dispatch(record: Dict[str, Any]) -> None:
    now = time.time()
    try:
        from gateway.status import get_process_start_time
        owner_started_at = get_process_start_time(__import__("os").getpid())
    except Exception:
        owner_started_at = None
    task_payload = {
        key: record.get(key)
        for key in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch")
        if key in record
    }
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, owner_pid,
                owner_started_at, task_json, origin_session_id)
               VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (record["delegation_id"], record.get("session_key", ""),
             record.get("origin_ui_session_id", ""), record.get("parent_session_id"),
             record["dispatched_at"], now, __import__("os").getpid(),
             owner_started_at, json.dumps(task_payload),
             record.get("origin_session_id", "")),
        )
    _prune_durable_records()


def _delete_durable_delegation(delegation_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """DELETE FROM async_delegations WHERE delegation_id=?
               AND state='running' AND delivery_state='pending'
               AND delivery_tombstoned_at IS NULL""",
            (delegation_id,),
        )


def _prune_durable_records() -> None:
    """Retire only delivery facts that completed the full retention window."""
    now = time.time()
    cutoff = now - _DURABLE_RETENTION_SECONDS
    with _DB_LOCK, _transaction() as conn:
        for table in ("async_delegations", "ordinary_completion_deliveries"):
            conn.execute(
                f"""UPDATE {table} SET delivery_tombstoned_at=?
                    WHERE delivery_tombstoned_at IS NULL
                      AND delivery_state NOT IN ('pending', 'effect_started')""",
                (now,),
            )
            conn.execute(
                f"DELETE FROM {table} WHERE delivery_tombstoned_at < ?",
                (cutoff,),
            )
        terminal_count = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE state NOT IN ('running','finalizing')"
        ).fetchone()[0]
        excess = max(0, terminal_count - _MAX_RETAINED_COMPLETED)
        if excess:
            conn.execute(
                """DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing')
                       AND delivery_state NOT IN ('pending', 'effect_started', 'delivered')
                       AND delivery_state NOT LIKE 'recovery_%'
                     ORDER BY updated_at ASC LIMIT ?
                   )""",
                (excess,),
            )
        pending_count = conn.execute(
            """SELECT COUNT(*) FROM async_delegations
               WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'"""
        ).fetchone()[0]
        overflow = max(0, pending_count - _MAX_DURABLE_PENDING)
        if overflow:
            conn.execute(
                """DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                     ORDER BY updated_at ASC LIMIT ?
                   )""",
                (overflow,),
            )


def _persist_completion(event: Dict[str, Any], result: Dict[str, Any]) -> None:
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
               event_json=?, result_json=?, delivery_state='pending'
               WHERE delegation_id=? AND state IN ('running', 'finalizing')
                 AND delivery_state='pending' AND delivery_tombstoned_at IS NULL""",
            (event.get("status", "completed"), event.get("completed_at", now), now,
             json.dumps(event), json.dumps(result), event["delegation_id"]),
        )


def _note_delivery_attempt(delegation_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET delivery_attempts=delivery_attempts+1, updated_at=? WHERE delegation_id=?",
            (time.time(), delegation_id),
        )


def recover_abandoned_delegations() -> int:
    """Classify records whose owning process disappeared as outcome unknown."""
    pid_exists: Optional[Callable[[int], bool]] = None
    process_start_time: Optional[Callable[[int], Optional[int]]] = None
    try:
        from gateway.status import (
            _pid_exists as pid_exists,
            get_process_start_time as process_start_time,
        )
    except Exception:
        pass
    now = time.time()
    recovered = 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, origin_session, origin_ui_session_id,
                      parent_session_id, dispatched_at, owner_pid,
                      owner_started_at, task_json, origin_session_id
               FROM async_delegations WHERE state IN ('running','finalizing')"""
        ).fetchall()
        for row in rows:
            (delegation_id, session_key, origin_ui, parent_id, dispatched_at,
             pid, started, task_json, origin_session_id) = row
            try:
                live = _delivery_owner_live(
                    pid, started, pid_exists, process_start_time
                )
            except (TypeError, ValueError) as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
                parked_payload = _corrupt_completion_payload(
                    expected_type="async_delegation",
                    row_id=str(delegation_id),
                    diagnostic=diagnostic,
                )
                conn.execute(
                    """UPDATE async_delegations
                       SET state='unknown', completed_at=?, updated_at=?,
                           event_json=?, delivery_state='parked_corrupt',
                           delivery_claim=NULL, delivery_claimed_at=NULL,
                           owner_pid=NULL, owner_started_at=NULL
                       WHERE delegation_id=? AND state IN ('running','finalizing')""",
                    (now, now, parked_payload, delegation_id),
                )
                logger.warning(
                    "Parking corrupt abandoned async row %s: %s",
                    delegation_id,
                    diagnostic,
                )
                continue
            if live is True:
                continue
            if live is None:
                conn.execute(
                    """UPDATE async_delegations
                       SET state='unknown', completed_at=?, updated_at=?,
                           delivery_state='recovery_owner_unverifiable',
                           delivery_tombstoned_at=COALESCE(delivery_tombstoned_at, ?),
                           delivery_claim=NULL, delivery_claimed_at=NULL,
                           owner_pid=NULL, owner_started_at=NULL
                       WHERE delegation_id=? AND state IN ('running','finalizing')""",
                    (now, now, now, delegation_id),
                )
                logger.warning(
                    "Async delegation %s has an unverifiable owner; retaining "
                    "no-replay recovery state.",
                    delegation_id,
                )
                recovered += 1
                continue
            try:
                task = json.loads(task_json or "{}")
                if not isinstance(task, dict):
                    raise ValueError(
                        f"task_json must decode to an object, got {type(task).__name__}"
                    )
            except (TypeError, ValueError) as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
                parked_payload = _corrupt_completion_payload(
                    expected_type="async_delegation",
                    row_id=str(delegation_id),
                    diagnostic=diagnostic,
                )
                conn.execute(
                    """UPDATE async_delegations
                       SET state='unknown', completed_at=?, updated_at=?,
                           event_json=?, delivery_state='parked_corrupt',
                           delivery_claim=NULL, delivery_claimed_at=NULL,
                           owner_pid=NULL, owner_started_at=NULL
                       WHERE delegation_id=? AND state IN ('running','finalizing')""",
                    (now, now, parked_payload, delegation_id),
                )
                logger.warning(
                    "Parking corrupt abandoned async row %s: %s",
                    delegation_id,
                    diagnostic,
                )
                continue
            event = {
                "type": "async_delegation", "delegation_id": delegation_id,
                "session_key": session_key, "origin_ui_session_id": origin_ui,
                # Restore the durable wake target so completions recovered
                # after a restart remain routable to api_server sessions.
                "origin_session_id": origin_session_id or "",
                "parent_session_id": parent_id, "goal": task.get("goal", ""),
                "goals": task.get("goals"), "context": task.get("context"),
                "toolsets": task.get("toolsets"), "role": task.get("role"),
                "model": task.get("model"), "is_batch": bool(task.get("is_batch")),
                "status": "unknown", "summary": None,
                "error": "Delegation owner exited before recording a terminal result; outcome unknown.",
                "dispatched_at": dispatched_at, "completed_at": now,
            }
            result = {"status": "unknown", "summary": None, "error": event["error"]}
            conn.execute(
                """UPDATE async_delegations SET state='unknown', completed_at=?,
                   updated_at=?, event_json=?, result_json=?, delivery_state='pending'
                   WHERE delegation_id=?""",
                (now, now, json.dumps(event), json.dumps(result), delegation_id),
            )
            recovered += 1
    return recovered


def _bind_async_routing_fields(
    event: Dict[str, Any], routing: Dict[str, Any]
) -> Optional[str]:
    """Replace persisted routing aliases only when they agree with row authority."""
    for field, authoritative in routing.items():
        if authoritative is None:
            authoritative = ""
        if not isinstance(authoritative, str):
            return f"authoritative routing field {field} is invalid"
        candidate = event.get(field)
        if candidate is not None and not isinstance(candidate, str):
            return f"routing field {field} is invalid"
        if field in event and (candidate or "") != authoritative:
            return f"routing field {field} does not match its durable row"
        event[field] = authoritative
    if "origin_session" in event:
        candidate = event.pop("origin_session")
        if not isinstance(candidate, str) or candidate != event["session_key"]:
            return "origin_session does not match its durable row"
    return None


def _decode_restored_completion(
    payload: str,
    *,
    expected_type: str,
    row_id: str,
    routing: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        event = _load_durable_event_payload(payload)
    except (TypeError, ValueError) as exc:
        return None, type(exc).__name__
    if not isinstance(event, dict):
        return None, f"expected object, got {type(event).__name__}"
    if event.get("type") != expected_type:
        return event, f"unexpected event type ({type(event.get('type')).__name__})"
    if expected_type == "async_delegation":
        if event.get("delegation_id") != row_id:
            return event, "delegation identity does not match its durable row"
        if routing is not None:
            diagnostic = _bind_async_routing_fields(event, routing)
            if diagnostic is not None:
                return event, diagnostic
    elif _ordinary_completion_delivery_id(event) is None:
        return event, "ordinary completion identity is invalid"
    return event, None


def _corrupt_completion_payload(
    *, expected_type: str, row_id: str, diagnostic: str
) -> str:
    return json.dumps(
        {
            "type": expected_type,
            "corrupt_delivery_row": row_id,
            "corrupt_delivery_diagnostic": diagnostic,
        },
        sort_keys=True,
    )


def _reconcile_committed_ack_events(conn, events, now: float) -> int:
    """Settle checkpointed parent effects before abandoned claims can replay."""
    reconciled = 0
    for raw_event in events or ():
        if not isinstance(raw_event, dict):
            continue
        event = _durable_event_payload(raw_event)
        if event.get("type") == "async_delegation":
            delegation_id = event.get("delegation_id")
            if not isinstance(delegation_id, str) or not delegation_id:
                continue
            conn.execute(
                """INSERT INTO async_delegations
                   (delegation_id, origin_session, origin_ui_session_id, state,
                    dispatched_at, updated_at, delivery_state,
                    delivery_tombstoned_at)
                   VALUES (?, '', '', 'tombstone', ?, ?,
                           'recovery_committed_ack_failed', ?)
                   ON CONFLICT(delegation_id) DO UPDATE SET
                     delivery_state=CASE
                       WHEN async_delegations.delivery_state='delivered'
                       THEN 'delivered'
                       ELSE 'recovery_committed_ack_failed'
                     END,
                     delivery_claim=NULL, delivery_claimed_at=NULL,
                     owner_pid=NULL, owner_started_at=NULL,
                     delivery_tombstoned_at=COALESCE(
                       async_delegations.delivery_tombstoned_at, excluded.delivery_tombstoned_at
                     ),
                     updated_at=excluded.updated_at""",
                (delegation_id, now, now, now),
            )
            reconciled += 1
            continue
        delivery_id = _ordinary_completion_delivery_id(event)
        if delivery_id is None:
            continue
        payload = json.dumps(_durable_event_payload(event), sort_keys=True)
        _migrate_ordinary_completion_legacy_row(
            conn, event, delivery_id, payload, now
        )
        conn.execute(
            """INSERT INTO ordinary_completion_deliveries
               (delivery_id, event_json, delivery_state, updated_at,
                delivery_tombstoned_at)
               VALUES (?, ?, 'recovery_committed_ack_failed', ?, ?)
               ON CONFLICT(delivery_id) DO UPDATE SET
                 delivery_state=CASE
                   WHEN ordinary_completion_deliveries.delivery_state='delivered'
                   THEN 'delivered'
                   ELSE 'recovery_committed_ack_failed'
                 END,
                 delivery_claim=NULL, delivery_claimed_at=NULL,
                 delivery_owner_pid=NULL, delivery_owner_started_at=NULL,
                 delivery_tombstoned_at=COALESCE(
                   ordinary_completion_deliveries.delivery_tombstoned_at,
                   excluded.delivery_tombstoned_at
                 ),
                 updated_at=excluded.updated_at""",
            (delivery_id, payload, now, now),
        )
        reconciled += 1
    return reconciled


def reconcile_committed_completion_acks(events) -> int:
    """Reconcile durable ACK-only facts without re-running provider effects."""
    with _DB_LOCK, _transaction() as conn:
        reconciled = _reconcile_committed_ack_events(conn, events, time.time())
    _prune_durable_records()
    return reconciled


def restore_undelivered_completions(target_queue) -> int:
    """Enqueue durable retryable completions as fresh turns after process start.

    Every restored event is stamped ``restored=True`` (in-memory only — the
    stamp is added after the durable payload is deserialized and is never
    persisted). Restored events originate from a *previous* process, so no
    consumer in THIS process implicitly owns them: drain paths that run
    without an ownership filter (the legacy single-session behavior) must
    leave them queued for a consumer that can positively prove ownership,
    otherwise a brand-new session adopts a dead session's delegation
    results seconds after boot (#64484).
    """
    recover_abandoned_delegations()
    pid_exists: Optional[Callable[[int], bool]] = None
    process_start_time: Optional[Callable[[int], Optional[int]]] = None
    try:
        from gateway.status import (
            _pid_exists as pid_exists,
            get_process_start_time as process_start_time,
        )
    except Exception:
        pass
    restored_events = []
    with _DB_LOCK, _transaction() as conn:
        now = time.time()
        claimed_async_rows = conn.execute(
            """SELECT delegation_id, event_json, owner_pid, owner_started_at
               FROM async_delegations
               WHERE state != 'running' AND delivery_state='effect_started'
                 AND event_json IS NOT NULL"""
        ).fetchall()
        for delegation_id, payload, owner_pid, owner_started_at in claimed_async_rows:
            try:
                live = _delivery_owner_live(
                    owner_pid,
                    owner_started_at,
                    pid_exists,
                    process_start_time,
                )
            except (TypeError, ValueError) as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
                parked_payload = _corrupt_completion_payload(
                    expected_type="async_delegation",
                    row_id=delegation_id,
                    diagnostic=diagnostic,
                )
                conn.execute(
                    """UPDATE async_delegations
                       SET event_json=?, delivery_state='parked_corrupt',
                           delivery_claim=NULL, delivery_claimed_at=NULL,
                           owner_pid=NULL, owner_started_at=NULL, updated_at=?
                       WHERE delegation_id=? AND delivery_state='effect_started'""",
                    (parked_payload, now, delegation_id),
                )
                logger.warning(
                    "Parking corrupt async completion row %s: %s",
                    delegation_id,
                    diagnostic,
                )
                continue
            if live:
                continue
            conn.execute(
                """UPDATE async_delegations
                   SET delivery_state='recovery_effect_started_owner_lost',
                       delivery_claim=NULL, delivery_claimed_at=NULL,
                       owner_pid=NULL, owner_started_at=NULL, updated_at=?
                   WHERE delegation_id=? AND delivery_state='effect_started'""",
                (now, delegation_id),
            )
            logger.warning(
                "Async completion %s lost its delivery owner after effect start; "
                "retaining recovery state without provider replay.",
                delegation_id,
            )

        rows = conn.execute(
            """SELECT delegation_id, event_json, delivery_attempts,
                      origin_session, origin_ui_session_id, origin_session_id,
                      parent_session_id
               FROM async_delegations
               WHERE state != 'running' AND delivery_state='pending'
                 AND event_json IS NOT NULL
               ORDER BY completed_at, delegation_id"""
        ).fetchall()
        for (
            delegation_id,
            payload,
            attempts,
            origin_session,
            origin_ui_session_id,
            origin_session_id,
            parent_session_id,
        ) in rows:
            try:
                attempt_count = _delivery_attempt_count(attempts)
            except (TypeError, ValueError) as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
                parked_payload = _corrupt_completion_payload(
                    expected_type="async_delegation",
                    row_id=delegation_id,
                    diagnostic=diagnostic,
                )
                conn.execute(
                    """UPDATE async_delegations
                       SET event_json=?, delivery_state='parked_corrupt', updated_at=?
                       WHERE delegation_id=? AND delivery_state='pending'""",
                    (parked_payload, now, delegation_id),
                )
                logger.warning(
                    "Parking corrupt async completion row %s: %s",
                    delegation_id,
                    diagnostic,
                )
                continue
            # 'parked' (not 'delivered'): nothing was handed to a consumer.
            if attempt_count >= _MAX_DELIVERY_ATTEMPTS:
                conn.execute(
                    """UPDATE async_delegations SET delivery_state='parked', updated_at=?
                       WHERE delegation_id=?""",
                    (now, delegation_id),
                )
                logger.warning(
                    "Async delegation %s exhausted its %d delivery attempts "
                    "without a session that could own it; parking it instead of "
                    "replaying on every restart (result remains queryable).",
                    delegation_id,
                    _MAX_DELIVERY_ATTEMPTS,
                )
                continue
            event, diagnostic = _decode_restored_completion(
                payload,
                expected_type="async_delegation",
                row_id=delegation_id,
                routing={
                    "session_key": origin_session,
                    "origin_ui_session_id": origin_ui_session_id,
                    "origin_session_id": origin_session_id,
                    "parent_session_id": parent_session_id,
                },
            )
            if diagnostic is not None:
                parked_payload = _corrupt_completion_payload(
                    expected_type="async_delegation",
                    row_id=delegation_id,
                    diagnostic=diagnostic,
                )
                conn.execute(
                    """UPDATE async_delegations
                       SET event_json=?, delivery_state='parked_corrupt', updated_at=?
                       WHERE delegation_id=? AND delivery_state='pending'""",
                    (parked_payload, now, delegation_id),
                )
                logger.warning(
                    "Parking corrupt async completion row %s: %s",
                    delegation_id,
                    diagnostic,
                )
                continue
            if event is None:
                continue
            event["restored"] = True
            restored_events.append(event)
        claimed_rows = conn.execute(
            """SELECT delivery_id, event_json, delivery_owner_pid,
                      delivery_owner_started_at
               FROM ordinary_completion_deliveries
               WHERE delivery_state='effect_started'"""
        ).fetchall()
        for delivery_id, _payload, owner_pid, owner_started_at in claimed_rows:
            try:
                live = _delivery_owner_live(
                    owner_pid,
                    owner_started_at,
                    pid_exists,
                    process_start_time,
                )
            except (TypeError, ValueError) as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
                parked_payload = _corrupt_completion_payload(
                    expected_type="completion",
                    row_id=delivery_id,
                    diagnostic=diagnostic,
                )
                conn.execute(
                    """UPDATE ordinary_completion_deliveries
                       SET event_json=?, delivery_state='parked_corrupt',
                           delivery_claim=NULL, delivery_claimed_at=NULL,
                           delivery_owner_pid=NULL,
                           delivery_owner_started_at=NULL, updated_at=?
                       WHERE delivery_id=? AND delivery_state='effect_started'""",
                    (parked_payload, now, delivery_id),
                )
                logger.warning(
                    "Parking corrupt ordinary completion row %s: %s",
                    delivery_id,
                    diagnostic,
                )
                continue
            if live:
                continue
            conn.execute(
                """UPDATE ordinary_completion_deliveries
                   SET delivery_state='recovery_effect_started_owner_lost',
                       delivery_claim=NULL, delivery_claimed_at=NULL,
                       delivery_owner_pid=NULL,
                       delivery_owner_started_at=NULL, updated_at=?
                   WHERE delivery_id=? AND delivery_state='effect_started'""",
                (now, delivery_id),
            )
            logger.warning(
                "Ordinary completion %s lost its delivery owner after effect start; "
                "retaining recovery state without provider replay.",
                delivery_id,
            )
        pending_rows = conn.execute(
            """SELECT delivery_id, event_json
               FROM ordinary_completion_deliveries
               WHERE delivery_state='pending' ORDER BY updated_at, delivery_id"""
        ).fetchall()
        legacy_migrations = {}
        mismatches = {}
        for delivery_id, payload in pending_rows:
            event, diagnostic = _decode_restored_completion(
                payload, expected_type="completion", row_id=delivery_id
            )
            if diagnostic is not None:
                parked_payload = _corrupt_completion_payload(
                    expected_type="completion",
                    row_id=delivery_id,
                    diagnostic=diagnostic,
                )
                conn.execute(
                    """UPDATE ordinary_completion_deliveries
                       SET event_json=?, delivery_state='parked_corrupt', updated_at=?
                       WHERE delivery_id=? AND delivery_state='pending'""",
                    (parked_payload, now, delivery_id),
                )
                logger.warning(
                    "Parking corrupt ordinary completion row %s: %s",
                    delivery_id,
                    diagnostic,
                )
                continue
            if event is None:
                continue
            normalized_id = _ordinary_completion_delivery_id(event)
            normalized_payload = json.dumps(
                _durable_event_payload(event), sort_keys=True
            )
            if normalized_id != delivery_id:
                target = (normalized_id, normalized_payload)
                if delivery_id == _ordinary_completion_legacy_delivery_id(event):
                    legacy_migrations[delivery_id] = target
                else:
                    mismatches[delivery_id] = target
            else:
                conn.execute(
                    """UPDATE ordinary_completion_deliveries SET event_json=?
                       WHERE delivery_id=? AND delivery_state='pending'""",
                    (normalized_payload, delivery_id),
                )

        for delivery_id, (
            normalized_id,
            normalized_payload,
        ) in legacy_migrations.items():
            target_exists = conn.execute(
                "SELECT 1 FROM ordinary_completion_deliveries WHERE delivery_id=?",
                (normalized_id,),
            ).fetchone()
            if target_exists:
                conn.execute(
                    """UPDATE ordinary_completion_deliveries
                       SET event_json=?, delivery_state='parked_corrupt', updated_at=?
                       WHERE delivery_id=? AND delivery_state='pending'""",
                    (
                        _corrupt_completion_payload(
                            expected_type="completion",
                            row_id=delivery_id,
                            diagnostic="duplicate legacy ordinary completion identity",
                        ),
                        now,
                        delivery_id,
                    ),
                )
            else:
                conn.execute(
                    """UPDATE ordinary_completion_deliveries
                       SET delivery_id=?, event_json=? WHERE delivery_id=?
                         AND delivery_state='pending'""",
                    (normalized_id, normalized_payload, delivery_id),
                )

        cycle_rows = set()
        visited_rows = set()
        for start in mismatches:
            if start in visited_rows:
                continue
            path = []
            positions = {}
            current = start
            while current in mismatches and current not in positions:
                if current in visited_rows:
                    break
                positions[current] = len(path)
                path.append(current)
                current = mismatches[current][0]
            if current in positions:
                cycle_rows.update(path[positions[current] :])
            visited_rows.update(path)

        for delivery_id, (normalized_id, normalized_payload) in mismatches.items():
            if delivery_id in cycle_rows:
                conn.execute(
                    """UPDATE ordinary_completion_deliveries SET event_json=?
                       WHERE delivery_id=? AND delivery_state='pending'""",
                    (normalized_payload, normalized_id),
                )
                continue
            diagnostic = "ordinary completion identity does not match its durable row"
            parked_payload = _corrupt_completion_payload(
                expected_type="completion",
                row_id=delivery_id,
                diagnostic=diagnostic,
            )
            conn.execute(
                """UPDATE ordinary_completion_deliveries
                   SET event_json=?, delivery_state='parked_corrupt', updated_at=?
                   WHERE delivery_id=? AND delivery_state='pending'""",
                (parked_payload, now, delivery_id),
            )
            logger.warning(
                "Parking corrupt ordinary completion row %s: %s",
                delivery_id,
                diagnostic,
            )
        if cycle_rows:
            logger.warning(
                "Repaired a closed ordinary completion identity cycle across %d rows",
                len(cycle_rows),
            )
        ordinary_rows = conn.execute(
            """SELECT delivery_id, event_json, delivery_attempts
               FROM ordinary_completion_deliveries
               WHERE delivery_state='pending' ORDER BY updated_at, delivery_id"""
        ).fetchall()
        for delivery_id, payload, attempts in ordinary_rows:
            try:
                attempt_count = _delivery_attempt_count(attempts)
            except (TypeError, ValueError) as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
                parked_payload = _corrupt_completion_payload(
                    expected_type="completion",
                    row_id=delivery_id,
                    diagnostic=diagnostic,
                )
                conn.execute(
                    """UPDATE ordinary_completion_deliveries
                       SET event_json=?, delivery_state='parked_corrupt', updated_at=?
                       WHERE delivery_id=? AND delivery_state='pending'""",
                    (parked_payload, now, delivery_id),
                )
                logger.warning(
                    "Parking corrupt ordinary completion row %s: %s",
                    delivery_id,
                    diagnostic,
                )
                continue
            if attempt_count >= _MAX_DELIVERY_ATTEMPTS:
                conn.execute(
                    """UPDATE ordinary_completion_deliveries
                       SET delivery_state='dropped', updated_at=?
                       WHERE delivery_id=?""",
                    (now, delivery_id),
                )
                continue
            event, diagnostic = _decode_restored_completion(
                payload, expected_type="completion", row_id=delivery_id
            )
            if diagnostic is not None:
                parked_payload = _corrupt_completion_payload(
                    expected_type="completion",
                    row_id=delivery_id,
                    diagnostic=diagnostic,
                )
                conn.execute(
                    """UPDATE ordinary_completion_deliveries
                       SET event_json=?, delivery_state='parked_corrupt', updated_at=?
                       WHERE delivery_id=? AND delivery_state='pending'""",
                    (parked_payload, now, delivery_id),
                )
                logger.warning(
                    "Parking corrupt ordinary completion row %s: %s",
                    delivery_id,
                    diagnostic,
                )
                continue
            if event is None:
                continue
            event["restored"] = True
            restored_events.append(event)
    for event in restored_events:
        target_queue.put(event)
    return len(restored_events)


def mark_completion_delivered(delegation_id: str) -> bool:
    """Atomically acknowledge successful injection of a durable completion."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered', delivered_at=?, updated_at=?
               WHERE delegation_id=? AND delivery_state!='delivered'""",
            (now, now, delegation_id),
        )
        return cur.rowcount == 1


def claim_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Claim one pending completion across competing consumers/processes."""
    now = time.time()
    owner_pid, owner_started_at = _current_process_identity()
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT delivery_state FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()
        if row is None:
            return True  # legacy event created before durable dispatch
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='effect_started',
                      delivery_claim=?, delivery_claimed_at=?,
                      owner_pid=?, owner_started_at=?,
                      delivery_attempts=delivery_attempts+1, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
            (
                claim_id,
                now,
                owner_pid,
                owner_started_at,
                now,
                delegation_id,
                now - 300,
            ),
        )
        return cur.rowcount == 1


def _ordinary_completion_delivery_id(evt: Dict[str, Any]) -> Optional[str]:
    """Return canonical tuple authority or a sanitized envelope fingerprint."""
    if evt.get("type", "completion") != "completion":
        return None
    from tools.process_registry import ProcessRegistry

    identity = ProcessRegistry._completion_durable_identity(evt)
    if identity is None:
        return None
    return json.dumps(identity, ensure_ascii=False, separators=(",", ":"))


def _ordinary_completion_legacy_delivery_id(evt: Dict[str, Any]) -> Optional[str]:
    """Return the former tuple key only when the current envelope owns its core."""
    if _ordinary_completion_delivery_id(evt) is None:
        return None
    identity = (
        "completion",
        evt["session_id"],
        evt["started_at"],
        evt.get("session_key", ""),
    )
    return json.dumps(identity, ensure_ascii=False, separators=(",", ":"))


def _durable_event_payload(evt: Dict[str, Any]) -> Dict[str, Any]:
    """Strip process-local delivery state before persistence or replay."""
    return {
        key: value
        for key, value in evt.items()
        if not key.startswith("_completion_delivery_")
    }


def _load_durable_event_payload(payload: str) -> Any:
    """Decode persisted event data without restoring process-local authority."""
    event = json.loads(payload)
    return _durable_event_payload(event) if isinstance(event, dict) else event


def _migrate_ordinary_completion_legacy_row(
    conn: sqlite3.Connection,
    evt: Dict[str, Any],
    delivery_id: str,
    payload: str,
    now: float,
) -> None:
    """Re-key an honest tuple-era row before a current event can republish it."""
    legacy_id = _ordinary_completion_legacy_delivery_id(evt)
    if legacy_id is None or legacy_id == delivery_id:
        return
    legacy = conn.execute(
        """SELECT event_json, delivery_state, delivery_tombstoned_at
           FROM ordinary_completion_deliveries WHERE delivery_id=?""",
        (legacy_id,),
    ).fetchone()
    if legacy is None:
        return
    try:
        stored = _load_durable_event_payload(legacy[0])
    except (TypeError, ValueError):
        return
    if not isinstance(stored, dict) or _ordinary_completion_delivery_id(stored) != delivery_id:
        return
    target = conn.execute(
        "SELECT delivery_state FROM ordinary_completion_deliveries WHERE delivery_id=?",
        (delivery_id,),
    ).fetchone()
    if target is None:
        conn.execute(
            """UPDATE ordinary_completion_deliveries
               SET delivery_id=?, event_json=? WHERE delivery_id=?""",
            (delivery_id, payload, legacy_id),
        )
        return
    if legacy[1] != "pending" and target[0] == "pending":
        state = (
            "recovery_legacy_identity_conflict"
            if legacy[1] == "effect_started"
            else legacy[1]
        )
        conn.execute(
            """UPDATE ordinary_completion_deliveries
               SET delivery_state=?, delivery_claim=NULL,
                   delivery_claimed_at=NULL, delivery_owner_pid=NULL,
                   delivery_owner_started_at=NULL,
                   delivery_tombstoned_at=COALESCE(delivery_tombstoned_at, ?),
                   updated_at=?
               WHERE delivery_id=? AND delivery_state='pending'""",
            (state, legacy[2] or now, now, delivery_id),
        )


def persist_event_delivery(evt: Dict[str, Any]) -> bool:
    """Create the durable ordinary-completion receipt before queue delivery."""
    delivery_id = _ordinary_completion_delivery_id(evt)
    if delivery_id is None:
        return True
    now = time.time()
    payload = json.dumps(_durable_event_payload(evt), sort_keys=True)
    with _DB_LOCK, _transaction() as conn:
        _migrate_ordinary_completion_legacy_row(conn, evt, delivery_id, payload, now)
        conn.execute(
            """INSERT OR IGNORE INTO ordinary_completion_deliveries
               (delivery_id, event_json, delivery_state, updated_at)
               VALUES (?, ?, 'pending', ?)""",
            (delivery_id, payload, now),
        )
    _prune_durable_records()
    return True


def record_completion_delivery_recovery(
    session_id: str, reason: str, event: Dict[str, Any]
) -> bool:
    """Persist a missing in-memory suffix/marker disposition before clearing it."""
    if not session_id or not reason or not isinstance(event, dict):
        return False
    payload = json.dumps(
        _durable_event_payload(event), sort_keys=True, separators=(",", ":")
    )
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO completion_delivery_recoveries
               (session_id, event_json, reason, recovered_at)
               VALUES (?, ?, ?, ?)""",
            (session_id, payload, reason, time.time()),
        )
    return True


def get_completion_delivery_recoveries(session_id: str) -> List[Dict[str, Any]]:
    """Return durable missing-state dispositions for one session."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT event_json, reason, recovered_at
               FROM completion_delivery_recoveries
               WHERE session_id=? ORDER BY recovered_at, reason""",
            (session_id,),
        ).fetchall()
    return [
        {
            "event": _load_durable_event_payload(row[0]),
            "reason": row[1],
            "recovered_at": row[2],
        }
        for row in rows
    ]


def claim_event_delivery(evt: Dict[str, Any], consumer: str) -> Optional[str]:
    """Claim a durable completion or reject a stale heartbeat event."""
    if evt.get("type") == "heartbeat":
        from tools.runtime_heartbeat import runtime_heartbeat

        return "" if runtime_heartbeat.is_event_current(evt) else None
    if evt.get("type") == "async_delegation":
        delegation_id = str(evt.get("delegation_id") or "")
        if not delegation_id:
            return None
        with _DB_LOCK, _transaction() as conn:
            row = conn.execute(
                """SELECT origin_session, origin_ui_session_id,
                          origin_session_id, parent_session_id, delivery_state
                   FROM async_delegations WHERE delegation_id=?""",
                (delegation_id,),
            ).fetchone()
            if row is None:
                evt["_completion_delivery_legacy_async"] = True
            else:
                evt.pop("_completion_delivery_legacy_async", None)
            if row is not None and row[4] not in {"pending", "effect_started"}:
                return None
            if row is not None:
                diagnostic = _bind_async_routing_fields(
                    evt,
                    {
                        "session_key": row[0],
                        "origin_ui_session_id": row[1],
                        "origin_session_id": row[2],
                        "parent_session_id": row[3],
                    },
                )
                if diagnostic is not None:
                    conn.execute(
                        """UPDATE async_delegations
                           SET event_json=?, delivery_state='parked_corrupt',
                               delivery_claim=NULL, delivery_claimed_at=NULL,
                               owner_pid=NULL, owner_started_at=NULL, updated_at=?
                           WHERE delegation_id=?
                             AND delivery_state IN ('pending', 'effect_started')""",
                        (
                            _corrupt_completion_payload(
                                expected_type="async_delegation",
                                row_id=delegation_id,
                                diagnostic=diagnostic,
                            ),
                            time.time(),
                            delegation_id,
                        ),
                    )
                    return None
    retained_claim = str(evt.get("_completion_delivery_retained_claim_id") or "")
    if retained_claim:
        now = time.time()
        if evt.get("type") == "async_delegation":
            delegation_id = str(evt.get("delegation_id") or "")
            if not delegation_id:
                return None
            with _DB_LOCK, _transaction() as conn:
                cur = conn.execute(
                    """UPDATE async_delegations
                       SET delivery_attempts=delivery_attempts+1, updated_at=?
                       WHERE delegation_id=? AND delivery_state='effect_started'
                         AND delivery_claim=? AND delivery_attempts<?""",
                    (now, delegation_id, retained_claim, _MAX_DELIVERY_ATTEMPTS),
                )
                if cur.rowcount == 0:
                    conn.execute(
                        """UPDATE async_delegations SET delivery_state='dropped',
                                  delivery_claim=NULL, delivery_claimed_at=NULL,
                                  updated_at=?
                           WHERE delegation_id=? AND delivery_state='effect_started'
                             AND delivery_claim=? AND delivery_attempts>=?""",
                        (now, delegation_id, retained_claim, _MAX_DELIVERY_ATTEMPTS),
                    )
            return retained_claim if cur.rowcount == 1 else None
        delivery_id = _ordinary_completion_delivery_id(evt)
        if delivery_id is None:
            return None
        with _DB_LOCK, _transaction() as conn:
            payload = json.dumps(_durable_event_payload(evt), sort_keys=True)
            _migrate_ordinary_completion_legacy_row(
                conn, evt, delivery_id, payload, now
            )
            cur = conn.execute(
                """UPDATE ordinary_completion_deliveries
                   SET delivery_attempts=delivery_attempts+1, updated_at=?
                   WHERE delivery_id=? AND delivery_state='effect_started'
                     AND delivery_claim=? AND delivery_attempts<?""",
                (now, delivery_id, retained_claim, _MAX_DELIVERY_ATTEMPTS),
            )
            if cur.rowcount == 0:
                conn.execute(
                    """UPDATE ordinary_completion_deliveries
                       SET delivery_state='dropped', delivery_claim=NULL,
                           delivery_claimed_at=NULL, delivery_owner_pid=NULL,
                           delivery_owner_started_at=NULL, updated_at=?
                       WHERE delivery_id=? AND delivery_state='effect_started'
                         AND delivery_claim=? AND delivery_attempts>=?""",
                    (now, delivery_id, retained_claim, _MAX_DELIVERY_ATTEMPTS),
                )
        return retained_claim if cur.rowcount == 1 else None
    if evt.get("type") == "async_delegation":
        delegation_id = str(evt.get("delegation_id") or "")
        if not delegation_id:
            return ""
        claim_id = f"{consumer}:{__import__('os').getpid()}:{uuid.uuid4().hex}"
        return claim_id if claim_completion_delivery(delegation_id, claim_id) else None
    delivery_id = _ordinary_completion_delivery_id(evt)
    if delivery_id is None:
        return ""
    owner_pid, owner_started_at = _current_process_identity()
    claim_id = f"{consumer}:{owner_pid}:{uuid.uuid4().hex}"
    now = time.time()
    payload = json.dumps(_durable_event_payload(evt), sort_keys=True)
    with _DB_LOCK, _transaction() as conn:
        _migrate_ordinary_completion_legacy_row(conn, evt, delivery_id, payload, now)
        conn.execute(
            """INSERT OR IGNORE INTO ordinary_completion_deliveries
               (delivery_id, event_json, delivery_state, updated_at)
               VALUES (?, ?, 'pending', ?)""",
            (delivery_id, payload, now),
        )
        cur = conn.execute(
            """UPDATE ordinary_completion_deliveries
               SET delivery_state='effect_started', delivery_claim=?,
                   delivery_claimed_at=?, delivery_owner_pid=?,
                   delivery_owner_started_at=?,
                   delivery_attempts=delivery_attempts+1, updated_at=?
               WHERE delivery_id=? AND delivery_state='pending'""",
            (
                claim_id,
                now,
                owner_pid,
                owner_started_at,
                now,
                delivery_id,
            ),
        )
    return claim_id if cur.rowcount == 1 else None


def release_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Release a failed delivery claim so another consumer may retry.

    Attempts are counted at claim time, so a row that keeps being claimed and
    released has burned real delivery attempts. Once the budget is exhausted
    the row converges to a terminal ``dropped`` state instead of returning to
    ``pending`` — otherwise an undeliverable completion replays on every
    gateway restart forever (restore_undelivered_completions only restores
    pending rows).
    """
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        capped = conn.execute(
            """UPDATE async_delegations SET delivery_state='dropped',
                      delivery_claim=NULL, delivery_claimed_at=NULL,
                      owner_pid=NULL, owner_started_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='effect_started'
                 AND delivery_claim=? AND delivery_attempts>=?""",
            (now, delegation_id, claim_id, _MAX_DELIVERY_ATTEMPTS),
        )
        if capped.rowcount == 1:
            logger.warning(
                "Async delegation %s exhausted its %d delivery attempts; "
                "marking terminally dropped (result remains queryable).",
                delegation_id, _MAX_DELIVERY_ATTEMPTS,
            )
            return True
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='pending', delivery_claim=NULL,
                      delivery_claimed_at=NULL, owner_pid=NULL,
                      owner_started_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='effect_started'
                 AND delivery_claim=?""",
            (now, delegation_id, claim_id),
        )
        return cur.rowcount == 1


def drop_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Terminally drop a claimed completion that can never be delivered.

    Used when the delivery target is permanently gone — the spawning session
    ended at an explicit user boundary (/new, reset) rather than a compression
    rotation. Marking the row ``dropped`` (not ``delivered``) keeps the ack
    honest, and (not ``pending``) keeps restart recovery from replaying a
    completion that will be fail-closed dropped again every time.
    """
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='dropped',
                      updated_at=?, delivery_claim=NULL, delivery_claimed_at=NULL,
                      owner_pid=NULL, owner_started_at=NULL
               WHERE delegation_id=? AND delivery_state='effect_started'
                 AND delivery_claim=?""",
            (now, delegation_id, claim_id),
        )
        return cur.rowcount == 1


def mark_completion_delivery_recovery(
    evt: Dict[str, Any], claim_id: str, reason: str
) -> bool:
    """Durably stop replaying an effect whose suffix lost commit authority."""
    if not claim_id:
        return False
    now = time.time()
    if evt.get("type") == "async_delegation":
        delegation_id = str(evt.get("delegation_id") or "")
        if not delegation_id:
            return False
        with _DB_LOCK, _transaction() as conn:
            recovery_state = f"recovery_{reason}"
            cur = conn.execute(
                """UPDATE async_delegations SET delivery_state=?, delivery_claim=NULL,
                          delivery_claimed_at=NULL, owner_pid=NULL,
                          owner_started_at=NULL, updated_at=?
                   WHERE delegation_id=? AND delivery_state='effect_started'
                     AND delivery_claim=?""",
                (recovery_state, now, delegation_id, claim_id),
            )
            if cur.rowcount == 1:
                return True
            row = conn.execute(
                "SELECT delivery_state FROM async_delegations WHERE delegation_id=?",
                (delegation_id,),
            ).fetchone()
            if row is not None:
                return row[0] == "delivered" or str(row[0]).startswith("recovery_")
            cur = conn.execute(
                """INSERT OR IGNORE INTO async_delegations
                   (delegation_id, origin_session, origin_ui_session_id, state,
                    dispatched_at, updated_at, delivery_state)
                   VALUES (?, '', '', 'tombstone', ?, ?, ?)""",
                (delegation_id, now, now, recovery_state),
            )
            return cur.rowcount == 1
    delivery_id = _ordinary_completion_delivery_id(evt)
    if delivery_id is None:
        return False
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE ordinary_completion_deliveries
               SET delivery_state=?, delivery_claim=NULL,
                   delivery_claimed_at=NULL, delivery_owner_pid=NULL,
                   delivery_owner_started_at=NULL, updated_at=?
               WHERE delivery_id=? AND delivery_state='effect_started'
                 AND delivery_claim=?""",
            (f"recovery_{reason}", now, delivery_id, claim_id),
        )
        return cur.rowcount == 1


def complete_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Acknowledge a provider-terminal, SessionDB-committed delivery claim."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered',
                      delivered_at=?, updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL, owner_pid=NULL,
                      owner_started_at=NULL
               WHERE delegation_id=? AND delivery_state='effect_started'
                 AND delivery_claim=?""",
            (now, now, delegation_id, claim_id),
        )
        return cur.rowcount == 1


def complete_event_delivery(evt: Dict[str, Any], claim_id: str) -> bool:
    if not claim_id:
        return True
    if evt.get("type") == "async_delegation":
        return complete_completion_delivery(
            str(evt.get("delegation_id") or ""), claim_id
        )
    delivery_id = _ordinary_completion_delivery_id(evt)
    if delivery_id is None:
        return True
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE ordinary_completion_deliveries
               SET delivery_state='delivered', delivered_at=?, updated_at=?,
                   delivery_claim=NULL, delivery_claimed_at=NULL,
                   delivery_owner_pid=NULL, delivery_owner_started_at=NULL
               WHERE delivery_id=? AND delivery_state='effect_started'
                 AND delivery_claim=?""",
            (now, now, delivery_id, claim_id),
        )
        return cur.rowcount == 1


def release_event_delivery(evt: Dict[str, Any], claim_id: str) -> bool:
    if not claim_id:
        return True
    if evt.get("type") == "async_delegation":
        if evt.get("_completion_delivery_legacy_async") is True:
            return True
        return release_completion_delivery(
            str(evt.get("delegation_id") or ""), claim_id
        )
    delivery_id = _ordinary_completion_delivery_id(evt)
    if delivery_id is None:
        return True
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        capped = conn.execute(
            """UPDATE ordinary_completion_deliveries
               SET delivery_state='dropped', delivery_claim=NULL,
                   delivery_claimed_at=NULL, delivery_owner_pid=NULL,
                   delivery_owner_started_at=NULL, updated_at=?
               WHERE delivery_id=? AND delivery_state='effect_started'
                 AND delivery_claim=? AND delivery_attempts>=?""",
            (now, delivery_id, claim_id, _MAX_DELIVERY_ATTEMPTS),
        )
        if capped.rowcount == 1:
            return True
        cur = conn.execute(
            """UPDATE ordinary_completion_deliveries
               SET delivery_state='pending', delivery_claim=NULL,
                   delivery_claimed_at=NULL, delivery_owner_pid=NULL,
                   delivery_owner_started_at=NULL, updated_at=?
               WHERE delivery_id=? AND delivery_state='effect_started'
                 AND delivery_claim=?""",
            (now, delivery_id, claim_id),
        )
        return cur.rowcount == 1


def get_durable_event_delivery(evt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the ordinary completion's durable delivery receipt."""
    delivery_id = _ordinary_completion_delivery_id(evt)
    if delivery_id is None:
        return None
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            """SELECT delivery_state, delivery_attempts, delivered_at,
                      delivery_claim, event_json
               FROM ordinary_completion_deliveries WHERE delivery_id=?""",
            (delivery_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "delivery_id": delivery_id,
        "delivery_state": row[0],
        "delivery_attempts": row[1],
        "delivered_at": row[2],
        "delivery_claim": row[3],
        "event": _load_durable_event_payload(row[4]),
    }


def get_durable_delegation(delegation_id: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            """SELECT origin_session, state, dispatched_at, completed_at,
                      result_json, delivery_state, delivery_attempts,
                      origin_session_id
               FROM async_delegations WHERE delegation_id=?""", (delegation_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "delegation_id": delegation_id, "origin_session": row[0], "state": row[1],
        "dispatched_at": row[2], "completed_at": row[3],
        "result": json.loads(row[4]) if row[4] else None,
        "delivery_state": row[5], "delivery_attempts": row[6],
        "origin_session_id": row[7] or "",
    }


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    """Lazily create (or grow) the shared daemon executor.

    We never shrink — ThreadPoolExecutor can't resize — but if the configured
    cap grows between calls we rebuild a larger pool. Existing in-flight
    futures keep running on the old pool until it's garbage collected.
    """
    global _admission_cap, _executor, _executor_max_workers
    with _executor_lock:
        if _executor is None or max_workers > _executor_max_workers:
            # Daemon threads: thread_name_prefix aids debugging in stack dumps.
            _executor = _DaemonThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="async-delegate",
            )
            _executor_max_workers = max_workers
        with _admission_condition:
            _admission_cap = max_workers
            _admission_condition.notify_all()
        return _executor


def _admit_worker(delegation_id: str) -> bool:
    """Wait off-thread until this queued record owns a process-wide slot."""
    with _admission_condition:
        while True:
            record = _records.get(delegation_id)
            if record is None or record.get("status") != "queued":
                _pending_admission_ids.discard(delegation_id)
                return False
            running = sum(
                1
                for current in _records.values()
                if current.get("status") in ("running", "stalling")
            )
            if running < _admission_cap:
                _pending_admission_ids.discard(delegation_id)
                record["status"] = "running"
                record["_progress_ts"] = time.time()
                # The runner owns cleanup from this point onward.
                record.pop("_on_not_started", None)
                return True
            _admission_condition.wait()


def _try_register_queued(record: Dict[str, Any]) -> bool:
    """Atomically reserve and durably register one backlog slot for ``record``."""
    delegation_id = record["delegation_id"]
    with _records_lock:
        running = sum(
            1
            for current in _records.values()
            if current.get("status") in ("running", "stalling")
        )
        available_runner_slots = max(0, _admission_cap - running)
        if len(_pending_admission_ids) >= _admission_cap + available_runner_slots:
            return False
        _persist_dispatch(record)
        _pending_admission_ids.add(delegation_id)
        _records[delegation_id] = record
        return True


def _invoke_not_started(callback: Optional[Callable[[str], None]], status: str) -> None:
    """Finalize caller-owned resources when no runner took ownership."""
    if not callable(callback):
        return
    try:
        callback(status)
    except Exception:
        logger.exception("Async delegation pre-admission finalization failed")


def active_count() -> int:
    """Number of queued or running async delegation units.

    A unit is one dispatch: a single subagent OR a whole fan-out batch. A batch
    counts as ONE here because it occupies one async-pool slot (the capacity
    semantics ``dispatch_async_delegation_batch`` relies on). For the count of
    actual concurrent child subagents (batch expanded), use
    ``active_task_count()``.
    """
    with _records_lock:
        return sum(
            1 for r in _records.values()
            if r.get("status") in _ACTIVE_STATUSES
        )


def active_for_session(origin_ui_session_id: str) -> int:
    """Number of live async delegations owned by one UI session."""
    if not origin_ui_session_id:
        return 0
    with _records_lock:
        return sum(
            1
            for r in _records.values()
            if r.get("status") in _ACTIVE_STATUSES
            and str(r.get("origin_ui_session_id") or "")
            == origin_ui_session_id
        )


def active_task_count() -> int:
    """Number of async delegation TASKS (child subagents) currently running.

    Unlike ``active_count()`` (units/slots), this expands a batch to its child
    count: a running batch of N tasks contributes N, a single subagent
    contributes 1. This is the truthful "how many subagents are actually
    working right now" figure for observability, where a 3-task batch shown as
    "1" undercounts real concurrent work. Falls back to counting a batch as 1
    if its goal list is missing.
    """
    with _records_lock:
        total = 0
        for r in _records.values():
            if r.get("status") not in {"running", "finalizing"}:
                continue
            if r.get("is_batch"):
                goals = r.get("goals")
                total += len(goals) if isinstance(goals, (list, tuple)) and goals else 1
            else:
                total += 1
        return total


def _matches_session_selectors(
    record: Dict[str, Any],
    *,
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
) -> bool:
    """Authorize lifecycle control only by the spawning parent principal."""
    del session_key, origin_ui_session_id
    record_parent = str(record.get("parent_session_id") or "")
    return bool(parent_session_id and record_parent == str(parent_session_id))


def has_live_for_session(
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
) -> bool:
    """Whether a session still owns any live async delegation.

    Live states are defined by ``_ACTIVE_STATUSES``.
    """
    if not parent_session_id:
        return False
    with _records_lock:
        return any(
            r.get("status") in _ACTIVE_STATUSES
            and _matches_session_selectors(
                r,
                session_key=session_key,
                origin_ui_session_id=origin_ui_session_id,
                parent_session_id=parent_session_id,
            )
            for r in _records.values()
        )


def _new_delegation_id() -> str:
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _prune_completed_locked() -> None:
    """Drop the oldest completed records beyond the retention cap.

    Caller must hold ``_records_lock``.
    """
    completed = [
        (rid, r)
        for rid, r in _records.items()
        if r.get("status") not in _ACTIVE_STATUSES
    ]
    if len(completed) <= _MAX_RETAINED_COMPLETED:
        return
    # Oldest-first by completion time (fall back to dispatch time).
    completed.sort(key=lambda kv: kv[1].get("completed_at") or kv[1].get("dispatched_at") or 0)
    for rid, _ in completed[: len(completed) - _MAX_RETAINED_COMPLETED]:
        _records.pop(rid, None)


def _current_origin_session_id() -> str:
    """Raw session id of the ORIGINATING api_server request, or ``""``.

    The obvious source — ``HERMES_SESSION_ID`` via ``get_session_env`` — is
    NOT safe to read at dispatch time: constructing a child agent
    (``agent/agent_init.py``) calls ``set_current_session_id(child.session_id)``,
    clobbering that ContextVar *and* ``os.environ`` with the subagent's
    internal ``{timestamp}_{uuid}`` id moments before the dispatch code reads
    it, so the completion wake would self-post into the subagent's own
    (unread) session instead of the spawner's.

    The request-scoped ``HERMES_SESSION_CHAT_ID`` binding survives child
    construction: ``_bind_api_server_session`` binds ``chat_id`` to the raw
    ``X-Hermes-Session-Id``, and its only writer is ``set_session_vars`` —
    ``set_current_session_id`` never touches it. Gate on the platform: on
    push platforms ``chat_id`` is a chat, not a session, so yield ``""``
    there.
    """
    try:
        from gateway.session_context import get_session_env

        if get_session_env("HERMES_SESSION_PLATFORM", "") != "api_server":
            return ""
        return get_session_env("HERMES_SESSION_CHAT_ID", "") or ""
    except Exception:
        return ""


def dispatch_async_delegation(
    *,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    parent_session_id: Optional[str] = None,
    runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "",
    origin_session_id: str = "",
    interrupt_fn: Optional[Callable[[], None]] = None,
    on_not_started: Optional[Callable[[str], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    progress_fn: Optional[Callable[[], tuple]] = None,
) -> Dict[str, Any]:
    """Spawn ``runner`` on the daemon executor and return a handle immediately.

    Parameters
    ----------
    goal, context, toolsets, role, model
        The dispatch-time task spec, captured verbatim for the rich
        completion block.
    session_key
        The gateway session_key (from ``tools.approval.get_current_session_key``)
        captured on the parent thread BEFORE dispatch, because the daemon
        worker thread won't carry the contextvar. Used to route the
        completion back to the originating session.
    parent_session_id
        The durable ``state.db`` session id of the parent agent that spawned
        the delegation. Carried on the completion event so the gateway can
        pin routing to the spawning session instead of recovering the latest
        ``ended_at IS NULL`` row for the peer tuple (#57498).
    runner
        Zero-arg callable that builds + runs the child and returns the same
        result dict ``_run_single_child`` produces. Runs on the worker thread.
    interrupt_fn
        Optional callable to signal the child to stop (used on shutdown /
        explicit cancel).
    on_not_started
        Optional callable invoked exactly once with the terminal status when
        registration, submission, or queued cancellation ends the delegation
        before its runner owns cleanup.
    progress_fn
        Optional zero-arg callable returning ``(token, in_tool)`` where
        ``token`` is any comparable snapshot of the child's progress (api
        call count + current tool) and ``in_tool`` says whether the child is
        currently inside a tool call. Sampled by the stale monitor; a frozen
        token past the stale threshold marks the delegation stuck (see the
        stale-detection block at the top of this module). When omitted, the
        delegation is not monitored.
    max_async_children
        Maximum number of runners admitted at once and queued registrations
        retained process-wide. Excess registrations are rejected immediately.

    Returns
    -------
    dict
        ``{"status": "dispatched", "delegation_id": ...}`` on success.
    """
    from tools.runtime_heartbeat import preflight_current_heartbeat

    heartbeat_interval = preflight_current_heartbeat()
    delegation_id = _new_delegation_id()
    dispatched_at = time.time()
    record: Dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": goal,
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "role": role,
        "model": model,
        "session_key": session_key,
        "origin_ui_session_id": origin_ui_session_id,
        "origin_session_id": origin_session_id,
        "parent_session_id": parent_session_id,
        "status": "queued",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "_on_not_started": on_not_started,
        "progress_fn": progress_fn,
        # Stale-monitor bookkeeping (see _stale_monitor_loop).
        "_progress_token": None,
        "_progress_ts": dispatched_at,
        "_interrupted_at": None,
    }
    executor = _get_executor(max_async_children)
    if not _try_register_queued(record):
        _invoke_not_started(on_not_started, "error")
        return {
            "status": "rejected",
            "error": _BACKLOG_FULL_ERROR,
        }

    def _worker() -> None:
        if not _admit_worker(delegation_id):
            return
        if progress_fn is not None:
            _ensure_stale_monitor()
        result: Dict[str, Any] = {}
        status = "error"
        try:
            result = runner() or {}
            status = result.get("status") or "completed"
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception("Async delegation %s crashed", delegation_id)
            result = {
                "status": "error",
                "summary": None,
                "error": f"{type(exc).__name__}: {exc}",
                "api_calls": 0,
                "duration_seconds": round(time.time() - dispatched_at, 2),
            }
            status = "error"
        finally:
            _finalize(delegation_id, result, status)

    from tools.runtime_heartbeat import inspect_delegation, runtime_heartbeat

    submit_error = None
    with _records_lock:
        try:
            # Keep heartbeat installation atomic with the worker's terminal
            # claim. A fast worker may finish before submit() returns, but it
            # cannot cancel until arm() has installed the target.
            executor.submit(propagate_context_to_thread(_worker))
        except Exception as exc:  # pragma: no cover — pool submit failure is rare
            submit_error = exc
        else:
            runtime_heartbeat.arm(
                delegation_id,
                caller_id=session_key,
                kind="delegation",
                interval=heartbeat_interval,
                inspect=lambda _id=delegation_id: inspect_delegation(_id),
            )
    if submit_error is not None:
        with _records_lock:
            _records.pop(delegation_id, None)
            _pending_admission_ids.discard(delegation_id)
        _delete_durable_delegation(delegation_id)
        _invoke_not_started(on_not_started, "error")
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation: {submit_error}",
        }
    if progress_fn is not None:
        _ensure_stale_monitor()

    logger.info(
        "Dispatched async delegation %s (session_key=%s): %s",
        delegation_id, session_key or "<cli>", (goal or "")[:80],
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize(delegation_id: str, result: Dict[str, Any], status: str) -> None:
    """Mark a record complete and push the completion event onto the queue."""
    claimed = _begin_finalization(delegation_id)
    if claimed is None:
        return
    event_record, _interrupt_fn, on_not_started = claimed

    if event_record.get("_stall_grace_expired"):
        completed_at = event_record.get("completed_at") or time.time()
        duration = round(
            completed_at - (event_record.get("dispatched_at") or completed_at), 2
        )
        stall_in_tool = event_record.get("_stall_in_tool")
        status = "stalled"
        result = {
            "status": "stalled",
            "summary": None,
            "error": (
                f"Async delegation {delegation_id} stalled and only returned "
                "after its interruption grace window expired."
            ),
            "api_calls": 0,
            "duration_seconds": duration,
            "exit_reason": "stalled",
            "stalled_after_quiet_seconds": event_record.get("_stall_quiet_seconds"),
            "stall_threshold_seconds": event_record.get("_stall_threshold_seconds"),
            "stall_phase": "in_tool" if stall_in_tool else "idle",
            "stall_grace_seconds": _STALL_GRACE_SECONDS,
        }

    _invoke_not_started(on_not_started, status)
    _push_completion_event(event_record, result, status)
    _finish_finalization(delegation_id, status)


def _begin_finalization(
    delegation_id: str,
) -> Optional[
    tuple[
        Dict[str, Any],
        Optional[Callable[[], None]],
        Optional[Callable[[str], None]],
    ]
]:
    """Atomically claim terminal delivery while keeping the record active."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or record.get("status") not in (
            "queued",
            "running",
            "stalling",
            "cancelling",
        ):
            return
        # Stay active until durable persistence and queue publication finish;
        # otherwise process shutdown can kill this daemon worker in the narrow
        # gap after status flips but before SQLite is committed.
        record["status"] = "finalizing"
        record["completed_at"] = time.time()
        interrupt_fn = record.get("interrupt_fn")
        record["interrupt_fn"] = None  # drop the closure; child is done
        record["progress_fn"] = None  # stop stale-monitor sampling
        on_not_started = record.pop("_on_not_started", None)
        event_record = dict(record)
        _admission_condition.notify_all()

    try:
        from tools.runtime_heartbeat import runtime_heartbeat

        runtime_heartbeat.cancel(delegation_id)
    except Exception:
        logger.debug(
            "Failed to cancel heartbeat for delegation %s",
            delegation_id,
            exc_info=True,
        )

    return event_record, interrupt_fn, on_not_started


def _finish_finalization(delegation_id: str, status: str) -> None:
    with _records_lock:
        record = _records.get(delegation_id)
        if record is not None:
            record["status"] = status
        _prune_completed_locked()


def _push_completion_event(
    record: Dict[str, Any], result: Dict[str, Any], status: str
) -> None:
    """Push a type='async_delegation' event onto the shared completion queue.

    Best-effort: a failure here must not crash the worker, but it WOULD mean a
    silently-lost result, so we log loudly.
    """
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s finished but process_registry import failed; "
            "result lost: %s",
            record.get("delegation_id"), exc,
        )
        return

    summary = result.get("summary")
    error = result.get("error")
    dispatched_at = record.get("dispatched_at") or time.time()
    completed_at = record.get("completed_at") or time.time()

    evt = {
        "type": "async_delegation",
        "delegation_id": record.get("delegation_id"),
        # session_key routes the completion back to the originating gateway
        # session; empty string => CLI (single-session) path.
        "session_key": record.get("session_key", ""),
        "origin_ui_session_id": record.get("origin_ui_session_id", ""),
        "origin_session_id": record.get("origin_session_id", ""),
        "parent_session_id": record.get("parent_session_id"),
        "goal": record.get("goal", ""),
        "context": record.get("context"),
        "toolsets": record.get("toolsets"),
        "role": record.get("role"),
        "model": result.get("model") or record.get("model"),
        "status": status,
        "summary": summary,
        "error": error,
        "api_calls": result.get("api_calls", 0),
        "duration_seconds": result.get(
            "duration_seconds", round(completed_at - dispatched_at, 2)
        ),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
        "exit_reason": result.get("exit_reason"),
    }
    # Structured stall metadata (#51690) — additive, present only on
    # stall-monitor finalizations.
    for _k in (
        "stalled_after_quiet_seconds",
        "stall_threshold_seconds",
        "stall_phase",
        "stall_grace_seconds",
    ):
        if _k in result:
            evt[_k] = result[_k]
    _persist_completion(evt, result)
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s: failed to enqueue completion event; "
            "result lost: %s",
            record.get("delegation_id"), exc,
        )


def dispatch_async_delegation_batch(
    *,
    goals: List[str],
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    parent_session_id: Optional[str] = None,
    runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "",
    origin_session_id: str = "",
    interrupt_fn: Optional[Callable[[], None]] = None,
    on_not_started: Optional[Callable[[str], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    delegation_id: Optional[str] = None,
    progress_fn: Optional[Callable[[], tuple]] = None,
) -> Dict[str, Any]:
    """Dispatch a WHOLE fan-out batch as ONE background unit.

    Unlike ``dispatch_async_delegation`` (which backs a single subagent),
    ``runner`` here runs the entire batch — it builds and joins on every child
    in parallel and returns the combined ``{"results": [...],
    "total_duration_seconds": N}`` dict that the synchronous path would have
    returned. We occupy ONE async slot for the whole batch (the in-batch
    parallelism is bounded separately by ``max_concurrent_children``), so a
    single ``delegate_task`` fan-out never exhausts the async pool by itself.

    When the batch finishes, a SINGLE completion event is pushed onto the
    shared ``process_registry.completion_queue`` carrying the full per-task
    ``results`` list, so the consolidated summaries re-enter the conversation
    as one message once every child is done — the chat is never blocked while
    they run.

    Returns ``{"status": "dispatched", "delegation_id": ...}`` after the
    batch is registered. Excess process-wide backlog is rejected immediately;
    runner admission remains off the foreground thread. If no runner takes
    ownership, ``on_not_started`` receives the terminal status exactly once.
    """
    from tools.runtime_heartbeat import preflight_current_heartbeat

    heartbeat_interval = preflight_current_heartbeat()
    delegation_id = delegation_id or _new_delegation_id()
    dispatched_at = time.time()
    n = len(goals)
    # A combined goal label for status listings / the completion header.
    combined_goal = (
        goals[0] if n == 1 else f"{n} parallel subagents: " + "; ".join(g[:40] for g in goals)
    )
    record: Dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": combined_goal,
        "goals": list(goals),
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "role": role,
        "model": model,
        "session_key": session_key,
        "origin_ui_session_id": origin_ui_session_id,
        "origin_session_id": origin_session_id,
        "parent_session_id": parent_session_id,
        "status": "queued",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "_on_not_started": on_not_started,
        "is_batch": True,
        "progress_fn": progress_fn,
        "_progress_token": None,
        "_progress_ts": dispatched_at,
        "_interrupted_at": None,
    }
    executor = _get_executor(max_async_children)
    if not _try_register_queued(record):
        _invoke_not_started(on_not_started, "error")
        return {
            "status": "rejected",
            "error": _BACKLOG_FULL_ERROR,
        }

    def _worker() -> None:
        if not _admit_worker(delegation_id):
            return
        if progress_fn is not None:
            _ensure_stale_monitor()
        combined: Dict[str, Any] = {}
        status = "error"
        try:
            combined = runner() or {}
            # Batch status: completed unless every child errored/was interrupted.
            child_results = combined.get("results") or []
            if child_results and all(
                (r.get("status") not in ("completed", "success"))
                for r in child_results
            ):
                status = "error"
            else:
                status = "completed"
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception("Async delegation batch %s crashed", delegation_id)
            combined = {
                "results": [],
                "error": f"{type(exc).__name__}: {exc}",
                "total_duration_seconds": round(time.time() - dispatched_at, 2),
            }
            status = "error"
        finally:
            _finalize_batch(delegation_id, combined, status)

    from tools.runtime_heartbeat import inspect_delegation, runtime_heartbeat

    submit_error = None
    with _records_lock:
        try:
            # Match the single-child path: terminal claim cannot race ahead
            # of heartbeat installation.
            executor.submit(propagate_context_to_thread(_worker))
        except Exception as exc:  # pragma: no cover
            submit_error = exc
        else:
            runtime_heartbeat.arm(
                delegation_id,
                caller_id=session_key,
                kind="delegation",
                interval=heartbeat_interval,
                inspect=lambda _id=delegation_id: inspect_delegation(_id),
            )
    if submit_error is not None:
        with _records_lock:
            _records.pop(delegation_id, None)
            _pending_admission_ids.discard(delegation_id)
        _delete_durable_delegation(delegation_id)
        _invoke_not_started(on_not_started, "error")
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation batch: {submit_error}",
        }
    if progress_fn is not None:
        _ensure_stale_monitor()

    logger.info(
        "Dispatched async delegation batch %s (%d task(s), session_key=%s)",
        delegation_id, n, session_key or "<cli>",
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize_batch(
    delegation_id: str, combined: Dict[str, Any], status: str
) -> None:
    """Mark a batch record complete and push ONE combined completion event."""
    claimed = _begin_finalization(delegation_id)
    if claimed is None:
        return
    event_record, _interrupt_fn, on_not_started = claimed

    _invoke_not_started(on_not_started, status)
    _push_batch_completion_event(event_record, combined, status)
    _finish_finalization(delegation_id, status)


def _push_batch_completion_event(
    event_record: Dict[str, Any], combined: Dict[str, Any], status: str
) -> None:
    """Push a combined async-delegation batch completion event."""
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s finished but process_registry import "
            "failed; result lost: %s",
            event_record.get("delegation_id"), exc,
        )
        return

    dispatched_at = event_record.get("dispatched_at") or time.time()
    completed_at = event_record.get("completed_at") or time.time()
    evt = {
        "type": "async_delegation",
        "delegation_id": event_record.get("delegation_id"),
        "session_key": event_record.get("session_key", ""),
        "origin_ui_session_id": event_record.get("origin_ui_session_id", ""),
        "origin_session_id": event_record.get("origin_session_id", ""),
        "parent_session_id": event_record.get("parent_session_id"),
        "goal": event_record.get("goal", ""),
        "goals": event_record.get("goals"),
        "context": event_record.get("context"),
        "toolsets": event_record.get("toolsets"),
        "role": event_record.get("role"),
        "model": event_record.get("model"),
        "status": status,
        "is_batch": True,
        # The full per-task results list — the formatter renders a
        # consolidated multi-task block from this.
        "results": combined.get("results") or [],
        # Per-task live transcript log paths (cache/delegation/live/...).
        # They persist after completion and double as the full-fidelity
        # operational record of each child's run.
        "live_transcripts": combined.get("live_transcripts"),
        "error": combined.get("error"),
        "total_duration_seconds": combined.get("total_duration_seconds"),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
    }
    # Structured stall metadata (#51690) — additive, present only on
    # stall-monitor finalizations.
    for _k in (
        "stalled_after_quiet_seconds",
        "stall_threshold_seconds",
        "stall_phase",
        "stall_grace_seconds",
    ):
        if _k in combined:
            evt[_k] = combined[_k]
    _persist_completion(evt, combined)
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s: failed to enqueue completion event; "
            "result lost: %s",
            event_record.get("delegation_id"), exc,
        )


def _ensure_stale_monitor() -> None:
    """Start (once) the module-level stale-delegation monitor thread.

    One daemon thread serves every dispatch; it exits on its own when no
    monitorable records remain, and is restarted by the next dispatch that
    carries a ``progress_fn``.
    """
    global _monitor_thread
    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return
        _monitor_stop.clear()
        _monitor_thread = threading.Thread(
            target=_stale_monitor_loop,
            name="async-delegate-stale-monitor",
            daemon=True,
        )
        _monitor_thread.start()


def _stale_monitor_loop() -> None:
    """Sweep running delegations for stalled progress.

    Per sweep, for every running record with a ``progress_fn``:

    - Sample ``(token, in_tool)``. A changed token refreshes the record's
      progress timestamp — a child that keeps advancing is never touched, no
      matter how long it runs.
    - A frozen token past the idle/in-tool threshold marks the record
      ``stalling``: we call ``interrupt_fn`` so a responsive-but-slow child
      can unwind and deliver its (partial) result through the normal
      ``_finalize`` path with full fidelity.
    - A ``stalling`` record whose runner still hasn't returned after the
      grace window is force-finalized with one terminal ``stalled`` event so
      the owning session hears an outcome and the async slot frees. A late
      runner return after that is ignored by ``_begin_finalization``.
    """
    while not _monitor_stop.wait(_STALE_CHECK_INTERVAL):
        now = time.time()
        stalled: List[tuple] = []  # (delegation_id, is_batch, quiet_for, in_tool)
        expired: List[str] = []  # stalling past grace → force-finalize
        any_monitorable = False
        with _records_lock:
            for record in _records.values():
                status = record.get("status")
                if status == "stalling":
                    any_monitorable = True
                    interrupted_at = record.get("_interrupted_at") or now
                    if now - interrupted_at >= _STALL_GRACE_SECONDS:
                        expired.append(record["delegation_id"])
                    continue
                if status != "running":
                    continue
                progress_fn = record.get("progress_fn")
                if progress_fn is None:
                    continue
                any_monitorable = True
                try:
                    token, in_tool = progress_fn()
                except Exception:
                    # An unreadable child must not look permanently healthy —
                    # keep the last timestamp running instead of refreshing it.
                    token, in_tool = record.get("_progress_token"), False
                if token != record.get("_progress_token"):
                    record["_progress_token"] = token
                    record["_progress_ts"] = now
                    continue
                quiet_for = now - (record.get("_progress_ts") or now)
                limit = (
                    _STALE_IN_TOOL_SECONDS if in_tool else _STALE_IDLE_SECONDS
                )
                if quiet_for >= limit:
                    record["status"] = "stalling"
                    record["_interrupted_at"] = now
                    # Structured stall context for the terminal event and
                    # status listings (#51690): how long progress was frozen,
                    # which threshold applied, and whether the child was
                    # inside a tool when it went quiet.
                    record["_stall_quiet_seconds"] = round(quiet_for, 2)
                    record["_stall_threshold_seconds"] = limit
                    record["_stall_in_tool"] = bool(in_tool)
                    stalled.append(
                        (
                            record["delegation_id"],
                            bool(record.get("is_batch")),
                            quiet_for,
                            in_tool,
                        )
                    )
        for delegation_id, _is_batch, quiet_for, in_tool in stalled:
            logger.warning(
                "Async delegation %s made no progress for %.0fs "
                "(in_tool=%s) — interrupting; grace window %.0fs",
                delegation_id, quiet_for, in_tool, _STALL_GRACE_SECONDS,
            )
            with _records_lock:
                record = _records.get(delegation_id)
                fn = record.get("interrupt_fn") if record else None
            if callable(fn):
                try:
                    fn()
                except Exception as exc:
                    logger.debug(
                        "Async delegation %s stall interrupt failed: %s",
                        delegation_id, exc,
                    )
        for delegation_id in expired:
            _finalize_stalled(delegation_id)
        if not any_monitorable:
            return


def _finalize_stalled(delegation_id: str) -> None:
    """Record stall grace expiry without freeing a live worker's capacity."""
    with _records_lock:
        record = _records.get(delegation_id)
        if (
            record is None
            or record.get("status") != "stalling"
            or record.get("_stall_grace_expired")
        ):
            return
        record["_stall_grace_expired"] = True
    logger.error(
        "Async delegation %s ignored the stall interrupt; retaining its "
        "capacity slot until the worker exits",
        delegation_id,
    )

def _children_activity_from_token(token: Any, now: float) -> Optional[List]:
    """Parse a progress token into per-child activity dicts (best-effort).

    delegate_tool's ``_batch_progress`` emits one ``(api_call_count,
    current_tool, last_activity_ts)`` tuple per child. Foreign token shapes
    (custom dispatchers) degrade to ``None`` entries rather than raising —
    the token contract is intentionally opaque to the registry.
    """
    try:
        parts = list(token)
    except TypeError:
        return None
    out: List[Optional[Dict[str, Any]]] = []
    for part in parts:
        if isinstance(part, (list, tuple)) and len(part) >= 2:
            entry: Dict[str, Any] = {
                "api_calls": part[0],
                "current_tool": part[1],
            }
            if len(part) >= 3 and isinstance(part[2], (int, float)):
                entry["seconds_since_activity"] = round(
                    max(0.0, now - float(part[2])), 1
                )
            out.append(entry)
        else:
            out.append(None)
    return out


def list_async_delegations() -> List[Dict[str, Any]]:
    """Snapshot of async delegations (running + recently completed).

    Safe to call from any thread. Excludes the non-serialisable callables
    and private monitor bookkeeping, but exposes computed live-status
    fields for UIs (#51690):

    - ``seconds_since_progress``: how long the stale monitor has seen a
      frozen progress token (running/stalling records).
    - ``children_activity``: per-child ``{api_calls, current_tool,
      seconds_since_activity}`` sampled live from the dispatch's
      ``progress_fn``.
    - ``stalled_after_quiet_seconds`` / ``stall_threshold_seconds`` /
      ``stall_in_tool``: stall context once the monitor has tripped.
    """
    now = time.time()
    samplers: Dict[str, Callable] = {}
    with _records_lock:
        items = []
        for r in _records.values():
            item = {
                k: v
                for k, v in r.items()
                if k not in {"interrupt_fn", "progress_fn"}
                and not k.startswith("_")
            }
            status = r.get("status")
            if status in ("running", "stalling"):
                ts = r.get("_progress_ts")
                if ts:
                    item["seconds_since_progress"] = round(now - ts, 1)
                fn = r.get("progress_fn")
                if callable(fn):
                    samplers[r["delegation_id"]] = fn
            if status in ("stalling", "stalled"):
                for src, dst in (
                    ("_stall_quiet_seconds", "stalled_after_quiet_seconds"),
                    ("_stall_threshold_seconds", "stall_threshold_seconds"),
                    ("_stall_in_tool", "stall_in_tool"),
                ):
                    if r.get(src) is not None:
                        item[dst] = r.get(src)
            items.append(item)

    # Sample live activity OUTSIDE the lock — progress_fn reads child-agent
    # attributes and must never run under _records_lock (a slow or broken
    # sampler would block every dispatch/finalize in the process).
    for item in items:
        fn = samplers.get(item.get("delegation_id"))
        if fn is None:
            continue
        try:
            token, in_tool = fn()
        except Exception:
            continue
        activity = _children_activity_from_token(token, now)
        if activity is not None:
            item["children_activity"] = activity
        item["in_tool"] = bool(in_tool)
    return items


def list_async_delegations_for_owner(*, parent_session_id: str) -> List[Dict[str, Any]]:
    """Return the calling parent's delegation status without task payloads."""
    if not parent_session_id:
        return []
    allowed = {
        "delegation_id", "status", "dispatched_at", "completed_at", "is_batch",
        "seconds_since_progress", "children_activity", "in_tool",
        "stalled_after_quiet_seconds", "stall_threshold_seconds", "stall_in_tool",
    }
    return [
        {key: value for key, value in item.items() if key in allowed}
        for item in list_async_delegations()
        if _matches_session_selectors(item, parent_session_id=parent_session_id)
    ]


def interrupt_all(reason: str = "shutdown") -> int:
    """Cancel every queued or running async delegation. Returns how many.

    Used on ``/stop`` and gateway shutdown so a dangling background subagent
    can't keep burning tokens with no one listening. The child still emits a
    completion event (status='interrupted') via the normal finalize path.
    """
    count = 0
    with _records_lock:
        targets = [
            r["delegation_id"] for r in _records.values()
            if r.get("status") in ("queued", "running", "stalling")
        ]
        targets.sort(key=lambda delegation_id: _records[delegation_id]["status"] != "queued")
    for delegation_id in targets:
        count += int(_interrupt_record(delegation_id, reason))
    if count:
        logger.info("Interrupted %d async delegation(s) (%s)", count, reason)
    return count


def _interrupt_record(delegation_id: str, reason: str) -> bool:
    """Interrupt a running unit or finalize a queued unit without starting it."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or record.get("status") not in (
            "queued",
            "running",
            "stalling",
        ):
            return False
        queued = record.get("status") == "queued"
        if queued:
            record["status"] = "cancelling"
            _admission_condition.notify_all()
        interrupt_fn = record.get("interrupt_fn")
        is_batch = bool(record.get("is_batch"))

    try:
        from tools.runtime_heartbeat import runtime_heartbeat

        runtime_heartbeat.cancel(delegation_id)
    except Exception:
        logger.debug("Could not cancel delegation heartbeat", exc_info=True)

    if callable(interrupt_fn):
        try:
            interrupt_fn()
        except Exception as exc:
            logger.debug(
                "interrupt %s failed for async delegation %s: %s",
                reason,
                delegation_id,
                exc,
            )
            if not queued:
                return False
    elif not queued:
        return False

    if queued:
        error = f"Async delegation cancelled before worker admission ({reason})."
        if is_batch:
            _finalize_batch(
                delegation_id,
                {"results": [], "error": error, "total_duration_seconds": 0},
                "interrupted",
            )
        else:
            _finalize(
                delegation_id,
                {
                    "status": "interrupted",
                    "summary": None,
                    "error": error,
                    "api_calls": 0,
                    "duration_seconds": 0,
                },
                "interrupted",
            )
    return True


def interrupt_for_session(
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
    reason: str = "session_end",
) -> int:
    """Cancel queued or running async delegations owned by ONE session.

    A delegation's lifecycle is bound to the session that spawned it: when
    that session ends, its in-flight background subagents must end with it —
    a completed orphan would otherwise sit on the shared completion queue
    with no live owner, either leaking into another chat or burning tokens
    with no one listening (#55578).

    Selectors (any matching field claims the record):
    - ``origin_ui_session_id``: the live TUI tab/window that commissioned it.
    - ``session_key``: the durable routing key captured at dispatch.
    - ``parent_session_id``: the spawning agent's durable session-db id —
      the right selector for gateway chats, whose ``session_key`` (the
      platform conversation key) SURVIVES a ``/new`` reset while the
      session id rotates.

    Returns how many were interrupted.
    """
    if not parent_session_id:
        return 0
    count = 0
    with _records_lock:
        targets = [
            r["delegation_id"] for r in _records.values()
            if r.get("status") in ("queued", "running", "stalling")
            and _matches_session_selectors(
                r,
                session_key=session_key,
                origin_ui_session_id=origin_ui_session_id,
                parent_session_id=parent_session_id,
            )
        ]
        targets.sort(key=lambda delegation_id: _records[delegation_id]["status"] != "queued")
    for delegation_id in targets:
        count += int(_interrupt_record(delegation_id, reason))
    if count:
        logger.info(
            "Interrupted %d async delegation(s) for ending session (%s)",
            count, reason,
        )
    return count


def _reset_for_tests() -> None:
    """Test-only: clear all state and tear down the executor + monitor."""
    global _admission_cap, _executor, _executor_max_workers, _monitor_thread
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
        _executor_max_workers = 0
    _monitor_stop.set()
    with _monitor_lock:
        thread = _monitor_thread
        _monitor_thread = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    with _admission_condition:
        _records.clear()
        _pending_admission_ids.clear()
        _admission_cap = 0
        _admission_condition.notify_all()
    try:
        from tools.runtime_heartbeat import runtime_heartbeat

        runtime_heartbeat.cancel_all()
    except Exception:
        pass
