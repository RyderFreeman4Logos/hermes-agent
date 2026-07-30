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
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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

_records_lock = threading.Lock()
# delegation_id -> record dict. Kept for the lifetime of the run plus a short
# tail after completion so `list_async_delegations()` can show recent results.
_records: Dict[str, Dict[str, Any]] = {}

_DEFAULT_MAX_ASYNC_CHILDREN = 3
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
            origin_session_id TEXT NOT NULL DEFAULT ''
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
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE async_delegations ADD COLUMN {name} {sql_type}")


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
        for key in (
            "goal", "goals", "context", "toolsets", "role", "model",
            "is_batch", "provider", "reasoning_effort",
        )
        if key in record
    }
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO async_delegations
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
        conn.execute("DELETE FROM async_delegations WHERE delegation_id=?", (delegation_id,))


def _prune_durable_records() -> None:
    """Bound terminal history, preferring delivered records for deletion."""
    now = time.time()
    cutoff = now - _DURABLE_RETENTION_SECONDS
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "DELETE FROM async_delegations WHERE delivery_state='delivered' AND updated_at < ?",
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
                     ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                              updated_at ASC LIMIT ?
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
    """Persist terminal completion. Retry once on failure; never silent-drop."""
    now = time.time()
    args = (
        event.get("status", "completed"), event.get("completed_at", now), now,
        json.dumps(event), json.dumps(result), event["delegation_id"],
    )
    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            with _DB_LOCK, _transaction() as conn:
                conn.execute(
                    """UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
                       event_json=?, result_json=?, delivery_state='pending'
                       WHERE delegation_id=?""",
                    args,
                )
            return
        except Exception as exc:  # noqa: BLE001 — must surface loudly
            last_exc = exc
            logger.error(
                "Async delegation %s: _persist_completion failed (attempt %d/2): %s",
                event.get("delegation_id"), attempt + 1, exc, exc_info=True,
            )
    if last_exc is not None:
        raise last_exc


def _note_delivery_attempt(delegation_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET delivery_attempts=delivery_attempts+1, updated_at=? WHERE delegation_id=?",
            (time.time(), delegation_id),
        )


def recover_abandoned_delegations() -> int:
    """Classify records whose owning process disappeared as outcome unknown."""
    try:
        from gateway.status import _pid_exists, get_process_start_time
    except Exception:
        return 0
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
            live = False
            if pid:
                live = _pid_exists(int(pid))
                if live and started is not None:
                    live = get_process_start_time(int(pid)) == int(started)
            if live:
                continue
            task = json.loads(task_json or "{}")
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


_TERMINAL_LIVE_STATUSES = frozenset({
    "completed", "error", "interrupted", "failed", "cancelled", "canceled", "unknown",
})
_NONTERMINAL_LIVE_STATUSES = frozenset({"running", "finalizing", "pending", "dispatched"})


def _read_live_manifest(delegation_id: str) -> Optional[Dict[str, Any]]:
    """Best-effort load of cache/delegation/live/<id>/manifest.json."""
    try:
        from tools.delegation_live_log import live_transcript_root

        mp = live_transcript_root() / delegation_id / "manifest.json"
        if not mp.is_file():
            return None
        data = json.loads(mp.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("live manifest read failed for %s: %s", delegation_id, exc)
        return None


def _task_log_has_terminal_marker(log_path: Optional[str]) -> bool:
    if not log_path:
        return False
    try:
        path = Path(log_path) if not isinstance(log_path, Path) else log_path
        if not path.is_file():
            return False
        # Read a small tail — markers are written at the end of the log.
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            tail = fh.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            s = line.strip()
            if not s:
                continue
            # LiveTranscriptWriter.finalize / subagent.complete markers
            if "end status=" in s:
                return True
            if "status=completed" in s or "status=error" in s or "status=interrupted" in s:
                return True
            if "status=failed" in s or "status=cancelled" in s or "status=canceled" in s:
                return True
        return False
    except Exception:
        return False


def inspect_live_completion_evidence(delegation_id: str) -> Optional[Dict[str, Any]]:
    """Return terminal live evidence for a delegation, or None if still live/unknown.

    Prefer the live manifest (``completed`` stamp + per-task statuses). Fall back
    to task-log terminal markers when the manifest is incomplete.
    """
    if not delegation_id:
        return None
    manifest = _read_live_manifest(delegation_id)
    if not manifest:
        return None

    tasks = [t for t in (manifest.get("tasks") or []) if isinstance(t, dict)]
    statuses = [str(t.get("status") or "").lower() for t in tasks]
    terminal_tasks = [
        s for s in statuses if s in _TERMINAL_LIVE_STATUSES
    ]
    nonterminal = [
        s for s in statuses if s in _NONTERMINAL_LIVE_STATUSES or not s
    ]

    # Manifest-level completed stamp is authoritative once present.
    completed_stamp = manifest.get("completed")
    all_tasks_terminal = bool(tasks) and not nonterminal and len(terminal_tasks) == len(tasks)

    if not completed_stamp and not all_tasks_terminal:
        # Last-chance: every task log has a terminal marker even if status
        # fields lagged.
        logs_terminal = all(
            _task_log_has_terminal_marker(t.get("log")) for t in tasks
        ) if tasks else False
        if not logs_terminal:
            return None

    # Aggregate status from task statuses.
    status = "completed"
    if any(s in {"error", "failed"} for s in statuses):
        if all(s in {"error", "failed", "interrupted", "cancelled", "canceled"} for s in statuses if s):
            status = "error"
        else:
            status = "completed"  # mixed — still terminal aggregate
    elif any(s in {"interrupted", "cancelled", "canceled"} for s in statuses):
        status = "interrupted"

    summary_bits: List[str] = []
    for t in tasks:
        goal = str(t.get("goal") or "")[:80]
        st = str(t.get("status") or "?")
        summary_bits.append(f"task[{t.get('index')}]: {st}" + (f" — {goal}" if goal else ""))
    summary = "; ".join(summary_bits) if summary_bits else (
        f"live transcript reports terminal completion"
        + (f" at {completed_stamp}" if completed_stamp else "")
    )

    evidence = (
        f"live manifest terminal (status={status}"
        + (f", completed={completed_stamp}" if completed_stamp else "")
        + f", tasks={len(tasks)})"
    )
    return {
        "status": status,
        "summary": summary,
        "evidence": evidence,
        "completed_stamp": completed_stamp,
        "task_count": len(tasks),
        "manifest": manifest,
        "reclaimed_from_live": True,
    }


def _owner_process_is_alive(
    owner_pid: Optional[int],
    owner_started_at: Optional[int] = None,
) -> bool:
    """True iff the recorded owner process identity is still running.

    Matches ``recover_abandoned_delegations``: a bare PID existence check is
    insufficient because PIDs are reused. When ``owner_started_at`` is known,
    require ``get_process_start_time(pid) == owner_started_at``. When start
    time cannot be read, fall back to PID existence only (conservative: treat
    as alive so we do not steal from a living-but-opaque owner).
    """
    try:
        pid = int(owner_pid or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists, get_process_start_time
    except Exception:
        # Status helpers unavailable — psutil is safe across platforms.
        try:
            import psutil  # type: ignore

            return bool(psutil.pid_exists(pid))
        except Exception:
            pass
        # os.kill(pid, 0) sends CTRL_C_EVENT on Windows (bpo-14484). Without
        # a safe liveness helper, preserve ownership rather than signal it.
        if sys.platform != "win32":
            try:
                os.kill(pid, 0)  # windows-footgun: ok — explicit non-Windows branch
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            except OSError:
                return False
        return True
    if not _pid_exists(pid):
        return False
    if owner_started_at is None:
        return True
    try:
        live_start = get_process_start_time(pid)
    except Exception:
        live_start = None
    if live_start is None:
        # Cannot disambiguate PID reuse — keep conservative "alive".
        return True
    try:
        return int(live_start) == int(owner_started_at)
    except (TypeError, ValueError):
        return True


# Back-compat alias used by older tests / call sites.
def _owner_pid_is_alive(
    owner_pid: Optional[int],
    owner_started_at: Optional[int] = None,
) -> bool:
    return _owner_process_is_alive(owner_pid, owner_started_at)


def local_process_should_enqueue_delegation(
    owner_pid: Optional[int],
    owner_started_at: Optional[int] = None,
    *,
    local_pid: Optional[int] = None,
) -> bool:
    """Whether THIS process may put a completion on its local completion queue.

    Multi-TUI hosts share one durable SQLite table but each process has its own
    ``completion_queue``. Enqueuing a foreign-owned event poisons the wrong
    poller: it cannot prove ownership and historically *dropped* the payload
    (#28). Only the living owner process (or any process when the owner is
    dead / unknown / PID-reused) may enqueue.
    """
    me = int(local_pid if local_pid is not None else os.getpid())
    try:
        pid = int(owner_pid or 0)
    except (TypeError, ValueError):
        pid = 0
    if pid <= 0:
        return True
    if pid == me:
        return True
    return not _owner_process_is_alive(pid, owner_started_at)


# delegation_id → monotonic last requeue time (flood control for #28 reinject)
_local_pending_requeue_at: Dict[str, float] = {}
_LOCAL_PENDING_REQUEUE_COOLDOWN_S = 15.0


def _restore_durable_event_routing(
    evt: Dict[str, Any],
    *,
    origin_session: Any,
    origin_ui_session_id: Any,
    parent_session_id: Any,
    origin_session_id: Any,
) -> Dict[str, Any]:
    """Backfill a replayed event's missing routing from its durable row.

    Old completion payloads can have an empty ``origin_session_id`` (normal for
    non-api-server sessions) and, after a partial finalize, may also be missing
    the otherwise durable session/UI selectors.  Preserve any populated event
    field so a stale/foreign event cannot be rewritten into another chat; only
    use the durable dispatch record to restore an absent return address.
    """
    for field, value in (
        ("session_key", origin_session),
        ("origin_ui_session_id", origin_ui_session_id),
        ("parent_session_id", parent_session_id),
        ("origin_session_id", origin_session_id),
    ):
        if not str(evt.get(field) or "").strip() and value is not None:
            restored = str(value).strip()
            if restored:
                evt[field] = restored
    return evt


def requeue_local_pending_async_completions(target_queue) -> int:
    """Re-enqueue durable pending completions owned by this process.

    Covers the #28 gap where another process reclaimed durable state to
    ``completed/pending`` but enqueued (or dropped) the event on the *wrong*
    completion queue. Owner process then never saw a queue event.

    Must be called from **idle watchers** (TUI notification poller timeout,
    gateway async-completion loop) as well as ``drain_notifications`` —
    otherwise a parent that is only blocked on an empty in-memory queue will
    never wake after a foreign reclaim (#28 review finding).

    Per-id cooldown prevents busy-loop floods; consumers still must prove
    ownership before injection.
    """
    if target_queue is None:
        return 0
    me = os.getpid()
    now = time.time()
    enqueued = 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, owner_pid, owner_started_at, event_json,
                      origin_session, origin_ui_session_id, parent_session_id,
                      origin_session_id
               FROM async_delegations
               WHERE state NOT IN ('running','finalizing')
                 AND delivery_state='pending'
                 AND event_json IS NOT NULL
               ORDER BY completed_at, delegation_id"""
        ).fetchall()
    for (
        delegation_id,
        owner_pid,
        owner_started_at,
        payload,
        origin_session,
        origin_ui_session_id,
        parent_session_id,
        origin_session_id,
    ) in rows:
        if not local_process_should_enqueue_delegation(
            owner_pid, owner_started_at, local_pid=me
        ):
            continue
        last = _local_pending_requeue_at.get(delegation_id, 0.0)
        if (now - last) < _LOCAL_PENDING_REQUEUE_COOLDOWN_S:
            continue
        try:
            evt = json.loads(payload)
        except Exception:
            continue
        if not isinstance(evt, dict):
            continue
        evt = dict(evt)
        _restore_durable_event_routing(
            evt,
            origin_session=origin_session,
            origin_ui_session_id=origin_ui_session_id,
            parent_session_id=parent_session_id,
            origin_session_id=origin_session_id,
        )
        evt["restored"] = True
        evt.setdefault("type", "async_delegation")
        evt["delegation_id"] = delegation_id
        try:
            target_queue.put(evt)
        except Exception as exc:
            logger.error(
                "Failed to requeue local pending async completion %s: %s",
                delegation_id, exc,
            )
            continue
        _local_pending_requeue_at[delegation_id] = now
        enqueued += 1
    # Cap bookkeeping growth
    if len(_local_pending_requeue_at) > 256:
        cutoff = now - 3600.0
        stale = [k for k, ts in _local_pending_requeue_at.items() if ts < cutoff]
        for k in stale:
            _local_pending_requeue_at.pop(k, None)
    return enqueued


def _arm_pending_delivery_heartbeat(
    delegation_id: str,
    *,
    caller_id: str,
    provider: str = "",
) -> bool:
    """Wake a parent once if a terminal delegation is still undelivered.

    The normal running-target heartbeat is cancelled at completion.  Keep a
    separate one-interval watchdog while its durable delivery remains pending:
    a lost/foreign queue event can then wake the rightful idle parent without
    extending the configured warm-KV cadence.  The first inspection arms the
    timer; the expiry inspection reports the still-pending row as STUCK, which
    intentionally produces one owner-routed check-in.
    """
    delegation_id = str(delegation_id or "")
    caller_id = str(caller_id or "")
    if not delegation_id or not caller_id:
        return False

    first_inspection = True

    def inspect_delivery() -> Dict[str, Any]:
        nonlocal first_inspection
        if first_inspection:
            first_inspection = False
            return {
                "alive": True,
                "progress": True,
                "evidence": "awaiting durable async completion delivery",
            }
        durable = get_durable_delegation(delegation_id)
        if (
            durable is not None
            and str(durable.get("state") or "") not in {"running", "finalizing"}
            and str(durable.get("delivery_state") or "") == "pending"
        ):
            return {
                "alive": False,
                "evidence": "async completion remains durable-pending after one warm-KV interval",
            }
        return {"alive": False, "evidence": "async completion delivery is no longer pending"}

    try:
        from tools.runtime_heartbeat import runtime_heartbeat

        return bool(
            runtime_heartbeat.arm(
                delegation_id,
                caller_id=caller_id,
                kind="async_delivery",
                provider=provider,
                inspect=inspect_delivery,
            )
        )
    except Exception:
        logger.debug(
            "Could not arm pending-delivery heartbeat for %s", delegation_id,
            exc_info=True,
        )
        return False


def _cancel_pending_delivery_heartbeat(delegation_id: str) -> None:
    try:
        from tools.runtime_heartbeat import runtime_heartbeat

        runtime_heartbeat.cancel(delegation_id)
    except Exception:
        logger.debug(
            "Could not cancel delivery heartbeat for %s",
            delegation_id,
            exc_info=True,
        )


def ensure_owner_pending_delivery_heartbeats() -> int:
    """Arm one configured-interval fallback for local terminal pending rows."""
    me = os.getpid()
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, origin_session, origin_ui_session_id,
                      owner_pid, owner_started_at, task_json
               FROM async_delegations
               WHERE state NOT IN ('running','finalizing')
                 AND delivery_state='pending'
               ORDER BY completed_at, delegation_id"""
        ).fetchall()
    try:
        from tools.runtime_heartbeat import runtime_heartbeat
    except Exception:
        return 0

    armed = 0
    for (
        delegation_id,
        origin_session,
        origin_ui_session_id,
        owner_pid,
        owner_started_at,
        task_json,
    ) in rows:
        if not local_process_should_enqueue_delegation(
            owner_pid, owner_started_at, local_pid=me
        ):
            continue
        caller_id = str(origin_session or origin_ui_session_id or "")
        if not caller_id:
            continue
        if delegation_id in runtime_heartbeat.outstanding_for_caller(caller_id):
            continue
        try:
            task = json.loads(task_json or "{}")
        except Exception:
            task = {}
        if _arm_pending_delivery_heartbeat(
            str(delegation_id),
            caller_id=caller_id,
            provider=str(task.get("provider") or ""),
        ):
            armed += 1
    return armed


def poll_owner_local_async_completions(target_queue=None) -> int:
    """Idle-watcher helper: reclaim orphans + reinject owner-local pending.

    Safe to call on a timer from TUI/gateway loops that otherwise only block
    on the in-memory completion queue. Returns total events enqueued by the
    reclaim path + local requeue path (best-effort).
    """
    try:
        from tools.process_registry import process_registry as _pr
    except Exception:
        _pr = None
    if target_queue is None:
        target_queue = getattr(_pr, "completion_queue", None) if _pr is not None else None
    if target_queue is None:
        return 0
    n = 0
    try:
        n += int(reclaim_orphaned_completed_delegations(target_queue, enqueue=True) or 0)
    except Exception:
        logger.debug("poll_owner_local reclaim failed", exc_info=True)
    try:
        n += int(requeue_local_pending_async_completions(target_queue) or 0)
    except Exception:
        logger.debug("poll_owner_local requeue failed", exc_info=True)
    try:
        ensure_owner_pending_delivery_heartbeats()
    except Exception:
        logger.debug(
            "poll_owner_local pending-delivery heartbeat arm failed",
            exc_info=True,
        )
    return n


def reclaim_orphaned_completed_delegations(
    target_queue=None, *, enqueue: bool = True
) -> int:
    """Reclaim durable running/finalizing rows whose live transcripts are terminal.

    Covers the case where a child finished and wrote a completed live
    manifest/log, but ``_persist_completion`` never landed (process race,
    finalize crash, etc.). Owner-dead rows remain ``recover_abandoned_delegations``'s
    job; this path only acts when live evidence proves terminal completion.

    Durable state is updated on any process that observes live terminal evidence.
    Queue publication is restricted to the living owner process (or any process
    when the owner is dead / PID-reused) so multi-TUI hosts do not poison foreign
    queues (#28).
    """
    try:
        from tools.process_registry import process_registry as _pr
    except Exception:
        _pr = None
    if target_queue is None and enqueue:
        target_queue = getattr(_pr, "completion_queue", None) if _pr is not None else None

    now = time.time()
    me = os.getpid()
    reclaimed = 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, origin_session, origin_ui_session_id,
                      parent_session_id, dispatched_at, owner_pid,
                      owner_started_at, task_json, origin_session_id, delivery_state
               FROM async_delegations
               WHERE state IN ('running','finalizing')
               ORDER BY dispatched_at, delegation_id"""
        ).fetchall()
    events_to_enqueue: List[Dict[str, Any]] = []
    for row in rows:
        (delegation_id, session_key, origin_ui, parent_id, dispatched_at,
         owner_pid, owner_started_at, task_json, origin_session_id, delivery_state) = row
        # Already-delivered rows should not be re-enqueued even if state lagging.
        if delivery_state == "delivered":
            continue
        live = inspect_live_completion_evidence(delegation_id)
        if not live:
            continue
        try:
            task = json.loads(task_json or "{}")
        except Exception:
            task = {}
        status = str(live.get("status") or "completed")
        summary = live.get("summary")
        completed_at = now
        event = {
            "type": "async_delegation",
            "delegation_id": delegation_id,
            "session_key": session_key or "",
            "origin_ui_session_id": origin_ui or "",
            "origin_session_id": origin_session_id or "",
            "parent_session_id": parent_id,
            "goal": task.get("goal", ""),
            "goals": task.get("goals"),
            "context": task.get("context"),
            "toolsets": task.get("toolsets"),
            "role": task.get("role"),
            "model": task.get("model"),
            "is_batch": bool(task.get("is_batch")) or bool(task.get("goals")),
            "status": status,
            "summary": summary,
            "error": None if status == "completed" else (
                f"Reclaimed from live transcript with terminal status={status}"
            ),
            "dispatched_at": dispatched_at,
            "completed_at": completed_at,
            "reclaimed_from_live": True,
            "live_evidence": live.get("evidence"),
            "owner_pid": owner_pid,
            "owner_started_at": owner_started_at,
        }
        result = {
            "status": status,
            "summary": summary,
            "error": event.get("error"),
            "reclaimed_from_live": True,
        }
        try:
            # Only transition rows that are STILL non-terminal + pending so a
            # concurrent real finalize cannot be clobbered into a double-fire.
            with _DB_LOCK, _transaction() as conn:
                cur = conn.execute(
                    """UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
                       event_json=?, result_json=?, delivery_state='pending'
                       WHERE delegation_id=? AND state IN ('running','finalizing')
                         AND delivery_state='pending'""",
                    (status, completed_at, now, json.dumps(event),
                     json.dumps(result), delegation_id),
                )
                if cur.rowcount != 1:
                    continue
            reclaimed += 1
            # Only the living owner (or any process if owner is dead) may put
            # this on a local completion queue (#28 cross-process poison).
            if local_process_should_enqueue_delegation(
                owner_pid, owner_started_at, local_pid=me
            ):
                events_to_enqueue.append(event)
            else:
                logger.info(
                    "Reclaimed orphaned completed async delegation %s durable-only "
                    "(owner_pid=%s still alive elsewhere; not enqueuing on pid=%s)",
                    delegation_id, owner_pid, me,
                )
            # Keep in-memory view consistent when present.
            with _records_lock:
                rec = _records.get(delegation_id)
                if rec is not None:
                    rec["status"] = status
                    rec["completed_at"] = completed_at
            logger.warning(
                "Reclaimed orphaned completed async delegation %s from live "
                "transcript (durable was still running/finalizing): %s",
                delegation_id, live.get("evidence"),
            )
        except Exception as exc:
            logger.error(
                "Failed to reclaim orphaned delegation %s: %s",
                delegation_id, exc, exc_info=True,
            )

    if enqueue and target_queue is not None:
        for evt in events_to_enqueue:
            try:
                # restored=True so ownership filters fail-closed on foreign sessions
                evt_out = dict(evt)
                evt_out["restored"] = True
                target_queue.put(evt_out)
            except Exception as exc:
                logger.error(
                    "Reclaimed %s but enqueue failed: %s",
                    evt.get("delegation_id"), exc,
                )
    return reclaimed


def restore_undelivered_completions(target_queue) -> int:
    """Enqueue durable pending completions as fresh turns after process start.

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
    # Owner-alive zombies: live transcript finished but durable still running.
    # Persist only here; this function re-reads event_json and enqueues once so
    # reclaim must not double-put onto target_queue.
    try:
        reclaim_orphaned_completed_delegations(target_queue, enqueue=False)
    except Exception as exc:
        logger.warning("reclaim_orphaned_completed_delegations failed: %s", exc)
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, event_json, origin_session,
                      origin_ui_session_id, parent_session_id, origin_session_id
               FROM async_delegations
               WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                 AND event_json IS NOT NULL
               ORDER BY completed_at, delegation_id"""
        ).fetchall()
        for (
            _delegation_id,
            payload,
            origin_session,
            origin_ui_session_id,
            parent_session_id,
            origin_session_id,
        ) in rows:
            evt = json.loads(payload)
            if isinstance(evt, dict):
                _restore_durable_event_routing(
                    evt,
                    origin_session=origin_session,
                    origin_ui_session_id=origin_ui_session_id,
                    parent_session_id=parent_session_id,
                    origin_session_id=origin_session_id,
                )
                evt["restored"] = True
            target_queue.put(evt)
    return len(rows)


def mark_completion_delivered(delegation_id: str) -> bool:
    """Atomically acknowledge successful injection of a durable completion."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered', delivered_at=?, updated_at=?
               WHERE delegation_id=? AND delivery_state!='delivered'""",
            (now, now, delegation_id),
        )
        delivered = cur.rowcount == 1
    if delivered:
        _cancel_pending_delivery_heartbeat(delegation_id)
    return delivered


def claim_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Claim one pending completion across competing consumers/processes."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT delivery_state FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()
        if row is None:
            return True  # legacy event created before durable dispatch
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_claim=?, delivery_claimed_at=?,
                      delivery_attempts=delivery_attempts+1, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
            (claim_id, now, now, delegation_id, now - 300),
        )
        return cur.rowcount == 1


def claim_event_delivery(evt: Dict[str, Any], consumer: str) -> Optional[str]:
    """Claim a durable delegation event; non-durable events need no token."""
    if evt.get("type") != "async_delegation":
        return ""
    delegation_id = str(evt.get("delegation_id") or "")
    if not delegation_id:
        return ""
    claim_id = f"{consumer}:{__import__('os').getpid()}:{uuid.uuid4().hex}"
    return claim_id if claim_completion_delivery(delegation_id, claim_id) else None


def release_completion_delivery(
    delegation_id: str, claim_id: str, consume_attempt: bool = True
) -> bool:
    """Release a failed delivery claim so another consumer may retry.

    Attempts are counted at claim time. The default release therefore keeps a
    claim's count and terminally drops a row once the retry budget is exhausted.
    A best-effort busy-steer is only a hint, not a delivery attempt: callers
    can set ``consume_attempt=False`` to atomically release its claim and roll
    back the increment without allowing the count to fall below zero.
    """
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        if not consume_attempt:
            cur = conn.execute(
                """UPDATE async_delegations SET delivery_claim=NULL,
                          delivery_claimed_at=NULL,
                          delivery_attempts=CASE WHEN delivery_attempts > 0
                              THEN delivery_attempts - 1 ELSE 0 END,
                          updated_at=?
                   WHERE delegation_id=? AND delivery_state='pending'
                     AND delivery_claim=?""",
                (now, delegation_id, claim_id),
            )
            return cur.rowcount == 1
        capped = conn.execute(
            """UPDATE async_delegations SET delivery_state='dropped',
                      delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
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
            """UPDATE async_delegations SET delivery_claim=NULL,
                      delivery_claimed_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
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
                      updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, delegation_id, claim_id),
        )
        dropped = cur.rowcount == 1
    if dropped:
        _cancel_pending_delivery_heartbeat(delegation_id)
    return dropped


def complete_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Acknowledge acceptance for the consumer holding this claim."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered',
                      delivered_at=?, updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, now, delegation_id, claim_id),
        )
        delivered = cur.rowcount == 1
    if delivered:
        _cancel_pending_delivery_heartbeat(delegation_id)
    return delivered


def complete_event_delivery(evt: Dict[str, Any], claim_id: str) -> None:
    if claim_id and evt.get("type") == "async_delegation":
        complete_completion_delivery(str(evt.get("delegation_id") or ""), claim_id)


def release_event_delivery(
    evt: Dict[str, Any], claim_id: str, consume_attempt: bool = True
) -> None:
    if claim_id and evt.get("type") == "async_delegation":
        release_completion_delivery(
            str(evt.get("delegation_id") or ""),
            claim_id,
            consume_attempt=consume_attempt,
        )


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
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is None or max_workers > _executor_max_workers:
            # Daemon threads: thread_name_prefix aids debugging in stack dumps.
            _executor = _DaemonThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="async-delegate",
            )
            _executor_max_workers = max_workers
        return _executor


def active_count() -> int:
    """Number of async delegations currently running."""
    with _records_lock:
        return sum(1 for r in _records.values() if r.get("status") in {"running", "finalizing"})


def _new_delegation_id() -> str:
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _prune_completed_locked() -> None:
    """Drop the oldest completed records beyond the retention cap.

    Caller must hold ``_records_lock``.
    """
    completed = [
        (rid, r)
        for rid, r in _records.items()
        if r.get("status") != "running"
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
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
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
    max_async_children
        Concurrency cap. When at capacity the dispatch is REJECTED (the caller
        should fall back to sync or tell the user) rather than queued, so a
        runaway model can't pile up unbounded background work.

    Returns
    -------
    dict
        ``{"status": "dispatched", "delegation_id": ...}`` on success, or
        ``{"status": "rejected", "error": ...}`` when at capacity.
    """
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
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
    }
    # Capacity check and record insert under ONE lock hold — checking
    # active_count() separately would let two concurrent dispatches (e.g.
    # from different gateway sessions) both pass the check and exceed the cap.
    with _records_lock:
        running = sum(
            1 for r in _records.values() if r.get("status") == "running"
        )
        if running >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"running). Wait for one to finish (its result will re-enter "
                    f"the chat), or run this task synchronously "
                    f"(background=false). Raise delegation.max_concurrent_children in "
                    f"config.yaml to allow more concurrent background subagents."
                ),
            }
        _records[delegation_id] = record

    _persist_dispatch(record)
    executor = _get_executor(max_async_children)

    def _worker() -> None:
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

    try:
        # Propagate the dispatching profile so the detached child resolves
        # get_hermes_home() under the right profile.
        executor.submit(propagate_context_to_thread(_worker))
    except Exception as exc:  # pragma: no cover — pool submit failure is rare
        with _records_lock:
            _records.pop(delegation_id, None)
        _delete_durable_delegation(delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation: {exc}",
        }

    # Async delegations use the same per-target lifecycle as managed
    # terminal work. Completion cancels this ID only (in _finalize).
    from tools.runtime_heartbeat import (
        get_current_provider,
        inspect_delegation,
        runtime_heartbeat,
    )
    runtime_heartbeat.arm(
        delegation_id,
        caller_id=session_key,
        kind="delegation",
        provider=get_current_provider(),
        inspect=lambda _id=delegation_id: inspect_delegation(_id),
    )

    logger.info(
        "Dispatched async delegation %s (session_key=%s): %s",
        delegation_id, session_key or "<cli>", (goal or "")[:80],
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize(delegation_id: str, result: Dict[str, Any], status: str) -> None:
    """Mark a record complete and push the completion event onto the queue."""
    # Cancellation is intentionally per delegation ID, never per caller.
    try:
        from tools.runtime_heartbeat import runtime_heartbeat
        runtime_heartbeat.cancel(delegation_id)
    except Exception:
        logger.debug("Failed to cancel heartbeat for async delegation %s", delegation_id, exc_info=True)
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None:
            return
        # Stay active until durable persistence and queue publication finish;
        # otherwise process shutdown can kill this daemon worker in the narrow
        # gap after status flips but before SQLite is committed.
        record["status"] = "finalizing"
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None  # drop the closure; child is done
        event_record = dict(record)

    _push_completion_event(event_record, result, status)
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
    persisted = False
    try:
        _persist_completion(evt, result)
        persisted = True
    except Exception as exc:
        logger.error(
            "Async delegation %s: durable completion persist failed after retry; "
            "will rely on live-transcript reclaim: %s",
            record.get("delegation_id"), exc, exc_info=True,
        )
    if persisted:
        _arm_pending_delivery_heartbeat(
            str(record.get("delegation_id") or ""),
            caller_id=str(
                record.get("session_key") or record.get("origin_ui_session_id") or ""
            ),
            provider=str(record.get("provider") or ""),
        )
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
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    delegation_id: Optional[str] = None,
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

    Returns ``{"status": "dispatched", "delegation_id": ...}`` on success or
    ``{"status": "rejected", "error": ...}`` when the async pool is at
    capacity.
    """
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
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "is_batch": True,
    }
    with _records_lock:
        running = sum(
            1 for r in _records.values() if r.get("status") == "running"
        )
        if running >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"running). Wait for one to finish (its result will re-enter "
                    f"the chat), or raise delegation.max_concurrent_children in "
                    f"config.yaml to allow more concurrent background units."
                ),
            }
        _records[delegation_id] = record

    _persist_dispatch(record)
    executor = _get_executor(max_async_children)

    def _worker() -> None:
        combined: Dict[str, Any] = {}
        status = "error"
        try:
            combined = runner() or {}
            # Batch status: completed unless every child errored/was interrupted.
            # Preserve an absent/malformed result envelope for the completion
            # consumer. Normalizing ``None`` to ``[]`` makes an unknown batch
            # appear to be a clean, empty success and bypasses its fail-closed
            # visible-response policy.
            child_results = combined.get("results")
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

    try:
        # Propagate the dispatching profile to the detached batch children.
        executor.submit(propagate_context_to_thread(_worker))
    except Exception as exc:  # pragma: no cover
        with _records_lock:
            _records.pop(delegation_id, None)
        _delete_durable_delegation(delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation batch: {exc}",
        }

    from tools.runtime_heartbeat import (
        get_current_provider,
        inspect_delegation,
        runtime_heartbeat,
    )
    runtime_heartbeat.arm(
        delegation_id,
        caller_id=session_key,
        kind="delegation",
        provider=get_current_provider(),
        inspect=lambda _id=delegation_id: inspect_delegation(_id),
    )

    logger.info(
        "Dispatched async delegation batch %s (%d task(s), session_key=%s)",
        delegation_id, n, session_key or "<cli>",
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize_batch(
    delegation_id: str, combined: Dict[str, Any], status: str
) -> None:
    """Mark a batch record complete and push ONE combined completion event."""
    try:
        from tools.runtime_heartbeat import runtime_heartbeat
        runtime_heartbeat.cancel(delegation_id)
    except Exception:
        logger.debug("Failed to cancel heartbeat for async delegation batch %s", delegation_id, exc_info=True)
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None:
            return
        record["status"] = "finalizing"
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None
        event_record = dict(record)

    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s finished but process_registry import "
            "failed; result lost: %s",
            delegation_id, exc,
        )
        return

    dispatched_at = event_record.get("dispatched_at") or time.time()
    completed_at = event_record.get("completed_at") or time.time()
    evt = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
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
        # Preserve the runner's result envelope verbatim. In particular,
        # ``None`` must remain observable by consumers so they fail closed
        # instead of treating it as an empty successful batch.
        "results": combined.get("results"),
        # Per-task live transcript log paths (cache/delegation/live/...).
        # They persist after completion and double as the full-fidelity
        # operational record of each child's run.
        "live_transcripts": combined.get("live_transcripts"),
        "error": combined.get("error"),
        "total_duration_seconds": combined.get("total_duration_seconds"),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
    }
    persisted = False
    try:
        _persist_completion(evt, combined)
        persisted = True
    except Exception as exc:
        logger.error(
            "Async delegation batch %s: durable completion persist failed after "
            "retry; will rely on live-transcript reclaim: %s",
            delegation_id, exc, exc_info=True,
        )
    if persisted:
        _arm_pending_delivery_heartbeat(
            delegation_id,
            caller_id=str(
                event_record.get("session_key")
                or event_record.get("origin_ui_session_id")
                or ""
            ),
            provider=str(event_record.get("provider") or ""),
        )
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s: failed to enqueue completion event; "
            "result lost: %s",
            delegation_id, exc,
        )
    finally:
        with _records_lock:
            record = _records.get(delegation_id)
            if record is not None:
                record["status"] = status
            _prune_completed_locked()


def list_async_delegations() -> List[Dict[str, Any]]:
    """Snapshot of async delegations (running + recently completed).

    Safe to call from any thread. Excludes the non-serialisable interrupt_fn.
    """
    with _records_lock:
        return [
            {k: v for k, v in r.items() if k != "interrupt_fn"}
            for r in _records.values()
        ]


def interrupt_all(reason: str = "shutdown") -> int:
    """Signal every running async delegation to stop. Returns how many.

    Used on ``/stop`` and gateway shutdown so a dangling background subagent
    can't keep burning tokens with no one listening. The child still emits a
    completion event (status='interrupted') via the normal finalize path.
    """
    count = 0
    with _records_lock:
        targets = [
            r for r in _records.values() if r.get("status") == "running"
        ]
    for r in targets:
        fn = r.get("interrupt_fn")
        if callable(fn):
            try:
                fn()
                count += 1
            except Exception as exc:
                logger.debug(
                    "interrupt_all: %s interrupt failed: %s",
                    r.get("delegation_id"), exc,
                )
    if count:
        logger.info("Interrupted %d async delegation(s) (%s)", count, reason)
    return count


def interrupt_for_session(
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
    reason: str = "session_end",
) -> int:
    """Signal running async delegations owned by ONE session to stop.

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
    if not session_key and not origin_ui_session_id and not parent_session_id:
        return 0
    count = 0
    with _records_lock:
        targets = [
            r for r in _records.values()
            if r.get("status") == "running"
            and (
                (origin_ui_session_id and str(r.get("origin_ui_session_id") or "") == origin_ui_session_id)
                or (session_key and str(r.get("session_key") or "") == session_key)
                or (parent_session_id and str(r.get("parent_session_id") or "") == parent_session_id)
            )
        ]
    for r in targets:
        fn = r.get("interrupt_fn")
        if callable(fn):
            try:
                fn()
                count += 1
            except Exception as exc:
                logger.debug(
                    "interrupt_for_session: %s interrupt failed: %s",
                    r.get("delegation_id"), exc,
                )
    if count:
        logger.info(
            "Interrupted %d async delegation(s) for ending session (%s)",
            count, reason,
        )
    return count


def _reset_for_tests() -> None:
    """Test-only: clear all state and tear down the executor."""
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = None
        _executor_max_workers = 0
    with _records_lock:
        _records.clear()
