"""Per-target runtime heartbeat/watchdog support.

Long-running managed terminal processes and async delegations are tracked
individually.  A target completion cancels only its own timer; every caller LLM
activation resets all *remaining* targets owned by that caller so completions
cannot collapse the check-in cadence into a short-poll loop.
"""

from __future__ import annotations

import contextvars
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

DOCUMENTED_FALLBACK_TTL_SECONDS = 3300
_current_provider: contextvars.ContextVar[str] = contextvars.ContextVar(
    "runtime_heartbeat_provider", default=""
)


def set_current_provider(provider: str | None) -> contextvars.Token[str]:
    """Bind the live caller provider so tool-worker context can resolve its TTL."""
    return _current_provider.set(str(provider or ""))


def reset_current_provider(token: contextvars.Token[str]) -> None:
    _current_provider.reset(token)


def get_current_provider() -> str:
    return _current_provider.get() or ""


def canonical_provider_family(provider: str | None) -> str:
    """Return the family used after exact provider lookup.

    Provider identity is deliberately preserved for exact configuration keys
    such as ``custom:pm``.  The family fallback lets a generic ``custom`` or
    ``openai`` policy cover endpoint/model variants without guessing a TTL.
    """
    value = (provider or "").strip().lower()
    if not value:
        return ""
    if value.startswith("custom:"):
        return "custom"
    if value.startswith("openai-"):
        return "openai"
    if ":" in value:
        return value.split(":", 1)[0]
    return value


def resolve_kv_cache_ttl(
    runtime_config: Optional[Dict[str, Any]], provider: str | None,
    *, fallback: int = DOCUMENTED_FALLBACK_TTL_SECONDS,
) -> int:
    """Resolve TTL: exact provider, canonical family, default, documented fallback."""
    cfg = runtime_config if isinstance(runtime_config, dict) else {}
    ttl_cfg = cfg.get("kv_cache_ttl")
    if not isinstance(ttl_cfg, dict):
        return fallback
    providers = ttl_cfg.get("providers")
    providers = providers if isinstance(providers, dict) else {}
    lookup = (provider or "").strip().lower()
    lowered = {str(k).strip().lower(): v for k, v in providers.items()}
    candidates = (lookup, canonical_provider_family(lookup), "default")
    for candidate in candidates:
        value = lowered.get(candidate) if candidate else None
        if value is None and candidate == "default":
            value = ttl_cfg.get("default")
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return fallback


def _runtime_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        runtime = config.get("runtime") if isinstance(config, dict) else {}
        return runtime if isinstance(runtime, dict) else {}
    except Exception:
        logger.debug("Could not load runtime heartbeat configuration", exc_info=True)
        return {}


def _heartbeat_settings(runtime: Dict[str, Any], provider: str | None) -> tuple[bool, float]:
    heartbeat = runtime.get("heartbeat") if isinstance(runtime, dict) else {}
    heartbeat = heartbeat if isinstance(heartbeat, dict) else {}
    if not heartbeat.get("enabled", True) or heartbeat.get("mode", "per_target") != "per_target":
        return False, 0.0
    ttl = resolve_kv_cache_ttl(runtime, provider)
    try:
        ratio = float(heartbeat.get("safety_ratio", 0.8))
    except (TypeError, ValueError):
        ratio = 0.8
    try:
        minimum = float(heartbeat.get("min_interval_seconds", 60))
    except (TypeError, ValueError):
        minimum = 60.0
    try:
        maximum = float(heartbeat.get("max_interval_seconds", ttl))
    except (TypeError, ValueError):
        maximum = float(ttl)
    minimum = max(1.0, minimum)
    maximum = max(minimum, maximum)
    return True, max(minimum, min(maximum, ttl * max(0.0, ratio)))


def _reset_on_caller_activation(runtime: Dict[str, Any]) -> bool:
    """Return whether caller activity may reschedule outstanding targets."""
    heartbeat = runtime.get("heartbeat") if isinstance(runtime, dict) else {}
    heartbeat = heartbeat if isinstance(heartbeat, dict) else {}
    raw = heartbeat.get("reset_on_caller_activation", True)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"false", "0", "no", "off"}
    return bool(raw)


@dataclass
class _Target:
    target_id: str
    caller_id: str
    kind: str
    provider: str
    inspect: Callable[[], Dict[str, Any]]
    interval: float
    started_at: float
    due_at: float = 0.0
    generation: int = 0
    timer: Any = None
    baseline: Dict[str, Any] = field(default_factory=dict)


class RuntimeHeartbeat:
    """Thread-safe per-target heartbeat timer manager.

    ``inspect`` returns a compact snapshot.  Process snapshots are compared to
    their previous output/CPU values; a live PID alone never qualifies as
    ALIVE. Async delegations do not expose reliable per-turn activity, so
    running/finalizing snapshots explicitly report progress instead of
    comparing their static dispatch timestamp.
    """

    def __init__(
        self,
        *,
        config_loader: Callable[[], Dict[str, Any]] = _runtime_config,
        event_queue: Optional[queue.Queue] = None,
        timer_factory: Callable[[float, Callable[[], None]], Any] = threading.Timer,
    ) -> None:
        self._config_loader = config_loader
        self._event_queue = event_queue
        self._timer_factory = timer_factory
        self._lock = threading.RLock()
        self._targets: Dict[str, _Target] = {}

    @staticmethod
    def _key(target_id: str) -> str:
        return str(target_id)

    def _queue(self):
        if self._event_queue is not None:
            return self._event_queue
        from tools.process_registry import process_registry

        return process_registry.completion_queue

    def _config(self) -> Dict[str, Any]:
        config = self._config_loader()
        return config if isinstance(config, dict) else {}

    def arm(
        self,
        target_id: str,
        *,
        caller_id: str,
        kind: str,
        provider: str | None = None,
        inspect: Callable[[], Dict[str, Any]],
    ) -> bool:
        """Arm/replace one target. Returns false when heartbeat is disabled."""
        if not target_id or not caller_id:
            return False
        runtime = self._config()
        enabled, interval = _heartbeat_settings(runtime, provider)
        key = self._key(target_id)
        with self._lock:
            existing = self._targets.pop(key, None)
            if existing and existing.timer is not None:
                existing.timer.cancel()
            if not enabled:
                return False
            target = _Target(
                target_id=str(target_id), caller_id=str(caller_id), kind=str(kind),
                provider=str(provider or ""), inspect=inspect, interval=interval,
                started_at=time.time(),
            )
            # Register before inspecting. A racing completion can now cancel this
            # exact entry; it can never leave a timer armed after completion.
            self._targets[key] = target
        try:
            baseline = dict(inspect() or {})
        except Exception:
            baseline = {}
        with self._lock:
            if self._targets.get(key) is not target:
                return False
            if baseline.get("alive") is False:
                self._targets.pop(key, None)
                return False
            target.baseline = baseline
            self._schedule_locked(key, target)
        return True

    def _schedule_locked(self, key: str, target: _Target) -> None:
        target.generation += 1
        generation = target.generation
        target.due_at = time.time() + target.interval
        timer = self._timer_factory(
            target.interval, lambda: self._fire(key, generation)
        )
        # Timer-like fakes used by unit tests may not expose daemon.
        if hasattr(timer, "daemon"):
            timer.daemon = True
        target.timer = timer
        timer.start()

    def cancel(self, target_id: str) -> bool:
        """Cancel one target only; sibling targets remain armed."""
        with self._lock:
            target = self._targets.pop(self._key(target_id), None)
            if target is None:
                return False
            if target.timer is not None:
                target.timer.cancel()
            return True

    def reset_for_caller(self, caller_id: str) -> int:
        """Reset all outstanding targets of exactly one caller to a full TTL window."""
        if not caller_id:
            return 0
        runtime = self._config()
        if not _reset_on_caller_activation(runtime):
            return 0
        count = 0
        with self._lock:
            for key, target in tuple(self._targets.items()):
                if target.caller_id != str(caller_id):
                    continue
                enabled, interval = _heartbeat_settings(runtime, target.provider)
                if target.timer is not None:
                    target.timer.cancel()
                if not enabled:
                    self._targets.pop(key, None)
                    continue
                target.interval = interval
                self._schedule_locked(key, target)
                count += 1
        return count

    def outstanding_for_caller(self, caller_id: str) -> list[str]:
        with self._lock:
            return [target.target_id for target in self._targets.values() if target.caller_id == caller_id]

    def snapshot_active_targets(self) -> list[dict[str, int | str]]:
        """Return compact elapsed/TTL observability for active watchdog targets.

        Managed processes get their wall-clock age from the registry's durable
        ``ProcessSession.started_at`` instead of the watchdog arm time. Other
        target kinds fall back to their arm time because they have no process
        session. ``due_at`` is refreshed on every arm/re-arm, so the TTL is the
        remaining interval until that target's next heartbeat.
        """
        now = time.time()
        with self._lock:
            targets = tuple(self._targets.values())

        rows: list[dict[str, int | str]] = []
        for target in targets:
            started_at = target.started_at
            if target.kind == "process":
                try:
                    from tools.process_registry import process_registry

                    session = process_registry.get(target.target_id)
                    session_started_at = getattr(session, "started_at", None)
                    if isinstance(session_started_at, (int, float)) and session_started_at > 0:
                        started_at = float(session_started_at)
                except Exception:
                    logger.debug("Could not read process start time for %s", target.target_id, exc_info=True)

            elapsed_s = max(0, int(now - started_at))
            ttl_remaining_s = max(0, int(math.ceil(target.due_at - now)))
            rows.append(
                {
                    "session_id": target.target_id,
                    "elapsed_s": elapsed_s,
                    "ttl_remaining_s": ttl_remaining_s,
                }
            )
        return rows

    @staticmethod
    def _assess(target: _Target, snapshot: Dict[str, Any]) -> tuple[str, str]:
        if not snapshot.get("alive", False):
            return "STUCK", str(snapshot.get("evidence") or "target is no longer alive")
        if target.kind == "process":
            old_output = int(target.baseline.get("output_size") or 0)
            new_output = int(snapshot.get("output_size") or 0)
            old_cpu = float(target.baseline.get("cpu_seconds") or 0.0)
            new_cpu = float(snapshot.get("cpu_seconds") or 0.0)
            if new_output > old_output:
                return "ALIVE", f"output grew {old_output}->{new_output} bytes"
            if new_cpu > old_cpu:
                return "ALIVE", f"CPU advanced {old_cpu:.2f}->{new_cpu:.2f}s"
            return "STUCK", str(snapshot.get("evidence") or "process is live but produced no output or CPU progress")
        if snapshot.get("progress") is True:
            return "ALIVE", str(snapshot.get("evidence") or "delegation in progress")
        old_activity = target.baseline.get("last_activity_at")
        new_activity = snapshot.get("last_activity_at")
        if new_activity != old_activity:
            return "ALIVE", str(snapshot.get("evidence") or "delegation activity advanced")
        evidence = str(snapshot.get("evidence") or "delegation remains active")
        return "STUCK", f"{evidence}; activity did not advance"

    def _fire(self, key: str, generation: int) -> None:
        with self._lock:
            target = self._targets.get(key)
            if target is None or target.generation != generation:
                return
            target.timer = None
        try:
            snapshot = dict(target.inspect() or {})
        except Exception as exc:  # a failed check must surface, not silently disappear
            snapshot = {"alive": False, "evidence": f"heartbeat inspection failed: {type(exc).__name__}: {exc}"}
        status, evidence = self._assess(target, snapshot)
        with self._lock:
            current = self._targets.get(key)
            if current is not target or current.generation != generation:
                return
            target.baseline = snapshot
            if status == "ALIVE":
                self._schedule_locked(key, target)
            else:
                self._targets.pop(key, None)
        event = {
            "type": "heartbeat",
            "target_id": target.target_id,
            "target_kind": target.kind,
            "session_id": target.target_id if target.kind == "process" else "",
            "session_key": target.caller_id,
            "status": status,
            "evidence": evidence[:500],
        }
        try:
            self._queue().put(event)
        except Exception:
            logger.exception("Unable to enqueue heartbeat for %s", target.target_id)


def inspect_process(target_id: str) -> Dict[str, Any]:
    """Inspect one managed process without treating PID existence as progress."""
    from tools.process_registry import process_registry

    session = process_registry.get(target_id)
    if session is None:
        return {"alive": False, "evidence": "process is not registered"}
    with session._lock:
        exited = bool(session.exited)
        output_size = len(session.output_buffer or "")
        pid = session.pid
    if exited:
        return {"alive": False, "output_size": output_size, "evidence": "process exited before completion delivery"}
    cpu_seconds = 0.0
    if pid:
        try:
            import psutil

            cpu = psutil.Process(pid).cpu_times()
            cpu_seconds = float(cpu.user + cpu.system)
        except Exception:
            # Non-local backends often have no host-visible PID. Output growth is
            # still a valid productive signal; lack of CPU access is not a PID-only
            # ALIVE decision.
            pass
    return {"alive": True, "output_size": output_size, "cpu_seconds": cpu_seconds}


def inspect_delegation(target_id: str) -> Dict[str, Any]:
    """Inspect one async delegation by its durable runtime status/activity."""
    from tools.async_delegation import list_async_delegations

    record = next((r for r in list_async_delegations() if r.get("delegation_id") == target_id), None)
    if record is None:
        return {"alive": False, "evidence": "delegation record is no longer available"}
    status = str(record.get("status") or "unknown")
    if status in {"running", "finalizing"}:
        return {
            "alive": True,
            "progress": True,
            "evidence": (
                "delegation in progress "
                f"(no granular activity tracking; status={status})"
            ),
        }
    return {"alive": False, "evidence": f"delegation status={status}"}


runtime_heartbeat = RuntimeHeartbeat()
