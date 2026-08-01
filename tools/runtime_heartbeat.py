"""Per-target warm-KV heartbeats for managed background work."""

from __future__ import annotations

import atexit
import contextvars
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_current_provider: contextvars.ContextVar[str] = contextvars.ContextVar(
    "runtime_heartbeat_provider", default=""
)


class HeartbeatConfigError(ValueError):
    """The active provider has no valid exact heartbeat interval."""


def canonical_runtime_provider_identity(agent) -> str:
    """Return the active provider's existing canonical routing identity."""
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    requested = str(
        getattr(agent, "requested_provider", "") or ""
    ).strip().lower()
    if requested.startswith("custom:"):
        return requested
    if provider == "custom" or requested == "custom":
        from hermes_cli.runtime_provider import canonical_custom_identity

        recovered = canonical_custom_identity(
            base_url=getattr(agent, "base_url", None),
            model=getattr(agent, "model", None),
        )
        return str(recovered or requested or provider).strip().lower()
    return provider or requested


def set_current_provider(provider: str | None) -> contextvars.Token[str]:
    return _current_provider.set(str(provider or "").strip().lower())


def reset_current_provider(token: contextvars.Token[str]) -> None:
    _current_provider.reset(token)


def get_current_provider() -> str:
    return _current_provider.get()


def bind_agent_provider(agent) -> contextvars.Token[str]:
    return set_current_provider(canonical_runtime_provider_identity(agent))


def resolve_heartbeat_interval(
    runtime_config: Optional[Dict[str, Any]], provider: str | None
) -> Optional[int]:
    """Resolve only the exact positive-integer provider mapping."""
    runtime = runtime_config if isinstance(runtime_config, dict) else {}
    heartbeat = runtime.get("heartbeat")
    if not isinstance(heartbeat, dict) or heartbeat.get("enabled") is not True:
        return None
    if heartbeat.get("mode", "per_target") != "per_target":
        return None

    provider_id = str(provider or "").strip().lower()
    providers = (runtime.get("warm_kv_timeout") or {}).get("providers")
    if not isinstance(providers, dict):
        providers = {}
    exact = {
        str(key).strip().lower(): value for key, value in providers.items()
    }.get(provider_id)
    if (
        not provider_id
        or not isinstance(exact, int)
        or isinstance(exact, bool)
        or exact <= 0
    ):
        raise HeartbeatConfigError(
            "runtime.warm_kv_timeout.providers must contain a positive "
            f"integer exact mapping for canonical provider {provider_id or '<missing>'}"
        )
    return exact


def _runtime_config() -> Dict[str, Any]:
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly() or {}
    runtime = config.get("runtime") if isinstance(config, dict) else None
    return runtime if isinstance(runtime, dict) else {}


def preflight_current_heartbeat() -> Optional[int]:
    """Validate the current provider before a managed target is created."""
    return resolve_heartbeat_interval(_runtime_config(), get_current_provider())


@dataclass
class _Target:
    target_id: str
    caller_id: str
    kind: str
    interval: int
    inspect: Callable[[], Dict[str, Any]]
    started_at: float
    generation: int = 0
    timer: Any = None
    baseline: Dict[str, Any] = field(default_factory=dict)
    publishing: bool = False


class RuntimeHeartbeat:
    """One process coordinator with isolated per-owner, per-target timers."""

    def __init__(
        self,
        *,
        event_queue: Optional[queue.Queue] = None,
        timer_factory: Callable[[float, Callable[[], None]], Any] = threading.Timer,
    ) -> None:
        self._event_queue = event_queue
        self._timer_factory = timer_factory
        self._lock = threading.RLock()
        self._publication_done = threading.Condition(self._lock)
        self._targets: Dict[str, _Target] = {}

    def _queue(self):
        if self._event_queue is not None:
            return self._event_queue
        from tools.process_registry import process_registry

        return process_registry.completion_queue

    def arm(
        self,
        target_id: str,
        *,
        caller_id: str,
        kind: str,
        interval: Optional[int],
        inspect: Callable[[], Dict[str, Any]],
    ) -> bool:
        """Arm after target creation using its already validated interval."""
        if interval is None:
            return False
        if (
            not target_id
            or not caller_id
            or not isinstance(interval, int)
            or isinstance(interval, bool)
            or interval <= 0
        ):
            raise ValueError("heartbeat target, owner, and interval must be valid")
        key = str(target_id)
        target = _Target(
            target_id=key,
            caller_id=str(caller_id),
            kind=str(kind),
            interval=interval,
            inspect=inspect,
            started_at=time.time(),
        )
        self.cancel(key)
        with self._lock:
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
        timer = self._timer_factory(
            target.interval, lambda: self._fire(key, generation)
        )
        if hasattr(timer, "daemon"):
            timer.daemon = True
        target.timer = timer
        timer.start()

    def cancel(self, target_id: str) -> bool:
        with self._lock:
            target = self._targets.pop(str(target_id), None)
            if target is None:
                return False
            if target.timer is not None:
                target.timer.cancel()
            while target.publishing:
                self._publication_done.wait()
            return True

    def cancel_for_caller(self, caller_id: str) -> int:
        with self._lock:
            target_ids = [
                target.target_id
                for target in self._targets.values()
                if target.caller_id == str(caller_id)
            ]
        return sum(self.cancel(target_id) for target_id in target_ids)

    def cancel_all(self) -> int:
        with self._lock:
            target_ids = list(self._targets)
        return sum(self.cancel(target_id) for target_id in target_ids)

    def reset_for_caller(self, caller_id: str) -> int:
        """Rearm one owner's live targets with each stored exact interval."""
        count = 0
        with self._lock:
            for key, target in tuple(self._targets.items()):
                if target.caller_id != str(caller_id):
                    continue
                if target.timer is not None:
                    target.timer.cancel()
                self._schedule_locked(key, target)
                count += 1
        return count

    def outstanding_for_caller(self, caller_id: str) -> list[str]:
        with self._lock:
            return [
                target.target_id
                for target in self._targets.values()
                if target.caller_id == str(caller_id)
            ]

    @staticmethod
    def _assess(target: _Target, snapshot: Dict[str, Any]) -> tuple[str, str]:
        if target.kind == "process":
            old_output = int(target.baseline.get("output_size") or 0)
            new_output = int(snapshot.get("output_size") or 0)
            old_cpu = float(target.baseline.get("cpu_seconds") or 0.0)
            new_cpu = float(snapshot.get("cpu_seconds") or 0.0)
            if new_output > old_output:
                return "ALIVE", f"output grew {old_output}->{new_output} bytes"
            if new_cpu > old_cpu:
                return "ALIVE", f"CPU advanced {old_cpu:.2f}->{new_cpu:.2f}s"
            return "STUCK", str(
                snapshot.get("evidence")
                or "process is live but produced no output or CPU progress"
            )
        if snapshot.get("progress") is True:
            return "ALIVE", str(
                snapshot.get("evidence") or "delegation in progress"
            )
        return "STUCK", str(snapshot.get("evidence") or "target made no progress")

    def _fire(self, key: str, generation: int) -> None:
        with self._lock:
            target = self._targets.get(key)
            if target is None or target.generation != generation:
                return
            target.timer = None
        inspection_error = None
        try:
            snapshot = dict(target.inspect() or {})
        except Exception as exc:
            inspection_error = exc
            snapshot = {
                "alive": True,
                "evidence": f"heartbeat inspection failed: {type(exc).__name__}: {exc}",
            }
        if snapshot.get("alive") is not True:
            self.cancel(key)
            return
        if inspection_error is not None:
            status, evidence = "UNKNOWN", str(snapshot["evidence"])
        else:
            status, evidence = self._assess(target, snapshot)
        with self._lock:
            current = self._targets.get(key)
            if current is not target or current.generation != generation:
                return
            if inspection_error is None:
                target.baseline = snapshot
                self._schedule_locked(key, target)
            target.publishing = True
        try:
            self._queue().put(
                {
                    "type": "heartbeat",
                    "target_id": target.target_id,
                    "target_kind": target.kind,
                    "session_id": target.target_id if target.kind == "process" else "",
                    "session_key": target.caller_id,
                    "status": status,
                    "evidence": evidence[:500],
                    "elapsed_s": max(0, int(time.time() - target.started_at)),
                }
            )
        finally:
            with self._lock:
                target.publishing = False
                if inspection_error is not None and self._targets.get(key) is target:
                    self._targets.pop(key, None)
                self._publication_done.notify_all()


def inspect_process(target_id: str) -> Dict[str, Any]:
    from tools.process_registry import process_registry

    session = process_registry.get(target_id)
    if session is None:
        return {"alive": False, "evidence": "process is not registered"}
    with session._lock:
        exited = bool(session.exited)
        output_size = int(session.output_size)
        pid = session.pid
        pid_scope = session.pid_scope
        host_start_time = session.host_start_time
    if exited:
        return {"alive": False, "evidence": "process exited"}
    cpu_seconds = 0.0
    if pid and pid_scope == "host":
        try:
            import psutil

            process_registry._remember_local_descendants(session)
            with session._lock:
                tracked = dict(session._tracked_descendants)
            processes = [(pid, host_start_time), *tracked.items()]
            for process_pid, expected_start in processes:
                if process_pid != pid and expected_start is None:
                    continue
                if (
                    expected_start is not None
                    and process_registry._safe_host_start_time(process_pid)
                    != expected_start
                ):
                    continue
                cpu = psutil.Process(process_pid).cpu_times()
                cpu_seconds += float(cpu.user + cpu.system)
        except Exception:
            pass
    return {
        "alive": True,
        "output_size": output_size,
        "cpu_seconds": cpu_seconds,
    }


def inspect_delegation(target_id: str) -> Dict[str, Any]:
    from tools.async_delegation import list_async_delegations

    record = next(
        (
            item
            for item in list_async_delegations()
            if item.get("delegation_id") == target_id
        ),
        None,
    )
    status = str((record or {}).get("status") or "")
    if status == "running":
        return {
            "alive": True,
            "progress": True,
            "evidence": f"delegation in progress; status={status}",
        }
    if status == "finalizing":
        return {
            "alive": True,
            "progress": True,
            "evidence": "delegation finalizing",
        }
    if status == "stalling":
        return {
            "alive": True,
            "progress": False,
            "evidence": "delegation interrupt requested; status=stalling",
        }
    return {"alive": False, "evidence": f"delegation status={status or 'missing'}"}


runtime_heartbeat = RuntimeHeartbeat()
atexit.register(runtime_heartbeat.cancel_all)
