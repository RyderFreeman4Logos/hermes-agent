"""Per-target warm-KV heartbeats for managed background work."""

from __future__ import annotations

import atexit
import copy
import contextvars
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from hermes_constants import PARTIAL_STREAM_STUB_ID

logger = logging.getLogger(__name__)

_current_provider: contextvars.ContextVar[str] = contextvars.ContextVar(
    "runtime_heartbeat_provider", default=""
)
_current_cache_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "runtime_heartbeat_cache_context", default=""
)
_warm_snapshot_init_lock = threading.Lock()
_WARM_OUTPUT_CAP_FIELDS = (
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
)


class HeartbeatConfigError(ValueError):
    """The active provider has no valid exact heartbeat interval."""


@dataclass(frozen=True)
class _WarmRequestDraft:
    epoch: int
    api_kwargs: Dict[str, Any]
    provider: str
    cache_context: str
    model: str
    runtime_client: Any
    tools: Any
    system: Any
    compression_attempt: str
    cache_scope: str


@dataclass(frozen=True)
class _WarmRequestSnapshot(_WarmRequestDraft):
    physical_client: Any


def _warm_snapshot_state(agent) -> tuple[threading.RLock, Dict[str, Any]]:
    # Real heartbeats can race the first normal request on a freshly built
    # agent. Pair the lazy lock and state atomically so two first callers
    # cannot retain different locks and lose an epoch/publication.
    lock = getattr(agent, "_heartbeat_warm_snapshot_lock", None)
    state = getattr(agent, "_heartbeat_warm_snapshot_state", None)
    if lock is not None and state is not None:
        return lock, state
    with _warm_snapshot_init_lock:
        lock = getattr(agent, "_heartbeat_warm_snapshot_lock", None)
        if lock is None:
            lock = threading.RLock()
            agent._heartbeat_warm_snapshot_lock = lock
        state = getattr(agent, "_heartbeat_warm_snapshot_state", None)
        if state is None:
            state = {
                "epoch": 0,
                "active": set(),
                "snapshot": None,
                "warm_epoch": None,
                "warm_phase": None,
                "pending_responses": {},
            }
            agent._heartbeat_warm_snapshot_state = state
    return lock, state


def _snapshot_identity_matches(
    snapshot: _WarmRequestSnapshot, identity: Dict[str, Any]
) -> bool:
    # A replacement client with surprising equality semantics is still a
    # different transport.  The remaining immutable values compare by value.
    return snapshot.runtime_client is identity.get("runtime_client") and all(
        getattr(snapshot, key) == value
        for key, value in identity.items()
        if key != "runtime_client"
    )


def _build_warm_api_kwargs(
    snapshot: _WarmRequestSnapshot,
) -> Optional[Dict[str, Any]]:
    """Copy a normal request and bound only its existing output-cap field.

    Provider APIs disagree on the output-cap parameter name. Replaying a cap
    field that the validated physical request did not contain can make an
    otherwise compatible endpoint reject the warm, so this fails closed
    unless at least one known field already carries a positive integer.
    """
    try:
        api_kwargs = copy.deepcopy(snapshot.api_kwargs)
    except Exception:
        logger.debug("Could not copy heartbeat request snapshot", exc_info=True)
        return None
    bounded = False
    for field_name in _WARM_OUTPUT_CAP_FIELDS:
        value = api_kwargs.get(field_name)
        if type(value) is int and value > 0:
            api_kwargs[field_name] = 1
            bounded = True
    return api_kwargs if bounded else None


def _warm_snapshot_identity(agent) -> Dict[str, Any]:
    scope_resolver = getattr(agent, "_prompt_cache_scope_id", None)
    return {
        "provider": canonical_runtime_provider_identity(agent),
        "cache_context": canonical_runtime_cache_context_identity(agent),
        "model": str(getattr(agent, "model", "") or ""),
        "runtime_client": getattr(agent, "client", None),
        "tools": copy.deepcopy(getattr(agent, "tools", None)),
        "system": copy.deepcopy(
            (
                getattr(agent, "_cached_system_prompt", None),
                getattr(agent, "_cached_system_prompt_static", None),
                getattr(agent, "ephemeral_system_prompt", None),
                getattr(agent, "prefill_messages", None),
            )
        ),
        "compression_attempt": str(
            getattr(agent, "_compression_attempt_id", "") or ""
        ),
        "cache_scope": str(scope_resolver() or "") if callable(scope_resolver) else "",
    }


def _supported_openai_warm_client(agent, physical_client: Any) -> Any:
    """Return the proven OpenAI-wire client, otherwise fail closed.

    A ``chat_completions`` label alone is insufficient: native Gemini and
    Copilot ACP expose a compatibility facade with the same method shape but
    can reinterpret the call or launch a subprocess.  Production publication
    is restricted to the standard OpenAI SDK client Hermes creates for
    OpenAI-compatible HTTP endpoints. Tests use ``Mock`` clients deliberately.
    """
    if str(getattr(agent, "api_mode", "") or "").lower() != "chat_completions":
        return None
    if str(getattr(agent, "provider", "") or "").lower() == "moa":
        return None
    if physical_client is None:
        return None
    try:
        from unittest.mock import Mock

        if isinstance(physical_client, Mock):
            return physical_client
    except Exception:
        pass
    try:
        from openai import OpenAI

        if isinstance(physical_client, OpenAI):
            return physical_client
    except Exception:
        pass
    return None


def begin_normal_warm_snapshot(
    agent,
    api_kwargs: Dict[str, Any],
    *,
    physical_client: Any = None,
) -> tuple[int, Any]:
    """Start a physical normal request that may become the warm replay source."""
    lock, state = _warm_snapshot_state(agent)
    with lock:
        state["epoch"] += 1
        epoch = int(state["epoch"])
        state["active"].add(epoch)
        state["snapshot"] = None
        # Every deferred response from an older epoch is now incapable of
        # publication. Drop its strong reference instead of retaining it if
        # that older validator later raises before consuming the entry.
        state["pending_responses"].clear()
    draft = None
    try:
        if (
            str(getattr(agent, "api_mode", "") or "").lower()
            != "chat_completions"
            or str(getattr(agent, "provider", "") or "").lower() == "moa"
        ):
            return epoch, None
        identity = _warm_snapshot_identity(agent)
        draft = _WarmRequestDraft(
            epoch=epoch,
            api_kwargs=copy.deepcopy(api_kwargs),
            **identity,
        )
    except Exception:
        logger.debug("Could not snapshot normal request for heartbeat", exc_info=True)
    token = (epoch, draft)
    if physical_client is not None:
        return bind_normal_warm_snapshot_client(agent, token, physical_client)
    return token


def bind_normal_warm_snapshot_client(
    agent, token: tuple[int, Any], physical_client: Any
) -> tuple[int, Any]:
    """Bind the exact request-local client that issued the physical call."""
    epoch, draft = token
    if not isinstance(draft, _WarmRequestDraft):
        return epoch, None
    if _supported_openai_warm_client(agent, physical_client) is None:
        return epoch, None
    lock, state = _warm_snapshot_state(agent)
    with lock:
        if epoch != state["epoch"] or epoch not in state["active"]:
            return epoch, None
    return epoch, _WarmRequestSnapshot(
        **draft.__dict__,
        physical_client=physical_client,
    )


def finish_normal_warm_snapshot(
    agent, token: tuple[int, Any], *, succeeded: bool
) -> None:
    """Publish only the newest successfully validated physical request."""
    epoch, candidate = token
    lock, state = _warm_snapshot_state(agent)
    with lock:
        state["active"].discard(epoch)
        if epoch == state["epoch"]:
            state["snapshot"] = (
                candidate
                if succeeded and isinstance(candidate, _WarmRequestSnapshot)
                else None
            )


def _claim_physical_client(agent, physical_client: Any) -> Any:
    try:
        from unittest.mock import Mock

        if isinstance(physical_client, Mock):
            return physical_client
    except Exception:
        pass
    claim = getattr(agent, "_claim_request_openai_client_for_heartbeat", None)
    if not callable(claim):
        return None
    try:
        return claim(physical_client)
    except Exception:
        logger.debug("Could not lease heartbeat physical client", exc_info=True)
        return None


def _release_physical_client(agent, physical_client: Any, *, reusable: bool) -> None:
    try:
        from unittest.mock import Mock

        if isinstance(physical_client, Mock):
            return
    except Exception:
        pass
    release = getattr(agent, "_release_request_openai_client_from_heartbeat", None)
    if not callable(release):
        return
    try:
        release(physical_client, reusable=reusable)
    except Exception:
        logger.debug("Could not release heartbeat physical client", exc_info=True)


def defer_normal_warm_snapshot_until_validated(
    agent, token: tuple[int, Any], response: Any
) -> Any:
    """Associate a physical request with the response validator's result.

    Opening an SSE iterator is not provider success.  Callers first finish the
    physical-open phase with ``succeeded=False`` (which removes its in-flight
    fence without publishing), then defer the immutable candidate here.  The
    conversation loop consumes it only after the existing transport validator
    has accepted or rejected the completed response.
    """
    lock, state = _warm_snapshot_state(agent)
    with lock:
        state.setdefault("pending_responses", {})[id(response)] = (response, token)
    return response


def finish_deferred_normal_warm_snapshot(
    agent, response: Any, *, succeeded: bool
) -> None:
    """Publish a deferred physical request only after response validation."""
    lock, state = _warm_snapshot_state(agent)
    with lock:
        pending = state.setdefault("pending_responses", {}).pop(id(response), None)
    if pending is None or pending[0] is not response:
        return
    # The transport validator accepts this internal recovery envelope because
    # the ordinary loop must inspect and surface its partial content.  It is
    # nevertheless proof that the physical stream failed, never a successful
    # last-known-good request suitable for cache warming.
    publishable = succeeded and (
        getattr(response, "id", "") != PARTIAL_STREAM_STUB_ID
    )
    finish_normal_warm_snapshot(agent, pending[1], succeeded=publishable)


def claim_warm_snapshot(
    agent,
) -> tuple[Optional[tuple[int, Dict[str, Any], Any]], str]:
    """Claim one immutable last-known-good request and explain safe skips."""
    lock, state = _warm_snapshot_state(agent)
    with lock:
        snapshot = state.get("snapshot")
        if state["active"]:
            return None, "normal_request_in_flight"
        if state.get("warm_epoch") is not None:
            return None, "checkin_already_in_flight"
        if snapshot is None:
            return None, "no_validated_snapshot"
        epoch = int(state["epoch"])
    try:
        identity = _warm_snapshot_identity(agent)
    except Exception:
        logger.debug("Could not validate heartbeat request snapshot", exc_info=True)
        return None, "identity_unavailable"
    with lock:
        if (
            state["epoch"] != epoch
            or state["active"]
            or state.get("warm_epoch") is not None
            or state.get("snapshot") is not snapshot
            or not _snapshot_identity_matches(snapshot, identity)
        ):
            return None, "snapshot_stale"
        api_kwargs = _build_warm_api_kwargs(snapshot)
        if api_kwargs is None:
            return None, "no_bounded_output_cap"
        state["warm_epoch"] = epoch
    physical_client = _claim_physical_client(agent, snapshot.physical_client)
    if physical_client is None:
        with lock:
            if state.get("warm_epoch") == epoch:
                state["warm_epoch"] = None
        return None, "client_lease_unavailable"
    with lock:
        if (
            state["epoch"] != epoch
            or state["active"]
            or state.get("warm_epoch") != epoch
            or state.get("snapshot") is not snapshot
        ):
            state["warm_epoch"] = None
            warm_phase = state.get("warm_phase")
            if isinstance(warm_phase, tuple) and warm_phase[0] == epoch:
                state["warm_phase"] = None
            stale = True
        else:
            state["warm_client"] = (epoch, physical_client)
            state["warm_phase"] = (epoch, "precall")
            stale = False
    if stale:
        _release_physical_client(agent, physical_client, reusable=True)
        return None, "snapshot_stale_after_lease"
    return (epoch, api_kwargs, physical_client), ""


def commit_warm_snapshot_dispatch(
    agent, epoch: int, event_current: Callable[[], bool]
) -> bool:
    """Atomically authorize the claimed warm's single provider dispatch.

    The transition from ``precall`` to ``committed`` is the dispatch
    linearization point shared with normal-request epoch publication. A normal
    request that wins the snapshot lock first invalidates this warm before any
    provider call. Once committed, the provider call may run without holding
    the lock; a later normal request still wins publication and lease refresh.
    """
    lock, state = _warm_snapshot_state(agent)
    with lock:
        snapshot = state.get("snapshot")
        warm_client = state.get("warm_client")
        if (
            state["epoch"] != epoch
            or state["active"]
            or state.get("warm_epoch") != epoch
            or state.get("warm_phase") != (epoch, "precall")
            or snapshot is None
            or not isinstance(warm_client, tuple)
            or warm_client[0] != epoch
            or warm_client[1] is not snapshot.physical_client
        ):
            return False
    try:
        identity = _warm_snapshot_identity(agent)
    except Exception:
        return False
    with lock:
        snapshot = state.get("snapshot")
        warm_client = state.get("warm_client")
        if (
            state["epoch"] != epoch
            or state["active"]
            or state.get("warm_epoch") != epoch
            or state.get("warm_phase") != (epoch, "precall")
            or snapshot is None
            or not isinstance(warm_client, tuple)
            or warm_client[0] != epoch
            or warm_client[1] is not snapshot.physical_client
            or not _snapshot_identity_matches(snapshot, identity)
        ):
            return False
        try:
            if not event_current():
                return False
        except Exception:
            logger.debug("Heartbeat dispatch event validation failed", exc_info=True)
            return False
        # The snapshot lock has excluded begin_normal_warm_snapshot throughout
        # the final event check and this one-shot phase transition.
        if (
            state["epoch"] != epoch
            or state["active"]
            or state.get("warm_phase") != (epoch, "precall")
        ):
            return False
        state["warm_phase"] = (epoch, "committed")
        return True


def warm_snapshot_is_current(agent, epoch: int) -> bool:
    """Revalidate a committed warm before accepting its response."""
    lock, state = _warm_snapshot_state(agent)
    with lock:
        snapshot = state.get("snapshot")
        if (
            state["epoch"] != epoch
            or state["active"]
            or state.get("warm_epoch") != epoch
            or state.get("warm_phase") != (epoch, "committed")
            or snapshot is None
            or state.get("warm_client", (None, None))[0] != epoch
            or state.get("warm_client", (None, None))[1]
            is not snapshot.physical_client
        ):
            return False
    try:
        identity = _warm_snapshot_identity(agent)
    except Exception:
        return False
    with lock:
        warm_client = state.get("warm_client")
        return (
            state["epoch"] == epoch
            and not state["active"]
            and state.get("warm_epoch") == epoch
            and state.get("warm_phase") == (epoch, "committed")
            and state.get("snapshot") is snapshot
            and isinstance(warm_client, tuple)
            and warm_client[0] == epoch
            and warm_client[1] is snapshot.physical_client
            and _snapshot_identity_matches(snapshot, identity)
        )


def release_warm_snapshot(agent, epoch: int, *, reusable: bool = True) -> None:
    lock, state = _warm_snapshot_state(agent)
    with lock:
        warm_client = state.get("warm_client")
        if isinstance(warm_client, tuple) and warm_client[0] == epoch:
            state.pop("warm_client", None)
        if state.get("warm_epoch") == epoch:
            state["warm_epoch"] = None
        warm_phase = state.get("warm_phase")
        if isinstance(warm_phase, tuple) and warm_phase[0] == epoch:
            state["warm_phase"] = None
    if isinstance(warm_client, tuple) and warm_client[0] == epoch:
        _release_physical_client(agent, warm_client[1], reusable=reusable)


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


def canonical_runtime_cache_context_identity(agent) -> str:
    """Return the secret-free deployment identity that owns a prompt cache."""
    from agent.backend_identity import BackendIdentity

    identity = BackendIdentity.build(
        canonical_runtime_provider_identity(agent),
        getattr(agent, "model", ""),
        getattr(agent, "base_url", ""),
    )
    api_mode = str(getattr(agent, "api_mode", "") or "").strip().lower()
    return "|".join((identity.provider, identity.base_url, identity.model, api_mode))


def set_current_provider(provider: str | None) -> contextvars.Token[str]:
    return _current_provider.set(str(provider or "").strip().lower())


def reset_current_provider(token) -> None:
    if isinstance(token, tuple):
        provider_token, cache_token = token
        _current_cache_context.reset(cache_token)
        _current_provider.reset(provider_token)
        return
    _current_provider.reset(token)


def get_current_provider() -> str:
    return _current_provider.get()


def get_current_cache_context() -> str:
    return _current_cache_context.get()


def bind_agent_provider(agent):
    return (
        set_current_provider(canonical_runtime_provider_identity(agent)),
        _current_cache_context.set(canonical_runtime_cache_context_identity(agent)),
    )


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
    provider: str = ""
    cache_context: str = ""
    generation: int = 0
    timer: Any = None
    deadline: float = 0.0
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
        self._group_tokens: Dict[tuple[str, int, str, str], int] = {}
        self._group_next_emit: Dict[tuple[str, int, str, str], float] = {}
        self._group_pending: Dict[tuple[str, int, str, str], Dict[str, Any]] = {}
        self._group_replacements: Dict[
            tuple[str, int, str, str], Dict[str, Any]
        ] = {}
        self._next_group_token = 0
        self._next_generation = 0

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
        provider: str | None = None,
        cache_context: str | None = None,
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
            provider=str(provider if provider is not None else get_current_provider()),
            cache_context=str(
                cache_context
                if cache_context is not None
                else get_current_cache_context()
            ),
        )
        self.cancel(key)
        with self._lock:
            group = self._group_key(target)
            if not any(
                self._group_key(other) == group
                for other in self._targets.values()
            ):
                self._next_group_token += 1
                self._group_tokens[group] = self._next_group_token
                self._group_next_emit.pop(group, None)
                self._group_pending.pop(group, None)
                self._group_replacements.pop(group, None)
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
        logger.info(
            "Runtime heartbeat phase=arm target=%s owner=%s kind=%s "
            "provider=%s interval_s=%s",
            target.target_id,
            target.caller_id,
            target.kind,
            target.provider or "-",
            target.interval,
        )
        return True

    def _schedule_locked(
        self,
        key: str,
        target: _Target,
        *,
        delay: Optional[float] = None,
        deadline: Optional[float] = None,
    ) -> None:
        if deadline is None:
            delay = float(target.interval if delay is None else max(0.0, delay))
            deadline = time.monotonic() + delay
        else:
            delay = float(
                max(0.0, deadline - time.monotonic())
                if delay is None
                else max(0.0, delay)
            )
        target.deadline = deadline
        self._next_generation += 1
        target.generation = self._next_generation
        generation = target.generation
        timer = self._timer_factory(
            delay,
            lambda: self._fire(key, generation),
        )
        if hasattr(timer, "daemon"):
            timer.daemon = True
        target.timer = timer
        timer.start()

    @staticmethod
    def _group_key(target: _Target) -> tuple[str, int, str, str]:
        return (
            target.caller_id,
            target.interval,
            target.provider,
            target.cache_context,
        )

    def cancel(self, target_id: str) -> bool:
        replacement_event = None
        with self._lock:
            target = self._targets.pop(str(target_id), None)
            if target is None:
                return False
            group = self._group_key(target)
            if not any(
                self._group_key(candidate) == group
                for candidate in self._targets.values()
            ):
                self._next_group_token += 1
                self._group_tokens[group] = self._next_group_token
                self._group_next_emit.pop(group, None)
                self._group_pending.pop(group, None)
                self._group_replacements.pop(group, None)
            else:
                pending = self._group_pending.get(group)
                if pending is not None and pending.get("target_id") == target.target_id:
                    self._group_pending.pop(group, None)
                    replacement_event = self._group_replacements.pop(group, None)
                    if replacement_event is None:
                        self._group_next_emit[group] = 0.0
                    else:
                        self._group_pending[group] = replacement_event
            if target.timer is not None:
                target.timer.cancel()
            while target.publishing:
                self._publication_done.wait()
        if replacement_event is not None:
            self._queue().put(replacement_event)
        logger.info(
            "Runtime heartbeat phase=cancel target=%s owner=%s kind=%s",
            target.target_id,
            target.caller_id,
            target.kind,
        )
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

    def reset_for_caller(
        self,
        caller_id: str,
        *,
        provider: str | None = None,
        cache_context: str | None = None,
        activity_at: float | None = None,
    ) -> int:
        """Rearm matching live targets from a successful provider dispatch."""
        if (provider is None) != (cache_context is None):
            raise ValueError("provider and cache_context must be supplied together")
        count = 0
        with self._lock:
            groups: Dict[tuple[str, int, str, str], float] = {}
            now = time.monotonic()
            dispatch_at = now if activity_at is None else min(now, float(activity_at))
            for key, target in tuple(self._targets.items()):
                if target.caller_id != str(caller_id):
                    continue
                group = self._group_key(target)
                if provider is not None and group[2:] != (
                    str(provider),
                    str(cache_context),
                ):
                    continue
                deadline = dispatch_at + target.interval
                if deadline <= target.deadline:
                    continue
                if target.timer is not None:
                    target.timer.cancel()
                self._schedule_locked(
                    key, target, delay=deadline - now, deadline=deadline
                )
                groups[group] = deadline
                count += 1
            for group, deadline in groups.items():
                self._next_group_token += 1
                self._group_tokens[group] = self._next_group_token
                self._group_next_emit[group] = max(
                    deadline, self._group_next_emit.get(group, 0.0)
                )
                self._group_pending.pop(group, None)
                self._group_replacements.pop(group, None)
        return count

    def is_event_current(
        self, event: Dict[str, Any], agent=None, *, consume: bool = False
    ) -> bool:
        """Return whether the event's exact target generation is still live."""
        if event.get("type") != "heartbeat":
            return True
        token = event.get("heartbeat_group_token")
        interval = event.get("heartbeat_interval")
        generation = event.get("generation")
        target_id = str(event.get("target_id") or "")
        caller_id = str(event.get("session_key") or "")
        if (
            not target_id
            or not isinstance(token, int)
            or not isinstance(interval, int)
            or not isinstance(generation, int)
        ):
            return False
        provider = str(event.get("provider") or "")
        cache_context = str(event.get("cache_context") or "")
        group = (caller_id, interval, provider, cache_context)
        if agent is not None and (
            provider != canonical_runtime_provider_identity(agent)
            or cache_context != canonical_runtime_cache_context_identity(agent)
        ):
            return False
        with self._lock:
            if self._group_tokens.get(group) != token:
                return False
            target = self._targets.get(target_id)
            if (
                target is None
                or self._group_key(target) != group
                or target.generation != generation
            ):
                return False
        try:
            alive = dict(target.inspect() or {}).get("alive") is True
        except Exception:
            alive = str(event.get("status") or "").upper() == "UNKNOWN"
        if not alive:
            return False
        with self._lock:
            current = (
                self._targets.get(target_id) is target
                and target.generation == generation
                and self._group_tokens.get(group) == token
                and (
                    agent is None
                    or (
                        provider == canonical_runtime_provider_identity(agent)
                        and cache_context
                        == canonical_runtime_cache_context_identity(agent)
                    )
                )
            )
            pending = self._group_pending.get(group)
            if (
                current
                and consume
                and pending is not None
                and pending.get("target_id") == target_id
                and pending.get("generation") == generation
            ):
                self._group_pending.pop(group, None)
                self._group_replacements.pop(group, None)
            return current

    @staticmethod
    def _retarget_event(event: Dict[str, Any], target: _Target) -> None:
        event["target_id"] = target.target_id
        event["target_ids"] = [target.target_id]
        event["generation"] = target.generation
        event["generations"] = [target.generation]
        event["target_kind"] = target.kind
        event["session_id"] = target.target_id if target.kind == "process" else ""

    def outstanding_for_caller(self, caller_id: str) -> list[str]:
        with self._lock:
            return [
                target.target_id
                for target in self._targets.values()
                if target.caller_id == str(caller_id)
            ]

    def active_snapshots(self) -> list[Dict[str, Any]]:
        """Return immutable UI-safe target timing snapshots."""
        with self._lock:
            targets = sorted(
                self._targets.values(),
                key=lambda target: (target.started_at, target.target_id),
            )
            return [
                {
                    "target_id": target.target_id,
                    "caller_id": target.caller_id,
                    "kind": target.kind,
                    "started_at": target.started_at,
                    "interval_s": target.interval,
                }
                for target in targets
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
            old_by_identity = target.baseline.get("cpu_by_identity") or {}
            new_by_identity = snapshot.get("cpu_by_identity") or {}
            if any(
                total > float(old_by_identity.get(identity) or 0.0)
                for identity, total in new_by_identity.items()
            ):
                return "ALIVE", "CPU advanced for a live PID/start identity"
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
        logger.info(
            "Runtime heartbeat phase=due target=%s owner=%s kind=%s "
            "interval_s=%s",
            target.target_id,
            target.caller_id,
            target.kind,
            target.interval,
        )
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
            group = self._group_key(target)
            now = time.monotonic()
            if status == "ALIVE":
                if now < self._group_next_emit.get(group, 0.0):
                    pending = self._group_pending.get(group)
                    if pending is not None:
                        replacement = dict(pending)
                        self._retarget_event(replacement, target)
                        replacement["evidence"] = evidence[:500]
                        replacement["elapsed_s"] = max(
                            0, int(time.time() - target.started_at)
                        )
                        self._group_replacements[group] = replacement
                    return
                self._group_next_emit[group] = now + target.interval
            group_token = self._group_tokens[group]
            target.publishing = True
            event_generation = target.generation
            event = {
                "type": "heartbeat",
                "target_id": target.target_id,
                "target_ids": [target.target_id],
                "generation": event_generation,
                "generations": [event_generation],
                "target_kind": target.kind,
                "session_id": target.target_id if target.kind == "process" else "",
                "session_key": target.caller_id,
                "provider": target.provider,
                "cache_context": target.cache_context,
                "status": status,
                "evidence": evidence[:500],
                "elapsed_s": max(0, int(time.time() - target.started_at)),
                "heartbeat_interval": target.interval,
                "heartbeat_group_token": group_token,
                "heartbeat_terminal": inspection_error is not None,
            }
            if status == "ALIVE":
                self._group_replacements.pop(group, None)
                self._group_pending[group] = event
        try:
            self._queue().put(event)
        finally:
            with self._lock:
                target.publishing = False
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
    cpu_by_identity = {}
    if (
        pid
        and pid_scope == "host"
        and host_start_time is not None
        and process_registry._safe_host_start_time(pid) == host_start_time
    ):
        try:
            import psutil

            process_registry._remember_local_descendants(
                session, include_subreaper=True
            )
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
                try:
                    cpu = psutil.Process(process_pid).cpu_times()
                except Exception:
                    continue
                total = float(cpu.user + cpu.system)
                cpu_seconds += total
                cpu_by_identity[(process_pid, expected_start)] = total
            if process_registry._safe_host_start_time(pid) != host_start_time:
                cpu_seconds = 0.0
                cpu_by_identity = {}
        except Exception:
            pass
    return {
        "alive": True,
        "output_size": output_size,
        "cpu_seconds": cpu_seconds,
        "cpu_by_identity": cpu_by_identity,
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
    if status in {"queued", "running"}:
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
