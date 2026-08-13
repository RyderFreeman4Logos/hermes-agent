"""Shared, isolated warm-KV heartbeats for one agent owner."""

from __future__ import annotations

import copy
import logging
import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_current_owner: ContextVar[Any] = ContextVar("runtime_heartbeat_owner", default=None)


class HeartbeatConfigError(ValueError):
    """The current route has no valid warm-KV interval."""


def resolve_interval(config: dict[str, Any], provider: str) -> int:
    """Return the exact configured interval for *provider*.

    A family fallback could warm a request on the wrong route, which cannot
    preserve the route's provider-side KV cache.
    """
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_id = str(provider or "").strip().lower()
    interval = providers.get(provider_id) if isinstance(providers, dict) else None
    if not provider_id or type(interval) is not int or interval <= 0:
        raise HeartbeatConfigError(
            "runtime.warm_kv_timeout.providers requires a positive exact mapping "
            f"for {provider_id or '<missing>'}"
        )
    return interval


def _runtime_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        runtime = (load_config_readonly() or {}).get("runtime", {})
        if runtime.get("heartbeat", {}).get("enabled") is True:
            return runtime.get("warm_kv_timeout", {})
    except Exception:
        logger.debug("Could not load warm-KV heartbeat configuration", exc_info=True)
    return {}


def current_owner() -> Any:
    return _current_owner.get()


@contextmanager
def bind_owner(agent: Any):
    """Expose the active agent to tool workers spawned by its turn."""
    token = _current_owner.set(agent)
    try:
        yield
    finally:
        _current_owner.reset(token)


def _provider_id(agent: Any) -> str:
    requested = str(getattr(agent, "requested_provider", "") or "").strip().lower()
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    return requested if requested.startswith("custom:") else provider or requested


def _system_prefix(messages: Any) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
            break
        prefix.append(copy.deepcopy(message))
    return prefix


@dataclass
class _OwnerState:
    owner_ref: Callable[[], Any]
    interval: int
    children: set[tuple[str, str]] = field(default_factory=set)
    snapshot: dict[str, Any] | None = None
    timer: Any = None
    warming: bool = False


class RuntimeHeartbeat:
    """Own one timer while an agent has background work or is session-idle."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        timer_factory: Callable[[float, Callable[[], None]], Any] = threading.Timer,
    ) -> None:
        self._config = config
        self._timer_factory = timer_factory
        self._lock = threading.RLock()
        self._owners: dict[int, _OwnerState] = {}
        self._session_owners: dict[str, Callable[[], Any]] = {}
        self._child_owners: dict[tuple[str, str], Callable[[], Any]] = {}

    def _interval(self, agent: Any) -> int:
        return resolve_interval(self._config if self._config is not None else _runtime_config(), _provider_id(agent))

    @staticmethod
    def _key(agent: Any) -> int:
        return id(agent)

    @staticmethod
    def _owner_ref(agent: Any) -> Callable[[], Any]:
        try:
            return weakref.ref(agent)
        except TypeError:
            return lambda: agent

    def _state(self, agent: Any) -> _OwnerState | None:
        key = self._key(agent)
        state = self._owners.get(key)
        if state is not None and state.owner_ref() is agent:
            return state
        try:
            state = _OwnerState(owner_ref=self._owner_ref(agent), interval=self._interval(agent))
        except HeartbeatConfigError:
            return None
        self._owners[key] = state
        session_id = str(getattr(agent, "session_id", "") or "")
        if session_id:
            self._session_owners[session_id] = state.owner_ref
        return state

    def _agent_for_session(self, session_id: str) -> Any:
        owner_ref = self._session_owners.get(str(session_id or ""))
        return owner_ref() if callable(owner_ref) else None

    def timer_for(self, agent: Any) -> Any:
        with self._lock:
            state = self._owners.get(self._key(agent))
            return state.timer if state is not None and state.owner_ref() is agent else None

    def register_child(self, agent: Any, kind: str, child_id: str) -> None:
        """Keep the shared timer armed while this managed child is live."""
        if agent is None:
            return
        with self._lock:
            state = self._state(agent)
            if state is None:
                return
            state.children.add((kind, str(child_id)))
            self._child_owners[(kind, str(child_id))] = state.owner_ref
            if state.timer is None:
                self._restart_locked(agent, state)

    def register_current_child(self, kind: str, child_id: str) -> None:
        """Register a tool-created child for the agent currently running a turn."""
        self.register_child(current_owner(), kind, child_id)

    def complete_child(self, agent: Any, kind: str, child_id: str) -> None:
        """Remove one child; cancel only after the final child exits."""
        if agent is None:
            return
        with self._lock:
            state = self._owners.get(self._key(agent))
            if state is None or state.owner_ref() is not agent:
                return
            state.children.discard((kind, str(child_id)))
            self._child_owners.pop((kind, str(child_id)), None)
            if state.children:
                return
            self._cancel_locked(agent, state)

    def register_process(self, session_id: str, process_id: str) -> None:
        """Associate a managed process with its owning conversation."""
        with self._lock:
            agent = self._agent_for_session(session_id)
        if agent is not None:
            self.register_child(agent, "process", process_id)

    def complete_process(self, session_id: str, process_id: str) -> None:
        with self._lock:
            owner_ref = self._child_owners.get(("process", str(process_id)))
            agent = owner_ref() if callable(owner_ref) else self._agent_for_session(session_id)
        if agent is not None:
            self.complete_child(agent, "process", process_id)

    def complete_delegation(self, session_id: str, delegation_id: str) -> None:
        with self._lock:
            owner_ref = self._child_owners.get(("subagent", str(delegation_id)))
            agent = owner_ref() if callable(owner_ref) else self._agent_for_session(session_id)
        if agent is not None:
            self.complete_child(agent, "subagent", delegation_id)

    def capture_successful_request(self, agent: Any, api_kwargs: dict[str, Any]) -> None:
        """Remember only a cacheable prefix from a successful normal request."""
        if (
            agent is None
            or not isinstance(api_kwargs, dict)
            or getattr(agent, "api_mode", "") != "chat_completions"
        ):
            return
        with self._lock:
            state = self._state(agent)
            if state is None:
                return
            messages = _system_prefix(api_kwargs.get("messages"))
            if not messages:
                return
            state.snapshot = {
                "model": api_kwargs.get("model"),
                "messages": messages,
                "tools": copy.deepcopy(api_kwargs.get("tools")),
            }

    def on_caller_active(self, agent: Any) -> None:
        """A new caller turn supersedes any idle-session warmth."""
        with self._lock:
            state = self._owners.get(self._key(agent))
            if state is not None and state.owner_ref() is agent and not state.children:
                self._cancel_locked(agent, state)

    def on_loop_stop(self, agent: Any, *, completed: bool) -> None:
        """Restart child warmth after every parent loop; arm idle warmth on success."""
        with self._lock:
            state = self._owners.get(self._key(agent))
            if state is None or state.owner_ref() is not agent:
                return
            if state.children or (completed and state.snapshot is not None):
                self._restart_locked(agent, state)
            elif not state.children:
                self._cancel_locked(agent, state)

    def _restart_locked(self, agent: Any, state: _OwnerState) -> None:
        if state.timer is not None:
            state.timer.cancel()
        timer = self._timer_factory(state.interval, lambda: self._fire(agent))
        if hasattr(timer, "daemon"):
            timer.daemon = True
        state.timer = timer
        timer.start()

    def _cancel_locked(self, agent: Any, state: _OwnerState) -> None:
        if state.timer is not None:
            state.timer.cancel()
            state.timer = None

    def _fire(self, agent: Any) -> None:
        with self._lock:
            state = self._owners.get(self._key(agent))
            if state is None or state.owner_ref() is not agent or state.warming:
                return
            snapshot = state.snapshot
            if snapshot is None:
                return
            state.warming = True
        try:
            request = {
                "model": snapshot["model"],
                "messages": copy.deepcopy(snapshot["messages"]),
                "tools": copy.deepcopy(snapshot["tools"]),
                "tool_choice": "none",
                "max_tokens": 1,
                "stream": False,
            }
            client = agent._create_request_openai_client(
                reason="heartbeat_warm", api_kwargs=request
            )
            try:
                client.chat.completions.create(**request)
            finally:
                agent._close_request_openai_client(client, reason="heartbeat_warm_complete")
        except Exception:
            # The warm request is deliberately invisible unless it fails; callers
            # retain their normal conversation path and history either way.
            logger.warning("Warm-KV heartbeat failed for session %s", getattr(agent, "session_id", ""), exc_info=True)
        finally:
            with self._lock:
                state = self._owners.get(self._key(agent))
                if state is not None and state.owner_ref() is agent:
                    state.warming = False
                    if state.children or state.snapshot is not None:
                        self._restart_locked(agent, state)


runtime_heartbeat = RuntimeHeartbeat()
