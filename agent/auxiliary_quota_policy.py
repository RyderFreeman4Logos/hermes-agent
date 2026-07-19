"""Immutable planner for opt-in quota-only auxiliary fallback.

The legacy auxiliary router remains authoritative when ``fallback_on`` is
absent.  This module only plans the closed policy: it snapshots config and
turns configured entries into concrete routes that cannot discover ambient
providers later in the request.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

from agent.secret_scope import get_secret

__all__ = [
    "AuxiliaryPolicyError",
    "ClosedFallbackPlan",
    "FrozenRoute",
    "capture_closed_plan",
    "normalize_route_runtime",
    "parse_fallback_policy",
    "thaw_json_payload",
]

_SUPPORTED_TRIGGERS = frozenset({"quota_exhausted"})
_MISSING = object()
_JSON_THAW_MAX_DEPTH = 32
_JSON_THAW_MAX_NODES = 10_000
_DIRECT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class AuxiliaryPolicyError(ValueError):
    """Raised when a closed quota policy cannot be snapshotted safely."""


def parse_fallback_policy(config: Mapping[str, Any]) -> Optional[frozenset[str]]:
    """Return ``None`` for legacy mode and an empty set for invalid opt-in."""

    if "fallback_on" not in config:
        return None
    value = config.get("fallback_on")
    if not isinstance(value, (list, tuple)) or not value:
        return frozenset()
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return frozenset()
    triggers = frozenset(item.strip().lower() for item in value)
    return triggers if triggers <= _SUPPORTED_TRIGGERS else frozenset()


def _freeze(value: Any) -> Any:
    node_count = 0
    active_containers: set[int] = set()

    def freeze(item: Any, depth: int) -> Any:
        nonlocal node_count
        node_count += 1
        if node_count > _JSON_THAW_MAX_NODES:
            raise AuxiliaryPolicyError("Auxiliary policy graph exceeds the node limit")
        if depth > _JSON_THAW_MAX_DEPTH:
            raise AuxiliaryPolicyError("Auxiliary policy graph exceeds the depth limit")
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise AuxiliaryPolicyError(
                    "Auxiliary policy graph contains a non-finite float"
                )
            return item

        is_mapping = isinstance(item, Mapping)
        is_sequence = isinstance(item, (list, tuple))
        if not (is_mapping or is_sequence):
            raise AuxiliaryPolicyError(
                f"Auxiliary policy graph contains non-JSON value {type(item).__name__}"
            )

        container_id = id(item)
        if container_id in active_containers:
            raise AuxiliaryPolicyError("Auxiliary policy graph contains a cycle")
        active_containers.add(container_id)
        try:
            if is_mapping:
                frozen = {}
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise AuxiliaryPolicyError(
                            "Auxiliary policy graph mapping keys must be strings"
                        )
                    frozen[key] = freeze(child, depth + 1)
                return MappingProxyType(frozen)
            return tuple(freeze(child, depth + 1) for child in item)
        finally:
            active_containers.remove(container_id)

    return freeze(value, 0)


def thaw_json_payload(value: Any) -> Any:
    """Return a bounded plain-container copy for a provider request body."""

    node_count = 0
    active_containers: set[int] = set()

    def thaw(item: Any, depth: int) -> Any:
        nonlocal node_count
        node_count += 1
        if node_count > _JSON_THAW_MAX_NODES:
            raise AuxiliaryPolicyError("Auxiliary request payload exceeds the node limit")
        if depth > _JSON_THAW_MAX_DEPTH:
            raise AuxiliaryPolicyError("Auxiliary request payload exceeds the depth limit")

        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise AuxiliaryPolicyError(
                    "Auxiliary request payload contains a non-finite float"
                )
            return item

        is_mapping = isinstance(item, Mapping)
        is_sequence = isinstance(item, (list, tuple))
        if not (is_mapping or is_sequence):
            raise AuxiliaryPolicyError(
                f"Auxiliary request payload contains non-JSON value {type(item).__name__}"
            )

        container_id = id(item)
        if container_id in active_containers:
            raise AuxiliaryPolicyError("Auxiliary request payload contains a cycle")
        active_containers.add(container_id)
        try:
            if is_mapping:
                thawed = {}
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise AuxiliaryPolicyError(
                            "Auxiliary request payload mapping keys must be strings"
                        )
                    thawed[key] = thaw(child, depth + 1)
                return thawed
            return [thaw(child, depth + 1) for child in item]
        finally:
            active_containers.remove(container_id)

    return thaw(value, 0)


def normalize_route_runtime(route: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy sentinels and aliases without resolving ambient state."""

    normalized = dict(route)
    provider = normalized.get("provider")
    if isinstance(provider, str) and provider.strip().lower() == "openai":
        normalized["provider"] = "custom"
        if not normalized.get("base_url"):
            normalized["base_url"] = _DIRECT_OPENAI_BASE_URL
    model = normalized.get("model")
    if isinstance(model, str) and model.strip().lower() == "auto":
        normalized["model"] = None
    return normalized


def _route_key(route: Mapping[str, Any]) -> Any:
    direct = route.get("api_key")
    if callable(direct) or str(direct or "").strip():
        return direct
    key_env = str(route.get("key_env") or route.get("api_key_env") or "").strip()
    return get_secret(key_env) if key_env else None


def _materialize_named_custom(route: Mapping[str, Any]) -> dict[str, Any]:
    materialized = normalize_route_runtime(route)
    key = _route_key(materialized)
    if key:
        materialized["api_key"] = key

    provider = str(materialized.get("provider") or "").strip()
    lower = provider.lower()
    if lower in {"", "auto", "main", "custom"}:
        return materialized
    head, separator, tail = provider.partition(":")
    if not (separator and head.lower() == "custom" and tail.strip()):
        return materialized

    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider

        named = _get_named_custom_provider(provider)
    except Exception:
        named = None
    if not isinstance(named, Mapping):
        return materialized

    for field in ("base_url", "model", "api_mode"):
        if not materialized.get(field) and named.get(field):
            materialized[field] = named[field]
    if not materialized.get("api_key"):
        named_key = _route_key(named)
        if named_key:
            materialized["api_key"] = named_key
    return materialized


@dataclass(frozen=True)
class FrozenRoute:
    """One concrete request-local provider route."""

    index: int
    declared_provider: str
    provider: str
    model: Optional[str]
    base_url: str
    api_key: Any = None
    api_mode: Optional[str] = None
    timeout: Optional[float] = None


@dataclass(frozen=True)
class ClosedFallbackPlan:
    """All mutable routing authority captured before the primary request."""

    task: Optional[str]
    policy: frozenset[str]
    config: Mapping[str, Any]
    main_runtime: Mapping[str, Any]
    primary: Optional[FrozenRoute]
    candidates: tuple[FrozenRoute, ...]

    @property
    def quota_enabled(self) -> bool:
        return "quota_exhausted" in self.policy


def _as_timeout(value: Any) -> Optional[float]:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _route_from_mapping(
    entry: Mapping[str, Any],
    *,
    index: int,
    main_runtime: Mapping[str, Any],
    require_base_url: bool = True,
    closed_candidate: bool = False,
) -> Optional[FrozenRoute]:
    entry = normalize_route_runtime(entry)
    declared = str(entry.get("provider") or "").strip()
    normalized = declared.lower()
    model = str(entry.get("model") or "").strip()
    base_url = str(entry.get("base_url") or "").strip()
    api_key = _route_key(entry)
    api_mode = (
        str(entry.get("api_mode") or entry.get("transport") or "").strip() or None
    )
    route_timeout = entry.get("timeout")

    requested_main = normalized == "main"
    if requested_main:
        declared = "main"
        configured_model = model
        configured_api_mode = api_mode
        entry = dict(main_runtime)
        normalized = str(entry.get("provider") or "").strip().lower()
        model = configured_model or str(entry.get("model") or "").strip()
        base_url = str(entry.get("base_url") or "").strip()
        api_key = _route_key(entry)
        api_mode = (
            configured_api_mode or str(entry.get("api_mode") or "").strip() or None
        )

    head, separator, tail = normalized.partition(":")
    named_custom = separator and head == "custom" and tail not in {"", "auto", "main"}
    if normalized in {"", "auto"} or (head == "custom" and tail in {"auto", "main"}):
        return None
    if (closed_candidate and not model) or (require_base_url and not base_url):
        return None
    if (normalized == "custom" or named_custom) and not base_url:
        return None
    if closed_candidate and callable(api_key):
        return None
    if closed_candidate and requested_main and not api_key:
        return None
    if base_url and not api_key:
        # Explicitly freeze the intentional no-auth case instead of allowing
        # the generic resolver to borrow an ambient pool/env credential.
        api_key = "no-key-required"

    if closed_candidate and not (
        normalized == "custom" or named_custom or requested_main
    ):
        return None

    if normalized == "custom" or named_custom or (requested_main and closed_candidate):
        resolver_provider = "custom"
    else:
        # A base URL makes this route concrete.  Keeping the declared provider
        # preserves provider-specific wire/model behavior without credential
        # discovery because base/key are passed explicitly by the executor.
        resolver_provider = normalized

    return FrozenRoute(
        index=index,
        declared_provider=declared,
        provider=resolver_provider,
        model=model or None,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
        timeout=_as_timeout(route_timeout),
    )


def capture_closed_plan(
    task: Optional[str],
    config: Mapping[str, Any],
    *,
    main_runtime: Optional[Mapping[str, Any]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Any = None,
    api_mode: Optional[str] = None,
) -> Optional[ClosedFallbackPlan]:
    """Capture a closed plan, or return ``None`` for the untouched legacy path."""

    policy = parse_fallback_policy(config)
    if policy is None:
        return None

    runtime = normalize_route_runtime(main_runtime or {})
    task_config = _materialize_named_custom(config)
    chain = task_config.get("fallback_chain")
    materialized_chain: list[Any] = []
    if isinstance(chain, (list, tuple)):
        materialized_chain = [
            _materialize_named_custom(item) if isinstance(item, Mapping) else item
            for item in chain
        ]
    task_config["fallback_chain"] = materialized_chain

    configured_provider = str(task_config.get("provider") or "").strip().lower()
    configured_base_url = str(task_config.get("base_url") or "").strip()
    provider_changed = (
        provider is not None and str(provider).strip().lower() != configured_provider
    )
    base_url_changed = (
        base_url is not None and str(base_url).strip() != configured_base_url
    )
    if provider_changed or base_url_changed:
        primary_config = {
            "provider": (
                provider
                if provider is not None
                else ("custom" if base_url_changed else task_config.get("provider"))
            ),
            "model": model if model is not None else task_config.get("model"),
            "base_url": base_url,
            "api_key": api_key,
            "api_mode": api_mode,
        }
    else:
        primary_config = dict(task_config)
    for field, value in (
        ("provider", provider),
        ("model", model),
        ("base_url", base_url),
        ("api_key", api_key),
        ("api_mode", api_mode),
    ):
        if value is not None:
            primary_config[field] = value
    primary = _route_from_mapping(
        primary_config,
        index=-1,
        main_runtime=runtime,
        require_base_url=False,
    )

    candidates: tuple[FrozenRoute, ...] = ()
    if materialized_chain and all(
        isinstance(entry, Mapping) for entry in materialized_chain
    ):
        admitted = tuple(
            _route_from_mapping(
                entry,
                index=index,
                main_runtime=runtime,
                closed_candidate=True,
            )
            for index, entry in enumerate(materialized_chain)
        )
        if all(route is not None for route in admitted):
            candidates = tuple(route for route in admitted if route is not None)

    # Deferred candidate credentials are deliberately inadmissible above. Do
    # not retain their callable in the immutable diagnostic snapshot either.
    task_config["fallback_chain"] = [
        {
            **entry,
            "api_key": None,
        }
        if isinstance(entry, Mapping) and callable(entry.get("api_key"))
        else entry
        for entry in materialized_chain
    ]
    return ClosedFallbackPlan(
        task=task,
        policy=policy,
        config=_freeze(task_config),
        main_runtime=_freeze(runtime),
        primary=primary,
        candidates=candidates,
    )
