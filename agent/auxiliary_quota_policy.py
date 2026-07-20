"""Immutable planner for opt-in quota-only auxiliary fallback.

The legacy auxiliary router remains authoritative when ``fallback_on`` is
absent.  This module only plans the closed policy: it snapshots config and
turns configured entries into concrete routes that cannot discover ambient
providers later in the request.
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

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

    for field in (
        "base_url",
        "model",
        "api_mode",
        "timeout",
        "max_output_tokens",
        "supports_vision",
        "context_length",
        "source",
    ):
        if materialized.get(field) is None and named.get(field) is not None:
            materialized[field] = named[field]
        elif not materialized.get(field) and named.get(field):
            materialized[field] = named[field]
    for field in ("request_overrides", "extra_body", "extra_headers"):
        merged: dict[str, Any] = {}
        if isinstance(named.get(field), Mapping):
            merged.update(named[field])
        if isinstance(materialized.get(field), Mapping):
            merged.update(materialized[field])
        if merged:
            materialized[field] = merged
    if not materialized.get("api_key"):
        named_key = _route_key(named)
        if named_key:
            materialized["api_key"] = named_key
    return materialized


@dataclass(frozen=True)
class FrozenCredentialBinding:
    """Secret-safe credential captured for exactly one route."""

    kind: str
    identity: str
    source: str
    value: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class FrozenRoute:
    """One concrete request-local provider route."""

    index: int
    declared_provider: str
    provider: str
    source: str
    model: str
    base_url: str
    credential: FrozenCredentialBinding = field(repr=False)
    api_mode: Optional[str] = None
    timeout: Optional[float] = None
    request_overrides: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )
    extra_body: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )
    extra_headers: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )
    max_output_tokens: Optional[int] = None
    supports_vision: Optional[bool] = None
    context_length: Optional[int] = None
    refresh_policy: str = "none"

    @property
    def api_key(self) -> Any:
        if self.credential.kind == "no_auth":
            return "no-key-required"
        return self.credential.value


@dataclass(frozen=True)
class ClosedFallbackPlan:
    """All mutable routing authority captured before the primary request."""

    task: Optional[str]
    policy: frozenset[str]
    config: Mapping[str, Any] = field(repr=False)
    main_runtime: Mapping[str, Any] = field(repr=False)
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
    return timeout if math.isfinite(timeout) and timeout > 0 else None


def _as_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _plain_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _frozen_mapping(value: Any) -> Mapping[str, Any]:
    return _freeze(value) if isinstance(value, Mapping) else MappingProxyType({})


def _is_explicit_local_route(provider: str, base_url: str) -> bool:
    if provider.lower() in {"local", "ollama", "lmstudio", "llama.cpp", "vllm"}:
        return True
    try:
        host = (urlparse(base_url).hostname or "").lower()
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _credential_binding(
    value: Any,
    *,
    source: str,
    no_auth: bool,
) -> Optional[FrozenCredentialBinding]:
    if no_auth:
        return FrozenCredentialBinding("no_auth", f"no-auth:{source}", source)
    if callable(value):
        target = getattr(value, "__func__", value)
        label = f"{type(target).__module__}.{type(target).__qualname__}:{id(target)}"
        digest = hashlib.sha256(f"{source}:{label}".encode()).hexdigest()
        return FrozenCredentialBinding("callable", f"callable:{digest}", source, value)
    if isinstance(value, str) and value.strip() and value != "no-key-required":
        digest = hashlib.sha256(
            source.encode() + b"\0" + value.encode("utf-8", "surrogatepass")
        ).hexdigest()
        return FrozenCredentialBinding("static", f"static:{digest}", source, value)
    return None


_DEFAULT_ENDPOINTS = {
    "anthropic": ("https://api.anthropic.com", "anthropic"),
    "nous": ("https://inference-api.nousresearch.com/v1", None),
    "openai-codex": ("https://chatgpt.com/backend-api/codex", "codex"),
    "openrouter": ("https://openrouter.ai/api/v1", None),
    "xai": ("https://api.x.ai/v1", None),
    "xai-oauth": ("https://api.x.ai/v1", None),
}
_OPENROUTER_FROZEN_HEADERS = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "X-OpenRouter-Categories": "productivity,cli-agent",
}


def _route_from_mapping(
    entry: Mapping[str, Any],
    *,
    index: int,
    main_runtime: Mapping[str, Any],
    require_base_url: bool = True,
    closed_candidate: bool = False,
) -> Optional[FrozenRoute]:
    original = dict(entry)
    declared = str(original.get("provider") or "").strip()
    requested_main = declared.lower() == "main"
    if requested_main:
        configured = original
        original = dict(main_runtime)
        for name in (
            "model",
            "api_mode",
            "transport",
            "timeout",
            "request_overrides",
            "extra_body",
            "extra_headers",
            "max_output_tokens",
            "supports_vision",
            "context_length",
        ):
            if name in configured and configured.get(name) is not None:
                original[name] = configured[name]
    original = _materialize_named_custom(original)
    entry = normalize_route_runtime(original)
    resolved = str(entry.get("provider") or "").strip().lower()
    if requested_main:
        declared = "main"
    if resolved in {"", "auto", "main"}:
        return None
    head, separator, tail = resolved.partition(":")
    named_custom = separator and head == "custom" and tail not in {"", "auto", "main"}
    if head == "custom" and tail in {"auto", "main"}:
        return None

    model = str(entry.get("model") or "").strip()
    base_url = str(entry.get("base_url") or "").strip()
    api_mode = (
        str(entry.get("api_mode") or entry.get("transport") or "").strip() or None
    )
    if not base_url and resolved in _DEFAULT_ENDPOINTS:
        base_url, default_mode = _DEFAULT_ENDPOINTS[resolved]
        api_mode = api_mode or default_mode
    if not model:
        frozen_main_model = str(main_runtime.get("model") or "").strip()
        if frozen_main_model:
            model = frozen_main_model
        elif resolved == "custom" and base_url == _DIRECT_OPENAI_BASE_URL:
            model = "gpt-4o-mini"
    if not model or (require_base_url and not base_url):
        return None
    if (resolved == "custom" or named_custom) and not base_url:
        return None
    if resolved in {"bedrock", "copilot-acp"} or api_mode in {
        "bedrock",
        "process",
        "external-process",
    }:
        return None

    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    direct_key = entry.get("api_key")
    if key_env:
        api_key = get_secret(key_env)
        if not api_key:
            return None
        credential_source = f"env:{key_env}"
    else:
        api_key = direct_key
        credential_source = str(entry.get("credential_source") or declared or resolved)
    no_auth = _is_explicit_local_route(declared or resolved, base_url) and not api_key
    binding = _credential_binding(
        api_key,
        source=credential_source,
        no_auth=no_auth,
    )
    if binding is None:
        return None

    resolver_provider = "custom" if named_custom else resolved
    source = str(entry.get("source") or declared or resolved)
    extra_headers = (
        dict(_OPENROUTER_FROZEN_HEADERS) if resolved == "openrouter" else {}
    )
    extra_headers.update(_plain_mapping(entry.get("extra_headers")))
    return FrozenRoute(
        index=index,
        declared_provider=declared or resolved,
        provider=resolver_provider,
        source=source,
        model=model,
        base_url=base_url,
        credential=binding,
        api_mode=api_mode,
        timeout=_as_timeout(entry.get("timeout")),
        request_overrides=_frozen_mapping(entry.get("request_overrides")),
        extra_body=_frozen_mapping(entry.get("extra_body")),
        extra_headers=_frozen_mapping(extra_headers),
        max_output_tokens=_as_positive_int(entry.get("max_output_tokens")),
        supports_vision=(
            entry.get("supports_vision")
            if isinstance(entry.get("supports_vision"), bool)
            else None
        ),
        context_length=_as_positive_int(entry.get("context_length")),
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
    task_config = dict(config)
    chain = task_config.get("fallback_chain")
    materialized_chain: list[Any] = []
    if isinstance(chain, (list, tuple)):
        materialized_chain = [
            dict(item) if isinstance(item, Mapping) else item
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

    def diagnostic_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = dict(value)
        if "api_key" in snapshot:
            snapshot["api_key"] = "[BOUND]" if snapshot.get("api_key") else None
        if "extra_headers" in snapshot:
            snapshot["extra_headers"] = "[FROZEN-ON-ROUTE]"
        return snapshot

    task_snapshot = diagnostic_snapshot(task_config)
    task_snapshot["fallback_chain"] = [
        diagnostic_snapshot(entry) if isinstance(entry, Mapping) else entry
        for entry in materialized_chain
    ]
    return ClosedFallbackPlan(
        task=task,
        policy=policy,
        config=_freeze(task_snapshot),
        main_runtime=_freeze(diagnostic_snapshot(runtime)),
        primary=primary,
        candidates=candidates,
    )
