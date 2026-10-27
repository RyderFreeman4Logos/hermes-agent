"""Default delegation strategy (``delegation.backend: default``).

Every ``delegate_task`` task takes the route the ``delegation`` config section describes: a
``delegation.base_url`` endpoint, a ``delegation.provider`` pin, ``delegation.model`` and
``delegation.reasoning_effort``, or, when those are empty, the parent's own route. It adds the configured pool tier parameter. Bundled ``kind: backend``,
so it is always loaded; other strategies are plugins
of the same kind (``agent/delegation_provider.py``).
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.delegation_provider import DelegationProvider


class DefaultDelegationProvider(DelegationProvider):
    """Route every task through the ``delegation`` section."""

    @property
    def name(self) -> str:
        return "default"

    @property
    def display_name(self) -> str:
        return "Default"

    @property
    def summary(self) -> str:
        return "Uses the configured tier pool, otherwise delegation settings or the parent's model."

    def description(self, config: Mapping[str, Any]) -> str:
        return ("Omitted model_profile uses standard. Unknown names and a pool without "
                "standard fail closed. A non-empty pool is the only routing source.")

    def atomic_routes(self, config: Mapping[str, Any]) -> bool:
        # A configured pool is one admission unit: never start part of a misconfigured batch.
        from tools.delegate_tool import _model_pool
        return bool(_model_pool(dict(config)))

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        from tools.delegate_tool import _available_model_profile_names
        names = _available_model_profile_names(dict(config))
        prop = {"type": "string", "description": "Configured tier; omitted uses standard."}
        if names:
            prop["enum"] = names
        return {"model_profile": prop}

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        from tools.delegate_tool import _model_pool, _available_model_profile_names
        from agent.delegation_provider import DelegationRouteError
        cfg = dict(config)
        pool = _model_pool(cfg)
        name = str(task.get("model_profile") or "").strip()
        if not pool and not name:
            return {k: v for k, v in cfg.items() if k != "service_tier"}
        names = _available_model_profile_names(cfg)
        if pool and "standard" not in names:
            raise DelegationRouteError("Non-empty model_pool requires an explicit 'standard' profile.")
        name = name or "standard"
        if name not in names:
            raise DelegationRouteError(f"Unknown {name!r}. Configured: {', '.join(names) or '(none)'}.")
        return {**pool[name], "exclusive_route": True, "label": name,
                "reasoning_effort": pool[name].get("reasoning_effort", config.get("reasoning_effort")),
                "fallback_providers": []}


def register(ctx) -> None:
    ctx.register_delegation_provider(DefaultDelegationProvider())
