"""Fail-closed URL identity normalization for model/provider routes."""

from __future__ import annotations

from typing import Any

from utils import normalize_route_base_url


def should_clear_context_pin(
    configured_model: Any,
    active_model: Any,
    configured_base_url: Any,
    active_base_url: Any,
    configured_provider: Any,
    active_provider: Any,
) -> bool:
    """True when a configured ``model.context_length`` pin no longer matches its runtime route.

    Fail-closed: any error during route comparison returns ``True`` (drop the pin)
    so a stale window never silently inflates the compression threshold.
    """
    configured_model = str(configured_model or "").strip()
    if configured_model and configured_model != str(active_model or "").strip():
        return True
    try:
        from agent.agent_init import _context_route_mismatch

        return _context_route_mismatch(
            configured_base_url,
            active_base_url,
            configured_provider,
            active_provider,
        )
    except Exception:
        return True


async def should_clear_context_pin_async(
    configured_model: Any,
    active_model: Any,
    configured_base_url: Any,
    active_base_url: Any,
    configured_provider: Any,
    active_provider: Any,
) -> bool:
    """Async wrapper for ``should_clear_context_pin``.

    Offloads the route comparison to a worker thread so async gateway
    handlers never run it on the event loop — the resolution chain is
    cache-only (``allow_network=False``) but can still do cold-start disk
    I/O. Shares all logic with the sync version — no code duplication.
    """
    import asyncio

    return await asyncio.to_thread(
        should_clear_context_pin,
        configured_model,
        active_model,
        configured_base_url,
        active_base_url,
        configured_provider,
        active_provider,
    )
