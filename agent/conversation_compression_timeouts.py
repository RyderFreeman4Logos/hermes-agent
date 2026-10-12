"""Compression idle/ceiling resolution.

The facade keeps ``resolve_context_compression_timeouts`` as the public name.
This leaf owns the read: compression idle and ceiling, the official aux-request
floor (#114594), and a configured auxiliary/fallback candidate budget. A
positive finite ``auxiliary.compression.timeout``, or a positive timeout on a
named fallback provider+model, raises the ceiling. Non-finite and overflow
values are ignored. A compression-only dict (no ``auxiliary`` key) is not a
full config.
"""

from __future__ import annotations

import contextlib
import math
from typing import Optional


def _positive_finite(raw: object) -> Optional[float]:
    with contextlib.suppress(TypeError, ValueError, OverflowError):
        parsed = float(raw)  # type: ignore[arg-type]
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    return None


def configured_compression_candidate_budget(config_root: object) -> Optional[float]:
    """Longest valid auxiliary/fallback compression timeout, or None."""
    if not isinstance(config_root, dict):
        return None
    auxiliary = config_root.get("auxiliary", {})
    compression = auxiliary.get("compression", {}) if isinstance(auxiliary, dict) else {}
    if not isinstance(compression, dict):
        return None
    budget = _positive_finite(compression.get("timeout")) if compression.get("timeout") is not None else None
    chain = compression.get("fallback_chain")
    if isinstance(chain, list):
        for entry in chain:
            if not isinstance(entry, dict):
                continue
            if not str(entry.get("provider") or "").strip() or not str(entry.get("model") or "").strip():
                continue
            raw = entry.get("timeout")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            parsed = _positive_finite(raw)
            if parsed is not None:
                budget = max(budget or 0.0, parsed)
    return budget


def resolve_context_compression_timeouts(compression_cfg: Optional[dict] = None) -> tuple[float, float]:
    """Return ``(idle_timeout_seconds, total_ceiling_seconds)``.

    ``idle_timeout_seconds <= 0`` disables the progress-aware wrapper. The
    ceiling is at least one idle window when idle is positive.
    """
    from agent.conversation_compression import (
        DEFAULT_CONTEXT_TIMEOUT_SECONDS,
        DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS,
    )

    idle = DEFAULT_CONTEXT_TIMEOUT_SECONDS
    ceiling = DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS
    config_root: dict = {}
    cfg = compression_cfg
    if cfg is None:
        with contextlib.suppress(Exception):
            from hermes_cli.config import load_config
            raw = load_config()
            config_root = raw if isinstance(raw, dict) else {}
        maybe = config_root.get("compression")
        cfg = maybe if isinstance(maybe, dict) else {}
    elif isinstance(cfg, dict) and ("compression" in cfg or "auxiliary" in cfg):
        # Full config. The compute-host waiter nests compression; the agent
        # tests put auxiliary next to the context_* keys. A bare compression
        # section has neither key.
        config_root = cfg
        maybe = cfg.get("compression")
        cfg = maybe if isinstance(maybe, dict) else cfg
    if isinstance(cfg, dict):
        with contextlib.suppress(TypeError, ValueError, OverflowError):
            if cfg.get("context_timeout_seconds") is not None:
                parsed = float(cfg["context_timeout_seconds"])
                if math.isfinite(parsed):
                    idle = parsed
        with contextlib.suppress(TypeError, ValueError, OverflowError):
            if cfg.get("context_total_ceiling_seconds") is not None:
                parsed = float(cfg["context_total_ceiling_seconds"])
                if math.isfinite(parsed) and parsed > 0:
                    ceiling = parsed
    aux_timeout = configured_compression_candidate_budget(config_root)
    if aux_timeout is not None:
        ceiling = max(ceiling, aux_timeout)
    if idle > 0:
        ceiling = max(ceiling, idle)
        # #114594: the aux summary attempt needs up to its own budget. Only raises.
        from agent.auxiliary_client import _effective_aux_timeout
        _aux_budget = float(_effective_aux_timeout("compression", None))
        if _aux_budget > 0:
            ceiling = max(ceiling, _aux_budget)
            idle = max(idle, min(_aux_budget, ceiling))
    return idle, ceiling
