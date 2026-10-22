"""Compression time limits: the configured idle and total limits, and the fence deadline shared live
with the summary stream (sibling of ``conversation_compression``).

Compression waits a total time limit for its summary. That limit stops a summary that trickles
out a token at a time; on a laptop GPU, reading a 150K-token conversation alone can take longer
than the whole limit. Time llama-server spends reading the prompt therefore pushes the fence
deadline back (``CompressionCommitFence.extend_deadline``), and hosts read the deadline from the
fence instead of adding the configured limit to the time they started waiting.
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.conversation_compression import CompressionCommitFence

# Defaults for the in-agent progress-aware wrap; mirror hermes_cli.config.DEFAULT_CONFIG["compression"] keys.
DEFAULT_CONTEXT_TIMEOUT_SECONDS = 120.0
DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS = 600.0


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


def effective_ceiling(fence: CompressionCommitFence, ceiling: float, wait_started: float) -> float:
    """The total limit for a host that started waiting at *wait_started*: the configured
    *ceiling*, or longer once prompt reading pushed the fence deadline back."""
    deadline = fence.deadline_monotonic
    return ceiling if deadline is None else max(ceiling, deadline - wait_started)


@contextlib.contextmanager
def compression_deadline_scope(fence: Optional[CompressionCommitFence]):
    """Share *fence*'s deadline with the summary stream, live, and push it back while the
    server reads the prompt. A stream checks the deadline as it reads, so a later summary
    call in the same attempt sees the extended deadline too."""
    from agent.auxiliary_client import aux_stream_deadline
    from agent.local_prompt_progress import prompt_reading_hook

    if fence is None:
        yield
        return
    with aux_stream_deadline(lambda: fence.deadline_monotonic), prompt_reading_hook(fence.extend_deadline):
        yield
