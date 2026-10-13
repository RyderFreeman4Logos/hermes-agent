from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import CheckpointContextEngine


def final_request_exceeds_hard_wire_budget(engine: CheckpointContextEngine, messages: Sequence[Mapping[str, Any]], *, overhead_tokens: int = 0) -> bool:
    """Return whether the exact model-facing message list exceeds its hard cap."""
    return engine._estimate_wire_tokens(messages) + int(overhead_tokens) > engine.hard_max_wire_tokens
