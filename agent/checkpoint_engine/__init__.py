"""Opt-in checkpoint ContextEngine (DESIGN.md §10 item 1: shadow no-op)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

__all__ = ["CheckpointContextEngine"]


class CheckpointContextEngine(ContextEngine):
    """Selectable ``checkpoint`` engine that does not mutate messages."""

    @property
    def name(self) -> str:
        return "checkpoint"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        return False

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        return messages
