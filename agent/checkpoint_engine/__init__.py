"""Opt-in checkpoint ContextEngine (DESIGN.md §10 item 1: shadow no-op)."""

from __future__ import annotations

import hashlib
import json
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

    @staticmethod
    def _has_inflight_tools(messages: List[Dict[str, Any]]) -> bool:
        pending = set()
        for message in messages:
            if message.get("role") == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls is None:
                    continue
                if not isinstance(tool_calls, list):
                    return True
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        return True
                    tool_call_id = tool_call.get("id")
                    if not isinstance(tool_call_id, str) or not tool_call_id:
                        return True
                    pending.add(tool_call_id)
            elif message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                if isinstance(tool_call_id, str):
                    pending.discard(tool_call_id)
        return bool(pending)

    @staticmethod
    def _capture_snapshot(messages: List[Dict[str, Any]]) -> tuple[int, str]:
        content_hash = hashlib.sha256(
            json.dumps(
                messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest()
        return id(messages), content_hash

    def _snapshot_is_current(
        self, messages: List[Dict[str, Any]], snapshot: tuple[int, str]
    ) -> bool:
        return self._capture_snapshot(messages) == snapshot

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        try:
            snapshot = self._capture_snapshot(messages)
        except (TypeError, ValueError):
            return messages
        if self._has_inflight_tools(messages) or not self._snapshot_is_current(
            messages, snapshot
        ):
            return messages
        return messages
