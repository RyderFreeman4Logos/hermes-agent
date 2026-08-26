"""Opt-in checkpoint ContextEngine (DESIGN.md §10 item 1: shadow no-op)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

__all__ = [
    "ActiveIntent",
    "CausalGroup",
    "CheckpointContextEngine",
    "DeterministicLanes",
    "Effect",
]


@dataclass(frozen=True)
class ActiveIntent:
    """The newest non-empty user turn, kept outside checkpoint prose."""

    content: str
    event_indices: tuple[int, ...]


@dataclass(frozen=True)
class Effect:
    """A tool action whose completion requires a trusted receipt."""

    tool_call_id: str
    operation: Optional[str]
    status: str
    event_indices: tuple[int, ...]


@dataclass(frozen=True)
class CausalGroup:
    """Contiguous events that must be planned as one causal shard unit."""

    event_indices: tuple[int, ...]


@dataclass(frozen=True)
class DeterministicLanes:
    """Intent and effect facts extracted without model inference."""

    active_intent: Optional[ActiveIntent]
    effects: tuple[Effect, ...]


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
    def _tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return []
        return [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]

    @classmethod
    def _plan_causal_groups(
        cls, messages: List[Dict[str, Any]]
    ) -> tuple[CausalGroup, ...]:
        """Partition events without separating calls from their tool results."""
        groups = []
        pending = set()
        group_start = 0
        for index, message in enumerate(messages):
            if message.get("role") == "assistant":
                pending.update(
                    tool_call_id
                    for tool_call in cls._tool_calls(message)
                    if isinstance(tool_call_id := tool_call.get("id"), str)
                    and tool_call_id
                )
            elif message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                if isinstance(tool_call_id, str):
                    pending.discard(tool_call_id)

            if pending:
                continue
            groups.append(CausalGroup(tuple(range(group_start, index + 1))))
            group_start = index + 1

        if group_start < len(messages):
            groups.append(CausalGroup(tuple(range(group_start, len(messages)))))
        return tuple(groups)

    @classmethod
    def _extract_deterministic_lanes(
        cls, messages: List[Dict[str, Any]]
    ) -> DeterministicLanes:
        """Extract active intent and conservative tool-effect state."""
        active_intent = None
        effects = []
        effect_positions = {}
        for index, message in enumerate(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    active_intent = ActiveIntent(content, (index,))
            elif message.get("role") == "assistant":
                for tool_call in cls._tool_calls(message):
                    tool_call_id = tool_call.get("id")
                    if not isinstance(tool_call_id, str) or not tool_call_id:
                        continue
                    function = tool_call.get("function")
                    operation = (
                        function.get("name")
                        if isinstance(function, dict)
                        and isinstance(function.get("name"), str)
                        else None
                    )
                    effect_positions[tool_call_id] = len(effects)
                    effects.append(Effect(tool_call_id, operation, "issued", (index,)))
            elif message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id not in effect_positions:
                    continue
                effect_index = effect_positions[tool_call_id]
                effect = effects[effect_index]
                effects[effect_index] = Effect(
                    effect.tool_call_id,
                    effect.operation,
                    "unknown",
                    (*effect.event_indices, index),
                )
        return DeterministicLanes(active_intent, tuple(effects))

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
