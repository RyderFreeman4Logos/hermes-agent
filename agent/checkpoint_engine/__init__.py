"""Opt-in checkpoint ContextEngine (DESIGN.md §10 item 1: shadow no-op)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    "MapFact",
    "MapShard",
]


_MAP_CONCURRENCY_CAP = 2
_MAP_MAX_TOKENS = 1024
_MAP_TASK = "compression"


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


@dataclass(frozen=True)
class MapFact:
    """One model-derived fact tied to source events or marked uncertain."""

    kind: str
    text: str
    source_event_ids: tuple[int, ...]
    uncertain: bool = False


@dataclass(frozen=True)
class MapShard:
    """Validated typed Map output for one causally complete shard."""

    source_event_ids: tuple[int, ...]
    facts: tuple[MapFact, ...]


class CheckpointContextEngine(ContextEngine):
    """Selectable ``checkpoint`` engine that does not mutate messages."""

    def __init__(
        self,
        *,
        auxiliary_client: Any = None,
        map_concurrency: Optional[int] = None,
    ) -> None:
        self._auxiliary_client = auxiliary_client
        self._map_concurrency = self._bounded_map_concurrency(map_concurrency)
        self.last_map_shards: tuple[MapShard, ...] = ()

    @property
    def map_concurrency(self) -> int:
        """Bounded Map worker count; never exceeds the v1 cap."""
        return self._map_concurrency

    @staticmethod
    def _bounded_map_concurrency(value: Optional[int]) -> int:
        if value is None:
            try:
                from hermes_cli.config import load_config_readonly

                config = load_config_readonly()
                checkpoint = config.get("checkpoint", {})
                value = checkpoint.get("map_concurrency", _MAP_CONCURRENCY_CAP)
            except (AttributeError, ImportError, OSError, TypeError, ValueError):
                value = _MAP_CONCURRENCY_CAP
        if isinstance(value, bool) or not isinstance(value, int):
            return _MAP_CONCURRENCY_CAP
        return min(max(value, 1), _MAP_CONCURRENCY_CAP)

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

    @staticmethod
    def _map_prompt(
        messages: List[Dict[str, Any]], group: CausalGroup
    ) -> List[Dict[str, str]]:
        event_ids = group.event_indices
        payload = {
            "source_event_ids": event_ids,
            "events": [
                {"source_event_id": event_id, "message": messages[event_id]}
                for event_id in event_ids
            ],
        }
        return [
            {
                "role": "system",
                "content": (
                    "Return only JSON with exactly source_event_ids and facts. "
                    "source_event_ids must cover this shard exactly. Each fact "
                    "needs kind, text, and source_event_ids from this shard, or "
                    "uncertain: true when it has no source. Do not call tools."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]

    def _configured_auxiliary_response(self, messages: List[Dict[str, str]]) -> Any:
        """Call only explicit compression candidates and their configured chain."""
        try:
            from agent import auxiliary_client

            config = auxiliary_client._get_auxiliary_task_config(_MAP_TASK)
            chain = config.get("fallback_chain", [])
            candidates = [("checkpoint-primary", config)]
            if isinstance(chain, list):
                candidates.extend(
                    (f"fallback_chain[{index}]({entry.get('provider', '')})", entry)
                    for index, entry in enumerate(chain)
                    if isinstance(entry, dict)
                )
            timeout = auxiliary_client._effective_aux_timeout(_MAP_TASK, None)
            extra_body = auxiliary_client._get_task_extra_body(_MAP_TASK)
            extra_body.pop("reasoning", None)
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            return None

        for label, entry in candidates:
            provider = str(entry.get("provider", "")).strip()
            model = str(entry.get("model", "")).strip()
            if not provider or not model or provider.lower() == "auto":
                continue
            try:
                client, resolved_model = auxiliary_client._resolve_fallback_entry(entry)
                if client is None:
                    continue
                response = auxiliary_client._call_fallback_candidate_sync(
                    client,
                    resolved_model or model,
                    label,
                    task=_MAP_TASK,
                    messages=messages,
                    temperature=0,
                    max_tokens=_MAP_MAX_TOKENS,
                    tools=[],
                    effective_timeout=timeout,
                    effective_extra_body=extra_body,
                    reasoning_config={"enabled": False},
                )
            except Exception:
                continue
            if response is not None:
                return response
        return None

    def _call_map(self, messages: List[Dict[str, str]]) -> Any:
        if self._auxiliary_client is not None:
            return self._auxiliary_client.complete(
                messages=messages,
                max_tokens=_MAP_MAX_TOKENS,
                tools=[],
            )
        return self._configured_auxiliary_response(messages)

    @staticmethod
    def _response_content(response: Any) -> Optional[str]:
        if isinstance(response, str):
            return response
        choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        if not isinstance(message, dict) and message is None:
            return None
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        if tool_calls:
            return None
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        return content if isinstance(content, str) else None

    @staticmethod
    def _event_ids(value: Any) -> Optional[tuple[int, ...]]:
        if not isinstance(value, list) or any(
            isinstance(event_id, bool) or not isinstance(event_id, int)
            for event_id in value
        ):
            return None
        event_ids = tuple(value)
        return event_ids if len(set(event_ids)) == len(event_ids) else None

    @classmethod
    def _parse_map_shard(
        cls, response: Any, group: CausalGroup
    ) -> Optional[MapShard]:
        content = cls._response_content(response)
        if content is None:
            return None
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict) or set(payload) != {"source_event_ids", "facts"}:
            return None
        source_event_ids = cls._event_ids(payload["source_event_ids"])
        if source_event_ids is None or set(source_event_ids) != set(group.event_indices):
            return None
        facts = payload["facts"]
        if not isinstance(facts, list):
            return None

        parsed_facts = []
        for fact in facts:
            if not isinstance(fact, dict) or not {"kind", "text"} <= set(fact):
                return None
            if set(fact) - {"kind", "text", "source_event_ids", "uncertain"}:
                return None
            kind = fact["kind"]
            text = fact["text"]
            uncertain = fact.get("uncertain", False)
            if not isinstance(kind, str) or not kind or not isinstance(text, str) or not text:
                return None
            if not isinstance(uncertain, bool):
                return None
            fact_event_ids = cls._event_ids(fact.get("source_event_ids"))
            if fact_event_ids is None:
                if "source_event_ids" in fact or not uncertain:
                    return None
                fact_event_ids = ()
            elif not fact_event_ids or not set(fact_event_ids) <= set(source_event_ids):
                return None
            parsed_facts.append(MapFact(kind, text, fact_event_ids, uncertain))
        return MapShard(tuple(group.event_indices), tuple(parsed_facts))

    def _map_group(
        self, messages: List[Dict[str, Any]], group: CausalGroup
    ) -> Optional[MapShard]:
        try:
            return self._parse_map_shard(
                self._call_map(self._map_prompt(messages, group)), group
            )
        except Exception:
            return None

    def _map_shards(
        self, messages: List[Dict[str, Any]], groups: tuple[CausalGroup, ...]
    ) -> Optional[tuple[MapShard, ...]]:
        if not groups:
            return ()
        with ThreadPoolExecutor(max_workers=self._map_concurrency) as executor:
            mapped = tuple(executor.map(lambda group: self._map_group(messages, group), groups))
        if any(shard is None for shard in mapped):
            return None
        return tuple(shard for shard in mapped if shard is not None)

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
        self.last_map_shards = ()
        mapped = self._map_shards(messages, self._plan_causal_groups(messages))
        if mapped is None or not self._snapshot_is_current(messages, snapshot):
            return messages
        self.last_map_shards = mapped
        return messages
