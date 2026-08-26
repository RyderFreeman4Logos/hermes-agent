"""Opt-in checkpoint ContextEngine (DESIGN.md §10 item 1: shadow no-op)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional

from agent.context_engine import ContextEngine, sanitize_memory_context

__all__ = [
    "ActiveIntent",
    "CausalGroup",
    "CheckpointContextEngine",
    "DeterministicLanes",
    "Effect",
    "MapFact",
    "MapShard",
    "ReducedState",
]


_MAP_CONCURRENCY_CAP = 2
_MAP_MAX_TOKENS = 1024
_MAP_SHARD_TARGET_INPUT_TOKENS = 12_000
_MAP_SHARD_MAX_INPUT_TOKENS = 16_000
_MAP_MAX_SHARDS_CAP = 32
_DEFAULT_MAX_MAP_SHARDS = 32
_MAP_TOTAL_INPUT_TOKENS = _MAP_SHARD_TARGET_INPUT_TOKENS * _DEFAULT_MAX_MAP_SHARDS
_MAP_TOTAL_OUTPUT_TOKENS = _MAP_MAX_TOKENS * _DEFAULT_MAX_MAP_SHARDS
_SEMANTIC_REDUCE_MAX_TOKENS = 2048
_MAP_TASK = "compression"
_DEFAULT_TARGET_WIRE_TOKENS = 48_000
_DEFAULT_HARD_MAX_WIRE_TOKENS = 60_000
_DEFAULT_OUTPUT_RESERVE_TOKENS = 4_096
_AUTO_COMPRESSION_COOLDOWN_SECONDS = 60.0
_ACTION_STATES = frozenset(
    {"planned", "issued", "running", "succeeded", "failed", "unknown"}
)
_TERMINAL_ACTION_STATES = frozenset({"succeeded", "failed", "unknown"})


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
    identity: Optional[str] = None
    supersedes: tuple[str, ...] = ()
    action_state: Optional[str] = None


@dataclass(frozen=True)
class MapShard:
    """Validated typed Map output for one causally complete shard."""

    source_event_ids: tuple[int, ...]
    facts: tuple[MapFact, ...]


@dataclass(frozen=True)
class ReducedState:
    """Deterministic state passed to the cheap semantic reducer."""

    active_intent: Optional[ActiveIntent]
    effects: tuple[Effect, ...]
    facts: tuple[MapFact, ...]
    plans: tuple[MapFact, ...]


class CheckpointContextEngine(ContextEngine):
    """Selectable ``checkpoint`` engine that does not mutate messages."""

    def __init__(
        self,
        *,
        auxiliary_client: Any = None,
        main_model: Any = None,
        map_concurrency: Optional[int] = None,
        max_map_shards: Optional[int] = None,
        semantic_reducer: Optional[Callable[[ReducedState], Any]] = None,
        mode: Optional[str] = None,
        target_wire_tokens: Optional[int] = None,
        hard_max_wire_tokens: Optional[int] = None,
        token_counter: Optional[Callable[[Any], int]] = None,
        tool_schemas: Any = (),
        output_reserve_tokens: Optional[int] = None,
    ) -> None:
        checkpoint_config = self._checkpoint_config()
        self._auxiliary_client = auxiliary_client
        self._main_model = main_model
        self._map_concurrency = self._bounded_map_concurrency(map_concurrency)
        self._max_map_shards = self._bounded_max_map_shards(max_map_shards)
        self._semantic_reducer = semantic_reducer
        configured_mode = checkpoint_config.get("mode", "shadow")
        self._mode = mode if mode in {"shadow", "live"} else configured_mode
        if self._mode not in {"shadow", "live"}:
            self._mode = "shadow"
        configured_hard_max = checkpoint_config.get(
            "hard_max_wire_tokens", _DEFAULT_HARD_MAX_WIRE_TOKENS
        )
        self._hard_max_wire_tokens = self._positive_int(
            hard_max_wire_tokens
            if hard_max_wire_tokens is not None
            else configured_hard_max,
            _DEFAULT_HARD_MAX_WIRE_TOKENS,
        )
        configured_target = checkpoint_config.get(
            "target_wire_tokens", _DEFAULT_TARGET_WIRE_TOKENS
        )
        self._target_wire_tokens = min(
            self._positive_int(
                target_wire_tokens
                if target_wire_tokens is not None
                else configured_target,
                _DEFAULT_TARGET_WIRE_TOKENS,
            ),
            self._hard_max_wire_tokens,
        )
        self._token_counter = token_counter if callable(token_counter) else None
        self._tool_schemas = tool_schemas
        self._output_reserve_tokens = self._positive_int(
            output_reserve_tokens
            if output_reserve_tokens is not None
            else _DEFAULT_OUTPUT_RESERVE_TOKENS,
            _DEFAULT_OUTPUT_RESERVE_TOKENS,
            allow_zero=True,
        )
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.provider = ""
        self.api_mode = ""
        self.model_thresholds: Dict[str, float] = {}
        self.threshold_percent = type(self).threshold_percent
        self.threshold_tokens = 0
        self._context_length = 0
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.last_request_tokens = 0
        self.last_trigger_reason: Optional[str] = None
        self.last_focus_topic = ""
        self.last_memory_context = ""
        self._automatic_cooldown_until = 0.0
        self._automatic_cooldown_reason = ""
        self._last_compress_aborted = False
        self._last_summary_error: Optional[str] = None
        self._last_compression_made_progress = False
        self._verify_compaction_cleared_threshold = False
        self.awaiting_real_usage_after_compression = False
        self.last_map_shards: tuple[MapShard, ...] = ()
        self.last_map_externalized_groups: tuple[CausalGroup, ...] = ()
        self.last_reduced_state: Optional[ReducedState] = None
        self.last_checkpoint_text: Optional[str] = None
        self.last_candidate: Optional[List[Dict[str, Any]]] = None
        self.last_wire_tokens: Optional[int] = None
        self.last_degradation_steps: tuple[str, ...] = ()

    @staticmethod
    def _checkpoint_config() -> Dict[str, Any]:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
            checkpoint = config.get("checkpoint", {})
            return checkpoint if isinstance(checkpoint, dict) else {}
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            return {}

    @staticmethod
    def _positive_int(value: Any, default: int, *, allow_zero: bool = False) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            return default
        if value > 0 or allow_zero and value == 0:
            return value
        return default

    @property
    def map_concurrency(self) -> int:
        """Bounded Map worker count; never exceeds the v1 cap."""
        return self._map_concurrency

    @staticmethod
    def _bounded_map_concurrency(value: Optional[int]) -> int:
        if value is None:
            value = CheckpointContextEngine._checkpoint_config().get(
                "map_concurrency", _MAP_CONCURRENCY_CAP
            )
        if isinstance(value, bool) or not isinstance(value, int):
            return _MAP_CONCURRENCY_CAP
        return min(max(value, 1), _MAP_CONCURRENCY_CAP)

    @property
    def max_map_shards(self) -> int:
        """Hard cap for Map requests in one checkpoint candidate."""
        return self._max_map_shards

    @staticmethod
    def _bounded_max_map_shards(value: Optional[int]) -> int:
        if value is None:
            value = CheckpointContextEngine._checkpoint_config().get(
                "max_map_shards", _DEFAULT_MAX_MAP_SHARDS
            )
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return _DEFAULT_MAX_MAP_SHARDS
        return min(value, _MAP_MAX_SHARDS_CAP)

    @property
    def name(self) -> str:
        return "checkpoint"

    @property
    def context_length(self) -> int:
        return self._context_length

    @context_length.setter
    def context_length(self, value: Any) -> None:
        context_length = value if isinstance(value, int) and not isinstance(value, bool) else 0
        context_length = max(0, context_length)
        if context_length == getattr(self, "_context_length", 0):
            return
        self._context_length = context_length
        if hasattr(self, "threshold_percent"):
            self.threshold_tokens = int(context_length * self.threshold_percent)

    @staticmethod
    def _usage_tokens(value: Any) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        usage = usage if isinstance(usage, dict) else {}
        self.last_input_tokens = self._usage_tokens(usage.get("input_tokens"))
        self.last_output_tokens = self._usage_tokens(usage.get("output_tokens"))
        self.last_cache_read_tokens = self._usage_tokens(usage.get("cache_read_tokens"))
        self.last_cache_write_tokens = self._usage_tokens(usage.get("cache_write_tokens"))
        self.last_reasoning_tokens = self._usage_tokens(usage.get("reasoning_tokens"))
        provider_prompt = self._usage_tokens(usage.get("prompt_tokens"))
        if not provider_prompt:
            provider_prompt = (
                self.last_input_tokens
                + self.last_cache_read_tokens
                + self.last_cache_write_tokens
            )
        self.last_prompt_tokens = provider_prompt
        self.last_completion_tokens = self._usage_tokens(
            usage.get("completion_tokens", self.last_output_tokens)
        )
        self.last_total_tokens = self._usage_tokens(
            usage.get("total_tokens", self.last_prompt_tokens + self.last_completion_tokens)
        )
        if self._verify_compaction_cleared_threshold:
            self._verify_compaction_cleared_threshold = False
            if self.last_prompt_tokens >= self.threshold_tokens > 0:
                self._set_automatic_cooldown("ineffective")
            else:
                self._clear_automatic_cooldown()
        elif self.last_prompt_tokens < self.threshold_tokens:
            self._clear_automatic_cooldown()
        self.awaiting_real_usage_after_compression = False

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        super().update_model(
            model=model,
            context_length=context_length,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            api_mode=api_mode,
        )
        self._clear_automatic_cooldown()

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._clear_automatic_cooldown()
        self.last_request_tokens = 0
        self.last_trigger_reason = None

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        return self.should_compress_info(prompt_tokens)[0]

    def should_compress_info(
        self, prompt_tokens: Optional[int] = None
    ) -> tuple[bool, Optional[str]]:
        """Return the automatic trigger decision without warning below threshold.

        The host treats a non-empty reason as an overflow warning, so the
        detailed ``below_threshold`` state remains available on the engine
        without spuriously warning on healthy turns.
        """
        tokens = self._usage_tokens(prompt_tokens)
        if prompt_tokens is None:
            tokens = self.last_prompt_tokens or self.last_request_tokens
        if self.threshold_tokens <= 0:
            self.last_trigger_reason = "threshold_unavailable"
            return False, None
        if tokens < self.threshold_tokens:
            self.last_trigger_reason = "below_threshold"
            return False, None
        if reason := self._automatic_cooldown_reason_now():
            self.last_trigger_reason = reason
            return False, reason
        self.last_trigger_reason = None
        return True, None

    def record_completed_compaction(
        self, *, used_fallback: bool = False, feasibility_skip: bool = False
    ) -> None:
        del used_fallback, feasibility_skip
        self._verify_compaction_cleared_threshold = True
        self._clear_automatic_cooldown()

    def _set_automatic_cooldown(self, reason: str) -> None:
        self._automatic_cooldown_until = time.monotonic() + _AUTO_COMPRESSION_COOLDOWN_SECONDS
        self._automatic_cooldown_reason = reason
        self.last_trigger_reason = f"cooldown:{reason}"

    def _clear_automatic_cooldown(self) -> None:
        self._automatic_cooldown_until = 0.0
        self._automatic_cooldown_reason = ""

    def _automatic_cooldown_reason_now(self) -> Optional[str]:
        if self._automatic_cooldown_until <= time.monotonic():
            self._clear_automatic_cooldown()
            return None
        return f"cooldown:{self._automatic_cooldown_reason or 'retry'}"

    def _reclaimable_groups(
        self, messages: List[Dict[str, Any]], groups: Optional[tuple[CausalGroup, ...]] = None
    ) -> tuple[CausalGroup, ...]:
        groups = groups if groups is not None else self._plan_causal_groups(messages)
        lanes = self._extract_deterministic_lanes(messages)
        active_indices = set(lanes.active_intent.event_indices) if lanes.active_intent else set()
        return tuple(
            group
            for group in groups
            if not active_indices.intersection(group.event_indices)
            and not any(messages[index].get("role") == "system" for index in group.event_indices)
        )

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        try:
            return bool(self._reclaimable_groups(messages))
        except (AttributeError, IndexError, TypeError, ValueError):
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

    def _map_input_tokens(
        self, messages: List[Dict[str, Any]], group: CausalGroup
    ) -> int:
        """Estimate the complete Map request before admission."""
        return self._token_count(self._map_prompt(messages, group))

    def _plan_map_shards(
        self, messages: List[Dict[str, Any]], groups: tuple[CausalGroup, ...]
    ) -> Optional[tuple[CausalGroup, ...]]:
        """Pack adjacent causal units; retain units with no proven safe split."""
        shards = []
        current_indices: List[int] = []
        externalized = []

        def finish_current() -> None:
            nonlocal current_indices
            if current_indices:
                shards.append(CausalGroup(tuple(current_indices)))
                current_indices = []

        for group in groups:
            unit_tokens = self._map_input_tokens(messages, group)
            if unit_tokens > _MAP_SHARD_MAX_INPUT_TOKENS:
                finish_current()
                externalized.append(group)
                continue
            candidate = CausalGroup(tuple((*current_indices, *group.event_indices)))
            if current_indices and self._map_input_tokens(messages, candidate) > _MAP_SHARD_TARGET_INPUT_TOKENS:
                finish_current()
            current_indices.extend(group.event_indices)
        finish_current()

        self.last_map_externalized_groups = tuple(externalized)
        total_input = sum(self._map_input_tokens(messages, shard) for shard in shards)
        if (
            len(shards) > self._max_map_shards
            or total_input > _MAP_TOTAL_INPUT_TOKENS
            or len(shards) * _MAP_MAX_TOKENS > _MAP_TOTAL_OUTPUT_TOKENS
        ):
            return None
        return tuple(shards)

    def _configured_auxiliary_response(
        self, messages: List[Dict[str, str]], max_tokens: int
    ) -> Any:
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
                    max_tokens=max_tokens,
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
        return self._configured_auxiliary_response(messages, _MAP_MAX_TOKENS)

    def _call_semantic_reduce(self, messages: List[Dict[str, str]]) -> Any:
        if self._auxiliary_client is not None:
            return self._auxiliary_client.complete(
                messages=messages,
                max_tokens=_SEMANTIC_REDUCE_MAX_TOKENS,
                tools=[],
            )
        return self._configured_auxiliary_response(messages, _SEMANTIC_REDUCE_MAX_TOKENS)

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

    @staticmethod
    def _identities(value: Any) -> Optional[tuple[str, ...]]:
        if not isinstance(value, list) or any(
            not isinstance(identity, str) or not identity.strip()
            for identity in value
        ):
            return None
        identities = tuple(value)
        return identities if len(set(identities)) == len(identities) else None

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
            if set(fact) - {
                "kind",
                "text",
                "source_event_ids",
                "uncertain",
                "identity",
                "supersedes",
                "action_state",
            }:
                return None
            kind = fact["kind"]
            text = fact["text"]
            uncertain = fact.get("uncertain", False)
            if not isinstance(kind, str) or not kind or not isinstance(text, str) or not text:
                return None
            if not isinstance(uncertain, bool):
                return None
            identity = fact.get("identity")
            if identity is not None and (
                not isinstance(identity, str) or not identity.strip()
            ):
                return None
            supersedes = cls._identities(fact.get("supersedes", []))
            if supersedes is None:
                return None
            action_state = fact.get("action_state")
            if action_state is not None and action_state not in _ACTION_STATES:
                return None
            if action_state is not None and identity is None:
                return None
            fact_event_ids = cls._event_ids(fact.get("source_event_ids"))
            if fact_event_ids is None:
                if "source_event_ids" in fact or not uncertain:
                    return None
                fact_event_ids = ()
            elif not fact_event_ids or not set(fact_event_ids) <= set(source_event_ids):
                return None
            parsed_facts.append(
                MapFact(
                    kind,
                    text,
                    fact_event_ids,
                    uncertain,
                    identity,
                    supersedes,
                    action_state,
                )
            )
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

    @staticmethod
    def _fact_identity(fact: MapFact) -> str:
        return fact.identity or f"{fact.kind}:{fact.text}"

    @staticmethod
    def _fact_position(fact: MapFact) -> int:
        return max(fact.source_event_ids, default=-1)

    @classmethod
    def _newer_fact(cls, existing: MapFact, candidate: MapFact) -> MapFact:
        """Keep terminal action state from being reopened by stale Map prose."""
        if existing.action_state and candidate.action_state:
            if (
                existing.action_state in _TERMINAL_ACTION_STATES
                and candidate.action_state not in _TERMINAL_ACTION_STATES
            ):
                return existing
            if (
                candidate.action_state == "planned"
                and existing.action_state != "planned"
            ):
                return existing
        return candidate if cls._fact_position(candidate) >= cls._fact_position(existing) else existing

    @classmethod
    def _reduce(
        cls, lanes: DeterministicLanes, shards: tuple[MapShard, ...]
    ) -> ReducedState:
        """Merge typed Map facts; plan prose never creates a tool effect."""
        records: Dict[str, MapFact] = {}
        for shard in shards:
            for fact in sorted(shard.facts, key=cls._fact_position):
                identity = cls._fact_identity(fact)
                for superseded in fact.supersedes:
                    records.pop(superseded, None)
                previous = records.get(identity)
                records[identity] = (
                    fact if previous is None else cls._newer_fact(previous, fact)
                )
        facts = tuple(
            sorted(records.values(), key=lambda fact: (cls._fact_position(fact), cls._fact_identity(fact)))
        )
        return ReducedState(
            lanes.active_intent,
            lanes.effects,
            facts,
            tuple(fact for fact in facts if fact.action_state == "planned"),
        )

    @staticmethod
    def _semantic_reduce_prompt(
        state: ReducedState, *, focus_topic: str = "", memory_context: str = ""
    ) -> List[Dict[str, str]]:
        payload = {
            "focus_topic": focus_topic,
            "memory_context": memory_context,
            "active_intent": (
                {
                    "content": state.active_intent.content,
                    "source_event_ids": state.active_intent.event_indices,
                }
                if state.active_intent is not None
                else None
            ),
            "effects": [
                {
                    "tool_call_id": effect.tool_call_id,
                    "operation": effect.operation,
                    "status": effect.status,
                    "source_event_ids": effect.event_indices,
                }
                for effect in state.effects
            ],
            "facts": [
                {
                    "kind": fact.kind,
                    "text": fact.text,
                    "identity": fact.identity,
                    "supersedes": fact.supersedes,
                    "action_state": fact.action_state,
                    "source_event_ids": fact.source_event_ids,
                    "uncertain": fact.uncertain,
                }
                for fact in state.facts
            ],
        }
        return [
            {
                "role": "system",
                "content": (
                    "Write one concise continuity checkpoint from this typed state. "
                    "Keep active intent, trusted effects, verification, and source ids. "
                    "Plans are not effects. Return text only. Do not call tools."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]

    def _semantic_checkpoint(
        self, state: ReducedState, *, focus_topic: str = "", memory_context: str = ""
    ) -> Optional[str]:
        try:
            if self._semantic_reducer is not None:
                candidate = self._semantic_reducer(state)
            else:
                candidate = self._response_content(
                    self._call_semantic_reduce(
                        self._semantic_reduce_prompt(
                            state,
                            focus_topic=focus_topic,
                            memory_context=memory_context,
                        )
                    )
                )
        except Exception:
            return None
        if not isinstance(candidate, str):
            return None
        checkpoint = candidate.strip()
        return checkpoint or None

    @staticmethod
    def _source_reference(fact: MapFact) -> str:
        return ",".join(str(event_id) for event_id in fact.source_event_ids)

    def _render_checkpoint(
        self, semantic_checkpoint: str, state: ReducedState, detail_level: int
    ) -> str:
        """Add deterministic lanes while degrading only safe historical detail."""
        details = []
        completed_refs = []
        for fact in state.facts:
            reference = self._source_reference(fact)
            if fact.action_state in _TERMINAL_ACTION_STATES and detail_level >= 1:
                completed_refs.append(f"{self._fact_identity(fact)}@{reference}")
                continue
            if fact.kind in {"tool_body", "tool_result"} and detail_level >= 2:
                details.append(f"- {fact.kind} ref: {reference}")
                continue
            text = fact.text
            active_index = (
                max(state.active_intent.event_indices)
                if state.active_intent is not None
                else -1
            )
            if (
                fact.kind == "decision"
                and detail_level >= 3
                and self._fact_position(fact) < active_index
                and len(text) > 160
            ):
                text = f"{text[:157].rstrip()}..."
            action = (
                f" [{fact.action_state}]"
                if fact.action_state is not None
                else ""
            )
            details.append(f"- {fact.kind}{action}: {text} (events: {reference})")
        if completed_refs:
            details.append(f"- completed action refs: {', '.join(completed_refs)}")
        if state.effects:
            details.append(
                "Trusted effects:\n"
                + "\n".join(
                    f"- {effect.operation or 'tool'} [{effect.status}] "
                    f"(events: {','.join(str(index) for index in effect.event_indices)})"
                    for effect in state.effects
                )
            )
        return semantic_checkpoint if not details else f"{semantic_checkpoint}\n\n" + "\n".join(details)

    @staticmethod
    def _rough_token_count(value: Any) -> int:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            text = str(value)
        try:
            from agent.model_metadata import (
                estimate_request_tokens_rough,
                estimate_tokens_rough,
            )

            host_estimate = (
                estimate_request_tokens_rough(value)
                if isinstance(value, list)
                and all(isinstance(message, dict) and "role" in message for message in value)
                else estimate_tokens_rough(text)
            )
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            host_estimate = 0
        if not text.isascii():
            fallback = len(text)
        elif isinstance(value, (dict, list, tuple)):
            fallback = (len(text) + 1) // 2
        else:
            fallback = (len(text) + 3) // 4
        return max(1, host_estimate, fallback)

    def _token_count(self, value: Any) -> int:
        if self._token_counter is not None:
            try:
                count = self._token_counter(value)
            except Exception:
                return self._rough_token_count(value)
            if not isinstance(count, bool) and isinstance(count, int) and count >= 0:
                return count
        return self._rough_token_count(value)

    def _estimate_wire_tokens(self, candidate: List[Dict[str, Any]]) -> int:
        schema_tokens = self._token_count(self._tool_schemas) if self._tool_schemas else 0
        return (
            sum(self._token_count(message) for message in candidate)
            + schema_tokens
            + self._output_reserve_tokens
        )

    @staticmethod
    def _tail_groups(
        messages: List[Dict[str, Any]],
        groups: tuple[CausalGroup, ...],
        active_intent: ActiveIntent,
    ) -> tuple[CausalGroup, ...]:
        active_indices = set(active_intent.event_indices)
        return tuple(
            group
            for group in groups
            if not active_indices.intersection(group.event_indices)
            and not any(messages[index].get("role") == "system" for index in group.event_indices)
        )

    def _projection(
        self,
        messages: List[Dict[str, Any]],
        tail_groups: tuple[CausalGroup, ...],
        active_intent: ActiveIntent,
        checkpoint: str,
    ) -> List[Dict[str, Any]]:
        systems = [dict(message) for message in messages if message.get("role") == "system"]
        tail = [
            dict(messages[index])
            for group in tail_groups
            for index in group.event_indices
        ]
        while tail and tail[-1].get("role") == "user":
            tail_groups = tail_groups[:-1]
            tail = [
                dict(messages[index])
                for group in tail_groups
                for index in group.event_indices
            ]
        active = [dict(messages[index]) for index in active_intent.event_indices]
        return [
            *systems,
            {"role": "system", "content": checkpoint},
            *tail,
            *active,
        ]

    def _render_candidate(
        self,
        messages: List[Dict[str, Any]],
        groups: tuple[CausalGroup, ...],
        state: ReducedState,
        semantic_checkpoint: str,
    ) -> Optional[tuple[List[Dict[str, Any]], int, tuple[str, ...], str]]:
        if state.active_intent is None:
            return None
        tail_groups = self._tail_groups(messages, groups, state.active_intent)
        detail_level = 0
        steps = []
        while True:
            checkpoint = self._render_checkpoint(
                semantic_checkpoint, state, detail_level
            )
            candidate = self._projection(
                messages, tail_groups, state.active_intent, checkpoint
            )
            wire_tokens = self._estimate_wire_tokens(candidate)
            if wire_tokens <= self._target_wire_tokens:
                break
            if detail_level < 3:
                detail_level += 1
                steps.append(("completed", "tool_bodies", "decisions")[detail_level - 1])
                continue
            if tail_groups:
                tail_groups = tail_groups[1:]
                steps.append("tail")
                continue
            break
        if wire_tokens > self._hard_max_wire_tokens:
            return None
        return candidate, wire_tokens, tuple(steps), checkpoint

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        self._last_compress_aborted = False
        self._last_summary_error = None
        self._last_compression_made_progress = False
        self.last_request_tokens = self._usage_tokens(current_tokens) or self._estimate_wire_tokens(messages)
        self.last_focus_topic = focus_topic.strip() if isinstance(focus_topic, str) else ""
        self.last_memory_context = (
            sanitize_memory_context(memory_context)
            if isinstance(memory_context, str) and memory_context.strip()
            else ""
        )
        if force:
            self._clear_automatic_cooldown()
        elif reason := self._automatic_cooldown_reason_now():
            self.last_trigger_reason = reason
            return messages
        try:
            snapshot = self._capture_snapshot(messages)
        except (TypeError, ValueError):
            self._set_automatic_cooldown("snapshot_unavailable")
            return messages
        if self._has_inflight_tools(messages):
            self.last_trigger_reason = "in_flight_tools"
            return messages
        if not self._snapshot_is_current(messages, snapshot):
            self.last_trigger_reason = "stale_snapshot"
            return messages
        self.last_map_shards = ()
        self.last_map_externalized_groups = ()
        self.last_reduced_state = None
        self.last_checkpoint_text = None
        self.last_candidate = None
        self.last_wire_tokens = None
        self.last_degradation_steps = ()
        groups = self._plan_causal_groups(messages)
        if (
            current_tokens is not None
            and not force
            and not self._reclaimable_groups(messages, groups)
        ):
            self._set_automatic_cooldown("nothing_reclaimable")
            return messages
        map_groups = self._plan_map_shards(messages, groups)
        if map_groups is None:
            self._set_automatic_cooldown("map_budget_exceeded")
            self._last_compress_aborted = True
            self._last_summary_error = "checkpoint Map exceeded its token budget"
            return messages
        mapped = self._map_shards(messages, map_groups)
        if mapped is None:
            self._set_automatic_cooldown("map_failed")
            self._last_compress_aborted = True
            self._last_summary_error = "checkpoint Map failed"
            return messages
        if not self._snapshot_is_current(messages, snapshot):
            self.last_trigger_reason = "stale_snapshot"
            return messages
        self.last_map_shards = mapped
        reduced = self._reduce(self._extract_deterministic_lanes(messages), mapped)
        checkpoint = self._semantic_checkpoint(
            reduced,
            focus_topic=self.last_focus_topic,
            memory_context=self.last_memory_context,
        )
        if checkpoint is None:
            self._set_automatic_cooldown("semantic_reduce_failed")
            self._last_compress_aborted = True
            self._last_summary_error = "checkpoint semantic Reduce failed"
            return messages
        if not self._snapshot_is_current(messages, snapshot):
            self.last_trigger_reason = "stale_snapshot"
            return messages
        rendered = self._render_candidate(messages, groups, reduced, checkpoint)
        if rendered is None:
            self._set_automatic_cooldown("candidate_rejected")
            self._last_compress_aborted = True
            self._last_summary_error = "checkpoint candidate exceeded its wire budget"
            return messages
        if not self._snapshot_is_current(messages, snapshot):
            self.last_trigger_reason = "stale_snapshot"
            return messages
        candidate, wire_tokens, steps, checkpoint_text = rendered
        self.last_reduced_state = reduced
        self.last_checkpoint_text = checkpoint_text
        self.last_candidate = candidate
        self.last_wire_tokens = wire_tokens
        self.last_degradation_steps = steps
        if self._mode != "live":
            self._set_automatic_cooldown("shadow")
            return messages
        if candidate == messages:
            self._set_automatic_cooldown("ineffective")
            return messages
        self.compression_count += 1
        self._last_compression_made_progress = True
        return candidate
