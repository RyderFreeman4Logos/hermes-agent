from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from collections.abc import Iterable
from typing import Any

from agent.context_engine import ContextEngine
from .core import (
    ActiveIntent, CausalGroup, CheckpointGeneration, CheckpointRejected,
    DeterministicLanes, DurableCheckpointStore, Effect, MapDisposition,
    MapFact, MapShard, ReducedState, StructuredOutputPolicy, TaskEpoch,
    TraceRecord, count_request_tokens, parse_map_response, prepare_provider_request,
)


class CheckpointContextEngine(ContextEngine):
    """Host-owned checkpoint projection with an optional auxiliary Map step."""

    def __init__(self, config: Mapping[str, Any] | None = None, *, store: DurableCheckpointStore | None = None, session_id: str = "default", map_caller: Callable[..., Any] | None = None) -> None:
        cfg = config or {}
        self.mode = str(cfg.get("mode", "shadow"))
        if self.mode not in {"shadow", "live"}:
            raise ValueError("checkpoint.mode must be shadow or live")
        self.trace = bool(cfg.get("trace", False))
        self.target_wire_tokens = int(cfg.get("target_wire_tokens", 48_000))
        self.hard_max_wire_tokens = int(cfg.get("hard_max_wire_tokens", 60_000))
        self.map_concurrency = int(cfg.get("map_concurrency", 2))
        self.max_map_shards = int(cfg.get("max_map_shards", 32))
        if self.target_wire_tokens <= 0 or self.hard_max_wire_tokens < self.target_wire_tokens:
            raise ValueError("checkpoint wire budgets are invalid")
        if self.map_concurrency < 1 or self.max_map_shards < 1:
            raise ValueError("checkpoint scheduler limits are invalid")
        self.policy = StructuredOutputPolicy(str(cfg.get("structured_output", cfg.get("policy", "preferred"))).lower())
        self.context_length = int(cfg.get("context_length", self.hard_max_wire_tokens))
        self.threshold_percent = float(cfg.get("threshold_percent", .75))
        self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self.protect_first_n = int(cfg.get("protect_first_n", 3))
        self.protect_last_n = int(cfg.get("protect_last_n", 6))
        self.last_prompt_tokens = self.last_completion_tokens = self.last_total_tokens = 0
        self.compression_count = 0
        self.generation = 0
        self.last_rejection: str | None = None
        self.last_trace: TraceRecord | None = None
        self._store = store or DurableCheckpointStore()
        self.session_id = session_id
        self._map_caller = map_caller

    @property
    def name(self) -> str:
        return "checkpoint"

    def update_from_response(self, usage: Mapping[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        self.last_total_tokens = int(usage.get("total_tokens", self.last_prompt_tokens + self.last_completion_tokens) or 0)

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        return bool(prompt_tokens is not None and self.threshold_tokens and prompt_tokens >= self.threshold_tokens)

    @staticmethod
    def _role(message: Mapping[str, Any]) -> str:
        return str(message.get("role", ""))

    @staticmethod
    def _row_id(message: Mapping[str, Any], index: int) -> int:
        value = message.get("_row_id", message.get("id", index))
        return int(value) if isinstance(value, int) else index

    def _has_inflight_tools(self, messages: Sequence[Mapping[str, Any]]) -> bool:
        calls = {str(c.get("id")) for m in messages if self._role(m) == "assistant" for c in (m.get("tool_calls") or ()) if isinstance(c, Mapping) and c.get("id")}
        results = {str(m.get("tool_call_id")) for m in messages if self._role(m) == "tool" and m.get("tool_call_id")}
        return bool(calls - results)

    def _capture_revision(self, messages: Sequence[Mapping[str, Any]]):
        return self._store.revision(self.session_id, messages)

    def _extract_deterministic_lanes(self, messages: Sequence[Mapping[str, Any]]) -> DeterministicLanes:
        users = [(i, m) for i, m in enumerate(messages) if self._role(m) == "user"]
        active = None
        if users:
            i, m = users[-1]
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(x.get("text", "")) for x in content if isinstance(x, Mapping))
            active = ActiveIntent(str(content), (i,), (self._row_id(m, i),))
        effects: list[Effect] = []
        for i, m in enumerate(messages):
            if self._role(m) != "tool":
                continue
            rid = str(m.get("tool_call_id", f"tool:{i}"))
            status = "failed" if str(m.get("status", "")).lower() in {"error", "failed"} else ("succeeded" if str(m.get("status", "")).lower() in {"success", "succeeded"} else "observed")
            effects.append(Effect(rid, m.get("name"), status, (i,), (self._row_id(m, i),)))
        recent = tuple(range(max(0, len(messages) - self.protect_last_n), len(messages)))
        return DeterministicLanes(active, tuple(effects), (), recent)

    def _plan_causal_groups(self, messages: Sequence[Mapping[str, Any]]) -> tuple[CausalGroup, ...]:
        groups: list[CausalGroup] = []
        used: set[int] = set()
        for i, m in enumerate(messages):
            if i in used:
                continue
            ids = {str(c.get("id")) for c in (m.get("tool_calls") or ()) if isinstance(c, Mapping) and c.get("id")} if self._role(m) == "assistant" else set()
            if ids:
                members = [i]
                for j in range(i + 1, len(messages)):
                    if self._role(messages[j]) == "tool" and str(messages[j].get("tool_call_id")) in ids:
                        members.append(j)
                used.update(members)
                groups.append(CausalGroup(tuple(members)))
            else:
                groups.append(CausalGroup((i,)))
                used.add(i)
        return tuple(groups)

    def _plan_map_shards(self, messages: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, ...], ...]:
        groups = self._plan_causal_groups(messages)
        size = max(1, (len(groups) + self.max_map_shards - 1) // self.max_map_shards)
        shards: list[tuple[int, ...]] = []
        for n in range(0, len(groups), size):
            indices = tuple(i for group in groups[n:n + size] for i in group.event_indices)
            if indices:
                shards.append(indices)
        return tuple(shards[:self.max_map_shards])

    def _local_map(self, messages: Sequence[Mapping[str, Any]], event_ids: tuple[int, ...]) -> MapShard:
        # Deterministic facts are observations only.  Assistant prose cannot
        # promote an action to succeeded; only a tool receipt can do that.
        facts: list[MapFact] = []
        for i in event_ids:
            m = messages[i]
            if self._role(m) == "tool":
                status = str(m.get("status", "observed"))
                facts.append(MapFact("tool_result", str(m.get("content", "")), (self._row_id(m, i),), uncertain=status not in {"success", "succeeded"}, fact_id=f"local:{i}"))
        dispositions = tuple(MapDisposition(self._row_id(messages[i], i), "observed") for i in event_ids)
        return MapShard(tuple(self._row_id(messages[i], i) for i in event_ids), tuple(facts), dispositions)

    def _call_map(self, messages: Sequence[Mapping[str, Any]], event_ids: tuple[int, ...]) -> MapShard:
        if self._map_caller is None:
            return self._local_map(messages, event_ids)
        prompt = [{"role": "user", "content": json.dumps({"source_event_ids": event_ids, "messages": [messages[i] for i in event_ids]}, default=str)}]
        request = prepare_provider_request(prompt, policy=self.policy, schema=__import__("agent.checkpoint_engine.core", fromlist=["MapResponse"]).MapResponse.schema())
        raw = self._map_caller(request)
        if isinstance(raw, Mapping) and "choices" in raw:
            raw = raw["choices"][0]["message"]["content"]
        return parse_map_response(raw, expected_source_event_ids=event_ids)

    def _reduce(self, lanes: DeterministicLanes, shards: Iterable[MapShard], messages: Sequence[Mapping[str, Any]]) -> ReducedState:
        all_facts: list[MapFact] = []
        dispositions: list[MapDisposition] = []
        for shard in shards:
            all_facts.extend(shard.facts)
            dispositions.extend(shard.dispositions)
        epochs = tuple(TaskEpoch(f"epoch:{i}", d.source_event_id) for i, d in enumerate(dispositions) if d.status in {"in_progress", "unresolved"})
        return ReducedState(lanes.active_intent, lanes.effects, tuple(all_facts), tuple(dispositions), epochs)

    def _render_checkpoint(self, reduced: ReducedState) -> str:
        lines = ["CHECKPOINT (host-authored; raw transcript remains authoritative)"]
        if reduced.active_intent:
            lines.append(f"Active intent: {reduced.active_intent.content}")
        for effect in reduced.effects:
            lines.append(f"Observed effect {effect.tool_call_id}: {effect.status}")
        for fact in reduced.facts:
            state = "uncertain" if fact.uncertain else "observed"
            lines.append(f"{state} {fact.kind}: {fact.text}")
        for epoch in reduced.epochs:
            lines.append(f"Open epoch {epoch.epoch_id} from event {epoch.opened_by_event_id}")
        return "\n".join(lines)

    def _projection(self, messages: Sequence[Mapping[str, Any]], checkpoint: str) -> list[dict[str, Any]]:
        system = [dict(m) for m in messages if self._role(m) == "system"]
        users = [dict(m) for m in messages if self._role(m) == "user"]
        latest = users[-1:] if users else []
        recent = [dict(messages[i]) for i in range(max(0, len(messages) - self.protect_last_n), len(messages))]
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for m in system + [{"role": "assistant", "content": checkpoint}] + latest + recent:
            key = json.dumps(m, sort_keys=True, default=str)
            if key not in seen:
                out.append(m)
                seen.add(key)
        return out

    def _estimate_wire_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        return count_request_tokens({"messages": list(messages)})

    def compress(self, messages: list[dict[str, Any]], current_tokens: int | None = None, focus_topic: str | None = None, force: bool = False, memory_context: str = "", **kwargs: Any) -> list[dict[str, Any]]:
        if not isinstance(messages, list) or self._has_inflight_tools(messages):
            return messages
        revision = self._capture_revision(messages)
        try:
            lanes = self._extract_deterministic_lanes(messages)
            shards = tuple(self._call_map(messages, ids) for ids in self._plan_map_shards(messages))
            reduced = self._reduce(lanes, shards, messages)
            checkpoint = self._render_checkpoint(reduced)
            candidate = self._projection(messages, checkpoint)
            if self._estimate_wire_tokens(candidate) > self.hard_max_wire_tokens:
                raise CheckpointRejected("projected request exceeds hard wire budget")
            generation = CheckpointGeneration(self.generation + 1, revision.revision, revision.source_event_ids, DurableCheckpointStore.signature(candidate), self.mode)
            if self.mode == "live" and not self._store.compare_and_swap(self.session_id, revision, generation):
                raise CheckpointRejected("transcript changed during checkpoint")
            self.generation = generation.generation
            self.compression_count += 1
            self.last_rejection = None
            if self.trace:
                reduced_hash = hashlib.sha256(json.dumps(asdict(reduced), default=str, sort_keys=True).encode()).hexdigest()
                self.last_trace = TraceRecord(self.generation, revision.revision, "auxiliary", ("configured",), self.policy.value, "", "", reduced_hash, count_request_tokens(candidate), 0, 0, "stop", hashlib.sha256(checkpoint.encode()).hexdigest(), "unknown", False)
                self._store.append_trace(self.session_id, self.last_trace)
            return messages if self.mode == "shadow" else candidate
        except (CheckpointRejected, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.last_rejection = str(exc)
            return messages
