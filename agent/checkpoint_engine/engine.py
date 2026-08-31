from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from collections.abc import Iterable
from threading import Lock
from typing import Any

from agent.context_engine import ContextEngine
from .core import (
    ActiveIntent, CausalGroup, CheckpointGeneration, CheckpointRejected,
    ArtifactReference, ContentAddressedArtifacts, DeterministicLanes,
    DurableCheckpointStore, Effect, HostLifecycleEvent, MapDisposition,
    MapFact, MapResponse, MapShard, ReducedState, StructuredOutputPolicy,
    TaskEpoch, ToolExecutionReceipt, ToolReceipt,
    TraceRecord, count_request_tokens, parse_map_response, prepare_provider_request,
)


class CheckpointContextEngine(ContextEngine):
    """Host-owned checkpoint projection with an optional auxiliary Map step."""

    def __init__(self, config: Mapping[str, Any] | None = None, *, store: DurableCheckpointStore | None = None, session_id: str = "default", map_caller: Callable[..., Any] | None = None, artifact_root: str | None = None) -> None:
        cfg = config or {}
        self.mode = str(cfg.get("mode", "shadow"))
        if self.mode not in {"shadow", "live"}:
            raise ValueError("checkpoint.mode must be shadow or live")
        if self.mode == "live" and cfg.get("raw_history", True) is False:
            raise ValueError("live checkpoint mode requires retained raw history")
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
        previous = self._store.generation(session_id)
        self.generation = previous.generation if previous else 0
        self._map_caller = map_caller
        self._map_routes = tuple(cfg.get("map_routes", ()))
        self._before_commit: Callable[[], Any] | None = None
        self._host_events: tuple[HostLifecycleEvent, ...] = ()
        self._tool_receipts: tuple[ToolExecutionReceipt, ...] = ()
        self._tool_receipt_lock = Lock()
        self._last_route_attempts: list[str] = []
        self._artifacts = ContentAddressedArtifacts(artifact_root) if artifact_root else None

    @property
    def name(self) -> str:
        return "checkpoint"

    def record_tool_receipt(self, receipt: ToolExecutionReceipt) -> None:
        """Record host authority from the actual tool-dispatch boundary."""
        if not isinstance(receipt, ToolExecutionReceipt):
            raise TypeError("checkpoint receipt must be typed")
        with self._tool_receipt_lock:
            self._tool_receipts = (*self._tool_receipts, receipt)

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

    def task_epochs(self, events: Sequence[HostLifecycleEvent]) -> tuple[TaskEpoch, ...]:
        """Derive epochs only from host lifecycle events, never model prose."""
        opened: dict[tuple[str, int], int] = {}
        closed: dict[tuple[str, int], int] = {}
        for event in sorted(events, key=lambda item: item.event_id):
            key = (event.task_id, event.epoch)
            if event.kind in {"task_started", "epoch_opened"}:
                opened.setdefault(key, event.event_id)
            elif event.kind in {"task_finished", "epoch_closed"}:
                closed[key] = event.event_id
        return tuple(
            TaskEpoch(f"{task_id}:{epoch}", event_id, closed.get((task_id, epoch)))
            for (task_id, epoch), event_id in opened.items()
        )

    def _extract_deterministic_lanes(self, messages: Sequence[Mapping[str, Any]], *, tool_receipts: Sequence[ToolExecutionReceipt] = ()) -> DeterministicLanes:
        users = [(i, m) for i, m in enumerate(messages) if self._role(m) == "user"]
        active = None
        if users:
            i, m = users[-1]
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(x.get("text", "")) for x in content if isinstance(x, Mapping))
            active = ActiveIntent(str(content), (i,), (self._row_id(m, i),))
        receipt_by_call = {receipt.tool_call_id: receipt for receipt in tool_receipts}
        effects: list[Effect] = []
        for i, m in enumerate(messages):
            if self._role(m) != "tool":
                continue
            rid = str(m.get("tool_call_id", f"tool:{i}"))
            receipt = receipt_by_call.get(rid)
            # A transcript field is display data.  Only a host receipt can
            # authorize an effect; absent one, this remains observational.
            status = receipt.status if receipt else "observed"
            source_ids = receipt.source_event_ids if receipt else (self._row_id(m, i),)
            typed_receipt = ToolReceipt(
                receipt.tool_call_id, receipt.tool_name, receipt.status,
                receipt.source_event_ids,
            ) if receipt else None
            effects.append(Effect(rid, receipt.tool_name if receipt else m.get("name"), status, (i,), source_ids, typed_receipt))
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
                facts.append(MapFact("tool_result", str(m.get("content", "")), (self._row_id(m, i),), uncertain=True, fact_id=f"local:{i}"))
        dispositions = tuple(MapDisposition(self._row_id(messages[i], i), "observed") for i in event_ids)
        return MapShard(tuple(self._row_id(messages[i], i) for i in event_ids), tuple(facts), dispositions)

    def _call_map(self, messages: Sequence[Mapping[str, Any]], event_ids: tuple[int, ...]) -> MapShard:
        self._last_route_attempts = []
        if self._map_caller is None:
            self._last_route_attempts.append("local")
            return self._local_map(messages, event_ids)
        source_ids = tuple(self._row_id(messages[i], i) for i in event_ids)
        source_events = {
            str(self._row_id(messages[i], i)): str(messages[i].get("content", ""))
            for i in event_ids
        }
        prompt = [{"role": "user", "content": json.dumps({"source_event_ids": source_ids, "messages": [messages[i] for i in event_ids]}, default=str)}]
        routes = self._map_routes or ({},)
        last_error: Exception | None = None
        for route in routes:
            route = dict(route)
            self._last_route_attempts.append(str(route.get("model", route.get("name", "configured"))))
            if self.policy is StructuredOutputPolicy.REQUIRED and route.get("structured_output") is False:
                continue
            try:
                request = prepare_provider_request(prompt, model=route.get("model"), policy=self.policy, schema=MapResponse.schema(), route_capabilities=route)
                raw = self._map_caller(request)
                if isinstance(raw, Mapping) and "choices" in raw:
                    raw = raw["choices"][0]["message"]["content"]
                return parse_map_response(raw, expected_source_event_ids=source_ids, source_events=source_events)
            except (RuntimeError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
        if last_error:
            raise last_error
        raise CheckpointRejected("no configured structured-capable map route")

    def _reduce(self, lanes: DeterministicLanes, shards: Iterable[MapShard], messages: Sequence[Mapping[str, Any]], *, host_events: Sequence[HostLifecycleEvent] = ()) -> ReducedState:
        all_facts: list[MapFact] = []
        dispositions: list[MapDisposition] = []
        for shard in shards:
            all_facts.extend(shard.facts)
            dispositions.extend(shard.dispositions)
        epochs = self.task_epochs(host_events)
        return ReducedState(lanes.active_intent, lanes.effects, tuple(all_facts), tuple(dispositions), epochs)

    def reduce_host_state(
        self,
        messages: Sequence[Mapping[str, Any]], *,
        host_events: Sequence[HostLifecycleEvent] = (),
        tool_receipts: Sequence[ToolExecutionReceipt] = (),
    ) -> ReducedState:
        lanes = self._extract_deterministic_lanes(messages, tool_receipts=tool_receipts)
        shards = (self._local_map(messages, tuple(range(len(messages)))),) if messages else ()
        return self._reduce(lanes, shards, messages, host_events=host_events)

    def externalize_artifact(self, content: str, *, media_type: str = "text/plain") -> ArtifactReference:
        if self._artifacts is None:
            raise CheckpointRejected("artifact store is not configured")
        return ArtifactReference(self._artifacts.put(content), media_type)

    def checkpoint_artifact_read(self, artifact_id: str) -> str:
        if self._artifacts is None:
            raise CheckpointRejected("artifact store is not configured")
        return self._artifacts.read(artifact_id)

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
        tail_start = max(0, len(messages) - self.protect_last_n)
        recent_indices: set[int] = set()
        for group in self._plan_causal_groups(messages):
            if any(i >= tail_start for i in group.event_indices):
                recent_indices.update(group.event_indices)
        recent = [dict(messages[i]) for i in sorted(recent_indices)]
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        checkpoint_message = {"role": "assistant", "content": checkpoint, "checkpoint_projection": True}
        for m in system + [checkpoint_message] + latest + recent:
            key = json.dumps(m, sort_keys=True, default=str)
            if key not in seen:
                out.append(m)
                seen.add(key)
        return out

    def _estimate_wire_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        return count_request_tokens({"messages": list(messages)})

    def prepare_provider_request(self, messages: Sequence[Mapping[str, Any]], *, model: str | None = None, tools: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        return prepare_provider_request(messages, model=model, tools=tools, policy=StructuredOutputPolicy.DISABLED)

    def compress(self, messages: list[dict[str, Any]], current_tokens: int | None = None, focus_topic: str | None = None, force: bool = False, memory_context: str = "", **kwargs: Any) -> list[dict[str, Any]]:
        if not isinstance(messages, list) or self._has_inflight_tools(messages):
            return messages
        self._store.record_raw(self.session_id, messages)
        raw_messages = self._store.raw_messages(self.session_id)
        source_messages = raw_messages or tuple(messages)
        revision = self._capture_revision(messages)
        try:
            self._host_events = tuple(kwargs.get("host_events", self._host_events))
            self._tool_receipts = tuple(kwargs.get("tool_receipts", self._tool_receipts))
            lanes = self._extract_deterministic_lanes(source_messages, tool_receipts=self._tool_receipts)
            shards = tuple(self._call_map(source_messages, ids) for ids in self._plan_map_shards(source_messages))
            reduced = self._reduce(lanes, shards, source_messages, host_events=self._host_events)
            checkpoint = self._render_checkpoint(reduced)
            candidate = self._projection(source_messages, checkpoint)
            if self._estimate_wire_tokens(candidate) > self.hard_max_wire_tokens:
                raise CheckpointRejected("projected request exceeds hard wire budget")
            raw_ids = tuple(self._row_id(message, index) for index, message in enumerate(source_messages))
            raw_ranges = self._event_ranges(raw_ids)
            generation = CheckpointGeneration(
                self.generation + 1, revision.revision, raw_ids,
                DurableCheckpointStore.signature(candidate), self.mode,
                self.generation or None, raw_ranges, reduced.artifacts,
            )
            if self._before_commit:
                self._before_commit()
            if not self._store.compare_and_swap(self.session_id, revision, generation):
                raise CheckpointRejected("transcript changed during checkpoint")
            self.generation = generation.generation
            self.compression_count += 1
            self.last_rejection = None
            if self.trace:
                reduced_hash = hashlib.sha256(json.dumps(asdict(reduced), default=str, sort_keys=True).encode()).hexdigest()
                prompt_hash = hashlib.sha256(json.dumps(candidate, default=str, sort_keys=True).encode()).hexdigest()
                schema_hash = hashlib.sha256(json.dumps(MapResponse.schema(), sort_keys=True).encode()).hexdigest()
                code_snapshot = dict(kwargs.get("code_snapshot") or {})
                commit_snapshot = dict(kwargs.get("commit_code_snapshot") or code_snapshot)
                if code_snapshot and commit_snapshot and code_snapshot != commit_snapshot:
                    raise CheckpointRejected("host code changed during checkpoint")
                code_head = str(code_snapshot.get("head", ""))
                code_tree = str(code_snapshot.get("tree", ""))
                dirty = bool(code_snapshot.get("dirty", False))
                dirty_diff_hash = str(code_snapshot.get("dirty_diff_hash", ""))
                configured_route = str(kwargs.get("configured_route", ""))
                physical_model = str(kwargs.get("physical_model", ""))
                final_request = prepare_provider_request(
                    candidate, model=physical_model or None, policy=StructuredOutputPolicy.DISABLED
                )
                identity_complete = bool(
                    code_head and code_tree and "dirty" in code_snapshot
                    and "dirty_diff_hash" in code_snapshot and physical_model
                    and (configured_route or self._last_route_attempts)
                )
                self.last_trace = TraceRecord(
                    self.generation, revision.revision, "auxiliary",
                    tuple(self._last_route_attempts), self.policy.value, schema_hash,
                    prompt_hash, reduced_hash, count_request_tokens(candidate), 0,
                    0, "stop", hashlib.sha256(checkpoint.encode()).hexdigest(),
                    code_tree, dirty, source="raw_transcript",
                    projection_hash=prompt_hash,
                    artifact_graph_hash=hashlib.sha256(json.dumps(reduced.artifacts, sort_keys=True).encode()).hexdigest(),
                    execution_provenance_hash=hashlib.sha256(json.dumps([asdict(effect) for effect in reduced.effects], default=str, sort_keys=True).encode()).hexdigest(),
                    continuation_window_hash=hashlib.sha256(json.dumps(candidate[-self.protect_last_n:], default=str, sort_keys=True).encode()).hexdigest(),
                    physical_model=physical_model,
                    extractor_hash=hashlib.sha256(b"checkpoint-extractor-v1").hexdigest(),
                    tokens_counting_mode="conservative_4_chars",
                    configured_route=configured_route,
                    actual_wire_mode=str(kwargs.get("wire_mode", self.policy.value)),
                    fallback_rejection=str(kwargs.get("fallback_rejection", "")),
                    final_request_hash=hashlib.sha256(json.dumps(final_request, sort_keys=True, default=str).encode()).hexdigest(),
                    code_head=code_head,
                    dirty_diff_hash=dirty_diff_hash,
                    map_shard_provenance=tuple(shard.source_event_ids for shard in shards),
                    execution_identity_complete=identity_complete,
                    benchmark_admissible=identity_complete,
                )
                self._store.append_trace(self.session_id, self.last_trace)
            return messages if self.mode == "shadow" else candidate
        except (CheckpointRejected, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.last_rejection = str(exc)
            return messages

    @staticmethod
    def _event_ranges(event_ids: Sequence[int]) -> tuple[tuple[int, int], ...]:
        if not event_ids:
            return ()
        ordered = sorted(set(event_ids))
        ranges: list[tuple[int, int]] = []
        start = previous = ordered[0]
        for event_id in ordered[1:]:
            if event_id != previous + 1:
                ranges.append((start, previous))
                start = event_id
            previous = event_id
        ranges.append((start, previous))
        return tuple(ranges)
