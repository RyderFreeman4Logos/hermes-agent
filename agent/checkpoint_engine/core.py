"""Small, typed building blocks for the opt-in checkpoint engine.

The raw transcript remains authoritative.  This module only builds a
request-scoped projection and publishes it after a revision check.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence


class StructuredOutputPolicy(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    DISABLED = "disabled"


class StructuredOutputUnavailable(RuntimeError):
    """The selected route cannot honor a required structured response."""


class CheckpointRejected(RuntimeError):
    pass


class CheckpointMapCallRejected(CheckpointRejected):
    """A failed physical Map attempt with its transport-boundary record."""

    def __init__(self, identity: Mapping[str, Any], cause: Exception) -> None:
        super().__init__(str(cause))
        self.identity = dict(identity)


@dataclass(frozen=True)
class HostLifecycleEvent:
    """A host-authored lifecycle boundary; models cannot create one."""

    event_id: int
    kind: str
    task_id: str
    epoch: int


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    media_type: str = "text/plain"


@dataclass(frozen=True)
class ToolExecutionReceipt:
    """The only authority for whether a side effect completed."""

    tool_call_id: str
    tool_name: str
    effect_class: str
    status: str
    exit_code: int | None
    error_type: str | None
    artifact_refs: tuple[ArtifactReference | str, ...]
    source_event_ids: tuple[int, ...]


def prepare_provider_request(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    policy: StructuredOutputPolicy = StructuredOutputPolicy.DISABLED,
    schema: Mapping[str, Any] | None = None,
    route_capabilities: Mapping[str, Any] | None = None,
    tokenizer: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Prepare the exact provider payload without sending it.

    This is shared by the real send path and the checkpoint wire gate.  A
    required policy never turns into a prompt instruction or ``extra_body``.
    """
    try:
        policy = StructuredOutputPolicy(policy)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown structured-output policy") from exc
    caps = route_capabilities or {}
    if policy is StructuredOutputPolicy.REQUIRED and caps.get("structured_output") is False:
        raise StructuredOutputUnavailable("route does not support structured output")
    request: dict[str, Any] = {"messages": [dict(m) for m in messages]}
    if model:
        request["model"] = model
    if tools:
        request["tools"] = [dict(tool) for tool in tools]
    if policy is not StructuredOutputPolicy.DISABLED and schema is not None and caps.get("structured_output", True) is not False:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "checkpoint_map", "strict": True, "schema": dict(schema)},
        }
    request["estimated_input_tokens"] = count_request_tokens(request, tokenizer=tokenizer)
    return request


def count_request_tokens(value: Any, *, tokenizer: Callable[[str], int] | None = None) -> int:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if tokenizer:
        return max(0, int(tokenizer(text)))
    # ponytail: conservative four-character estimate; replace only when the
    # host exposes its exact tokenizer.
    return (len(text) + 3) // 4


@dataclass(frozen=True)
class EvidenceSpan:
    event_id: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        if self.start_char < 0 or self.end_char < self.start_char:
            raise ValueError("invalid evidence span")

    def text(self, source: str) -> str:
        if self.end_char > len(source):
            raise ValueError("evidence span exceeds source bounds")
        return source[self.start_char:self.end_char]


def extract_canonical_evidence(
    span: EvidenceSpan, events: Mapping[str | int, str]
) -> str:
    """Extract evidence from the host-owned event, never model text."""
    source = events.get(span.event_id)
    if source is None:
        source = events.get(int(span.event_id)) if str(span.event_id).isdigit() else None
    if not isinstance(source, str):
        raise ValueError("evidence event not found")
    return span.text(source)


@dataclass(frozen=True)
class ActiveIntent:
    content: str
    event_indices: tuple[int, ...]
    source_event_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ToolReceipt:
    receipt_id: str
    operation: str
    status: str
    source_event_ids: tuple[int, ...]
    repo_head: str | None = None
    tree_hash: str | None = None


@dataclass(frozen=True)
class Effect:
    tool_call_id: str
    operation: str | None
    status: str
    event_indices: tuple[int, ...]
    source_event_ids: tuple[int, ...] = ()
    receipt: ToolReceipt | None = None


@dataclass(frozen=True)
class CausalGroup:
    event_indices: tuple[int, ...]


@dataclass(frozen=True)
class MapFact:
    kind: str
    text: str
    source_event_ids: tuple[int, ...]
    evidence: tuple[EvidenceSpan, ...] = ()
    uncertain: bool = False
    fact_id: str | None = None
    summary: str = ""


@dataclass(frozen=True)
class MapDisposition:
    source_event_id: int
    status: str
    fact_ids: tuple[str, ...] = ()
    recovery_ref: str | None = None


@dataclass(frozen=True)
class MapShard:
    source_event_ids: tuple[int, ...]
    facts: tuple[MapFact, ...]
    dispositions: tuple[MapDisposition, ...]


@dataclass(frozen=True)
class DeterministicLanes:
    active_intent: ActiveIntent | None
    effects: tuple[Effect, ...]
    constraints: tuple[str, ...] = ()
    recent_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class TaskEpoch:
    epoch_id: str
    opened_by_event_id: int
    closed_by_event_id: int | None = None


@dataclass(frozen=True)
class ReducedState:
    active_intent: ActiveIntent | None
    effects: tuple[Effect, ...]
    facts: tuple[MapFact, ...]
    dispositions: tuple[MapDisposition, ...]
    epochs: tuple[TaskEpoch, ...] = ()
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranscriptRevision:
    revision: int
    source_event_ids: tuple[int, ...]
    signature: str


@dataclass(frozen=True)
class CheckpointGeneration:
    generation: int
    source_revision: int
    source_event_ids: tuple[int, ...]
    checkpoint_hash: str
    mode: str
    parent_generation: int | None = None
    raw_event_ranges: tuple[tuple[int, int], ...] = ()
    artifact_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceRecord:
    generation: int
    source_revision: int
    model: str
    route_attempts: tuple[str, ...]
    structured_wire_mode: str
    schema_hash: str
    prompt_hash: str
    reducer_hash: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    finish_reason: str
    response_hash: str
    code_tree: str
    dirty: bool
    source: str = "raw_transcript"
    projection_hash: str = ""
    artifact_graph_hash: str = ""
    execution_provenance_hash: str = ""
    continuation_window_hash: str = ""
    physical_model: str = ""
    prompt_hash_version: str = "v1"
    extractor_hash: str = ""
    tokens_counting_mode: str = "conservative"
    configured_route: str = ""
    actual_wire_mode: str = ""
    fallback_rejection: str = ""
    final_request_hash: str = ""
    code_head: str = ""
    dirty_diff_hash: str = ""
    map_shard_provenance: tuple[tuple[int, ...], ...] = ()
    map_attempt_records: tuple["MapAttemptRecord", ...] = ()
    execution_identity_complete: bool = False
    benchmark_admissible: bool = False


@dataclass(frozen=True)
class MapAttemptRecord:
    configured_route: str | None = None
    physical_model: str | None = None
    actual_wire_mode: str | None = None
    fallback_rejection: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    finish_reason: str | None = None
    response_hash: str | None = None
    code_head: str | None = None
    code_tree: str | None = None
    dirty: bool | None = None
    dirty_diff_hash: str | None = None


class MapResponse:
    @classmethod
    def schema(cls, source_event_ids: Sequence[int] = ()) -> dict[str, Any]:
        event_indexes = list(range(len(source_event_ids)))
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "facts": {"type": "array", "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string"},
                        "text": {"type": "string"},
                        "summary": {"type": "string"},
                        "uncertain": {"type": "boolean"},
                        "evidence": {"type": "array", "items": {
                            "type": "object", "additionalProperties": False,
                            "properties": {
                                "event_index": {"type": "integer", "enum": event_indexes},
                                "start_char": {"type": "integer", "minimum": 0},
                                "end_char": {"type": "integer", "minimum": 0},
                            },
                            "required": ["event_index", "start_char", "end_char"],
                        }},
                    },
                    "required": ["kind", "evidence"],
                }},
            },
            "required": ["facts"],
        }


def _event_index(span: Mapping[str, Any], source_count: int) -> int:
    index = span.get("event_index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < source_count:
        raise ValueError("invalid evidence event index")
    return index


def parse_map_response(
    raw: str | Mapping[str, Any], *, expected_source_event_ids: tuple[int, ...],
    source_events: Mapping[str | int, str] | None = None,
) -> MapShard:
    """Parse only the canonical schema; no fences, aliases, or salvage."""
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, Mapping) or "facts" not in payload:
        raise ValueError("invalid map schema")
    source_ids = tuple(expected_source_event_ids)
    if set(payload) - {"schema_version", "source_event_ids", "facts"}:
        raise ValueError("invalid map schema")
    if "source_event_ids" in payload and tuple(payload["source_event_ids"]) != source_ids:
        raise ValueError("map source range mismatch")
    facts: list[MapFact] = []
    for item in payload.get("facts", ()):
        if not isinstance(item, Mapping) or set(item) - {"kind", "text", "summary", "source_event_ids", "uncertain", "evidence"}:
            raise ValueError("invalid map fact")
        if "source_event_ids" in item and not set(item["source_event_ids"]) <= set(source_ids):
            raise ValueError("fact has invalid source ids")
        raw_evidence = item.get("evidence", ())
        if (not isinstance(raw_evidence, (list, tuple))
                or not all(isinstance(span, Mapping) for span in raw_evidence)):
            raise ValueError("invalid evidence")
        evidence = tuple(
            EvidenceSpan(
                str(source_ids[0] if len(source_ids) == 1 else source_ids[_event_index(span, len(source_ids))]),
                int(span["start_char"]), int(span["end_char"]),
            )
            for span in raw_evidence
        )
        if not evidence:
            raise ValueError("fact requires evidence")
        if source_events is None:
            raise ValueError("fact evidence requires source events")
        canonical_text = "\n".join(
            extract_canonical_evidence(span, source_events) for span in evidence
        )
        ids = tuple(dict.fromkeys(int(span.event_id) for span in evidence))
        identity = json.dumps(
            [str(item.get("kind", "observation")), canonical_text, ids,
             [(e.event_id, e.start_char, e.end_char) for e in evidence]],
            sort_keys=True, separators=(",", ":"),
        )
        fact_id = "fact:" + sha256(identity.encode()).hexdigest()[:16]
        facts.append(MapFact(
            str(item.get("kind", "observation")), canonical_text, ids, evidence,
            bool(item.get("uncertain", False)), fact_id,
            str(item.get("summary", item.get("text", ""))),
        ))
    dispositions = tuple(
        MapDisposition(event_id, "unresolved", recovery_ref=f"session-event:{event_id}")
        for event_id in source_ids
    )
    return MapShard(source_ids, tuple(facts), dispositions)


class DurableCheckpointStore:
    """Tiny append-only CAS store; integrations may wrap SessionDB."""
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else None
        self._lock = threading.RLock()
        self._revisions: dict[str, TranscriptRevision] = {}
        self._generations: dict[str, CheckpointGeneration] = {}
        self._traces: dict[str, list[TraceRecord]] = {}
        self._raw_events: dict[str, list[dict[str, Any]]] = {}
        self._artifacts: dict[str, str] = {}
        self._load()

    @property
    def _state_path(self) -> Path | None:
        return self.root / "checkpoint-state.json" if self.root else None

    def _load(self) -> None:
        path = self._state_path
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for session_id, item in payload.get("revisions", {}).items():
                self._revisions[session_id] = TranscriptRevision(**item)
            for session_id, item in payload.get("generations", {}).items():
                item["source_event_ids"] = tuple(item.get("source_event_ids", ()))
                item["raw_event_ranges"] = tuple(tuple(pair) for pair in item.get("raw_event_ranges", ()))
                item["artifact_dependencies"] = tuple(item.get("artifact_dependencies", ()))
                self._generations[session_id] = CheckpointGeneration(**item)
            for session_id, items in payload.get("traces", {}).items():
                for item in items:
                    item["map_attempt_records"] = tuple(
                        MapAttemptRecord(**record)
                        for record in item.get("map_attempt_records", ())
                    )
                self._traces[session_id] = [TraceRecord(**item) for item in items]
            self._raw_events = {
                session_id: [dict(event) for event in events]
                for session_id, events in payload.get("raw_events", {}).items()
            }
            self._artifacts = {
                str(digest): str(content)
                for digest, content in payload.get("artifacts", {}).items()
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # An incomplete optional store must not make shadow operation crash.
            return

    def _persist(self) -> None:
        path = self._state_path
        if path is None:
            return
        payload = {
            "revisions": {key: asdict(value) for key, value in self._revisions.items()},
            "generations": {key: asdict(value) for key, value in self._generations.items()},
            "traces": {key: [asdict(value) for value in values] for key, values in self._traces.items()},
            "raw_events": self._raw_events,
            "artifacts": self._artifacts,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")

    @staticmethod
    def signature(messages: Sequence[Mapping[str, Any]]) -> str:
        blob = json.dumps(list(messages), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        return sha256(blob.encode()).hexdigest()

    def revision(self, session_id: str, messages: Sequence[Mapping[str, Any]]) -> TranscriptRevision:
        with self._lock:
            old = self._revisions.get(session_id)
            ids = tuple(int(m["_row_id"]) for m in messages if isinstance(m.get("_row_id"), int))
            signature = self.signature(messages)
            if old is None or old.signature != signature:
                old = TranscriptRevision((old.revision + 1 if old else 1), ids, signature)
                self._revisions[session_id] = old
                self._persist()
            return old

    def compare_and_swap(self, session_id: str, revision: TranscriptRevision, generation: CheckpointGeneration) -> bool:
        with self._lock:
            if self._revisions.get(session_id) != revision:
                return False
            self._generations[session_id] = generation
            self._persist()
            return True

    def generation(self, session_id: str) -> CheckpointGeneration | None:
        return self._generations.get(session_id)

    def append_trace(self, session_id: str, trace: TraceRecord) -> None:
        with self._lock:
            self._traces.setdefault(session_id, []).append(trace)
            self._persist()

    def traces(self, session_id: str) -> tuple[TraceRecord, ...]:
        return tuple(self._traces.get(session_id, ()))

    def record_raw(self, session_id: str, messages: Sequence[Mapping[str, Any]]) -> None:
        """Retain immutable input events; rendered projections are excluded."""
        with self._lock:
            events = self._raw_events.setdefault(session_id, [])
            seen = {event.get("_row_id") for event in events}
            for message in messages:
                if message.get("checkpoint_projection") or message.get("_checkpoint"):
                    continue
                row_id = message.get("_row_id")
                if row_id is not None and row_id in seen:
                    continue
                event = dict(message)
                if row_id is None:
                    event["_row_id"] = len(events)
                events.append(event)
            self._persist()

    def raw_messages(self, session_id: str) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(event) for event in self._raw_events.get(session_id, ()))

    def put_artifact(self, content: str) -> str:
        digest = sha256(content.encode()).hexdigest()
        with self._lock:
            self._artifacts.setdefault(digest, content)
            self._persist()
        return digest

    def read_artifact(self, digest: str) -> str:
        with self._lock:
            content = self._artifacts[digest]
        if sha256(content.encode()).hexdigest() != digest:
            raise ValueError("artifact hash mismatch")
        return content


class ContentAddressedArtifacts:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, content: str) -> str:
        digest = sha256(content.encode()).hexdigest()
        path = self.root / digest
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        return digest

    def read(self, digest: str) -> str:
        content = (self.root / digest).read_text(encoding="utf-8")
        if sha256(content.encode()).hexdigest() != digest:
            raise ValueError("artifact hash mismatch")
        return content
