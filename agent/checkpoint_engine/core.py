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
    if policy is not StructuredOutputPolicy.DISABLED and schema is not None:
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
        return source[self.start_char:self.end_char]


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


class MapResponse:
    VERSION = 1

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "integer", "enum": [cls.VERSION]},
                "source_event_ids": {"type": "array", "items": {"type": "integer"}},
                "facts": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["schema_version", "source_event_ids", "facts"],
        }


def parse_map_response(raw: str | Mapping[str, Any], *, expected_source_event_ids: tuple[int, ...]) -> MapShard:
    """Parse only the canonical schema; no fences, aliases, or salvage."""
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, Mapping) or payload.get("schema_version") != MapResponse.VERSION:
        raise ValueError("invalid map schema")
    source_ids = tuple(payload.get("source_event_ids", ()))
    if source_ids != tuple(expected_source_event_ids):
        raise ValueError("map source range mismatch")
    facts: list[MapFact] = []
    for n, item in enumerate(payload.get("facts", ())):
        if not isinstance(item, Mapping) or set(item) - {"kind", "text", "source_event_ids", "uncertain", "evidence"}:
            raise ValueError("invalid map fact")
        ids = tuple(item.get("source_event_ids", ()))
        if not ids or not set(ids) <= set(source_ids):
            raise ValueError("fact has invalid source ids")
        evidence = tuple(EvidenceSpan(str(s["event_id"]), int(s["start_char"]), int(s["end_char"])) for s in item.get("evidence", ()))
        facts.append(MapFact(str(item.get("kind", "observation")), str(item.get("text", "")), ids, evidence, bool(item.get("uncertain", False)), f"fact:{n}"))
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
            return old

    def compare_and_swap(self, session_id: str, revision: TranscriptRevision, generation: CheckpointGeneration) -> bool:
        with self._lock:
            if self._revisions.get(session_id) != revision:
                return False
            self._generations[session_id] = generation
            return True

    def generation(self, session_id: str) -> CheckpointGeneration | None:
        return self._generations.get(session_id)

    def append_trace(self, session_id: str, trace: TraceRecord) -> None:
        with self._lock:
            self._traces.setdefault(session_id, []).append(trace)

    def traces(self, session_id: str) -> tuple[TraceRecord, ...]:
        return tuple(self._traces.get(session_id, ()))


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
