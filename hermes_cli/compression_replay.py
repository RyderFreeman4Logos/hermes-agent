"""Deterministic, side-effect-free replay of an exported compression trace."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


class ReplaySafetyError(RuntimeError):
    """Replay attempted to cross the recorded, read-only boundary."""


class ReplayIntegrityError(ValueError):
    """The corpus was changed or contains an invalid manifest."""


@dataclass(frozen=True)
class ReplayConfig:
    engine: str = "checkpoint"
    provider: Optional[str] = None
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    fallback_policy: str = "strict-single-route"
    read_only: bool = True
    max_concurrency: int = 2
    execute_real_tools: bool = False
    timeout: Optional[float] = None
    repetitions: int = 1
    inference_parameters: Mapping[str, Any] = field(default_factory=dict)
    engine_version: str = "compression-replay-v1"
    prompt_version: str = "compression-replay-prompt-v1"
    schema_version: str = "compression-replay-schema-v1"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.fallback_policy not in {"strict-single-route", "configured-only", "configured-fallback", "production", "production-fallback"}:
            raise ValueError("fallback_policy must be strict-single-route or configured fallback")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if not self.read_only:
            raise ReplaySafetyError("compression replay is always read-only")

    @property
    def policy_label(self) -> str:
        return "strict-single-route" if self.fallback_policy == "strict-single-route" else "configured-fallback"


class RecordedToolAdapter:
    """Resolve requested tool calls only against exported continuation rows."""

    def __init__(self, events: Iterable[Mapping[str, Any]]) -> None:
        self._by_id: Dict[str, Mapping[str, Any]] = {}
        self._by_name: Dict[str, Mapping[str, Any]] = {}
        for event in events:
            if event.get("role") != "tool":
                continue
            call_id = event.get("tool_call_id")
            name = event.get("tool_name")
            if call_id:
                self._by_id[str(call_id)] = event
            if name:
                self._by_name[str(name)] = event

    def execute(self, call: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a recorded result, never invoke a command/network executor."""
        call_id = call.get("id") or call.get("tool_call_id")
        name = call.get("name") or call.get("tool_name")
        event = self._by_id.get(str(call_id)) if call_id else None
        event = event or (self._by_name.get(str(name)) if name else None)
        if event is None:
            raise ReplaySafetyError(f"unrecorded tool call blocked: {name or call_id or 'unknown'}")
        return {
            "recorded": True,
            "tool_call_id": event.get("tool_call_id"),
            "tool_name": event.get("tool_name"),
            "content": event.get("content"),
            "effect_disposition": event.get("effect_disposition"),
        }


# Kept public for callers that used the name in evaluation notebooks.
RecordedResponseAdapter = RecordedToolAdapter


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayIntegrityError(f"cannot read {path}") from exc


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    path.chmod(0o600)


def _verify_manifest(root: Path) -> Dict[str, Any]:
    manifest = _load(root / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ReplayIntegrityError("manifest is missing its file hash map")
    for relative, expected in manifest["files"].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or not isinstance(expected, str):
            raise ReplayIntegrityError(f"invalid manifest path: {relative}")
        path = root / relative_path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ReplayIntegrityError(f"manifest hash mismatch: {relative}")
    return manifest


def _git_provenance() -> Dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        return {"git_commit": sha, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


class ReplayRunner:
    def __init__(
        self,
        corpus: Path,
        config: Optional[ReplayConfig] = None,
        *,
        infer: Optional[Callable[[Dict[str, Any]], Any]] = None,
        tool_executor: Any = None,
    ) -> None:
        self.corpus = Path(corpus).resolve()
        self.config = config or ReplayConfig()
        self.infer = infer
        self.tool_executor = tool_executor
        self.manifest = _verify_manifest(self.corpus)
        self.points = sorted(
            (self.corpus / "points").glob("*/metadata.json"),
            key=lambda path: int(path.parent.name) if path.parent.name.isdigit() else path.parent.name,
        )

    def _route(self, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        recorded = (metadata.get("route") if isinstance(metadata.get("route"), Mapping) else {}) or {}
        route = {
            "provider": self.config.provider if self.config.provider is not None else recorded.get("provider"),
            "model": self.config.model if self.config.model is not None else recorded.get("model"),
            "reasoning_effort": self.config.reasoning_effort if self.config.reasoning_effort is not None else recorded.get("reasoning_effort"),
        }
        if recorded.get("physical_model") is not None:
            route["physical_model"] = recorded["physical_model"]
        route["fallback_policy"] = self.config.fallback_policy
        if recorded.get("fallback_chain") is not None:
            route["fallback_chain"] = recorded["fallback_chain"]
        return route

    def _cache_key(self, metadata: Mapping[str, Any], route: Mapping[str, Any]) -> str:
        return _canonical_hash({
            "source_content_hash": self.manifest.get("source_content_hash"),
            "source_event_hash": metadata.get("source_event_hash"),
            "engine": self.config.engine,
            "engine_version": self.config.engine_version,
            "prompt_version": self.config.prompt_version,
            "schema_version": self.config.schema_version,
            "route": route,
            "reasoning_effort": route.get("reasoning_effort"),
            "inference_parameters": dict(self.config.inference_parameters),
            "fallback_policy": self.config.fallback_policy,
            "timeout": self.config.timeout,
            "repetitions": self.config.repetitions,
            "seed": self.config.seed,
        })

    def _point(self, metadata_path: Path) -> Dict[str, Any]:
        point = metadata_path.parent
        return {
            "metadata": _load(metadata_path),
            "pre_context": _load(point / "pre-context.json"),
            "post_context": _load(point / "post-context.json"),
            "continuation": _read_jsonl(point / "continuation.jsonl"),
        }

    def _candidate(self, point: Mapping[str, Any], route: Mapping[str, Any], state: Any) -> Any:
        if self.infer is None:
            # No provider is contacted by default. This makes an export useful
            # for structural/replay tests while an explicit adapter can perform
            # a model benchmark.
            return point["post_context"]
        request = {
            "engine": self.config.engine,
            "pre_context": state,
            "source_post_context": point["post_context"],
            "route": dict(route),
            "reasoning_effort": route.get("reasoning_effort"),
            "fallback_policy": self.config.fallback_policy,
            "inference_parameters": dict(self.config.inference_parameters),
            "prompt_version": self.config.prompt_version,
            "schema_version": self.config.schema_version,
            "read_only": True,
        }
        result = self.infer(request)
        return result if result is not None else point["post_context"]

    @staticmethod
    def _tool_calls(candidate: Any) -> List[Mapping[str, Any]]:
        if not isinstance(candidate, Mapping):
            return []
        calls = candidate.get("tool_calls")
        return calls if isinstance(calls, list) else []

    def _continuation(self, point: Mapping[str, Any], candidate: Any) -> Dict[str, Any]:
        if self.config.execute_real_tools:
            raise ReplaySafetyError("real tool execution is prohibited in Mode C")
        adapter = RecordedToolAdapter(point["continuation"])
        responses = [adapter.execute(call) for call in self._tool_calls(candidate)]
        return {"tool_calls": len(responses), "responses": responses, "recorded_only": True}

    def run(self, *, mode: str = "A", output: Optional[Path] = None, rerun: bool = False) -> Dict[str, Any]:
        mode = mode.upper()
        if mode not in {"A", "B", "C"}:
            raise ValueError("mode must be A, B, or C")
        if self.config.execute_real_tools:
            raise ReplaySafetyError("real tool execution is prohibited in compression replay")
        run_id = _canonical_hash({"manifest": self.manifest.get("source_content_hash"), "config": asdict(self.config), "mode": mode})[:16]
        run_dir = Path(output or (self.corpus / "runs" / run_id)).resolve()
        if run_dir == self.corpus:
            raise ValueError("replay output must be separate from the export corpus")
        run_dir.mkdir(parents=True, exist_ok=True)
        results_dir = run_dir / "points"
        results_dir.mkdir(exist_ok=True)
        route = self._route({})
        results: List[Dict[str, Any]] = []
        state: Any = None
        for index, metadata_path in enumerate(self.points, 1):
            point = self._point(metadata_path)
            metadata = point["metadata"]
            route = self._route(metadata)
            result_path = results_dir / f"{index}.json"
            cache_key = self._cache_key(metadata, route)
            cache_path = run_dir / "cache" / f"{cache_key}.json"
            if not rerun and result_path.exists():
                result = _load(result_path)
                if result.get("status") == "completed":
                    results.append(result)
                    state = result.get("candidate", state)
                    continue
            if not rerun and cache_path.exists():
                result = _load(cache_path)
                if result.get("status") != "completed":
                    result = None
                else:
                    result["from_cache"] = True
            else:
                result = None
            if result is None:
                if mode == "A" or state is None:
                    state = point["pre_context"]
                try:
                    candidate = self._candidate(point, route, state)
                    continuation = self._continuation(point, candidate) if mode == "C" else None
                    result = {
                        "generation": index,
                        "route": dict(route),
                        "reasoning_effort": route.get("reasoning_effort"),
                        "candidate": candidate,
                        "continuation": continuation,
                        "cache_key": cache_key,
                        "from_cache": False,
                        "status": "completed",
                    }
                except ReplaySafetyError:
                    raise
                except Exception as exc:
                    result = {
                        "generation": index,
                        "route": dict(route),
                        "reasoning_effort": route.get("reasoning_effort"),
                        "cache_key": cache_key,
                        "from_cache": False,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                cache_path.parent.mkdir(exist_ok=True)
                _atomic_json(cache_path, result)
            # Mode A resets from the point pre-state on every iteration. Mode B
            # carries only the replay result forward, never the original result.
            if mode == "B":
                state = result.get("candidate", state)
            else:
                state = None
            _atomic_json(result_path, result)
            results.append(result)
        manifest = {
            "schema_version": "compression-replay-run-v1",
            "run_id": run_id,
            "mode": mode,
            "policy_label": self.config.policy_label,
            "config": asdict(self.config),
            "route": route,
            "export_manifest_hash": hashlib.sha256((self.corpus / "manifest.json").read_bytes()).hexdigest(),
            "provenance": _git_provenance(),
            "started_at": time.time(),
            "completed_points": sum(result.get("status") == "completed" for result in results),
            "failed_points": sum(result.get("status") != "completed" for result in results),
            "points": results,
        }
        _atomic_json(run_dir / "run-manifest.json", manifest)
        return manifest


def run_replay(corpus: Path, config: Optional[ReplayConfig] = None, **kwargs: Any) -> Dict[str, Any]:
    return ReplayRunner(corpus, config, **kwargs).run()


def compare_replays(run_a: Path, run_b: Path, output: Path) -> Dict[str, Any]:
    """Create a small self-contained paired report without claiming quality."""
    a, b = Path(run_a), Path(run_b)
    manifest_a, manifest_b = _load(a / "run-manifest.json"), _load(b / "run-manifest.json")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    points_a, points_b = manifest_a.get("points", []), manifest_b.get("points", [])
    if len(points_a) != len(points_b):
        raise ValueError("replay reports have different compression-point counts")
    for left, right in zip(points_a, points_b):
        if left.get("generation") != right.get("generation"):
            raise ValueError("replay reports are not paired by generation")
        rows.append({"generation": left.get("generation"), "a_status": left.get("status"), "b_status": right.get("status")})
    summary = {"run_a": manifest_a.get("run_id"), "run_b": manifest_b.get("run_id"), "paired_points": len(rows), "note": "Replay report contains structural outcomes; continuation quality requires an evaluator."}
    _atomic_json(output / "summary.json", summary)
    (output / "catastrophic-errors.jsonl").write_bytes(b"")
    (output / "catastrophic-errors.jsonl").chmod(0o600)
    (output / "failures").mkdir(exist_ok=True)
    (output / "per-session").mkdir(exist_ok=True)
    points_dir = output / "per-compression-point"
    points_dir.mkdir(exist_ok=True)
    for row in rows:
        _atomic_json(points_dir / f"{row['generation']}.json", row)
    _atomic_json(output / "per-session" / "paired.json", summary)
    with (output / "paired-metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generation", "a_status", "b_status"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.md").write_text(f"# Compression replay comparison\n\nPaired points: {len(rows)}.\n", encoding="utf-8")
    run_manifest = {"schema_version": "compression-replay-report-v1", "summary": summary, "run_a": manifest_a, "run_b": manifest_b}
    _atomic_json(output / "run-manifest.json", run_manifest)
    sums = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return summary


compare = compare_replays
