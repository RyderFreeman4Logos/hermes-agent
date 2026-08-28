"""Read-only export of durable compression traces.

The exporter deliberately uses ``SessionDB(read_only=True)`` and writes only to
an explicitly supplied corpus directory.  Historical sessions without a
committed trace are reported as incomplete rather than being silently promoted
to exact data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


_SCHEMA_VERSION = "compression-trace-export-v1"
_SECRET_KEY = re.compile(r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|cookie|password|secret|credential)", re.I)
_SECRET_VALUE = re.compile(
    r"(?i)\b(?:bearer\s+|sk-|gh[opsu]_)[A-Za-z0-9_./+=:-]{12,}"
    r"|\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*[^\s,;]+"
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _redact(value: Any, *, enabled: bool = True) -> Any:
    if not enabled:
        return value
    if isinstance(value, Mapping):
        return {str(k): "[REDACTED]" if _SECRET_KEY.search(str(k)) else _redact(v, enabled=True) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, enabled=True) for v in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    return value


def _write(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(body)
    os.replace(temporary, path)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return hashlib.sha256(body).hexdigest()


def _db_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            digest.update(candidate.name.encode())
            digest.update(candidate.read_bytes())
    return digest.hexdigest()


def _rows_jsonl(rows: Iterable[Mapping[str, Any]], *, redact: bool) -> bytes:
    return b"".join(_json_bytes(_redact(dict(row), enabled=redact)) for row in rows)


def _artifact_body(db: Any, artifact_id: Optional[str]) -> Optional[bytes]:
    if not artifact_id:
        return None
    body = db.get_checkpoint_artifact(artifact_id)
    if body is None:
        return None
    if hashlib.sha256(body).hexdigest() != artifact_id:
        return None
    return body


def _session_events(db: Any, session_id: str) -> List[Dict[str, Any]]:
    return db.get_messages(session_id, include_inactive=True)


def _export_artifact_body(body: bytes, *, redact: bool) -> bytes:
    if not redact:
        return body
    try:
        return _json_bytes(_redact(json.loads(body.decode("utf-8")), enabled=True))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _SECRET_VALUE.sub("[REDACTED]", body.decode("utf-8", "replace")).encode("utf-8")


def _artifact_export_path(artifact_id: str, body: bytes, *, redact: bool) -> Path:
    return Path("artifacts") / (f"{artifact_id}.redacted" if redact and _export_artifact_body(body, redact=True) != body else artifact_id)


def _compression_run_evidence(
    db: Any, run: Mapping[str, Any], event_by_id: Mapping[int, Mapping[str, Any]]
) -> tuple:
    source_ids = [int(i) for i in run.get("source_event_ids") or [] if isinstance(i, int) and i > 0]
    return (
        source_ids,
        [event_by_id[i] for i in source_ids if i in event_by_id],
        _artifact_body(db, run.get("config_artifact_id")),
        _artifact_body(db, run.get("pre_projection_artifact_id")),
        _artifact_body(db, run.get("post_projection_artifact_id")),
    )


def _compression_run_is_exact(run: Mapping[str, Any], evidence: tuple) -> bool:
    source_ids, raw_events, config_body, pre_body, post_body = evidence
    return bool(
        source_ids
        and len(raw_events) == len(source_ids)
        and config_body
        and pre_body
        and post_body
        and run.get("status") == "committed"
    )


def discover_compression_sessions(
    db_path: Path, *, min_compressions: int = 0, sort: str = "compression-count-desc"
) -> List[Dict[str, Any]]:
    """List sessions and trace-derived facts without opening the DB writable."""
    from hermes_state import SessionDB

    db_path = Path(db_path)
    db = SessionDB(db_path, read_only=True)
    try:
        with db._read_ctx() as conn:
            rows = conn.execute(
                """SELECT s.*, COUNT(cr.run_id) AS compression_count,
                          MAX(cr.run_id) AS last_compression_run
                     FROM sessions s LEFT JOIN compression_runs cr ON cr.session_id = s.id
                    GROUP BY s.id ORDER BY s.started_at DESC"""
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["compression_count"] = int(item.get("compression_count") or 0)
            if item["compression_count"] < min_compressions:
                continue
            if item["compression_count"]:
                events = _session_events(db, str(item["id"]))
                event_by_id = {
                    int(event["id"]): event
                    for event in events
                    if event.get("id") is not None
                }
                runs = db.list_compression_runs(str(item["id"]))
                item["trace_classification"] = (
                    "exact"
                    if all(
                        _compression_run_is_exact(run, _compression_run_evidence(db, run, event_by_id))
                        for run in runs
                    )
                    else "partial"
                )
            else:
                item["trace_classification"] = "partial" if item.get("message_count") else "unusable"
            result.append(item)
        if sort == "compression-count-desc":
            result.sort(key=lambda item: (-item["compression_count"], -(item.get("started_at") or 0)))
        elif sort == "started-at":
            result.sort(key=lambda item: item.get("started_at") or 0)
        return result
    finally:
        db.close()


def inspect_compression_session(db_path: Path, session_id: str) -> Dict[str, Any]:
    """Return a JSON-safe inspection record for one session or raise ``ValueError``."""
    from hermes_state import SessionDB

    db = SessionDB(Path(db_path), read_only=True)
    try:
        resolved = db.resolve_session_id(session_id)
        session = db.get_session(resolved) if resolved else None
        if not session:
            raise ValueError(f"session not found: {session_id}")
        runs = db.list_compression_runs(resolved)
        events = _session_events(db, resolved)
        return {
            "session": session,
            "compression_count": len(runs),
            "compressions": runs,
            "event_count": len(events),
            "tool_call_count": sum(1 for event in events if event.get("role") == "tool" or event.get("tool_calls")),
            "has_failed_verification": any(str(event.get("finish_reason") or "").lower() in {"error", "failed"} or str(event.get("effect_disposition") or "").lower() == "failed" for event in events),
            "has_chinese_messages": any(re.search(r"[\u3400-\u9fff]", str(event.get("content") or "")) for event in events),
        }
    finally:
        db.close()


def export_compression_trace(
    db_path: Path,
    session_id: str,
    output: Path,
    *,
    redaction: str = "default",
) -> Dict[str, Any]:
    """Export all committed boundaries for *session_id* into *output*.

    ``redaction='none'`` is explicit and local-only; the function never uploads
    data or invokes a model/tool executor.
    """
    if redaction not in {"default", "none"}:
        raise ValueError("redaction must be 'default' or 'none'")
    db_path, output = Path(db_path).resolve(), Path(output).resolve()
    if output == db_path or db_path in output.parents:
        raise ValueError("export output must not be the source database or its directory")
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(stat.S_IRWXU)
    redact = redaction != "none"

    from hermes_state import SessionDB

    source_hash = _db_hash(db_path)
    db = SessionDB(db_path, read_only=True)
    files: Dict[str, str] = {}
    try:
        resolved = db.resolve_session_id(session_id)
        session = db.get_session(resolved) if resolved else None
        if not session:
            raise ValueError(f"session not found: {session_id}")
        runs = db.list_compression_runs(resolved)
        all_events = _session_events(db, resolved)
        if not runs and all_events:
            # ponytail: one conservative projection when no durable boundary exists;
            # retain events rather than inventing a generation split.
            reconstructed_class = (
                "exact-events-projection-reconstructed"
                if any(event.get("compacted") or event.get("_compressed_summary") for event in all_events)
                else "partial"
            )
            runs = [{
                "run_id": "reconstructed-1",
                "session_id": resolved,
                "source_event_ids": [int(event["id"]) for event in all_events if event.get("id") is not None],
                "config_artifact_id": None,
                "pre_projection_artifact_id": None,
                "post_projection_artifact_id": None,
                "status": "reconstructed",
                "boundary_kind": "reconstructed",
                "continuation_session_id": resolved,
                "_reconstructed_classification": reconstructed_class,
            }]
        event_by_id = {int(event["id"]): event for event in all_events if event.get("id") is not None}
        classification = "exact"
        points = []
        for generation, run in enumerate(runs, 1):
            source_ids, raw_events, config_body, pre_body, post_body = _compression_run_evidence(
                db, run, event_by_id
            )
            point_class = run.get("_reconstructed_classification") or (
                "exact" if _compression_run_is_exact(
                    run, (source_ids, raw_events, config_body, pre_body, post_body)
                ) else "partial"
            )
            if point_class == "partial":
                classification = "partial"
            elif point_class == "exact-events-projection-reconstructed" and classification == "exact":
                classification = point_class
            point_dir = output / "points" / str(generation)
            point_dir.mkdir(parents=True, exist_ok=True)
            point_dir.chmod(stat.S_IRWXU)

            try:
                config = json.loads(config_body.decode("utf-8")) if config_body else {}
            except (ValueError, UnicodeDecodeError):
                config = {}
            try:
                pre_context = json.loads(pre_body.decode("utf-8")) if pre_body else {"messages": raw_events}
            except (ValueError, UnicodeDecodeError):
                pre_context = {"messages": raw_events}
            try:
                post_context = json.loads(post_body.decode("utf-8")) if post_body else {}
            except (ValueError, UnicodeDecodeError):
                post_context = {}

            continuation_id = run.get("continuation_session_id") or resolved
            continuation_events = _session_events(db, continuation_id)
            if continuation_id == resolved and source_ids:
                continuation_events = [event for event in continuation_events if int(event.get("id", 0)) > max(source_ids)]
            metadata = {
                "schema_version": _SCHEMA_VERSION,
                "trace_classification": point_class,
                "session_id": resolved,
                "compression_id": f"run-{run['run_id']}",
                "generation": generation,
                "source_revision": run.get("run_id"),
                "source_event_ids": source_ids,
                "source_event_hash": hashlib.sha256(_rows_jsonl(raw_events, redact=False)).hexdigest(),
                "engine": config.get("engine", "unknown") if isinstance(config, dict) else "unknown",
                "engine_version": config.get("engine_version") if isinstance(config, dict) else None,
                "route": {key: config.get(key) for key in ("provider", "model", "physical_model", "reasoning_effort", "fallback_policy", "fallback_chain") if isinstance(config, dict) and key in config},
                "inference_parameters": config.get("inference_parameters", {}) if isinstance(config, dict) else {},
                "status": run.get("status"),
                "boundary_kind": run.get("boundary_kind"),
                "continuation_session_id": continuation_id,
            }
            point_files = {
                "pre-context.json": _json_bytes(_redact(pre_context, enabled=redact)),
                "post-context.json": _json_bytes(_redact(post_context, enabled=redact)),
                "raw-events.jsonl": _rows_jsonl(raw_events, redact=redact),
                "continuation.jsonl": _rows_jsonl(continuation_events, redact=redact),
                "metadata.json": _json_bytes(_redact(metadata, enabled=redact)),
                "README.md": (f"# Compression generation {generation}\n\nClassification: `{point_class}`.\n\nSource events: {len(source_ids)}.\n").encode(),
            }
            point_hashes = {}
            for name, body in point_files.items():
                relative = Path("points") / str(generation) / name
                files[relative.as_posix()] = _write(output / relative, body)
                point_hashes[name] = files[relative.as_posix()]
            for body in (config_body, pre_body, post_body):
                if body:
                    artifact_id = hashlib.sha256(body).hexdigest()
                    relative = _artifact_export_path(artifact_id, body, redact=redact)
                    if relative.as_posix() not in files:
                        files[relative.as_posix()] = _write(output / relative, _export_artifact_body(body, redact=redact))
            points.append({**metadata, "files": point_hashes})

        if not runs:
            classification = "unusable"
        compression_lines = b"".join(_json_bytes(_redact(point, enabled=redact)) for point in points)
        files["compression-points.jsonl"] = _write(output / "compression-points.jsonl", compression_lines)
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "session_id": resolved,
            "source_db": str(db_path),
            "source_content_hash": source_hash,
            "trace_classification": classification,
            "redaction": "default" if redact else "none",
            "created_at": time.time(),
            "compression_count": len(points),
            "files": dict(sorted(files.items())),
        }
        manifest_body = _json_bytes(manifest)
        _write(output / "manifest.json", manifest_body)
        return manifest
    finally:
        db.close()


# Short aliases keep the module pleasant for callers and CLI adapters.
discover_sessions = discover_compression_sessions
inspect_session = inspect_compression_session
inspect_trace = inspect_compression_session
list_compression_sessions = discover_compression_sessions
export_trace = export_compression_trace
export_session = export_compression_trace
