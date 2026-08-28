import hashlib
import json
import subprocess

import pytest

from hermes_state import SessionDB
from hermes_cli.compression_trace import discover_compression_sessions, export_compression_trace
from hermes_cli.compression_replay import (
    ReplayConfig,
    ReplaySafetyError,
    ReplayRunner,
)


def _source(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("source-session", "cli", model="stored-model")
    ids = [
        db.append_message("source-session", "user", "Keep the branch and run tests."),
        db.append_message("source-session", "assistant", "I will inspect it."),
        db.append_message("source-session", "tool", "test failed", tool_name="terminal", effect_disposition="failed"),
    ]
    run_id = db.store_compression_run(
        "source-session",
        ids,
        {"engine": "lean", "provider": "stored-provider", "reasoning_effort": "low"},
        {"messages": [{"role": "user", "content": "Keep the branch and run tests."}]},
        {"messages": [{"role": "user", "content": "Keep the branch and run tests."}, {"role": "assistant", "content": "I will inspect it."}]},
    )
    db.complete_compression_run(run_id, "source-session", "in_place")
    db.close()
    return path


def test_discovery_only_labels_committed_evidenced_traces_exact(tmp_path):
    source = _source(tmp_path)
    db = SessionDB(source)
    db.create_session("prepared-session", "cli", model="stored-model")
    prepared_id = db.append_message("prepared-session", "user", "Prepared only.")
    db.store_compression_run(
        "prepared-session",
        [prepared_id],
        {"engine": "lean"},
        {"messages": []},
        {"messages": []},
    )

    db.create_session("partial-session", "cli", model="stored-model")
    partial_run_id = db.store_compression_run(
        "partial-session",
        [999999],
        {"engine": "lean"},
        {"messages": []},
        {"messages": []},
    )
    db.complete_compression_run(partial_run_id, "partial-session", "in_place")

    db.create_session("reconstructed-session", "cli", model="stored-model")
    db.append_message("reconstructed-session", "assistant", "Reconstructed.", _compressed_summary=True)
    db.close()

    discovered = {
        item["id"]: item
        for item in discover_compression_sessions(source)
    }

    assert discovered["source-session"]["trace_classification"] == "exact"
    assert discovered["prepared-session"]["trace_classification"] != "exact"
    assert discovered["partial-session"]["trace_classification"] != "exact"
    assert discovered["reconstructed-session"]["trace_classification"] != "exact"


def test_export_manifest_hash_round_trip_and_source_is_read_only(tmp_path):
    source = _source(tmp_path)
    before = source.read_bytes()
    output = tmp_path / "corpus"

    manifest = export_compression_trace(source, "source-session", output)

    assert manifest["trace_classification"] == "exact"
    assert (output / "manifest.json").exists()
    point = output / "points" / "1"
    for name in ("pre-context.json", "post-context.json", "raw-events.jsonl", "continuation.jsonl", "metadata.json", "README.md"):
        assert (point / name).exists()
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected
    assert source.read_bytes() == before


def test_replay_propagates_route_and_reasoning_and_labels_policy(tmp_path):
    source = _source(tmp_path)
    corpus = tmp_path / "corpus"
    export_compression_trace(source, "source-session", corpus)
    seen = []

    def infer(request):
        seen.append(request)
        return {"messages": request["pre_context"]["messages"]}

    runner = ReplayRunner(
        corpus,
        ReplayConfig(provider="custom:qwen-local", model="Qwen3.8-27B-NVFP4", reasoning_effort="none", fallback_policy="strict-single-route"),
        infer=infer,
    )
    result = runner.run(mode="A")
    assert result["policy_label"] == "strict-single-route"
    assert result["route"]["provider"] == "custom:qwen-local"
    assert result["route"]["model"] == "Qwen3.8-27B-NVFP4"
    assert result["route"]["reasoning_effort"] == "none"
    assert seen[0]["route"] == result["route"]

    fallback = ReplayRunner(corpus, ReplayConfig(fallback_policy="configured-only"), infer=infer).run(mode="A")
    assert fallback["policy_label"] == "configured-fallback"


def test_mode_c_rejects_real_tool_execution(tmp_path):
    source = _source(tmp_path)
    corpus = tmp_path / "corpus"
    export_compression_trace(source, "source-session", corpus)
    runner = ReplayRunner(corpus, ReplayConfig(execute_real_tools=True), infer=lambda request: {"tool_calls": [{"name": "terminal"}]})
    with pytest.raises(ReplaySafetyError):
        runner.run(mode="C")


def test_replay_defaults_are_safe():
    config = ReplayConfig()
    assert config.read_only is True
    assert config.max_concurrency == 2
    assert config.execute_real_tools is False


def test_export_and_replay_leave_source_and_worktree_unchanged(tmp_path):
    source = _source(tmp_path)
    corpus = tmp_path / "corpus"
    before_db = source.read_bytes()
    before_status = subprocess.check_output(["git", "status", "--porcelain"], text=True)

    export_compression_trace(source, "source-session", corpus)
    ReplayRunner(corpus).run(mode="A")

    assert source.read_bytes() == before_db
    assert subprocess.check_output(["git", "status", "--porcelain"], text=True) == before_status
