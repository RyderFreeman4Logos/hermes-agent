from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace

from agent import cache_prefix_diagnostics as diagnostics


def _request():
    return {
        "model": "model",
        "instructions": "system secret",
        "input": [{"role": "user", "content": "prompt secret"}],
        "tools": [{"type": "function", "name": "tool secret"}],
        "prompt_cache_key": "scope secret",
    }


def test_disabled_does_not_create_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_enabled", lambda: False)
    diagnostics.record_attempt_start(_request(), correlation="raw-id", attempt=0)
    assert not (tmp_path / "cache").exists()


def test_records_only_digests_lengths_and_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_enabled", lambda: True)
    token = diagnostics.record_attempt_start(_request(), correlation="raw-id", attempt=0)
    diagnostics.record_attempt_terminal(token, SimpleNamespace(usage=SimpleNamespace(input_tokens_details=SimpleNamespace(cached_tokens=7))))
    path = tmp_path / "cache" / "codex-cache-prefix.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    text = path.read_text()
    for sentinel in ("system secret", "prompt secret", "tool secret", "scope secret", "raw-id"):
        assert sentinel not in text
    assert rows[0]["kind"] == "start"
    assert rows[0]["segments"]["messages"]["bytes"] > 0
    assert rows[1]["kind"] == "terminal"
    assert rows[1]["cache_tokens"] == 7


def test_symlink_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_enabled", lambda: True)
    (tmp_path / "cache").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    diagnostics.record_attempt_start(_request(), correlation="id", attempt=0)
    assert not (tmp_path / "elsewhere").exists()


def test_concurrent_records_are_complete_json_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_enabled", lambda: True)
    threads = [threading.Thread(target=diagnostics.record_attempt_start, args=(_request(),), kwargs={"correlation": str(i), "attempt": i}) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    path = tmp_path / "cache" / "codex-cache-prefix.jsonl"
    assert len(path.read_text().splitlines()) == 8
    for line in path.read_text().splitlines():
        json.loads(line)


def test_real_final_kwargs_boundary(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_enabled", lambda: True)
    seen = {}
    request = _request()

    def create(**kwargs):
        seen.update(kwargs)
        return object()

    diagnostics.call_final_codex_create(create, request, correlation="id", attempt=1)
    assert seen == request
    assert "input" not in (tmp_path / "cache" / "codex-cache-prefix.jsonl").read_text()
