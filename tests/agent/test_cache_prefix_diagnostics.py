from __future__ import annotations

import json
import multiprocessing
import os
from types import SimpleNamespace

from agent import cache_prefix_diagnostics as diagnostics


def request():
    return {
        "model": "model-secret",
        "instructions": "system secret",
        "input": [{"role": "user", "content": "old secret"}, {"role": "assistant", "content": "later secret"}],
        "tools": [{"type": "function", "name": "tool secret"}],
        "prompt_cache_key": "scope secret",
    }


def enable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_enabled", lambda: True)


def rows(tmp_path):
    return [json.loads(x) for x in (tmp_path / "cache" / "codex-cache-prefix.jsonl").read_text().splitlines()]


def _process_record(home):
    import os
    os.environ["HERMES_HOME"] = str(home)
    diagnostics._enabled = lambda: True
    token = diagnostics.begin_attempt(request(), session_id="process", turn_id="turn", api_id="api", ordinal=0, retry=0)
    diagnostics.finish_attempt(token)


def test_disabled_does_not_create_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_enabled", lambda: False)
    assert diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0) is None
    assert not (tmp_path / "cache").exists()


def test_terminal_row_is_after_consumption_and_redacted(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch)
    token = diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=2, retry=1, role="primary")
    assert token is not None
    diagnostics.finish_attempt(token, SimpleNamespace(usage=SimpleNamespace(input_tokens=10, input_tokens_details=SimpleNamespace(cached_tokens=7))))
    data = rows(tmp_path)
    assert len(data) == 1 and data[0]["kind"] == "terminal"
    assert data[0]["ordinal"] == 2 and data[0]["retry"] == 1
    assert data[0]["usage"] == {"cache_read": 7, "uncached_input": 3}
    assert [x["key"] for x in data[0]["components"]] == ["system", "tools", "scope", "history:0", "history:1"]
    text = (tmp_path / "cache" / "codex-cache-prefix.jsonl").read_text()
    for secret in ("system secret", "old secret", "later secret", "tool secret", "scope secret", "model-secret"):
        assert secret not in text
    assert '"session":"s"' not in text and '"turn":"t"' not in text


def test_exception_is_one_error_row_without_exception_text(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch)
    token = diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0)
    diagnostics.finish_attempt(token, error=ValueError("sentinel exception"))
    row = rows(tmp_path)[0]
    assert row["kind"] == "error" and "sentinel" not in json.dumps(row)


def test_usage_zero_and_unknown_are_distinct(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch)
    for usage in (SimpleNamespace(input_tokens=0, input_tokens_details=SimpleNamespace(cached_tokens=0)), None):
        token = diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0)
        diagnostics.finish_attempt(token, SimpleNamespace(usage=usage))
    assert rows(tmp_path)[0]["usage"] == {"cache_read": 0, "uncached_input": 0}
    assert rows(tmp_path)[1]["usage"] == {"cache_read": None, "uncached_input": None}


def test_component_change_changes_only_component_digest(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch)
    a = request(); b = request(); b["tools"][0]["name"] = "changed"
    for req in (a, b):
        token = diagnostics.begin_attempt(req, session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0)
        diagnostics.finish_attempt(token)
    first, second = rows(tmp_path)
    assert first["components"][1]["hmac"] != second["components"][1]["hmac"]
    assert first["components"][0] == second["components"][0]


def test_real_sdk_boundary_stream_terminal_and_error(tmp_path, monkeypatch):
    from agent.codex_runtime import run_codex_stream
    enable(tmp_path, monkeypatch)
    seen = []
    class Agent:
        provider = "openai-codex"
        session_id = "session raw sentinel"
        _current_api_request_id = "session raw sentinel:task:turn raw sentinel:api:2"
        _interrupt_requested = False
        _fallback_index = 0
        model = "model"
        def _is_codex_backend(self): return True
        def _fire_stream_delta(self, text): pass
        def _fire_reasoning_delta(self, text): pass
        def _touch_activity(self, text): pass
    agent = Agent()
    def create(**kwargs):
        seen.append(kwargs)
        assert len(rows(tmp_path)) == 0 if (tmp_path / "cache" / "codex-cache-prefix.jsonl").exists() else True
        return iter([{"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 4, "input_tokens_details": {"cached_tokens": 0}}}}])
    client = SimpleNamespace(responses=SimpleNamespace(create=create), base_url="https://chatgpt.com/backend-api/codex")
    result = run_codex_stream(agent, request(), client=client)
    assert result.status == "completed"
    assert seen and rows(tmp_path)[0]["ordinal"] == 2
    assert rows(tmp_path)[0]["usage"] == {"cache_read": 0, "uncached_input": 4}
    assert "raw sentinel" not in json.dumps(rows(tmp_path))


def test_unsafe_ancestor_output_key_and_permissions_fail_closed(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch)
    parent = tmp_path / "link"
    parent.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(parent))
    assert diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0) is None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    directory = tmp_path / "cache"
    directory.mkdir(mode=0o700)
    key = directory / "codex-cache-prefix.key"
    key.symlink_to(tmp_path / "victim")
    assert diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0) is None
    key.unlink()
    token = diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0)
    output = directory / "codex-cache-prefix.jsonl"
    output.symlink_to(tmp_path / "victim")
    diagnostics.finish_attempt(token)
    assert not (tmp_path / "victim").exists()
    output.unlink()
    key.chmod(0o644)
    assert diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0) is None


def test_insecure_existing_home_fails_closed_without_artifacts(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o777)
    enable(home, monkeypatch)
    assert diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0) is None
    assert not (home / "cache").exists()

    home.chmod(0o700)
    current_uid = os.getuid()
    monkeypatch.setattr(diagnostics.os, "getuid", lambda: current_uid + 1)
    assert diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0) is None
    assert not (home / "cache").exists()


def test_private_existing_home_allows_diagnostic_artifact(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    enable(home, monkeypatch)
    token = diagnostics.begin_attempt(request(), session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0)
    assert token is not None
    diagnostics.finish_attempt(token)
    assert len(rows(home)) == 1


def test_insecure_existing_home_does_not_disrupt_provider(tmp_path, monkeypatch):
    from agent.codex_runtime import run_codex_stream

    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o777)
    enable(home, monkeypatch)
    seen = []

    class Agent:
        provider = "openai-codex"
        session_id = "session"
        _current_api_request_id = "session:task:turn:api:0"
        _interrupt_requested = False
        _fallback_index = 0
        model = "model"

        def _is_codex_backend(self): return True
        def _fire_stream_delta(self, text): pass
        def _fire_reasoning_delta(self, text): pass
        def _touch_activity(self, text): pass

    def create(**kwargs):
        seen.append(kwargs)
        return iter([{"type": "response.completed", "response": {"status": "completed"}}])

    client = SimpleNamespace(responses=SimpleNamespace(create=create), base_url="https://chatgpt.com/backend-api/codex")
    result = run_codex_stream(Agent(), request(), client=client)
    assert result.status == "completed"
    assert seen
    assert not (home / "cache").exists()


def test_file_sequence_and_rotation_continue(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch)
    monkeypatch.setattr(diagnostics, "_MAX_BYTES", 1700)
    for number in range(5):
        token = diagnostics.begin_attempt(request(), session_id="session", turn_id=f"turn-{number}", api_id=f"api-{number}", ordinal=number, retry=0)
        diagnostics.finish_attempt(token)
    directory = tmp_path / "cache"
    older = [json.loads(line) for line in (directory / "codex-cache-prefix.jsonl.1").read_text().splitlines()]
    latest = rows(tmp_path)
    assert [r["sequence"] for r in older + latest] == sorted(r["sequence"] for r in older + latest)
    assert latest[-1]["sequence"] == 4


def test_multiprocess_append_is_lossless(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch)
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_process_record, args=(tmp_path,)) for _ in range(4)]
    for process in processes: process.start()
    for process in processes: process.join()
    assert all(process.exitcode == 0 for process in processes)
    assert len(rows(tmp_path)) == 4


def test_history_overflow_digest_distinguishes_same_count_suffix(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch)
    prefix = [{"role": "user", "content": f"history-{index}"} for index in range(diagnostics._MAX_HISTORY)]
    first = request()
    second = request()
    first["input"] = prefix + [{"role": "user", "content": "suffix-a"}]
    second["input"] = prefix + [{"role": "user", "content": "suffix-b"}]
    for req in (first, second):
        token = diagnostics.begin_attempt(req, session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0)
        diagnostics.finish_attempt(token)
    first_overflow = rows(tmp_path)[0]["components"][-1]
    second_overflow = rows(tmp_path)[1]["components"][-1]
    assert first_overflow["key"] == second_overflow["key"] == "history:overflow"
    assert first_overflow["hmac"] != second_overflow["hmac"]
    assert first_overflow["bytes"] == len(diagnostics._encoded(first["input"][diagnostics._MAX_HISTORY:]))
