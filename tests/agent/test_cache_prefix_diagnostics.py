from __future__ import annotations

import json
import multiprocessing
import os
import threading
from types import SimpleNamespace

import pytest

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


def test_managed_rewrite_digest_matches_sdk_kwargs(tmp_path, monkeypatch):
    from agent import relay_llm, relay_runtime
    from agent.codex_runtime import run_codex_stream

    enable(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="session-managed", platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="turn-1", task_id="task-1")
    lease.host.retain_managed_execution("test.cache_prefix")
    seen = []

    class Agent:
        provider = "openai-codex"
        session_id = "session-managed"
        _current_api_request_id = "turn-1:api:0"
        _interrupt_requested = False
        _fallback_index = 0
        model = "model"
        def _is_codex_backend(self): return True
        def _fire_stream_delta(self, text): pass
        def _fire_reasoning_delta(self, text): pass
        def _touch_activity(self, text): pass

    pre_wire = {}
    original_execute = lease.host.relay.llm.stream_execute

    async def rewrite_once(name, request, callback, *args, **kwargs):
        content = dict(request.content)
        pre_wire["instructions"] = content.get("instructions")
        pre_wire["prompt_cache_key"] = content.get("prompt_cache_key")
        content["instructions"] = "rewritten instructions"
        content["prompt_cache_key"] = "rewritten scope"
        from nemo_relay import LLMRequest
        rewritten = LLMRequest(getattr(request, "headers", {}) or {}, content)
        return await original_execute(name, rewritten, callback, *args, **kwargs)

    monkeypatch.setattr(lease.host.relay.llm, "stream_execute", rewrite_once)

    def create(**kwargs):
        seen.append(kwargs)
        return iter([{"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 2, "input_tokens_details": {"cached_tokens": 0}}}}])

    client = SimpleNamespace(responses=SimpleNamespace(create=create), base_url="https://chatgpt.com/backend-api/codex")
    try:
        result = run_codex_stream(Agent(), request(), client=client)
    finally:
        lease.host.release_managed_execution("test.cache_prefix")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()
    assert result is not None and result.status == "completed" and len(seen) == 1
    sent = seen[0]
    assert sent["instructions"] == "rewritten instructions"
    assert pre_wire["instructions"] == "system secret"
    key = (tmp_path / "cache" / "codex-cache-prefix.key").read_bytes()
    expected = diagnostics._components(sent, key)
    stale = diagnostics._components({**sent, "instructions": pre_wire["instructions"], "prompt_cache_key": pre_wire["prompt_cache_key"]}, key)
    recorded = {item["key"]: item for item in rows(tmp_path)[0]["components"]}
    assert recorded["system"]["hmac"] == expected[0]["hmac"]
    assert recorded["system"]["hmac"] != stale[0]["hmac"]
    assert recorded["scope"]["hmac"] == expected[2]["hmac"]
    text = (tmp_path / "cache" / "codex-cache-prefix.jsonl").read_text()
    assert "rewritten instructions" not in text and "system secret" not in text


def test_delayed_trailing_drain_error_is_recorded_after_iterator_error(tmp_path, monkeypatch):
    from agent.codex_runtime import run_codex_stream
    import httpx

    enable(tmp_path, monkeypatch)
    phases = []
    drain_started = threading.Event()
    drain_release = threading.Event()
    append = diagnostics._append

    def release_drain():
        if drain_started.wait(2):
            drain_release.set()

    release_thread = threading.Thread(target=release_drain, daemon=True)
    release_thread.start()

    def record_append(row):
        phases.append(f"diagnostic:{row['kind']}")
        return append(row)

    monkeypatch.setattr(diagnostics, "_append", record_append)

    class DelayedTrailingFailure:
        def __init__(self):
            self._terminal = True

        def __iter__(self):
            return self

        def __next__(self):
            if self._terminal:
                self._terminal = False
                phases.append("terminal")
                return {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": 4, "input_tokens_details": {"cached_tokens": 0}},
                    },
                }
            phases.append("drain_started")
            drain_started.set()
            assert drain_release.wait(2)
            phases.append("drain_error")
            raise httpx.ReadError("late trailing read")

    class Agent:
        provider = "openai-codex"
        session_id = "session"
        _current_api_request_id = "session:task:turn:api:2"
        _interrupt_requested = False
        _fallback_index = 0
        model = "model"

        def _is_codex_backend(self): return True
        def _fire_stream_delta(self, text): pass
        def _fire_reasoning_delta(self, text): pass
        def _touch_activity(self, text): pass
        def _client_log_context(self): return "test"

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return DelayedTrailingFailure()

    client = SimpleNamespace(responses=SimpleNamespace(create=create), base_url="https://chatgpt.com/backend-api/codex")
    result = run_codex_stream(Agent(), request(), client=client)
    release_thread.join(2)

    assert result is not None
    assert result.status == "completed"
    assert len(calls) == 1
    assert phases == ["terminal", "drain_started", "drain_error", "diagnostic:error"]
    row = rows(tmp_path)[0]
    assert row["kind"] == "error"
    assert row["usage"] == {"cache_read": None, "uncached_input": None}
    assert "late trailing read" not in json.dumps(row)


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


def test_codex_opener_counts_a_finished_create_not_its_start(tmp_path, monkeypatch):
    from agent import relay_llm
    from agent.codex_runtime import run_codex_stream
    import httpx

    enable(tmp_path, monkeypatch)
    contexts = []
    real_send = relay_llm.physical_send

    def capture(request, callback):
        def wrapped(final):
            context = relay_llm._PHYSICAL_DIAGNOSTICS.get()
            if context is not None and context not in contexts:
                contexts.append(context)
            return callback(final)

        return real_send(request, wrapped)

    monkeypatch.setattr(relay_llm, "physical_send", capture)

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
        def _client_log_context(self): return "test"

    def refusing(**kwargs):
        raise TypeError("sdk build rejected the request")

    client = SimpleNamespace(responses=SimpleNamespace(create=refusing), base_url="https://chatgpt.com/backend-api/codex")
    with pytest.raises(TypeError, match="sdk build rejected"):
        run_codex_stream(Agent(), request(), client=client)
    assert [context.ordinals for context in contexts] == [[0]]
    row = rows(tmp_path)[0]
    assert row["kind"] == "error" and "sdk build rejected" not in json.dumps(row)

    contexts.clear()

    class Wrote(httpx.WriteError):
        pass

    def wrote(**kwargs):
        request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
        raise Wrote("body reached the transport", request=request)

    client = SimpleNamespace(responses=SimpleNamespace(create=wrote), base_url="https://chatgpt.com/backend-api/codex")
    with pytest.raises(httpx.WriteError):
        run_codex_stream(Agent(), request(), client=client)
    assert [context.ordinals for context in contexts] == [[0]]
    row = rows(tmp_path)[1]
    assert row["kind"] == "error" and "body reached" not in json.dumps(row)

    contexts.clear()
    seen = []

    def opened(**kwargs):
        seen.append(list(contexts[-1].ordinals))
        return iter([{"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}}}}])

    client = SimpleNamespace(responses=SimpleNamespace(create=opened), base_url="https://chatgpt.com/backend-api/codex")
    result = run_codex_stream(Agent(), request(), client=client)
    assert result.status == "completed" and seen == [[]]
    assert [context.ordinals for context in contexts] == [[0]]


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
