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


def _sdk_sse_client(create):
    """Exercise the production SDK and httpx hook; only the network is replaced."""
    import httpx

    class Events(httpx.SyncByteStream):
        def __init__(self, events):
            self.events = events

        def __iter__(self):
            for event in self.events:
                yield ("data: " + json.dumps(event) + "\n\n").encode()

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=Events(create(**json.loads(request.content))))

    return _real_codex_client(httpx.MockTransport(handler), max_retries=0)


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
    client = _sdk_sse_client(create)
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

    client = _sdk_sse_client(create)
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

    def record_append(row, **kwargs):
        phases.append(f"diagnostic:{row['kind']}")
        return append(row, **kwargs)

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

    client = _sdk_sse_client(create)
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

    client = _sdk_sse_client(create)
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
    # A busy nonblocking lock skips that writer instead of waiting.
    assert 1 <= len(rows(tmp_path)) <= 4


def test_unsupported_opener_never_invents_dispatch_evidence(tmp_path, monkeypatch):
    from agent import relay_llm
    from agent.codex_runtime import run_codex_stream
    import httpx

    enable(tmp_path, monkeypatch)
    contexts = []
    real_scope = relay_llm._diagnostic_scope

    def capture(context):
        contexts.append(context)
        return real_scope(context)

    monkeypatch.setattr(relay_llm, "_diagnostic_scope", capture)

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
    assert [context.ordinals for context in contexts] == [[]]
    assert not (tmp_path / "cache" / "codex-cache-prefix.jsonl").exists()

    contexts.clear()

    class Wrote(httpx.WriteError):
        pass

    def wrote(**kwargs):
        request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
        raise Wrote("body reached the transport", request=request)

    client = SimpleNamespace(responses=SimpleNamespace(create=wrote), base_url="https://chatgpt.com/backend-api/codex")
    with pytest.raises(httpx.WriteError):
        run_codex_stream(Agent(), request(), client=client)
    assert [context.ordinals for context in contexts] == [[]]
    assert not (tmp_path / "cache" / "codex-cache-prefix.jsonl").exists()

    contexts.clear()
    seen = []

    def opened(**kwargs):
        seen.append(list(contexts[-1].ordinals))
        return iter([{"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}}}}])

    client = SimpleNamespace(responses=SimpleNamespace(create=opened), base_url="https://chatgpt.com/backend-api/codex")
    result = run_codex_stream(Agent(), request(), client=client)
    assert result.status == "completed" and seen == [[]]
    assert [context.ordinals for context in contexts] == [[]]


def _codex_agent():
    class Agent:
        provider = "openai-codex"
        session_id = "session"
        _current_api_request_id = "session:task:turn:api:0"
        _interrupt_requested = False
        _fallback_index = 0
        model = "model"

        def _is_codex_backend(self):
            return True

        def _fire_stream_delta(self, text):
            pass

        def _fire_reasoning_delta(self, text):
            pass

        def _touch_activity(self, text):
            pass

        def _client_log_context(self):
            return "test"

        def _buffer_diagnostic_status(self, text):
            pass

    return Agent()


def _real_codex_client(transport, *, max_retries):
    import httpx
    from openai import OpenAI

    http_client = httpx.Client(transport=transport)
    return OpenAI(
        api_key="test-key",
        base_url="https://chatgpt.com/backend-api/codex",
        http_client=http_client,
        max_retries=max_retries,
    )


def test_real_client_prebuild_is_not_a_dispatch(tmp_path, monkeypatch):
    """A TypeError before the SDK builds a request never reaches the httpx request hook."""
    from agent import relay_llm
    from agent.codex_runtime import run_codex_stream
    import httpx

    enable(tmp_path, monkeypatch)
    contexts = []
    real_scope = relay_llm._diagnostic_scope

    def capture(context):
        contexts.append(context)
        return real_scope(context)

    monkeypatch.setattr(relay_llm, "_diagnostic_scope", capture)
    dispatches = []

    def handler(request):
        dispatches.append(request)
        return httpx.Response(200, json={"id": "resp", "output": [], "status": "completed"})

    client = _real_codex_client(httpx.MockTransport(handler), max_retries=0)
    with pytest.raises(TypeError):
        run_codex_stream(_codex_agent(), {"model": "model", "input": object()}, client=client)
    assert dispatches == []
    assert [context.ordinals for context in contexts] == [[]]
    assert not (tmp_path / "cache").exists()


def test_real_client_write_error_is_one_dispatch(tmp_path, monkeypatch):
    """A WriteError after the request hook is one dispatch, not a later SDK return."""
    from agent import relay_llm
    from agent.codex_runtime import run_codex_stream
    import httpx

    enable(tmp_path, monkeypatch)
    contexts = []
    real_scope = relay_llm._diagnostic_scope

    def capture(context):
        contexts.append(context)
        return real_scope(context)

    monkeypatch.setattr(relay_llm, "_diagnostic_scope", capture)
    dispatches = []

    def handler(request):
        dispatches.append(1)
        raise httpx.WriteError("body reached the transport", request=request)

    from openai import APIConnectionError

    client = _real_codex_client(httpx.MockTransport(handler), max_retries=0)
    try:
        with pytest.raises(APIConnectionError) as raised:
            run_codex_stream(_codex_agent(), request(), client=client)
        assert isinstance(raised.value.__cause__, httpx.WriteError)
        assert dispatches == [1]
        assert [context.ordinals for context in contexts] == [[0]]
        assert client._client.event_hooks["request"] == []
        data = rows(tmp_path)
        assert len(data) == 1 and data[0]["kind"] == "error"
        assert data[0]["usage"]["cache_read"] is None
        text = (tmp_path / "cache" / "codex-cache-prefix.jsonl").read_text()
        assert "body reached" not in text
        assert "chatgpt.com" not in text
    finally:
        client.close()


def test_real_client_sdk_retries_are_distinct_dispatches(tmp_path, monkeypatch):
    """Default SDK retries are separate hook fires inside one responses.create."""
    from agent import relay_llm
    from agent.codex_runtime import run_codex_stream
    import httpx

    enable(tmp_path, monkeypatch)
    contexts = []
    real_scope = relay_llm._diagnostic_scope

    def capture(context):
        contexts.append(context)
        return real_scope(context)

    monkeypatch.setattr(relay_llm, "_diagnostic_scope", capture)
    dispatches = []

    def handler(request):
        dispatches.append(1)
        if len(dispatches) < 3:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"type":"response.completed","response":{"id":"resp","status":"completed",'
                b'"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0}}}}\n\n'
            ),
        )

    client = _real_codex_client(httpx.MockTransport(handler), max_retries=2)
    result = run_codex_stream(_codex_agent(), request(), client=client)
    assert result.status == "completed"
    assert dispatches == [1, 1, 1]
    assert [context.ordinals for context in contexts] == [[0, 1, 2]]
    data = rows(tmp_path)
    assert [row["dispatch_ordinal"] for row in data] == [0, 1, 2]
    assert [row["kind"] for row in data] == ["error", "error", "terminal"]
    assert all(row["usage"]["cache_read"] is None for row in data[:2])


def test_cancelled_await_and_unfinished_future_are_not_dispatches():
    """Cancel after the await starts, and a returned unfinished coroutine, never reach httpx."""
    import asyncio
    from agent import relay_llm

    context = relay_llm._PhysicalDiagnosticContext("provider", "model", {"api_mode": "codex_responses"})

    async def cancelled(_request):
        raise asyncio.CancelledError

    async def unfinished(_request):
        raise AssertionError("a returned coroutine must not run")

    with relay_llm._diagnostic_scope(context):
        pending = relay_llm.physical_send({"n": 1}, cancelled)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(pending)
        assert context.ordinals == []
        returned = relay_llm.physical_send({"n": 2}, unfinished)
        assert context.ordinals == []
        returned.close()


def test_dispatch_context_does_not_cross_calls():
    """A hook armed for one client must not count a later call on another client."""
    import httpx
    from agent import relay_llm

    first_hits = []
    second_hits = []

    def first_handler(request):
        first_hits.append(1)
        return httpx.Response(200, json={"ok": True})

    def second_handler(request):
        second_hits.append(1)
        return httpx.Response(200, json={"ok": True})

    first = httpx.Client(transport=httpx.MockTransport(first_handler))
    second = httpx.Client(transport=httpx.MockTransport(second_handler))
    context = relay_llm._PhysicalDiagnosticContext("provider", "model", {"api_mode": "chat_completions"})

    def send(client):
        def callback(_request, _client=client):
            return _client.request("POST", "https://example.invalid/v1")
        return callback

    with relay_llm._diagnostic_scope(context):
        relay_llm.physical_send({"n": 1}, send(first))
        assert context.ordinals == [0]
        relay_llm.physical_send({"n": 2}, send(second))
    assert context.ordinals == [0, 1]
    assert first_hits == [1] and second_hits == [1]
    first.close()
    second.close()


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

@pytest.mark.parametrize("enabled", [False, True])
def test_transport_stage_digest_and_default_off(tmp_path, monkeypatch, enabled):
    import httpx
    from agent.codex_runtime import run_codex_stream

    enable(tmp_path, monkeypatch)
    monkeypatch.setattr(diagnostics, "_enabled", lambda: enabled)
    sent = []
    sentinel = "transport-stage-private-sentinel"

    class Rewrite(httpx.Auth):
        def auth_flow(self, built):
            body = json.loads(built.content)
            body["instructions"] = sentinel
            yield httpx.Request(built.method, built.url, json=body)

    def handler(built):
        sent.append(json.loads(built.content))
        return httpx.Response(200, headers={"content-type":"text/event-stream"},
                             content=b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n')

    client = _real_codex_client(httpx.MockTransport(handler), max_retries=0)
    client._client.auth = Rewrite()
    try:
        assert run_codex_stream(_codex_agent(), request(), client=client).status == "completed"
        assert len(sent) == 1
        assert client._client.event_hooks["request"] == []
        if not enabled:
            assert not (tmp_path / "cache").exists()
            return
        data = rows(tmp_path)
        assert len(data) == 1 and data[0]["dispatch_ordinal"] == 0
        key = (tmp_path / "cache" / "codex-cache-prefix.key").read_bytes()
        assert data[0]["components"] == diagnostics._components(sent[0], key)
        assert sentinel not in json.dumps(data)
        assert all(len(x["hmac"]) == 64 and isinstance(x["bytes"], int) for x in data[0]["components"])
        for name in ("codex-cache-prefix.key", "codex-cache-prefix.jsonl"):
            assert (tmp_path / "cache" / name).stat().st_mode & 0o777 == 0o600
    finally:
        client.close()


def test_real_async_shared_client_dispatch_contexts(monkeypatch):
    import asyncio
    import httpx
    from openai import AsyncOpenAI
    from agent import relay_llm

    contexts = []
    scope = relay_llm._diagnostic_scope

    def capture(context):
        if all(existing is not context for existing in contexts):
            contexts.append(context)
        return scope(context)

    monkeypatch.setattr(relay_llm, "_diagnostic_scope", capture)

    async def run():
        entered = 0
        together = asyncio.Event()

        async def handler(built):
            nonlocal entered
            entered += 1
            if entered == 2:
                together.set()
            await asyncio.wait_for(together.wait(), 2)
            return httpx.Response(200, json={"id":"fixture", "object":"chat.completion", "created":0,
                "model":"model", "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = AsyncOpenAI(api_key="fixture", base_url="https://example.invalid/v1", http_client=http, max_retries=0)

            async def callback(request):
                return await client.chat.completions.create(**request)

            results = await asyncio.gather(*(relay_llm.execute_async(
                {"model":"model", "messages":[{"role":"user","content":"fixture"}]}, callback,
                name="provider", model_name="model", metadata={"api_mode":"chat_completions"},
            ) for _ in range(2)))
            assert len(results) == entered == 2
            assert http.event_hooks["request"] == []

    asyncio.run(run())
    assert [context.ordinals for context in contexts] == [[0], [0]]


@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses"])
def test_public_auxiliary_call_observes_actual_dispatch(tmp_path, monkeypatch, api_mode):
    import httpx
    from agent import auxiliary_client, relay_llm

    enable(tmp_path, monkeypatch)
    contexts, sent = [], []
    scope = relay_llm._diagnostic_scope

    def capture(context):
        if all(existing is not context for existing in contexts):
            contexts.append(context)
        return scope(context)

    monkeypatch.setattr(relay_llm, "_diagnostic_scope", capture)

    def handler(built):
        sent.append(json.loads(built.content))
        if api_mode == "chat_completions":
            return httpx.Response(200, json={"id":"fixture", "object":"chat.completion", "created":0,
                "model":"model", "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})
        item = {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}]}
        events = [
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": {
                "status": "completed", "output": [item],
                "usage": {"input_tokens": 12, "output_tokens": 1, "total_tokens": 13,
                          "input_tokens_details": {"cached_tokens": 8}},
            }},
        ]
        return httpx.Response(200, headers={"content-type":"text/event-stream"},
                              content="".join("data: " + json.dumps(event) + "\n\n" for event in events).encode())

    client = _real_codex_client(httpx.MockTransport(handler), max_retries=0)
    # Replace network construction only; resolution, conversion, aux hooks and Relay run for real.
    monkeypatch.setattr(auxiliary_client, "_create_openai_client", lambda **kwargs: client)
    try:
        result = auxiliary_client.call_llm(task="fixture", provider="custom", model="model",
            base_url="https://example.invalid/v1", api_key="fixture", api_mode=api_mode,
            messages=[{"role":"user", "content":"auxiliary-private-sentinel"}], timeout=5)
        assert result.choices[0].message.content == "ok"
        assert len(sent) == 1
        assert [context.ordinals for context in contexts] == [[0]]
        assert client._client.event_hooks["request"] == []
        if api_mode == "codex_responses":
            data = rows(tmp_path)
            assert len(data) == 1 and data[0]["kind"] == "terminal"
            key = (tmp_path / "cache" / "codex-cache-prefix.key").read_bytes()
            assert data[0]["components"] == diagnostics._components(sent[0], key)
            assert "auxiliary-private-sentinel" not in json.dumps(data)
            assert data[0]["usage"]["cache_read"] == 8
            assert result.usage.prompt_tokens == 12 and result.usage.total_tokens == 13
    finally:
        client.close()


def test_profile_config_and_sink_share_owner(tmp_path, monkeypatch):
    """Enabled profile B must not write its key or rows into disabled launch profile A."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    a, b = tmp_path / "a", tmp_path / "b"
    for home, enabled in ((a, False), (b, True)):
        home.mkdir(mode=0o700)
        (home / "config.yaml").write_text(json.dumps({"agent": {"codex_cache_diagnostics": {"enabled": enabled}}}))
    monkeypatch.setenv("HERMES_HOME", str(a))
    observed = []
    for home in (a, b, a):
        binding = set_hermes_home_override(home)
        try:
            enabled = diagnostics._enabled()
            token = diagnostics.begin_attempt({"model": "fixture", "input": []}, session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0)
            diagnostics.finish_attempt(token)
            observed.append({"profile": home.name, "enabled": enabled, "recorded": token is not None})
        finally:
            reset_hermes_home_override(binding)
    assert observed == [
        {"profile": "a", "enabled": False, "recorded": False},
        {"profile": "b", "enabled": True, "recorded": True},
        {"profile": "a", "enabled": False, "recorded": False},
    ]
    assert (b / "cache" / "codex-cache-prefix.jsonl").is_file()
    assert not (a / "cache" / "codex-cache-prefix.jsonl").exists()
    assert not (a / "cache" / "codex-cache-prefix.key").exists()


def test_public_codex_sink_keeps_origin_across_scope_exit(tmp_path, monkeypatch):
    """A→B→A public Codex stream persists only while B is enabled, including after scope reset."""
    import httpx
    from agent import secret_scope
    from agent.codex_runtime import run_codex_stream
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    a, b = tmp_path / "a", tmp_path / "b"
    for home, enabled in ((a, False), (b, True)):
        home.mkdir(mode=0o700)
        (home / "config.yaml").write_text(json.dumps({"agent": {"codex_cache_diagnostics": {"enabled": enabled}}}))
    monkeypatch.setenv("HERMES_HOME", str(a))
    sends = []

    def handler(request):
        del request
        sends.append(1)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":2,"input_tokens_details":{"cached_tokens":1}}}}\n\n',
        )

    client = _real_codex_client(httpx.MockTransport(handler), max_retries=0)
    secret_scope.set_multiplex_active(True)
    gates = []
    try:
        for home in (a, b, a):
            binding = set_hermes_home_override(home)
            secrets = secret_scope.set_secret_scope({}, profile_home=str(home))
            try:
                gates.append(diagnostics._enabled())
                assert run_codex_stream(_codex_agent(), {"model": "fixture", "input": []}, client=client).status == "completed"
            finally:
                secret_scope.reset_secret_scope(secrets)
                reset_hermes_home_override(binding)
        # Begin under B, leave B, then finalize. Ambient home at finish must not own the row.
        binding = set_hermes_home_override(b)
        secrets = secret_scope.set_secret_scope({}, profile_home=str(b))
        try:
            token = diagnostics.begin_attempt({"model": "fixture", "input": []}, session_id="late-b", turn_id="late-b", api_id="late-b", ordinal=9, retry=0)
            assert token is not None and token[1]["home"] == str(b)
        finally:
            secret_scope.reset_secret_scope(secrets)
            reset_hermes_home_override(binding)
        binding = set_hermes_home_override(a)
        secrets = secret_scope.set_secret_scope({}, profile_home=str(a))
        try:
            assert diagnostics._enabled() is False
            diagnostics.finish_attempt(token)
        finally:
            secret_scope.reset_secret_scope(secrets)
            reset_hermes_home_override(binding)
    finally:
        secret_scope.set_multiplex_active(False)
        client.close()
    assert gates == [False, True, False] and len(sends) == 3
    data = rows(b)
    assert len(data) == 2 and data[-1]["kind"] == "terminal"
    assert not (a / "cache" / "codex-cache-prefix.jsonl").exists()


def _read_private_key(home):
    os.environ["HERMES_HOME"] = str(home)
    diagnostics._enabled = lambda: True
    diagnostics.begin_attempt({}, session_id="", turn_id="", api_id="", ordinal=0, retry=0)


def test_nonregular_key_and_lock_fail_closed_without_blocking(tmp_path):
    """A FIFO at the production cache leaf must not block the request hook."""
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    cache = home / "cache"
    cache.mkdir(mode=0o700)
    for name in ("codex-cache-prefix.key", "codex-cache-prefix.lock"):
        leaf = cache / name
        if leaf.exists():
            leaf.unlink()
        os.mkfifo(leaf, 0o600)
        child = multiprocessing.get_context("fork").Process(target=_read_private_key, args=(home,))
        child.start()
        child.join(2)
        blocked = child.is_alive()
        if blocked:
            child.terminate()
            child.join(2)
        assert not child.is_alive()
        assert not blocked, name


def _child_begin_finish(home, phase, ready):
    os.environ["HERMES_HOME"] = str(home)
    diagnostics._enabled = lambda: True
    ready.set()
    token = diagnostics.begin_attempt({"model": "m", "input": []}, session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0)
    if phase == "begin":
        return
    diagnostics.finish_attempt(token)


@pytest.mark.parametrize("phase", ["begin", "finish"])
def test_held_regular_lock_fails_open(tmp_path, phase):
    """A cooperating holder of the 0600 lock must not stall begin or finish."""
    import fcntl
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    enable_home = home
    os.environ["HERMES_HOME"] = str(enable_home)
    diagnostics._enabled = lambda: True
    token = diagnostics.begin_attempt({"model": "m", "input": []}, session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0)
    diagnostics.finish_attempt(token)
    lock = home / "cache" / "codex-cache-prefix.lock"
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    ready = multiprocessing.get_context("fork").Event()
    child = multiprocessing.get_context("fork").Process(target=_child_begin_finish, args=(home, phase, ready))
    try:
        child.start()
        assert ready.wait(2)
        child.join(2)
        assert not child.is_alive()
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(2)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_process_key_lock_contention_skips(tmp_path, monkeypatch):
    """A held in-process key lock is skipped, not waited on."""
    enable(tmp_path, monkeypatch)
    diagnostics._key_lock.acquire()
    try:
        assert diagnostics.begin_attempt({"model": "m", "input": []}, session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0) is None
    finally:
        diagnostics._key_lock.release()
    token = diagnostics.begin_attempt({"model": "m", "input": []}, session_id="s", turn_id="t", api_id="a", ordinal=0, retry=0)
    assert token is not None
