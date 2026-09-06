from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent import cache_request_capture as capture


def _enable(monkeypatch, *, strict: bool = False) -> None:
    monkeypatch.setattr(
        capture,
        "_settings",
        lambda: {"enabled": True, "strict_write": strict},
    )


def _captures(tmp_path: Path) -> list[dict]:
    files = sorted((tmp_path / "debug" / "cache-requests").glob("*.json"))
    return [json.loads(path.read_text()) for path in files]


def _capture(request: dict[str, Any], **identity: Any) -> None:
    details = {
        "route": "https://provider.example.test/v1",
        "provider": "test-provider",
        "model": str(request.get("model") or "test-model"),
        "api_mode": "chat_completions",
    }
    details.update(identity)
    capture.capture_provider_request(request, **details)


def _first_difference(left, right, path=()):
    if type(left) is not type(right) or not isinstance(left, (dict, list)):
        return path if left != right else None
    if isinstance(left, list):
        if len(left) != len(right):
            return path + ("length",)
        for index, (left_child, right_child) in enumerate(zip(left, right)):
            difference = _first_difference(left_child, right_child, path + (index,))
            if difference is not None:
                return difference
        return None
    for key in left.keys() | right.keys():
        if key not in left or key not in right:
            return path + (key,)
        difference = _first_difference(left[key], right[key], path + (key,))
        if difference is not None:
            return difference
    return None


def test_capture_is_disabled_by_default(monkeypatch, tmp_path):
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(capture, "_settings", lambda: {})

    _capture({"model": "m"})

    assert cast(dict[str, Any], DEFAULT_CONFIG)["debug"]["cache_requests"] == {
        "enabled": False,
        "strict_write": False,
    }
    assert not (tmp_path / "debug").exists()


def test_capture_persists_redacted_serialized_body_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    request = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "Authorization": "Bearer secret",
        "callback_url": "https://user:secret@example.test/callback",
    }

    _capture(request)

    captured = _captures(tmp_path)[0]
    assert captured["request"]["Authorization"] == "[REDACTED]"
    assert "secret" not in json.dumps(captured)
    body = base64.b64decode(captured["body_bytes"]["data"])
    assert body == json.dumps(
        captured["request"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def test_exact_capture_preserves_one_character_system_prompt_diff(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    base = {"model": "m", "messages": [{"role": "system", "content": "A"}]}
    changed = {"model": "m", "messages": [{"role": "system", "content": "B"}]}

    _capture(base, provider="p", model="m")
    _capture(changed, provider="p", model="m")

    requests = [item["request"] for item in _captures(tmp_path)]
    assert requests[0]["messages"][0]["content"] == "A"
    assert requests[1]["messages"][0]["content"] == "B"
    assert _first_difference(requests[0], requests[1]) == ("messages", 0, "content")


def test_exact_capture_preserves_tool_reorder(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    first = {"tools": [{"name": "one"}, {"name": "two"}], "body": "body"}
    second = {"tools": [{"name": "two"}, {"name": "one"}], "body": "body"}

    _capture(first)
    _capture(second)

    requests = [item["request"] for item in _captures(tmp_path)]
    assert [tool["name"] for tool in requests[0]["tools"]] == ["one", "two"]
    assert [tool["name"] for tool in requests[1]["tools"]] == ["two", "one"]
    assert _first_difference(requests[0], requests[1]) == ("tools", 0, "name")


def test_exact_capture_preserves_input_and_prompt_cache_key(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    first = {
        "model": "grok-4.6",
        "messages": [{"role": "system", "content": "A"}],
        "input": "INPUT-A",
        "tools": [{"name": "TOOL-A"}],
        "prompt_cache_key": "PCK-A",
    }
    second = {
        "model": "grok-4.6",
        "messages": [{"role": "system", "content": "A"}],
        "input": "INPUT-B",
        "tools": [{"name": "TOOL-B"}],
        "prompt_cache_key": "PCK-B",
    }

    _capture(first)
    _capture(second)

    requests = [item["request"] for item in _captures(tmp_path)]
    assert requests[0]["input"] == "INPUT-A"
    assert requests[1]["input"] == "INPUT-B"
    assert requests[0]["prompt_cache_key"] == "PCK-A"
    assert requests[1]["prompt_cache_key"] == "PCK-B"
    assert requests[0]["tools"][0]["name"] == "TOOL-A"
    assert requests[1]["tools"][0]["name"] == "TOOL-B"


def test_persist_redacts_hostile_identity_and_neutral_scalar(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    route_secret = "opaque-" + "route-secret"
    scalar_secret = "opaque-" + "scalar-secret"
    _capture(
        {"context": f"postgres://user:{scalar_secret}@example.test/db"},
        route=f"https://user:{route_secret}@example.test/v1",
    )
    safe_route = "openai-responses"
    _capture({"context": "benign"}, route=safe_route)
    captures = _captures(tmp_path)
    serialized = json.dumps(captures)

    assert not any(marker in serialized for marker in (route_secret, scalar_secret))
    assert captures[0]["route"]["model"] == "test-model"
    assert captures[1]["route"]["route"] == safe_route


def test_primary_route_identity_and_persisted_scalars_are_strictly_sanitized(
    monkeypatch, tmp_path
):
    from agent import relay_llm

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    markers = (
        "query-credential-marker",
        "redis-password-marker",
        "fallback-secret-marker",
    )

    class HostileValue:
        def __str__(self) -> str:
            return "token=fallback-secret-marker"

    def _primary_attempt(route: str | None) -> None:
        relay_llm._execute_attempt(
            {
                "model": "m",
                "callback": "https://provider.example.test/cb?code=query-credential-marker",
                "cache": "redis://:redis-password-marker@host/0",
                "fallback": HostileValue(),
            },
            lambda request: relay_llm.capture_transport_request(request),
            name="openai",
            model_name="m",
            metadata={"api_mode": "chat_completions", "route": route},
        )

    _primary_attempt("https://provider.example.test/v1")
    captures = _captures(tmp_path)
    serialized = json.dumps(captures)
    assert not any(marker in serialized for marker in markers)
    assert captures[0]["route"]["route"] == "https://provider.example.test/v1"

    for route in (None, "", " ", "unknown"):
        _primary_attempt(route)
    assert len(_captures(tmp_path)) == 1


def test_transport_and_ordinary_secrets_are_redacted(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    request = {
        "headers": {
            "Authorization": "fake-auth-value",
            "X-API-Key": "fake-api-value",
            "Cookie": "fake-cookie-value",
        },
        "refresh_token": "fake-refresh-value",
        "password": "fake-password-value",
        "token": "fake-token-value",
        "access_token": "fake-access-token-value",
        "client_secret": "fake-client-secret-value",
        "private_key": "fake-private-key-value",
        "credentials": "fake-credentials-value",
        "key": "fake-key-value",
        "secret": "fake-secret-value",
        "messages": [{"role": "user", "content": "prompt-body", "password": "fake-nested-password"}],
        "body": "body-value",
        "cache_control": {"type": "ephemeral"},
        "tools": [{"function": {"parameters": {"properties": {"api_key": {"type": "string"}}}}}],
    }

    _capture(request)

    path = next((tmp_path / "debug" / "cache-requests").glob("*.json"))
    serialized = path.read_text()
    saved = json.loads(serialized)["request"]
    for value in (
        "fake-auth-value",
        "fake-api-value",
        "fake-cookie-value",
        "fake-refresh-value",
        "fake-password-value",
        "fake-token-value",
        "fake-access-token-value",
        "fake-client-secret-value",
        "fake-private-key-value",
        "fake-credentials-value",
        "fake-key-value",
        "fake-secret-value",
        "fake-nested-password",
    ):
        assert value not in serialized
    assert saved["headers"]["Authorization"] == "[REDACTED]"
    assert saved["headers"]["X-API-Key"] == "[REDACTED]"
    assert saved["headers"]["Cookie"] == "[REDACTED]"
    assert saved["refresh_token"] == "[REDACTED]"
    assert saved["password"] == "[REDACTED]"
    assert saved["token"] == "[REDACTED]"
    assert saved["access_token"] == "[REDACTED]"
    assert saved["client_secret"] == "[REDACTED]"
    assert saved["private_key"] == "[REDACTED]"
    assert saved["credentials"] == "[REDACTED]"
    assert saved["key"] == "[REDACTED]"
    assert saved["secret"] == "[REDACTED]"
    assert saved["messages"][0]["content"] == "prompt-body"
    assert saved["messages"][0]["password"] == "[REDACTED]"
    assert saved["body"] == "body-value"
    assert saved["cache_control"] == {"type": "ephemeral"}
    assert saved["tools"][0]["function"]["parameters"]["properties"]["api_key"] == "[REDACTED]"


def test_capture_redacts_secret_key_tokens_without_redacting_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    markers = {
        "AWS_SECRET_ACCESS_KEY": "redaction-class-aws-marker",
        "database_secret": "redaction-class-suffix-marker",
        "x-api-key": "redaction-class-conventional-marker",
    }
    nested_markers = {
        "provider_secret_access_key": "redaction-class-nested-marker",
    }

    _capture(
        {
            **markers,
            "nested": nested_markers,
            "api_mode": "chat_completions",
            "model": "test-model",
            "route": "primary-route",
        }
    )

    serialized = next((tmp_path / "debug" / "cache-requests").glob("*.json")).read_text()
    saved = json.loads(serialized)["request"]
    assert not any(marker in serialized for marker in (*markers.values(), *nested_markers.values()))
    assert all(saved[key] == "[REDACTED]" for key in markers)
    assert saved["nested"]["provider_secret_access_key"] == "[REDACTED]"
    assert saved["api_mode"] == "chat_completions"
    assert saved["model"] == "test-model"
    assert saved["route"] == "primary-route"


def test_preserved_payload_redacts_structured_secret_values(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    request = {
        "messages": [
            {
                "content": {
                    "password": {"value": "dict-secret-marker"},
                    "token": ["list-secret-marker"],
                }
            }
        ]
    }

    _capture(request)

    serialized = next((tmp_path / "debug" / "cache-requests").glob("*.json")).read_text()
    saved = json.loads(serialized)["request"]
    assert "dict-secret-marker" not in serialized
    assert "list-secret-marker" not in serialized
    assert saved["messages"][0]["content"]["password"] == "[REDACTED]"
    assert saved["messages"][0]["content"]["token"] == "[REDACTED]"


def test_capture_is_non_fatal_by_default(monkeypatch):
    from agent import relay_llm

    _enable(monkeypatch)
    monkeypatch.setattr(capture, "_persist", lambda payload: (_ for _ in ()).throw(OSError("disk")))
    called = []

    result = relay_llm._execute_attempt(
        {"messages": []},
        lambda request: called.append(request) or "response",
        name="provider",
        model_name="m",
        metadata=None,
    )

    assert result == "response"
    assert called == [{"messages": []}]


def test_strict_write_propagates_capture_failure(monkeypatch):
    _enable(monkeypatch, strict=True)
    monkeypatch.setattr(capture, "_persist", lambda payload: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(OSError, match="disk"):
        _capture({"messages": []})


def test_capture_rejects_incomplete_or_conflicting_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)

    capture.capture_provider_request({"model": "m"})
    _capture({"model": "m"}, route=" ")
    _capture({"model": "m"}, provider="unknown")
    _capture({"model": "m"}, api_mode=" ")
    _capture({"model": "m"}, model="other-model")

    assert not (tmp_path / "debug").exists()


def test_atomic_private_capture_files(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)

    _capture({"model": "m"})

    root = tmp_path / "debug" / "cache-requests"
    path = next(root.glob("*.json"))
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(root).st_mode & 0o777 == 0o700
    assert not list(root.glob(".tmp-*"))


def test_capture_records_each_physical_attempt(monkeypatch, tmp_path):
    from agent import relay_llm

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    request = {"model": "m", "messages": [{"role": "user", "content": "x"}]}

    for retry_count in (0, 1):
        relay_llm._execute_attempt(
            dict(request),
            lambda final_request: relay_llm.capture_transport_request(final_request),
            name="provider",
            model_name="m",
            metadata={
                "api_mode": "chat_completions",
                "api_request_id": "turn:api:0",
                "retry_count": retry_count,
                "route": "https://provider.example.test/v1",
            },
        )

    captures = _captures(tmp_path)
    assert len(captures) == 2
    assert [item["physical_attempt"]["retry"] for item in captures] == [0, 1]


def test_openai_capture_matches_final_sdk_kwargs(monkeypatch):
    from agent import chat_completion_helpers, relay_llm

    opened = []
    captured = []

    class Completions:
        def create(self, **kwargs):
            opened.append(dict(kwargs))
            return "response"

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openai",
        client=client,
    )
    monkeypatch.setattr(
        relay_llm,
        "capture_transport_request",
        lambda request: captured.append(dict(request)),
    )

    request = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    result = chat_completion_helpers._dispatch_nonstreaming_api_request(
        agent, request, make_client=lambda *args, **kwargs: client
    )

    assert result == "response"
    assert captured == opened


def test_anthropic_capture_matches_stream_and_fallback_kwargs(monkeypatch):
    from agent import anthropic_adapter, relay_llm

    opened = []
    captured = []

    class Messages:
        def stream(self, **kwargs):
            opened.append(dict(kwargs))
            raise RuntimeError("stream not supported")

        def create(self, **kwargs):
            opened.append(dict(kwargs))
            return "response"

    monkeypatch.setattr(
        relay_llm,
        "capture_transport_request",
        lambda request: captured.append(dict(request)),
    )
    request = {
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "instructions": "drop",
        "input": "drop",
        "store": True,
        "parallel_tool_calls": True,
        "stream": True,
    }

    result = anthropic_adapter.create_anthropic_message(
        SimpleNamespace(messages=Messages()), request
    )

    assert result == "response"
    assert captured == opened
    assert captured == [
        {"model": "m", "messages": [{"role": "user", "content": "x"}]},
        {"model": "m", "messages": [{"role": "user", "content": "x"}]},
    ]


def test_codex_moa_and_streaming_bounded_sends_keep_capture_context(
    monkeypatch, tmp_path
):
    from agent import auxiliary_client as auxiliary, relay_llm

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    request = {"model": "m", "messages": [{"role": "user", "content": "x"}]}

    for provider, api_mode in (
        ("openai-codex", "codex_responses"),
        ("moa", "chat_completions"),
        ("streaming", "chat_completions"),
    ):
        with relay_llm._transport_capture_context(
            name=provider,
            model_name="m",
            metadata={
                "api_mode": api_mode,
                "route": "https://provider.example.test/v1",
            },
        ):
            auxiliary._create_bounded(
                lambda: relay_llm.capture_transport_request(request), 1
            )

    captures = _captures(tmp_path)
    assert [item["request"] for item in captures] == [request] * 3
    assert [item["route"]["provider"] for item in captures] == [
        "openai-codex",
        "moa",
        "streaming",
    ]


def test_bedrock_stream_fallback_captures_each_final_opener(monkeypatch, tmp_path):
    from agent import bedrock_adapter, relay_llm

    opened = []
    captured = []

    class Client:
        def converse_stream(self, **kwargs):
            opened.append(dict(kwargs))
            raise RuntimeError("access denied")

        def converse(self, **kwargs):
            opened.append(dict(kwargs))
            return {"response": "ok"}

    monkeypatch.setattr(
        bedrock_adapter, "_get_bedrock_runtime_client", lambda region: Client()
    )
    monkeypatch.setattr(
        bedrock_adapter, "is_streaming_access_denied_error", lambda exc: True
    )
    monkeypatch.setattr(
        bedrock_adapter, "normalize_converse_response", lambda response: response
    )
    monkeypatch.setattr(
        relay_llm,
        "capture_transport_request",
        lambda request: captured.append(dict(request)),
    )

    result = bedrock_adapter.call_converse_stream(
        "us-east-1", "m", [{"role": "user", "content": [{"text": "x"}]}]
    )

    assert result == {"response": "ok"}
    assert captured == opened


def test_codex_app_server_turn_start_and_steer_capture_once(monkeypatch):
    from agent import relay_llm
    from agent.transports.codex_app_server import CodexAppServerError
    from agent.transports.codex_app_server_session import CodexAppServerSession

    events = []

    class Client:
        def request(self, method, params, timeout=None):
            events.append(("open", method, dict(params)))
            raise CodexAppServerError(code=-1, message="opener failed")

        def stderr_tail(self, _count):
            return []

    monkeypatch.setattr(
        relay_llm,
        "capture_transport_request",
        lambda request: events.append(("capture", dict(request))),
    )
    session = CodexAppServerSession()
    session._client = cast(Any, Client())
    session._thread_id = "thread-1"

    result = session.run_turn("start")
    session._active_turn_id = "turn-1"
    assert session.request_steer("steer") is False

    assert result.error is not None
    assert [event[:2] for event in events] == [
        ("capture", {"threadId": "thread-1", "input": [{"type": "text", "text": "start"}]}),
        ("open", "turn/start"),
        (
            "capture",
            {
                "threadId": "thread-1",
                "input": [{"type": "text", "text": "steer"}],
                "expectedTurnId": "turn-1",
            },
        ),
        ("open", "turn/steer"),
    ]
    assert events[0][1] == events[1][2]
    assert events[2][1] == events[3][2]


def test_native_gemini_sync_and_stream_capture_once(monkeypatch):
    from agent import relay_llm
    from agent.gemini_native_adapter import GeminiNativeClient

    events = []

    class HTTP:
        def post(self, _url, *, json, headers, timeout):
            events.append(("open", dict(json)))
            raise RuntimeError("sync opener failed")

        def stream(self, _method, _url, *, json, headers, timeout):
            events.append(("open", dict(json)))
            raise RuntimeError("stream opener failed")

        def close(self):
            return None

    monkeypatch.setattr(
        relay_llm,
        "capture_transport_request",
        lambda request: events.append(("capture", dict(request))),
    )
    client = GeminiNativeClient(api_key="test", http_client=cast(Any, HTTP()))
    kwargs = {
        "model": "gemini-test",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with pytest.raises(RuntimeError, match="sync opener failed"):
        client.chat.completions.create(**kwargs)
    stream = client.chat.completions.create(**kwargs, stream=True)
    assert [kind for kind, _request in events] == ["capture", "open"]
    with pytest.raises(RuntimeError, match="stream opener failed"):
        next(stream)

    assert [kind for kind, _request in events] == ["capture", "open"] * 2
    assert events[0][1] == events[1][1]
    assert events[2][1] == events[3][1]


def test_auxiliary_direct_and_deadline_capture_before_failure(monkeypatch):
    from agent import auxiliary_client as auxiliary, relay_llm

    events = []

    class Completions:
        def create(self, **kwargs):
            events.append(("open", dict(kwargs)))
            raise RuntimeError("opener failed")

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(
        relay_llm,
        "capture_transport_request",
        lambda request: events.append(("capture", dict(request))),
    )

    direct = {"model": "direct", "messages": []}
    with pytest.raises(RuntimeError, match="opener failed"):
        auxiliary._create_with_progress(client, direct)

    deadline = {"model": "deadline", "messages": [], "timeout": 1}
    with auxiliary.aux_host_candidate_deadline(1):
        with pytest.raises(RuntimeError, match="opener failed"):
            auxiliary._create_with_progress(client, deadline)

    assert [kind for kind, _request in events] == ["capture", "open"] * 2
    assert events[0][1] == events[1][1] == direct
    assert events[2][1] == events[3][1]
    assert events[2][1]["model"] == "deadline"


def test_auxiliary_stream_and_fallback_openers_capture_before_sdk(monkeypatch):
    from agent import auxiliary_client as auxiliary, relay_llm

    events = []

    class Completions:
        def create(self, **kwargs):
            events.append(("open", dict(kwargs)))
            if kwargs.get("stream"):
                raise RuntimeError("stream not supported")
            return "response"

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(
        relay_llm,
        "capture_transport_request",
        lambda request: events.append(("capture", dict(request))),
    )
    monkeypatch.setattr(auxiliary, "_aux_progress_active", lambda: True)

    request = {"model": "aux-stream", "messages": []}
    result = auxiliary._create_with_progress(client, request)

    assert result == "response"
    assert [kind for kind, _request in events] == ["capture", "open"] * 2
    assert events[0][1] == events[1][1]
    assert events[0][1]["stream"] is True
    assert events[2][1] == events[3][1] == request


def test_create_bounded_is_used_for_deadline_create_with_progress(monkeypatch):
    from agent import auxiliary_client as auxiliary, relay_llm

    events = []
    bounded = []

    class Completions:
        def create(self, **kwargs):
            events.append(("open", dict(kwargs)))
            raise RuntimeError("opener failed")

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    real_bounded = auxiliary._create_bounded

    def wrapped(create_fn, timeout_s):
        bounded.append(timeout_s)
        return real_bounded(create_fn, timeout_s)

    monkeypatch.setattr(auxiliary, "_create_bounded", wrapped)
    monkeypatch.setattr(
        relay_llm,
        "capture_transport_request",
        lambda request: events.append(("capture", dict(request))),
    )

    deadline = {"model": "deadline", "messages": [], "timeout": 1}
    with auxiliary.aux_host_candidate_deadline(1):
        with pytest.raises(RuntimeError, match="opener failed"):
            auxiliary._create_with_progress(client, deadline)

    assert bounded == [1]
    assert [kind for kind, _request in events] == ["capture", "open"]
    assert events[0][1] == events[1][1]
    assert events[0][1]["model"] == "deadline"


def test_codex_runtime_stream_opener_captures_before_sdk(monkeypatch):
    from agent import codex_runtime, relay_llm

    events = []

    class Client:
        def __init__(self):
            self.responses = self

        def create(self, **kwargs):
            events.append(("open", dict(kwargs)))
            raise RuntimeError("codex opener failed")

    monkeypatch.setattr(
        relay_llm,
        "capture_transport_request",
        lambda request: events.append(("capture", dict(request))),
    )

    agent = SimpleNamespace(
        _interrupt_requested=False,
        session_id="",
        provider="openai-codex",
        _fire_stream_delta=lambda text: None,
        _fire_reasoning_delta=lambda text: None,
        _fire_streamed_codex_commentary=lambda text: None,
        _touch_activity=lambda *args, **kwargs: None,
        _is_codex_backend=lambda: False,
        _claim_stream_writer=lambda: object(),
    )
    with pytest.raises(RuntimeError, match="codex opener failed"):
        codex_runtime.run_codex_stream(agent, {"model": "codex-m"}, client=Client())

    assert [kind for kind, _request in events] == ["capture", "open"]
    assert events[0][1] == events[1][1]
    assert events[0][1]["stream"] is True


def test_execute_current_no_turn_sets_capture_context(monkeypatch, tmp_path):
    from agent import relay_llm, relay_runtime

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    monkeypatch.setattr(relay_runtime, "active_turn", lambda: None)

    def callback(request):
        relay_llm.capture_transport_request(request)
        return "ok"

    result = relay_llm.execute_current(
        {"model": "m", "messages": []},
        callback,
        name="provider",
        model_name="m",
        metadata={
            "api_mode": "chat_completions",
            "route": "https://provider.example.test/v1",
            "api_request_id": "no-turn:0",
        },
    )

    assert result == "ok"
    captures = _captures(tmp_path)
    assert len(captures) == 1
    assert captures[0]["route"]["provider"] == "provider"


@pytest.mark.asyncio
async def test_execute_current_async_no_turn_sets_capture_context(
    monkeypatch, tmp_path
):
    from agent import relay_llm, relay_runtime

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    monkeypatch.setattr(relay_runtime, "active_turn", lambda: None)

    async def callback(request):
        relay_llm.capture_transport_request(request)
        return "ok"

    result = await relay_llm.execute_current_async(
        {"model": "m", "messages": []},
        callback,
        name="provider",
        model_name="m",
        metadata={
            "api_mode": "chat_completions",
            "route": "https://provider.example.test/v1",
            "api_request_id": "no-turn-async:0",
        },
    )

    assert result == "ok"
    captures = _captures(tmp_path)
    assert len(captures) == 1


def test_codex_direct_auxiliary_stream_captures_once(monkeypatch, tmp_path):
    from agent import auxiliary_client as auxiliary

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    opened = []
    base_url = "https://chatgpt.com/backend-api/codex"

    class Responses:
        def create(self, **kwargs):
            opened.append(dict(kwargs))
            return SimpleNamespace(output=[], usage=None)

    real_client = SimpleNamespace(
        api_key="[REDACTED]",
        base_url=base_url,
        responses=Responses(),
        close=lambda: None,
    )
    client = auxiliary.CodexAuxiliaryClient(real_client, "gpt-5.4")
    monkeypatch.setattr(
        auxiliary,
        "_resolve_task_provider_model",
        lambda *args, **kwargs: (
            "openai-codex",
            "gpt-5.4",
            base_url,
            "[REDACTED]",
            "codex_responses",
        ),
    )
    monkeypatch.setattr(
        auxiliary,
        "_get_cached_client",
        lambda *args, **kwargs: (client, "gpt-5.4"),
    )
    monkeypatch.setattr(auxiliary, "_get_task_extra_body", lambda task: {})
    monkeypatch.setattr(
        auxiliary, "_effective_provider_for_client", lambda *args: "openai-codex"
    )
    monkeypatch.setattr(auxiliary, "_acquire_sync_aux_semaphore", lambda task: None)
    monkeypatch.setattr(
        auxiliary,
        "_build_call_kwargs",
        lambda provider, model, messages, **kwargs: {
            "model": model,
            "messages": messages,
        },
    )
    result = auxiliary.call_llm(
        task="moa_aggregator",
        provider="openai-codex",
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    assert result.choices
    assert len(opened) == 1
    captures = _captures(tmp_path)
    assert len(captures) == 1
    saved = captures[0]
    assert saved["route"]["provider"] == "openai-codex"
    assert saved["route"]["api_mode"] == "codex_responses"
    assert saved["request"] == opened[0]


@pytest.mark.asyncio
async def test_async_codex_fallback_captures_once(monkeypatch, tmp_path):
    from agent import auxiliary_client as auxiliary

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    opened = []

    class Responses:
        def create(self, **kwargs):
            opened.append(dict(kwargs))
            return SimpleNamespace(output=[], usage=None)

    real_client = SimpleNamespace(
        api_key="[REDACTED]",
        base_url="https://chatgpt.com/backend-api/codex",
        responses=Responses(),
        close=lambda: None,
    )
    client = auxiliary.AsyncCodexAuxiliaryClient(
        auxiliary.CodexAuxiliaryClient(real_client, "gpt-5.4")
    )

    @auxiliary._relay_auxiliary_call_async
    async def run(task):
        auxiliary._set_relay_auxiliary_route(
            "openai-codex", "gpt-5.4", "codex_responses"
        )
        return await auxiliary._relay_async_completion(
            client, {"model": "gpt-5.4", "messages": []}
        )

    await run("title_generation")

    assert len(opened) == 1
    assert len(_captures(tmp_path)) == 1


def test_bedrock_nonstream_converse_captures_before_sdk(monkeypatch):
    from agent import bedrock_adapter, relay_llm

    opened = []
    captured = []

    class Client:
        def converse(self, **kwargs):
            opened.append(dict(kwargs))
            return {"output": {"message": {"content": [{"text": "ok"}]}}}

    monkeypatch.setattr(
        bedrock_adapter, "_get_bedrock_runtime_client", lambda region: Client()
    )
    monkeypatch.setattr(
        bedrock_adapter, "normalize_converse_response", lambda response: response
    )
    monkeypatch.setattr(
        relay_llm,
        "capture_transport_request",
        lambda request: captured.append(dict(request)),
    )

    bedrock_adapter.call_converse(
        "us-east-1", "m", [{"role": "user", "content": [{"text": "x"}]}]
    )

    assert captured == opened


def _first_byte(left: bytes, right: bytes) -> int | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit if len(left) != len(right) else None


def _openai_chat_response(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl_probe",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }


def _send_openai_sdk(*, tmp_path: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
    import httpx
    from openai import OpenAI

    from agent import relay_llm

    wire: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        wire["content"] = bytes(request.content)
        wire["headers"] = dict(request.headers)
        wire["url"] = str(request.url)
        return httpx.Response(200, json=_openai_chat_response(str(kwargs["model"])))

    client = OpenAI(
        api_key="sk-probe-not-a-real-key",
        base_url="https://provider.example.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    def send(request: dict[str, Any]) -> Any:
        relay_llm.capture_transport_request(request)
        return client.chat.completions.create(**request)

    relay_llm._execute_attempt(
        dict(kwargs),
        send,
        name="openai",
        model_name=str(kwargs["model"]),
        metadata={
            "api_mode": "chat_completions",
            "route": "https://provider.example.test/v1",
            "api_request_id": "sdk-wire:0",
        },
    )
    captures = _captures(tmp_path)
    assert captures, "expected one persisted capture"
    captured = captures[-1]
    return {
        "wire": wire,
        "captured": captured,
        "body": base64.b64decode(captured["body_bytes"]["data"]),
        "serialized": json.dumps(captured),
    }


def test_openai_sdk_wire_body_is_captured_exactly(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    kwargs = {
        "model": "probe-model",
        "messages": [
            {"role": "system", "content": "SYS-A"},
            {"role": "user", "content": "hello-prompt"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "tool-desc",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "prompt_cache_key": "PCK-A",
    }

    result = _send_openai_sdk(tmp_path=tmp_path, kwargs=kwargs)

    assert result["body"] == result["wire"]["content"]
    assert result["captured"]["request"]["messages"] == kwargs["messages"]
    assert result["captured"]["request"]["tools"] == kwargs["tools"]
    assert result["captured"]["request"]["prompt_cache_key"] == "PCK-A"
    assert "sk-probe-not-a-real-key" not in result["serialized"]
    assert result["captured"]["request"]["messages"][1]["content"] == "hello-prompt"
    assert "chatgpt.com" not in result["serialized"]
    assert "api.openai.com" not in result["serialized"]


def test_openai_sdk_wire_first_diff_is_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    base = {
        "model": "probe-model",
        "messages": [{"role": "system", "content": "SYS-A"}],
        "tools": [{"type": "function", "function": {"name": "one"}}],
        "prompt_cache_key": "PCK-A",
    }
    message = {
        **base,
        "messages": [{"role": "system", "content": "SYS-B"}],
    }
    tools = {
        **base,
        "tools": [{"type": "function", "function": {"name": "two"}}],
    }
    scope = {**base, "prompt_cache_key": "PCK-B"}

    first = _send_openai_sdk(tmp_path=tmp_path, kwargs=base)
    second = _send_openai_sdk(tmp_path=tmp_path, kwargs=message)
    third = _send_openai_sdk(tmp_path=tmp_path, kwargs=tools)
    fourth = _send_openai_sdk(tmp_path=tmp_path, kwargs=scope)

    assert _first_byte(first["body"], second["body"]) == _first_byte(
        first["wire"]["content"], second["wire"]["content"]
    )
    assert _first_difference(first["captured"]["request"], second["captured"]["request"]) == (
        "messages",
        0,
        "content",
    )
    assert _first_difference(first["captured"]["request"], third["captured"]["request"]) == (
        "tools",
        0,
        "function",
        "name",
    )
    assert _first_difference(first["captured"]["request"], fourth["captured"]["request"]) == (
        "prompt_cache_key",
    )
    assert json.loads(first["wire"]["content"])["prompt_cache_key"] == "PCK-A"
    assert json.loads(fourth["wire"]["content"])["prompt_cache_key"] == "PCK-B"


# Synthetic fixtures only. Never interpolate these into assertion messages.
_FIXTURE_URL_USER = "probeuser"
_FIXTURE_URL_PASS = "canarypass-9b2e"
_FIXTURE_PRIVATE_URL = (
    f"https://{_FIXTURE_URL_USER}:{_FIXTURE_URL_PASS}@private.example.test/v1/hidden"
)
_FIXTURE_API_KEY = "sk-ant-api03-opaqueCanaryValueNotReal000"
_FIXTURE_PROMPT = "keep-this-nonsensitive-prompt"


def _fixture_absent(blob: bytes | str, *markers: str) -> bool:
    data = blob if isinstance(blob, bytes) else blob.encode("utf-8")
    return all(marker.encode("utf-8") not in data for marker in markers)


def _capture_file_bytes(tmp_path: Path) -> bytes:
    folder = tmp_path / "debug" / "cache-requests"
    return b"".join(path.read_bytes() for path in sorted(folder.glob("*.json")))


def _identity_meta(request_id: str = "probe:0") -> dict[str, Any]:
    return {
        "api_mode": "chat_completions",
        "route": "https://provider.example.test/v1",
        "api_request_id": request_id,
        "retry_count": 0,
    }


def test_httpx_body_embedded_secrets_are_sanitized_not_exact_wire(monkeypatch, tmp_path):
    """F1: raw transport bytes must not bypass the sanitizer."""
    import httpx
    from agent import relay_llm

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    wire: dict[str, Any] = {}
    sent = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["count"] += 1
        wire["content"] = bytes(request.content)
        return httpx.Response(200, json={"ok": True})

    body = json.dumps(
        {
            "model": "probe-model",
            "messages": [{"role": "user", "content": f"{_FIXTURE_PROMPT} {_FIXTURE_PRIVATE_URL}"}],
            "api_key": _FIXTURE_API_KEY,
        },
        separators=(",", ":"),
    ).encode("utf-8")

    def send_cb(request: dict[str, Any]) -> Any:
        relay_llm.capture_transport_request(request)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return client.send(
                httpx.Request(
                    "POST",
                    "https://provider.example.test/v1/chat/completions",
                    content=body,
                )
            )

    relay_llm._execute_attempt(
        {
            "model": "probe-model",
            "messages": [{"role": "user", "content": _FIXTURE_PROMPT}],
        },
        send_cb,
        name="openai",
        model_name="probe-model",
        metadata=_identity_meta("f1:0"),
    )

    if sent["count"] != 1:
        pytest.fail("expected one MockTransport send")
    captures = _captures(tmp_path)
    if len(captures) != 1:
        pytest.fail("expected one persisted capture")
    captured = captures[0]
    persisted = base64.b64decode(captured["body_bytes"]["data"])
    file_bytes = _capture_file_bytes(tmp_path)
    markers = (_FIXTURE_URL_PASS, _FIXTURE_API_KEY, _FIXTURE_URL_USER)
    if not _fixture_absent(persisted, *markers) or not _fixture_absent(file_bytes, *markers):
        pytest.fail("capture leaked a fixture marker")
    if captured["request"]["messages"][0]["content"] != _FIXTURE_PROMPT:
        pytest.fail("nonsensitive prompt was not retained")
    if captured["body_bytes"].get("status") == "exact_wire":
        pytest.fail("sensitive body claimed exact-wire identity")
    if persisted == wire["content"]:
        pytest.fail("changed bytes were claimed equal to the wire body")
    if _FIXTURE_PROMPT.encode("utf-8") not in persisted and _FIXTURE_PROMPT.encode(
        "utf-8"
    ) not in json.dumps(captured["request"]).encode("utf-8"):
        pytest.fail("nonsensitive prompt missing from sanitized capture")


def test_strict_write_does_not_abort_unread_stream_send(monkeypatch, tmp_path):
    """F2: unread stream bodies must not raise into Client.send."""
    import httpx
    from agent import relay_llm

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch, strict=True)
    sent = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["count"] += 1
        return httpx.Response(200, json={"ok": True})

    def send_cb(request: dict[str, Any]) -> Any:
        relay_llm.capture_transport_request(request)
        req = httpx.Request(
            "POST",
            "https://provider.example.test/v1",
            content=iter([b'{"stream":true}']),
        )
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return client.send(req)

    result = relay_llm._execute_attempt(
        {
            "model": "probe-model",
            "messages": [{"role": "user", "content": "stream-prompt"}],
        },
        send_cb,
        name="openai",
        model_name="probe-model",
        metadata=_identity_meta("f2:0"),
    )

    if sent["count"] != 1:
        pytest.fail("streaming Client.send was aborted before transport")
    if result is None:
        pytest.fail("streaming send returned no response")
    captures = _captures(tmp_path)
    if len(captures) != 1:
        pytest.fail("expected kwargs fallback capture for unread stream")
    captured = captures[0]
    if captured["body_bytes"].get("status") == "exact_wire":
        pytest.fail("unread stream claimed exact-wire identity")
    if captured["request"]["messages"][0]["content"] != "stream-prompt":
        pytest.fail("nonsensitive stream prompt was not retained")
    persisted = base64.b64decode(captured["body_bytes"]["data"])
    if b'"stream":true' in persisted:
        pytest.fail("capture consumed an unread streaming body")


def test_buffered_httpx_retries_persist_each_send(monkeypatch, tmp_path):
    import httpx
    from agent import relay_llm

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    sent = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["count"] += 1
        return httpx.Response(200, json={"ok": True})

    def send_cb(request: dict[str, Any]) -> Any:
        relay_llm.capture_transport_request(request)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            client.send(
                httpx.Request(
                    "POST",
                    "https://provider.example.test/v1",
                    content=b'{"retry":1,"prompt":"keep-a"}',
                )
            )
            return client.send(
                httpx.Request(
                    "POST",
                    "https://provider.example.test/v1",
                    content=b'{"retry":2,"prompt":"keep-b"}',
                )
            )

    relay_llm._execute_attempt(
        {
            "model": "probe-model",
            "messages": [{"role": "user", "content": "retry-prompt"}],
        },
        send_cb,
        name="openai",
        model_name="probe-model",
        metadata=_identity_meta("retry:0"),
    )

    if sent["count"] != 2:
        pytest.fail("expected two MockTransport sends")
    captures = _captures(tmp_path)
    if len(captures) != 2:
        pytest.fail("retry HTTP bodies were dropped")
    bodies = [base64.b64decode(item["body_bytes"]["data"]) for item in captures]
    if b'"retry":1' not in bodies[0] or b'"retry":2' not in bodies[1]:
        pytest.fail("persisted retry bodies were not the buffered sends")
    if any(item["body_bytes"].get("status") not in {"exact_wire", "sanitized"} for item in captures):
        pytest.fail("buffered retry captures lacked a wire status")


@pytest.mark.asyncio
async def test_async_httpx_buffered_body_is_captured(monkeypatch, tmp_path):
    import httpx
    from agent import relay_llm

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    sent = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["count"] += 1
        return httpx.Response(200, json={"ok": True})

    async def send_cb(request: dict[str, Any]) -> Any:
        relay_llm.capture_transport_request(request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await client.send(
                httpx.Request(
                    "POST",
                    "https://provider.example.test/v1",
                    content=b'{"async_wire":true,"prompt":"async-prompt"}',
                )
            )

    await relay_llm._execute_attempt_async(
        {
            "model": "probe-model",
            "messages": [{"role": "user", "content": "async-prompt"}],
        },
        send_cb,
        name="openai",
        model_name="probe-model",
        metadata=_identity_meta("async:0"),
    )

    if sent["count"] != 1:
        pytest.fail("expected one async MockTransport send")
    captures = _captures(tmp_path)
    if len(captures) != 1:
        pytest.fail("expected one async capture")
    captured = captures[0]
    persisted = base64.b64decode(captured["body_bytes"]["data"])
    if b'"async_wire":true' not in persisted:
        pytest.fail("async buffered body was not captured")
    if captured["body_bytes"].get("status") == "kwargs_fallback":
        pytest.fail("async httpx send was labeled kwargs fallback")
    if captured["request"]["messages"][0]["content"] != "async-prompt":
        pytest.fail("nonsensitive async prompt was not retained")


def test_non_httpx_fallback_is_not_exact_wire(monkeypatch, tmp_path):
    from agent import relay_llm

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)

    def send_cb(request: dict[str, Any]) -> Any:
        relay_llm.capture_transport_request(request)
        return "ok"

    relay_llm._execute_attempt(
        {
            "model": "probe-model",
            "messages": [{"role": "user", "content": "fallback-prompt"}],
            "Authorization": "Bearer not-a-real-credential",
        },
        send_cb,
        name="bedrock",
        model_name="probe-model",
        metadata=_identity_meta("fallback:0"),
    )

    captured = _captures(tmp_path)[0]
    if captured["body_bytes"].get("status") != "kwargs_fallback":
        pytest.fail("non-httpx fallback lacked an explicit non-wire status")
    if captured["request"]["messages"][0]["content"] != "fallback-prompt":
        pytest.fail("nonsensitive fallback prompt was not retained")
    if captured["request"]["Authorization"] != "[REDACTED]":
        pytest.fail("fallback Authorization was not redacted")
    serialized = json.dumps(captured)
    if "not-a-real-credential" in serialized:
        pytest.fail("fallback capture leaked an auth fixture")
