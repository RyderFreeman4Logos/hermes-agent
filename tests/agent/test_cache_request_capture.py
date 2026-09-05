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
