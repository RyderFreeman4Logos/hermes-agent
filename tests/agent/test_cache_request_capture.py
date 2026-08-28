from __future__ import annotations

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

    capture.capture_provider_request({"model": "m"})

    assert cast(dict[str, Any], DEFAULT_CONFIG)["debug"]["cache_requests"] == {
        "enabled": False,
        "strict_write": False,
    }
    assert not (tmp_path / "debug").exists()


def test_exact_capture_preserves_one_character_system_prompt_diff(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    base = {"model": "m", "messages": [{"role": "system", "content": "A"}]}
    changed = {"model": "m", "messages": [{"role": "system", "content": "B"}]}

    capture.capture_provider_request(base, provider="p", model="m")
    capture.capture_provider_request(changed, provider="p", model="m")

    requests = [item["request"] for item in _captures(tmp_path)]
    assert requests[0]["messages"][0]["content"] == "A"
    assert requests[1]["messages"][0]["content"] == "B"
    assert _first_difference(requests[0], requests[1]) == ("messages", 0, "content")


def test_exact_capture_preserves_tool_reorder(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    first = {"tools": [{"name": "one"}, {"name": "two"}], "body": "body"}
    second = {"tools": [{"name": "two"}, {"name": "one"}], "body": "body"}

    capture.capture_provider_request(first)
    capture.capture_provider_request(second)

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

    capture.capture_provider_request(first)
    capture.capture_provider_request(second)

    requests = [item["request"] for item in _captures(tmp_path)]
    assert requests[0]["input"] == "INPUT-A"
    assert requests[1]["input"] == "INPUT-B"
    assert requests[0]["prompt_cache_key"] == "PCK-A"
    assert requests[1]["prompt_cache_key"] == "PCK-B"
    assert requests[0]["tools"][0]["name"] == "TOOL-A"
    assert requests[1]["tools"][0]["name"] == "TOOL-B"


def test_only_transport_secrets_are_redacted(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    request = {
        "headers": {
            "Authorization": "Bearer auth-secret",
            "X-API-Key": "api-secret",
            "Cookie": "cookie-secret",
        },
        "refresh_token": "refresh-secret",
        "messages": [{"role": "user", "content": "prompt-secret"}],
        "body": "body-secret",
        "cache_control": {"type": "ephemeral"},
        "tools": [{"function": {"parameters": {"properties": {"api_key": {"type": "string"}}}}}],
    }

    capture.capture_provider_request(request)

    path = next((tmp_path / "debug" / "cache-requests").glob("*.json"))
    serialized = path.read_text()
    saved = json.loads(serialized)["request"]
    assert "auth-secret" not in serialized
    assert "api-secret" not in serialized
    assert "cookie-secret" not in serialized
    assert "refresh-secret" not in serialized
    assert saved["headers"]["Authorization"] == "[REDACTED]"
    assert saved["headers"]["X-API-Key"] == "[REDACTED]"
    assert saved["headers"]["Cookie"] == "[REDACTED]"
    assert saved["refresh_token"] == "[REDACTED]"
    assert saved["messages"][0]["content"] == "prompt-secret"
    assert saved["body"] == "body-secret"
    assert saved["cache_control"] == {"type": "ephemeral"}
    assert saved["tools"][0]["function"]["parameters"]["properties"]["api_key"] == {
        "type": "string"
    }


def test_capture_is_non_fatal_by_default(monkeypatch):
    from agent import physical_attempt_diagnostics, relay_llm

    _enable(monkeypatch)
    monkeypatch.setattr(capture, "_persist", lambda payload: (_ for _ in ()).throw(OSError("disk")))
    monkeypatch.setattr(physical_attempt_diagnostics, "start_attempt", lambda *args, **kwargs: None)
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
        capture.capture_provider_request({"messages": []})


def test_atomic_private_capture_files(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)

    capture.capture_provider_request({"model": "m"})

    root = tmp_path / "debug" / "cache-requests"
    path = next(root.glob("*.json"))
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(root).st_mode & 0o777 == 0o700
    assert not list(root.glob(".tmp-*"))


def test_capture_records_each_physical_attempt(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics, relay_llm

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    monkeypatch.setattr(physical_attempt_diagnostics, "start_attempt", lambda *args, **kwargs: None)
    request = {"model": "m", "messages": [{"role": "user", "content": "x"}]}

    for retry_count in (0, 1):
        relay_llm._execute_attempt(
            dict(request),
            lambda final_request: relay_llm.capture_transport_request(final_request),
            name="provider",
            model_name="m",
            metadata={"api_request_id": "turn:api:0", "retry_count": retry_count},
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



def test_codex_direct_auxiliary_stream_captures_once(monkeypatch, tmp_path):
    from agent import auxiliary_client as auxiliary
    from agent import physical_attempt_diagnostics

    monkeypatch.setattr(capture, "get_hermes_home", lambda: tmp_path)
    _enable(monkeypatch)
    monkeypatch.setattr(
        physical_attempt_diagnostics,
        "start_attempt",
        lambda *args, **kwargs: None,
    )
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
    monkeypatch.setattr(auxiliary, "_effective_provider_for_client", lambda *args: "openai-codex")
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
    assert saved["route"] == {
        "api_mode": "codex_responses",
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "route": base_url,
    }
    assert saved["physical_attempt"]["correlation"].startswith("aux-")
    assert saved["physical_attempt"] == {
        "attempt_id": f"{saved['physical_attempt']['correlation']}:attempt:0",
        "correlation": saved["physical_attempt"]["correlation"],
        "retry": 0,
    }
    assert saved["request"] == opened[0]


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
