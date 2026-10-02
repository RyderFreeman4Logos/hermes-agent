"""Standard-child policy denials are terminal through public delegation (#161).

Only outbound HTTP is scripted; admission, child, SDK and recovery remain real.
"""

import json
import socket
import sys
import threading
from copy import deepcopy

import httpx
import pytest

from agent.error_classifier import _CONTENT_POLICY_BLOCKED_PATTERNS


@pytest.mark.parametrize("profile", ["standard", "review"])
@pytest.mark.parametrize("status,body,entry", [
    *[(400, token, "direct") for token in _CONTENT_POLICY_BLOCKED_PATTERNS],
    (401, "content_filter: token expired", "registry"),
    (503, "content_filter: no auth available", "registry"),
])
def test_public_child_policy_terminal(tmp_path, monkeypatch, profile, status, body, entry):
    _public_child(tmp_path, monkeypatch, profile, status, body, entry)


@pytest.mark.parametrize("profile", ["standard", "review"])
@pytest.mark.parametrize("case,streaming", [
    ("chat_filter", False), ("chat_filter", True), ("chat_refusal", False),
    ("responses_failed", True), ("responses_failed_output", True), ("responses_filter", True),
    ("chat_partial", True), ("responses_refusal", True), ("responses_refusal_delta", True),
    ("chat_plain", False), ("responses_plain", True), ("chat_mixed", False),
    ("chat_empty", False), ("chat_length", False), ("chat_malformed", False),
    ("responses_mixed", True),
])
def test_public_child_response_policy(tmp_path, monkeypatch, profile, case, streaming):
    _public_child(tmp_path, monkeypatch, profile, 200, "", "direct", case, streaming)



def _response(request, payload, case):
    """Real SDK wire shapes, including policy events after a delivered delta."""
    text = "I cannot help with that request." if case.endswith("plain") else "PUBLIC_PROBE_OK"
    if request.url.path.endswith("/responses"):
        status, error, incomplete = "completed", None, None
        content = [{"type": "output_text", "text": text, "annotations": []}]
        if case == "responses_refusal":
            content = [{"type": "refusal", "refusal": "fixture safety refusal"}]
        elif case == "responses_refusal_delta":
            # Compatible backends stream the refusal and omit output_item.done.
            content = [{"type": "refusal", "refusal": "fixture safety refusal"}]
        elif case == "responses_filter":
            status, content, incomplete = "incomplete", [], {"reason": "content_filter"}
        elif case in {"responses_failed", "responses_failed_output"}:
            status = "failed"
            content = content if case.endswith("output") else []
            error = {"code": "content_filter", "message": "Request rejected: content_filter"}
        if case == "responses_mixed":
            content.append({"type": "refusal", "refusal": "fixture safety refusal"})
        item = {"type": "message", "id": "msg_fixture", "role": "assistant", "status": "completed", "content": content}
        response = {"id": "resp_fixture", "object": "response", "created_at": 0, "model": payload["model"],
                    "status": status, "output": [item] if content else [], "error": error,
                    "incomplete_details": incomplete, "usage": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}}
        events = []
        if case == "responses_refusal_delta":
            events.append({"type": "response.refusal.delta", "item_id": "msg_fixture", "output_index": 0,
                           "content_index": 0, "delta": "fixture safety refusal", "sequence_number": 1})
        elif content:
            events.append({"type": "response.output_item.done", "output_index": 0, "item": item})
        events.append({"type": "response." + status, "response": response})
        wire = "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events)
        return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"}, content=wire.encode())
    content, refusal, reason = text, None, "stop"
    if case in {"chat_filter", "chat_refusal"}:
        content, refusal = None, "fixture safety refusal"
        reason = "content_filter" if case == "chat_filter" else "stop"
    elif case == "chat_mixed":
        refusal = "fixture safety refusal"
    elif case == "chat_empty":
        content = None
    elif case == "chat_length":
        reason = "length"
    response = {"id": "fixture", "object": "chat.completion", "created": 0, "model": payload["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "refusal": refusal}, "finish_reason": reason}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}}
    if case == "chat_malformed":
        response["choices"] = []
    if payload.get("stream"):
        delta = {"role": "assistant", "content": content, "refusal": refusal}
        if case == "chat_partial":
            delta, reason = {"role": "assistant", "content": "Partial fixture answer."}, None
        chunk = {**response, "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
        end = json.dumps({"error": {"code": "content_filter", "message": "Request rejected: content_filter"}}) if case == "chat_partial" else "[DONE]"
        wire = "data: " + json.dumps(chunk) + "\n\ndata: " + end + "\n\n"
        return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"}, content=wire.encode())
    return httpx.Response(200, request=request, json=response)


def _public_child(tmp_path, monkeypatch, profile, status, body, entry, case="error", streaming=False):
    from hermes_cli.config import atomic_config_write
    from run_agent import AIAgent
    from tools.delegate_tool import delegate_task
    from tools.registry import registry

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def denied(*args, **kwargs):
        raise AssertionError("Unexpected real network access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    url = "https://accepted.invalid/v1"
    route = {
        "provider": "custom", "model": "gpt-4o-mini", "base_url": url,
        "api_key": "fixture-accepted-key", "api_mode": "chat_completions",
        "request_overrides": {"extra_body": {"probe_route": "accepted"}},
        "fallback_chain": [{"provider": "custom", "model": "gpt-4o", "base_url": url,
                            "api_key": "fixture-backup-key"}],
    }
    if case.startswith("responses_"):
        route["api_mode"] = "codex_responses"
        route["fallback_chain"][0]["api_mode"] = "codex_responses"
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "custom", "default": "gpt-4o-mini", "base_url": url,
                  "context_length": 128000, "streaming": streaming},
        "agent": {"max_retries": 2, "auto_recovery_cycles": 0},
        "compression": {"enabled": False},
        "delegation": {"model": "forbidden-global-model", "provider": "forbidden-global-provider",
                       "max_spawn_depth": 2, "child_timeout_seconds": 45,
                       "model_pool": {"standard": route, "review": route}},
    })
    children, accepted, settled, attempts, fallbacks, verdicts = [], [], [], [], [], []

    def identity(child):
        return (child.model, child.provider, child.requested_provider, child.base_url,
                child.api_mode, child.api_key, child.client, child._credential_pool,
                child._credential_pool_entry_id, deepcopy(child.request_overrides))

    def observe(frame, event, value):
        name = frame.f_code.co_name
        if name == "_build_children" and event == "return" and isinstance(value, tuple):
            children.extend(item[2] for item in value[0])
            accepted.extend(identity(child) for child in children)
        elif name == "try_activate_fallback" and event == "call":
            fallbacks.append(frame.f_locals["reason"])
        elif name in {"handle_api_error", "check_api_response"} and event == "return":
            verdicts.append(value)
            settled.append(identity(frame.f_locals["agent"]))

    def send(client, request, **kwargs):
        assert request.url.host == "accepted.invalid", str(request.url)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, request=request, json={"object": "list", "data": [
                {"id": model, "object": "model", "context_length": 128000}
                for model in ("gpt-4o-mini", "gpt-4o")
            ]})
        assert request.url.path.endswith("/responses" if case.startswith("responses_") else "/chat/completions")
        payload = json.loads(request.content)
        assert bool(payload.get("stream")) == (streaming or case.startswith("responses_")), payload
        attempts.append((payload["model"], request.headers["authorization"]))
        assert len(attempts) <= 2, attempts
        if len(attempts) == 1:
            assert payload["probe_route"] == "accepted"
            if case == "error":
                return httpx.Response(status, request=request, headers={"x-should-retry": "false"},
                                      json={"error": {"message": body}})
        if case != "error":
            return _response(request, payload, case if len(attempts) == 1 else "ok")
        return httpx.Response(200, request=request, json={
            "id": "fixture", "object": "chat.completion", "created": 0, "model": payload["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "PUBLIC_PROBE_OK"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        })

    monkeypatch.setattr(httpx.Client, "send", send)
    parent = AIAgent(provider="custom", model="gpt-4o-mini", base_url="https://parent.invalid/v1",
                     api_key="fixture-parent-key", enabled_toolsets=[], quiet_mode=True,
                     skip_context_files=True, skip_memory=True, skip_background_review=True,
                     save_trajectories=False, max_iterations=3)
    # The model-facing registry is synchronous only for an orchestrator child.
    parent._delegate_depth = 1
    parent._delegate_model_profile = "standard"
    previous, previous_thread = sys.getprofile(), threading.getprofile()
    try:
        sys.setprofile(observe)
        threading.setprofile(observe)
        args = {"goal": "Reply PUBLIC_PROBE_OK.", "model_profile": profile, "max_iterations": 3}
        if entry == "registry":
            raw = registry.dispatch("delegate_task", args, parent_agent=parent)
        else:
            raw = delegate_task(**args, background=False, parent_agent=parent)
        result = json.loads(raw) if isinstance(raw, str) else raw
    finally:
        sys.setprofile(previous)
        threading.setprofile(previous_thread)
        parent.close()

    assert len(children) == 1, result
    child = children[0]
    assert child._delegate_model_profile == profile
    assert attempts[0] == (route["model"], "Bearer " + route["api_key"])
    policy = case not in {"chat_plain", "responses_plain", "chat_mixed", "chat_empty", "chat_length", "chat_malformed", "responses_mixed"}
    if profile == "standard" and policy:
        assert len(attempts) == 1, attempts
        assert len(verdicts) == 1
        assert not fallbacks, fallbacks
        # Compare before normal child teardown closes the accepted client.
        assert settled == accepted
        assert not child._fallback_activated
        assert result["results"][0]["status"] == "failed", result
        assert "PUBLIC_PROBE_OK" not in result["results"][0]["summary"]
        assert verdicts[0].action == "return"
        assert verdicts[0].result["failure_reason"] == "content_policy_blocked"
        assert verdicts[0].result["failure_retryable"] is False
        assert verdicts[0].result["completed"] is False
    elif case in {"chat_plain", "responses_plain", "chat_mixed", "responses_refusal", "responses_refusal_delta", "responses_mixed"}:
        assert len(attempts) == 1
        assert not fallbacks
        assert result["results"][0]["status"] == "completed", result
    elif case in {"chat_empty", "chat_length", "responses_failed_output"}:
        assert attempts == [(route["model"], "Bearer fixture-accepted-key")] * 2
        assert not fallbacks
        assert result["results"][0]["status"] == "completed", result
    else:
        # Outside the standard contract, retain the existing configured fallback.
        assert attempts == [(route["model"], "Bearer fixture-accepted-key"),
                            ("gpt-4o", "Bearer fixture-backup-key")]
        assert fallbacks
        assert result["results"][0]["status"] == "completed", result
        assert result["results"][0]["summary"] == "PUBLIC_PROBE_OK"
