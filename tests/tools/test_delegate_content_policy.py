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
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "custom", "default": "gpt-4o-mini", "base_url": url,
                  "context_length": 128000, "streaming": False},
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
        elif name == "handle_api_error" and event == "return":
            verdicts.append(value)
            settled.append(identity(frame.f_locals["agent"]))

    def send(client, request, **kwargs):
        assert request.url.host == "accepted.invalid", str(request.url)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, request=request, json={"object": "list", "data": [
                {"id": model, "object": "model", "context_length": 128000}
                for model in ("gpt-4o-mini", "gpt-4o")
            ]})
        assert request.url.path.endswith("/chat/completions")
        payload = json.loads(request.content)
        assert not payload.get("stream"), payload
        attempts.append((payload["model"], request.headers["authorization"]))
        assert len(attempts) <= 2, attempts
        if len(attempts) == 1:
            assert payload["probe_route"] == "accepted"
            return httpx.Response(status, request=request, headers={"x-should-retry": "false"},
                                  json={"error": {"message": body}})
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
    assert len(verdicts) == 1
    if profile == "standard":
        assert len(attempts) == 1, attempts
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
    else:
        # Outside the standard contract, retain the existing configured fallback.
        assert attempts == [(route["model"], "Bearer fixture-accepted-key"),
                            ("gpt-4o", "Bearer fixture-backup-key")]
        assert fallbacks
        assert result["results"][0]["status"] == "completed", result
        assert result["results"][0]["summary"] == "PUBLIC_PROBE_OK"
