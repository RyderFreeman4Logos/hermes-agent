"""Public accepted-vs-selected identity for delegated children.

The HTTP send is the existing scripted SDK fixture. Admission, the child,
and the conversation loop stay real. A scripted send is not provider acceptance.
"""

import json
import socket
from types import SimpleNamespace

import httpx
import pytest

from hermes_cli.config import atomic_config_write
from run_agent import AIAgent
from tools.delegate_tool import delegate_task


def _deny_network(*_args, **_kwargs):
    raise AssertionError("Unexpected real network access")


def _ok(request, model):
    return httpx.Response(200, request=request, json={
        "id": "fixture", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "PUBLIC_PROBE_OK"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
    })


def _parent(tmp_path):
    return AIAgent(
        provider="custom", model="gpt-4o-mini", base_url="https://parent.invalid/v1",
        api_key="fixture-parent-key", enabled_toolsets=[], quiet_mode=True,
        skip_context_files=True, skip_memory=True, skip_background_review=True,
        save_trajectories=False, max_iterations=3, session_db=None,
    )


@pytest.mark.parametrize("ingress", ["task", "top", "default"])
def test_public_fallback_reports_actual_accepted_route(tmp_path, monkeypatch, ingress):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    url = "https://accepted.invalid/v1"
    route = {
        "provider": "custom", "model": "gpt-4o-mini", "base_url": url,
        "api_key": "fixture-primary-key", "api_mode": "chat_completions",
        "request_overrides": {"extra_body": {"probe_route": "accepted"}},
        "fallback_chain": [{"provider": "custom", "model": "gpt-4o", "base_url": url,
                            "api_key": "fixture-backup-key"}],
    }
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "custom", "default": "gpt-4o-mini", "base_url": url, "context_length": 128000, "streaming": False},
        "agent": {"max_retries": 2, "auto_recovery_cycles": 0},
        "compression": {"enabled": False},
        "delegation": {"max_spawn_depth": 1, "model_pool": {"standard": route, "review": route}},
    })
    attempts = []

    def send(_client, request, **_kwargs):
        assert request.url.host == "accepted.invalid"
        if request.url.path.endswith("/models"):
            return httpx.Response(200, request=request, json={"object": "list", "data": [
                {"id": model, "object": "model"} for model in ("gpt-4o-mini", "gpt-4o")
            ]})
        payload = json.loads(request.content)
        attempts.append((payload["model"], request.headers["authorization"]))
        if len(attempts) == 1:
            return httpx.Response(
                429, request=request, headers={"x-should-retry": "false"},
                json={"error": {"message": "rate limit"}},
            )
        return _ok(request, payload["model"])

    monkeypatch.setattr(httpx.Client, "send", send)
    parent = _parent(tmp_path)
    kwargs = {"goal": "Reply PUBLIC_PROBE_OK.", "parent_agent": parent, "max_iterations": 3}
    if ingress == "task":
        kwargs["tasks"] = [{"goal": kwargs.pop("goal"), "model_profile": "review"}]
    elif ingress == "top":
        kwargs["model_profile"] = "review"
    try:
        result = json.loads(delegate_task(**kwargs))
    finally:
        parent.close()
    entry = result["results"][0]
    assert attempts == [
        ("gpt-4o-mini", "Bearer fixture-primary-key"),
        ("gpt-4o", "Bearer fixture-backup-key"),
    ]
    assert entry["status"] == "completed"
    assert (entry["model"], entry["provider"]) == ("gpt-4o", "custom")


def test_named_alias_keeps_runtime_provider_canonical(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    url = "https://accepted.invalid/v1"
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "custom", "default": "gpt-4o-mini", "base_url": url, "context_length": 128000, "streaming": False},
        "providers": {"named-review": {"name": "Named Review", "base_url": url, "api_key": "fixture-named-key",
                                       "models": {"gpt-4o-mini": {}}}},
        "agent": {"max_retries": 0, "auto_recovery_cycles": 0},
        "compression": {"enabled": False},
        "delegation": {"max_spawn_depth": 1, "model_pool": {"standard": {
            "provider": "named-review", "model": "gpt-4o-mini", "base_url": url,
            "api_key": "fixture-named-key", "api_mode": "chat_completions",
        }}},
    })

    def send(_client, request, **_kwargs):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, request=request, json={"object": "list", "data": [
                {"id": "gpt-4o-mini", "object": "model"}
            ]})
        return _ok(request, "gpt-4o-mini")

    monkeypatch.setattr(httpx.Client, "send", send)
    parent = _parent(tmp_path)
    try:
        result = json.loads(delegate_task(
            tasks=[{"goal": "Reply PUBLIC_PROBE_OK.", "model_profile": "standard"}],
            parent_agent=parent, max_iterations=2,
        ))
    finally:
        parent.close()
    entry = result["results"][0]
    assert entry["status"] == "completed"
    assert (entry["provider"], entry.get("requested_provider")) == ("custom", "named-review")


def test_failed_schema_names_invalid_contract():
    from tools.delegate_tool import _run_single_child
    from tests.tools.test_delegate_output_schema import ADDRESS_SCHEMA, _StubChild, _StubParent

    child = _StubChild(["not json"])
    child._delegate_output_schema = ADDRESS_SCHEMA

    def failed(user_message, task_id=None, **_kwargs):
        child.calls.append(user_message)
        return {
            "final_response": "not json", "completed": False, "failed": True,
            "error": "provider failed", "api_calls": 1, "messages": [],
        }

    child.run_conversation = failed
    entry = _run_single_child(0, "produce the address", child, _StubParent())
    assert entry["status"] == "failed"
    assert entry["schema_valid"] is False
    assert "output_schema" in entry["error"]
    assert "provider failed" not in entry["error"]


def test_codex_fabricated_error_does_not_report_selected_as_accepted(monkeypatch):
    from tools.delegate_tool import _run_single_child

    child = SimpleNamespace(
        api_mode="codex_app_server", model="gpt-selected", provider="openai-codex",
        requested_provider="openai-codex", tool_progress_callback=None,
        _delegate_role="leaf", _delegate_depth=1, _subagent_id=None,
        _delegate_selected_llm_route=("gpt-selected", "openai-codex"),
        session_prompt_tokens=0, session_completion_tokens=0, session_estimated_cost_usd=0.0,
        session_reasoning_tokens=0, _credential_pool=None,
    )

    def boom(self):
        raise RuntimeError("workspace seed failed")

    monkeypatch.setattr("tools.delegate_tool_child_run._ChildRun.seed_workspace", boom)
    entry = _run_single_child(0, "goal", child, SimpleNamespace(_current_task_id=None, _delegate_depth=0, _touch_activity=lambda *_a: None))
    assert entry["status"] == "error"
    assert (entry["model"], entry["provider"]) == (None, None)
