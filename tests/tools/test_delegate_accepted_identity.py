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
from tools import delegate_tool
from tools.delegate_tool import delegate_task


def _deny_network(*_args, **_kwargs):
    raise AssertionError("Unexpected real network access")


def _chat(request, model, content, finish):
    return httpx.Response(200, request=request, json={
        "id": "fixture", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
    })


def _ok(request, model):
    return _chat(request, model, "PUBLIC_PROBE_OK", "stop")


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
    primary_url = "https://primary.invalid/v1"
    accepted_url = "https://accepted.invalid/v1"
    route = {
        "provider": "named-primary", "model": "gpt-4o-mini", "base_url": primary_url,
        "api_key": "fixture-primary-key", "api_mode": "chat_completions",
        "request_overrides": {"extra_body": {"probe_route": "accepted"}},
        "fallback_chain": [{"provider": "named-fallback", "model": "gpt-4o", "base_url": accepted_url,
                            "api_key": "fixture-backup-key"}],
    }
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "custom", "default": "gpt-4o-mini", "base_url": primary_url, "context_length": 128000, "streaming": False},
        "providers": {
            "named-primary": {"name": "Primary", "base_url": primary_url, "api_key": "fixture-primary-key"},
            "named-fallback": {"name": "Fallback", "base_url": accepted_url, "api_key": "fixture-backup-key"},
        },
        "agent": {"max_retries": 2, "auto_recovery_cycles": 0},
        "compression": {"enabled": False},
        "delegation": {"max_spawn_depth": 1, "model_pool": {"standard": route, "review": route}},
    })
    attempts = []
    children = []
    original_build = delegate_tool._build_child_preserving_parent_tools

    def capture_child(**kwargs):
        child = original_build(**kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", capture_child)

    def send(_client, request, **_kwargs):
        assert request.url.host in {"primary.invalid", "accepted.invalid"}
        if request.url.path.endswith("/models"):
            model = "gpt-4o-mini" if request.url.host == "primary.invalid" else "gpt-4o"
            return httpx.Response(200, request=request, json={"object": "list", "data": [
                {"id": model, "object": "model"}
            ]})
        payload = json.loads(request.content)
        attempts.append((request.url.host, payload["model"], request.headers["authorization"]))
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
        ("primary.invalid", "gpt-4o-mini", "Bearer fixture-primary-key"),
        ("accepted.invalid", "gpt-4o", "Bearer fixture-backup-key"),
    ]
    assert entry["status"] == "completed"
    assert (entry["model"], entry["provider"]) == ("gpt-4o", "named-fallback")
    accepted_route = children[0]._delegate_accepted_llm_route
    assert accepted_route[:3] == ("gpt-4o", "named-fallback", "named-fallback")
    assert accepted_route[3] is not None and accepted_route[3].rstrip("/") == accepted_url
    assert accepted_url not in json.dumps(entry)


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
        _delegate_selected_llm_route=("gpt-selected", "openai-codex", "openai-codex", None),
        session_prompt_tokens=0, session_completion_tokens=0, session_estimated_cost_usd=0.0,
        session_reasoning_tokens=0, _credential_pool=None,
    )

    def boom(self):
        raise RuntimeError("workspace seed failed")

    monkeypatch.setattr("tools.delegate_tool_child_run._ChildRun.seed_workspace", boom)
    entry = _run_single_child(0, "goal", child, SimpleNamespace(_current_task_id=None, _delegate_depth=0, _touch_activity=lambda *_a: None))
    assert entry["status"] == "error"
    assert (entry["model"], entry["provider"]) == (None, None)


@pytest.mark.parametrize("ingress", ["goal", "task", "top"])
def test_public_outer_exception_preserves_selected_route_authority(tmp_path, monkeypatch, ingress):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    selected_url = "https://selected.invalid/v1"
    route = {
        "provider": "named-selected", "model": "gpt-selected", "base_url": selected_url,
        "api_key": "fixture-selected-key", "api_mode": "chat_completions",
    }
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "custom", "default": "gpt-parent", "base_url": "https://parent.invalid/v1",
                  "context_length": 128000, "streaming": False},
        "providers": {"named-selected": {"name": "Selected", "base_url": selected_url,
                                           "api_key": "fixture-selected-key", "models": {"gpt-selected": {}}}},
        "agent": {"max_retries": 0, "auto_recovery_cycles": 0},
        "compression": {"enabled": False},
        "delegation": {"max_spawn_depth": 1, "model_pool": {"standard": route, "fast": route}},
    })
    parent = _parent(tmp_path)
    children = []
    original_agent = AIAgent

    def capture_child(**kwargs):
        child = original_agent(**kwargs)
        children.append(child)
        return child

    monkeypatch.setattr("run_agent.AIAgent", capture_child)

    def fail_during_workspace_seed(self):
        child = self.child
        child.model, child.provider = "fallback-model", "fallback-provider"
        child.requested_provider, child.base_url = "fallback-owner", "https://fallback.invalid/v1"
        raise RuntimeError("workspace seed failed")

    monkeypatch.setattr("tools.delegate_tool_child_run._ChildRun.seed_workspace", fail_during_workspace_seed)
    kwargs = {"goal": "exercise selected route on outer failure", "parent_agent": parent}
    if ingress == "task":
        kwargs["tasks"] = [{"goal": kwargs.pop("goal"), "model_profile": "fast"}]
    elif ingress == "top":
        kwargs["model_profile"] = "fast"
    try:
        result = json.loads(delegate_task(**kwargs))
    finally:
        parent.close()
    entry = result["results"][0]
    assert entry["status"] == "error"
    assert children[0]._delegate_selected_llm_route == (
        "gpt-selected", "custom", "named-selected", selected_url,
    )
    assert (entry["model"], entry["provider"], entry["requested_provider"]) == (
        "gpt-selected", "custom", "named-selected",
    )
    assert "fallback-model" not in json.dumps(entry)
    assert "fallback-owner" not in json.dumps(entry)
    assert selected_url not in json.dumps(entry)


_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


def _scripted_pool(tmp_path, monkeypatch, url, route):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "custom", "default": "gpt-4o-mini", "base_url": url,
                  "context_length": 128000, "streaming": False},
        "agent": {"max_retries": 2, "auto_recovery_cycles": 0},
        "compression": {"enabled": False},
        "delegation": {"max_spawn_depth": 1, "model_pool": {"standard": route, "review": route}},
    })


@pytest.mark.parametrize("ingress", ["task", "top", "default"])
@pytest.mark.parametrize("terminal", ["length", "empty", "thinking", "invalid", "refusal", "failed", "success"])
def test_empty_length_response_keeps_last_usable_fallback(tmp_path, monkeypatch, ingress, terminal):
    """Restoring primary only replaces fallback identity after a usable response."""
    url = "https://accepted.invalid/v1"
    route = {
        "provider": "custom", "model": "gpt-4o-mini", "base_url": url,
        "api_key": "fixture-primary-key", "api_mode": "chat_completions",
        "fallback_chain": [{"provider": "custom", "model": "gpt-4o", "base_url": url,
                            "api_key": "fixture-backup-key"}],
    }
    _scripted_pool(tmp_path, monkeypatch, url, route)
    attempts = []

    def send(_client, request, **_kwargs):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, request=request, json={"object": "list", "data": [
                {"id": model, "object": "model"} for model in ("gpt-4o-mini", "gpt-4o")
            ]})
        payload = json.loads(request.content)
        attempts.append((payload["model"], request.headers["authorization"]))
        n = len(attempts)
        assert n <= 5, attempts
        if n == 1 or (terminal == "invalid" and n == 3) or (terminal in {"length", "empty", "thinking"} and n == 4):
            return httpx.Response(200, request=request, json={
                "id": "fixture", "object": "chat.completion", "created": 0,
                "model": payload["model"], "choices": [],
            })
        if n == 2:
            # Usable text MUST violate the schema so the correction restores primary.
            return _chat(request, "unrelated-body-model", "PUBLIC_OK", "stop")
        if n == 3 and terminal not in {"invalid", "failed"}:
            content, finish = {
                "length": ("", "length"),
                "empty": (None, "stop"),
                "thinking": ("<think>private reasoning</think>", "stop"),
                "refusal": ("Policy refused.", "content_filter"),
                "success": ('{"city":"Oslo"}', "stop"),
            }[terminal]
            return _chat(request, "unrelated-body-model", content, finish)
        return httpx.Response(401, request=request, headers={"x-should-retry": "false"},
                              json={"error": {"message": "Invalid API key"}})

    monkeypatch.setattr(httpx.Client, "send", send)
    parent = _parent(tmp_path)
    try:
        kwargs = {"goal": "Return the city.", "output_schema": _SCHEMA,
                  "parent_agent": parent, "max_iterations": 4}
        if ingress == "task":
            kwargs["tasks"] = [{"goal": kwargs.pop("goal"), "model_profile": "review"}]
        elif ingress == "top":
            kwargs["model_profile"] = "review"
        result = json.loads(delegate_task(**kwargs))
    finally:
        parent.close()
    primary = ("gpt-4o-mini", "Bearer fixture-primary-key")
    fallback = ("gpt-4o", "Bearer fixture-backup-key")
    expected = [primary, fallback, primary]
    if terminal in {"length", "empty", "thinking"}:
        expected += [primary, fallback]
    elif terminal != "success" and not (terminal == "refusal" and ingress == "default"):
        # The standard tier ends on refusal; it must not issue another fallback.
        expected += [fallback]
    assert attempts == expected
    entry = result["results"][0]
    assert entry["schema_valid"] is (terminal == "success")
    assert (entry["model"], entry["provider"]) == (
        "gpt-4o-mini" if terminal == "success" else "gpt-4o", "custom",
    )


def test_tool_call_without_text_is_usable(tmp_path, monkeypatch):
    url = "https://accepted.invalid/v1"
    route = {
        "provider": "custom", "model": "custom/gpt-4o-mini", "base_url": url,
        "api_key": "fixture-primary-key", "api_mode": "chat_completions",
    }
    _scripted_pool(tmp_path, monkeypatch, url, route)
    sent = []

    def send(_client, request, **_kwargs):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, request=request, json={"object": "list", "data": [
                {"id": "gpt-4o-mini", "object": "model"}
            ]})
        sent.append((json.loads(request.content)["model"], request.headers["authorization"]))
        if len(sent) == 1:
            return httpx.Response(200, request=request, json={
                "id": "fixture", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-1", "type": "function",
                    "function": {"name": "delegate_task", "arguments": "{}"},
                }]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            })
        return httpx.Response(401, request=request, headers={"x-should-retry": "false"},
                              json={"error": {"message": "Invalid API key"}})

    monkeypatch.setattr(httpx.Client, "send", send)
    parent = _parent(tmp_path)
    try:
        result = json.loads(delegate_task(
            tasks=[{"goal": "Reply PUBLIC_PROBE_OK.", "model_profile": "review"}],
            parent_agent=parent, max_iterations=3,
        ))
    finally:
        parent.close()
    assert sent == [("gpt-4o-mini", "Bearer fixture-primary-key")] * 2
    assert result["results"][0]["status"] == "failed"
    assert (result["results"][0]["model"], result["results"][0]["provider"]) == ("gpt-4o-mini", "custom")


@pytest.mark.parametrize("model_only", [False, True])
@pytest.mark.parametrize("failure", ["outer", "http"])
@pytest.mark.parametrize("ingress", ["goal", "task"])
def test_inherited_route_survives_outer_failure(tmp_path, monkeypatch, model_only, failure, ingress):
    """Freeze effective legacy routes before both workspace and HTTP failures."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "custom", "default": "gpt-4o-mini",
                  "base_url": "https://parent.invalid/v1", "context_length": 128000, "streaming": False},
        "agent": {"max_retries": 0, "auto_recovery_cycles": 0},
        "compression": {"enabled": False},
        "delegation": {"max_spawn_depth": 1, **({"model": "gpt-4o"} if model_only else {})},
    })
    sends = []

    def send(_client, request, **_kwargs):
        assert request.url.host == "parent.invalid"
        if request.url.path.endswith("/models"):
            return httpx.Response(200, request=request, json={"data": [
                {"id": "gpt-4o-mini"}, {"id": "gpt-4o"},
            ]})
        sends.append((json.loads(request.content)["model"], request.headers["authorization"]))
        return httpx.Response(401, request=request, headers={"x-should-retry": "false"},
                              json={"error": {"message": "Invalid API key"}})

    monkeypatch.setattr(httpx.Client, "send", send)
    if failure == "outer":
        def boom(self):
            raise RuntimeError("workspace seed failed")
        monkeypatch.setattr("tools.delegate_tool_child_run._ChildRun.seed_workspace", boom)
    parent = _parent(tmp_path)
    try:
        kwargs = {"goal": "inherit", "parent_agent": parent, "max_iterations": 2}
        if ingress == "task":
            kwargs["tasks"] = [{"goal": kwargs.pop("goal")}]
        entry = json.loads(delegate_task(**kwargs))["results"][0]
    finally:
        parent.close()
    expected = "gpt-4o" if model_only else "gpt-4o-mini"
    assert sends == ([] if failure == "outer" else [(expected, "Bearer fixture-parent-key")])
    assert entry["status"] == ("error" if failure == "outer" else "failed")
    assert (entry["model"], entry["provider"]) == (expected, "custom")
