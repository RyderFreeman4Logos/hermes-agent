from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tools.delegate_tool as delegate_tool


def _parent(*, depth: int = 1) -> MagicMock:
    parent = MagicMock()
    parent.base_url = "https://parent.invalid/v1"
    parent.api_key = "parent-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "parent-model"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.provider_require_parameters = False
    parent.provider_data_collection = ""
    parent.openrouter_min_coding_score = None
    parent.reasoning_config = {"enabled": True, "effort": "low"}
    parent._fallback_chain = [{"provider": "openrouter", "model": "parent-fallback"}]
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = []
    parent.disabled_toolsets = []
    parent.prefill_messages = None
    parent.request_overrides = {"parent_only": True}
    parent.max_tokens = None
    parent.session_id = "parent-session"
    return parent


def _creds(model: str, *, fallback_chain=None) -> dict:
    return {
        "model": model,
        "provider": "custom:pm",
        "base_url": "https://pm.invalid/v1",
        "api_key": f"{model}-key",
        "api_mode": "chat_completions",
        "request_overrides": {"profile": model},
        "max_output_tokens": None,
        "command": None,
        "args": [],
        "reasoning_effort": "high",
        "fallback_chain": fallback_chain,
    }


def test_named_profile_overlays_route_and_normalizes_fallback_chain(monkeypatch):
    cfg = {
        "provider": "openrouter",
        "model": "global-model",
        "model_pool": {
            "smart": {
                "provider": "custom:pm",
                "model": "gpt-smart",
                "api_mode": "chat_completions",
                "reasoning_effort": "high",
                "fallback_chain": [
                    {"provider": "openai-codex", "model": "gpt-fallback"}
                ],
            }
        },
    }
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "custom",
            "model": "gpt-smart",
            "base_url": "https://pm.invalid/v1",
            "api_key": "pm-key",
            "api_mode": "chat_completions",
            "request_overrides": {"profile": "smart"},
        },
    )

    creds = delegate_tool._resolve_delegation_credentials(
        cfg, _parent(), model_profile="smart"
    )

    assert creds["provider"] == "custom:pm"
    assert creds["model"] == "gpt-smart"
    assert creds["reasoning_effort"] == "high"
    assert creds["fallback_chain"] == [
        {"provider": "openai-codex", "model": "gpt-fallback"}
    ]
    assert creds["request_overrides"] == {"profile": "smart"}


def test_unknown_explicit_or_default_profile_fails_closed():
    cfg = {
        "default_profile": "missing",
        "model_pool": {"fast": {"provider": "openrouter", "model": "m"}},
    }
    with pytest.raises(ValueError, match="missing"):
        delegate_tool._resolve_delegation_credentials(cfg, _parent())
    with pytest.raises(ValueError, match="other"):
        delegate_tool._resolve_delegation_credentials(
            cfg, _parent(), model_profile="other"
        )


def test_cross_endpoint_profile_without_key_refuses_parent_secret():
    cfg = {
        "model_pool": {
            "other": {
                "provider": "custom",
                "model": "m",
                "base_url": "https://other.invalid/v1",
            }
        }
    }
    with pytest.raises(ValueError, match="refusing to inherit"):
        delegate_tool._resolve_delegation_credentials(
            cfg, _parent(), model_profile="other"
        )


@patch("tools.delegate_tool._load_config", return_value={})
@patch("run_agent.AIAgent")
def test_child_route_uses_profile_reasoning_and_fallback_without_parent_leak(
    agent_cls, _config
):
    child = MagicMock()
    child.session_id = "child"
    child._session_init_model_config = {}
    agent_cls.return_value = child
    parent = _parent()

    delegate_tool._build_child_agent(
        task_index=0,
        goal="route",
        context=None,
        toolsets=None,
        model="profile-model",
        max_iterations=2,
        task_count=1,
        parent_agent=parent,
        override_provider="custom:pm",
        override_base_url="https://pm.invalid/v1",
        override_api_key="profile-key",
        inherit_parent_api_key=False,
        override_api_mode="chat_completions",
        override_reasoning_effort="high",
        override_fallback_chain=[
            {"provider": "openai-codex", "model": "profile-fallback"}
        ],
    )

    kwargs = agent_cls.call_args.kwargs
    assert kwargs["api_key"] == "profile-key"
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert kwargs["fallback_model"] == [
        {"provider": "openai-codex", "model": "profile-fallback"}
    ]
    assert kwargs["request_overrides"] == {}


@patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], []))
@patch("tools.delegate_tool._run_single_child")
@patch("tools.delegate_tool._resolve_delegation_credentials")
@patch("tools.delegate_tool._load_config")
@patch("tools.delegate_tool._build_child_preserving_parent_tools")
def test_batch_freezes_each_task_profile_before_child_construction(
    build_child, load_config, resolve_creds, run_child, _live
):
    load_config.return_value = {
        "max_iterations": 10,
        "max_spawn_depth": 3,
        "model_pool": {
            "fast": {"provider": "custom:pm", "model": "fast-model"},
            "smart": {"provider": "custom:pm", "model": "smart-model"},
        },
    }
    resolve_creds.side_effect = [_creds("fast-model"), _creds("smart-model")]
    first = SimpleNamespace(
        session_id="child-1",
        _delegate_role="leaf",
        _session_init_model_config={},
        tool_progress_callback=None,
    )
    second = SimpleNamespace(
        session_id="child-2",
        _delegate_role="leaf",
        _session_init_model_config={},
        tool_progress_callback=None,
    )
    build_child.side_effect = [first, second]
    run_child.side_effect = [
        {"task_index": 0, "status": "completed", "summary": "one", "api_calls": 1},
        {"task_index": 1, "status": "completed", "summary": "two", "api_calls": 1},
    ]

    result = delegate_tool.delegate_task(
        tasks=[
            {
                "goal": "Review the first production routing path",
                "model_profile": "fast",
            },
            {
                "goal": "Review the second production routing path",
                "model_profile": "smart",
            },
        ],
        parent_agent=_parent(),
    )

    assert "error" not in json.loads(result)
    assert [call.kwargs["model_profile"] for call in resolve_creds.call_args_list] == [
        "fast",
        "smart",
    ]
    assert [call.kwargs["model"] for call in build_child.call_args_list] == [
        "fast-model",
        "smart-model",
    ]


def test_model_facing_dispatch_forwards_top_level_profile(monkeypatch):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._delegate_depth = 1
    captured = {}

    def fake_delegate(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("tools.delegate_tool.delegate_task", fake_delegate)

    assert agent._dispatch_delegate_task(
        {"goal": "inspect", "model_profile": "smart"}
    ) == "ok"
    assert captured["model_profile"] == "smart"


def test_model_profile_schema_is_optional_and_cache_stable(monkeypatch):
    config = {"model_pool": {"fast": {"model": "m"}}}
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: config)

    first = delegate_tool._build_dynamic_schema_overrides()
    config["model_pool"] = {"smart": {"model": "n"}}
    second = delegate_tool._build_dynamic_schema_overrides()

    for schema in (first, second):
        parameters = schema["parameters"]
        assert "model_profile" not in parameters.get("required", [])
        assert parameters["properties"]["model_profile"] == {
            "type": "string",
            "description": (
                "Optional configured delegation.model_pool profile. "
                "For batches, tasks[].model_profile overrides this value."
            ),
        }
        assert parameters["properties"]["tasks"]["items"]["properties"][
            "model_profile"
        ] == parameters["properties"]["model_profile"]
