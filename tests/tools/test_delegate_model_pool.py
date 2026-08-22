"""delegate_task model_profile / delegation.model_pool (issue #117).

A requested profile must pin the child to that profile's primary +
fallback_chain. Global delegation.model must not win, and the child must
not inherit an empty/parent chain that retries the same exhausted model.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_agent,
    _build_dynamic_schema_overrides,
    delegate_task,
)


STANDARD_POOL = {
    "standard": {
        "provider": "opencode-go",
        "model": "deepseek-v4-flash",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": "profile-key",
        "fallback_chain": [
            {"provider": "openrouter", "model": "fb-one"},
            {"provider": "openrouter", "model": "fb-two"},
            {"provider": "openrouter", "model": "fb-three"},
            {"provider": "openrouter", "model": "fb-four"},
        ],
    },
    "test": {
        "provider": "custom",
        "model": "tiny-test",
        "base_url": "http://127.0.0.1:8/v1",
        "api_key": "test-key",
        "fallback_chain": [{"provider": "custom", "model": "tiny-backup"}],
    },
}

PINNED_CFG = {
    "max_iterations": 10,
    "model": "gpt-5.6-terra",
    "provider": "openai-codex",
    "model_pool": STANDARD_POOL,
}


def _parent():
    parent = MagicMock()
    parent.base_url = "https://api.openai.com/v1"
    parent.api_key = "parent-key"
    parent.provider = "openai-codex"
    parent.api_mode = "codex_responses"
    parent.model = "gpt-5.6-terra"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._fallback_chain = [
        {"provider": "openai-codex", "model": "gpt-5.6-terra"}
    ]
    parent.enabled_toolsets = ["terminal"]
    parent.disabled_toolsets = []
    return parent


def _child_kwargs(goal="do work", **delegate_kw):
    parent = _parent()
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        child = MagicMock()
        child.run_conversation.return_value = {
            "final_response": "ok",
            "completed": True,
            "api_calls": 1,
        }
        child.close = MagicMock()
        return child

    with patch("tools.delegate_tool._load_config", return_value=PINNED_CFG), patch(
        "run_agent.AIAgent", side_effect=_capture
    ):
        raw = delegate_task(goal=goal, parent_agent=parent, **delegate_kw)
    return json.loads(raw), captured


class TestModelProfileSchema:
    def test_schema_exposes_model_profile_top_level_and_per_task(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "model_profile" in props
        assert "model_profile" in props["tasks"]["items"]["properties"]

    def test_dynamic_schema_enum_is_configured_pool_keys(self):
        with patch("tools.delegate_tool._load_config", return_value=PINNED_CFG):
            overrides = _build_dynamic_schema_overrides()
        props = overrides["parameters"]["properties"]
        assert set(props["model_profile"]["enum"]) == {"standard", "test"}
        task_enum = props["tasks"]["items"]["properties"]["model_profile"]["enum"]
        assert set(task_enum) == {"standard", "test"}


class TestModelProfileResolution:
    def test_profile_primary_beats_global_delegation_model(self):
        payload, kwargs = _child_kwargs(model_profile="standard")
        assert "error" not in payload
        assert kwargs["model"] == "deepseek-v4-flash"
        assert kwargs["provider"] == "opencode-go"
        assert kwargs["model"] != "gpt-5.6-terra"

    def test_profile_fallback_chain_not_parent_or_global_pin(self):
        _, kwargs = _child_kwargs(model_profile="standard")
        chain = kwargs["fallback_model"]
        assert isinstance(chain, list)
        assert [e["model"] for e in chain] == [
            "fb-one",
            "fb-two",
            "fb-three",
            "fb-four",
        ]
        assert all(e["model"] != "gpt-5.6-terra" for e in chain)

    def test_unknown_profile_fails_closed(self):
        payload, kwargs = _child_kwargs(model_profile="does-not-exist")
        assert "error" in payload
        assert "does-not-exist" in payload["error"]
        assert kwargs == {}

    def test_per_task_profile_beats_top_level(self):
        parent = _parent()
        seen = []

        def _capture(**kwargs):
            seen.append(kwargs)
            child = MagicMock()
            child.run_conversation.return_value = {
                "final_response": "ok",
                "completed": True,
                "api_calls": 1,
            }
            child.close = MagicMock()
            return child

        with patch("tools.delegate_tool._load_config", return_value=PINNED_CFG), patch(
            "run_agent.AIAgent", side_effect=_capture
        ):
            raw = delegate_task(
                tasks=[
                    {"goal": "use test profile", "model_profile": "test"},
                    {"goal": "use standard profile"},
                ],
                model_profile="standard",
                parent_agent=parent,
            )
        payload = json.loads(raw)
        assert "error" not in payload
        assert len(seen) == 2
        assert seen[0]["model"] == "tiny-test"
        assert seen[1]["model"] == "deepseek-v4-flash"
        assert seen[0]["fallback_model"][0]["model"] == "tiny-backup"

    def test_dispatch_forwards_model_profile(self):
        from run_agent import AIAgent

        forwarded = {}

        def _fake_delegate(**kwargs):
            forwarded.update(kwargs)
            return json.dumps({"ok": True})

        agent = MagicMock(spec=AIAgent)
        agent._delegate_depth = 0
        with patch("tools.delegate_tool.delegate_task", _fake_delegate):
            AIAgent._dispatch_delegate_task(
                agent, {"goal": "x", "model_profile": "standard"}
            )
        assert forwarded.get("model_profile") == "standard"


class TestBuildChildOverrideChain:
    def test_override_fallback_chain_beats_parent_inherit(self):
        parent = _parent()
        profile_chain = [{"provider": "openrouter", "model": "fb-one"}]
        with patch("run_agent.AIAgent") as mock_agent:
            mock_agent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="g",
                context=None,
                toolsets=None,
                model="deepseek-v4-flash",
                max_iterations=5,
                parent_agent=parent,
                task_count=1,
                override_fallback_chain=profile_chain,
            )
        _, kwargs = mock_agent.call_args
        assert kwargs["fallback_model"] == profile_chain
        assert kwargs["model"] == "deepseek-v4-flash"
