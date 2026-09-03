"""Focused model-pool child-routing regressions."""

from __future__ import annotations

import json
import logging
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
        "api_key": "[REDACTED]",
        "fallback_chain": [
            {"provider": "openrouter", "model": "fb-one"},
            {"provider": "openrouter", "model": "fb-two"},
        ],
    },
    "test": {
        "provider": "custom",
        "model": "tiny-test",
        "base_url": "http://127.0.0.1:8/v1",
        "api_key": "[REDACTED]",
        "fallback_chain": [{"provider": "custom", "model": "tiny-backup"}],
    },
}
CFG = {"model": "gpt-5.6-terra", "provider": "openai-codex", "model_pool": STANDARD_POOL}


def parent():
    p = MagicMock()
    p.base_url = "https://api.openai.com/v1"
    p.api_key = "[REDACTED]"
    p.provider = "openai-codex"
    p.api_mode = "codex_responses"
    p.model = "gpt-5.6-terra"
    p.platform = "cli"
    p.providers_allowed = p.providers_ignored = p.providers_order = p.provider_sort = None
    p._session_db = None
    p._delegate_depth = 0
    p._active_children = []
    p._active_children_lock = threading.Lock()
    p._print_fn = p.tool_progress_callback = p.thinking_callback = None
    p._fallback_chain = [{"provider": "openai-codex", "model": "gpt-5.6-terra"}]
    p.enabled_toolsets, p.disabled_toolsets = ["terminal"], []
    return p


def captured_child(**kwargs):
    child = MagicMock()
    child.run_conversation.return_value = {"final_response": "ok", "completed": True, "api_calls": 1}
    child.close = MagicMock()
    LAST_KWARGS.clear()
    LAST_KWARGS.update(kwargs)
    return child


LAST_KWARGS = {}


def spawn(**kwargs):
    with patch("tools.delegate_tool._load_config", return_value=CFG), patch(
        "run_agent.AIAgent", side_effect=captured_child
    ):
        return json.loads(delegate_task(parent_agent=parent(), **kwargs))


class TestModelPoolSchema:
    def test_static_schema_has_top_level_and_task_profile(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "model_profile" in props
        assert "model_profile" in props["tasks"]["items"]["properties"]

    def test_dynamic_schema_enum_tracks_pool(self):
        with patch("tools.delegate_tool._load_config", return_value=CFG):
            props = _build_dynamic_schema_overrides()["parameters"]["properties"]
        assert set(props["model_profile"]["enum"]) == {"standard", "test"}
        assert set(props["tasks"]["items"]["properties"]["model_profile"]["enum"]) == {
            "standard",
            "test",
        }


class TestModelPoolRouting:
    def test_omitted_profile_uses_standard_and_profile_chain(self):
        payload = spawn(goal="do work")
        assert "error" not in payload
        assert LAST_KWARGS["model"] == "deepseek-v4-flash"
        assert LAST_KWARGS["provider"] == "opencode-go"
        assert [x["model"] for x in LAST_KWARGS["fallback_model"]] == [
            "fb-one",
            "fb-two",
        ]

    def test_requested_profile_beats_global_and_task_profile_beats_top_level(self):
        seen = []

        def capture(**kwargs):
            seen.append(kwargs)
            return captured_child(**kwargs)

        tasks = [{"goal": "use test profile", "model_profile": "test"}, {"goal": "use standard"}]
        with patch("tools.delegate_tool._load_config", return_value=CFG), patch(
            "run_agent.AIAgent", side_effect=capture
        ):
            payload = json.loads(
                delegate_task(tasks=tasks, model_profile="standard", parent_agent=parent())
            )
        assert "error" not in payload
        assert [x["model"] for x in seen] == ["tiny-test", "deepseek-v4-flash"]
        assert seen[0]["fallback_model"][0]["model"] == "tiny-backup"

    def test_profile_overrides_request_options(self):
        profile = {
            **STANDARD_POOL["standard"],
            "request_overrides": {"reasoning_effort": "low"},
            "max_output_tokens": 123,
        }
        cfg = {"model_pool": {"standard": profile}}
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "run_agent.AIAgent", side_effect=captured_child
        ):
            payload = json.loads(
                delegate_task(goal="do work", model_profile="standard", parent_agent=parent())
            )
        assert "error" not in payload
        assert LAST_KWARGS["fast_mode_overrides"] == {"reasoning_effort": "low"}
        assert LAST_KWARGS["max_tokens"] == 123

    def test_unknown_profile_fails_before_child_construction(self):
        with patch("tools.delegate_tool._load_config", return_value=CFG), patch(
            "run_agent.AIAgent", side_effect=AssertionError("must not construct")
        ):
            payload = json.loads(
                delegate_task(goal="do work", model_profile="missing", parent_agent=parent())
            )
        assert "error" in payload and "missing" in payload["error"]

    def test_missing_standard_fails_closed_even_when_pool_order_changes(self):
        cfg = {"model_pool": {"test": STANDARD_POOL["test"], "fast": STANDARD_POOL["standard"]}}
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "run_agent.AIAgent", side_effect=AssertionError("must not construct")
        ):
            payload = json.loads(delegate_task(goal="do work", parent_agent=parent()))
        assert "error" in payload and "standard" in payload["error"].lower()

    def test_explicit_profile_missing_standard_fails_before_child_construction(self):
        cfg = {"model_pool": {"fast": STANDARD_POOL["test"]}}
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "run_agent.AIAgent", side_effect=AssertionError("must not construct")
        ):
            payload = json.loads(
                delegate_task(goal="do work", model_profile="fast", parent_agent=parent())
            )
        assert "error" in payload and "standard" in payload["error"].lower()

    def test_per_task_profile_missing_standard_fails_before_child_construction(self):
        cfg = {"model_pool": {"fast": STANDARD_POOL["test"]}}
        tasks = [
            {"goal": "use fast profile", "model_profile": "fast"},
            {"goal": "use the inherited profile"},
        ]
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "run_agent.AIAgent", side_effect=AssertionError("must not construct")
        ):
            payload = json.loads(
                delegate_task(
                    tasks=tasks,
                    model_profile="fast",
                    parent_agent=parent(),
                )
            )
        assert "error" in payload and "standard" in payload["error"].lower()


class TestBuildChildFallbackOverride:
    def test_profile_chain_beats_parent_chain(self):
        p = parent()
        chain = [{"provider": "openrouter", "model": "fb-one"}]
        with patch("run_agent.AIAgent", return_value=MagicMock()) as agent:
            _build_child_agent(
                task_index=0,
                goal="g",
                context=None,
                toolsets=None,
                model="deepseek-v4-flash",
                max_iterations=5,
                parent_agent=p,
                task_count=1,
                override_fallback_chain=chain,
            )
        assert agent.call_args.kwargs["fallback_model"] == chain


class TestDispatchForwarding:
    def test_dispatch_forwards_profile(self):
        from run_agent import AIAgent

        forwarded = {}
        with patch("tools.delegate_tool.delegate_task", side_effect=lambda **kw: forwarded.update(kw) or "{}"):
            agent = MagicMock(spec=AIAgent)
            agent._delegate_depth = 0
            AIAgent._dispatch_delegate_task(agent, {"goal": "x", "model_profile": "standard"})
        assert forwarded["model_profile"] == "standard"


class TestRoutingLog:
    def test_route_log_contains_safe_selection_details(self, caplog):
        with caplog.at_level(logging.INFO, logger="tools.delegate_tool"):
            spawn(goal="do work")
        text = caplog.text.lower()
        assert "standard" in text and "deepseek-v4-flash" in text and "fb-one" in text
        assert "[redacted]" not in text
