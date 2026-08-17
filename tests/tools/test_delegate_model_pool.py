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
from tools.registry import registry


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


_BANNED_SCHEMA_WORDS = (
    "provider",
    "model",
    "fallback_chain",
    "fallback",
    "primary",
)


def _profile_schema_texts(fn):
    props = fn.get("parameters", {}).get("properties", {})
    texts = [props.get("model_profile", {}).get("description") or ""]
    nested = (
        (props.get("tasks") or {})
        .get("items", {})
        .get("properties", {})
        .get("model_profile", {})
    )
    texts.append(nested.get("description") or "")
    return "\n".join(texts).lower().replace("model_profile", "")


class TestOmittedProfileUsesPoolDefault:
    def test_omitted_profile_uses_standard_not_global_pin(self):
        payload, kwargs = _child_kwargs()
        assert "error" not in payload
        assert kwargs["model"] == "deepseek-v4-flash"
        assert kwargs["provider"] == "opencode-go"
        assert [e["model"] for e in kwargs["fallback_model"]] == [
            "fb-one",
            "fb-two",
            "fb-three",
            "fb-four",
        ]

    def test_omitted_profile_uses_first_pool_key_when_no_standard(self):
        cfg = {
            "max_iterations": 10,
            "model": "gpt-5.6-terra",
            "provider": "openai-codex",
            "model_pool": {
                "test": STANDARD_POOL["test"],
                "fast": STANDARD_POOL["standard"],
            },
        }
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

        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "run_agent.AIAgent", side_effect=_capture
        ):
            raw = delegate_task(goal="do work", parent_agent=parent)
        payload = json.loads(raw)
        assert "error" not in payload
        assert captured["model"] == "tiny-test"
        assert captured["provider"] == "custom"


class TestSchemaIsTierNamesOnly:
    def test_static_and_dynamic_schema_omit_model_provider_fallback(self):
        static = _profile_schema_texts(DELEGATE_TASK_SCHEMA)
        with patch("tools.delegate_tool._load_config", return_value=PINNED_CFG):
            overrides = _build_dynamic_schema_overrides()
        dynamic = _profile_schema_texts(overrides)
        for blob in (static, dynamic):
            for banned in _BANNED_SCHEMA_WORDS:
                assert banned not in blob, banned

    def test_unknown_profile_error_lists_tier_names_only(self):
        payload, _ = _child_kwargs(model_profile="does-not-exist")
        err = payload["error"].lower().replace("model_profile", "")
        assert "does-not-exist" in err
        assert "standard" in err
        assert "test" in err
        for banned in ("provider", "model", "fallback", "gpt-5.6", "deepseek"):
            assert banned not in err, banned


class _LiveCfg:
    def __init__(self, data):
        self.data = data

    def __call__(self):
        return self.data


class TestLiveConfigReread:
    def test_get_definitions_sees_mutated_pool_keys(self):
        loader = _LiveCfg(dict(PINNED_CFG))
        with patch("tools.delegate_tool._load_config", side_effect=loader):
            a = registry.get_definitions({"delegate_task"}, quiet=True)
            loader.data = {
                "max_iterations": 10,
                "model": "gpt-5.6-terra",
                "provider": "openai-codex",
                "model_pool": {
                    "fast": {
                        "provider": "custom",
                        "model": "fast-model",
                        "fallback_chain": [
                            {"provider": "custom", "model": "fast-fb"}
                        ],
                    },
                    "standard": STANDARD_POOL["standard"],
                },
            }
            b = registry.get_definitions({"delegate_task"}, quiet=True)
        enum_a = a[0]["function"]["parameters"]["properties"]["model_profile"]["enum"]
        enum_b = b[0]["function"]["parameters"]["properties"]["model_profile"]["enum"]
        assert set(enum_a) == {"standard", "test"}
        assert set(enum_b) == {"fast", "standard"}

    def test_second_spawn_sees_mutated_pool_chain(self):
        loader = _LiveCfg(dict(PINNED_CFG))
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

        with patch("tools.delegate_tool._load_config", side_effect=loader), patch(
            "run_agent.AIAgent", side_effect=_capture
        ):
            r1 = json.loads(delegate_task(goal="first", parent_agent=parent))
            loader.data = {
                "max_iterations": 10,
                "model": "gpt-5.6-terra",
                "provider": "openai-codex",
                "model_pool": {
                    "standard": {
                        "provider": "opencode-go",
                        "model": "deepseek-v4-flash",
                        "base_url": "http://127.0.0.1:9/v1",
                        "api_key": "profile-key",
                        "fallback_chain": [
                            {"provider": "openrouter", "model": "new-fb"}
                        ],
                    }
                },
            }
            r2 = json.loads(delegate_task(goal="second", parent_agent=parent))
        assert "error" not in r1 and "error" not in r2
        assert [e["model"] for e in seen[0]["fallback_model"]] == [
            "fb-one",
            "fb-two",
            "fb-three",
            "fb-four",
        ]
        assert [e["model"] for e in seen[1]["fallback_model"]] == ["new-fb"]
