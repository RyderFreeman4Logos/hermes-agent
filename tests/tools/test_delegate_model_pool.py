"""delegate_task model_profile / delegation.model_pool (issue #117).

A requested profile pins the child's primary route and may override the
routing owner's fallback policy. Global delegation.model must not win, and a
pinned child must never borrow the parent's fallback chain.
"""

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
import pytest
import yaml

from tools.registry import registry


@pytest.fixture(autouse=True)
def owned_named_provider(tmp_path, monkeypatch):
    """The tier's named identity is independently declared, not borrowed from its key."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"providers": {"fixture-opencode-go": {
        "base_url": "http://127.0.0.1:9/v1", "api_key": "fixture-opencode-owned-key",
    }}}))


STANDARD_POOL = {
    "standard": {
        "provider": "fixture-opencode-go",
        "model": "deepseek-v4-flash",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": "profile-key",
        "fallback_chain": [
            {"provider": "openrouter", "model": "fb-one", "api_key": "sk-or-test"},
            {"provider": "openrouter", "model": "fb-two", "api_key": "sk-or-test"},
            {"provider": "openrouter", "model": "fb-three", "api_key": "sk-or-test"},
            {"provider": "openrouter", "model": "fb-four", "api_key": "sk-or-test"},
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
        assert kwargs["provider"] == "custom"
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

    def test_profile_fallback_drops_unauthenticated_cloud_hop(self, monkeypatch):
        """A hop cannot borrow another provider's ambient key; its scoped key is accepted."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-owned-by-openrouter")
        profile = dict(STANDARD_POOL["standard"])
        profile["fallback_chain"] = [
            {"provider": "xai", "model": "grok-unauth"},
            {"provider": "openrouter", "model": "or-scoped", "key_env": "PROFILE_OR_KEY"},
        ]
        monkeypatch.setenv("PROFILE_OR_KEY", "sk-or-profile-owned")
        cfg = dict(PINNED_CFG)
        cfg["model_pool"] = {"standard": profile, "test": STANDARD_POOL["test"]}
        parent = _parent()
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            child = MagicMock()
            child.run_conversation.return_value = {"final_response": "ok", "completed": True, "api_calls": 1}
            return child

        with patch("tools.delegate_tool._load_config", return_value=cfg), patch("run_agent.AIAgent", side_effect=capture):
            raw = delegate_task(goal="do work", parent_agent=parent, model_profile="standard")
        payload = json.loads(raw)
        assert "error" not in payload
        chain = captured["fallback_model"]
        assert [entry["model"] for entry in chain] == ["or-scoped"]
        assert chain[0]["api_key"] == "sk-or-profile-owned"

    def test_profile_fallback_presence_excludes_global_policy_and_keeps_owned_chain(self):
        route = {key: value for key, value in STANDARD_POOL["standard"].items()
                 if key != "fallback_chain"}
        declared = [{"provider": "openrouter", "model": "owner-backup"}]
        for profile_chain, owner_chain, expected in (
            (None, declared, []),
            (None, None, []),
            ([], declared, []),
            ([{"provider": "openrouter", "model": "profile-backup", "api_key": "sk-or-test"}], declared, ["profile-backup"]),
        ):
            profile = dict(route)
            if profile_chain is not None:
                profile["fallback_chain"] = profile_chain
            cfg = {"model_pool": {"standard": profile}, "fallback_providers": owner_chain}
            captured = {}

            def capture(**kwargs):
                captured.update(kwargs)
                child = MagicMock()
                child.run_conversation.return_value = {"final_response": "ok", "completed": True, "api_calls": 1}
                return child

            with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
                "run_agent.AIAgent", side_effect=capture
            ):
                payload = json.loads(delegate_task(goal="verify fallback policy", parent_agent=_parent()))
            assert "error" not in payload
            assert (captured["model"], captured["api_key"]) == ("deepseek-v4-flash", "profile-key")
            assert [entry["model"] for entry in captured["fallback_model"] or []] == expected

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

    def test_explicit_profile_missing_standard_fails_before_child_construction(self):
        cfg = {"model_pool": {"fast": STANDARD_POOL["test"]}}
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "run_agent.AIAgent", side_effect=AssertionError("must not construct")
        ):
            payload = json.loads(
                delegate_task(goal="do work", model_profile="fast", parent_agent=_parent())
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
                    parent_agent=_parent(),
                )
            )
        assert "error" in payload and "standard" in payload["error"].lower()

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
        profile_chain = [{"provider": "openrouter", "model": "fb-one", "api_key": "sk-or-test"}]
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
    def test_advertised_schema_agrees_with_omit_resolution(self):
        with patch("tools.delegate_tool._load_config", return_value=PINNED_CFG):
            overrides = _build_dynamic_schema_overrides()
        advertised = "\n".join(
            [
                overrides.get("description") or "",
                overrides["parameters"]["properties"]["model_profile"]["description"],
                overrides["parameters"]["properties"]["tasks"]["items"]["properties"][
                    "model_profile"
                ]["description"],
            ]
        ).lower()
        assert "inherit" not in advertised
        assert "standard" in advertised
        assert "fail closed" in advertised.replace("-", " ")
        assert "first" not in advertised

        payload, kwargs = _child_kwargs()
        assert "error" not in payload
        assert kwargs["model"] == "deepseek-v4-flash"
        assert kwargs["provider"] == "custom"

    def test_omitted_profile_uses_standard_not_global_pin(self):
        payload, kwargs = _child_kwargs()
        assert "error" not in payload
        assert kwargs["model"] == "deepseek-v4-flash"
        assert kwargs["provider"] == "custom"
        assert [e["model"] for e in kwargs["fallback_model"]] == [
            "fb-one",
            "fb-two",
            "fb-three",
            "fb-four",
        ]

    def test_omitted_profile_fails_when_standard_missing(self):
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
        assert "error" in payload
        err = payload["error"].lower()
        assert "standard" in err
        assert captured == {}

    def test_missing_standard_is_order_independent(self):
        def _run(pool):
            cfg = {
                "max_iterations": 10,
                "model": "gpt-5.6-terra",
                "provider": "openai-codex",
                "model_pool": pool,
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
                payload = json.loads(delegate_task(goal="do work", parent_agent=parent))
            return payload, captured

        a, captured_a = _run(
            {"test": STANDARD_POOL["test"], "fast": STANDARD_POOL["standard"]}
        )
        b, captured_b = _run(
            {"fast": STANDARD_POOL["standard"], "test": STANDARD_POOL["test"]}
        )
        assert "error" in a and "error" in b
        assert captured_a == {} and captured_b == {}
        assert "standard" in a["error"].lower()
        assert "standard" in b["error"].lower()

    def test_resolved_route_is_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="tools.delegate_tool"):
            payload, kwargs = _child_kwargs()
        assert "error" not in payload
        text = caplog.text.lower()
        assert "standard" in text
        assert kwargs["model"].lower() in text
        assert kwargs["provider"].lower() in text
        assert "fb-one" in text
        assert "reasoning" in text


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
                        "base_url": "http://127.0.0.1:8/v1",
                        "api_key": "test-key",
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
                        "provider": "fixture-opencode-go",
                        "model": "deepseek-v4-flash",
                        "base_url": "http://127.0.0.1:9/v1",
                        "api_key": "profile-key",
                        "fallback_chain": [
                            {"provider": "openrouter", "model": "new-fb", "api_key": "sk-or-test"}
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
