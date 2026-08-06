#!/usr/bin/env python3
"""Tests for per-subagent named model pool (model_profile).

Covers the model_pool feature added in
``feat(delegation): named model pool for per-subagent model selection``:

  1. Profile present  → child uses the profile's provider/model.
  2. Profile omitted  → falls back to the global delegation config.
  3. Per-task model_profile in a batch fans out across profiles.
  4. Profile-level reasoning_effort overrides the global level.
  5. Profile provider resolves through resolve_runtime_provider (custom:).
  6. Unknown profiles and malformed fallback chains fail closed.

Uses mock AIAgent instances — no real LLM calls.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import unittest
from unittest.mock import MagicMock, patch

import tools.delegate_tool as delegate_tool
from tools.delegate_tool import (
    _available_model_profile_names,
    _build_child_agent,
    _build_dynamic_schema_overrides,
    _resolve_delegation_credentials,
    _resolve_model_profile,
    delegate_task,
)


def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


# =========================================================================
# _resolve_model_profile helper
# =========================================================================


class TestResolveModelProfile(unittest.TestCase):
    """Unit tests for the _resolve_model_profile overlay helper."""

    def test_none_profile_returns_none(self):
        self.assertIsNone(_resolve_model_profile({}, None))
        self.assertIsNone(_resolve_model_profile({"model_pool": {}}, None))

    def test_empty_profile_returns_none(self):
        self.assertIsNone(_resolve_model_profile({"model_pool": {}}, "  "))

    def test_unknown_profile_returns_none(self):
        cfg = {"model_pool": {"fast": {"provider": "x", "model": "m"}}}
        self.assertIsNone(_resolve_model_profile(cfg, "nope"))

    def test_known_profile_overlays_fields(self):
        cfg = {
            "model": "global-model",
            "provider": "openrouter",
            "model_pool": {
                "smart": {
                    "provider": "custom:pm",
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "high",
                }
            },
        }
        merged = _resolve_model_profile(cfg, "smart")
        self.assertIsNotNone(merged)
        self.assertEqual(merged["provider"], "custom:pm")
        self.assertEqual(merged["model"], "gpt-5.6-terra")
        self.assertEqual(merged["reasoning_effort"], "high")

    def test_profile_empty_values_do_not_overlay(self):
        """Empty strings in the profile must not clobber global values."""
        cfg = {
            "model": "global-model",
            "provider": "openrouter",
            "model_pool": {"fast": {"provider": "custom:localrouter", "model": ""}},
        }
        merged = _resolve_model_profile(cfg, "fast")
        # provider overridden, but empty model does not clear the global model
        self.assertEqual(merged["provider"], "custom:localrouter")
        self.assertEqual(merged["model"], "global-model")

    def test_malformed_profile_field_types_fail_closed(self):
        invalid = {
            "provider": [],
            "model": 7,
            "base_url": {},
            "api_key": ["secret"],
            "api_mode": [],
            "reasoning_effort": {"effort": "high"},
        }
        for field, value in invalid.items():
            profile = {"model": "m"}
            profile[field] = value
            cfg = {"model_pool": {"bad": profile}}
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, field
            ):
                _resolve_model_profile(cfg, "bad")

    def test_invalid_profile_values_fail_closed(self):
        for field, value in (
            ("api_mode", "not-a-wire"),
            ("reasoning_effort", "instant"),
            ("base_url", "not-a-url"),
        ):
            cfg = {"model_pool": {"bad": {field: value, "model": "m"}}}
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, field
            ):
                _resolve_model_profile(cfg, "bad")


# =========================================================================
# _resolve_delegation_credentials with model_profile
# =========================================================================


class TestResolveDelegationCredentialsModelProfile(unittest.TestCase):
    """model_profile flows through _resolve_delegation_credentials."""

    def test_profile_provider_model_win_over_global(self):
        parent = _make_mock_parent()
        cfg = {
            "model": "global-model",
            "provider": "openrouter",
            "model_pool": {
                "fast": {
                    "provider": "custom:localrouter",
                    "model": "grok-4.5",
                    "base_url": "http://localhost:8000/v1",
                    "api_key": "fast-key",
                    "reasoning_effort": "high",
                }
            },
        }
        creds = _resolve_delegation_credentials(cfg, parent, model_profile="fast")
        self.assertEqual(creds["model"], "grok-4.5")
        # custom: with a base_url resolves to provider="custom"
        self.assertEqual(creds["provider"], "custom")
        self.assertEqual(creds["base_url"], "http://localhost:8000/v1")
        self.assertEqual(creds["api_key"], "fast-key")
        self.assertFalse(creds["inherit_parent_api_key"])
        self.assertEqual(creds["reasoning_effort"], "high")

    def test_unknown_explicit_profile_fails_closed(self):
        parent = _make_mock_parent()
        cfg = {
            "model": "global-model",
            "provider": "openrouter",
            "model_pool": {"fast": {"provider": "x", "model": "m"}},
        }
        with self.assertRaisesRegex(ValueError, "Unknown or invalid.*nope"):
            _resolve_delegation_credentials(cfg, parent, model_profile="nope")

    def test_unknown_default_profile_fails_closed(self):
        cfg = {
            "default_profile": "missing",
            "model_pool": {"fast": {"provider": "x", "model": "m"}},
        }
        with self.assertRaisesRegex(ValueError, "Unknown or invalid.*missing"):
            _resolve_delegation_credentials(cfg, _make_mock_parent())

    def test_no_profile_uses_global(self):
        parent = _make_mock_parent()
        cfg = {
            "model": "qwen2.5-coder",
            "provider": "openrouter",
            "base_url": "http://localhost:1234/v1",
            "api_key": "local-key",
        }
        creds = _resolve_delegation_credentials(cfg, parent, model_profile=None)
        self.assertEqual(creds["model"], "qwen2.5-coder")
        self.assertEqual(creds["provider"], "custom")
        self.assertEqual(creds["api_key"], "local-key")
        self.assertIsNone(creds["inherit_parent_api_key"])
        # No profile → no reasoning_effort surfaced
        self.assertIsNone(creds["reasoning_effort"])

    def test_profile_custom_provider_resolves_via_runtime(self):
        """A custom:<name> provider in a profile resolves through the runtime."""
        parent = _make_mock_parent()
        cfg = {
            "model_pool": {
                "smart": {
                    "provider": "custom:pm",
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "high",
                }
            },
        }
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider"
        ) as mock_rt:
            mock_rt.return_value = {
                "provider": "custom",
                "api_mode": "chat_completions",
                "base_url": "http://pm.example.com/v1",
                "api_key": "pm-key",
                "model": "gpt-5.6-terra",
            }
            creds = _resolve_delegation_credentials(cfg, parent, model_profile="smart")
        # resolve_runtime_provider was called with the profile's provider
        _, kwargs = mock_rt.call_args
        self.assertEqual(kwargs["requested"], "custom:pm")
        # When the runtime resolves to "custom", the configured provider
        # (custom:pm) is preserved verbatim — see _resolve_delegation_credentials.
        self.assertEqual(creds["provider"], "custom:pm")
        self.assertEqual(creds["api_key"], "pm-key")
        self.assertEqual(creds["reasoning_effort"], "high")

    def test_cross_endpoint_profile_never_inherits_parent_secret(self):
        parent = _make_mock_parent()
        parent.api_key = "parent-secret"
        cfg = {
            "api_key": "global-secret",
            "model_pool": {
                "other": {
                    "provider": "custom",
                    "model": "m",
                    "base_url": "https://other.invalid/v1",
                }
            }
        }
        with (
            patch("tools.delegate_tool._load_config", return_value=cfg),
            patch("run_agent.AIAgent") as MockAgent,
        ):
            result = delegate_task(
                goal="do not run", model_profile="other", parent_agent=parent
            )
        self.assertIn("refusing to inherit", result)
        MockAgent.assert_not_called()

    def test_same_endpoint_profile_can_inherit_parent_secret(self):
        parent = _make_mock_parent()
        cfg = {
            "model_pool": {
                "same": {
                    "provider": "custom",
                    "model": "m",
                    "base_url": parent.base_url,
                }
            }
        }
        creds = _resolve_delegation_credentials(cfg, parent, model_profile="same")
        self.assertIsNone(creds["api_key"])
        self.assertTrue(creds["inherit_parent_api_key"])


class TestBuildChildAgentCredentialInheritance(unittest.TestCase):
    @patch("tools.delegate_tool._load_config", return_value={})
    @patch("run_agent.AIAgent")
    def test_backend_boundary_fails_closed_but_compatible_calls_inherit(
        self, MockAgent, _mock_cfg
    ):
        MockAgent.return_value = MagicMock()
        parent = _make_mock_parent()
        parent.api_key = "parent-secret"

        common = {
            "task_index": 0,
            "goal": "test",
            "context": None,
            "toolsets": None,
            "model": None,
            "max_iterations": 1,
            "task_count": 1,
            "parent_agent": parent,
        }
        with self.assertRaisesRegex(ValueError, "different delegation backend"):
            _build_child_agent(
                **common,
                override_provider="custom",
                override_base_url="https://profile.invalid/v1",
            )
        MockAgent.assert_not_called()

        with self.assertRaisesRegex(ValueError, "different delegation backend"):
            _build_child_agent(
                **common,
                override_provider="custom",
                override_base_url="https://profile.invalid/v1",
                inherit_parent_api_key=True,
            )
        with self.assertRaisesRegex(ValueError, "different delegation backend"):
            _build_child_agent(
                **common,
                override_provider="custom",
                inherit_parent_api_key=True,
            )
        MockAgent.assert_not_called()

        with self.assertRaisesRegex(ValueError, "different delegation backend"):
            _build_child_agent(
                **common,
                override_base_url=parent.base_url,
                inherit_parent_api_key=False,
            )
        MockAgent.assert_not_called()

        _build_child_agent(
            **common,
            override_provider="custom",
            override_base_url=parent.base_url,
            inherit_parent_api_key=True,
        )
        self.assertEqual(MockAgent.call_args.kwargs["api_key"], "parent-secret")

        MockAgent.reset_mock()
        _build_child_agent(
            **common,
            override_provider="custom",
            override_base_url=parent.base_url,
        )
        self.assertEqual(MockAgent.call_args.kwargs["api_key"], "parent-secret")

        MockAgent.reset_mock()
        _build_child_agent(**common)
        self.assertEqual(MockAgent.call_args.kwargs["api_key"], "parent-secret")

        MockAgent.reset_mock()
        _build_child_agent(
            **common,
            override_provider="custom",
            override_base_url="https://profile.invalid/v1",
            override_api_key="",
            inherit_parent_api_key=False,
        )
        self.assertEqual(MockAgent.call_args.kwargs["api_key"], "")

    @patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], []))
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("tools.delegate_tool._load_config")
    @patch("run_agent.AIAgent")
    def test_all_named_profile_paths_recheck_inheritance_at_builder(
        self, MockAgent, mock_cfg, mock_creds, _mock_live
    ):
        cfg = {
            "max_spawn_depth": 3,
            "orchestrator_enabled": True,
            "model_pool": {"other": {"model": "m"}},
        }
        mock_cfg.return_value = cfg
        mock_creds.return_value = {
            "model": "m",
            "provider": "custom",
            "base_url": "https://other.invalid/v1",
            "api_key": None,
            "inherit_parent_api_key": True,
            "api_mode": "chat_completions",
        }
        cases = (
            ("sync", {"goal": "sync", "model_profile": "other"}, 0),
            (
                "background",
                {"goal": "background", "model_profile": "other", "background": True},
                0,
            ),
            (
                "batch",
                {
                    "tasks": [{"goal": "one"}, {"goal": "two"}],
                    "model_profile": "other",
                },
                0,
            ),
            (
                "orchestrator",
                {"goal": "nested", "model_profile": "other", "role": "orchestrator"},
                1,
            ),
        )

        for name, kwargs, depth in cases:
            with self.subTest(path=name), self.assertRaisesRegex(
                ValueError, "different delegation backend"
            ):
                delegate_task(parent_agent=_make_mock_parent(depth=depth), **kwargs)
        MockAgent.assert_not_called()


# =========================================================================
# reasoning_effort override in _build_child_agent
# =========================================================================


class TestBuildChildAgentProfileReasoningEffort(unittest.TestCase):
    """override_reasoning_effort (from a profile) beats the global level."""

    @patch("tools.delegate_tool._load_config")
    @patch("run_agent.AIAgent")
    def test_profile_effort_overrides_global(self, MockAgent, mock_cfg):
        mock_cfg.return_value = {"max_iterations": 50, "reasoning_effort": "low"}
        MockAgent.return_value = MagicMock()
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "xhigh"}

        _build_child_agent(
            task_index=0,
            goal="test",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=50,
            task_count=1,
            parent_agent=parent,
            override_reasoning_effort="high",
        )
        call_kwargs = MockAgent.call_args[1]
        # Profile effort "high" wins over global "low" and parent "xhigh"
        self.assertEqual(
            call_kwargs["reasoning_config"], {"enabled": True, "effort": "high"}
        )

    @patch("tools.delegate_tool._load_config")
    @patch("run_agent.AIAgent")
    def test_profile_effort_none_disables(self, MockAgent, mock_cfg):
        mock_cfg.return_value = {"max_iterations": 50}
        MockAgent.return_value = MagicMock()
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "high"}

        _build_child_agent(
            task_index=0,
            goal="test",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=50,
            task_count=1,
            parent_agent=parent,
            override_reasoning_effort="none",
        )
        call_kwargs = MockAgent.call_args[1]
        self.assertEqual(call_kwargs["reasoning_config"], {"enabled": False})

    @patch("tools.delegate_tool._load_config")
    @patch("run_agent.AIAgent")
    def test_no_profile_effort_falls_back_to_global(self, MockAgent, mock_cfg):
        """When override_reasoning_effort is None, global delegation level applies."""
        mock_cfg.return_value = {"max_iterations": 50, "reasoning_effort": "medium"}
        MockAgent.return_value = MagicMock()
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "xhigh"}

        _build_child_agent(
            task_index=0,
            goal="test",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=50,
            task_count=1,
            parent_agent=parent,
            # No override_reasoning_effort → global "medium" wins
        )
        call_kwargs = MockAgent.call_args[1]
        self.assertEqual(
            call_kwargs["reasoning_config"], {"enabled": True, "effort": "medium"}
        )


# =========================================================================
# delegate_task end-to-end with model_profile
# =========================================================================


class TestDelegateTaskModelProfile(unittest.TestCase):
    """Integration: model_profile reaches the child agent via delegate_task."""

    @patch("tools.delegate_tool._load_config")
    def test_unknown_profile_returns_tool_error(self, mock_cfg):
        mock_cfg.return_value = {
            "model_pool": {"fast": {"provider": "x", "model": "m"}}
        }
        result = delegate_task(
            goal="Do not run", model_profile="missing", parent_agent=_make_mock_parent()
        )
        self.assertIn("Unknown or invalid delegation model profile 'missing'", result)

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_top_level_profile_reaches_child(self, mock_creds, mock_cfg):
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model_pool": {"fast": {"model": "grok-4.5"}},
        }
        mock_creds.return_value = {
            "model": "grok-4.5",
            "provider": "custom",
            "base_url": "http://localhost:8000/v1",
            "api_key": "fast-key",
            "inherit_parent_api_key": False,
            "api_mode": "chat_completions",
            "reasoning_effort": "high",
        }
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            delegate_task(
                goal="Test profile routing", model_profile="fast", parent_agent=parent
            )

            # _resolve_delegation_credentials was called with the profile
            _, kwargs = mock_creds.call_args
            self.assertEqual(kwargs.get("model_profile"), "fast")
            # The child got the profile's model + reasoning effort
            _, child_kwargs = MockAgent.call_args
            self.assertEqual(child_kwargs["model"], "grok-4.5")
            self.assertEqual(child_kwargs["provider"], "custom")
            self.assertEqual(child_kwargs["api_key"], "fast-key")

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_per_task_profile_overrides_top_level(self, mock_creds, mock_cfg):
        """A per-task model_profile beats the top-level one in a batch."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model_pool": {
                "fast": {"model": "grok-4.5"},
                "smart": {"model": "gpt-5.6-terra"},
            },
        }

        fast_creds = {
            "model": "grok-4.5",
            "provider": "custom",
            "base_url": "http://localhost:8000/v1",
            "api_key": "fast-key",
            "inherit_parent_api_key": False,
            "api_mode": "chat_completions",
            "reasoning_effort": "high",
        }
        smart_creds = {
            "model": "gpt-5.6-terra",
            "provider": "custom",
            "base_url": "http://pm.example.com/v1",
            "api_key": "smart-key",
            "inherit_parent_api_key": False,
            "api_mode": "chat_completions",
            "reasoning_effort": "high",
        }
        mock_creds.side_effect = [fast_creds, smart_creds]

        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child

            delegate_task(
                tasks=[
                    {"goal": "task A", "model_profile": "fast"},
                    {"goal": "task B", "model_profile": "smart"},
                ],
                parent_agent=parent,
            )

            self.assertEqual(mock_creds.call_count, 2)
            # The first per-task call used "fast"
            first_task_kwargs = mock_creds.call_args_list[0][1]
            self.assertEqual(first_task_kwargs.get("model_profile"), "fast")
            # The second per-task call used "smart"
            second_task_kwargs = mock_creds.call_args_list[1][1]
            self.assertEqual(second_task_kwargs.get("model_profile"), "smart")
            # The last child built got the smart profile's model
            _, last_child_kwargs = MockAgent.call_args
            self.assertEqual(last_child_kwargs["model"], "gpt-5.6-terra")
            self.assertEqual(last_child_kwargs["api_key"], "smart-key")

    @patch("tools.delegate_tool._load_config")
    @patch("run_agent.AIAgent")
    def test_batch_prevalidates_all_profiles_before_spawning(self, MockAgent, mock_cfg):
        mock_cfg.return_value = {
            "model_pool": {
                "valid": {
                    "model": "m",
                    "base_url": "https://valid.invalid/v1",
                    "api_key": "valid-key",
                },
                "invalid": {"provider": ["not", "a", "string"], "model": "m"},
            }
        }
        result = delegate_task(
            tasks=[
                {"goal": "first", "model_profile": "valid"},
                {"goal": "second", "model_profile": "invalid"},
            ],
            parent_agent=_make_mock_parent(),
        )
        self.assertIn("provider must be a string", result)
        MockAgent.assert_not_called()


# =========================================================================
# Schema: dynamic enum of profile names
# =========================================================================


class TestModelProfileSchemaEnum(unittest.TestCase):
    """_build_dynamic_schema_overrides injects the profile enum."""

    def setUp(self):
        # Each test models a newly constructed session. Production deliberately
        # freezes this snapshot until a successful context compression.
        self._previous_snapshot = delegate_tool._MODEL_POOL_SCHEMA_NAMES
        delegate_tool._MODEL_POOL_SCHEMA_NAMES = None

    def tearDown(self):
        delegate_tool._MODEL_POOL_SCHEMA_NAMES = self._previous_snapshot

    @patch("tools.delegate_tool._load_config")
    def test_enum_lists_profiles_when_present(self, mock_cfg):
        mock_cfg.return_value = {
            "model_pool": {
                "fast": {"provider": "x", "model": "m"},
                "smart": {"provider": "y", "model": "n"},
            }
        }
        overrides = _build_dynamic_schema_overrides()
        prop = overrides["parameters"]["properties"]["model_profile"]
        self.assertEqual(prop["enum"], ["fast", "smart"])

    @patch("tools.delegate_tool._load_config")
    def test_no_enum_when_no_profiles(self, mock_cfg):
        mock_cfg.return_value = {"model_pool": {}}
        overrides = _build_dynamic_schema_overrides()
        prop = overrides["parameters"]["properties"]["model_profile"]
        self.assertNotIn("enum", prop)

    @patch("tools.delegate_tool._load_config")
    def test_available_names_sorted(self, mock_cfg):
        mock_cfg.return_value = {
            "model_pool": {
                "zeta": {"provider": "x", "model": "m"},
                "alpha": {"provider": "y", "model": "n"},
            }
        }
        self.assertEqual(_available_model_profile_names(), ["alpha", "zeta"])

    @patch("tools.delegate_tool._load_config")
    def test_model_profile_optional_when_pool_empty(self, mock_cfg):
        """Empty model_pool: model_profile is not required (upstream-compatible)."""
        mock_cfg.return_value = {"model_pool": {}}
        overrides = _build_dynamic_schema_overrides()
        self.assertNotIn("model_profile", overrides["parameters"].get("required", []))
        prop = overrides["parameters"]["properties"]["model_profile"]
        self.assertNotIn("enum", prop)
        self.assertIn("optional", prop["description"].lower())
        # The static schema itself also leaves it optional.
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA

        self.assertNotIn(
            "model_profile",
            DELEGATE_TASK_SCHEMA["parameters"].get("required", []),
        )

    @patch("tools.delegate_tool._load_config")
    def test_model_profile_optional_when_pool_nonempty(self, mock_cfg):
        """Non-empty pool: model_profile remains optional; enum is published."""
        mock_cfg.return_value = {
            "model_pool": {
                "fast": {"provider": "x", "model": "m"},
                "smart": {"provider": "y", "model": "n"},
            }
        }
        overrides = _build_dynamic_schema_overrides()
        self.assertNotIn("model_profile", overrides["parameters"].get("required", []))
        prop = overrides["parameters"]["properties"]["model_profile"]
        self.assertEqual(prop["enum"], ["fast", "smart"])
        self.assertIn("optional", prop["description"].lower())
        self.assertIn("default_profile", prop["description"])
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA

        self.assertNotIn(
            "model_profile",
            DELEGATE_TASK_SCHEMA["parameters"].get("required", []),
        )

    @patch("tools.delegate_tool._load_config")
    def test_schema_prose_matches_top_level_and_per_task_parameters(self, mock_cfg):
        mock_cfg.return_value = {"model_pool": {"fast": {"model": "m"}}}
        overrides = _build_dynamic_schema_overrides()
        properties = overrides["parameters"]["properties"]
        task_properties = properties["tasks"]["items"]["properties"]
        self.assertEqual(properties["model_profile"]["type"], "string")
        self.assertEqual(task_properties["model_profile"]["type"], "string")
        self.assertIn("model_profile", overrides["description"])
        self.assertIn("tasks[]", overrides["description"])


# =========================================================================
# Per-profile fallback_chain
# =========================================================================


class TestResolveModelProfileFallbackChain(unittest.TestCase):
    """_resolve_model_profile normalizes a profile's fallback_chain."""

    def test_fallback_chain_extracted_and_normalized(self):
        cfg = {
            "model_pool": {
                "fast": {
                    "provider": "custom:localrouter",
                    "model": "grok-4.5",
                    "reasoning_effort": "high",
                    "fallback_chain": [
                        {"provider": "openai-codex", "model": "gpt-5.6-terra"},
                        {"provider": "nous", "model": "hermes-4"},
                    ],
                }
            }
        }
        merged = _resolve_model_profile(cfg, "fast")
        self.assertIsNotNone(merged)
        chain = merged["_profile_fallback_chain"]
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0]["provider"], "openai-codex")
        self.assertEqual(chain[0]["model"], "gpt-5.6-terra")
        self.assertEqual(chain[1]["provider"], "nous")

    def test_malformed_fallback_chain_entries_fail_closed(self):
        for entry in (
            {"provider": "", "model": "no-provider"},
            {"provider": "no-model", "model": ""},
            "not-a-dict",
        ):
            cfg = {
                "model_pool": {
                    "fast": {
                        "provider": "x",
                        "model": "m",
                        "fallback_chain": [entry],
                    }
                }
            }
            with self.subTest(entry=entry), self.assertRaisesRegex(
                ValueError, r"fallback_chain\[0\]"
            ):
                _resolve_model_profile(cfg, "fast")

    def test_malformed_fallback_fields_fail_closed(self):
        invalid = {
            "provider": 123,
            "model": ["m"],
            "base_url": {},
            "api_key": ["secret"],
            "api_mode": "not-a-wire",
            "reasoning_effort": 7,
        }
        for field, value in invalid.items():
            entry = {"provider": "p", "model": "m", field: value}
            cfg = {
                "model_pool": {
                    "fast": {"model": "m", "fallback_chain": [entry]}
                }
            }
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, r"fallback_chain\[0\]"
            ):
                _resolve_model_profile(cfg, "fast")

        cfg = {
            "model_pool": {
                "fast": {
                    "model": "m",
                    "fallback_chain": [
                        {"provider": "p", "model": "m", "base_url": "not-a-url"}
                    ],
                }
            }
        }
        with self.assertRaisesRegex(ValueError, r"fallback_chain\[0\].*base_url"):
            _resolve_model_profile(cfg, "fast")

    def test_non_list_fallback_chain_fails_closed(self):
        cfg = {
            "model_pool": {
                "fast": {"provider": "x", "model": "m", "fallback_chain": {}}
            }
        }
        with self.assertRaisesRegex(ValueError, "fallback_chain must be a list"):
            _resolve_model_profile(cfg, "fast")

    def test_no_fallback_chain_is_absent(self):
        cfg = {"model_pool": {"fast": {"provider": "x", "model": "m"}}}
        merged = _resolve_model_profile(cfg, "fast")
        self.assertNotIn("_profile_fallback_chain", merged)


class TestResolveCredentialsFallbackChain(unittest.TestCase):
    """_resolve_delegation_credentials surfaces fallback_chain + default_profile."""

    def test_fallback_chain_returned_in_creds(self):
        cfg = {
            "base_url": "http://localhost:8000/v1",
            "model_pool": {
                "fast": {
                    "provider": "custom:localrouter",
                    "model": "grok-4.5",
                    "api_key": "fast-key",
                    "fallback_chain": [
                        {"provider": "openai-codex", "model": "gpt-5.6-terra"},
                    ],
                }
            },
        }
        parent = _make_mock_parent()
        creds = _resolve_delegation_credentials(cfg, parent, model_profile="fast")
        self.assertEqual(creds["fallback_chain"][0]["provider"], "openai-codex")
        self.assertEqual(creds["fallback_chain"][0]["model"], "gpt-5.6-terra")

    def test_missing_fallback_chain_keeps_inheritance_signal(self):
        cfg = {
            "base_url": "http://localhost:8000/v1",
            "model_pool": {
                "fast": {"provider": "x", "model": "m", "api_key": "fast-key"}
            },
        }
        parent = _make_mock_parent()
        creds = _resolve_delegation_credentials(cfg, parent, model_profile="fast")
        self.assertIsNone(creds["fallback_chain"])

    def test_default_profile_used_when_model_profile_none(self):
        """Internal path (model_profile=None) falls back to default_profile."""
        cfg = {
            "base_url": "http://localhost:8000/v1",
            "default_profile": "fast",
            "model_pool": {
                "fast": {
                    "provider": "custom:localrouter",
                    "model": "grok-4.5",
                    "api_key": "fast-key",
                    "fallback_chain": [
                        {"provider": "openai-codex", "model": "gpt-5.6-terra"},
                    ],
                }
            },
        }
        parent = _make_mock_parent()
        creds = _resolve_delegation_credentials(cfg, parent, model_profile=None)
        # default_profile "fast" was applied: its model + chain surfaced.
        self.assertEqual(creds["model"], "grok-4.5")
        self.assertEqual(creds["fallback_chain"][0]["provider"], "openai-codex")

    def test_no_default_profile_degrades_to_global(self):
        """When model_profile=None and no default_profile, global config used."""
        cfg = {
            "provider": "openrouter",
        }
        parent = _make_mock_parent()
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider"
        ) as mock_resolve:
            mock_resolve.return_value = {
                "provider": "custom",
                "model": "global-model",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "key",
                "api_mode": "chat_completions",
            }
            creds = _resolve_delegation_credentials(cfg, parent, model_profile=None)
        # No profile → no chain, global provider resolved.
        self.assertIsNone(creds["fallback_chain"])
        self.assertIsNotNone(creds["provider"])


class TestBuildChildAgentFallbackChain(unittest.TestCase):
    """_build_child_agent hands the per-profile chain to AIAgent's fallback_model."""

    @patch("tools.delegate_tool._load_config")
    @patch("run_agent.AIAgent")
    def test_profile_chain_overrides_parent_chain(self, MockAgent, mock_cfg):
        """A per-profile fallback_chain takes precedence over parent's chain."""
        mock_cfg.return_value = {"max_iterations": 50}
        MockAgent.return_value = MagicMock()
        parent = _make_mock_parent()
        # Parent has its own fallback chain; the profile's must win.
        parent._fallback_chain = [
            {"provider": "parent-provider", "model": "parent-model"}
        ]

        profile_chain = [
            {"provider": "openai-codex", "model": "gpt-5.6-terra"},
            {"provider": "nous", "model": "hermes-4"},
        ]
        _build_child_agent(
            task_index=0,
            goal="test",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=50,
            task_count=1,
            parent_agent=parent,
            override_fallback_chain=profile_chain,
        )
        call_kwargs = MockAgent.call_args[1]
        self.assertEqual(call_kwargs["fallback_model"], profile_chain)

    @patch("tools.delegate_tool._load_config")
    @patch("run_agent.AIAgent")
    def test_no_profile_chain_inherits_parent_chain(self, MockAgent, mock_cfg):
        """Without a profile chain, the parent's chain is inherited (back-compat)."""
        mock_cfg.return_value = {"max_iterations": 50}
        MockAgent.return_value = MagicMock()
        parent = _make_mock_parent()
        parent_chain = [
            {"provider": "parent-provider", "model": "parent-model"}
        ]
        parent._fallback_chain = parent_chain

        _build_child_agent(
            task_index=0,
            goal="test",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=50,
            task_count=1,
            parent_agent=parent,
            # No override_fallback_chain → parent chain inherited
        )
        call_kwargs = MockAgent.call_args[1]
        self.assertEqual(call_kwargs["fallback_model"], parent_chain)

    @patch("tools.delegate_tool._load_config")
    @patch("run_agent.AIAgent")
    def test_empty_profile_chain_disables_parent_chain(self, MockAgent, mock_cfg):
        mock_cfg.return_value = {"max_iterations": 50}
        MockAgent.return_value = MagicMock()
        parent = _make_mock_parent()
        parent._fallback_chain = [{"provider": "p", "model": "m"}]

        _build_child_agent(
            task_index=0,
            goal="test",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=50,
            task_count=1,
            parent_agent=parent,
            override_fallback_chain=[],
        )
        call_kwargs = MockAgent.call_args[1]
        self.assertEqual(call_kwargs["fallback_model"], [])


class TestDelegateTaskFallbackChainIntegration(unittest.TestCase):
    """delegate_task threads the profile's fallback_chain end-to-end."""

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("run_agent.AIAgent")
    def test_profile_chain_reaches_child_agent(
        self, MockAgent, mock_creds, mock_cfg
    ):
        """The profile's fallback_chain reaches _build_child_agent."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model_pool": {"fast": {"model": "grok-4.5"}},
        }
        chain = [{"provider": "openai-codex", "model": "gpt-5.6-terra"}]
        mock_creds.return_value = {
            "model": "grok-4.5",
            "provider": "custom",
            "base_url": "http://localhost:8000/v1",
            "api_key": "fast-key",
            "api_mode": "chat_completions",
            "reasoning_effort": "high",
            "fallback_chain": chain,
        }
        parent = _make_mock_parent(depth=0)

        mock_child = MagicMock()
        mock_child.run_conversation.return_value = {
            "final_response": "done",
            "completed": True,
            "api_calls": 1,
        }
        MockAgent.return_value = mock_child

        delegate_task(
            goal="Test fallback chain routing",
            model_profile="fast",
            parent_agent=parent,
        )
        _, child_kwargs = MockAgent.call_args
        self.assertEqual(child_kwargs["fallback_model"], chain)

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("run_agent.AIAgent")
    def test_per_task_profiles_have_distinct_chains(
        self, MockAgent, mock_creds, mock_cfg
    ):
        """Each per-task profile carries its own fallback_chain."""
        mock_cfg.return_value = {
            "max_iterations": 45,
            "model_pool": {
                "fast": {"model": "grok-4.5"},
                "smart": {"model": "gpt-5.6-terra"},
            },
        }
        fast_chain = [{"provider": "openai-codex", "model": "gpt-5.6-terra"}]
        smart_chain = [{"provider": "nous", "model": "hermes-4"}]
        mock_creds.side_effect = [
            {  # fast task
                "model": "grok-4.5",
                "provider": "custom",
                "base_url": "http://localhost:8000/v1",
                "api_key": "fast-key",
                "api_mode": "chat_completions",
                "reasoning_effort": "high",
                "fallback_chain": fast_chain,
            },
            {  # smart task
                "model": "gpt-5.6-terra",
                "provider": "custom",
                "base_url": "http://pm.example.com/v1",
                "api_key": "smart-key",
                "api_mode": "chat_completions",
                "reasoning_effort": "high",
                "fallback_chain": smart_chain,
            },
        ]
        parent = _make_mock_parent(depth=0)

        mock_child = MagicMock()
        mock_child.run_conversation.return_value = {
            "final_response": "done",
            "completed": True,
            "api_calls": 1,
        }
        MockAgent.return_value = mock_child

        delegate_task(
            tasks=[
                {"goal": "task A", "model_profile": "fast"},
                {"goal": "task B", "model_profile": "smart"},
            ],
            parent_agent=parent,
        )
        # Two children built; each got its own chain.
        all_kwargs = [c[1] for c in MockAgent.call_args_list]
        self.assertEqual(all_kwargs[0]["fallback_model"], fast_chain)
        self.assertEqual(all_kwargs[1]["fallback_model"], smart_chain)


def test_fresh_process_resolves_codex_primary_and_named_custom_fallback(tmp_path):
    """Configured model-pool routing resolves without network or credential output."""
    (tmp_path / "config.yaml").write_text(
        textwrap.dedent(
            """\
            providers:
              pm:
                base_url: https://pm.invalid/v1
                api_key: pm-test-secret
                api_mode: codex_responses
                default_model: gpt-5.4-mini
            delegation:
              default_profile: fast
              model_pool:
                fast:
                  provider: openai-codex
                  model: gpt-5.4-mini
                  base_url: https://chatgpt.com/backend-api/codex
                  api_key: codex-test-secret
                  api_mode: codex_responses
                  reasoning_effort: minimal
                  fallback_chain:
                    - provider: pm
                      model: gpt-5.4-mini
                      reasoning_effort: none
            """
        ),
        encoding="utf-8",
    )
    code = """
import socket
from types import SimpleNamespace
socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network"))
from tools.delegate_tool import _load_config, _resolve_delegation_credentials
from hermes_cli.runtime_provider import resolve_runtime_provider
creds = _resolve_delegation_credentials(_load_config(), SimpleNamespace(), "fast")
assert creds["provider"] == "openai-codex"
assert creds["model"] == "gpt-5.4-mini"
assert creds["api_mode"] == "codex_responses"
assert creds["reasoning_effort"] == "minimal"
assert creds["fallback_chain"] == [{"provider": "pm", "model": "gpt-5.4-mini", "reasoning_effort": "none"}]
fallback = resolve_runtime_provider(requested="pm", target_model="gpt-5.4-mini")
assert fallback["provider"] == "custom"
assert fallback["api_mode"] == "codex_responses"
assert fallback["base_url"] == "https://pm.invalid/v1"
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


if __name__ == "__main__":
    unittest.main()
