"""Public delegate route: schema enum, registry dispatch, then the real child builder.

The probe must not call ``delegate_task`` directly. Explicit ``review`` selects the
configured review model, provider, and fallback. An omitted profile selects the
configured standard route.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.registry import registry


def _parent():
    parent = MagicMock()
    parent.base_url = "https://parent.example/v1"
    parent.api_key = "parent-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "parent-model"
    parent.platform = "cli"
    parent.providers_allowed = ["Anthropic"]
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.request_overrides = {"extra_body": {"thinking": {"type": "disabled"}}}
    parent._fallback_chain = [{"provider": "openrouter", "model": "parent-fallback"}]
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


_POOL = {
    "delegation": {
        "model_pool": {
            "standard": {
                "provider": "deepseek",
                "model": "std-model",
                "base_url": "http://std/v1",
                "api_key": "std-key",
                "fallback_chain": [{"provider": "deepseek", "model": "std-fb-model"}],
            },
            "review": {
                "provider": "groq",
                "model": "rev-model",
                "base_url": "http://rev/v1",
                "api_key": "rev-key",
                "fallback_chain": [{"provider": "groq", "model": "rev-fb-model"}],
            },
        }
    }
}


def _schema():
    defs = registry.get_definitions({"delegate_task"})
    return next(item["function"] for item in defs if item["function"]["name"] == "delegate_task")


class TestPublicProfileDispatchProbe(unittest.TestCase):
    def test_schema_enum_lists_configured_profiles(self):
        with patch("hermes_cli.config.load_config_readonly", return_value=_POOL):
            schema = _schema()
        self.assertEqual(schema["parameters"]["properties"]["model_profile"]["enum"], ["standard", "review"])
        task_enum = schema["parameters"]["properties"]["tasks"]["items"]["properties"]["model_profile"]["enum"]
        self.assertEqual(task_enum, ["standard", "review"])

    def _dispatch(self, args):
        parent = _parent()
        with patch("hermes_cli.config.load_config_readonly", return_value=_POOL), patch(
            "run_agent.AIAgent"
        ) as built, patch(
            "tools.delegate_tool._run_single_child", return_value={"status": "completed"}
        ), patch(
            "tools.delegate_tool._run_batch",
            side_effect=lambda batch, background: json.dumps({"status": "built", "background": background}),
        ):
            built.return_value = MagicMock()
            raw = registry.dispatch("delegate_task", args, parent_agent=parent)
        self.assertIsInstance(raw, str)
        return json.loads(raw), built.call_args.kwargs

    def test_explicit_review_selects_configured_review_route(self):
        payload, kwargs = self._dispatch({
            "tasks": [{"goal": "review this", "model_profile": "review"}],
        })
        self.assertNotIn("error", payload)
        self.assertEqual(kwargs["model"], "rev-model")
        self.assertEqual(kwargs["base_url"], "http://rev/v1")
        self.assertEqual(kwargs["api_key"], "rev-key")
        self.assertEqual(kwargs["fallback_model"], [{"provider": "groq", "model": "rev-fb-model"}])
        self.assertNotEqual(kwargs["model"], "parent-model")
        self.assertNotEqual(kwargs["base_url"], "https://parent.example/v1")
        self.assertNotEqual(kwargs["fallback_model"], [{"provider": "openrouter", "model": "parent-fallback"}])

    def test_omitted_profile_selects_configured_standard_route(self):
        payload, kwargs = self._dispatch({"tasks": [{"goal": "default route"}]})
        self.assertNotIn("error", payload)
        self.assertEqual(kwargs["model"], "std-model")
        self.assertEqual(kwargs["base_url"], "http://std/v1")
        self.assertEqual(kwargs["api_key"], "std-key")
        self.assertEqual(kwargs["fallback_model"], [{"provider": "deepseek", "model": "std-fb-model"}])
        self.assertNotEqual(kwargs["model"], "parent-model")
        self.assertNotEqual(kwargs["base_url"], "https://parent.example/v1")


if __name__ == "__main__":
    unittest.main()
