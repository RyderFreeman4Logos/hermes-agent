"""Public child completion must not carry raw terminal output or a failed route.

The returned entry already drops those. ``subagent.complete`` has to match,
including when the child produced messages.
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _run_single_child


_SECRET = "personal-team-blocked:spending-limit"
_FAILED_ROUTE = "https://api.x.ai/v1"


def _child(events):
    child = MagicMock()
    child.session_id = "child-sess"
    child.model = "grok-4.6"
    child.provider = "xai-oauth"
    child.base_url = _FAILED_ROUTE
    child._credential_pool = None
    child.tool_progress_callback = lambda event, **kw: events.append((event, kw))
    child.run_conversation.return_value = {
        "final_response": "ok",
        "completed": True,
        "failed": False,
        "api_calls": 1,
        "messages": [
            {"role": "tool", "content": f"terminal: {_SECRET} at {_FAILED_ROUTE}"},
        ],
    }
    child.get_activity_summary.return_value = {}
    return child


class TestChildCompletionEventLeak(unittest.TestCase):
    def test_complete_event_omits_raw_tail_and_failed_route(self):
        parent = MagicMock()
        parent._touch_activity = lambda *_a, **_k: None
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent._current_task_id = None
        events = []
        with patch("tools.delegate_tool._HEARTBEAT_INTERVAL", 60):
            entry = _run_single_child(0, "goal", child=_child(events), parent_agent=parent)
        blob = repr(events)
        self.assertNotIn(_SECRET, blob)
        self.assertNotIn(_FAILED_ROUTE, blob)
        complete = [kw for event, kw in events if event == "subagent.complete"]
        self.assertEqual(len(complete), 1)
        self.assertEqual(complete[0].get("output_tail"), "")
        self.assertNotIn("provider", complete[0])
        self.assertNotIn("model", complete[0])
        # The returned entry is the other public surface; it stays clean too.
        self.assertNotIn(_SECRET, repr(entry))
        self.assertNotIn(_FAILED_ROUTE, repr(entry))

    def test_failed_child_hides_rejected_route_and_exception_text(self):
        parent = MagicMock()
        parent._touch_activity = lambda *_a, **_k: None
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent._current_task_id = None
        events = []
        child = MagicMock()
        child.session_id = "child-sess"
        child.model = "rejected-model"
        child.provider = "rejected-provider"
        child.base_url = _FAILED_ROUTE
        child._subagent_id = "sa-leak"
        child._credential_pool = None
        child._delegate_accepted_route = {"model": "rejected-model", "provider": "rejected-provider"}
        child.tool_progress_callback = lambda event, **kw: events.append((event, kw))
        child.run_conversation.side_effect = RuntimeError(f"provider said {_SECRET}")
        child.get_activity_summary.return_value = {}
        with patch("tools.delegate_tool._HEARTBEAT_INTERVAL", 60), \
             patch("tools.delegate_tool_child_run._register_subagent") as register:
            entry = _run_single_child(0, "goal", child=child, parent_agent=parent)
        blob = repr(events) + repr(entry) + repr(register.call_args)
        self.assertNotIn(_SECRET, blob)
        self.assertNotIn("rejected-model", blob)
        self.assertNotIn("rejected-provider", blob)
        self.assertNotIn(_FAILED_ROUTE, blob)
        self.assertEqual(entry["status"], "error")
        self.assertIn("Subagent raised", entry["error"])
        complete = [kw for event, kw in events if event == "subagent.complete"]
        self.assertEqual(complete[0].get("summary"), "Subagent raised RuntimeError")


if __name__ == "__main__":
    unittest.main()
