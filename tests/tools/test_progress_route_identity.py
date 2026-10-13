"""Public progress identity is the accepted route, never the construction route.

The production callback is built before the child exists, with the construction
model. ``subagent.start`` / ``subagent.progress`` / ``subagent.complete`` (and
the live list) stay blank until a shape-valid response stamps a route. An
unverified xAI 403 never publishes that construction identity. A later
accepted alternate route is published only after the stamp.
"""

import json
import threading
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _run_single_child
from tools.delegate_tool_progress import _build_child_progress_callback
from tools.delegate_tool_registry import _list_payload, list_active_subagents


_XAI_MODEL = "grok-4.6"
_ACCEPTED_MODEL = "claude-sonnet-4-6"
_RAW = "HTTP 403 personal-team-blocked:spending-limit raw body"
_PUBLIC = ("subagent.start", "subagent.progress", "subagent.complete")


class _Child:
    def __init__(self, result, *, subagent_id):
        self.session_id = "child-sess"
        self.model = _XAI_MODEL
        self.provider = "xai-oauth"
        self.base_url = "https://api.x.ai/v1"
        self.api_mode = "chat_completions"
        self._credential_pool = None
        self._delegate_successful_llm_route = None
        self._subagent_id = subagent_id
        self._delegate_depth = 1
        self._parent_subagent_id = None
        self._delegation_id = "deleg-1"
        self._parent_session_id = "parent-sess"
        self._delegate_role = "leaf"
        self._result = result
        self.tool_progress_callback = None
        self._saw_start = False

    def run_conversation(self, *_args, **_kwargs):
        cb = self.tool_progress_callback
        if stamp := getattr(self, "_stamp_before_tool", None):
            self._progress_ref["child"] = self
            self.model, self.provider = stamp
            self._delegate_successful_llm_route = stamp
        if callable(cb):
            cb("tool.started", "read_file", "notes", None)
        return self._result

    def get_activity_summary(self):
        return {}

    def close(self):
        return None


def _parent(events, snapshots):
    parent = MagicMock()
    parent._touch_activity = lambda *_a, **_k: None
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._current_task_id = None
    parent.session_id = "parent-sess"
    parent._delegate_spinner = None

    def parent_cb(*args, **kwargs):
        event = args[0] if args else None
        events.append((args, kwargs))
        if event in _PUBLIC:
            snapshots.append(list_active_subagents())

    parent.tool_progress_callback = parent_cb
    return parent


def _wire_production_callback(child, parent):
    """Same order as spawn: the callback is built with the construction model before the child exists."""
    ref = {}
    child.tool_progress_callback = _build_child_progress_callback(
        0, "goal", parent, 1, subagent_id=child._subagent_id, model=_XAI_MODEL, session_ref=ref,
    )
    child._progress_ref = ref
    return ref


def _run(result, *, subagent_id, stamp_before_tool=False):
    events, snapshots = [], []
    child = _Child(result, subagent_id=subagent_id)
    parent = _parent(events, snapshots)
    ref = _wire_production_callback(child, parent)
    if stamp_before_tool:
        child._stamp_before_tool = (_ACCEPTED_MODEL, "anthropic")
    del ref
    with patch("tools.delegate_tool._HEARTBEAT_INTERVAL", 60):
        entry = _run_single_child(0, "goal", child=child, parent_agent=parent)
    return entry, events, snapshots


def _public(events):
    rows = []
    for args, kwargs in events:
        event = args[0] if args else None
        if event in _PUBLIC:
            rows.append((event, kwargs.get("model"), kwargs.get("provider")))
    return rows


def test_unverified_xai_hides_construction_route_on_start_progress_complete_and_list():
    result = {
        "final_response": _RAW,
        "completed": False,
        "failed": True,
        "error": _RAW,
        "failure_reason": "billing",
        "billing_unverified": True,
        "billing_block": {"provider": "xai-oauth", "message": _RAW},
        "api_calls": 1,
        "messages": [{"role": "assistant", "content": "partial"}],
    }
    entry, events, snapshots = _run(result, subagent_id="sa-unverified")
    blob = json.dumps({"entry": entry, "events": events, "snapshots": snapshots}, default=str)
    assert _XAI_MODEL not in blob
    assert _RAW not in blob
    rows = _public(events)
    assert [row[0] for row in rows] == ["subagent.start", "subagent.progress", "subagent.complete"]
    assert rows == [(event, None, None) for event, _m, _p in rows]
    assert entry["model"] is None and entry["provider"] is None
    assert snapshots
    assert all(row.get("model") is None for snap in snapshots for row in snap)


def test_accepted_alternate_route_is_published_only_after_stamp():
    result = {
        "final_response": "ok",
        "completed": True,
        "failed": False,
        "api_calls": 2,
        "messages": [{"role": "assistant", "content": "ok"}],
    }
    entry, events, snapshots = _run(result, subagent_id="sa-accepted", stamp_before_tool=True)
    rows = _public(events)
    assert rows[0] == ("subagent.start", None, None)
    assert ("subagent.progress", _ACCEPTED_MODEL, "anthropic") in rows
    assert rows[-1] == ("subagent.complete", _ACCEPTED_MODEL, "anthropic")
    assert entry["status"] == "completed"
    assert entry["model"] == _ACCEPTED_MODEL and entry["provider"] == "anthropic"
    assert "ok" in (entry.get("summary") or "")
    listed_models = [row.get("model") for snap in snapshots for row in snap]
    assert _XAI_MODEL not in listed_models
    assert _ACCEPTED_MODEL in listed_models
