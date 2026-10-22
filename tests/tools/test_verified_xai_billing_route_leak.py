"""Verified xAI billing on a non-xAI accepted route must not leak the rejection.

The accepted route owns public identity. A verified xAI spending-limit body
on an anthropic/claude answer must not appear in the result, subagent.complete,
or the live log. Unverified xAI stays hidden. An accepted xAI route stays.
"""

import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _run_single_child


_RAW = "HTTP 403 personal-team-blocked:spending-limit raw body"
_XAI_MODEL = "grok-4.6"
_XAI_URL = "https://api.x.ai/v1"


def _parent():
    parent = MagicMock()
    parent._touch_activity = lambda *_a, **_k: None
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._current_task_id = None
    return parent


def _child(events, result, *, model, provider, route):
    child = MagicMock()
    child.session_id = "child-sess"
    child.model = model
    child.provider = provider
    child.base_url = _XAI_URL
    child.api_mode = "chat_completions"
    child._credential_pool = None
    child._delegate_successful_llm_route = route
    child.tool_progress_callback = lambda event, **kw: events.append((event, kw))
    child.run_conversation.return_value = result
    child.get_activity_summary.return_value = {}
    return child


def _run(result, *, model, provider, route):
    events = []
    child = _child(events, result, model=model, provider=provider, route=route)
    with patch("tools.delegate_tool._HEARTBEAT_INTERVAL", 60):
        entry = _run_single_child(0, "goal", child=child, parent_agent=_parent())
    return entry, events


def _blob(entry, events, logs):
    return json.dumps(entry, default=str) + repr(events) + "\n".join(logs)


def _logs_for(fn):
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("tools.delegate_tool")
    handler = _Capture()
    logger.addHandler(handler)
    old = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        return fn(), [r.getMessage() for r in records]
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)


def test_verified_xai_billing_on_non_xai_route_does_not_leak():
    result = {
        "final_response": _RAW,
        "completed": False,
        "failed": True,
        "error": _RAW,
        "failure_reason": "billing",
        "billing_unverified": False,
        "billing_block": {"provider": "xai", "message": _RAW},
        "api_calls": 2,
        "messages": [{"role": "tool", "content": f"{_RAW} {_XAI_URL} {_XAI_MODEL}"}],
    }

    def go():
        return _run(
            result,
            model="claude-sonnet-4-6",
            provider="anthropic",
            route=("claude-sonnet-4-6", "anthropic"),
        )

    (entry, events), logs = _logs_for(go)
    blob = _blob(entry, events, logs)
    assert _RAW not in blob
    assert _XAI_MODEL not in blob
    assert _XAI_URL not in blob
    assert entry["model"] == "claude-sonnet-4-6"
    assert entry["provider"] == "anthropic"
    complete = [kw for event, kw in events if event == "subagent.complete"]
    assert complete
    assert complete[0].get("summary") != _RAW


def test_unverified_xai_billing_stays_hidden():
    result = {
        "final_response": _RAW,
        "completed": False,
        "failed": True,
        "error": _RAW,
        "failure_reason": "billing",
        "billing_unverified": True,
        "billing_block": {"provider": "xai-oauth"},
        "api_calls": 1,
        "messages": [],
    }
    entry, events = _run(
        result, model=_XAI_MODEL, provider="xai-oauth", route=(_XAI_MODEL, "xai-oauth"),
    )
    blob = _blob(entry, events, [])
    assert _RAW not in blob
    assert entry["model"] is None
    assert entry["provider"] is None


def test_accepted_xai_route_is_published():
    result = {
        "final_response": "ok",
        "completed": True,
        "failed": False,
        "api_calls": 1,
        "messages": [],
    }
    entry, events = _run(
        result, model=_XAI_MODEL, provider="xai-oauth", route=(_XAI_MODEL, "xai-oauth"),
    )
    assert entry["status"] == "completed"
    assert entry["model"] == _XAI_MODEL
    assert entry["provider"] == "xai-oauth"
    complete = [kw for event, kw in events if event == "subagent.complete"]
    assert "ok" in (complete[0].get("summary") or "")


def test_schema_retry_keeps_accepted_non_xai_answer(monkeypatch):
    from tools.delegate_tool_child_run import _validate_child_output_schema

    calls = []

    class Child(SimpleNamespace):
        def run_conversation(self, user_message, task_id=None, **_kwargs):
            calls.append(user_message)
            self.model, self.provider = "claude-sonnet-4-6", "anthropic"
            self._delegate_successful_llm_route = (self.model, self.provider)
            return {
                "final_response": '{"city": "Oslo"}',
                "completed": True,
                "failed": False,
                "api_calls": 1,
                "messages": [],
            }

    child = Child(
        model=_XAI_MODEL,
        provider="xai",
        _delegate_successful_llm_route=(_XAI_MODEL, "xai"),
        _delegate_output_schema={"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}},
        session_id="sess",
    )
    monkeypatch.setattr(
        "tools.delegation_output_schema.validate_output",
        lambda text, _schema: (text.startswith("{"), [] if text.startswith("{") else ["city"]),
    )
    monkeypatch.setattr(
        "tools.delegation_output_schema.build_retry_message",
        lambda errors: "retry",
    )
    first = {
        "final_response": _RAW,
        "completed": False,
        "failed": True,
        "error": _RAW,
        "failure_reason": "billing",
        "billing_unverified": False,
        "billing_block": {"provider": "xai"},
        "api_calls": 1,
        "messages": [],
    }
    outcome = _validate_child_output_schema(child, first, 0, "child-0", None)
    assert outcome.valid is True
    assert first["final_response"] == '{"city": "Oslo"}'
    assert "billing_block" not in first
    assert "billing_unverified" not in first
    assert "error" not in first
    assert _RAW not in json.dumps(first)
    assert child._delegate_successful_llm_route == ("claude-sonnet-4-6", "anthropic")
    assert calls == ["retry"]
