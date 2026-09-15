"""Public delegation witnesses for accepted identity on outer failures."""

from contextlib import ExitStack
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tools.delegate_tool import delegate_task


ACCEPTED = ("configured-model", "configured-provider")
SELECTED = ("selected-model", "selected-provider")
_REAL_RUN_CONVERSATION = AIAgent.run_conversation


def _accepted_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="accepted response",
                tool_calls=None,
                reasoning_content=None,
                reasoning=None,
                reasoning_details=None,
                model_extra={},
            ),
            finish_reason="stop",
        )],
        model=ACCEPTED[0],
        usage=None,
    )


def _parent(events):
    return SimpleNamespace(
        base_url="https://configured.invalid/v1",
        api_key="synthetic-key",
        provider="configured-provider",
        api_mode="chat_completions",
        model="configured-model",
        platform="cli",
        enabled_toolsets=[],
        disabled_toolsets=[],
        request_overrides={},
        _fallback_chain=[],
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        _session_db=None,
        session_id=None,
        tool_progress_callback=lambda event_type, *_args, **kwargs: events.append(
            {"event": event_type, **kwargs}
        ),
    )


def _delegate_patches(run_conversation, *, timeout):
    stack = ExitStack()
    stack.enter_context(patch("model_tools.get_tool_definitions", return_value=[]))
    stack.enter_context(patch("model_tools.check_toolset_requirements", return_value={}))
    stack.enter_context(patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()))
    stack.enter_context(patch.object(AIAgent, "run_conversation", run_conversation))
    stack.enter_context(
        patch.object(AIAgent, "_interruptible_api_call", return_value=_accepted_response())
    )
    stack.enter_context(
        patch.object(AIAgent, "_interruptible_streaming_api_call", return_value=_accepted_response())
    )
    stack.enter_context(patch.object(AIAgent, "_persist_session"))
    stack.enter_context(patch.object(AIAgent, "_save_trajectory"))
    stack.enter_context(patch.object(AIAgent, "_cleanup_task_resources"))
    stack.enter_context(
        patch.object(
            AIAgent,
            "get_activity_summary",
            return_value={"api_call_count": 1, "current_tool": "synthetic-tool"},
        )
    )
    stack.enter_context(patch("tools.delegate_tool._get_child_timeout", return_value=timeout))
    return stack


@pytest.mark.parametrize(
    ("identity", "expected"),
    [("accepted", ACCEPTED), ("selected", SELECTED), ("none", (None, None))],
    ids=("accepted-over-selected", "selected", "not-accepted"),
)
def test_public_sync_outer_exception_reports_only_accepted_identity(monkeypatch, tmp_path, identity, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    events = []

    def accepted_then_raise(child, **_kwargs):
        if identity == "accepted":
            result = _REAL_RUN_CONVERSATION(child, **_kwargs)
            assert result["completed"] is True
            assert child._delegate_successful_llm_route == ACCEPTED
        if identity in {"accepted", "selected"}:
            child._delegate_model_profile = "fast"
            child.model, child.provider = SELECTED
        raise RuntimeError("synthetic outer child failure")

    with _delegate_patches(accepted_then_raise, timeout=None):
        payload = json.loads(
            delegate_task(
                tasks=[{"goal": "raise after the public child starts"}],
                parent_agent=_parent(events),
                background=False,
            )
        )

    entry = payload["results"][0]
    assert entry["status"] == "error"
    assert (entry["model"], entry["provider"]) == expected
    complete = [event for event in events if event["event"] == "subagent.complete"]
    assert complete
    assert (complete[-1]["model"], complete[-1]["provider"]) == expected


@pytest.mark.parametrize(
    ("identity", "expected"),
    [("accepted", ACCEPTED), ("selected", SELECTED), ("none", (None, None))],
    ids=("accepted-over-selected", "selected", "not-accepted"),
)
def test_public_background_timeout_persists_only_accepted_identity(monkeypatch, tmp_path, identity, expected):
    from tools import async_delegation
    from tools.process_registry import process_registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    release = threading.Event()

    def accepted_then_block(child, **_kwargs):
        if identity == "accepted":
            result = _REAL_RUN_CONVERSATION(child, **_kwargs)
            assert result["completed"] is True
            assert child._delegate_successful_llm_route == ACCEPTED
        if identity in {"accepted", "selected"}:
            child._delegate_model_profile = "fast"
            child.model, child.provider = SELECTED
        release.wait(5)
        return {"completed": True, "final_response": "released too late", "api_calls": 1}

    try:
        # Leave enough admission budget for the real first response even when
        # plugin discovery is cold; the post-acceptance block remains bounded.
        with _delegate_patches(accepted_then_block, timeout=2.0):
            handle = json.loads(
                delegate_task(
                    tasks=[{"goal": "time out after the public child starts"}],
                    parent_agent=_parent([]),
                    background=True,
                )
            )
            deadline = time.monotonic() + 5
            event = None
            while time.monotonic() < deadline:
                if not process_registry.completion_queue.empty():
                    candidate = process_registry.completion_queue.get_nowait()
                    if candidate.get("delegation_id") == handle["delegation_id"]:
                        event = candidate
                        break
                time.sleep(0.02)

        assert event is not None
        entry = event["results"][0]
        assert entry["status"] == "timeout"
        assert (entry["model"], entry["provider"]) == expected
        with async_delegation._connect() as conn:
            event_json, result_json = conn.execute(
                "SELECT event_json, result_json FROM async_delegations WHERE delegation_id=?",
                (handle["delegation_id"],),
            ).fetchone()
        persisted_event = json.loads(event_json)
        persisted_result = json.loads(result_json)
        assert (
            persisted_event["results"][0]["model"],
            persisted_event["results"][0]["provider"],
        ) == expected
        assert (
            persisted_result["results"][0]["model"],
            persisted_result["results"][0]["provider"],
        ) == expected
    finally:
        release.set()
