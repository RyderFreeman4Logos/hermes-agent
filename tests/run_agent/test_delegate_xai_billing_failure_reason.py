"""Public witnesses for xAI billing failure-reason attribution."""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.delegate_tool import delegate_task


class _BillingChild:
    tool_progress_callback = None
    _delegate_saved_tool_names: list = []
    _credential_pool = None
    _subagent_id = None
    _delegate_depth = 1
    _parent_subagent_id = None
    _delegate_output_schema = None
    _delegate_role = "leaf"
    api_mode = "chat_completions"
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_reasoning_tokens = 0
    session_estimated_cost_usd = 0.0
    session_cost_status = None

    def __init__(self, *, accepted_route, model, profile, result):
        self._delegate_successful_llm_route = accepted_route
        self.model = model
        self.provider = "xai-oauth" if "grok" in model else "configured-provider"
        self._delegate_model_profile = profile
        self._result = result

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 1, "current_tool": None}

    def run_conversation(self, **_kwargs):
        return dict(self._result)

    def close(self):
        return None


def _parent():
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
        tool_progress_callback=None,
    )


def _failed_result(*, reason, billing_block=None, billing_unverified=False):
    raw = "synthetic xAI spending-limit terminal"
    result = {
        "final_response": raw,
        "error": raw,
        "failure_reason": reason,
        "completed": False,
        "failed": True,
        "interrupted": False,
        "api_calls": 1,
        "messages": [],
    }
    if billing_block is not None:
        result["billing_block"] = billing_block
    if billing_unverified:
        result["billing_unverified"] = True
    return result


def _public_delegate(child, *, background):
    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch(
            "tools.delegate_tool._resolve_delegation_credentials",
            return_value={
                "provider": None,
                "model": None,
                "base_url": None,
                "api_key": None,
                "api_mode": None,
            },
        ),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=child),
    ):
        return json.loads(
            delegate_task(
                tasks=[{"goal": "exercise billing attribution"}],
                parent_agent=_parent(),
                background=background,
            )
        )


def test_public_sync_drops_unaccepted_xai_billing_reason_from_non_xai_identity():
    child = _BillingChild(
        accepted_route=("accepted-model", "accepted-provider"),
        model="grok-4.6",
        profile="standard",
        result=_failed_result(
            reason="billing",
            billing_block={"provider": "xai-oauth"},
            billing_unverified=True,
        ),
    )

    entry = _public_delegate(child, background=False)["results"][0]

    assert (entry["model"], entry["provider"]) == (
        "accepted-model",
        "accepted-provider",
    )
    assert entry["status"] == "failed"
    assert "spending-limit" not in entry["summary"]
    assert "spending-limit" not in entry["error"]
    assert "failure_reason" not in entry


def test_public_background_durably_drops_unaccepted_xai_billing_reason(monkeypatch, tmp_path):
    from tools import async_delegation
    from tools.process_registry import process_registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    child = _BillingChild(
        accepted_route=None,
        model="grok-4.6",
        profile="standard",
        result=_failed_result(
            reason="billing",
            billing_block={"provider": "xai-oauth"},
            billing_unverified=True,
        ),
    )

    handle = _public_delegate(child, background=True)
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
    with async_delegation._connect() as conn:
        event_json, result_json = conn.execute(
            "SELECT event_json, result_json FROM async_delegations WHERE delegation_id=?",
            (handle["delegation_id"],),
        ).fetchone()
    persisted_event = json.loads(event_json)
    persisted_result = json.loads(result_json)
    for entry in (
        event["results"][0],
        persisted_event["results"][0],
        persisted_result["results"][0],
    ):
        assert (entry["model"], entry["provider"]) == (None, None)
        assert "spending-limit" not in entry["summary"]
        assert "spending-limit" not in entry["error"]
        assert "failure_reason" not in entry


@pytest.mark.parametrize(
    ("accepted_route", "model", "profile", "result", "expected_reason"),
    [
        pytest.param(
            ("grok-4.6", "xai-oauth"),
            "grok-4.6",
            "premium",
            _failed_result(
                reason="billing",
                billing_block={"provider": "xai-oauth"},
            ),
            "billing",
            id="accepted-xai-billing",
        ),
        pytest.param(
            ("accepted-model", "accepted-provider"),
            "accepted-model",
            "standard",
            _failed_result(reason="rate_limit"),
            "rate_limit",
            id="non-billing-failure",
        ),
    ],
)
def test_public_sync_keeps_owned_failure_reason(
    accepted_route, model, profile, result, expected_reason
):
    child = _BillingChild(
        accepted_route=accepted_route,
        model=model,
        profile=profile,
        result=result,
    )

    entry = _public_delegate(child, background=False)["results"][0]

    assert (entry["model"], entry["provider"]) == accepted_route
    assert entry["status"] == "failed"
    assert entry["failure_reason"] == expected_reason
