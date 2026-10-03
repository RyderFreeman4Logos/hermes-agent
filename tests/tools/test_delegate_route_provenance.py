"""Public route identity and xAI billing hedges for delegated children.

Selected identity is the profile route frozen before the child runs.
Accepted identity is a successful SDK route stamp, not the mutable model.
Neither is proof of provider acceptance, and neither inherits parent creds.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import _build_children, delegate_task

_SPEND = "Your team has reached its spending limit."
_HEDGE_UNVERIFIED = "Subagent failed after an unverified provider billing error."
_HEDGE_FOREIGN = "Subagent failed with a provider error unrelated to its effective model."
_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


def _parent():
    parent = MagicMock()
    parent.base_url = "https://api.openai.com/v1"
    parent.api_key = "parent-secret"
    parent.provider = "openai"
    parent.api_mode = "chat_completions"
    parent.model = "gpt-parent"
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
    return parent


def _route(provider, model, api_mode=None):
    route = {
        "provider": "custom",
        "model": model,
        "base_url": f"http://127.0.0.1:9/{provider}",
        "api_key": f"{provider}-key",
        "xai_family": provider in {"xai", "xai-oauth"},
    }
    if api_mode:
        route["api_mode"] = api_mode
    return route


def _pool(profile, name):
    pool = {"standard": _route("standard", "standard-model")}
    pool[name] = dict(profile)
    return {"model_pool": pool}


def _billing_result(*, unverified, model_text="grok-4.6"):
    return {
        "final_response": _SPEND,
        "completed": False,
        "failed": True,
        "error": _SPEND,
        "failure_reason": "billing",
        "billing_block": {"provider": "xai-oauth", "message": _SPEND},
        "billing_unverified": unverified,
        "api_calls": 1,
        "messages": [],
        "model": model_text,
    }


def _run(profile, result, *, mutate=None, goal="route provenance", schema=None, profile_name="fast"):
    parent = _parent()
    built = {}

    def capture(**kwargs):
        child = MagicMock()
        child.model = kwargs.get("model")
        child.provider = "xai-oauth" if profile.get("xai_family") else kwargs.get("provider")
        child.api_mode = kwargs.get("api_mode")
        child.api_key = kwargs.get("api_key")
        child.run_conversation.return_value = result
        child.get_activity_summary.return_value = {"api_call_count": 1}
        built["child"] = child
        built["kwargs"] = kwargs
        if mutate:
            mutate(child)
        return child

    task = {"goal": goal, "model_profile": profile_name}
    if schema is not None:
        task["output_schema"] = schema
    with patch("tools.delegate_tool._load_config", return_value=_pool(profile, profile_name)), patch(
        "run_agent.AIAgent", side_effect=capture
    ):
        payload = json.loads(delegate_task(tasks=[task], parent_agent=parent))
    assert "error" not in payload
    return payload["results"][0], built


def _stamp(child, model, provider):
    child._delegate_accepted_llm_route = (model, provider)


def test_non_grok_child_hides_verified_xai_billing():
    entry, _ = _run(
        _route("deepseek", "deepseek-v4-flash"),
        _billing_result(unverified=False),
    )
    assert entry["summary"] == _HEDGE_FOREIGN
    assert entry["error"] == _HEDGE_FOREIGN
    assert "failure_reason" not in entry
    assert _SPEND not in json.dumps(entry)
    assert entry["model"] == "deepseek-v4-flash"
    assert entry["provider"] == "custom"


def test_unverified_standard_grok_child_hides_xai_billing():
    entry, _ = _run(
        _route("xai-oauth", "grok-4.6"),
        _billing_result(unverified=True),
        profile_name="standard",
    )
    assert entry["summary"] == _HEDGE_UNVERIFIED
    assert entry["error"] == _HEDGE_UNVERIFIED
    assert "failure_reason" not in entry
    assert _SPEND not in json.dumps(entry)


def test_verified_grok_child_keeps_owned_billing():
    entry, _ = _run(
        _route("xai-oauth", "grok-4.6"),
        _billing_result(unverified=False),
    )
    assert entry["failure_reason"] == "billing"
    assert entry["summary"] == _SPEND
    assert entry["error"] == _SPEND
    assert entry["provider"] == "custom"


def test_invalid_schema_error_names_the_contract():
    entry, _ = _run(
        _route("deepseek", "deepseek-v4-flash"),
        {
            "final_response": "not json",
            "completed": True,
            "api_calls": 1,
            "messages": [],
        },
        schema=_SCHEMA,
    )
    assert entry["schema_valid"] is False
    assert entry["status"] == "completed"
    assert "error" not in entry
    assert entry["schema_note"].startswith("Final answer does not satisfy the declared output_schema")


def test_schema_retry_clears_rejected_turn_provenance():
    first = {
        "final_response": "not json",
        "completed": False,
        "failed": True,
        "error": _SPEND,
        "failure_reason": "billing",
        "failure_retryable": False,
        "billing_block": {"provider": "xai-oauth"},
        "billing_unverified": True,
        "codex_turn_id": "turn-rejected",
        "api_calls": 1,
        "messages": [{"role": "assistant", "content": "not json"}],
    }
    retry = {
        "final_response": '{"city": "Oslo"}',
        "completed": True,
        "api_calls": 1,
        "messages": [{"role": "assistant", "content": '{"city": "Oslo"}'}],
    }

    def mutate(child):
        calls = {"n": 0}

        def turn(*_a, **_k):
            calls["n"] += 1
            if calls["n"] == 1:
                return first
            _stamp(child, "claude-sonnet", "openrouter")
            return retry

        child.run_conversation.side_effect = turn

    entry, _ = _run(
        _route("xai-oauth", "grok-4.6"),
        first,
        mutate=mutate,
        schema=_SCHEMA,
    )
    assert entry["status"] == "completed"
    assert entry["schema_valid"] is True
    assert entry["summary"] == '{"city": "Oslo"}'
    for key in ("error", "failure_reason", "billing_block", "billing_unverified"):
        assert key not in entry
    assert _SPEND not in json.dumps(entry)
    assert entry["model"] == "claude-sonnet"
    assert entry["provider"] == "openrouter"


def test_outer_exception_reports_frozen_selected_not_fallback():
    def mutate(child):
        child.model, child.provider = "fallback-model", "fallback-provider"
        child.run_conversation.side_effect = RuntimeError("worker exploded")

    entry, _ = _run(
        _route("selected", "selected-model"),
        {},
        mutate=mutate,
    )
    assert entry["status"] == "error"
    assert (entry["model"], entry["provider"]) == ("selected-model", "custom")
    assert "fallback-model" not in json.dumps(entry)


def test_accepted_route_beats_selected_on_failure():
    def mutate(child):
        _stamp(child, "accepted-model", "accepted-provider")
        child.model, child.provider = "fallback-model", "fallback-provider"

    entry, _ = _run(
        _route("selected", "selected-model"),
        {
            "final_response": "transport died",
            "completed": False,
            "failed": True,
            "error": "transport died",
            "failure_reason": "server_error",
            "api_calls": 2,
            "messages": [],
        },
        mutate=mutate,
    )
    assert (entry["model"], entry["provider"]) == ("accepted-model", "accepted-provider")
    assert entry["failure_reason"] == "server_error"


def test_selected_route_is_frozen_before_the_child_runs():
    parent = _parent()
    profile = _route("selected", "selected-model")
    creds = {
        "model": "selected-model",
        "provider": "custom",
        "base_url": profile["base_url"],
        "api_key": profile["api_key"],
        "api_mode": None,
    }
    with patch("tools.delegate_tool._load_config", return_value=_pool(profile, "fast")), patch(
        "run_agent.AIAgent", return_value=MagicMock(model=creds["model"], provider=creds["provider"])
    ):
        children, error = _build_children(
            [{"goal": "freeze", "model_profile": "fast"}],
            [None],
            [("fast", creds)],
            top_role="leaf",
            max_iterations=2,
            parent_agent=parent,
            routing_cfg={},
            live_deleg_id=None,
            live_writers=[],
        )
    assert error is None
    child = children[0][2]
    assert child._delegate_selected_llm_route == ("selected-model", "custom")
    child.model, child.provider = "fallback-model", "fallback-provider"
    assert child._delegate_selected_llm_route == ("selected-model", "custom")


def test_nonsuccess_codex_app_server_reports_accepted_only():
    def mutate(child):
        child.api_mode = "codex_app_server"
        child.model, child.provider = "mutable-model", "mutable-provider"

    entry, _ = _run(
        _route("openai-codex", "gpt-selected", "codex_app_server"),
        {
            "final_response": "app server failed",
            "completed": False,
            "failed": True,
            "error": "app server failed",
            "api_calls": 1,
            "messages": [],
        },
        mutate=mutate,
    )
    assert entry["model"] is None
    assert entry["provider"] is None


def test_codex_app_server_keeps_prior_accepted_route_on_later_failure():
    def mutate(child):
        child.api_mode = "codex_app_server"
        _stamp(child, "accepted-codex", "openai-codex")
        child.model, child.provider = "mutable-model", "mutable-provider"

    entry, _ = _run(
        _route("openai-codex", "gpt-selected", "codex_app_server"),
        {
            "final_response": "later failure",
            "completed": False,
            "failed": True,
            "error": "later failure",
            "api_calls": 2,
            "messages": [],
        },
        mutate=mutate,
    )
    assert (entry["model"], entry["provider"]) == ("accepted-codex", "openai-codex")


def test_omitted_profile_keeps_existing_bare_model_route():
    """#185 already owns credential isolation. A profile-less call stays on that path."""
    parent = _parent()
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        child = MagicMock()
        child.model = kwargs.get("model")
        child.provider = kwargs.get("provider")
        child.run_conversation.return_value = {"final_response": "ok", "completed": True, "api_calls": 1, "messages": []}
        return child

    with patch("tools.delegate_tool._load_config", return_value={
        "model": "configured-model",
        "provider": "custom",
        "base_url": "http://127.0.0.1:9/bare",
        "api_key": "configured-key",
    }), patch(
        "run_agent.AIAgent", side_effect=capture
    ):
        payload = json.loads(delegate_task(goal="bare", parent_agent=parent))
    assert "error" not in payload
    entry = payload["results"][0]
    assert entry["status"] == "completed"
    assert captured.get("api_key") != "parent-secret"


def test_unverified_billing_on_selected_non_xai_is_not_owned():
    entry, _ = _run(
        _route("deepseek", "deepseek-v4-flash"),
        _billing_result(unverified=True, model_text="deepseek-v4-flash"),
    )
    assert "failure_reason" not in entry
    assert _SPEND not in json.dumps(entry)
    assert (entry["model"], entry["provider"]) == ("deepseek-v4-flash", "custom")
