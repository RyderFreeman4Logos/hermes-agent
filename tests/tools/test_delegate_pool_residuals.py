"""Residual model-pool contracts not covered by the official pin.

Omitted model_profile is standard. A named tier does not inherit parent pools,
parent overrides, or an ambient key. A hop without its own credential is
dropped. Result identity is the accepted route, else the route selected before
the child ran. A fixed tier key does not lease another profile's pool on restore.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    _credentials_for_model_profile,
    _normalize_profile_fallback_chain,
)


def _parent():
    parent = MagicMock()
    parent.base_url = "https://parent.example/v1"
    parent.api_key = "parent-key"
    parent.provider = "openrouter"
    parent.requested_provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "parent-model"
    parent.platform = "cli"
    parent.request_overrides = {"extra_body": {"parent": True}}
    parent.acp_command = "parent-acp"
    parent.acp_args = ["--parent"]
    parent._fallback_chain = [{"provider": "openrouter", "model": "parent-fallback"}]
    parent._credential_pool = MagicMock(name="parent-pool")
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


_POOL = {
    "standard": {
        "provider": "openrouter",
        "model": "std-model",
        "api_key": "std-key",
        "base_url": "https://std.example/v1",
    },
    "complex": {
        "provider": "anthropic",
        "model": "complex-model",
        "api_key": "complex-key",
        "base_url": "https://api.anthropic.com",
        "fallback_chain": [{"provider": "openrouter", "model": "owned-hop", "api_key": "hop-key"}],
    },
}


def test_absent_pool_keeps_legacy_route():
    parent = _parent()
    with patch("tools.delegate_tool._resolve_delegation_credentials", return_value={"model": "legacy", "provider": "openrouter", "base_url": "u", "api_key": "k", "api_mode": "chat_completions"}) as legacy:
        creds = _credentials_for_model_profile({}, parent, None)
    legacy.assert_called_once()
    assert creds["model"] == "legacy"
    assert creds.get("exclusive_route") is not True


def test_omitted_profile_is_standard_and_explicit_complex_keeps_own_chain():
    parent = _parent()
    cfg = {"model": "global-model", "provider": "openrouter", "model_pool": _POOL, "fallback_providers": [{"provider": "openrouter", "model": "global-hop"}]}
    standard = _credentials_for_model_profile(cfg, parent, None)
    complex_route = _credentials_for_model_profile(cfg, parent, "complex")
    assert standard["model"] == "std-model"
    assert standard["api_key"] == "std-key"
    assert standard["fallback_chain"] == []
    assert complex_route["model"] == "complex-model"
    assert complex_route["fallback_chain"][0]["model"] == "owned-hop"
    assert all(entry["model"] != "global-hop" for entry in complex_route["fallback_chain"])
    assert complex_route["exclusive_route"] is True
    assert complex_route["fixed_api_key"] is True


def test_missing_standard_and_unknown_profile_fail_closed():
    parent = _parent()
    broken = {"model_pool": {"complex": _POOL["complex"]}}
    try:
        _credentials_for_model_profile(broken, parent, None)
    except ValueError as exc:
        assert "standard" in str(exc)
    else:
        raise AssertionError("missing standard was accepted")
    try:
        _credentials_for_model_profile({"model_pool": _POOL}, parent, "missing")
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("unknown profile was accepted")


def test_named_tier_does_not_attach_parent_pool_or_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    parent = _parent()
    cfg = {"model_pool": _POOL, "request_overrides": {"extra_body": {"global": True}}}
    creds = _credentials_for_model_profile(cfg, parent, "complex")
    assert creds["model"] == "complex-model"
    assert creds["api_key"] == "complex-key"
    assert creds["base_url"] == "https://api.anthropic.com"
    assert creds.get("command") in (None, "")
    assert "parent" not in json.dumps(creds.get("request_overrides") or {})
    assert "global" not in json.dumps(creds.get("request_overrides") or {})
    assert creds["exclusive_route"] is True
    assert creds["fixed_api_key"] is True
    assert creds["fallback_chain"][0]["model"] == "owned-hop"


def test_unauthenticated_fallback_hop_is_dropped_and_log_has_no_exception_text(caplog, monkeypatch):
    import logging
    from hermes_cli.auth import AuthError

    def _boom(entry):
        if entry.get("model") == "borrowed":
            raise AuthError("secret-traceback")
        return entry.get("api_key")

    monkeypatch.setattr("hermes_cli.fallback_config.resolve_entry_api_key", _boom)
    with caplog.at_level(logging.INFO, logger="tools.delegate_tool"):
        chain = _normalize_profile_fallback_chain([
            {"provider": "openrouter", "model": "owned", "api_key": "owned-key-value"},
            {"provider": "anthropic", "model": "borrowed"},
        ])
    assert [entry["model"] for entry in chain] == ["owned"]
    assert "secret-traceback" not in caplog.text
    assert "AuthError" in caplog.text


def test_result_identity_prefers_accepted_route_else_selected():
    from tools.delegate_tool_child_run import _fabricated_entry

    child = MagicMock()
    child.model = "fallback-model"
    child.provider = "fallback-provider"
    child._delegate_model_profile = "complex"
    child._delegate_successful_llm_route = None
    child.session_estimated_cost_usd = 0
    child.session_cost_status = None
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child._delegate_role = "leaf"
    selected = _fabricated_entry(0, "failed", "boom", child, 1)
    assert selected["model"] == "fallback-model"
    assert selected["provider"] == "fallback-provider"
    assert selected["model_profile"] == "complex"

    child._delegate_selected_route = ("selected-model", "selected-provider")
    child.model = "mutated-model"
    child.provider = "mutated-provider"
    failed = _fabricated_entry(0, "timeout", "slow", child, 1)
    assert (failed["model"], failed["provider"]) == ("selected-model", "selected-provider")

    child._delegate_successful_llm_route = ("accepted-model", "accepted-provider")
    accepted = _fabricated_entry(0, "error", "after accept", child, 1)
    assert (accepted["model"], accepted["provider"]) == ("accepted-model", "accepted-provider")
    assert accepted["model_profile"] == "complex"


def test_fixed_key_restore_does_not_load_another_pool(monkeypatch):
    from agent.agent_runtime_helpers import restore_primary_runtime

    agent = MagicMock()
    agent._fallback_activated = True
    agent._fallback_index = 1
    agent._primary_runtime = {"provider": "custom", "model": "std-model", "base_url": "https://std.example/v1", "api_key": "std-key", "requested_provider": "custom:std"}
    agent._delegation_fixed_api_key = True
    agent._credential_pool = MagicMock(provider="other")
    agent.model = "fallback-model"
    agent.provider = "openrouter"
    agent._provider_fallback_active = True
    agent._provider_fallback_route = ("fallback-model", "openrouter")
    agent._rate_limited_until = 0

    def _load(_key):
        raise AssertionError("fixed key must not load a pool")

    monkeypatch.setattr("agent.credential_pool.load_pool", _load)
    monkeypatch.setattr("agent.route_binding.reinstall_primary_runtime", lambda *args, **kwargs: setattr(agent, "model", "std-model"))
    assert restore_primary_runtime(agent) is True
    assert agent._credential_pool is None


def test_shared_url_keeps_own_key_and_rejects_the_other_owner(monkeypatch):
    from tools.delegate_tool_config import _require_pool_provider

    shared = "https://shared.example/v1"
    own = {"provider": "custom", "base_url": shared, "api_key": "owned-b", "source": "custom_provider:named-b"}
    foreign = {"provider": "custom", "base_url": shared, "api_key": "owned-a", "source": "custom_provider:named-a"}

    def _named(name):
        if name == "named-b":
            return {"name": "named-b", "provider_key": "named-b", "base_url": shared, "api_key": "owned-b"}
        return None

    monkeypatch.setattr("hermes_cli.runtime_provider_custom._get_named_custom_provider", _named)
    _require_pool_provider("named-b", own)
    try:
        _require_pool_provider("named-b", foreign)
    except ValueError as exc:
        assert "named-b" in str(exc)
    else:
        raise AssertionError("foreign key on the same URL was accepted")


_OWN = "https://shared.example/v1"


@pytest.mark.parametrize("declared", [
    "https://Shared.Example/v1",
    "https://shared.example:443/v1",
    "https://shared.example/v1/",
])
def test_canonical_endpoint_keeps_own_key(monkeypatch, declared):
    from tools.delegate_tool_config import _require_pool_provider

    runtime = {"provider": "custom", "base_url": _OWN, "api_key": "owned-b", "source": "custom_provider:named-b"}

    def _named(name):
        if name == "named-b":
            return {"name": "named-b", "provider_key": "named-b", "base_url": declared, "api_key": "owned-b"}
        return None

    monkeypatch.setattr("hermes_cli.runtime_provider_custom._get_named_custom_provider", _named)
    _require_pool_provider("named-b", runtime)


@pytest.mark.parametrize("declared", [
    "http://shared.example/v1",
    "https://shared.example/v2",
    "https://shared.example:8443/v1",
])
def test_distinct_endpoint_rejects_owned_key(monkeypatch, declared):
    from tools.delegate_tool_config import _require_pool_provider

    runtime = {"provider": "custom", "base_url": _OWN, "api_key": "owned-b", "source": "custom_provider:named-b"}

    def _named(name):
        if name == "named-b":
            return {"name": "named-b", "provider_key": "named-b", "base_url": declared, "api_key": "owned-b"}
        return None

    monkeypatch.setattr("hermes_cli.runtime_provider_custom._get_named_custom_provider", _named)
    with pytest.raises(ValueError, match="named-b"):
        _require_pool_provider("named-b", runtime)


def test_same_canonical_url_wrong_provider_still_rejected(monkeypatch):
    from tools.delegate_tool_config import _require_pool_provider

    runtime = {"provider": "custom", "base_url": _OWN, "api_key": "owned-b", "source": "custom_provider:named-b"}

    def _named(name):
        if name == "named-a":
            return {"name": "named-a", "provider_key": "named-a", "base_url": "https://Shared.Example/v1/", "api_key": "owned-a"}
        return None

    monkeypatch.setattr("hermes_cli.runtime_provider_custom._get_named_custom_provider", _named)
    with pytest.raises(ValueError, match="named-a"):
        _require_pool_provider("named-a", runtime)


def test_fallback_success_stamp_is_the_route_that_served():
    from tools.delegate_tool_child_run import _build_result_entry, _SchemaOutcome

    child = MagicMock()
    child.model = "served-model"
    child.provider = "anthropic"
    child._delegate_model_profile = "standard"
    child._delegate_successful_llm_route = ("served-model", "anthropic")
    child._delegate_selected_route = ("primary-model", "openrouter")
    child._fallback_activated = True
    child._primary_runtime = {"model": "primary-model"}
    child.session_estimated_cost_usd = 0
    child.session_cost_status = None
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child._delegate_role = "leaf"
    schema = _SchemaOutcome(None, None, [], 0)
    entry = _build_result_entry(child, {"final_response": "ok", "completed": True, "api_calls": 2}, 0, 1, schema)
    assert (entry["model"], entry["provider"]) == ("served-model", "anthropic")
    child._delegate_successful_llm_route = None
    child.model, child.provider = "mutated-model", "mutated-provider"
    failed = _build_result_entry(child, {"failed": True, "error": "down"}, 0, 1, schema)
    assert (failed["model"], failed["provider"]) == ("primary-model", "openrouter")


@pytest.mark.parametrize("outcome", ["success", "failed", "cancelled"])
def test_complete_logical_call_stamps_served_route_only_on_success(outcome):
    from agent.relay_llm import complete_logical_call
    from tools.delegate_tool_child_run import _build_result_entry, _SchemaOutcome

    child = MagicMock()
    child.model = "served-model"
    child.provider = "anthropic"
    child._delegate_model_profile = "standard"
    child._delegate_selected_route = ("primary-model", "openrouter")
    child._delegate_successful_llm_route = ("primary-model", "openrouter")
    child.session_estimated_cost_usd = 0
    child.session_cost_status = None
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child._delegate_role = "leaf"
    complete_logical_call("req-1", outcome=outcome, agent=child)
    schema = _SchemaOutcome(None, None, [], 0)
    result = {"final_response": "ok", "completed": True} if outcome == "success" else {"failed": True, "error": "down"}
    entry = _build_result_entry(child, result, 0, 1, schema)
    expected = ("served-model", "anthropic") if outcome == "success" else ("primary-model", "openrouter")
    assert (entry["model"], entry["provider"]) == expected
