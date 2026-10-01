import json
from unittest.mock import patch

from tests.tools.test_delegate_model_pool_boundaries import _fake_child, _parent
from tools.delegate_tool import delegate_task


URL = "https://explicit.invalid/v1"
OWNED_URL = "https://named.invalid/v1"


def _dispatch(config, runtime, named, captured):
    parent = _parent()
    resolver = patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        side_effect=runtime if isinstance(runtime, Exception) else None,
        return_value=None if isinstance(runtime, Exception) else runtime,
    )
    with patch("tools.delegate_tool._load_config", return_value=config), resolver, patch(
        "hermes_cli.runtime_provider_custom._get_named_custom_provider", return_value=named,
    ), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=_fake_child(parent, captured),
    ), patch("tools.delegate_tool._run_batch", return_value=json.dumps({"ok": True})):
        raw = delegate_task(goal="isolated identity probe", parent_agent=parent)
    return json.loads(raw)


def _pool_profile(**extra):
    return {"max_iterations": 4, "model_pool": {"standard": {
        "provider": "named-tier", "model": "tier-model", "base_url": URL,
        "api_key": "fixture-tier-owned-key", **extra,
    }}}


def _runtime(*, url=URL, key="named-tier-owned-key", provider="named-tier", overrides=None):
    return {
        "provider": "custom", "requested_provider": provider, "model": "tier-model",
        "base_url": url, "api_key": key, "api_mode": "chat_completions",
        "source": "local-runtime", "request_overrides": overrides or {},
    }


def _named(url=URL):
    return {"name": "named-tier", "base_url": url, "api_key": "named-tier-owned-key"}


def test_wrong_named_provider_same_url_is_rejected_before_child():
    captured = []
    result = _dispatch(_pool_profile(), _runtime(key="different-provider-key", provider="other-provider"), _named(), captured)
    assert "error" in result and "did not resolve its own provider credentials" in result["error"]
    assert not captured


def test_matching_named_provider_endpoint_key_and_owned_overrides_are_preserved():
    captured = []
    result = _dispatch(
        _pool_profile(request_overrides={"extra_body": {"tier": True}}),
        _runtime(overrides={"extra_body": {"provider": True}}), _named(), captured,
    )
    assert result == {"ok": True}
    assert captured[0]["override_api_key"] == "fixture-tier-owned-key"
    assert captured[0]["override_requested_provider"] == "named-tier"
    assert captured[0]["override_request_overrides"] == {"extra_body": {"provider": True, "tier": True}}


def test_different_named_url_keeps_explicit_endpoint_key_not_provider_key():
    captured = []
    profile = {"max_iterations": 4, "model_pool": {"standard": {
        "provider": "named-tier", "model": "tier-model", "base_url": URL,
        "api_key": "fixture-tier-owned-key", "request_overrides": {"extra_body": {"tier": True}},
    }}}
    result = _dispatch(profile, _runtime(url=OWNED_URL, overrides={"extra_body": {"provider": True}}), _named(OWNED_URL), captured)
    assert result == {"ok": True}
    assert captured[0]["override_base_url"] == URL
    assert captured[0]["override_api_key"] == "fixture-tier-owned-key"
    assert captured[0]["override_request_overrides"] == {"extra_body": {"tier": True}}


def test_exclusive_direct_endpoint_fails_closed_if_provider_resolution_throws():
    captured = []
    result = _dispatch(_pool_profile(), RuntimeError("resolver unavailable"), _named(), captured)
    assert "error" in result and "Cannot resolve delegation provider" in result["error"]
    assert not captured


def test_nonexclusive_direct_endpoint_preserves_legacy_resolver_failure_behavior():
    captured = []
    config = {"max_iterations": 4, "provider": "named-tier", "model": "legacy-model",
              "base_url": URL, "api_key": "legacy-explicit-key"}
    result = _dispatch(config, RuntimeError("resolver unavailable"), _named(), captured)
    assert result == {"ok": True}
    assert captured[0]["override_api_key"] == "legacy-explicit-key"
    assert captured[0]["override_base_url"] == URL


def test_exclusive_direct_endpoint_rejects_missing_or_incomplete_runtime():
    for runtime in (None, {}):
        captured = []
        result = _dispatch(_pool_profile(), runtime, _named(), captured)
        assert "error" in result
        assert not captured
