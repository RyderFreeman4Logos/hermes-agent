"""Finite typed delegation boundary and resolver-owned identity contracts (#185)."""
import copy
import json
from unittest.mock import patch

import pytest
import yaml

from tests.tools.test_delegate_model_pool_boundaries import _fake_child, _parent
from tests.tools.test_review_identity_dispatch import URL, _dispatch, _named, _runtime
from tools.delegate_tool import delegate_task


_RUNTIME_FAULTS = [
    (field, 7) for field in (
        "model", "provider", "requested_provider", "base_url", "api_mode", "command", "source",
    )
] + [
    ("api_key", ["opaque-canary"]), ("api_key", True), ("api_key", {"opaque-canary": 1}),
    ("request_overrides", 7), ("request_overrides", "opaque-canary"),
    ("request_overrides", ["opaque-canary"]),
    ("args", 7), ("args", "opaque-canary"), ("args", {"opaque-canary": 1}),
    ("args", [7]), ("credential_pool", 7),
]
_CONFIG_FAULTS = [
    (field, bad) for field in ("model", "provider", "base_url", "api_key", "api_mode")
    for bad in (7, True, ["opaque-canary"], {"opaque-canary": 1})
] + [("request_overrides", bad) for bad in (7, "opaque-canary", ["opaque-canary"])]
_CASES = [
    ("runtime", surface, field, bad)
    for surface in ("fixed", "derived", "provider", "native", "legacy-direct", "legacy-provider")
    for field, bad in _RUNTIME_FAULTS
] + [
    ("config", surface, field, bad)
    for surface in ("fixed", "derived", "provider", "native", "legacy-direct", "legacy-provider", "inherit")
    for field, bad in _CONFIG_FAULTS
] + [("parent", "inherit", "request_overrides", 7)]


@pytest.mark.parametrize("origin,surface,field,bad", _CASES)
def test_consumed_shapes_refuse_before_child(origin, surface, field, bad, caplog):
    route = {"model": "fixture-model", "provider": "named-tier"}
    rt = _runtime()
    if surface in {"fixed", "derived", "legacy-direct"}:
        route["base_url"] = URL
    if surface in {"fixed", "legacy-direct"}:
        route["api_key"] = "fixed-key"
    if surface == "native":
        route["provider"] = "google"
        rt.update(provider="google", api_mode="google_genai", requested_provider="google")
    if surface == "inherit":
        route = {"model": "fixture-model"}
    if origin == "runtime":
        rt[field] = bad
    elif origin == "config":
        route[field] = bad
    cfg = {"model_pool": {"standard": route}} if surface in {"fixed", "derived", "provider", "native"} else route
    captured = []
    if origin == "parent":
        parent = _parent()
        parent.request_overrides = bad
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "tools.delegate_tool._build_child_preserving_parent_tools", side_effect=_fake_child(parent, captured)
        ):
            result = json.loads(delegate_task(goal="typed inherit", parent_agent=parent))
    else:
        result = _dispatch(cfg, rt, _named(), captured)
    assert "error" in result, result
    assert not captured
    assert "opaque-canary" not in result["error"]
    assert "opaque-canary" not in caplog.text


@pytest.mark.parametrize("surface", ["fixed", "derived", "provider"])
@pytest.mark.parametrize("credential", ["static", "callable"])
@pytest.mark.parametrize("owner", ["own", "foreign"])
def test_declared_identity_is_resolver_owned_not_request_tag(tmp_path, monkeypatch, surface, credential, owner):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    providers = {name: {"name": "Shared Label", "base_url": URL, "extra_body": {"owner": name}} for name in ("own", "foreign")}
    for entry in providers.values():
        entry.update({"api_key": "shared-key"} if credential == "static" else {"key_cmd": "fixture-token-command"})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"providers": providers}))
    from hermes_cli.runtime_provider import _tag, resolve_runtime_provider
    with patch("agent.command_token_source.build_command_token_provider", side_effect=lambda *args: lambda: "fixture-token"), patch(
        "hermes_cli.runtime_provider._try_resolve_from_custom_pool", return_value=None
    ):
        rt = resolve_runtime_provider(requested=owner, target_model="fixture-model")
    # The owning rung survives the outer resolver's request-tag overwrite.
    rt = _tag(rt, "own")
    route = {"model": "fixture-model", "provider": "own"}
    if surface != "provider":
        route["base_url"] = URL
    if surface == "fixed":
        route["api_key"] = "fixed-key"
    parent, captured = _parent(), []
    with patch("tools.delegate_tool._load_config", return_value={"model_pool": {"standard": route}}), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider", return_value=rt
    ), patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=_fake_child(parent, captured)), patch(
        "tools.delegate_tool._run_batch", return_value='{"ok": true}'
    ), patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])) as live:
        result = json.loads(delegate_task(goal="owned identity", parent_agent=parent))
    if owner == "foreign":
        assert "error" in result, result
        assert not captured
        live.assert_not_called()
    else:
        assert result == {"ok": True}
        assert captured[0]["override_request_overrides"] == {"extra_body": {"owner": "own"}}
        key = captured[0]["override_api_key"]
        assert key == "fixed-key" if surface == "fixed" else callable(key) if credential == "callable" else key == "shared-key"
        assert copy.deepcopy(rt["request_overrides"]) == {"extra_body": {"owner": "own"}}


@pytest.mark.parametrize("provider,named", [("openai", False), ("openai", True), ("custom:own", True)])
def test_declared_alias_and_direct_alias_positive(tmp_path, monkeypatch, provider, named):
    from hermes_cli.config import atomic_config_write
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from tools.delegate_tool_config import _resolve_delegation_credentials
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    providers = {"own" if provider == "custom:own" else "openai": {
        "name": "Own Label", "base_url": URL, "api_key": "fixture-token",
    }} if named else {}
    atomic_config_write(tmp_path / "config.yaml", {"providers": providers})
    with patch("hermes_cli.runtime_provider._try_resolve_from_custom_pool", return_value=None):
        rt = resolve_runtime_provider(requested=provider, target_model="fixture-model")
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=rt):
        creds = _resolve_delegation_credentials({"provider": provider, "model": "fixture-model"}, None, exclusive=True)
    assert creds["provider"] == provider
    assert creds["base_url"] == (URL if named else "https://api.openai.com/v1")


@pytest.mark.parametrize("provider,mode,key", [("gemini", "google_genai", "fixture-token"), ("bedrock", "bedrock_converse", "no-key-required"), ("acp", "acp", "fixture-process"), ("custom", "chat_completions", lambda: "fixture-token")])
def test_valid_native_acp_and_callable_personality(provider, mode, key):
    from tools.delegate_tool_config import _resolve_delegation_credentials
    rt = {"provider": "google" if provider == "gemini" else provider, "api_mode": mode, "api_key": key,
          "request_overrides": {"extra_body": {"nested": [None, 7, {"sdk": True}]}}}
    if provider == "custom":
        rt["base_url"] = URL
    if provider == "acp":
        rt.update(command="python", args=["-u", "agent.py"])
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=rt), patch("shutil.which", return_value="/fixture/python"):
        creds = _resolve_delegation_credentials({"provider": provider, "model": "fixture-model"}, None)
    assert creds["api_mode"] == mode
    assert creds["request_overrides"] == rt["request_overrides"]
    if provider == "acp":
        assert creds["args"] == ["-u", "agent.py"]


@pytest.mark.parametrize("field,value", [(field, value) for field, value in _RUNTIME_FAULTS if field not in {"source", "credential_pool"}])
def test_inherited_consumed_fields_refuse_before_dispatch(field, value):
    parent = _parent()
    parent.requested_provider, parent.acp_command, parent.acp_args = "openrouter", None, []
    setattr(parent, {"command": "acp_command", "args": "acp_args"}.get(field, field), value)
    with patch("tools.delegate_tool._load_config", return_value={}), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools"
    ) as child, patch("tools.delegation_live_log.create_live_transcripts") as live:
        result = json.loads(delegate_task(goal="finite inherited projection", parent_agent=parent))
    assert "error" in result
    assert not child.called and not live.called


@pytest.mark.parametrize("requested,resolved", [("openai", "openrouter"), ("kimi", "kimi-coding")])
def test_named_declaration_precedes_registered_alias(tmp_path, monkeypatch, requested, resolved):
    from hermes_cli.config import atomic_config_write
    from tools.delegate_tool_config import _resolve_delegation_credentials
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {"providers": {requested: {"base_url": URL, "api_key": "fixture-token"}}})
    rt = {"provider": resolved, "api_key": "fixture-token", "base_url": URL, "api_mode": "chat_completions"}
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=rt), pytest.raises(ValueError, match="own provider"):
        _resolve_delegation_credentials({"provider": requested, "model": "fixture-model"}, None, exclusive=True)
