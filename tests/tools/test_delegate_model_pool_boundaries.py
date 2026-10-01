"""Public regressions for model-pool route preparation and provenance."""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.redact import RedactingFormatter
from tools import async_delegation as ad
from tools.delegate_tool import delegate_task
from tools.delegate_tool_child_run import _lease_child_credential
from tools.delegate_tool_config import _resolve_child_credential_pool
from hermes_cli.config import validate_config_structure
from tools.process_registry import process_registry
from tools.process_registry_notifications import _format_batch_delegation, format_process_notification


def _parent():
    parent = MagicMock()
    parent.base_url = "https://parent.invalid/v1"
    parent.api_key = "fixture-parent-key"
    parent.provider = "openrouter"
    parent.api_mode, parent.request_overrides = "chat_completions", None
    parent.requested_provider, parent.acp_command, parent.acp_args = "openrouter", None, []
    parent.model = "parent-model"
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
    parent._fallback_chain = []
    parent.enabled_toolsets = ["terminal"]
    parent.disabled_toolsets = []
    parent.session_id = "fixture-parent-session"
    return parent

def _route(model: str, *, provider: str = "custom", port: int = 9, **extra):
    return {
        "provider": provider,
        "model": model,
        "base_url": f"http://127.0.0.1:{port}/v1",
        "api_key": f"fixture-{model}-key",
        "api_mode": "chat_completions",
        **extra,
    }


def _fake_child(parent, captured):
    def _build(**kwargs):
        captured.append(kwargs)
        child = SimpleNamespace(
            model=kwargs["model"],
            provider=kwargs["override_provider"],
            base_url=kwargs["override_base_url"],
            api_key=kwargs["override_api_key"],
            _delegate_role="leaf",
            _subagent_id=None,
            session_id="fixture-child",
            _delegate_depth=1,
            _interrupt_requested=False,
            tool_progress_callback=None,
            run_conversation=lambda **_kw: {"completed": True, "final_response": "ok", "api_calls": 0},
            get_activity_summary=lambda: {"api_call_count": 0, "current_tool": None, "max_iterations": 4},
            interrupt=lambda *a, **k: None,
            close=lambda: None,
        )
        parent._active_children.append(child)
        return child

    return _build


def _sync_result(_batch, _background):
    return json.dumps({"ok": True})


@pytest.mark.parametrize("runtime_key,should_refuse", [
    ("different-provider-key", True),
    ("named-tier-owned-key", False),
])
def test_exclusive_explicit_endpoint_enforces_named_provider_identity(runtime_key, should_refuse):
    url = "https://shared.invalid/v1"
    cfg = {"max_iterations": 4, "model_pool": {"standard": {
        "provider": "named-tier", "model": "tier-model", "base_url": url,
        "api_key": "fixture-tier-owned-key",
    }}}
    parent, captured = _parent(), []
    runtime = {
        "provider": "custom", "requested_provider": "different-provider",
        "model": "tier-model", "base_url": url, "api_key": runtime_key,
        "api_mode": "chat_completions", "source": f"custom_provider:{'named-tier' if not should_refuse else 'different-provider'}",
    }
    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime,
    ), patch(
        "hermes_cli.runtime_provider_custom._get_named_custom_provider",
        return_value={"name": "named-tier", "base_url": url, "api_key": "named-tier-owned-key"},
    ), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=_fake_child(parent, captured),
    ), patch("tools.delegate_tool._run_batch", side_effect=_sync_result):
        result = json.loads(delegate_task(goal="offline identity regression", parent_agent=parent))

    if should_refuse:
        assert "error" in result, result
        assert "did not resolve its own provider credentials" in result["error"]
        assert not captured
    else:
        assert result == {"ok": True}, result
        assert captured[0]["override_api_key"] == "fixture-tier-owned-key"
        assert captured[0]["override_provider"] == "custom"
        assert captured[0]["override_requested_provider"] == "named-tier"


@pytest.mark.parametrize("route,runtime,error", [
    ({"model": "tier-model"}, {}, "provider or base_url"),
    ({"model": "tier-model", "provider": "auto"}, {}, "auto"),
    ({**_route("tier-model"), "provider": "AUTO"}, {}, "auto"),
    ({"model": "tier-model", "base_url": "http://127.0.0.1:9/v1"}, {}, "api_key"),
    ({"model": "tier-model", "provider": "minimax"},
     {"provider": "openrouter", "requested_provider": "minimax", "api_key": "ambient-key",
      "base_url": "https://ambient.invalid/v1"}, "provider"),
    ({"model": "tier-model", "provider": "named-tier"},
     {"provider": "custom", "requested_provider": "named-tier", "api_key": "ambient-key",
      "base_url": "https://ambient.invalid/v1", "source": "local-runtime"}, "provider"),
    ({"model": "tier-model", "provider": "minimax", "base_url": "http://127.0.0.1:9/v1"},
     {"provider": "openrouter", "requested_provider": "minimax", "api_key": "ambient-key",
      "base_url": "http://127.0.0.1:9/v1"}, "provider"),
    ({**_route("tier-model"), "fallback_chain": [{"provider": "auto", "model": "backup"}]},
     {}, "fallback_chain"),
])
def test_exclusive_pool_refuses_ambient_routes_before_registry_child_construction(
    tmp_path, monkeypatch, route, runtime, error,
):
    from tools.registry import registry
    import yaml

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-key")
    cfg = {"delegation": {**_route("global-model"), "model_pool": {"standard": route}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime), patch(
        "run_agent.AIAgent"
    ) as constructor, patch("tools.delegation_live_log.create_live_transcripts") as transcripts:
        raw = registry.dispatch("delegate_task", {"goal": "offline"}, parent_agent=_parent())
        result = json.loads(raw) if isinstance(raw, str) else raw
    assert error in result["error"].lower(), result
    constructor.assert_not_called()
    transcripts.assert_not_called()


@pytest.mark.parametrize("route_kind", ["endpoint", "provider", "named"])
@pytest.mark.parametrize("fallback", [None, [], [{"provider": "minimax", "model": "owned-backup"}]])
@pytest.mark.parametrize("overrides", [None, {}, {"extra_body": {"tier_only": True}}])
def test_exclusive_pool_registry_constructor_uses_only_owned_route_across_homes(
    tmp_path, monkeypatch, route_kind, fallback, overrides,
):
    from tools.registry import registry
    import yaml

    parent = _parent()
    parent.acp_command = "parent-acp"
    parent.acp_args = ["parent-arg"]
    parent.request_overrides = {"extra_body": {"parent_only": True}}
    parent._fallback_chain = [{"provider": "openrouter", "model": "parent-backup"}]
    parent._credential_pool = MagicMock()
    parent._credential_pool.provider = "openrouter"
    parent._credential_pool.entries.return_value = []
    owned_pool = SimpleNamespace(provider="openrouter", has_credentials=lambda: True)
    parent.providers_allowed = ["parent-route"]
    homes = [tmp_path / "a", tmp_path / "b"]
    seen = []

    def constructor(**kwargs):
        seen.append(kwargs)
        child = MagicMock()
        child.provider = kwargs["provider"]
        child.requested_provider = kwargs["requested_provider"]
        child.base_url = kwargs["base_url"]
        child.api_key = kwargs["api_key"]
        child._credential_pool = None
        return child

    for home in homes:
        home.mkdir()
        owner = home.name
        route = {"provider": "openrouter", "model": f"tier-{owner}"}
        if route_kind == "endpoint":
            route = _route(f"tier-{owner}")
        elif route_kind == "named":
            route["provider"] = "named-tier"
        if fallback is not None:
            route["fallback_chain"] = fallback
        if overrides is not None:
            route["request_overrides"] = overrides
        cfg = {"model": {"provider": "openrouter", "default": "ambient-model"}, "delegation": {
            **_route("global-model"), "max_iterations": 4,
            "request_overrides": {"extra_body": {"global_only": True}},
            "fallback_providers": [{"provider": "openrouter", "model": "global-backup"}],
            "model_pool": {"standard": route},
        }}
        if route_kind == "named":
            cfg["providers"] = {"named-tier": {
                "base_url": f"https://tier-{owner}.invalid/v1", "api_key": f"owned-{owner}",
            }}
        (home / "config.yaml").write_text(yaml.safe_dump(cfg))

    with patch("run_agent.AIAgent", side_effect=constructor), patch(
        "tools.delegate_tool._run_batch", side_effect=_sync_result
    ), patch("tools.delegate_tool_config._loaded_pool", return_value=owned_pool) as pool_lookup:
        for home in (homes[0], homes[1], homes[0]):
            monkeypatch.setenv("HERMES_HOME", str(home))
            # Named custom providers exercise the real resolver. The built-in
            # provider's credential discovery is the only mocked auth boundary.
            runtime = {"provider": "openrouter", "base_url": f"https://tier-{home.name}.invalid/v1",
                       "api_key": f"owned-{home.name}", "api_mode": "chat_completions"}
            resolver = patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime)
            if route_kind == "provider":
                resolver.start()
            try:
                raw = registry.dispatch("delegate_task", {"goal": "offline"}, parent_agent=parent)
                result = json.loads(raw) if isinstance(raw, str) else raw
            finally:
                if route_kind == "provider":
                    resolver.stop()
            assert result == {"ok": True}, result
            actual = seen[-1]
            assert actual["model"] == f"tier-{home.name}"
            assert actual["provider"] == {"endpoint": "custom", "provider": "openrouter", "named": "named-tier"}[route_kind]
            assert actual["api_key"] == (f"fixture-tier-{home.name}-key" if route_kind == "endpoint" else f"owned-{home.name}")
            assert actual["base_url"] == ("http://127.0.0.1:9/v1" if route_kind == "endpoint" else runtime["base_url"])
            if route_kind == "provider":
                assert parent._active_children[-1]._credential_pool is owned_pool
            assert actual["request_overrides"] == (overrides or {})
            assert actual["fallback_model"] == (fallback or [])
            assert (actual["acp_command"], actual["acp_args"], actual["providers_allowed"]) == (None, [], None)
        if route_kind == "provider":
            assert pool_lookup.call_count == 3
            pool_lookup.assert_called_with("openrouter")
        else:
            assert all(call.args[0] != "custom" for call in pool_lookup.call_args_list)
    parent._credential_pool.acquire_lease.assert_not_called()

@pytest.mark.parametrize("pool,expected", [
    ("oops", "mapping"), ([], "mapping"), (["standard"], "mapping"),
    (0, "mapping"), (False, "mapping"), ("", "mapping"),
    ({" ": _route("x"), "standard": _route("s")}, "name"),
    ({1: _route("x"), "standard": _route("s")}, "name"),
    ({" standard ": _route("s")}, "name"),
    ({"standard": ["model"]}, "mapping"),
    ({"standard": {**_route("s"), "model": ["s"]}}, "model"),
    ({"standard": {**_route("s"), "provider": ["custom"]}}, "provider"),
    ({"standard": {**_route("s"), "base_url": ["url"]}}, "base_url"),
    ({"standard": {**_route("s"), "api_key": True}}, "api_key"),
    ({"standard": {**_route("s"), "fallback_chain": "backup"}}, "fallback_chain"),
    ({"standard": {**_route("s"), "fallback_chain": [{"provider": "custom"}]}}, "fallback_chain"),
    ({"standard": _route("s"), "fast": {"provider": "custom"}}, "model"),
    ({"fast": _route("f")}, "standard"),
])
def test_malformed_pool_class_refused_before_child_or_transcript(pool, expected):
    cfg = {"model": "global-model", "provider": "openrouter", "model_pool": pool}
    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools"
    ) as build, patch("tools.delegation_live_log.create_live_transcripts") as live:
        result = json.loads(delegate_task(goal="check route", parent_agent=_parent()))
    assert expected in result["error"], result
    build.assert_not_called()
    live.assert_not_called()
    assert any(
        issue.severity == "error" and expected in issue.message
        for issue in validate_config_structure({"delegation": cfg})
    )

@pytest.mark.parametrize("pool", [None, {}])
def test_absent_or_empty_pool_keeps_legacy_route(pool):
    cfg = {"model": "legacy-model", "model_pool": pool}
    captured = []
    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=_fake_child(_parent(), captured),
    ), patch("tools.delegate_tool._run_batch", side_effect=_sync_result):
        result = json.loads(delegate_task(goal="legacy", parent_agent=_parent()))
    assert result == {"ok": True}
    assert captured[0]["model"] == "legacy-model"
    assert not [i for i in validate_config_structure({"delegation": cfg}) if i.severity == "error"]

def test_malformed_pool_does_not_advertise_partial_tier_schema():
    from tools.delegate_tool import _build_dynamic_schema_overrides
    cfg = {"model_pool": {"standard": _route("s"), "fast": {"model": "oops"}}}
    with patch("tools.delegate_tool._load_config", return_value=cfg):
        props = _build_dynamic_schema_overrides()["parameters"]["properties"]
    assert "enum" not in props["model_profile"]
    assert "enum" not in props["tasks"]["items"]["properties"]["model_profile"]


@pytest.mark.parametrize("missing", ["model", "api_key"])
def test_incomplete_pool_route_refused_before_children_and_config_reports_it(missing):
    route = _route("pool-model")
    del route[missing]
    cfg = {"max_iterations": 4, "model_pool": {"standard": route}}
    parent = _parent()
    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools"
    ) as build:
        result = json.loads(delegate_task(goal="check route", parent_agent=parent))
    assert missing in result["error"]
    build.assert_not_called()
    assert any(missing in i.message and i.severity == "error" for i in validate_config_structure({"delegation": cfg}))


@pytest.mark.parametrize("same_endpoint", [True, False])
def test_named_pool_provider_can_resolve_its_own_endpoint_credential(same_endpoint):
    route = _route("pool-model", provider="named-provider")
    del route["api_key"]
    cfg = {"max_iterations": 4, "model_pool": {"standard": route}}
    parent = _parent()
    captured = []
    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value={"provider": "custom", "model": "pool-model", "base_url": route["base_url"] if same_endpoint else "http://127.0.0.1:8/v1", "api_key": "provider-key", "source": "custom_provider:named-provider"},
    ), patch("hermes_cli.runtime_provider_custom._get_named_custom_provider",
             return_value={"name": "named-provider", "base_url": route["base_url"], "api_key": "provider-key"}), patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=_fake_child(parent, captured)), patch(
        "tools.delegate_tool._run_batch", side_effect=_sync_result
    ):
        result = json.loads(delegate_task(goal="check route", parent_agent=parent))
    if same_endpoint:
        assert result == {"ok": True}
        assert captured[0]["override_api_key"] == "provider-key"
        assert captured[0]["override_api_key"] != parent.api_key
    else:
        assert "provider credentials" in result["error"]
        assert captured == []


def test_direct_endpoint_keeps_canonical_custom_identity_through_lease():
    cfg = {"max_iterations": 4, "model_pool": {"standard": _route("fixture-model", provider="openrouter")}}
    parent = _parent()
    parent_pool = MagicMock()
    parent_pool.acquire_lease.return_value = "parent-lease"
    parent_pool.current.return_value = SimpleNamespace(
        api_key="fixture-parent-key", base_url="https://parent.invalid/v1"
    )
    parent._credential_pool = parent_pool
    captured = []

    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=_fake_child(parent, captured),
    ), patch("tools.delegate_tool._run_batch", side_effect=_sync_result):
        payload = json.loads(delegate_task(goal="inspect route identity", parent_agent=parent))

    assert payload == {"ok": True}
    assert captured[0]["override_provider"] == "custom"
    child = SimpleNamespace(_swap_credential=MagicMock())
    with patch("agent.credential_pool.get_custom_provider_pool_key", return_value=None):
        child_pool = _resolve_child_credential_pool(
            captured[0]["override_provider"], parent, captured[0]["override_base_url"]
        )
    if child_pool is not None:
        child._credential_pool = child_pool
    _lease_child_credential(child)
    assert child_pool is None
    child._swap_credential.assert_not_called()
    assert captured[0]["override_base_url"] == "http://127.0.0.1:9/v1"
    assert captured[0]["override_api_key"] == "fixture-fixture-model-key"


def test_public_named_shared_endpoint_keeps_explicit_owner_before_dispatch(tmp_path, monkeypatch):
    import yaml

    url = "http://127.0.0.1:9/v1"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Shared endpoint, separate independently owned provider identities/credentials.
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"providers": {
        "named-a": {"base_url": url, "api_key": "provider-owned-a"},
        "named-b": {"base_url": url, "api_key": "provider-owned-b"},
    }}))
    cfg = {"model_pool": {
        "standard": {"provider": "named-a", "model": "fixture-m", "base_url": url, "api_key": "fixed-a"},
        "other": {"provider": "named-b", "model": "fixture-m", "base_url": url, "api_key": "fixed-b"},
    }}
    parent = _parent()
    foreign = SimpleNamespace(id="foreign", api_key="pool-a", base_url=url, last_status="ok")
    foreign_pool = MagicMock()
    foreign_pool.entries.return_value = [foreign]
    foreign_pool.acquire_lease.return_value = "foreign"
    monkeypatch.setattr("agent.credential_pool.get_custom_provider_pool_key", lambda base_url, provider_name=None: "custom:a")
    monkeypatch.setattr("tools.delegate_tool_config._loaded_pool", lambda key: foreign_pool)
    seen = []

    def fake_agent(**kwargs):
        child = MagicMock()
        child.provider = kwargs["provider"]
        child.requested_provider = kwargs["requested_provider"]
        child.base_url = kwargs["base_url"]
        child.api_key = kwargs["api_key"]
        child._credential_pool = None
        child._swap_credential.side_effect = lambda entry: setattr(child, "api_key", entry.api_key)
        seen.append(child)
        return child

    with patch("tools.delegate_tool._load_config", return_value=cfg), patch("run_agent.AIAgent", side_effect=fake_agent), patch(
        "tools.delegate_tool._run_batch", side_effect=lambda *_: json.dumps({"ok": True})
    ):
        result = json.loads(delegate_task(goal="offline", model_profile="other", parent_agent=parent))
    assert result == {"ok": True}
    assert seen[0].requested_provider == "named-b"
    assert seen[0].api_key == "fixed-b"
    assert _lease_child_credential(seen[0]) == (None, None)
    assert seen[0].api_key == "fixed-b"
    foreign_pool.acquire_lease.assert_not_called()
    from agent.agent_runtime_helpers import _rebind_primary_credential_pool
    seen[0]._credential_pool = foreign_pool  # a fallback's foreign pool before primary restore
    _rebind_primary_credential_pool(
        seen[0], "custom", "fixture-m", lambda _: True,
        lambda: foreign_pool, foreign_pool, True,
    )
    assert seen[0]._credential_pool is None
    seen[0]._swap_credential.assert_not_called()


@pytest.mark.parametrize("pooled", [False, True])
def test_provider_only_fixed_and_derived_keys_stay_owned_before_dispatch(pooled):
    url = "http://127.0.0.1:9/v1"
    parent = _parent()
    resolved = {"provider": "custom", "model": "fixture-m", "base_url": url,
                "api_key": "provider-owned", "api_mode": "chat_completions", "source": "custom_provider:named-b"}
    pool = MagicMock()
    pool.has_credentials.return_value = True
    pool.provider = "custom:named-b"
    if pooled:
        resolved.update(api_key="pool-owned", source="pool:custom:named-b", credential_pool=pool)
    seen = []

    def fake_agent(**kwargs):
        child = MagicMock()
        child.provider = kwargs["provider"]
        child.requested_provider = kwargs["requested_provider"]
        child.api_key = kwargs["api_key"]
        child.base_url = kwargs["base_url"]
        child._credential_pool = None
        seen.append(child)
        return child

    fixed = {"provider": "named-b", "model": "fixture-m", "api_key": "tier-owned"}
    derived = {"provider": "named-b", "model": "fixture-m", "base_url": url}
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=resolved), patch(
        "hermes_cli.runtime_provider_custom._get_named_custom_provider",
        return_value={"name": "named-b", "base_url": url, "api_key": "provider-owned"},
    ), patch(
        "agent.credential_pool.get_custom_provider_pool_key", return_value="custom:named-b"
    ) as key_lookup, patch("agent.credential_pool.load_pool", return_value=pool), patch(
        "tools.delegate_tool._run_batch", side_effect=lambda *_: json.dumps({"ok": True})
    ), patch("run_agent.AIAgent", side_effect=fake_agent):
        for route in (fixed, derived):
            with patch("tools.delegate_tool._load_config", return_value={"model_pool": {"standard": route}}):
                assert json.loads(delegate_task(goal="offline", parent_agent=parent)) == {"ok": True}
    assert (seen[0].requested_provider, seen[0].api_key, seen[0]._credential_pool) == (
        "named-b", "tier-owned", None,
    )
    assert (seen[1].requested_provider, seen[1].api_key) == ("named-b", "pool-owned" if pooled else "provider-owned")
    assert seen[1]._credential_pool is pool
    assert any(call.kwargs.get("provider_name") == "named-b" for call in key_lookup.call_args_list)
    pool.acquire_lease.assert_not_called()


def test_explicit_pool_mode_must_be_supported_before_public_dispatch():
    from tools.delegate_tool import _build_dynamic_schema_overrides
    for mode in ([], {}, True, 42, "unsupported-wire"):
        cfg = {"model_pool": {"standard": {**_route("s"), "api_mode": mode}}}
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "tools.delegation_live_log.create_live_transcripts"
        ) as live, patch("tools.delegate_tool._build_child_preserving_parent_tools") as build:
            payload = json.loads(delegate_task(goal="offline", parent_agent=_parent()))
            props = _build_dynamic_schema_overrides()["parameters"]["properties"]
        assert "api_mode" in payload["error"]
        assert "enum" not in props["model_profile"]
        assert any("api_mode" in i.message and i.severity == "error" for i in validate_config_structure({"delegation": cfg}))
        live.assert_not_called()
        build.assert_not_called()
    for mode in (None, "responses"):
        cfg = {"model_pool": {"standard": {**_route("s"), "api_mode": mode}}}
        assert not [i for i in validate_config_structure({"delegation": cfg}) if i.severity == "error"]


def test_fallback_route_log_allowlists_labels_without_inline_key():
    opaque_key = "opaque-fixture-value"
    cfg = {
        "max_iterations": 4,
        "model_pool": {
            "standard": _route(
                "primary", fallback_chain=[{"provider": "custom", "model": "backup", "api_key": opaque_key}]
            )
        },
    }
    parent = _parent()
    captured = []
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    logger = logging.getLogger("tools.delegate_tool")
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "tools.delegate_tool._build_child_preserving_parent_tools",
            side_effect=_fake_child(parent, captured),
        ), patch("tools.delegate_tool._run_batch", side_effect=_sync_result):
            payload = json.loads(delegate_task(goal="inspect safe logging", parent_agent=parent))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    text = stream.getvalue()
    assert payload == {"ok": True}
    assert "custom" in text and "backup" in text
    assert opaque_key not in text
    assert "api_key" not in text


def test_unknown_later_profile_rejects_before_transcripts_or_children():
    cfg = {"max_iterations": 4, "model_pool": {"standard": _route("standard-model")}}
    parent = _parent()
    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegation_live_log.create_live_transcripts",
        return_value=(None, [None, None], []),
    ) as live, patch("tools.delegate_tool._build_child_preserving_parent_tools") as build:
        payload = json.loads(
            delegate_task(
                tasks=[
                    {"goal": "inspect routing ownership", "model_profile": "standard"},
                    {"goal": "inspect result delivery", "model_profile": "missing-tier"},
                ],
                parent_agent=parent,
            )
        )

    assert "missing-tier" in payload["error"]
    assert (live.call_count, build.call_count) == (0, 0)
    assert parent._active_children == []


def test_all_fast_tasks_do_not_resolve_unused_standard_but_standard_is_required():
    fast = _route("fast-model", port=8)
    cfg = {
        "max_iterations": 4,
        "model_pool": {
            "standard": {"provider": "minimax", "model": "unused-standard"},
            "fast": fast,
        },
    }
    parent = _parent()
    captured = []
    resolved = []

    def _resolve(route_cfg, _parent_agent, *, exclusive=False):
        assert exclusive
        resolved.append(route_cfg.get("provider"))
        if route_cfg.get("provider") == "minimax":
            raise ValueError("unused MiniMax credentials unavailable")
        return {
            "model": route_cfg.get("model"),
            "provider": "custom",
            "base_url": route_cfg.get("base_url"),
            "api_key": route_cfg.get("api_key"),
            "api_mode": "chat_completions",
            "request_overrides": None,
        }

    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegate_tool._resolve_delegation_credentials", side_effect=_resolve
    ), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=_fake_child(parent, captured),
    ), patch("tools.delegate_tool._run_batch", side_effect=_sync_result):
        payload = json.loads(
            delegate_task(
                tasks=[{"goal": "use the fast route", "model_profile": "fast"}],
                parent_agent=parent,
            )
        )

    assert payload == {"ok": True}
    assert resolved == ["custom"]
    assert captured[0]["model"] == "fast-model"

    missing_standard = {"max_iterations": 4, "model_pool": {"fast": fast}}
    with patch("tools.delegate_tool._load_config", return_value=missing_standard):
        refused = json.loads(
            delegate_task(
                tasks=[{"goal": "use the fast route", "model_profile": "fast"}],
                parent_agent=_parent(),
            )
        )
    assert "standard" in refused["error"].lower()


def _run_public_background_routes(tmp_path, monkeypatch, tasks):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {
        "max_iterations": 4,
        "independent_completions": True,
        "model_pool": {
            "standard": _route("standard-model", provider="custom", port=9),
            "fast": _route("fast-model", provider="custom", port=8),
        },
    }
    parent = _parent()
    captured_children = []
    persisted = []

    class _NoRunExecutor:
        def submit(self, _fn):
            return SimpleNamespace(add_done_callback=lambda _cb: None)

    with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        side_effect=_fake_child(parent, captured_children),
    ), patch("tools.delegate_tool_dispatch._resolve_async_wake_sid", return_value="fixture-origin"), patch(
        "gateway.session_context.session_history_delivery_supported", return_value=True
    ), patch(
        "tools.delegate_tool_dispatch._resolve_async_session_key", return_value=("fixture-key", "fixture-ui")
    ), patch("tools.delegate_tool_dispatch._detach_child"), patch(
        "tools.delegate_tool_config._get_independent_completions", return_value=True
    ), patch("tools.async_delegation._records", {}), patch(
        "tools.async_delegation._persist_dispatch", side_effect=lambda record: persisted.append(dict(record))
    ), patch("tools.async_delegation._get_executor", return_value=_NoRunExecutor()), patch(
        "tools.async_delegation._ensure_stale_monitor"
    ), patch("tools.delegate_tool._get_max_async_children", return_value=4):
        handle = json.loads(
            delegate_task(tasks=tasks, background=True, parent_agent=parent)
        )

    manifest_path = tmp_path / "cache" / "delegation" / "live" / handle["delegation_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return captured_children, persisted, manifest


def test_public_sole_fast_task_keeps_route_in_manifest_and_persisted_record(tmp_path, monkeypatch):
    children, persisted, manifest = _run_public_background_routes(
        tmp_path, monkeypatch, [{"goal": "run only fast", "model_profile": "fast"}]
    )

    assert children[0]["model"] == "fast-model"
    assert manifest["model"] == "fast-model"
    assert (manifest["tasks"][0]["model"], manifest["tasks"][0]["provider"]) == (
        "fast-model", "custom"
    )
    assert [record["model"] for record in persisted] == ["fast-model"]


def test_public_independent_units_keep_routes_in_manifest_dispatch_and_completion(tmp_path, monkeypatch):
    _children, persisted, manifest = _run_public_background_routes(
        tmp_path,
        monkeypatch,
        [
            {"goal": "run fast independently", "model_profile": "fast"},
            {"goal": "run standard independently", "model_profile": "standard"},
        ],
    )

    assert [(t["model"], t["provider"]) for t in manifest["tasks"]] == [
        ("fast-model", "custom"),
        ("standard-model", "custom"),
    ]
    assert manifest["model"] is None
    assert [record["model"] for record in persisted] == ["fast-model", "standard-model"]

    rendered = _format_batch_delegation(
        {
            "role": "leaf",
            "model": None,
            "goals": ["run fast independently", "run standard independently"],
            "results": [
                {"task_index": 0, "status": "completed", "summary": "fast done", "model": "fast-model", "provider": "custom"},
                {"task_index": 1, "status": "completed", "summary": "standard done", "model": "standard-model", "provider": "custom"},
            ],
        },
        "deleg-fixture",
        1.0,
    )
    assert "Model: fast-model" in rendered
    assert "Model: standard-model" in rendered
    assert rendered.index("Model: fast-model") < rendered.index("fast done")
    assert rendered.index("Model: standard-model") < rendered.index("standard done")


class _SelectedRouteFailureChild:
    def __init__(self, *, model: str, provider: str, outcome: str):
        self.model = model
        self.provider = provider
        self.outcome = outcome
        self.session_id = f"fixture-{outcome}-child"
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self._interrupt_requested = False
        self.tool_progress_callback = None
        self._release = threading.Event()

    def run_conversation(self, **_kwargs):
        if self.outcome == "exception":
            raise RuntimeError("synthetic selected-route failure")
        self._release.wait(timeout=5.0)
        return {"completed": False, "final_response": "", "api_calls": 1}

    def get_activity_summary(self):
        return {"api_call_count": 1, "current_tool": None, "max_iterations": 4}

    def interrupt(self):
        self._release.set()

    def close(self):
        self._release.set()


@pytest.mark.parametrize(("outcome", "expected_status"), [("exception", "error"), ("timeout", "timeout")])
def test_public_background_route_errors_keep_selected_provenance(
    tmp_path, monkeypatch, outcome, expected_status,
):
    """Public result, durable completion, and formatter keep the route selected before the run."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {
        "max_iterations": 4,
        "model_pool": {
            "standard": _route("standard-model", port=9),
            "fast": _route("fast-model", port=8),
        },
    }
    parent = _parent()

    def _build(**kwargs):
        child = _SelectedRouteFailureChild(
            model=kwargs["model"], provider=kwargs["override_provider"], outcome=outcome,
        )
        parent._active_children.append(child)
        return child

    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    try:
        with patch("tools.delegate_tool._load_config", return_value=cfg), patch(
            "tools.delegate_tool._build_child_preserving_parent_tools", side_effect=_build,
        ), patch(
            "tools.delegate_tool_dispatch._resolve_async_wake_sid", return_value="",
        ), patch(
            "tools.delegate_tool._get_child_timeout", return_value=0.05 if outcome == "timeout" else None,
        ):
            handle = json.loads(delegate_task(
                goal="exercise selected route failure",
                model_profile="fast",
                background=True,
                parent_agent=parent,
            ))

            assert handle["status"] == "dispatched"
            deadline = time.monotonic() + 5.0
            event = None
            while time.monotonic() < deadline:
                if process_registry.completion_queue.empty():
                    time.sleep(0.02)
                    continue
                candidate = process_registry.completion_queue.get_nowait()
                if candidate.get("delegation_id") == handle["delegation_id"]:
                    event = candidate
                    break
            assert event is not None
            (result,) = event["results"]
            assert (result["status"], result["model"], result["provider"]) == (
                expected_status, "fast-model", "custom",
            )

            durable = ad.get_durable_delegation(event["delegation_id"])
            assert durable is not None
            (persisted,) = durable["result"]["results"]
            assert (persisted["status"], persisted["model"], persisted["provider"]) == (
                expected_status, "fast-model", "custom",
            )

            rendered = format_process_notification(event)
            assert "Model: fast-model" in rendered
            assert "Provider: custom" in rendered
            assert rendered.index("Model: fast-model") < rendered.index("(no summary")
    finally:
        deadline = time.monotonic() + 2.0
        while ad.active_count() and time.monotonic() < deadline:
            time.sleep(0.02)
        ad._reset_for_tests()
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
