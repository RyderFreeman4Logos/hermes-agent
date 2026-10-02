"""Offline public constructor/restore regressions for named delegation routes."""
import json
import socket
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

URL = "https://shared.invalid/v1"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    def denied(*args, **kwargs):
        raise AssertionError("NO NETWORK ALLOWED")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def parent():
    return SimpleNamespace(
        model="parent-m", provider="custom", requested_provider="parent",
        base_url="https://parent.invalid/v1", api_key="fixture-parent", api_mode="chat_completions",
        _delegate_depth=0, _active_children=[], _active_children_lock=threading.Lock(),
        _session_db=None, session_id="offline-parent", enabled_toolsets=[], disabled_toolsets=[],
        tool_progress_callback=None, thinking_callback=None, _print_fn=None, _fallback_chain=[],
        _interrupt_requested=False, platform="subagent",
    )


def install_config(tmp_path, route):
    from hermes_cli.config import atomic_config_write
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "custom", "default": "parent-m", "base_url": "https://parent.invalid/v1"},
        "providers": {
            "owner-a": {"base_url": URL, "api_key": "fixture-provider-a", "api_mode": "chat_completions"},
            "owner-b": {"base_url": URL, "api_key": "fixture-provider-b", "api_mode": "chat_completions"},
        },
        "delegation": {"max_spawn_depth": 1, "model": "forbidden-global-m", "provider": "forbidden-global-p",
                       "model_pool": {"standard": route}},
    })


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("provider,expected", [
    ("owner-b", "codex_responses"), ("bedrock", "bedrock_converse"),
    ("google", "google_genai"), ("vertex", "google_genai"),
])
def test_public_tier_mode_reaches_constructor(tmp_path, direct, fixed, provider, expected):
    from tools.delegate_tool import delegate_task
    route = {"provider": provider, "model": "route-m", "api_mode": "responses"}
    if direct:
        route["base_url"] = URL
    if fixed:
        route["api_key"] = "fixture-fixed"
    install_config(tmp_path, route)
    captured = []

    def build(**kw):
        captured.append(kw)
        return SimpleNamespace(**kw, _credential_pool=None, session_id="offline-child")

    with ExitStack() as stack:
        if provider != "owner-b":
            # No native SDK/auth/network; the real resolver remains active for named routes.
            stack.enter_context(patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                "provider": provider, "base_url": URL, "api_key": "fixture-native", "api_mode": expected,
            }))
        stack.enter_context(patch("run_agent.AIAgent", side_effect=build))
        stack.enter_context(patch("tools.delegate_tool._run_batch", return_value='{"offline":true}'))
        result = json.loads(delegate_task(goal="offline route test", parent_agent=parent()))
    assert result == {"offline": True}, result
    assert captured[0]["api_mode"] == expected
    assert captured[0]["api_key"] == (
        "fixture-fixed" if fixed else "fixture-provider-b" if provider == "owner-b" else "fixture-native"
    )


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("cooldown_owner", ["none", "foreign", "primary", "monotonic"])
def test_public_fallback_restore_uses_only_primary_owner(tmp_path, direct, fixed, cooldown_owner):
    import run_agent
    from tools.delegate_tool import delegate_task
    from tools.delegate_tool_child_run import _lease_child_credential

    primary_overrides = {"extra_body": {"route": "primary"}}
    route = {"provider": "owner-b", "model": "fixture-primary-m", "api_mode": "chat_completions",
             "request_overrides": primary_overrides,
             "fallback_chain": [{"provider": "owner-a", "model": "fixture-fallback-m", "api_key": "fixture-owner-a"}]}
    if direct:
        route["base_url"] = URL
    if fixed:
        route["api_key"] = "fixture-fixed-b"
    install_config(tmp_path, route)
    original = run_agent.AIAgent
    children = []
    cooldown = {"owner-a": None, "owner-b": None}
    own = SimpleNamespace(provider="owner-b", has_credentials=lambda: True,
                          has_available=lambda **kw: False, next_available_at=lambda **kw: cooldown["owner-b"])
    foreign = SimpleNamespace(provider="owner-a", has_credentials=lambda: True,
                              has_available=lambda **kw: cooldown["owner-a"] is None,
                              next_available_at=lambda **kw: cooldown["owner-a"])

    def build(**kw):
        child = original(**kw)
        children.append(child)
        return child

    with ExitStack() as stack:
        stack.enter_context(patch("model_tools.get_tool_definitions", return_value=[]))
        stack.enter_context(patch("model_tools.check_toolset_requirements", return_value={}))
        stack.enter_context(patch("agent.process_bootstrap.OpenAI"))
        stack.enter_context(patch("agent.context_compressor.get_model_context_length", return_value=200_000))
        stack.enter_context(patch.object(run_agent, "AIAgent", side_effect=build))
        stack.enter_context(patch("tools.delegate_tool._run_batch", return_value='{"offline":true}'))
        result = json.loads(delegate_task(goal="real constructor offline", parent_agent=parent()))
        assert result == {"offline": True}, result
        child, = children
        try:
            key = "fixture-fixed-b" if fixed else "fixture-provider-b"
            assert child.api_key == key
            if fixed:
                assert _lease_child_credential(child) == (None, None)
            client = MagicMock()
            client.base_url, client.api_key = URL, "fixture-foreign"
            with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, None)), \
                    patch("agent.model_metadata.get_model_context_length", return_value=200_000), \
                    patch("agent.credential_pool.load_pool", return_value=foreign):
                assert child._try_activate_fallback() is True
            assert child._credential_pool is foreign
            assert child.model == "fixture-fallback-m"
            if cooldown_owner in {"foreign", "primary"}:
                cooldown["owner-a" if cooldown_owner == "foreign" else "owner-b"] = time.time() + 3600
            child._rate_limited_until = time.monotonic() + 3600 if cooldown_owner == "monotonic" else 0
            with patch("agent.credential_pool.load_pool", side_effect=lambda owner: own if owner == "owner-b" else foreign) as load:
                restored = child._restore_primary_runtime()
            blocked = cooldown_owner == "monotonic" or (not fixed and cooldown_owner == "primary")
            assert restored is not blocked
            if blocked:
                assert (child.model, child.api_key, child._credential_pool) == (
                    "fixture-fallback-m", "fixture-foreign", foreign,
                )
            else:
                assert (child.model, child.api_key, child._credential_pool) == (
                    "fixture-primary-m", key, None if fixed else own,
                )
                assert child.requested_provider == "owner-b"
                assert child.request_overrides == primary_overrides
            if fixed or cooldown_owner == "monotonic":
                load.assert_not_called()
            else:
                load.assert_called_once_with("owner-b")
        finally:
            child.close()


@pytest.mark.parametrize("credential", ["inline", "key_env", "provider", "callable"])
@pytest.mark.parametrize("owner", ["owner-b", "owner-a"])
@pytest.mark.parametrize("pool_enabled", [True, False])
def test_public_fallback_ownership_across_scopes(tmp_path, monkeypatch, credential, owner, pool_enabled):
    """Explicit hop keys cannot waive identity, even on equal-URL named routes."""
    import hermes_yaml as yaml
    from hermes_cli.fallback_config import resolve_entry_api_key
    from agent.secret_scope import (
        is_multiplex_active, reset_secret_scope, set_multiplex_active, set_secret_scope,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from tools.registry import registry

    monkeypatch.setenv("FALLBACK_TEST_KEY", "fixture-launch-profile")
    previous_mode = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for scope in ("a", "b", "a"):
            home = tmp_path / scope
            home.mkdir(exist_ok=True)
            expected_key = f"fixture-hop-{scope}"
            hop = {"provider": "owner-b", "model": "bare-model"}
            if credential == "inline":
                hop["api_key"] = expected_key
            elif credential == "key_env":
                hop["key_env"] = "FALLBACK_TEST_KEY"
            providers = {name: {"base_url": URL, "api_key": f"fixture-{name}-{scope}"}
                         for name in ("owner-a", "owner-b")}
            if credential == "callable":
                for value in providers.values():
                    value.pop("api_key")
                    value["key_cmd"] = "fixture-command-not-executed"
            (home / "config.yaml").write_text(yaml.safe_dump({
                "providers": providers,
                "delegation": {"model_pool": {"standard": {
                    "provider": "custom", "model": "primary-model",
                    "base_url": "http://127.0.0.1:9/v1", "api_key": "fixture-primary",
                    "fallback_chain": [hop],
                }}},
            }))
            home_token = set_hermes_home_override(home)
            secret_token = set_secret_scope({"FALLBACK_TEST_KEY": expected_key}, profile_home=str(home))
            captured, runtimes = [], []
            token_provider = MagicMock(return_value="fixture-token-not-requested")

            def resolve(**kwargs):
                # Resolve a real named declaration, then simulate a resolver identity fault.
                runtime = resolve_runtime_provider(**{**kwargs, "requested": owner})
                runtime["requested_provider"] = kwargs["requested"]
                runtimes.append(runtime)
                return runtime

            def build(**kwargs):
                captured.append(kwargs)
                return SimpleNamespace(**kwargs, _credential_pool=None, session_id="offline-child")

            try:
                with ExitStack() as stack:
                    if not pool_enabled:
                        stack.enter_context(patch("hermes_cli.runtime_provider._try_resolve_from_custom_pool", return_value=None))
                    stack.enter_context(patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve))
                    stack.enter_context(patch("agent.command_token_source.build_command_token_provider", return_value=token_provider))
                    stack.enter_context(patch("run_agent.AIAgent", side_effect=build))
                    stack.enter_context(patch("tools.delegate_tool._run_batch", return_value='{"offline":true}'))
                    raw = registry.dispatch("delegate_task", {"goal": "owned fallback"}, parent_agent=parent())
                    result = json.loads(raw) if isinstance(raw, str) else raw
                assert result == {"offline": True}
                assert captured[0]["model"] == "primary-model"
                chain = captured[0]["fallback_model"]
                assert bool(chain) is (owner == "owner-b")
                assert runtimes[0]["source"] in {f"custom_provider:{owner}", f"pool:custom:{owner}"}
                if owner == "owner-b":
                    assert chain[0]["provider"] == "owner-b"
                    if credential in {"inline", "key_env"}:
                        assert resolve_entry_api_key(chain[0]) == expected_key
                        if credential == "key_env":
                            assert chain[0]["key_env"] == "FALLBACK_TEST_KEY"
                            assert "api_key" not in chain[0]
                    elif credential == "callable":
                        assert runtimes[0]["api_key"] is token_provider
                    else:
                        assert runtimes[0]["api_key"] == f"fixture-owner-b-{scope}"
                token_provider.assert_not_called()
                assert all(entry.get("api_key") != "fixture-launch-profile" for entry in chain)
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)
    finally:
        set_multiplex_active(previous_mode)


@pytest.mark.parametrize("failure", ["auth", "resolver", "identity"])
def test_public_fallback_diagnostic_is_actionable_without_error_payload(tmp_path, caplog, failure):
    from hermes_cli.auth import AuthError
    from tools.registry import registry

    canary = "opaque-secret-canary"
    install_config(tmp_path, {
        "provider": "custom", "model": "primary-model", "base_url": "http://127.0.0.1:9/v1",
        "api_key": "fixture-primary", "fallback_chain": [{"provider": "owner-b", "model": "bare-model"}],
    })
    captured = []

    def build(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(**kwargs, _credential_pool=None, session_id="offline-child")

    fault = {
        "auth": {"side_effect": AuthError(f"Bearer {canary} https://user:{canary}@private.invalid", code=canary)},
        "resolver": {"side_effect": OSError(canary)},
        "identity": {"return_value": {"provider": "custom", "base_url": URL, "api_key": canary,
                                       "source": "custom_provider:owner-a"}},
    }[failure]
    with caplog.at_level("INFO", logger="tools.delegate_tool"), \
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", **fault), \
            patch("run_agent.AIAgent", side_effect=build), \
            patch("tools.delegate_tool._run_batch", return_value='{"offline":true}'):
        raw = registry.dispatch("delegate_task", {"goal": "safe reason"}, parent_agent=parent())
        result = json.loads(raw) if isinstance(raw, str) else raw
    assert result == {"offline": True}
    assert captured[0]["fallback_model"] == []
    assert canary not in caplog.text
    assert "private.invalid" not in caplog.text
    assert {
        "auth": "authentication failed; configure credentials or run hermes auth",
        "resolver": "runtime resolution failed; check provider configuration",
        "identity": "provider ownership mismatch; check provider name, endpoint and credentials",
    }[failure] in caplog.text
