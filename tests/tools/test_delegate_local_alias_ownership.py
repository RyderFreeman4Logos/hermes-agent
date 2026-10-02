"""Local fallback aliases retain endpoint ownership without waiving named identity (#359)."""
from contextlib import ExitStack, contextmanager
import json
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.test_delegate_pool_runtime_transitions import offline as offline, parent

LOCAL_URL = "http://127.0.0.1:11434/v1"


@contextmanager
def dispatched_child(tmp_path, hop, providers=None, resolver=None):
    import hermes_yaml as yaml
    import run_agent
    from tools.registry import registry

    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openrouter", "default": "ambient-model"},
        "providers": providers or {},
        "delegation": {"provider": "openrouter", "model": "forbidden-global", "model_pool": {
            "standard": {"provider": "custom", "model": "primary-model", "api_key": "fixture-primary",
                         "base_url": "http://127.0.0.1:9/v1", "fallback_chain": [hop]},
        }},
    }))
    original = run_agent.AIAgent
    children, clients = [], []

    def construct(**kwargs):
        child = original(**kwargs)
        children.append(child)
        return child

    def client(**kwargs):
        clients.append(kwargs)
        result = MagicMock()
        result.base_url, result.api_key = kwargs["base_url"], kwargs.get("api_key")
        return result

    with ExitStack() as stack:
        stack.enter_context(patch("model_tools.get_tool_definitions", return_value=[]))
        stack.enter_context(patch("model_tools.check_toolset_requirements", return_value={}))
        stack.enter_context(patch("agent.process_bootstrap.OpenAI", side_effect=client))
        stack.enter_context(patch("agent.auxiliary_client._create_openai_client", side_effect=client))
        stack.enter_context(patch("agent.context_compressor.get_model_context_length", return_value=200_000))
        stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=200_000))
        stack.enter_context(patch.object(run_agent, "AIAgent", side_effect=construct))
        stack.enter_context(patch("tools.delegate_tool._run_batch", return_value='{"offline":true}'))
        if resolver:
            stack.enter_context(patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolver))
        try:
            result = registry.dispatch("delegate_task", {"goal": "local alias ownership"}, parent_agent=parent())
            assert (json.loads(result) if isinstance(result, str) else result) == {"offline": True}
            child, = children
            assert (child.model, child.api_key) == ("primary-model", "fixture-primary")
            yield child, clients
        finally:
            for child in children:
                child.close()


@pytest.mark.parametrize("provider,key", [
    ("custom", None), ("ollama", None), ("vllm", None), ("llamacpp", None), ("ollama", "fixture-hop"),
])
def test_local_alias_reaches_real_constructor_and_activation(tmp_path, provider, key):
    hop = {"provider": provider, "model": "bare-model", "base_url": LOCAL_URL}
    if key:
        hop["api_key"] = key
    with dispatched_child(tmp_path, hop) as (child, clients):
        assert child._fallback_chain
        assert child._try_activate_fallback()
        assert child.model == "bare-model"
        assert str(child.base_url).rstrip("/") == LOCAL_URL
        assert clients[-1]["api_key"] == (key or "no-key-required")
        assert len(clients) == 2


@pytest.mark.parametrize("case", [
    "named-alias-own", "named-alias-direct", "named-alias-keyless", "foreign-inline", "foreign-key-env",
    "foreign-callable", "endpoint-mismatch", "ambient-key", "named-remote-keyless-local",
])
def test_alias_admission_preserves_named_and_endpoint_fences(tmp_path, monkeypatch, case):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    hop = {"provider": "ollama", "model": "bare-model", "base_url": LOCAL_URL}
    providers = {"ollama": {"base_url": LOCAL_URL, "api_key": "fixture-owned"},
                 "foreign": {"base_url": LOCAL_URL, "api_key": "fixture-foreign"}}
    calls = []
    token_provider = MagicMock(return_value="fixture-token-never-minted")
    if case in {"foreign-inline", "named-alias-direct"}:
        hop["api_key"] = "fixture-hop"
    if case == "foreign-key-env":
        hop["key_env"] = "ALIAS_TEST_KEY"
        monkeypatch.setenv("ALIAS_TEST_KEY", "fixture-hop")
    if case == "foreign-callable":
        providers = {name: {"base_url": LOCAL_URL, "key_cmd": "fixture-command-never-executed"}
                     for name in providers}
    if case in {"endpoint-mismatch", "ambient-key"}:
        providers = {}
    if case == "named-remote-keyless-local":
        hop["provider"] = "remote-owner"
        providers = {"remote-owner": {"base_url": "https://remote.invalid/v1"}}

    def resolve(**kwargs):
        # Faults alter a real resolver result, not an invented provider bundle.
        requested = "foreign" if case.startswith("foreign-") else kwargs["requested"]
        args = {**kwargs, "requested": requested}
        if case == "named-alias-own" or case.startswith("foreign-"):
            args.pop("explicit_base_url", None)
        if case == "named-remote-keyless-local":
            args["requested"] = "vllm"
        runtime = resolve_runtime_provider(**args)
        if case == "endpoint-mismatch":
            runtime["base_url"] = "http://127.0.0.1:8000/v1"
        if case == "ambient-key":
            runtime["api_key"] = "fixture-ambient"
        runtime["requested_provider"] = kwargs["requested"]
        calls.append(runtime)
        return runtime

    with patch("agent.command_token_source.build_command_token_provider", return_value=token_provider):
        with dispatched_child(tmp_path, hop, providers, resolve) as (child, clients):
            admitted = case == "named-alias-own"
            assert bool(child._fallback_chain) is admitted
            assert child._try_activate_fallback() is admitted
            assert len(clients) == (2 if admitted else 1)
            if admitted:
                assert child.model == "bare-model"
                assert clients[-1]["api_key"] == "fixture-owned"
            else:
                assert child.model == "primary-model"
            assert calls
            if case.startswith("foreign-"):
                assert calls[0]["source"] in {"pool:custom:foreign", "custom_provider:foreign"}
            token_provider.assert_not_called()


@pytest.mark.parametrize("credential", ["inline", "key_env", "api_key_env", "callable"])
def test_credential_freshness_at_activation(tmp_path, monkeypatch, credential):
    """Admission validates env keys but must not pin them for the same child's recovery."""
    from agent.secret_scope import (
        get_secret, is_multiplex_active, reset_secret_scope, set_multiplex_active, set_secret_scope,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("ROTATION_KEY", "fixture-launch")
    previous_mode = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for scope in ("a", "b", "a"):
            home = tmp_path / scope
            home.mkdir(exist_ok=True)
            before, after = f"fixture-before-{scope}", f"fixture-after-{scope}"
            hop = {"provider": "rotation-owner", "model": "bare-backup"}
            provider = {"base_url": "https://rotation.invalid/v1", "api_key": "fixture-provider"}
            if credential == "inline":
                hop.update(api_key="fixture-inline", key_env="ROTATION_KEY")
            elif credential == "callable":
                provider = {"base_url": "https://rotation.invalid/v1", "key_cmd": "fixture-not-executed"}
            else:
                hop[credential] = "ROTATION_KEY"
            home_token = set_hermes_home_override(home)
            secret_token = set_secret_scope({"ROTATION_KEY": before}, profile_home=str(home))
            try:
                with patch("agent.command_token_source._mint", side_effect=lambda *args: (get_secret("ROTATION_KEY"), 120)) as mint:
                    with dispatched_child(home, hop, {"rotation-owner": provider}) as (child, clients):
                        assert child._fallback_chain
                        mint.assert_not_called()
                        rotated = set_secret_scope({"ROTATION_KEY": after}, profile_home=str(home))
                        try:
                            assert child._try_activate_fallback()
                            mint.assert_not_called()
                            key = clients[-1]["api_key"]
                            assert (key() if callable(key) else key) == ("fixture-inline" if credential == "inline" else after)
                            assert len(clients) == 2
                        finally:
                            reset_secret_scope(rotated)
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)
    finally:
        set_multiplex_active(previous_mode)


@pytest.mark.parametrize("key_field", ["key_env", "api_key_env"])
@pytest.mark.parametrize("replacement", [{}, {"ROTATION_KEY": "   "}, None], ids=["removed", "empty", "unscoped"])
def test_credential_freshness_missing_key_refuses_ambient(tmp_path, monkeypatch, key_field, replacement):
    from agent.secret_scope import (
        is_multiplex_active, reset_secret_scope, set_multiplex_active, set_secret_scope,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("ROTATION_KEY", "fixture-launch")
    previous_mode = is_multiplex_active()
    set_multiplex_active(True)
    home_token = set_hermes_home_override(tmp_path)
    secret_token = set_secret_scope({"ROTATION_KEY": "fixture-before"}, profile_home=str(tmp_path))
    hop = {"provider": "rotation-owner", "model": "bare-backup", key_field: "ROTATION_KEY"}
    providers = {"rotation-owner": {"base_url": "https://rotation.invalid/v1", "api_key": "fixture-provider"}}
    try:
        with dispatched_child(tmp_path, hop, providers) as (child, clients):
            assert child._fallback_chain
            expired = set_secret_scope(replacement, profile_home=str(tmp_path))
            try:
                assert child._try_activate_fallback() is False
                assert (child.model, child.api_key) == ("primary-model", "fixture-primary")
                assert len(clients) == 1
            finally:
                reset_secret_scope(expired)
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_mode)
