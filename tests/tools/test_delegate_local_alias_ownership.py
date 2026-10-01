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
