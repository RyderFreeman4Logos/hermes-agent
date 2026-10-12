"""Stable-prefix cache routing after compression, without a custom-name capability grant."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.system_prompt import build_system_prompt
from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.chat_completions
    return get_transport("chat_completions")


def _tools(name="lookup"):
    return [{
        "type": "function",
        "function": {
            "name": name,
            "description": "Look something up.",
            "parameters": {"type": "object", "properties": {}},
        },
    }]


def _agent_with_rebuilt_prompt(volatile):
    """The production prefix producer, then the post-compression rebuild of its volatile tier."""
    agent = SimpleNamespace(
        load_soul_identity=False, skip_context_files=True, valid_tool_names=[],
        _task_completion_guidance=False, _tool_use_enforcement=False, _environment_probe=False,
        _kanban_worker_guidance="", _memory_store=None, _memory_manager=None,
        model="gpt-cache-model", provider="openai", platform="", pass_session_id=False,
        session_id="session-after-compress", _emit_status=lambda *_a, **_k: None,
    )
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
    ):
        first = build_system_prompt(agent)
    stable = agent._cached_system_prompt_static
    assert stable and first.startswith(stable)
    rebuilt = stable + "\n\n" + volatile
    return stable, first, rebuilt


def test_post_compress_request_reuses_unchanged_stable_prefix_key(transport):
    """A rebuilt volatile suffix must not change the key of the produced stable prefix."""
    stable, before, after = _agent_with_rebuilt_prompt("rebuilt volatile suffix")

    def key(prompt):
        return transport.build_kwargs(
            model="gpt-cache-model",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "next request"},
            ],
            tools=[],
            session_id="session-after-compress",
            static_system_prefix=stable,
            supports_prompt_cache_key=True,
        )["prompt_cache_key"]

    assert key(after) == key(before)
    assert key(after) != key("completely different prompt")


def test_named_custom_without_capability_emits_no_key(transport):
    """Naming an endpoint custom:<name> is not a prompt_cache_key grant."""
    from providers.base import ProviderProfile

    stable, prompt, _rebuilt = _agent_with_rebuilt_prompt("volatile")
    kwargs = transport.build_kwargs(
        model="strict-model",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": "request"}],
        tools=_tools(),
        session_id="same-route-session",
        provider_name="custom:strict-endpoint",
        provider_profile=ProviderProfile(name="custom", supports_prompt_cache_key=False),
        base_url="https://strict-endpoint.invalid/v1",
        static_system_prefix=stable,
    )
    assert "prompt_cache_key" not in kwargs
    assert "prompt_cache_key" not in kwargs.get("extra_body", {})


def test_named_custom_profile_request_reuses_stable_prefix_key(monkeypatch):
    """An explicit profile capability still keys the produced stable prefix."""
    from unittest.mock import MagicMock

    from hermes_cli import runtime_provider
    from providers import get_provider_profile
    from providers.base import ProviderProfile
    from run_agent import AIAgent

    custom_provider = {
        "name": "localrouter",
        "base_url": "https://localrouter.invalid/v1",
        "api_key": "localrouter-test-key",
    }
    monkeypatch.setattr(
        runtime_provider, "_get_named_custom_provider",
        lambda requested: custom_provider if requested == "custom:localrouter" else None,
    )
    monkeypatch.setattr("hermes_cli.config.load_config", dict)
    runtime = runtime_provider.resolve_runtime_provider(
        requested="custom:localrouter", target_model="local-model",
    )
    profile = get_provider_profile(runtime["provider"])
    assert profile is not None
    assert profile.supports_prompt_cache_key is False
    capable = ProviderProfile(name=profile.name, supports_prompt_cache_key=True)

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url=runtime["base_url"],
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent.client = MagicMock()
    agent.provider = runtime["provider"]
    agent.requested_provider = runtime["requested_provider"]
    agent.base_url = runtime["base_url"]
    agent.api_mode = runtime["api_mode"]
    agent.model = "local-model"
    agent.session_id = "same-route-session"
    stable, _first, rebuilt = _agent_with_rebuilt_prompt("rebuilt volatile suffix")
    object.__setattr__(agent, "_cached_system_prompt_static", stable)

    def emitted_request(user_text, prompt=rebuilt):
        request = agent._build_api_kwargs([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ])
        agent.client.chat.completions.create(**request)
        return agent.client.chat.completions.create.call_args_list[-1].kwargs

    with patch("providers.get_provider_profile", return_value=capable):
        first = emitted_request("first request")
        second = emitted_request("second request")
        changed = emitted_request("third request", prompt="different stable" + rebuilt[len(stable):])
    assert second["prompt_cache_key"].startswith("pck_")
    assert len(second["prompt_cache_key"]) == len("pck_") + 24
    assert second["prompt_cache_key"] == first["prompt_cache_key"]
    assert changed["prompt_cache_key"] != second["prompt_cache_key"]
