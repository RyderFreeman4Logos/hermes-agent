"""Stable-prefix cache routing after compression and for named custom Chat."""

import pytest

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


def test_post_compress_request_reuses_unchanged_stable_prefix_key(transport):
    """Compression may rebuild only the volatile system-prompt suffix."""
    marker = {"type": "ephemeral"}

    def key(volatile_suffix):
        return transport.build_kwargs(
            model="cache-model",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "stable prefix", "cache_control": marker},
                        {"type": "text", "text": volatile_suffix, "cache_control": marker},
                    ],
                },
                {"role": "user", "content": "next request"},
            ],
            tools=[],
            session_id="session-after-compress",
            supports_prompt_cache_key=True,
        )["prompt_cache_key"]

    assert key("rebuilt volatile suffix") == key("volatile before compression")


def test_named_custom_same_route_reuses_stable_prefix_key(transport):
    """Short-gap main requests keep one cache bucket for a stable prefix."""
    stable = {"type": "text", "text": "stable prefix", "cache_control": {"type": "ephemeral"}}

    def key(user_text):
        return transport.build_kwargs(
            model="grok-4.6",
            messages=[
                {"role": "system", "content": [stable, {"type": "text", "text": "volatile"}]},
                {"role": "user", "content": user_text},
            ],
            tools=_tools(),
            session_id="same-route-session",
            provider_name="custom:localrouter",
            base_url="https://localrouter.invalid/v1",
        )["prompt_cache_key"]

    first = key("first request")
    assert first.startswith("pck_")
    assert key("second request") == first


def test_named_custom_profile_request_reuses_stable_prefix_key(monkeypatch):
    """The resolved named-custom route emits a stable cache key on the wire."""
    from unittest.mock import MagicMock, patch

    from hermes_cli import runtime_provider
    from providers import get_provider_profile
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
    assert runtime["provider"] == "custom"
    assert runtime["requested_provider"] == "custom:localrouter"
    assert get_provider_profile(runtime["provider"]) is not None

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
    stable = {"type": "text", "text": "stable prefix", "cache_control": {"type": "ephemeral"}}

    def emitted_request(user_text, stable_text="stable prefix"):
        request = agent._build_api_kwargs([
            {"role": "system", "content": [{**stable, "text": stable_text}, {"type": "text", "text": "volatile"}]},
            {"role": "user", "content": user_text},
        ])
        agent.client.chat.completions.create(**request)
        return agent.client.chat.completions.create.call_args_list[-1].kwargs

    first = emitted_request("first request")
    second = emitted_request("second request")
    changed = emitted_request("third request", stable_text="changed prefix")
    assert second["prompt_cache_key"].startswith("pck_")
    assert len(second["prompt_cache_key"]) == len("pck_") + 24
    assert second["prompt_cache_key"] == first["prompt_cache_key"]
    assert changed["prompt_cache_key"] != second["prompt_cache_key"]
