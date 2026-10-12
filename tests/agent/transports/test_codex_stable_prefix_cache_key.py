"""Codex cache routing keeps the produced stable prefix when the suffix changes."""

import pytest

from agent.system_prompt import build_system_prompt
from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.codex
    return get_transport("codex_responses")


def test_short_gap_same_route_reuses_produced_stable_prefix_key(transport):
    """A rebuilt volatile suffix must not evict the key of the produced stable prefix."""
    from types import SimpleNamespace
    from unittest.mock import patch

    agent = SimpleNamespace(
        load_soul_identity=False, skip_context_files=True, valid_tool_names=[],
        _task_completion_guidance=False, _tool_use_enforcement=False, _environment_probe=False,
        _kanban_worker_guidance="", _memory_store=None, _memory_manager=None,
        model="gpt-5.6-luna", provider="openai", platform="", pass_session_id=False,
        session_id="same-route-session", _emit_status=lambda *_a, **_k: None,
    )
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
    ):
        first_prompt = build_system_prompt(agent)
    stable = agent._cached_system_prompt_static
    rebuilt = stable + "\n\nsecond volatile suffix"

    def key(prompt):
        return transport.build_kwargs(
            model="gpt-5.6-luna",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "same route"},
            ],
            tools=[],
            session_id="same-route-session",
            static_system_prefix=stable,
            base_url="https://api.openai.com/v1",
        )["prompt_cache_key"]

    first = key(first_prompt)
    assert first.startswith("pck_")
    assert key(rebuilt) == first
    assert key("unrelated prompt") != first


def test_marked_block_still_routes_without_the_string_prefix(transport):
    """Anthropic/Qwen marker policy stays: a marked first block is still the prefix."""
    def key(volatile):
        return transport.build_kwargs(
            model="gpt-5.6-luna",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "stable prefix", "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": volatile},
                    ],
                },
                {"role": "user", "content": "same route"},
            ],
            tools=[],
            session_id="same-route-session",
            base_url="https://api.openai.com/v1",
        )["prompt_cache_key"]

    assert key("first volatile suffix") == key("second volatile suffix")


def _codex_agent(memory):
    from tests.agent.transports.test_stable_prefix_cache_key import _prompt_agent

    agent = _prompt_agent(memory)
    agent.api_mode = "codex_responses"
    agent.model = "gpt-5.6-luna"
    agent.base_url = "https://api.openai.com/v1"
    agent._base_url_lower = agent.base_url.lower()
    agent._base_url_hostname = "api.openai.com"
    agent._is_codex_backend = lambda: False
    return agent


def _serialized_codex_key(agent):
    """Real Codex Responses serializer, local mock transport, no network."""
    import json

    import httpx
    from openai import OpenAI

    from agent.chat_completion_helpers import build_api_kwargs
    from agent.codex_runtime import _sanitize_consumer_codex_request
    from agent.sdk_transform_bypass import bypass_sdk_request_transform

    prompt = build_system_prompt(agent)
    request = build_api_kwargs(agent, [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "same route"},
    ])
    stream_kwargs = _sanitize_consumer_codex_request(agent, request)
    stream_kwargs["stream"] = True
    captured = {}

    def handler(http_request):
        captured.update(json.loads(http_request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=(
                'data: {"type":"response.completed","response":{"id":"resp_1",'
                '"output":[],"status":"completed"}}\n\n'
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAI(api_key="test-key", base_url=agent.base_url, http_client=http_client)
        stream = client.responses.create(**bypass_sdk_request_transform(stream_kwargs))
        list(stream)
    return captured.get("prompt_cache_key"), agent._cached_system_prompt_static


def test_codex_serialized_body_follows_rebuilt_stable_tier_only():
    """A rebuilt volatile suffix keeps the serialized Codex key; a rebuilt stable tier changes it."""
    first_key, stable = _serialized_codex_key(_codex_agent("memory one"))
    second_key, same_stable = _serialized_codex_key(_codex_agent("memory two"))
    changed = _codex_agent("memory two")
    changed.valid_tool_names = ["terminal"]
    changed_key, new_stable = _serialized_codex_key(changed)

    assert stable and stable == same_stable and new_stable != stable
    assert first_key and first_key.startswith("pck_") and first_key == second_key
    assert changed_key != first_key
