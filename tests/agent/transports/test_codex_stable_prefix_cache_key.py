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
