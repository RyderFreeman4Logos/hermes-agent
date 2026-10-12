"""Codex cache routing keeps a marked stable prefix when the suffix changes."""

import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.codex
    return get_transport("codex_responses")


def test_short_gap_same_route_reuses_marked_stable_prefix_key(transport):
    """A volatile suffix must not evict an unchanged marked prefix bucket."""
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
        )["prompt_cache_key"]

    first = key("first volatile suffix")
    assert first.startswith("pck_")
    assert key("second volatile suffix") == first
