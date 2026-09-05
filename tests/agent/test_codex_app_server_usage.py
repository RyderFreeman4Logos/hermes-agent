"""Regression tests for cache telemetry provenance across usage surfaces."""

from types import SimpleNamespace

import pytest

from agent.codex_runtime import _record_codex_app_server_usage
from agent.usage_pricing import normalize_usage
from tui_gateway import server


_ABSENT = object()


@pytest.mark.parametrize(
    ("api_mode", "usage_factory"),
    [
        (
            "chat_completions",
            lambda marker: SimpleNamespace(
                prompt_tokens=1_000,
                completion_tokens=100,
                prompt_tokens_details=(
                    SimpleNamespace()
                    if marker is _ABSENT
                    else SimpleNamespace(cached_tokens=marker)
                ),
            ),
        ),
        (
            "codex_responses",
            lambda marker: SimpleNamespace(
                input_tokens=1_000,
                output_tokens=100,
                input_tokens_details=(
                    SimpleNamespace()
                    if marker is _ABSENT
                    else SimpleNamespace(cached_tokens=marker)
                ),
            ),
        ),
        (
            "anthropic_messages",
            lambda marker: SimpleNamespace(
                input_tokens=1_000,
                output_tokens=100,
                **({} if marker is _ABSENT else {"cache_read_input_tokens": marker}),
            ),
        ),
        (
            "codex_app_server",
            lambda marker: SimpleNamespace(
                input_tokens=1_000,
                output_tokens=100,
                **({} if marker is _ABSENT else {"cached_input_tokens": marker}),
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ("marker", "expected_telemetry"),
    [(_ABSENT, "unavailable"), (None, "unavailable"), (0, "reported")],
    ids=["absent", "null", "explicit-zero"],
)
def test_normalize_usage_distinguishes_unavailable_from_explicit_zero(
    api_mode, usage_factory, marker, expected_telemetry
):
    usage = normalize_usage(
        usage_factory(marker),
        provider="anthropic" if api_mode == "anthropic_messages" else "openai",
        api_mode=api_mode,
    )

    assert usage.cache_read_tokens == 0
    assert usage.cache_telemetry == expected_telemetry


def _make_agent() -> SimpleNamespace:
    return SimpleNamespace(
        session_api_calls=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0,
        session_cost_status="unknown",
        session_cost_source="none",
        model="gpt-5.3-codex",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="",
        context_compressor=None,
        _session_db=None,
        session_id=None,
    )


def test_codex_app_server_latches_usage_and_notifies_tui():
    events = []
    agent = _make_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    usage = _record_codex_app_server_usage(
        agent,
        SimpleNamespace(
            token_usage_last={
                "inputTokens": 100,
                "cachedInputTokens": 900,
                "outputTokens": 700,
                "totalTokens": 1_700,
            }
        ),
    )

    assert usage["cache_telemetry"] == "reported"
    assert agent._first_turn_usage["cache_read_tokens"] == 900
    assert agent._last_turn_usage["cache_read_tokens"] == 900
    assert events[0][0] == "hit"
    assert events[0][2] == 900
    assert events[0][4]["state"] == "hit"


def test_cache_info_from_usage_preserves_terminal_cold_write():
    cache_info_from_usage = getattr(server, "_cache_info_from_usage", None)
    assert callable(cache_info_from_usage)

    info = cache_info_from_usage(
        {
            "prompt_tokens": 100,
            "cache_read_tokens": 0,
            "cache_write_tokens": 25,
            "cache_telemetry": "reported",
        }
    )

    assert info["state"] == "cold_write"
    assert info["pct"] == 0


def test_cache_info_from_usage_marks_post_compression_boundary():
    info = server._cache_info_from_usage(
        {
            "prompt_tokens": 100,
            "cache_read_tokens": 10,
            "cache_write_tokens": 0,
            "cache_telemetry": "reported",
            "cache_attribution": "post_compression",
        }
    )

    assert info["compression_bound"] is True
