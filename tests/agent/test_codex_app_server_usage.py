"""Regression tests for Codex app-server cache telemetry provenance."""

from types import SimpleNamespace

from agent.codex_runtime import _record_codex_app_server_usage
from tui_gateway.server import _cache_info_from_usage


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


def _record(cached_input_tokens_marker):
    usage = {
        "inputTokens": 2_000,
        "outputTokens": 700,
        "totalTokens": 2_700,
    }
    if cached_input_tokens_marker is not None:
        usage["cachedInputTokens"] = cached_input_tokens_marker
    return _record_codex_app_server_usage(
        _make_agent(),
        SimpleNamespace(token_usage_last=usage),
    )


def test_app_server_absent_cache_telemetry_is_unavailable_in_tui():
    usage = _record(None)

    assert usage["cache_telemetry"] == "unavailable"
    cache_info = _cache_info_from_usage(usage)
    assert cache_info["state"] == "unavailable"


def test_app_server_explicit_zero_cache_telemetry_is_reported_miss():
    usage = _record(0)

    assert usage["cache_telemetry"] == "reported"
    cache_info = _cache_info_from_usage(usage)
    assert cache_info["state"] == "miss"
