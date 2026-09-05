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


def test_codex_empty_compact_usage_does_not_clear_bound_latch():
    from agent.codex_runtime import (
        _record_codex_app_server_compaction,
        _record_codex_app_server_usage,
    )

    agent = _make_agent()
    agent.context_compressor = SimpleNamespace(
        compression_count=0,
        last_compression_rough_tokens=0,
        last_prompt_tokens=0,
        last_completion_tokens=0,
        awaiting_real_usage_after_compression=False,
        _verify_compaction_cleared_threshold=False,
    )
    compact = SimpleNamespace(
        compacted=True,
        thread_id="t1",
        turn_id="c1",
        token_usage_last=None,
    )

    _record_codex_app_server_compaction(agent, compact, approx_tokens=4_000, force=True)
    assert agent._awaiting_cache_usage_after_compression is True

    empty = _record_codex_app_server_usage(
        agent, compact, consume_compression_bound=False
    )
    assert empty["cache_telemetry"] == "unavailable"
    assert agent._awaiting_cache_usage_after_compression is True

    next_usage = _record_codex_app_server_usage(
        agent,
        SimpleNamespace(
            token_usage_last={
                "inputTokens": 800,
                "cachedInputTokens": 0,
                "outputTokens": 40,
                "totalTokens": 840,
            }
        ),
    )
    assert next_usage["cache_telemetry"] == "reported"
    assert agent._first_turn_usage["cache_attribution"] == "post_compression"
    assert agent._awaiting_cache_usage_after_compression is False


_CODEX_USAGE = {
    "empty": None,
    "unavailable": {"inputTokens": 800},
    "null": {"inputTokens": 800, "cachedInputTokens": None},
    "zero": {"inputTokens": 800, "cachedInputTokens": 0},
    "hit": {"inputTokens": 40, "cachedInputTokens": 760},
}


@pytest.mark.parametrize("compact_usage", _CODEX_USAGE)
@pytest.mark.parametrize("next_usage", _CODEX_USAGE)
def test_public_codex_compact_bookkeeping_matrix(compact_usage, next_usage):
    """Compact RPC accounting is not the next real-response cache observation."""
    from pathlib import Path
    from unittest.mock import MagicMock

    from agent.conversation_compression import compress_context
    from agent.transports.codex_app_server_session import TurnResult

    assert Path(compress_context.__code__.co_filename).resolve() == Path(__file__).resolve().parents[2] / "agent/conversation_compression.py"
    agent = _make_agent()
    agent.api_mode = "codex_app_server"
    agent._cached_system_prompt = "synthetic"
    agent._emit_status = lambda *a, **k: None
    agent._emit_warning = lambda *a, **k: None
    agent._awaiting_cache_usage_after_compression = False
    agent.context_compressor = SimpleNamespace(
        compression_count=0, update_from_response=lambda usage: None,
    )
    compact = TurnResult(token_usage_last=_CODEX_USAGE[compact_usage])
    agent._codex_session = MagicMock()
    agent._codex_session.compact_thread.return_value = compact
    events = []
    agent._tui_cache_callback = lambda *args: events.append(args)
    messages = [{"role": "user", "content": "synthetic"}]
    returned, _ = compress_context(agent, messages, "sys", force=True)
    assert returned is messages  # Codex, not Hermes, owns the rewritten thread.
    assert agent._codex_session.compact_thread.call_count == 1
    assert agent.context_compressor.compression_count == 1
    assert agent.session_api_calls == 1  # Proves accounting was not swallowed.
    assert not events
    assert agent._awaiting_cache_usage_after_compression is True
    usage = _record_codex_app_server_usage(agent, TurnResult(token_usage_last=_CODEX_USAGE[next_usage]))
    assert agent._awaiting_cache_usage_after_compression is False
    if next_usage == "empty":
        assert not events
        assert usage == {"cache_telemetry": "unavailable"}
    else:
        assert events[-1][4]["attribution"] == "post_compression"
        info = server._cache_info_from_usage(agent._first_turn_usage)
        assert info.get("compression_bound") is True
        assert info["state"] == {"unavailable": "unavailable", "null": "unavailable", "zero": "miss", "hit": "hit"}[next_usage]
        assert ("read_tokens" in info) == (next_usage in {"zero", "hit"})
    _record_codex_app_server_usage(agent, TurnResult(token_usage_last=_CODEX_USAGE["zero"]))
    assert "cache_attribution" not in agent._first_turn_usage
    assert "attribution" not in events[-1][4]


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("outcome", ["success", "error", "interrupted", "exception", "native", "off", "no-thread", "fence-cancel"])
def test_public_codex_boundary_outcome_matrix(pending, outcome):
    from unittest.mock import MagicMock

    from agent.conversation_compression import CompressionCommitFence, compress_context
    from agent.transports.codex_app_server_session import TurnResult

    agent = _make_agent()
    agent.api_mode = "codex_app_server"
    agent.codex_app_server_auto_compaction = outcome if outcome in {"native", "off"} else "hermes"
    agent._cached_system_prompt = "synthetic"
    agent._emit_status = lambda *a, **k: None
    agent._emit_warning = lambda *a, **k: None
    agent._awaiting_cache_usage_after_compression = pending
    agent.context_compressor = SimpleNamespace(compression_count=0, update_from_response=lambda usage: None)
    rpc = MagicMock()
    rpc.compact_thread.return_value = TurnResult(
        error="synthetic" if outcome == "error" else None,
        interrupted=outcome == "interrupted",
    )
    if outcome == "exception":
        rpc.compact_thread.side_effect = RuntimeError("synthetic compact failure")
    agent._codex_session = None if outcome == "no-thread" else rpc
    fence = CompressionCommitFence()
    if outcome == "fence-cancel":
        assert fence.cancel_before_commit()
    messages = [{"role": "user", "content": "synthetic"}]
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="synthetic compact failure"):
            compress_context(agent, messages, "sys", commit_fence=fence)
    else:
        returned, _ = compress_context(agent, messages, "sys", commit_fence=fence)
        assert returned is messages
    assert agent._awaiting_cache_usage_after_compression is (pending or outcome == "success")
    assert rpc.compact_thread.call_count == int(outcome in {"success", "error", "interrupted", "exception"})
    assert agent.context_compressor.compression_count == int(outcome == "success")
