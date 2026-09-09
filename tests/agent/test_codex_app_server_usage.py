"""Codex/loop cache telemetry: numeric accounting plus miss vs no_field."""

from types import SimpleNamespace

from agent.codex_runtime import _record_codex_app_server_usage
from agent.turn_usage import record_response_usage


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


def test_codex_app_server_usage_notifies_once_and_accounts_numeric_totals():
    events = []
    agent = _make_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    usage = _record_codex_app_server_usage(
        agent,
        SimpleNamespace(
            token_usage_last={
                "totalTokens": 130,
                "inputTokens": 80,
                "cachedInputTokens": 20,
                "outputTokens": 25,
                "reasoningOutputTokens": 5,
            }
        ),
    )

    assert len(events) == 1
    assert events[0][0] == "hit"
    assert events[0][2] == 20
    assert events[0][3] == 100
    assert events[0][4]["state"] == "hit"
    assert usage["cache_telemetry"] == "reported"
    assert agent._first_turn_usage["cache_telemetry"] == "reported"
    assert agent._last_turn_usage["cache_telemetry"] == "reported"
    assert agent._first_turn_usage["cache_read_tokens"] == 20
    assert agent.session_api_calls == 1
    assert agent.session_prompt_tokens == 100
    assert agent.session_completion_tokens == 25
    assert agent.session_total_tokens == 130
    assert agent.session_input_tokens == 80
    assert agent.session_output_tokens == 25
    assert agent.session_cache_read_tokens == 20
    assert agent.session_cache_write_tokens == 0
    assert agent.session_reasoning_tokens == 5
    assert not hasattr(agent, "session_cache_telemetry")


def test_codex_app_server_no_usage_still_notifies_once():
    events = []
    agent = _make_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    usage = _record_codex_app_server_usage(agent, SimpleNamespace(token_usage_last=None))

    assert usage == {}
    assert len(events) == 1
    assert events[0][0] == "no_field"
    assert events[0][4]["state"] == "no_field"
    assert agent.session_api_calls == 1
    assert not hasattr(agent, "session_cache_telemetry")
    assert not hasattr(agent, "_first_turn_usage")


def test_codex_app_server_present_zero_cache_is_miss_not_no_field():
    events = []
    agent = _make_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    usage = _record_codex_app_server_usage(
        agent,
        SimpleNamespace(
            token_usage_last={
                "totalTokens": 110,
                "inputTokens": 100,
                "cachedInputTokens": 0,
                "outputTokens": 10,
                "reasoningOutputTokens": 0,
            }
        ),
    )

    assert len(events) == 1
    assert events[0][0] == "miss"
    assert events[0][2] == 0
    assert events[0][3] == 100
    assert events[0][4]["state"] == "miss"
    assert usage["cache_telemetry"] == "reported"
    assert agent._first_turn_usage["cache_telemetry"] == "reported"
    assert agent._last_turn_usage["cache_telemetry"] == "reported"
    assert agent.session_prompt_tokens == 100
    assert agent.session_cache_read_tokens == 0
    assert not hasattr(agent, "session_cache_telemetry")


def test_codex_app_server_omitted_cache_field_is_no_field():
    events = []
    agent = _make_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    usage = _record_codex_app_server_usage(
        agent,
        SimpleNamespace(
            token_usage_last={
                "totalTokens": 110,
                "inputTokens": 100,
                "outputTokens": 10,
            }
        ),
    )

    assert len(events) == 1
    assert events[0][0] == "no_field"
    assert events[0][4]["state"] == "no_field"
    assert usage["cache_telemetry"] == "unavailable"
    assert agent._first_turn_usage["cache_telemetry"] == "unavailable"
    assert not hasattr(agent, "session_cache_telemetry")


def _make_loop_agent() -> SimpleNamespace:
    agent = _make_agent()
    agent.api_mode = "chat_completions"
    agent.provider = "openai"
    agent.model = "gpt-4o"
    agent.verbose_logging = False
    agent.quiet_mode = True
    agent.context_compressor = SimpleNamespace(
        awaiting_real_usage_after_compression=False,
        _verify_compaction_cleared_threshold=False,
        threshold_tokens=0,
        _context_probed=False,
        update_from_response=lambda _usage: None,
    )
    return agent


def test_loop_usage_present_zero_cache_notifies_miss():
    events = []
    agent = _make_loop_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    record_response_usage(
        agent,
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=4000,
                completion_tokens=10,
                total_tokens=4010,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            )
        ),
        messages=[{"role": "user", "content": "hi"}],
        api_call_count=1,
        api_duration=0.1,
        compression_attempts=0,
        max_compression_attempts=3,
    )

    assert len(events) == 1
    assert events[0][0] == "miss"
    assert events[0][4]["state"] == "miss"
    assert agent._first_turn_usage["cache_telemetry"] == "reported"
    assert agent._first_turn_usage["prompt_tokens"] == 4000
    assert agent._first_turn_usage["cache_read_tokens"] == 0
    assert not hasattr(agent, "session_cache_telemetry")


def test_loop_usage_absent_cache_field_notifies_no_field():
    events = []
    agent = _make_loop_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    record_response_usage(
        agent,
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=4000,
                completion_tokens=10,
                total_tokens=4010,
            )
        ),
        messages=[{"role": "user", "content": "hi"}],
        api_call_count=1,
        api_duration=0.1,
        compression_attempts=0,
        max_compression_attempts=3,
    )

    assert len(events) == 1
    assert events[0][0] == "no_field"
    assert events[0][4]["state"] == "no_field"
    assert agent._first_turn_usage["cache_telemetry"] == "unavailable"
    assert not hasattr(agent, "session_cache_telemetry")
