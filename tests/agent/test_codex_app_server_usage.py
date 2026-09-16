"""Codex/loop cache telemetry: numeric accounting plus miss vs no_field."""

from types import SimpleNamespace

from agent.codex_runtime import (
    _record_codex_app_server_usage,
    make_codex_app_server_event_bridge,
)
from agent.turn_usage import observe_response_usage, record_response_usage


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
    assert agent._first_turn_usage["cache_telemetry"] == "unavailable"


def test_codex_app_server_keeps_first_usage_note_not_last():
    events = []
    agent = _make_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    usage = _record_codex_app_server_usage(
        agent,
        SimpleNamespace(
            token_usage_first={
                "totalTokens": 110,
                "inputTokens": 100,
                "cachedInputTokens": 90,
                "outputTokens": 10,
            },
            token_usage_last={
                "totalTokens": 220,
                "inputTokens": 200,
                "cachedInputTokens": 0,
                "outputTokens": 20,
            },
        ),
    )

    assert events[0][0] == "hit"
    assert agent._first_turn_usage["cache_read_tokens"] == 90
    assert agent._last_turn_usage["cache_read_tokens"] == 0
    assert usage["cache_read_tokens"] == 0


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


def test_codex_app_server_invalid_cache_field_is_unavailable():
    events = []
    agent = _make_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    usage = _record_codex_app_server_usage(
        agent,
        SimpleNamespace(token_usage_last={
            "totalTokens": 110, "inputTokens": 100,
            "cachedInputTokens": True, "outputTokens": 10,
        }),
    )

    assert events[0][0] == "no_field"
    assert usage["cache_telemetry"] == "unavailable"
    assert usage["cache_read_tokens"] == 0


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


def test_loop_usage_less_first_response_stays_first_when_later_usage_arrives():
    events = []
    agent = _make_loop_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    record_response_usage(
        agent, SimpleNamespace(usage=None), messages=[], api_call_count=1,
        api_duration=0.1, compression_attempts=0, max_compression_attempts=3,
    )
    record_response_usage(
        agent,
        SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=10, total_tokens=110,
            prompt_tokens_details=SimpleNamespace(cached_tokens=90),
        )),
        messages=[], api_call_count=2, api_duration=0.1,
        compression_attempts=0, max_compression_attempts=3,
    )

    assert [event[0] for event in events] == ["no_field"]
    assert agent._first_turn_usage["cache_telemetry"] == "unavailable"


def test_loop_first_cache_observation_excludes_moa_advisor_usage():
    from agent.usage_pricing import CanonicalUsage

    events = []
    agent = _make_loop_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)
    agent.client = SimpleNamespace(
        consume_reference_usage=lambda: (
            CanonicalUsage(input_tokens=100, output_tokens=0), 0.0
        ),
        consume_and_save_trace=lambda *_args, **_kwargs: None,
    )

    record_response_usage(
        agent,
        SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=10, total_tokens=110,
            prompt_tokens_details=SimpleNamespace(cached_tokens=90),
        )),
        messages=[], api_call_count=1, api_duration=0.1,
        compression_attempts=0, max_compression_attempts=3,
    )

    assert events[0][0] == "hit"
    assert events[0][1] == 90
    assert agent._first_turn_usage["prompt_tokens"] == 100


def test_response_ingress_latches_length_miss_before_later_hit():
    events = []
    agent = _make_loop_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)

    observe_response_usage(
        agent,
        SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=2_000,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )),
    )
    observe_response_usage(
        agent,
        SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=2_000,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1_900),
        )),
    )

    assert [event[0] for event in events] == ["miss"]
    assert events[0][4]["request_index"] == 1
    assert agent._first_turn_usage["cache_read_tokens"] == 0
    assert agent._first_turn_usage["cache_telemetry"] == "reported"


def test_loop_accounting_uses_positive_fallback_after_nested_zero():
    agent = _make_loop_agent()

    record_response_usage(
        agent,
        SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=2_000,
            completion_tokens=100,
            total_tokens=2_100,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            prompt_cache_hit_tokens=1_900,
        )),
        messages=[{"role": "user", "content": "hi"}],
        api_call_count=1,
        api_duration=0.1,
        compression_attempts=0,
        max_compression_attempts=3,
    )

    assert agent.session_cache_read_tokens == 1_900
    assert agent.session_input_tokens == 100
    assert agent.session_prompt_tokens == 2_000


def test_app_server_usage_is_observed_at_notification_ingress():
    events = []
    agent = _make_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)
    bridge = make_codex_app_server_event_bridge(agent)

    bridge({
        "method": "thread/tokenUsage/updated",
        "params": {
            "tokenUsage": {
                "last": {
                    "inputTokens": 100,
                    "cachedInputTokens": 90,
                    "outputTokens": 10,
                    "totalTokens": 110,
                }
            }
        },
    })

    assert len(events) == 1
    assert events[0][0] == "hit"
    assert events[0][4]["request_index"] == 1
    assert agent._first_turn_usage["cache_read_tokens"] == 90


def test_partial_app_server_usage_does_not_latch_unavailable_before_real_usage():
    events = []
    agent = _make_agent()
    agent._tui_cache_callback = lambda *args: events.append(args)
    bridge = make_codex_app_server_event_bridge(agent)

    bridge({
        "method": "thread/tokenUsage/updated",
        "params": {
            "tokenUsage": {
                "total": {
                    "inputTokens": 100,
                    "cachedInputTokens": 0,
                    "outputTokens": 0,
                    "totalTokens": 100,
                },
                "modelContextWindow": 200_000,
            }
        },
    })

    assert events == []
    assert not hasattr(agent, "_first_turn_usage")

    bridge({
        "method": "thread/tokenUsage/updated",
        "params": {
            "tokenUsage": {
                "last": {
                    "inputTokens": 100,
                    "cachedInputTokens": 90,
                    "outputTokens": 10,
                    "totalTokens": 110,
                }
            }
        },
    })

    assert len(events) == 1
    assert events[0][0] == "hit"
    assert agent._first_turn_usage["cache_read_tokens"] == 90


def test_response_checker_observes_usage_before_length_recovery(monkeypatch):
    import agent.turn_recovery as recovery
    import agent.turn_response_check as checker

    observed = []
    response = SimpleNamespace(usage={"prompt_tokens": 100})
    agent = SimpleNamespace(
        quiet_mode=True,
        verbose_logging=False,
        provider="custom",
        api_mode="chat_completions",
        thinking_callback=None,
    )
    monkeypatch.setattr(recovery, "validate_response_shape", lambda *_args: (False, None))
    monkeypatch.setattr(checker, "_derive_finish_reason", lambda *_args: "length")
    monkeypatch.setattr(
        checker,
        "observe_response_usage",
        lambda seen_agent, seen_response: observed.append((seen_agent, seen_response)),
    )

    def recover(*_args, **_kwargs):
        assert observed == [(agent, response)]
        return SimpleNamespace(
            action="continue",
            result=None,
            messages=[],
            length_continue_retries=1,
            truncated_response_parts=[],
            truncated_tool_call_retries=0,
            retry_count=0,
            compression_attempts=0,
        )

    monkeypatch.setattr(checker, "recover_from_truncation", recover)
    verdict = checker.check_api_response(
        agent,
        response=response,
        _retry=SimpleNamespace(),
        thinking_spinner=None,
        messages=[],
        api_messages=[],
        api_kwargs={},
        active_system_prompt="",
        conversation_history=[],
        finish_reason=None,
        retry_count=0,
        max_retries=1,
        compression_attempts=0,
        max_compression_attempts=1,
        length_continue_retries=0,
        truncated_response_parts=[],
        truncated_tool_call_retries=0,
        current_turn_user_idx=0,
        api_call_count=1,
        api_request_id="r1",
        api_start_time=0,
        effective_task_id=None,
        turn_id=None,
        _preflight_compression_blocked=False,
        _last_preflight_pressure=None,
    )

    assert verdict.action == "continue"
