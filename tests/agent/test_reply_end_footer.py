from types import SimpleNamespace

from agent.turn_usage import _fold_reply_usage, _reply_cache_counts, mark_unaccounted_response, record_response_usage
from agent.usage_pricing import normalize_usage


def test_cache_read_zero_is_distinct_from_missing_and_invalid():
    measured_zero = normalize_usage(
        {"prompt_tokens": 100, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}
    )
    missing = normalize_usage({"prompt_tokens": 100, "completion_tokens": 1})
    fractional = normalize_usage(
        {"prompt_tokens": 100, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 1.5}}
    )

    assert measured_zero.cache_read_tokens == 0
    assert measured_zero.cache_read_known is True
    assert missing.cache_read_known is False
    assert fractional.cache_read_known is False


def test_usage_exactness_follows_the_selected_alias():
    invalid_selected = normalize_usage({
        "prompt_tokens": 20.5,
        "input_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 2.5},
        "cache_read_input_tokens": 2,
    })
    assert invalid_selected.prompt_tokens == 20
    assert invalid_selected.cache_read_tokens == 2
    assert invalid_selected.prompt_total_known is False
    assert invalid_selected.cache_read_known is False

    valid_selected = normalize_usage({
        "prompt_tokens": 20,
        "input_tokens": 21.5,
        "prompt_tokens_details": {"cached_tokens": 2},
        "cache_read_input_tokens": 2.5,
    })
    assert valid_selected.prompt_tokens == 20
    assert valid_selected.cache_read_tokens == 2
    assert valid_selected.prompt_total_known is True
    assert valid_selected.cache_read_known is True


def test_normal_text_response_attaches_footer_before_its_first_flush(monkeypatch):
    from agent.turn_final_response import finish_text_response

    monkeypatch.setattr("agent.agent_runtime_helpers.intent_ack_continuation_mode", lambda _agent: "off")
    monkeypatch.setattr("agent.agent_runtime_helpers.tool_results_this_turn", lambda _messages: 0)
    monkeypatch.setattr(
        "agent.turn_stop_gates.apply_stop_gates",
        lambda *_args, **_kwargs: SimpleNamespace(
            continue_turn=False,
            pending_verification_response=None,
            pending_verification_response_previewed=False,
        ),
    )
    monkeypatch.setattr(
        "agent.turn_finalizer.apply_llm_output_transform",
        lambda _agent, text, **_kwargs: (text, False, None),
    )
    monkeypatch.setattr("agent.turn_finalizer.time.monotonic", lambda: 12.0)

    flushed = []
    agent = SimpleNamespace(
        _reply_started_monotonic=10.0,
        _reply_usage={"cache_read_tokens": 4, "prompt_tokens": 10},
        _reply_usage_complete=True,
        _reply_elapsed_s=None,
        _strip_think_blocks=lambda text: text,
        _has_content_after_think_block=lambda text: bool(text),
        _emit_pending_fallback_notice=lambda: None,
        _clear_status_buffer=lambda: None,
        _build_assistant_message=lambda message, reason: {
            "role": "assistant", "content": message.content, "finish_reason": reason,
        },
        _interrupt_requested=False,
        _dropped_toolcall_retries=0,
        _flush_messages_to_session_db=lambda messages, _history: flushed.append(dict(messages[-1])),
        valid_tool_names=set(),
        quiet_mode=True,
    )
    messages = [{"role": "user", "content": "question"}]
    verdict = finish_text_response(
        agent,
        assistant_message=SimpleNamespace(content="Done.", tool_calls=[], reasoning_details=None),
        response=SimpleNamespace(),
        finish_reason="stop",
        messages=messages,
        api_messages=[],
        conversation_history=[],
        api_call_count=1,
        user_message="question",
        active_system_prompt="system",
        final_response=None,
        _turn_exit_reason="unknown",
        _preflight_compression_blocked=False,
        codex_ack_continuations=0,
        truncated_response_parts=[],
        length_continue_retries=0,
        _pending_verification_response=None,
        _pending_verification_response_previewed=False,
        effective_task_id="task",
    )

    assert verdict.action == "break"
    assert flushed[0]["display_metadata"]["reply_end"] == {
        "elapsed_s": 2.0, "cache_read": 4, "prompt_total": 10,
    }


def test_prompt_total_unknown_invalid_or_inconsistent_suppresses_exact_footer():
    samples = (
        {"prompt_tokens_details": {"cached_tokens": 10}},
        {"prompt_tokens": 20.5, "prompt_tokens_details": {"cached_tokens": 10}},
        {"prompt_tokens": 20, "prompt_tokens_details": {"cached_tokens": 25}},
    )
    for raw_usage in samples:
        normalized = normalize_usage(raw_usage)
        agent = SimpleNamespace(_reply_usage=None, _reply_usage_complete=True)
        _fold_reply_usage(
            agent,
            {"prompt_tokens": normalized.prompt_tokens, "cache_read_tokens": normalized.cache_read_tokens},
            normalized.cache_read_known,
            normalized.prompt_total_known,
        )
        assert _reply_cache_counts(agent) == (None, None)

    corrupt = SimpleNamespace(
        _reply_usage={"cache_read_tokens": 11, "prompt_tokens": 10},
        _reply_usage_complete=True,
    )
    assert _reply_cache_counts(corrupt) == (None, None)


def test_anthropic_prompt_total_requires_known_input_and_cache_components():
    complete = normalize_usage(
        {"input_tokens": 10, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 3},
        provider="anthropic",
        api_mode="anthropic_messages",
    )
    assert complete.prompt_total_known is True
    assert complete.prompt_tokens == 15

    for raw_usage in (
        {"input_tokens": 10, "cache_read_input_tokens": 2},
        {"input_tokens": 10, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1.5},
    ):
        usage = normalize_usage(raw_usage, provider="anthropic", api_mode="anthropic_messages")
        assert usage.prompt_total_known is False
        agent = SimpleNamespace(_reply_usage=None, _reply_usage_complete=True)
        _fold_reply_usage(
            agent,
            {"prompt_tokens": usage.prompt_tokens, "cache_read_tokens": usage.cache_read_tokens},
            usage.cache_read_known,
            usage.prompt_total_known,
        )
        assert _reply_cache_counts(agent) == (None, None)


def test_turn_cache_counts_accumulate_only_while_every_response_is_known():
    agent = SimpleNamespace(_reply_usage=None, _reply_usage_complete=True)
    _fold_reply_usage(agent, {"prompt_tokens": 100, "cache_read_tokens": 0}, True)
    _fold_reply_usage(agent, {"prompt_tokens": 50, "cache_read_tokens": 10}, True)
    assert _reply_cache_counts(agent) == (10, 150)

    _fold_reply_usage(agent, {"prompt_tokens": 20, "cache_read_tokens": 0}, False)
    assert _reply_cache_counts(agent) == (None, None)


def test_usage_less_or_unaccounted_response_never_reuses_previous_turn_totals():
    compressor = SimpleNamespace(awaiting_real_usage_after_compression=False)
    agent = SimpleNamespace(
        _reply_usage={"cache_read_tokens": 10, "prompt_tokens": 50},
        _reply_usage_complete=True,
        session_api_calls=0,
        model="test",
        provider="test",
        context_compressor=compressor,
    )

    record_response_usage(
        agent,
        SimpleNamespace(),
        messages=[],
        api_call_count=1,
        api_duration=0.1,
        compression_attempts=0,
        max_compression_attempts=3,
    )
    assert _reply_cache_counts(agent) == (None, None)

    agent._reply_usage_complete = True
    mark_unaccounted_response(agent)
    assert _reply_cache_counts(agent) == (None, None)
