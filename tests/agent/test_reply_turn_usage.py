"""Reply footer accounting is the whole user turn, not the last request.

``_last_turn_usage`` stays the latest usage-bearing response for its other
consumers. A missing cache field is not a measured zero, and a usage-less
request makes the turn aggregate incomplete.
"""

from types import SimpleNamespace

from agent.turn_usage import _reply_cache_counts, record_response_usage
from agent.usage_pricing import normalize_usage


def _agent():
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
        session_cost_status=None,
        session_cost_source=None,
        _last_turn_usage=None,
        _last_prompt_size_tokens=0,
        _reply_usage=None,
        _reply_usage_complete=True,
        context_compressor=SimpleNamespace(
            awaiting_real_usage_after_compression=False,
            update_from_response=lambda _usage: None,
            threshold_tokens=0,
        ),
        provider="openai",
        api_mode="chat_completions",
        model="m",
        base_url="",
        api_key="",
        session_id=None,
        _session_db=None,
        verbose_logging=False,
        quiet_mode=True,
        client=None,
        platform="cli",
    )


def _resp(usage):
    return SimpleNamespace(usage=usage, id=None)


def _record(agent, usage):
    record_response_usage(
        agent, _resp(usage), messages=[{"role": "user", "content": "hi"}],
        api_call_count=1, api_duration=0.1, compression_attempts=0, max_compression_attempts=3,
    )


def test_missing_cache_field_is_unknown_and_explicit_zero_is_known():
    missing = normalize_usage(
        SimpleNamespace(prompt_tokens=500, completion_tokens=50), provider="openai", api_mode="chat_completions",
    )
    explicit = normalize_usage(
        {"prompt_tokens": 500, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 0}},
        provider="openai", api_mode="chat_completions",
    )
    invalid = normalize_usage(
        {"prompt_tokens": 500, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": "nope"}},
        provider="openai", api_mode="chat_completions",
    )
    assert missing.cache_read_tokens == explicit.cache_read_tokens == 0
    assert missing.cache_read_known is False
    assert explicit.cache_read_known is True
    assert invalid.cache_read_known is False


def test_multi_request_reply_sums_hits_and_resets_for_the_next_turn():
    agent = _agent()
    _record(agent, {"prompt_tokens": 1000, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 800}})
    _record(agent, {"prompt_tokens": 400, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 100}})
    assert _reply_cache_counts(agent) == (900, 1400)
    # Other consumers still see only the latest response.
    assert agent._last_turn_usage["cache_read_tokens"] == 100
    assert agent._last_turn_usage["prompt_tokens"] == 400
    agent._reply_usage = None
    agent._reply_usage_complete = True
    agent._last_turn_usage = None
    _record(agent, {"prompt_tokens": 50, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 5}})
    assert _reply_cache_counts(agent) == (5, 50)


def test_usage_less_final_request_does_not_reuse_an_earlier_total():
    agent = _agent()
    _record(agent, {"prompt_tokens": 1000, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 800}})
    _record(agent, None)
    assert _reply_cache_counts(agent) == (None, None)
    # The last-request seam is unchanged: a usage-less response does not clear it.
    assert agent._last_turn_usage["cache_read_tokens"] == 800
