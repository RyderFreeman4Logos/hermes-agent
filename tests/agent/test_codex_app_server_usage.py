"""Codex app-server usage accounting must not setattr-loop cache telemetry."""

from types import SimpleNamespace

from agent.codex_runtime import _record_codex_app_server_usage


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
