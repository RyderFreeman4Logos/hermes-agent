"""Regression tests for API-only loop timing context."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import agent.conversation_loop as conversation_loop
from run_agent import AIAgent


UTC_MINUS_7 = timezone(timedelta(hours=-7))


def _timing_helper():
    helper = getattr(conversation_loop, "_loop_timing_context", None)
    assert helper is not None
    return helper


def test_loop_timing_context_records_boundaries_and_honors_config():
    helper = _timing_helper()
    agent = SimpleNamespace()
    first_start = datetime(2026, 8, 22, 11, 28, 0, tzinfo=UTC_MINUS_7)
    first_stop = datetime(2026, 8, 22, 11, 28, 3, tzinfo=UTC_MINUS_7)
    second_start = datetime(2026, 8, 22, 11, 28, 5, tzinfo=UTC_MINUS_7)

    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert "Current loop start: 2026-08-22T11:28:00-07:00" in helper(
            agent, now=first_start
        )
        helper(agent, now=first_stop, stop=True)
        context = helper(agent, now=second_start)

    assert "Previous loop start: 2026-08-22T11:28:00-07:00" in context
    assert "Previous loop stop: 2026-08-22T11:28:03-07:00" in context
    assert "Current loop start: 2026-08-22T11:28:05-07:00" in context

    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"agent": {"loop_timing_context": False}},
    ):
        assert helper(agent, now=second_start) == ""


def test_agent_forwarder_exposes_timing_context_to_loop_and_clears_it():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    seen_context = []

    def fake_loop(*args, **kwargs):
        seen_context.append(getattr(agent, "_loop_timing_context_text", ""))
        return {"final_response": "ok", "messages": [], "api_calls": 1}

    with (
        patch(
            "agent.conversation_loop._loop_timing_context",
            side_effect=["timing context", None],
            create=True,
        ) as timing,
        patch("agent.conversation_loop.run_conversation", side_effect=fake_loop),
    ):
        agent.run_conversation("hello")

    assert seen_context == ["timing context"]
    assert timing.call_count == 2
    assert timing.call_args_list[1].kwargs["stop"] is True
    assert agent._loop_timing_context_text == ""
