"""Tests for the end-of-turn completion explainer (#34452).

When a turn ends abnormally after tools (empty content after retries, a
partial/truncated stream, exhausted retries, or an iteration/budget limit)
the user should get a single user-visible explanation of why the reply
stopped instead of a blank or fragmentary response box.  Normal short
replies (e.g. ``Done.``) must stay quiet.

These tests exercise:
  1. ``_format_turn_completion_explanation`` — the pure reason→message map.
  2. ``_turn_completion_explainer_enabled`` — the env/config seam.
  3. An end-to-end ``run_conversation`` turn that exhausts empty-response
     retries and verifies the explanation reaches ``final_response``.

All assertions work under the mocked OpenAI SDK used elsewhere in this
suite (we patch ``run_agent.OpenAI`` and drive ``agent.client``), so they
pass identically in CI and locally.
"""

import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


# --------------------------------------------------------------------------
# Fixtures (mirrors tests/run_agent/test_tool_call_guardrail_runtime.py)
# --------------------------------------------------------------------------
def _mock_response(
    content="Hello", finish_reason="stop", tool_calls=None, reasoning_content=None
):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    # No fallback chain so empty responses exhaust deterministically.
    agent._fallback_chain = []
    return agent


# --------------------------------------------------------------------------
# 1. Pure formatter
# --------------------------------------------------------------------------
def test_explanation_quiet_for_normal_text_response():
    """A healthy text_response exit must NOT produce any explanation."""
    out = AIAgent._format_turn_completion_explanation(
        "text_response(finish_reason=stop)"
    )
    assert out == ""


def test_explanation_quiet_for_empty_reason():
    assert AIAgent._format_turn_completion_explanation("") == ""
    assert AIAgent._format_turn_completion_explanation("unknown") == ""
    # guardrail_halt surfaces its own message; explainer stays out of the way.
    assert AIAgent._format_turn_completion_explanation("guardrail_halt") == ""






def test_explanation_for_max_iterations_reached_prefix_match():
    """``max_iterations_reached(...)`` carries a parenthetical suffix."""
    out = AIAgent._format_turn_completion_explanation(
        "max_iterations_reached(10/10)"
    )
    assert "iteration" in out.lower()






# --------------------------------------------------------------------------
# 2. Enable/disable seam
# --------------------------------------------------------------------------
def test_explainer_enabled_by_default():
    agent = _make_agent()
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HERMES_TURN_COMPLETION_EXPLAINER", None)
        with patch("hermes_cli.config.load_config", return_value={}):
            assert agent._turn_completion_explainer_enabled() is True


def test_explainer_disabled_via_env():
    agent = _make_agent()
    with patch.dict(
        os.environ, {"HERMES_TURN_COMPLETION_EXPLAINER": "0"}, clear=False
    ):
        assert agent._turn_completion_explainer_enabled() is False




# --------------------------------------------------------------------------
# 3. End-to-end: empty-response exhaustion surfaces the explanation
# --------------------------------------------------------------------------
def test_run_conversation_empty_exhausted_surfaces_explanation():
    """Four empty responses in a row should exhaust retries and the final
    response should be the actionable explanation, not a bare '(empty)'."""
    agent = _make_agent(max_iterations=10)
    # 4 empty responses: retries 1..3 then the terminal on the 4th.
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop") for _ in range(8)
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do something")

    assert result["turn_exit_reason"] == "empty_response_exhausted"
    # The user must NOT be left with a bare sentinel; the explanation wins.
    assert result["final_response"] != "(empty)"
    assert result["final_response"].strip() != ""
    assert "No reply:" in result["final_response"]
    assert result.get("silent_noop") is not True
    assert agent.client.chat.completions.create.call_count == 4


def test_idle_completion_empty_stop_is_silent_and_keeps_notification():
    """A clean empty completion notification is durable but costs no retry turn."""
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[IMPORTANT: Background process proc_idle completed normally]",
            turn_origin="idle_completion",
            allow_silent_noop=True,
        )

    assert agent.client.chat.completions.create.call_count == 1
    assert result["final_response"] is None
    assert result["completed"] is True
    assert result["silent_noop"] is True
    assert result["turn_exit_reason"] == "idle_notification_noop"
    assert all(message.get("role") != "assistant" for message in result["messages"])
    assert any(
        message.get("role") == "user"
        and "proc_idle completed normally" in str(message.get("content"))
        for message in result["messages"]
    )


def test_idle_completion_reasoning_only_stop_or_success_is_silent_without_retries():
    """Reasoning without visible text is a no-op for an idle completion."""
    for finish_reason in ("stop", "success"):
        agent = _make_agent(max_iterations=10)
        agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content="",
                finish_reason=finish_reason,
                reasoning_content="The notification does not need a reply.",
            )
            for _ in range(8)
        ]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "[IMPORTANT: Background process proc_idle completed normally]",
                turn_origin="idle_completion",
                allow_silent_noop=True,
            )

        assert result["final_response"] is None
        assert result["silent_noop"] is True
        assert result["turn_exit_reason"] == "idle_notification_noop"
        assert result["api_calls"] == 1
        assert agent.client.chat.completions.create.call_count == 1


def test_idle_completion_streamed_reasoning_without_text_is_silent_without_retries():
    """Streamed reasoning alone must not enter the empty-response retry path."""
    agent = _make_agent(max_iterations=10)
    response = _mock_response(content="", finish_reason="stop")

    def _response_with_streamed_reasoning(**_kwargs):
        agent._current_streamed_reasoning_text = "No visible response is needed."
        return response

    agent.client.chat.completions.create.side_effect = _response_with_streamed_reasoning
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[IMPORTANT: Background process proc_idle completed normally]",
            turn_origin="idle_completion",
            allow_silent_noop=True,
        )

    assert result["final_response"] is None
    assert result["silent_noop"] is True
    assert result["turn_exit_reason"] == "idle_notification_noop"
    assert result["api_calls"] == 1
    assert agent.client.chat.completions.create.call_count == 1


def test_idle_completion_noop_instruction_is_api_only_and_persists_clean_notification():
    """The model sees the silent-noop hint, but the durable row never does."""
    notification = "[IMPORTANT: Background process proc_idle completed normally]"
    suffix = (
        "\n\n[Note: If this background completion requires no action, return an "
        "empty response with no text and no tool calls.]"
    )
    agent = _make_agent(max_iterations=10)
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            notification,
            turn_origin="idle_completion",
            allow_silent_noop=True,
        )

    api_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert api_messages[-1]["content"] == notification + suffix
    assert result["messages"][-1]["content"] == notification

    durable_user_rows = [
        call.kwargs
        for call in agent._session_db.append_message.call_args_list
        if call.kwargs.get("role") == "user"
    ]
    assert durable_user_rows[-1]["content"] == notification
    assert suffix not in str(durable_user_rows[-1].get("api_content") or "")


def test_first_api_call_reports_cache_hit_to_tui_callback():
    """Only the first provider call reports the cache state that diagnoses a cold prefix."""
    agent = _make_agent(max_iterations=10)
    response = _mock_response(content="Done.", finish_reason="stop")
    response.usage = SimpleNamespace(
        prompt_tokens=2_000,
        completion_tokens=10,
        total_tokens=2_010,
        prompt_tokens_details=SimpleNamespace(cached_tokens=1_740),
    )
    agent.client.chat.completions.create.side_effect = [response]
    cache_events = []
    agent._tui_cache_callback = lambda state, pct, read, prompt: cache_events.append(
        (state, pct, read, prompt)
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do something")

    assert result["final_response"] == "Done."
    assert cache_events == [("hit", 87, 1_740, 2_000)]


def test_idle_completion_timeout_or_error_is_not_silent():
    """Transport failures keep the normal retry/failure path, never no-op."""
    for failure in (TimeoutError("provider timed out"), RuntimeError("provider failed")):
        agent = _make_agent(max_iterations=10)
        with (
            patch.object(agent, "_interruptible_api_call", side_effect=failure),
            patch("agent.conversation_loop.time.sleep"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "[IMPORTANT: Background process proc_failed completed]",
                turn_origin="idle_completion",
                allow_silent_noop=True,
            )

        assert result.get("silent_noop") is not True
        assert result.get("error")


def test_run_conversation_partial_stream_recovery_surfaces_explanation():
    """A long recovered partial stream still needs the visible footer.

    Without this, the gateway marks the turn as previewed and suppresses
    the final send, leaving messaging users with a fragment and no reason.
    """
    agent = _make_agent(max_iterations=10)
    empty_stub = _mock_response(content=None, finish_reason="stop")
    recovered = (
        "I inspected the running gateway and found that the current turn "
        "stopped after the provider stream timed out."
    )

    def _fake_api_call(_api_kwargs):
        agent._current_streamed_assistant_text = recovered
        return empty_stub

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "do something",
            turn_origin="idle_completion",
            allow_silent_noop=True,
        )

    assert result["turn_exit_reason"] == "partial_stream_recovery"
    assert result["final_response"].startswith(recovered)
    assert "No reply:" in result["final_response"]
    assert result["response_previewed"] is False
    assert result.get("silent_noop") is not True


