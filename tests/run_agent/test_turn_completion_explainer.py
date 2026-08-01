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
def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
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


def test_heartbeat_silent_noop_leaves_no_durable_or_live_history():
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop")
    ]
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]

    with (
        patch.object(agent, "_persist_session") as persist,
        patch.object(agent, "_save_trajectory") as trajectory,
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_sync_external_memory_for_turn") as external_memory,
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] target remains ALIVE",
            conversation_history=history,
            turn_origin="heartbeat_warm",
            allow_silent_noop=True,
        )

    assert result["silent_noop"] is True
    assert result["final_response"] == ""
    assert result["messages"] == history
    assert agent._session_messages == history
    persist.assert_not_called()
    trajectory.assert_not_called()
    external_memory.assert_not_called()


def test_heartbeat_meaningful_tool_result_is_returned_and_persisted():
    agent = _make_agent(max_iterations=10)
    tool_call = SimpleNamespace(
        id="heartbeat-tool",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response(content="Target is still working.", finish_reason="stop"),
    ]
    persisted = []

    def _execute(_assistant, messages, _task_id, api_call_count=0):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "heartbeat-tool",
                "content": "process output grew",
            }
        )

    with (
        patch.object(agent, "_execute_tool_calls", side_effect=_execute),
        patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=lambda messages, _history=None: persisted.append(list(messages)),
        ),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            allow_silent_noop=True,
        )

    assert result["silent_noop"] is False
    assert result["final_response"] == "Target is still working."
    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert persisted[-1] == result["messages"]


def test_heartbeat_early_error_leaves_no_unmatched_synthetic_user_row():
    agent = _make_agent(max_iterations=10)
    agent.provider = "nous"
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]

    with (
        patch("agent.nous_rate_guard.nous_rate_limit_remaining", return_value=60),
        patch("agent.nous_rate_guard.format_remaining", return_value="1m"),
        patch.object(agent, "_try_activate_fallback", return_value=False),
        patch.object(agent, "_save_session_log") as save_log,
        patch.object(agent, "_flush_messages_to_session_db") as flush,
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            conversation_history=history,
            turn_origin="heartbeat_warm",
            allow_silent_noop=True,
        )

    assert result["messages"] == history
    assert agent._session_messages == history
    save_log.assert_not_called()
    flush.assert_not_called()


def test_heartbeat_does_not_consume_user_maintenance_triggers():
    agent = _make_agent(max_iterations=10)
    agent._user_turn_count = 9
    agent._turns_since_memory = 4
    agent._memory_nudge_interval = 5
    agent._memory_store = MagicMock()
    agent._iters_since_skill = 5
    agent._skill_nudge_interval = 5
    agent.valid_tool_names = {"memory", "skill_manage"}
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop"),
        _mock_response(content="real reply", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_spawn_background_review") as review,
    ):
        agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            allow_silent_noop=True,
        )
        assert agent._user_turn_count == 9
        assert agent._turns_since_memory == 4
        assert agent._iters_since_skill == 5
        review.assert_not_called()
        cached_system_prompt = agent._cached_system_prompt

        result = agent.run_conversation("real user turn")

    assert result["final_response"] == "real reply"
    assert agent._user_turn_count == 10
    assert agent._turns_since_memory == 0
    assert agent._iters_since_skill == 0
    assert agent._cached_system_prompt == cached_system_prompt
    review.assert_called_once()


def test_first_api_call_reports_cache_hit_to_tui_callback():
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
        result = agent.run_conversation("do something")

    assert result["turn_exit_reason"] == "partial_stream_recovery"
    assert result["final_response"].startswith(recovered)
    assert "No reply:" in result["final_response"]
    assert result["response_previewed"] is False
