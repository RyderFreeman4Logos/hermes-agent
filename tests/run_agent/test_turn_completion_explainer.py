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

import pytest

from agent.context_engine import ContextEngine
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


@pytest.mark.parametrize(
    ("cached_tokens", "expected_pct", "should_be_red"),
    [(1_880, 94, True), (1_900, 95, False)],
)
def test_first_api_call_reports_cache_hit_to_tui_callback(
    cached_tokens, expected_pct, should_be_red
):
    agent = _make_agent(max_iterations=10)
    agent.quiet_mode = False
    response = _mock_response(content="Done.", finish_reason="stop")
    response.usage = SimpleNamespace(
        prompt_tokens=2_000,
        completion_tokens=10,
        total_tokens=2_010,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
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
        patch.object(agent, "_vprint") as vprint,
    ):
        result = agent.run_conversation("do something")

    assert result["final_response"] == "Done."
    assert cache_events == [("hit", expected_pct, cached_tokens, 2_000)]
    assert "cache_attribution" not in agent._first_turn_usage
    cache_lines = [
        call.args[0] for call in vprint.call_args_list if "💾 Cache:" in call.args[0]
    ]
    assert len(cache_lines) == 1
    red_pct = f"\033[31m{expected_pct}%\033[0m"
    assert (red_pct in cache_lines[0]) is should_be_red
    assert "post-compression cold prefix" not in cache_lines[0]


def test_post_compression_cache_attribution_survives_retry_then_clears():
    class _RateLimitError(Exception):
        status_code = 429

        def __str__(self):
            return "Error code: 429 - Rate limit exceeded."

    class _NoOpContextEngine(ContextEngine):
        @property
        def name(self):
            return "no-op"

        def update_from_response(self, usage):
            pass

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(
            self,
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ):
            return messages

    def response(cached_tokens: int | None, *, tool_id: str = "", content: str = ""):
        tool_calls = []
        if tool_id:
            tool_calls = [
                SimpleNamespace(
                    id=tool_id,
                    type="function",
                    function=SimpleNamespace(name="noop", arguments="{}"),
                )
            ]
        result = _mock_response(
            content=content,
            finish_reason="tool_calls" if tool_id else "stop",
            tool_calls=tool_calls,
        )
        if cached_tokens is not None:
            result.usage = SimpleNamespace(
                prompt_tokens=2_000,
                completion_tokens=10,
                total_tokens=2_010,
                prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            )
        return result

    agent = _make_agent(max_iterations=10)
    agent.quiet_mode = False
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "noop",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    agent.valid_tool_names = {"noop"}
    engine = _NoOpContextEngine()
    setattr(agent, "context_compressor", engine)
    agent.client.chat.completions.create.side_effect = [
        response(2_000, tool_id="before"),
        _RateLimitError(),
        response(None, tool_id="no-usage"),
        response(1_880, tool_id="after"),
        response(1_900, content="Done."),
    ]
    cache_usage = []
    agent._tui_cache_callback = lambda *_args: cache_usage.append(
        dict(agent._first_turn_usage or {})
    )
    executions = 0

    def execute_tools(assistant_message, messages, _task_id, _api_call_count):
        nonlocal executions
        call_id = assistant_message.tool_calls[0].id
        messages.append({"role": "tool", "tool_call_id": call_id, "content": "ok"})
        if executions == 0:
            setattr(agent, "_awaiting_cache_usage_after_compression", True)
        executions += 1

    with (
        patch("run_agent.time.sleep", return_value=None),
        patch.object(agent, "_execute_tool_calls", side_effect=execute_tools),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_vprint") as vprint,
    ):
        result = agent.run_conversation("work")

    assert result["final_response"] == "Done."
    assert executions == 3
    assert len(cache_usage) == 2
    assert "cache_attribution" not in cache_usage[0]
    assert cache_usage[1]["cache_attribution"] == "post_compression"
    assert cache_usage[1]["cache_read_tokens"] == 1_880
    assert agent._first_turn_usage == cache_usage[1]
    assert getattr(agent, "_awaiting_cache_usage_after_compression") is False
    assert not hasattr(engine, "awaiting_real_usage_after_compression")
    assert agent.client.chat.completions.create.call_count == 5
    cache_lines = [
        call.args[0] for call in vprint.call_args_list if "💾 Cache:" in call.args[0]
    ]
    assert len(cache_lines) == 3
    assert ["post-compression cold prefix (expected)" in line for line in cache_lines] == [
        False,
        True,
        False,
    ]


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

