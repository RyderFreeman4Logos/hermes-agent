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
import queue
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_engine import ContextEngine
from agent.copilot_acp_client import CopilotACPClient
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


@pytest.fixture
def heartbeat_event(monkeypatch):
    event = {
        "type": "heartbeat",
        "target_id": "proc-heartbeat",
        "target_ids": ["proc-heartbeat"],
        "generations": [7],
        "generation": 7,
        "session_key": "owner-session",
        "provider": "openrouter",
        "cache_context": "test-cache-context",
        "status": "ALIVE",
        "evidence": "output grew",
    }
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        lambda candidate, agent=None, **_kwargs: candidate is event,
    )
    return event


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


def test_heartbeat_silent_noop_leaves_no_durable_or_live_history(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop")
    ]
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]
    agent._session_messages = history

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
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    assert result["final_response"] == ""
    assert result["messages"] == history
    assert agent._session_messages == history
    persist.assert_not_called()
    trajectory.assert_not_called()
    external_memory.assert_not_called()


def test_heartbeat_uses_one_provider_response_with_tools_disabled_on_wire(
    heartbeat_event,
):
    agent = _make_agent(max_iterations=10)
    agent.tools = [
        {
            "type": "function",
            "function": {"name": "web_search", "parameters": {}},
        }
    ]
    tool_call = SimpleNamespace(
        id="heartbeat-tool",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response(content="Target is still working.", finish_reason="stop"),
    ]
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]
    agent._session_messages = history
    persisted = []

    with (
        patch.object(agent, "_execute_tool_calls") as execute,
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
            conversation_history=history,
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    assert result["final_response"] == ""
    assert result["messages"] == history
    assert agent._session_messages == history
    assert agent.client.chat.completions.create.call_count == 1
    execute.assert_not_called()
    assert persisted == []
    request = agent.client.chat.completions.create.call_args.kwargs
    assert request["messages"][0]["role"] == "system"
    assert "You are helpful." in request["messages"][0]["content"]
    assert request["tools"] == agent.tools
    assert request["tool_choice"] == "none"
    assert request["stream"] is False


def test_heartbeat_matches_ordinary_effective_cache_prefix(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.ephemeral_system_prompt = "EPHEMERAL-SYSTEM"
    agent.tools = [
        {
            "type": "function",
            "function": {"name": "web_search", "parameters": {}},
        }
    ]
    agent.client.chat.completions.create.return_value = _mock_response("ok")

    heartbeat = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )
    heartbeat_request = agent.client.chat.completions.create.call_args.kwargs
    agent.client.chat.completions.create.reset_mock()

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("ordinary request")
    ordinary_request = agent.client.chat.completions.create.call_args.kwargs

    assert heartbeat["silent_noop"] is True
    assert heartbeat_request["messages"][0] == ordinary_request["messages"][0]
    assert heartbeat_request["tools"] == ordinary_request["tools"]


@pytest.mark.parametrize(
    ("threshold", "request_tokens"),
    [(100, 101), (400_000, 272_000)],
)
def test_heartbeat_skips_provider_at_compression_or_hard_limit(
    threshold, request_tokens, heartbeat_event
):
    agent = _make_agent(max_iterations=10)
    agent.context_compressor.threshold_tokens = threshold
    agent.client.chat.completions.create.return_value = _mock_response("unexpected")

    with (
        patch(
            "agent.conversation_loop.estimate_request_context_tokens",
            return_value=request_tokens,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    assert result["messages"] == []
    agent.client.chat.completions.create.assert_not_called()


def test_real_caller_activity_resets_heartbeat_deadline(heartbeat_event):
    from tools.approval import reset_current_session_key, set_current_session_key

    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop"),
        _mock_response(content="real reply", finish_reason="stop"),
    ]
    token = set_current_session_key("owner-session")
    try:
        with (
            patch(
                "tools.runtime_heartbeat.runtime_heartbeat.reset_for_caller"
            ) as reset_deadline,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation(
                "[HEARTBEAT] inspect target",
                turn_origin="heartbeat_warm",
                    heartbeat_event=heartbeat_event,
            )
            agent.run_conversation("real user turn")
    finally:
        reset_current_session_key(token)

    reset_deadline.assert_called_once_with("owner-session")


def test_heartbeat_early_error_leaves_no_unmatched_synthetic_user_row(
    heartbeat_event,
):
    agent = _make_agent(max_iterations=10)
    agent.provider = "nous"
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]
    agent._session_messages = history

    with (
        patch("agent.nous_rate_guard.nous_rate_limit_remaining", return_value=60),
        patch("agent.nous_rate_guard.format_remaining", return_value="1m"),
        patch.object(agent, "_try_activate_fallback", return_value=True) as fallback,
        patch.object(agent, "_save_session_log") as save_log,
        patch.object(agent, "_flush_messages_to_session_db") as flush,
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            conversation_history=history,
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["messages"] == history
    assert agent._session_messages == history
    assert agent.client.chat.completions.create.call_count == 1
    fallback.assert_not_called()
    save_log.assert_not_called()
    flush.assert_not_called()


def test_heartbeat_bypasses_ordinary_lifecycle_hooks(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.return_value = _mock_response("still alive")

    with (
        patch.dict(os.environ, {"HERMES_DUMP_REQUESTS": "1"}),
        patch("hermes_cli.lifecycle.invoke_hook") as invoke_hook,
        patch.object(agent, "_dump_api_request_debug") as dump_request,
        patch.object(agent, "_persist_session") as persist,
        patch.object(agent, "_save_trajectory") as trajectory,
        patch.object(agent, "_sync_external_memory_for_turn") as external_memory,
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    invoke_hook.assert_not_called()
    dump_request.assert_not_called()
    persist.assert_not_called()
    trajectory.assert_not_called()
    external_memory.assert_not_called()


def test_heartbeat_does_not_consume_user_maintenance_triggers(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent._user_turn_count = 9
    agent._turns_since_memory = 4
    agent._memory_nudge_interval = 5
    agent._memory_store = MagicMock()
    agent._iters_since_skill = 5
    agent._skill_nudge_interval = 5
    agent.valid_tool_names = {"memory", "skill_manage"}
    agent._memory_manager = MagicMock()
    agent._memory_manager.prefetch_all.return_value = ""
    agent._pending_steer = "queued user steer"
    agent._pending_redirect = "queued user redirect"
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
            heartbeat_event=heartbeat_event,
        )
        assert agent._user_turn_count == 9
        assert agent._turns_since_memory == 4
        assert agent._iters_since_skill == 5
        assert agent._pending_steer == "queued user steer"
        assert agent._pending_redirect == "queued user redirect"
        agent._memory_manager.on_turn_start.assert_not_called()
        agent._memory_manager.prefetch_all.assert_not_called()
        review.assert_not_called()
        cached_system_prompt = agent._cached_system_prompt

        agent._pending_steer = None
        agent._pending_redirect = None
        result = agent.run_conversation("real user turn")

    assert result["final_response"] == "real reply"
    assert agent._user_turn_count == 10
    assert agent._turns_since_memory == 0
    assert agent._iters_since_skill == 0
    assert agent._cached_system_prompt == cached_system_prompt
    agent._memory_manager.on_turn_start.assert_called_once()
    agent._memory_manager.prefetch_all.assert_called_once()
    review.assert_called_once()


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


def test_heartbeat_malformed_response_never_retries_or_falls_back(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[], usage=None),
        _mock_response(content="unexpected retry", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_try_activate_fallback", return_value=False) as fallback,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    assert agent.client.chat.completions.create.call_count == 1
    fallback.assert_not_called()


def test_heartbeat_skips_moa_fanout(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.provider = "moa"
    agent.client.chat.completions.prepare.side_effect = AssertionError(
        "heartbeat must not fan out through MoA"
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    agent.client.chat.completions.prepare.assert_not_called()
    agent.client.chat.completions.create.assert_not_called()


def test_anthropic_heartbeat_skips_without_any_transport_or_fallback(
    heartbeat_event,
):
    agent = _make_agent(max_iterations=10)
    agent.provider = "anthropic"
    agent.api_mode = "anthropic_messages"

    with (
        patch.object(agent, "_create_request_openai_client") as openai_client,
        patch.object(agent, "_create_request_anthropic_client") as anthropic_client,
        patch.object(agent, "_anthropic_messages_create") as anthropic_dispatch,
        patch.object(agent, "_try_activate_fallback") as fallback,
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    openai_client.assert_not_called()
    anthropic_client.assert_not_called()
    anthropic_dispatch.assert_not_called()
    agent.client.chat.completions.create.assert_not_called()
    fallback.assert_not_called()


def test_copilot_acp_heartbeat_skips_without_transport_dispatch(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.provider = "copilot-acp"
    agent.requested_provider = "copilot-acp"
    agent.api_mode = "chat_completions"

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_not_called()


def test_heartbeat_skips_provider_switched_during_final_target_inspection(
    monkeypatch,
):
    from tools.runtime_heartbeat import (
        RuntimeHeartbeat,
        canonical_runtime_cache_context_identity,
        canonical_runtime_provider_identity,
    )

    timers = []

    class Timer:
        def __init__(self, _interval, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    inspections = 0
    final_inspection_started = threading.Event()
    release_final_inspection = threading.Event()

    def inspect():
        nonlocal inspections
        inspections += 1
        if inspections == 4:
            final_inspection_started.set()
            assert release_final_inspection.wait(timeout=2)
        return {"alive": True, "progress": True}

    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=Timer)
    agent = _make_agent(max_iterations=10)
    old_client = agent.client
    old_client.chat.completions.create.return_value = _mock_response("old")
    manager.arm(
        "target",
        caller_id="owner-session",
        kind="delegation",
        interval=1700,
        inspect=inspect,
        provider=canonical_runtime_provider_identity(agent),
        cache_context=canonical_runtime_cache_context_identity(agent),
    )
    timers[0].callback()
    event = events.get_nowait()
    monkeypatch.setattr("tools.runtime_heartbeat.runtime_heartbeat", manager)

    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            agent.run_conversation(
                "[HEARTBEAT] inspect target",
                turn_origin="heartbeat_warm",
                    heartbeat_event=event,
            )
        )
    )
    worker.start()
    assert final_inspection_started.wait(timeout=2)

    new_client = MagicMock()
    new_client.chat.completions.create.return_value = _mock_response("new")
    agent.provider = "openai"
    agent.requested_provider = "openai"
    agent.base_url = "https://api.openai.com/v1"
    agent.client = new_client
    release_final_inspection.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert result[0]["silent_noop"] is True
    old_client.chat.completions.create.assert_not_called()
    new_client.chat.completions.create.assert_not_called()


def test_heartbeat_revalidates_generation_at_provider_boundary(
    heartbeat_event, monkeypatch
):
    agent = _make_agent(max_iterations=10)
    checks = iter((True, False))
    calls = []

    def current(candidate, agent=None, **kwargs):
        calls.append(kwargs)
        return next(checks)

    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        current,
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    assert calls == [{}, {"consume": True}]
    agent.client.chat.completions.create.assert_not_called()


def test_heartbeat_never_enters_request_middleware(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.return_value = _mock_response("still alive")

    with (
        patch("hermes_cli.middleware.run_llm_execution_middleware") as middleware,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    middleware.assert_not_called()
    agent.client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("status", ["STUCK", "UNKNOWN"])
def test_unhealthy_heartbeat_is_structured_visible_without_model_call(
    heartbeat_event, status
):
    agent = _make_agent(max_iterations=10)
    heartbeat_event["status"] = status
    heartbeat_event["evidence"] = "no output or CPU progress"

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is False
    assert status in result["final_response"]
    assert "no output or CPU progress" in result["final_response"]
    agent.client.chat.completions.create.assert_not_called()


def test_heartbeat_completion_preserves_unowned_marker_and_history(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    heartbeat_history = [{"role": "assistant", "content": "before heartbeat"}]
    ordinary_history = [{"role": "user", "content": "ordinary turn"}]

    def complete_heartbeat(**_kwargs):
        agent._inflight_turn_id = "ordinary-turn"
        agent._session_messages = ordinary_history
        return _mock_response(content="", finish_reason="stop")

    agent.client.chat.completions.create.side_effect = complete_heartbeat

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation(
            "[HEARTBEAT] inspect target",
            conversation_history=heartbeat_history,
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert agent._inflight_turn_id == "ordinary-turn"
    assert agent._session_messages is ordinary_history


def test_heartbeat_skips_custom_alias_resolved_to_copilot_acp(heartbeat_event):
    agent = _make_agent()
    agent.provider = "custom"
    agent.base_url = "acp://copilot"
    acp_client = CopilotACPClient(base_url=agent.base_url)
    acp_client._run_prompt = MagicMock(return_value=("", ""))
    agent.client = acp_client

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    acp_client._run_prompt.assert_not_called()


def test_heartbeat_allows_supported_custom_openai_transport(heartbeat_event):
    agent = _make_agent()
    agent.provider = "custom"
    agent.base_url = "https://custom.invalid/v1"

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_called_once()
