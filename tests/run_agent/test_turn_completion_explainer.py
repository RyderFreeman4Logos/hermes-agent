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
from contextlib import nullcontext
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


def _mock_stream_chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(
        role="assistant" if content else None,
        content=content,
        tool_calls=None,
        reasoning_content=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
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


def _prime_heartbeat_snapshot(agent, messages=None):
    from tools.runtime_heartbeat import (
        begin_normal_warm_snapshot,
        finish_normal_warm_snapshot,
    )

    token = begin_normal_warm_snapshot(
        agent,
        {
            "model": agent.model,
            "messages": [
                {"role": "system", "content": agent._cached_system_prompt},
                *list(
                    messages
                    or [{"role": "user", "content": "last physical request"}]
                ),
            ],
            "tools": list(agent.tools or []),
        },
        physical_client=agent.client,
    )
    finish_normal_warm_snapshot(agent, token, succeeded=True)
    agent.client.chat.completions.create.reset_mock()


def test_partial_stream_stub_cannot_publish_heartbeat_snapshot(heartbeat_event):
    from hermes_constants import PARTIAL_STREAM_STUB_ID
    from tools.runtime_heartbeat import (
        begin_normal_warm_snapshot,
        defer_normal_warm_snapshot_until_validated,
        finish_deferred_normal_warm_snapshot,
        finish_normal_warm_snapshot,
    )

    agent = _make_agent(max_iterations=10)
    token = begin_normal_warm_snapshot(
        agent,
        {
            "model": agent.model,
            "messages": [{"role": "user", "content": "partial request"}],
        },
        physical_client=agent.client,
    )
    finish_normal_warm_snapshot(agent, token, succeeded=False)
    partial = _mock_response("partial", finish_reason="length")
    partial.id = PARTIAL_STREAM_STUB_ID
    defer_normal_warm_snapshot_until_validated(agent, token, partial)

    # The generic validator can accept the recovery envelope so the normal
    # loop can surface its partial text. Publication must still fail closed.
    finish_deferred_normal_warm_snapshot(agent, partial, succeeded=True)
    agent.client.chat.completions.create.reset_mock()

    result = agent.run_conversation(
        "",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_not_called()


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
    _prime_heartbeat_snapshot(agent, history)

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


def test_live_heartbeat_uses_one_provider_response_with_tools_disabled_on_wire(
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
    _prime_heartbeat_snapshot(agent, history)
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


def test_stuck_heartbeat_is_structured_visible_without_model_call(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    heartbeat_event["status"] = "STUCK"
    heartbeat_event["evidence"] = "process is alive but made no progress"

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is False
    assert "STUCK" in result["final_response"]
    assert "process is alive but made no progress" in result["final_response"]
    agent.client.chat.completions.create.assert_not_called()


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

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("ordinary request")
    ordinary_request = agent.client.chat.completions.create.call_args.kwargs
    agent.client.chat.completions.create.reset_mock()

    heartbeat = agent.run_conversation(
        "[HEARTBEAT] this text must never enter the replayed request",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )
    heartbeat_request = agent.client.chat.completions.create.call_args.kwargs

    assert heartbeat["silent_noop"] is True
    assert heartbeat_request["messages"] == ordinary_request["messages"]
    assert heartbeat_request["tools"] == ordinary_request["tools"]
    assert heartbeat_request["stream"] is False
    assert heartbeat_request["tool_choice"] == "none"


def test_heartbeat_replays_exact_prompt_cache_key_and_api_kwargs(
    heartbeat_event,
):
    from tools.runtime_heartbeat import (
        begin_normal_warm_snapshot,
        finish_normal_warm_snapshot,
    )

    agent = _make_agent(max_iterations=10)
    physical_request = {
        "model": agent.model,
        "messages": [
            {"role": "system", "content": "stable prefix"},
            {"role": "user", "content": "last physical request"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {}},
            }
        ],
        "prompt_cache_key": "exact-normal-scope",
        "temperature": 0.25,
        "extra_headers": {"X-Request-Lineage": "normal-physical-request"},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    token = begin_normal_warm_snapshot(
        agent,
        physical_request,
        physical_client=agent.client,
    )
    finish_normal_warm_snapshot(agent, token, succeeded=True)
    agent.client.chat.completions.create.reset_mock()

    result = agent.run_conversation(
        "",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    warm_request = agent.client.chat.completions.create.call_args.kwargs
    expected_request = dict(physical_request)
    expected_request["stream"] = False
    expected_request["tool_choice"] = "none"
    expected_request.pop("stream_options")
    assert warm_request == expected_request


def test_alive_heartbeat_without_validated_snapshot_makes_no_provider_call(
    heartbeat_event,
):
    agent = _make_agent(max_iterations=10)

    result = agent.run_conversation(
        "[HEARTBEAT] no normal request has succeeded yet",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_not_called()


def test_heartbeat_skips_when_exact_physical_client_cannot_be_leased(
    heartbeat_event, monkeypatch
):
    from tools.runtime_heartbeat import (
        begin_normal_warm_snapshot,
        finish_normal_warm_snapshot,
    )

    agent = _make_agent(max_iterations=10)
    physical_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=MagicMock())
        )
    )
    monkeypatch.setattr(
        "tools.runtime_heartbeat._supported_openai_warm_client",
        lambda _agent, client: client,
    )
    token = begin_normal_warm_snapshot(
        agent,
        {
            "model": agent.model,
            "messages": [{"role": "user", "content": "physical"}],
        },
        physical_client=physical_client,
    )
    finish_normal_warm_snapshot(agent, token, succeeded=True)
    monkeypatch.setattr(
        agent,
        "_claim_request_openai_client_for_heartbeat",
        MagicMock(return_value=None),
    )

    result = agent.run_conversation(
        "",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    physical_client.chat.completions.create.assert_not_called()
    agent.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("failure", ["first_chunk_error", "malformed_chunk"])
def test_stream_open_without_validated_response_does_not_publish_snapshot(
    heartbeat_event, failure
):
    agent = _make_agent(max_iterations=10)
    agent._api_max_retries = 1
    wire_client = MagicMock()

    if failure == "first_chunk_error":
        class _BrokenStream:
            def __iter__(self):
                return self

            def __next__(self):
                raise RuntimeError("first chunk failed")

        wire_client.chat.completions.create.return_value = _BrokenStream()
    else:
        wire_client.chat.completions.create.return_value = iter(
            [SimpleNamespace(not_choices="malformed")]
        )

    with (
        patch.object(
            agent, "_create_request_openai_client", return_value=wire_client
        ),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation(
            "ordinary streaming request",
            stream_callback=lambda _delta: None,
        )

    agent.client.chat.completions.create.reset_mock()
    result = agent.run_conversation(
        "[HEARTBEAT] invalid stream must not seed a warm request",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_not_called()


def test_validated_stream_publishes_exact_physical_request_snapshot(
    heartbeat_event,
):
    agent = _make_agent(max_iterations=10)
    agent.tools = [
        {
            "type": "function",
            "function": {"name": "web_search", "parameters": {}},
        }
    ]
    wire_client = MagicMock()
    wire_client.chat.completions.create.return_value = iter(
        [
            _mock_stream_chunk(content="streamed reply"),
            _mock_stream_chunk(finish_reason="stop"),
        ]
    )

    with (
        patch.object(
            agent, "_create_request_openai_client", return_value=wire_client
        ),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "ordinary streaming request",
            stream_callback=lambda _delta: None,
        )

    assert result["completed"] is True
    physical_request = wire_client.chat.completions.create.call_args.kwargs
    wire_client.chat.completions.create.reset_mock()
    agent.client.chat.completions.create.reset_mock()

    heartbeat = agent.run_conversation(
        "[HEARTBEAT] replay the validated stream prefix",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert heartbeat["silent_noop"] is True
    warm_request = wire_client.chat.completions.create.call_args.kwargs
    assert warm_request["messages"] == physical_request["messages"]
    assert warm_request["tools"] == physical_request["tools"]
    assert warm_request["stream"] is False
    wire_client.chat.completions.create.assert_called_once()
    agent.client.chat.completions.create.assert_not_called()


def test_older_stream_validation_cannot_overwrite_newer_success(
    heartbeat_event,
):
    from tools.runtime_heartbeat import (
        begin_normal_warm_snapshot,
        defer_normal_warm_snapshot_until_validated,
        finish_deferred_normal_warm_snapshot,
        finish_normal_warm_snapshot,
    )

    agent = _make_agent(max_iterations=10)
    old_request = {
        "model": agent.model,
        "messages": [{"role": "user", "content": "old stream"}],
        "tools": [],
        "stream": True,
    }
    new_request = {
        "model": agent.model,
        "messages": [{"role": "user", "content": "new success"}],
        "tools": [],
    }

    old_token = begin_normal_warm_snapshot(
        agent, old_request, physical_client=agent.client
    )
    finish_normal_warm_snapshot(agent, old_token, succeeded=False)
    new_token = begin_normal_warm_snapshot(
        agent, new_request, physical_client=agent.client
    )
    finish_normal_warm_snapshot(agent, new_token, succeeded=False)

    new_response = _mock_response("new response")
    defer_normal_warm_snapshot_until_validated(agent, new_token, new_response)
    finish_deferred_normal_warm_snapshot(agent, new_response, succeeded=True)

    # The older stream finishes validation after the newer physical call.
    old_response = _mock_response("late old response")
    defer_normal_warm_snapshot_until_validated(agent, old_token, old_response)
    finish_deferred_normal_warm_snapshot(agent, old_response, succeeded=True)

    agent.run_conversation(
        "[HEARTBEAT] newest physical request must win",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    warm_request = agent.client.chat.completions.create.call_args.kwargs
    assert warm_request["messages"] == new_request["messages"]


def test_heartbeat_snapshot_is_single_flight(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    _prime_heartbeat_snapshot(agent)
    entered = threading.Event()
    release = threading.Event()

    def blocked_create(**_kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return _mock_response("warm")

    agent.client.chat.completions.create.side_effect = blocked_create
    results = []
    first = threading.Thread(
        target=lambda: results.append(
            agent.run_conversation(
                "",
                turn_origin="heartbeat_warm",
                heartbeat_event=heartbeat_event,
            )
        )
    )
    first.start()
    assert entered.wait(timeout=2)

    second = threading.Thread(
        target=lambda: results.append(
            agent.run_conversation(
                "",
                turn_origin="heartbeat_warm",
                heartbeat_event=heartbeat_event,
            )
        )
    )
    second.start()
    second.join(timeout=2)
    assert not second.is_alive()
    release.set()
    first.join(timeout=2)

    assert not first.is_alive()
    assert len(results) == 2
    assert all(result["silent_noop"] is True for result in results)
    assert agent.client.chat.completions.create.call_count == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "provider",
        "model",
        "client",
        "tools",
        "system",
        "compression",
        "cache_scope",
    ],
)
def test_heartbeat_snapshot_skips_after_cache_identity_mutation(
    heartbeat_event, monkeypatch, mutation
):
    agent = _make_agent(max_iterations=10)
    agent.tools = [{"type": "function", "function": {"name": "read_file"}}]
    _prime_heartbeat_snapshot(agent)
    original_client = agent.client

    if mutation == "provider":
        agent.provider = "openai"
    elif mutation == "model":
        agent.model = "different-model"
    elif mutation == "client":
        agent.client = MagicMock()
    elif mutation == "tools":
        agent.tools.append(
            {"type": "function", "function": {"name": "write_file"}}
        )
    elif mutation == "system":
        agent._cached_system_prompt = "changed system prompt"
    elif mutation == "compression":
        agent._compression_attempt_id = "new-compression-attempt"
    else:
        monkeypatch.setattr(agent, "_prompt_cache_scope_id", lambda: "new-scope")

    result = agent.run_conversation(
        "",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    original_client.chat.completions.create.assert_not_called()
    if agent.client is not original_client:
        agent.client.chat.completions.create.assert_not_called()


def test_normal_request_inflight_invalidates_previous_snapshot(heartbeat_event):
    from tools.runtime_heartbeat import (
        begin_normal_warm_snapshot,
        finish_normal_warm_snapshot,
    )

    agent = _make_agent(max_iterations=10)
    _prime_heartbeat_snapshot(agent)
    token = begin_normal_warm_snapshot(
        agent,
        {
            "model": agent.model,
            "messages": [{"role": "user", "content": "new inflight"}],
        },
    )
    try:
        result = agent.run_conversation(
            "",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )
    finally:
        finish_normal_warm_snapshot(agent, token, succeeded=False)

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    "transport",
    [
        "anthropic_messages",
        "codex_responses",
        "bedrock_converse",
        "moa",
        "native_gemini",
        "copilot_acp",
    ],
)
def test_unsupported_successful_physical_request_cannot_seed_warm_snapshot(
    heartbeat_event, transport
):
    from agent.gemini_native_adapter import GeminiNativeClient
    from tools.runtime_heartbeat import (
        begin_normal_warm_snapshot,
        finish_normal_warm_snapshot,
    )

    agent = _make_agent(max_iterations=10)
    old_client = agent.client
    _prime_heartbeat_snapshot(agent)

    if transport in {"anthropic_messages", "codex_responses", "bedrock_converse"}:
        agent.api_mode = transport
    elif transport == "moa":
        agent.provider = "moa"
    elif transport == "native_gemini":
        agent.provider = "custom"
        agent.client = GeminiNativeClient(
            api_key="test-key", http_client=MagicMock()
        )
        agent.client._create_chat_completion = MagicMock()
    else:
        agent.provider = "custom"
        agent.client = CopilotACPClient(base_url="acp://copilot")
        agent.client._run_prompt = MagicMock(return_value=("", ""))

    token = begin_normal_warm_snapshot(
        agent,
        {
            "model": agent.model,
            "messages": [{"role": "user", "content": "unsupported physical"}],
        },
        physical_client=agent.client,
    )
    finish_normal_warm_snapshot(agent, token, succeeded=True)

    result = agent.run_conversation(
        "",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    old_client.chat.completions.create.assert_not_called()
    if transport == "native_gemini":
        agent.client._create_chat_completion.assert_not_called()
    elif transport == "copilot_acp":
        agent.client._run_prompt.assert_not_called()


def test_snapshot_request_is_immutable_after_capture(heartbeat_event):
    from tools.runtime_heartbeat import (
        begin_normal_warm_snapshot,
        finish_normal_warm_snapshot,
    )

    agent = _make_agent(max_iterations=10)
    request = {
        "model": agent.model,
        "messages": [{"role": "user", "content": "captured"}],
        "tools": [],
    }
    token = begin_normal_warm_snapshot(
        agent, request, physical_client=agent.client
    )
    request["messages"][0]["content"] = "mutated later"
    finish_normal_warm_snapshot(agent, token, succeeded=True)

    agent.run_conversation(
        "",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    warm_request = agent.client.chat.completions.create.call_args.kwargs
    assert warm_request["messages"] == [
        {"role": "user", "content": "captured"}
    ]


@pytest.mark.parametrize(
    "failure_stage", ["event", "claim", "estimate", "create"]
)
def test_heartbeat_early_path_exceptions_are_silent(
    heartbeat_event, monkeypatch, failure_stage
):
    agent = _make_agent(max_iterations=10)
    _prime_heartbeat_snapshot(agent)

    if failure_stage == "event":
        monkeypatch.setattr(
            "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
            MagicMock(side_effect=RuntimeError("event failed")),
        )
    elif failure_stage == "claim":
        monkeypatch.setattr(
            "tools.runtime_heartbeat.claim_warm_snapshot",
            MagicMock(side_effect=RuntimeError("claim failed")),
        )
    elif failure_stage == "estimate":
        monkeypatch.setattr(
            "agent.conversation_loop.estimate_request_context_tokens",
            MagicMock(side_effect=RuntimeError("estimate failed")),
        )
    else:
        agent.client.chat.completions.create.side_effect = RuntimeError(
            "create failed"
        )

    result = agent.run_conversation(
        "",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    assert result["final_response"] == ""
    assert agent.client.chat.completions.create.call_count == (
        1 if failure_stage == "create" else 0
    )


def test_heartbeat_cache_scope_lookup_exception_does_not_escape(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    _prime_heartbeat_snapshot(agent)

    class _BrokenSessionDB:
        def get_session(self, _session_id):
            raise RuntimeError("SessionDB unavailable")

    agent._session_db = _BrokenSessionDB()

    result = agent.run_conversation(
        "",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    # Prompt-cache scope resolution intentionally falls back to the stable
    # session id when SessionDB is unavailable, so the existing snapshot is
    # still safe to replay.
    agent.client.chat.completions.create.assert_called_once()


def test_heartbeat_never_enters_auxiliary_runtime_scope(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    _prime_heartbeat_snapshot(agent)

    with (
        patch(
            "agent.auxiliary_client._normalize_main_runtime",
            side_effect=AssertionError("auxiliary runtime entered"),
        ) as normalize,
        patch(
            "agent.auxiliary_client.scoped_runtime_main",
            side_effect=AssertionError("auxiliary scope entered"),
        ) as scope,
    ):
        result = agent.run_conversation(
            "",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    normalize.assert_not_called()
    scope.assert_not_called()


def test_superseded_warm_response_does_not_refresh_heartbeat_lease(
    heartbeat_event,
):
    from tools.runtime_heartbeat import (
        begin_normal_warm_snapshot,
        finish_normal_warm_snapshot,
    )

    agent = _make_agent(max_iterations=10)
    _prime_heartbeat_snapshot(agent)
    newer_token = None

    def supersede_before_response(**_kwargs):
        nonlocal newer_token
        newer_token = begin_normal_warm_snapshot(
            agent,
            {
                "model": agent.model,
                "messages": [{"role": "user", "content": "new normal"}],
            },
        )
        return _mock_response("stale warm response")

    agent.client.chat.completions.create.side_effect = supersede_before_response
    try:
        with patch(
            "tools.runtime_heartbeat.runtime_heartbeat.reset_for_caller"
        ) as reset_deadline:
            result = agent.run_conversation(
                "",
                turn_origin="heartbeat_warm",
                heartbeat_event=heartbeat_event,
            )
    finally:
        if newer_token is not None:
            finish_normal_warm_snapshot(agent, newer_token, succeeded=False)

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_called_once()
    reset_deadline.assert_not_called()


def test_heartbeat_does_not_mutate_usage_tokens_history_or_session_db(
    heartbeat_event,
):
    agent = _make_agent(max_iterations=10)
    history = [
        {"role": "user", "content": "ordinary"},
        {"role": "assistant", "content": "answer"},
    ]
    agent._session_messages = history
    agent._first_turn_usage = {"input_tokens": 100, "cache_read_tokens": 90}
    agent._last_turn_usage = {"input_tokens": 110, "output_tokens": 5}
    counter_names = (
        "session_prompt_tokens",
        "session_completion_tokens",
        "session_total_tokens",
        "session_input_tokens",
        "session_output_tokens",
        "session_cache_read_tokens",
        "session_cache_write_tokens",
        "session_reasoning_tokens",
    )
    for index, name in enumerate(counter_names, start=1):
        setattr(agent, name, index * 10)
    before_counters = {name: getattr(agent, name) for name in counter_names}
    first_usage = dict(agent._first_turn_usage)
    last_usage = dict(agent._last_turn_usage)
    class _ReadOnlySessionDB:
        reads = 0

        def get_session(self, _session_id):
            self.reads += 1
            return None

        def __getattr__(self, name):
            raise AssertionError(f"heartbeat attempted SessionDB mutation: {name}")

    session_db = _ReadOnlySessionDB()
    agent._session_db = session_db
    _prime_heartbeat_snapshot(agent, history)
    response = _mock_response("warm")
    response.usage = SimpleNamespace(
        prompt_tokens=99_999,
        completion_tokens=999,
        total_tokens=100_998,
    )
    agent.client.chat.completions.create.return_value = response

    result = agent.run_conversation(
        "",
        conversation_history=history,
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["messages"] == history
    assert agent._session_messages is history
    assert agent._first_turn_usage == first_usage
    assert agent._last_turn_usage == last_usage
    assert {name: getattr(agent, name) for name in counter_names} == before_counters
    # Cache-lineage identity may read the session row, but no mutating DB
    # surface is available to this isolated path.
    assert session_db.reads >= 1


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
    _prime_heartbeat_snapshot(agent)

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


def test_successful_provider_dispatches_reset_exact_heartbeat_group(heartbeat_event):
    from tools.approval import reset_current_session_key, set_current_session_key
    from tools.runtime_heartbeat import (
        canonical_runtime_cache_context_identity,
        canonical_runtime_provider_identity,
    )

    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop"),
        _mock_response(content="real reply", finish_reason="stop"),
    ]
    _prime_heartbeat_snapshot(agent)
    token = set_current_session_key("owner-session")
    try:
        with (
            patch(
                "tools.runtime_heartbeat.runtime_heartbeat.reset_for_caller"
            ) as reset_deadline,
            patch("agent.conversation_loop.time.monotonic", return_value=123.0),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation(
                "[HEARTBEAT] inspect target",
                turn_origin="heartbeat_warm",
                heartbeat_event=heartbeat_event,
            )
            reset_deadline.assert_called_once_with(
                "owner-session",
                provider="openrouter",
                cache_context="test-cache-context",
                activity_at=123.0,
            )

            reset_deadline.reset_mock()
            agent.run_conversation("real user turn")
            reset_deadline.assert_called_once_with(
                "owner-session",
                provider=canonical_runtime_provider_identity(agent),
                cache_context=canonical_runtime_cache_context_identity(agent),
                activity_at=123.0,
            )
    finally:
        reset_current_session_key(token)


def test_failed_provider_calls_do_not_extend_heartbeat_lease(heartbeat_event):
    from tools.approval import reset_current_session_key, set_current_session_key

    agent = _make_agent(max_iterations=10)
    agent._api_max_retries = 1
    agent.client.chat.completions.create.side_effect = [
        RuntimeError("warm failed before response"),
        RuntimeError("ordinary failed before response"),
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

    reset_deadline.assert_not_called()


@pytest.mark.parametrize(
    ("streaming", "short_circuit"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=[
        "nonstream-physical",
        "nonstream-short-circuit",
        "stream-physical",
        "stream-short-circuit",
    ],
)
def test_provider_lease_requires_physical_relay_dispatch(
    tmp_path, monkeypatch, streaming, short_circuit
):
    relay = pytest.importorskip("nemo_relay")
    from agent import relay_llm, relay_runtime
    from tools.approval import reset_current_session_key, set_current_session_key

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    agent = _make_agent(max_iterations=10)
    host = relay_runtime.get_runtime()
    assert host is not None
    host.retain_managed_execution("test.provider_lease")
    physical_call = MagicMock()
    intercept_name = f"provider-lease-{'stream' if streaming else 'execute'}"
    wire_client = None

    if streaming:
        chunks = [
            _mock_stream_chunk(content="relay reply"),
            _mock_stream_chunk(finish_reason="stop"),
        ]
        physical_call.return_value = iter(chunks)
        wire_client = MagicMock()
        wire_client.chat.completions.create = physical_call
        agent.client = SimpleNamespace()

        def stream_intercept(request, next_call):
            async def generate():
                if short_circuit:
                    for chunk in chunks:
                        yield relay_llm._jsonable(chunk)
                    return
                upstream = await next_call(request)
                async for chunk in upstream:
                    yield chunk

            return generate()

        relay.intercepts.register_llm_stream_execution(
            intercept_name, 1, stream_intercept
        )
    else:
        assert agent.client is not None
        physical_call = agent.client.chat.completions.create
        physical_call.return_value = _mock_response(content="physical reply")

        def execute_intercept(_name, request, next_call):
            if short_circuit:
                return relay_llm._jsonable(_mock_response(content="relay reply"))
            return next_call(request)

        relay.intercepts.register_llm_execution(intercept_name, 1, execute_intercept)

    token = set_current_session_key("owner-session")
    try:
        with (
            patch(
                "tools.runtime_heartbeat.runtime_heartbeat.reset_for_caller"
            ) as reset_deadline,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(
                agent, "_create_request_openai_client", return_value=wire_client
            )
            if streaming
            else nullcontext(),
            patch.object(agent, "_close_request_openai_client")
            if streaming
            else nullcontext(),
        ):
            result = agent.run_conversation(
                "real user turn",
                stream_callback=(lambda _delta: None) if streaming else None,
            )
        assert result["completed"] is True
        assert physical_call.call_count == (0 if short_circuit else 1)
        assert reset_deadline.call_count == (0 if short_circuit else 1)
    finally:
        reset_current_session_key(token)
        if streaming:
            relay.intercepts.deregister_llm_stream_execution(intercept_name)
        else:
            relay.intercepts.deregister_llm_execution(intercept_name)
        host.release_managed_execution("test.provider_lease")
        relay_runtime._reset_for_tests()


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
    _prime_heartbeat_snapshot(agent, history)

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
    _prime_heartbeat_snapshot(agent)

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
    assert len(cache_usage) == 3
    assert "cache_attribution" not in cache_usage[0]
    assert cache_usage[1]["cache_attribution"] == "post_compression"
    assert cache_usage[1]["cache_read_tokens"] == 1_880
    assert "cache_attribution" not in cache_usage[2]
    assert cache_usage[2]["cache_read_tokens"] == 1_900
    assert agent._first_turn_usage == cache_usage[2]
    assert getattr(agent, "_awaiting_cache_usage_after_compression") is False
    assert not hasattr(engine, "awaiting_real_usage_after_compression")
    assert agent.client.chat.completions.create.call_count == 5
    cache_lines = [
        call.args[0] for call in vprint.call_args_list if "💾 Cache:" in call.args[0]
    ]
    assert len(cache_lines) == 3
    assert ["post-compression warmup (expected)" in line for line in cache_lines] == [
        False,
        True,
        False,
    ]
    assert "\033[31m94%\033[0m" not in cache_lines[1]


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
    _prime_heartbeat_snapshot(agent)
    agent.client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[], usage=None),
        _mock_response(content="unexpected retry", finish_reason="stop"),
    ]

    with (
        patch(
            "tools.runtime_heartbeat.runtime_heartbeat.reset_for_caller"
        ) as reset_deadline,
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
    reset_deadline.assert_not_called()
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


def test_gemini_chat_completions_heartbeat_skips_transport(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.provider = "gemini"
    agent.requested_provider = "gemini"
    agent.api_mode = "chat_completions"
    agent.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_not_called()


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
    _prime_heartbeat_snapshot(agent)
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
    _prime_heartbeat_snapshot(agent)
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
    _prime_heartbeat_snapshot(agent)
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


def test_unknown_heartbeat_is_structured_visible_without_model_call(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    heartbeat_event["status"] = "UNKNOWN"
    heartbeat_event["evidence"] = "no output or CPU progress"

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is False
    assert "UNKNOWN" in result["final_response"]
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
    _prime_heartbeat_snapshot(agent, heartbeat_history)

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


def test_heartbeat_skips_custom_alias_resolved_to_gemini_native(heartbeat_event):
    from agent.gemini_native_adapter import GeminiNativeClient

    agent = _make_agent()
    agent.provider = "custom"
    agent.requested_provider = "custom:gemini"
    agent.base_url = "https://generativelanguage.googleapis.com/v1beta"
    gemini_client = GeminiNativeClient(
        api_key="test-key", http_client=MagicMock()
    )
    gemini_client._create_chat_completion = MagicMock()
    agent.client = gemini_client

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    gemini_client._create_chat_completion.assert_not_called()


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
    _prime_heartbeat_snapshot(agent)

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_called_once()
