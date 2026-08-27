import copy
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tui_gateway import server


TIMING = (
    "[Agent loop timing]\n"
    "Previous loop start: 2026-08-25T09:00:00-07:00\n"
    "Previous loop stop: 2026-08-25T09:00:03-07:00\n"
    "Current loop start: 2026-08-25T09:33:25-07:00"
)


def _strict_provider_accepts_message_shape(messages):
    shape = [
        {
            "index": index,
            "role": message.get("role"),
            "has_cache_control": "cache_control" in message,
        }
        for index, message in enumerate(messages)
    ]
    logging.getLogger(__name__).info("strict-provider message shape=%s", shape)
    if any(item["role"] == "system" and item["index"] for item in shape):
        raise ValueError("System message must be at the beginning.")
    return shape


@pytest.mark.parametrize(
    ("running", "status", "expected"),
    [
        (False, "ignored", ""),
        (True, "appended", TIMING),
    ],
)
def test_loop_timing_never_starts_or_queues_a_user_turn(
    monkeypatch, running, status, expected
):
    agent = type("Agent", (), {"_loop_timing_context_text": ""})()
    session = {
        "agent": agent,
        "session_key": "timing-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": running,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
    }
    server._sessions["timing"] = session
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("loop timing started a new model turn")
        ),
    )
    monkeypatch.setattr(
        server,
        "_handle_busy_submit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("loop timing was queued as a user turn")
        ),
    )

    try:
        response = server.handle_request(
            {
                "id": "timing",
                "method": "prompt.submit",
                "params": {"session_id": "timing", "text": TIMING},
            }
        )
    finally:
        server._sessions.pop("timing", None)

    assert response["result"] == {"status": status}
    assert agent._loop_timing_context_text == expected
    assert session["history"] == []


def test_inflight_loop_timing_is_consumed_onto_outgoing_prompt():
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

    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="done",
                    reasoning_content=None,
                    reasoning=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        model="test/model",
        usage=None,
    )
    agent._cached_system_prompt = "You are helpful."
    agent._cached_system_prompt_static = "You are helpful."
    agent._use_prompt_caching = True
    agent._disable_streaming = True
    agent._loop_timing_context_text = TIMING

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
    shape = _strict_provider_accepts_message_shape(sent)
    assert shape[-1]["role"] == "user"
    assert len(shape) == 2
    assert agent._loop_timing_context_text == ""
    assert all(message.get("content") != TIMING for message in result["messages"])


def test_loop_timing_stays_unmarked_across_retry_and_fallback():
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

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._cached_system_prompt_static = "You are helpful."
    agent._use_prompt_caching = True
    agent._disable_streaming = True
    agent._loop_timing_context_text = TIMING
    agent._fallback_chain = [{"provider": "openrouter", "model": "fallback/model"}]
    agent._fallback_index = 0
    sent = []
    empty = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None))],
        model="test/model",
        usage=None,
    )
    complete = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="done", reasoning_content=None, reasoning=None, tool_calls=None
                ),
                finish_reason="stop",
            )
        ],
        model="fallback/model",
        usage=None,
    )
    outcomes = [UnicodeEncodeError("utf-8", "x", 0, 1, "retry"), empty, complete]

    def _send(**kwargs):
        sent.append(copy.deepcopy(kwargs["messages"]))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def _activate_fallback():
        agent._fallback_index = 1
        agent.provider = "openrouter"
        agent._use_prompt_caching = True
        agent._use_native_cache_layout = False
        return True

    agent.client.chat.completions.create.side_effect = _send
    with (
        patch.object(agent, "_try_activate_fallback", side_effect=_activate_fallback),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert len(sent) == 3
    prior = sent[0][-1]["content"][0]
    for attempt in sent:
        timing_blocks = [
            part
            for message in attempt
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
            if isinstance(part, dict) and part.get("text") == TIMING
        ]
        assert len(timing_blocks) == 1
        assert "cache_control" not in timing_blocks[0]
        assert all(not (index and message.get("role") == "system") for index, message in enumerate(attempt))
        assert attempt[-1]["content"][0] == prior


def test_non_chat_loop_timing_follows_cached_breakpoint_without_marker():
    agent = type("Agent", (), {"api_mode": "anthropic_messages"})()
    agent._loop_timing_context_text = TIMING
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "prior", "cache_control": {"type": "ephemeral"}}]}
    ]

    from agent.conversation_loop import _append_loop_timing_context

    sent = _append_loop_timing_context(agent, messages)

    assert sent[-1] == {"role": "system", "content": TIMING}
    assert "cache_control" not in sent[-1]
