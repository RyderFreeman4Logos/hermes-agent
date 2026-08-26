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
    timing_messages = [
        message for message in sent if message.get("content") == TIMING
    ]
    assert sent[-1] == {"role": "system", "content": TIMING}
    assert timing_messages == [{"role": "system", "content": TIMING}]
    assert agent._loop_timing_context_text == ""
    persisted = [message for message in result["messages"] if message.get("content") == TIMING]
    assert persisted == [{"role": "system", "content": TIMING, "display_kind": "hidden"}]
    assert all(message.get("role") != "user" for message in persisted)
