"""Regression coverage for API-only loop timing context."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock, patch

from agent import conversation_loop
from run_agent import AIAgent


UTC_MINUS_7 = timezone(timedelta(hours=-7))


def test_loop_timing_is_default_on_but_config_gated():
    timing_context: Callable[..., str | None] | None = getattr(
        conversation_loop, "_loop_timing_context", None
    )
    assert callable(timing_context), "loop timing context must be available to the turn forwarder"

    agent = SimpleNamespace()
    start = datetime(2026, 8, 22, 11, 28, 0, tzinfo=UTC_MINUS_7)
    stop = start + timedelta(seconds=3)
    next_start = stop + timedelta(seconds=2)

    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        context = timing_context(agent, now=start)
        assert context is not None
        assert "Current loop start: 2026-08-22T11:28:00-07:00" in context
        assert timing_context(agent, now=stop, stop=True) is None

    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"agent": {"loop_timing_context": False}},
    ):
        assert timing_context(agent, now=next_start) == ""

    assert agent._loop_timing_last_start == next_start
    assert agent._loop_timing_last_stop == stop


def test_next_cycle_keeps_prior_timing_in_historical_prefix():
    t1 = datetime(2026, 8, 26, 0, 0, 0, tzinfo=UTC_MINUS_7)
    t1_stop = datetime(2026, 8, 26, 0, 0, 3, tzinfo=UTC_MINUS_7)
    t2 = datetime(2026, 8, 26, 0, 1, 0, tzinfo=UTC_MINUS_7)
    t2_stop = datetime(2026, 8, 26, 0, 1, 2, tzinfo=UTC_MINUS_7)
    stamps = iter([t1, t1_stop, t2, t2_stop])
    real_timing = conversation_loop._loop_timing_context

    def timing(agent, *args, now=None, stop=False, **kwargs):
        return real_timing(agent, *args, now=next(stamps), stop=stop, **kwargs)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
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
                message=SimpleNamespace(content="done", tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="test/model",
        usage=None,
    )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent._fallback_chain = []

    with (
        patch("agent.conversation_loop._loop_timing_context", side_effect=timing),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("hello")
        second = agent.run_conversation("again", conversation_history=first["messages"])

    first_sent = agent.client.chat.completions.create.call_args_list[0].kwargs["messages"]
    second_sent = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]

    def timing_messages(messages):
        return [
            message
            for message in messages
            if "[Agent loop timing]" in str(message.get("content", ""))
        ]

    first_timing = timing_messages(first_sent)
    assert len(first_timing) == 1
    assert "cache_control" not in first_timing[0]

    persisted = timing_messages(first["messages"])
    assert persisted == [
        {
            "role": "system",
            "content": first_timing[0]["content"],
            "display_kind": "hidden",
        }
    ]

    def wire(message):
        return (message.get("role"), message.get("content"))

    assert [wire(message) for message in second_sent[: len(first_sent)]] == [
        wire(message) for message in first_sent
    ]
    second_timing = timing_messages(second_sent)
    assert "Current loop start: 2026-08-26T00:01:00-07:00" in second_timing[-1]["content"]
    assert "Previous loop start: 2026-08-26T00:00:00-07:00" not in second_timing[-1]["content"]
    assert all("cache_control" not in message for message in second_timing)


def test_inner_tool_continuation_persists_one_timing_row():
    start = datetime(2026, 8, 26, 0, 0, 0, tzinfo=UTC_MINUS_7)
    stop = start + timedelta(seconds=3)
    stamps = iter([start, stop])
    real_timing = conversation_loop._loop_timing_context

    def timing(agent, *args, now=None, stop=False, **kwargs):
        return real_timing(agent, *args, now=next(stamps), stop=stop, **kwargs)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="test_tool", arguments="{}"),
    )
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None, tool_calls=[tool_call]),
                    finish_reason="tool_calls",
                )
            ],
            model="test/model",
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="test/model",
            usage=None,
        ),
    ]
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent._fallback_chain = []
    agent.valid_tool_names = {"test_tool"}

    def execute_tool_call(assistant_message, messages, *_args):
        messages.append(
            {
                "role": "tool",
                "name": "test_tool",
                "tool_call_id": "call-1",
                "content": "ok",
            }
        )

    with (
        patch("agent.conversation_loop._loop_timing_context", side_effect=timing),
        patch.object(agent, "_execute_tool_calls", side_effect=execute_tool_call),
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["final_response"] == "done"
    assert agent.client.chat.completions.create.call_count == 2
    timing_rows = [
        message
        for message in result["messages"]
        if "[Agent loop timing]" in str(message.get("content", ""))
    ]
    assert len(timing_rows) == 1
    assert "Current loop start: 2026-08-26T00:00:00-07:00" in timing_rows[0]["content"]


def test_output_cap_compression_rearms_missing_timing_once():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent.model = "some/model"
    agent.max_tokens = 65_536
    agent.compression_enabled = True
    agent.context_compressor.context_length = 200_000
    agent.context_compressor.should_compress = MagicMock(return_value=False)
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._fallback_chain = []

    exc = Exception(
        "max_tokens: 65536 > context_window: 200000 "
        "- input_tokens: 199000 = available_tokens: 1000"
    )
    exc.status_code = 400
    exc.code = 400
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        exc,
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="some/model",
            usage=None,
        ),
    ]
    mock_compress = MagicMock(
        return_value=(
            [
                {"role": "assistant", "content": "previous"},
                {"role": "user", "content": "hello"},
            ],
            "You are helpful.",
        )
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent.context_compressor, "update_model"),
        patch.object(agent, "_compress_context", mock_compress),
    ):
        result = agent.run_conversation("hello")

    mock_compress.assert_called_once()
    retry_messages = agent.client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    retry_timing = [
        message
        for message in retry_messages
        if "[Agent loop timing]" in str(message.get("content", ""))
    ]
    persisted_timing = [
        message
        for message in result["messages"]
        if "[Agent loop timing]" in str(message.get("content", ""))
    ]
    assert len(retry_timing) == 1
    assert len(persisted_timing) == 1
    assert "cache_control" not in retry_timing[0]
