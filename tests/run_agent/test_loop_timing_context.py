"""Regression coverage for API-only loop timing context."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock, patch

from agent import conversation_loop
from hermes_state import SessionDB
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


def test_next_cycle_normalized_timing_stays_exact_once_through_tool_continuation():
    t1 = datetime(2026, 8, 26, 0, 0, 0, tzinfo=UTC_MINUS_7)
    t1_stop = datetime(2026, 8, 26, 0, 0, 3, tzinfo=UTC_MINUS_7)
    t2 = datetime(2026, 8, 26, 0, 1, 0, tzinfo=UTC_MINUS_7)
    t2_stop = datetime(2026, 8, 26, 0, 1, 2, tzinfo=UTC_MINUS_7)
    stamps = iter([t1, t1_stop, t2, t2_stop])
    real_timing = conversation_loop._loop_timing_context

    def timing(agent, *args, now=None, stop=False, **kwargs):
        return real_timing(agent, *args, now=next(stamps), stop=stop, **kwargs)

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
    done = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="test/model",
        usage=None,
    )
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        done,
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
        done,
    ]
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent._fallback_chain = []
    agent.valid_tool_names = {"test_tool"}

    def execute_tool_call(_assistant_message, messages, *_args):
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
        first = agent.run_conversation("hello")
        second = agent.run_conversation("again", conversation_history=first["messages"])

    assert second["final_response"] == "done"
    assert agent.client.chat.completions.create.call_count == 3
    first_sent = agent.client.chat.completions.create.call_args_list[0].kwargs["messages"]
    second_sent = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    continuation_sent = agent.client.chat.completions.create.call_args_list[2].kwargs[
        "messages"
    ]

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
    assert len(persisted) == 1
    assert persisted[0]["role"] == "system"
    assert persisted[0]["display_kind"] == "hidden"
    assert persisted[0]["content"] in first_timing[0]["content"]
    assert persisted[0]["display_metadata"]["loop_timing_turn_id"]

    def wire(message):
        return (message.get("role"), message.get("content"))

    assert [wire(message) for message in second_sent[: len(first_sent)]] == [
        wire(message) for message in first_sent
    ]
    current_stamp = "Current loop start: 2026-08-26T00:01:00-07:00"
    second_timing = timing_messages(second_sent)
    continuation_timing = timing_messages(continuation_sent)
    persisted_timing = timing_messages(second["messages"])
    assert all(
        len([message for message in rows if current_stamp in message["content"]]) == 1
        for rows in (second_timing, continuation_timing, persisted_timing)
    )
    assert "Previous loop start: 2026-08-26T00:00:00-07:00" not in second_timing[-1]["content"]
    assert all("cache_control" not in message for message in continuation_timing)


def test_inner_tool_continuation_persists_one_timing_row():
    start = datetime(2026, 8, 26, 0, 0, 0, tzinfo=UTC_MINUS_7)
    stop = start + timedelta(seconds=3)
    stamps = iter([start, stop])
    real_timing = conversation_loop._loop_timing_context

    def timing(agent, *args, now=None, stop=False, **kwargs):
        return real_timing(agent, *args, now=next(stamps), stop=stop, **kwargs)

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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


def test_three_turns_in_one_formatted_second_keep_distinct_timing_events():
    start = datetime(2026, 8, 26, 0, 0, 0, 100_000, tzinfo=UTC_MINUS_7)
    stamps = iter(
        [
            start + timedelta(microseconds=offset)
            for offset in (0, 50_000, 100_000, 150_000, 200_000, 250_000)
        ]
    )
    real_timing = conversation_loop._loop_timing_context

    def timing(agent, *args, now=None, stop=False, **kwargs):
        return real_timing(agent, *args, now=next(stamps), stop=stop, **kwargs)

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    done = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop")],
        model="test/model",
        usage=None,
    )
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [done, done, done]
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent._fallback_chain = []

    with (
        patch("agent.conversation_loop._loop_timing_context", side_effect=timing),
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("one")
        second = agent.run_conversation("two", conversation_history=first["messages"])
        third = agent.run_conversation("three", conversation_history=second["messages"])

    timing_rows = [
        message
        for message in third["messages"]
        if message.get("display_kind") == "hidden"
        and "[Agent loop timing]" in str(message.get("content", ""))
    ]
    assert len(timing_rows) == 3
    event_ids = [row["display_metadata"]["loop_timing_turn_id"] for row in timing_rows]
    assert len(set(event_ids)) == 3


def test_cold_session_replay_adds_current_same_second_timing_event(tmp_path):
    persisted = "[Agent loop timing]\nCurrent loop start: 2026-08-26T00:00:00-07:00"
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="timing-replay", source="test", model="test/model")
    db.append_messages_batch("timing-replay", [
        {"role": "user", "content": "previous"},
        {
            "role": "system",
            "content": persisted,
            "display_kind": "hidden",
            "display_metadata": {"loop_timing_turn_id": "old-turn"},
        },
    ])
    replay = db.get_messages_as_conversation("timing-replay")
    db.close()
    start = datetime(2026, 8, 26, 0, 0, 0, 500_000, tzinfo=UTC_MINUS_7)
    real_timing = conversation_loop._loop_timing_context

    def timing(agent, *args, now=None, stop=False, **kwargs):
        return real_timing(agent, *args, now=start + timedelta(microseconds=50_000 if stop else 0), stop=stop, **kwargs)

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop")],
        model="test/model",
        usage=None,
    )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent._fallback_chain = []

    with (
        patch("agent.conversation_loop._loop_timing_context", side_effect=timing),
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("current", conversation_history=replay)

    timing_rows = [
        message
        for message in result["messages"]
        if message.get("display_kind") == "hidden"
        and "[Agent loop timing]" in str(message.get("content", ""))
    ]
    assert len(timing_rows) == 2
    event_ids = [row["display_metadata"]["loop_timing_turn_id"] for row in timing_rows]
    assert event_ids[0] == "old-turn"
    assert event_ids[1] != "old-turn"


def test_output_cap_compression_rearms_missing_timing_once():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
    assert persisted_timing[0]["display_metadata"]["loop_timing_turn_id"]
