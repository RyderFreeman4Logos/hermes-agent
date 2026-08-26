"""Behavior tests for per-loop timing context injection."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import conversation_loop
from agent.conversation_loop import _loop_timing_context
from run_agent import AIAgent


UTC_MINUS_7 = timezone(timedelta(hours=-7))


def _agent():
    return SimpleNamespace()


def test_first_loop_includes_only_current_start():
    agent = _agent()
    current_start = datetime(2026, 8, 22, 11, 28, 3, tzinfo=UTC_MINUS_7)

    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        context = _loop_timing_context(agent, now=current_start)

    assert "Current loop start: 2026-08-22T11:28:03-07:00" in context
    assert "Previous loop start:" not in context
    assert "Previous loop stop:" not in context


def test_later_loop_includes_stamps_for_gap_and_duration():
    agent = _agent()
    previous_start = datetime(2026, 8, 22, 11, 28, 0, tzinfo=UTC_MINUS_7)
    previous_stop = datetime(2026, 8, 22, 11, 28, 3, tzinfo=UTC_MINUS_7)
    current_start = datetime(2026, 8, 22, 11, 28, 5, tzinfo=UTC_MINUS_7)

    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        _loop_timing_context(agent, now=previous_start)
        _loop_timing_context(agent, now=previous_stop, stop=True)
        context = _loop_timing_context(agent, now=current_start)

    assert "Previous loop start: 2026-08-22T11:28:00-07:00" in context
    assert "Previous loop stop: 2026-08-22T11:28:03-07:00" in context
    assert "Current loop start: 2026-08-22T11:28:05-07:00" in context
    assert current_start - previous_stop == timedelta(seconds=2)
    assert previous_stop - previous_start == timedelta(seconds=3)


def test_disabled_loop_timing_still_records_stamps():
    agent = _agent()
    current_start = datetime(2026, 8, 22, 11, 28, 3, tzinfo=UTC_MINUS_7)
    current_stop = datetime(2026, 8, 22, 11, 28, 7, tzinfo=UTC_MINUS_7)

    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"agent": {"loop_timing_context": False}},
    ):
        assert _loop_timing_context(agent, now=current_start) == ""
        assert _loop_timing_context(agent, now=current_stop, stop=True) is None

    assert agent._loop_timing_last_start == current_start
    assert agent._loop_timing_last_stop == current_stop


def test_loop_timing_config_hot_reloads_between_loops(tmp_path, monkeypatch):
    agent = _agent()
    first_start = datetime(2026, 8, 22, 11, 28, 0, tzinfo=UTC_MINUS_7)
    first_stop = datetime(2026, 8, 22, 11, 28, 3, tzinfo=UTC_MINUS_7)
    second_start = datetime(2026, 8, 22, 11, 28, 5, tzinfo=UTC_MINUS_7)
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config_path.write_text("agent:\n  loop_timing_context: false\n")
    assert _loop_timing_context(agent, now=first_start) == ""
    _loop_timing_context(agent, now=first_stop, stop=True)

    config_path.write_text("agent:\n  loop_timing_context: true\n")
    context = _loop_timing_context(agent, now=second_start)

    assert "Previous loop stop: 2026-08-22T11:28:03-07:00" in context
    assert "Current loop start: 2026-08-22T11:28:05-07:00" in context


def test_loop_timing_default_is_enabled():
    agent = _agent()
    current_start = datetime(2026, 8, 22, 11, 28, 3, tzinfo=UTC_MINUS_7)

    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert _loop_timing_context(agent, now=current_start)


def test_agent_forwarder_exposes_timing_context_to_loop_and_records_stop():
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

    seen_context = []

    def fake_loop(*args, **kwargs):
        seen_context.append(agent._loop_timing_context_text)
        return {"final_response": "ok", "messages": [], "api_calls": 1}

    with (
        patch(
            "agent.conversation_loop._loop_timing_context",
            side_effect=["timing context", None],
        ) as timing,
        patch("agent.conversation_loop.run_conversation", side_effect=fake_loop),
    ):
        agent.run_conversation("hello")

    assert seen_context == ["timing context"]
    assert timing.call_count == 2
    assert timing.call_args_list[1].kwargs["stop"] is True


def test_timing_context_is_injected_as_api_only_system_context():
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
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("hello")

    sent_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages[0] == {"role": "system", "content": "You are helpful."}
    timing_messages = [
        message
        for message in sent_messages[1:]
        if "[Agent loop timing]" in str(message.get("content", ""))
    ]
    assert len(timing_messages) == 1
    assert timing_messages[0]["role"] == "system"
    assert "Current loop start:" in timing_messages[0]["content"]
    assert "cache_control" not in timing_messages[0]


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
        second = agent.run_conversation(
            "again", conversation_history=first["messages"]
        )

    first_sent = agent.client.chat.completions.create.call_args_list[0].kwargs["messages"]
    second_sent = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]

    def _timing_messages(messages):
        return [
            message
            for message in messages
            if "[Agent loop timing]" in str(message.get("content", ""))
        ]

    first_timing = _timing_messages(first_sent)
    assert len(first_timing) == 1
    assert first_timing[0]["role"] == "system"
    assert "Current loop start: 2026-08-26T00:00:00-07:00" in first_timing[0]["content"]
    assert "cache_control" not in first_timing[0]

    persisted = _timing_messages(first["messages"])
    assert len(persisted) == 1
    assert persisted[0]["role"] == "system"
    assert persisted[0]["role"] != "user"

    def _wire(message):
        return (message.get("role"), message.get("content"))

    assert [_wire(message) for message in second_sent[: len(first_sent)]] == [
        _wire(message) for message in first_sent
    ]

    second_timing = _timing_messages(second_sent)
    assert any(
        message.get("content") == first_timing[0]["content"]
        and message.get("role") == "system"
        for message in second_timing
    )
    assert "Current loop start: 2026-08-26T00:01:00-07:00" in second_timing[-1]["content"]
    assert "Previous loop start: 2026-08-26T00:00:00-07:00" not in second_timing[-1]["content"]
    assert all("cache_control" not in message for message in second_timing)


def test_take_loop_timing_context_text_claims_once():
    agent = SimpleNamespace(_loop_timing_context_text="a")
    assert conversation_loop._take_loop_timing_context_text(agent) == "a"
    assert "_loop_timing_context_text" not in vars(agent)
    assert conversation_loop._take_loop_timing_context_text(agent) == ""


def test_take_loop_timing_context_text_preserves_late_write():
    agent = SimpleNamespace(_loop_timing_context_text="a")
    assert conversation_loop._take_loop_timing_context_text(agent) == "a"
    agent._loop_timing_context_text = "b"
    assert conversation_loop._take_loop_timing_context_text(agent) == "b"
