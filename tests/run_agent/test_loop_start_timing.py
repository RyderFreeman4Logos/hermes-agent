"""Append-only loop timing on real turn inputs.

The timing block belongs to the inbound user row.  It is persisted once before
the first provider request, then reused byte-for-byte for retries and tool-loop
continuations.  Only a successfully completed loop advances the small durable
stop scalar used by the next loop.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent


TZ = timezone(timedelta(hours=-7))
MARKER = "[Agent loop timing]"


def _response(content="done", *, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason="tool_calls" if tool_calls else "stop",
        )],
        model="test/model",
        usage=None,
    )


def _tool_call():
    return SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="test_tool", arguments="{}"),
    )


def _make_agent(tmp_path, sid: str, *, platform: str = "cli"):
    db = SessionDB(db_path=tmp_path / f"{sid}.db")
    db.create_session(sid, source=platform, model="test/model")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform=platform,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
            session_db=db,
            session_id=sid,
        )
    agent._session_db_created = True
    agent._cached_system_prompt = "SYSTEM"
    agent._skip_mcp_refresh = True
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent._fallback_chain = []
    agent.client = MagicMock()
    return agent, db


def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


@pytest.mark.parametrize(
    ("platform", "user_message", "persist_message", "display_kind"),
    [
        ("cli", "ordinary user input", None, None),
        (
            "discord",
            "[Background process completed]\nraw provider-facing notice",
            "background completion",
            "internal_notification",
        ),
        (
            "subagent",
            [
                {"type": "text", "text": "delegated task"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
            None,
            None,
        ),
    ],
)
def test_external_loop_start_is_decorated_once_before_persistence(
    tmp_path, monkeypatch, platform, user_message, persist_message, display_kind,
):
    """Ordinary, internal-completion, and child-loop starts share one ingress."""
    from agent import loop_timing

    original_user = deepcopy(user_message)
    agent, db = _make_agent(tmp_path, f"wake-{platform}", platform=platform)
    agent.client.chat.completions.create.return_value = _response()
    if isinstance(user_message, list):
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
    monkeypatch.setattr(
        loop_timing,
        "_now",
        lambda: datetime(2026, 9, 15, 10, 0, 0, tzinfo=TZ),
    )
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *_a, **_k: None)

    try:
        result = agent.run_conversation(
            user_message,
            persist_user_message=persist_message,
            persist_user_display_kind=display_kind,
        )

        sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
        sent_user = next(row for row in reversed(sent) if row.get("role") == "user")
        durable = db.get_messages_as_conversation(agent.session_id)
        durable_user = next(row for row in reversed(durable) if row.get("role") == "user")
        assert _text(sent_user["content"]).count(MARKER) == 1
        assert _text(durable_user["content"]).count(MARKER) == 0
        assert durable_user["api_content"] == sent_user["content"]
        assert not any(row.get("role") == "system" and MARKER in _text(row.get("content")) for row in durable)
        assert result["completed"] is True

        if isinstance(user_message, list):
            assert user_message == original_user
            assert sent_user["content"][:-1] == original_user
            assert durable_user["content"] == "delegated task\n[screenshot]"

            # A cold restore substitutes the structured sidecar back onto the wire.
            from agent.turn_context import build_api_messages

            replay, _system = build_api_messages(
                agent,
                durable,
                current_turn_user_idx=None,
                ext_prefetch_cache="",
                plugin_user_context="",
                moa_config=None,
                active_system_prompt="",
            )
            replay_user = next(row for row in reversed(replay) if row.get("role") == "user")
            assert replay_user["content"] == sent_user["content"]
        elif persist_message is not None:
            assert durable_user["content"] == persist_message
            assert durable_user["display_kind"] == display_kind
        else:
            assert durable_user["content"] == original_user
    finally:
        db.close()


def test_retry_and_tool_continuation_reuse_the_same_decorated_bytes(tmp_path, monkeypatch):
    """A provider retry and a synchronous tool continuation do not make new starts."""
    from agent import loop_timing

    agent, db = _make_agent(tmp_path, "continuation")
    transient = Exception("synthetic retryable timeout")
    transient.status_code = 408
    agent.client.chat.completions.create.side_effect = [
        transient,
        _response(content=None, tool_calls=[_tool_call()]),
        _response(),
    ]
    agent.valid_tool_names = {"test_tool"}
    times = iter([
        datetime(2026, 9, 15, 11, 0, 0, tzinfo=TZ),
        datetime(2026, 9, 15, 11, 0, 5, tzinfo=TZ),
    ])
    monkeypatch.setattr(loop_timing, "_now", lambda: next(times))
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *_a, **_k: None)

    def execute_tool(_assistant, messages, *_args):
        messages.append({
            "role": "tool",
            "name": "test_tool",
            "tool_call_id": "call-1",
            "content": "ok",
        })

    try:
        with patch.object(agent, "_execute_tool_calls", side_effect=execute_tool):
            result = agent.run_conversation("do work")
        requests = agent.client.chat.completions.create.call_args_list
        assert len(requests) == 3
        decorated = [
            next(row for row in call.kwargs["messages"] if row.get("role") == "user")["content"]
            for call in requests
        ]
        assert decorated[0] == decorated[1] == decorated[2]
        assert decorated[0].count(MARKER) == 1
        assert result["completed"] is True
    finally:
        db.close()


def test_next_turn_replays_the_prior_decorated_prefix_byte_for_byte(tmp_path, monkeypatch):
    """A later loop keeps the old start bytes and adds one new timing block."""
    from agent import loop_timing

    agent, db = _make_agent(tmp_path, "two-turn-prefix")
    agent.client.chat.completions.create.side_effect = [_response("one"), _response("two")]
    times = iter([
        datetime(2026, 9, 15, 11, 30, 0, tzinfo=TZ),
        datetime(2026, 9, 15, 11, 30, 2, tzinfo=TZ),
        datetime(2026, 9, 15, 11, 35, 0, tzinfo=TZ),
        datetime(2026, 9, 15, 11, 35, 3, tzinfo=TZ),
    ])
    monkeypatch.setattr(loop_timing, "_now", lambda: next(times))
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *_a, **_k: None)

    try:
        assert agent.run_conversation("first")["completed"] is True
        first_messages = deepcopy(
            agent.client.chat.completions.create.call_args_list[0].kwargs["messages"]
        )
        restored = db.get_messages_as_conversation(agent.session_id)
        assert agent.run_conversation("second", conversation_history=restored)["completed"] is True
        second_messages = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]

        first_user = next(row for row in first_messages if row.get("role") == "user")
        replayed_first = next(row for row in second_messages if row.get("role") == "user")
        current_second = next(row for row in reversed(second_messages) if row.get("role") == "user")
        assert replayed_first["content"] == first_user["content"]
        assert replayed_first["content"].count(MARKER) == 1
        assert current_second["content"].count(MARKER) == 1
        assert "Latest successful completed loop stop: 2026-09-15T11:30:02-07:00" in current_second["content"]
    finally:
        db.close()


def test_disabled_loop_timing_keeps_public_and_durable_input_unchanged(tmp_path, monkeypatch):
    from agent import loop_timing

    agent, db = _make_agent(tmp_path, "disabled")
    agent.client.chat.completions.create.return_value = _response()
    monkeypatch.setattr(loop_timing, "_enabled", lambda: False)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *_a, **_k: None)

    try:
        assert agent.run_conversation("plain input")["completed"] is True
        sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
        sent_user = next(row for row in reversed(sent) if row.get("role") == "user")
        durable = db.get_messages_as_conversation(agent.session_id)
        durable_user = next(row for row in reversed(durable) if row.get("role") == "user")
        assert sent_user["content"] == "plain input"
        assert durable_user["content"] == "plain input"
        assert db.get_session_model_config_value(agent.session_id, loop_timing.LOOP_STOP_KEY) is None
    finally:
        db.close()


def test_only_successful_completion_advances_session_scoped_stop(tmp_path, monkeypatch):
    """A failed N cannot relabel the latest successful N-1 stop as N's end."""
    from agent import loop_timing

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("main", source="cli", model="test/model")
    db.create_session(
        "branch",
        source="cli",
        model="test/model",
        model_config={"_branched_from": "main"},
        parent_session_id="main",
    )
    db.create_session("new", source="cli", model="test/model")
    agent = SimpleNamespace(
        _session_db=db,
        session_id="main",
        _loop_timing_start_decorated=True,
    )
    stop_n_minus_1 = datetime(2026, 9, 15, 12, 0, 3, tzinfo=TZ)
    failed_n = datetime(2026, 9, 15, 12, 5, 9, tzinfo=TZ)

    assert loop_timing.record_completed_loop_stop(
        agent,
        {"completed": True, "failed": False, "interrupted": False},
        now=stop_n_minus_1,
    )
    assert not loop_timing.record_completed_loop_stop(
        agent,
        {"completed": False, "failed": True, "interrupted": False},
        now=failed_n,
    )
    assert not loop_timing.record_completed_loop_stop(
        agent,
        {"completed": False, "failed": False, "interrupted": True},
        now=failed_n,
    )

    # A cold process for the same session sees N-1.  A branch/new session sees none.
    cold = SimpleNamespace(_session_db=SessionDB(db_path=db_path), session_id="main")
    branch = SimpleNamespace(_session_db=cold._session_db, session_id="branch")
    new_session = SimpleNamespace(_session_db=cold._session_db, session_id="new")
    start_n_plus_1 = datetime(2026, 9, 15, 12, 10, 0, tzinfo=TZ)
    try:
        current, persisted = loop_timing.decorate_loop_start_input(
            cold, "next", None, now=start_n_plus_1,
        )
        assert persisted == "next"
        assert "Latest successful completed loop stop: 2026-09-15T12:00:03-07:00" in current
        assert "12:05:09" not in current
        fresh, _ = loop_timing.decorate_loop_start_input(
            branch, "branch input", None, now=start_n_plus_1,
        )
        assert "Latest successful completed loop stop:" not in fresh
        new_input, _ = loop_timing.decorate_loop_start_input(
            new_session, "new session input", None, now=start_n_plus_1,
        )
        assert "Latest successful completed loop stop:" not in new_input

        # Rotation copies the scalar into the child model config; the parent is untouched.
        rotated = loop_timing.model_config_for_compression_child(
            cold, {"reasoning_effort": "high"},
        )
        assert rotated[loop_timing.LOOP_STOP_KEY] == "2026-09-15T12:00:03-07:00"
        assert db.get_session_model_config_value("main", loop_timing.LOOP_STOP_KEY) == rotated[loop_timing.LOOP_STOP_KEY]
    finally:
        cold._session_db.close()
        db.close()
