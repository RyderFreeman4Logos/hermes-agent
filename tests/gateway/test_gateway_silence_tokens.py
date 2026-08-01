"""Gateway intentional-silence token behavior."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from gateway.response_filters import (
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _event():
    return MessageEvent(
        text="side chatter",
        source=_source(),
        message_id="msg-42",
    )


def _runner(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-silent",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_blank_and_prose_mentions_are_not_silence():
    assert not is_intentional_silence_response("")
    assert not is_intentional_silence_response("Use NO_REPLY when no answer is needed.")
    assert not is_intentional_silence_response("The reply was [SILENT], intentionally.")


@pytest.mark.asyncio
async def test_heartbeat_silent_noop_skips_warning_and_transcript_fallback(
    monkeypatch, tmp_path
):
    runner = _runner(monkeypatch, tmp_path)
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]
    runner.session_store.load_transcript.return_value = history
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "",
            "messages": list(history),
            "history_offset": len(history),
            "api_calls": 1,
            "completed": True,
            "silent_noop": True,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="[HEARTBEAT] target remains ALIVE",
        source=_source(),
        internal=True,
        metadata={
            "turn_origin": "heartbeat_warm",
            "allow_silent_noop": True,
        },
    )

    response = await runner._handle_message_with_agent(
        event,
        _source(),
        "agent:main:telegram:group:-1001:12345",
        1,
    )

    assert response == ""
    runner.session_store.append_to_transcript.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["STUCK", "UNKNOWN"])
async def test_unhealthy_heartbeat_does_not_opt_into_silent_noop(
    monkeypatch, tmp_path, status
):
    runner = _runner(monkeypatch, tmp_path)
    caller_id = "agent:main:telegram:group:-1001:12345"
    runner.session_store._entries = {caller_id: object()}
    runner._inject_watch_notification = AsyncMock(return_value=True)

    await runner._handle_heartbeat_event(
        {
            "type": "heartbeat",
            "target_id": "proc-heartbeat",
            "session_key": caller_id,
            "status": status,
            "evidence": "not healthy",
        }
    )

    assert runner._inject_watch_notification.call_args.kwargs == {
        "turn_origin": "heartbeat_warm",
        "allow_silent_noop": False,
    }


@pytest.mark.asyncio
async def test_heartbeat_early_error_skips_transcript_fallback(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]
    runner.session_store.load_transcript.return_value = history
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "provider unavailable",
            "messages": list(history),
            "history_offset": len(history),
            "api_calls": 1,
            "completed": False,
            "failed": True,
            "error": "provider unavailable",
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="[HEARTBEAT] target remains ALIVE",
        source=_source(),
        internal=True,
        metadata={
            "turn_origin": "heartbeat_warm",
            "allow_silent_noop": True,
        },
    )

    await runner._handle_message_with_agent(
        event,
        _source(),
        "agent:main:telegram:group:-1001:12345",
        1,
    )

    runner.session_store.append_to_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_unhealthy_heartbeat_is_visible_without_starting_auto_title(
    monkeypatch, tmp_path
):
    runner = _runner(monkeypatch, tmp_path)
    runner._should_send_voice_reply = (
        gateway_run.GatewayRunner._should_send_voice_reply.__get__(
            runner, gateway_run.GatewayRunner
        )
    )
    runner._send_voice_reply = AsyncMock()
    runner._voice_mode[runner._voice_key(Platform.TELEGRAM, "-1001")] = "all"
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]
    runner.session_store.load_transcript.return_value = history
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Target is STUCK: no CPU or output progress.",
            "messages": list(history),
            "history_offset": len(history),
            "api_calls": 1,
            "completed": True,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="[HEARTBEAT] target remains STUCK",
        source=_source(),
        internal=True,
        metadata={
            "turn_origin": "heartbeat_warm",
            "allow_silent_noop": False,
        },
    )

    with patch("agent.title_generator.maybe_auto_title") as auto_title:
        response = await runner._handle_message_with_agent(
            event,
            _source(),
            "agent:main:telegram:group:-1001:12345",
            1,
        )

    assert response == "Target is STUCK: no CPU or output progress."
    auto_title.assert_not_called()
    runner._send_voice_reply.assert_not_awaited()
