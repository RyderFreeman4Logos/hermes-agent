"""Gateway intentional-silence token behavior."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

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
    runner._set_session_env = MagicMock()
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
@pytest.mark.parametrize("status", ["STUCK", "UNKNOWN", "CHECKIN_FAILED"])
async def test_unhealthy_heartbeat_is_directly_visible_without_model_turn(
    monkeypatch, tmp_path, status
):
    runner = _runner(monkeypatch, tmp_path)
    caller_id = "agent:main:telegram:group:-1001:12345"
    runner.session_store._entries = {caller_id: object()}
    runner._inject_watch_notification = AsyncMock(return_value=True)
    runner._deliver_platform_notice = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        lambda *_args, **_kwargs: True,
    )

    await runner._handle_heartbeat_event(
        {
            "type": "heartbeat",
            "target_id": "proc-heartbeat",
            "session_key": caller_id,
            "status": status,
            "evidence": "not healthy",
        }
    )

    runner._inject_watch_notification.assert_not_awaited()
    runner._deliver_platform_notice.assert_awaited_once()
    assert status in runner._deliver_platform_notice.await_args_list[0].args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["STUCK", "UNKNOWN", "CHECKIN_FAILED"])
async def test_raw_api_heartbeat_is_directly_visible_without_model_turn(
    monkeypatch, tmp_path, status
):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    caller_id = "raw-api-session"
    runner.session_store._entries = {caller_id: object()}
    runner._inject_watch_notification = AsyncMock(return_value=True)
    runner._deliver_platform_notice = AsyncMock(return_value=True)
    is_event_current = MagicMock(return_value=True)
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        is_event_current,
    )
    cache_context = (
        "custom:pm|https://user:password@pm.invalid/v1"
        "?api_key=secret#fragment|model-a|chat_completions"
    )
    provider_identity = "custom:https://user:password@pm.invalid?api_key=secret#fragment"

    await runner._handle_heartbeat_event(
        {
            "type": "heartbeat",
            "target_id": "proc-heartbeat",
            "session_key": caller_id,
            "status": status,
            "evidence": "not healthy",
            "generation": 17,
            "target_kind": "process",
            "provider": provider_identity,
            "cache_context": cache_context,
            "heartbeat_warm_reason": "provider_error:APIStatusError",
            "heartbeat_group_token": 23,
        }
    )

    runner._inject_watch_notification.assert_not_awaited()
    runner._deliver_platform_notice.assert_not_awaited()
    from gateway.status import read_runtime_status

    assert is_event_current.call_count == 2
    persisted_text = (tmp_path / "gateway_state.json").read_text(encoding="utf-8")
    runtime_status = read_runtime_status()
    assert runtime_status is not None
    persisted_notice = runtime_status["runtime_notices"][-1]

    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "fixture-gateway-key"})
    )
    request = MagicMock()
    request.headers = {"Authorization": "Bearer fixture-gateway-key"}
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: "test/model")
    response = await adapter._handle_health_detailed(request)
    assert response.status == 200
    api_text = response.text
    assert api_text is not None
    api_notice = json.loads(api_text)["runtime_notices"][-1]

    for serialized in (persisted_text, api_text):
        leaked = any(
            marker in serialized
            for marker in (
                cache_context,
                provider_identity,
                "cache_context",
                '"provider":',
                "user:password@",
                "api_key=secret",
                "#fragment",
            )
        )
        assert not leaked, "credential-shaped cache identity reached an operational notice"
    for notice in (persisted_notice, api_notice):
        assert notice["type"] == "runtime_heartbeat"
        assert notice["status"] == status
        assert notice["session_key"] == caller_id
        assert notice["target_id"] == "proc-heartbeat"
        assert notice["evidence"] == "not healthy"
        assert notice["reason"] == "provider_error:APIStatusError"
        assert notice["generation"] == 17
        assert notice["target_kind"] == "process"
        assert "provider" not in notice
        assert notice["heartbeat_group_token"] == 23
