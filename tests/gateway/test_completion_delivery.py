"""Gateway delivery regressions for terminal completions.

Claims remain pending through adapter scheduling and settle only after a real
model turn acknowledges them. Async delegation uses SQLite; ordinary process
notifications use the process registry's bounded delivery ledger.
"""

import asyncio
import json
import logging
import queue
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
    merge_pending_message_event,
)
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Any current/future durable compatibility path must stay in tmp state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _runner(adapter, *, origins=None):
    if origins is None:
        origins = {
            "agent:main:telegram:dm:12345:678": SimpleNamespace(
                origin=SessionSource(
                    platform=Platform.TELEGRAM,
                    chat_id="12345",
                    chat_type="dm",
                    user_id="678",
                )
            )
        }
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries=origins,
    )
    runner._session_source_cache = {}
    return runner


def _async_event(delegation_id="deleg_duplicate"):
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "agent:main:telegram:dm:12345:678",
        "goal": "Investigate flaky test",
        "status": "completed",
        "summary": "Found it",
        "api_calls": 1,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
        # PR #62479 stamps these on gateway-owned events. They must not
        # change the producer identity used for queue replay.
        "origin_profile": "default",
        "origin_hermes_home": "/tmp/hermes-default",
    }


def _completion_event(*, started_at, session_id="proc_reused"):
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "started_at": started_at,
        "command": "echo done",
        "exit_code": 0,
        "completion_reason": "exited",
        "output": "done\n",
    }


def _heartbeat_event(**overrides):
    event = {
        "type": "heartbeat",
        "target_id": "proc-heartbeat",
        "target_kind": "process",
        "session_id": "proc-heartbeat",
        "session_key": "agent:main:telegram:dm:12345:678",
        "status": "ALIVE",
        "evidence": "output grew 0->128 bytes",
    }
    event.update(overrides)
    return event


def _message_runner(monkeypatch, tmp_path):
    """Minimal real message-delivery path for response-return assertions."""
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda source: True
    runner._set_session_env = lambda context: []
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _generation: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:12345:678",
        session_id="sess-heartbeat",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
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


def test_gateway_heartbeat_event_injects_a_silent_warm_turn(monkeypatch, isolated_registry):
    """The idle gateway watcher consumes heartbeats instead of leaving them queued."""
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_heartbeat_event())
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    from tools.runtime_heartbeat import runtime_heartbeat

    monkeypatch.setattr(
        runtime_heartbeat,
        "snapshot_active_targets",
        lambda caller_id=None: [{"target_id": "proc-heartbeat", "elapsed_s": 17}],
    )

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()
    synth_event = adapter.handle_message.await_args.args[0]
    assert "checkin #1" in synth_event.text
    assert "Elapsed: 17s" in synth_event.text
    assert synth_event.metadata["turn_origin"] == "heartbeat_warm"
    assert synth_event.metadata["allow_silent_noop"] is True
    assert isolated.empty()


@pytest.mark.asyncio
async def test_gateway_heartbeat_silent_noop_returns_no_platform_response(
    monkeypatch, tmp_path
):
    """A warm no-op must reach the adapter as no response, never fallback text."""
    runner = _message_runner(monkeypatch, tmp_path)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="678",
    )
    event = MessageEvent(
        text='[HEARTBEAT] checkin #1. Background target "proc-heartbeat" is ALIVE.',
        source=source,
        message_id="heartbeat-noop",
        internal=True,
        metadata={
            "turn_origin": "heartbeat_warm",
            "allow_silent_noop": True,
        },
    )

    class _HeartbeatCaptureAdapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(
                PlatformConfig(
                    enabled=True, token="fake-token", typing_indicator=False
                ),
                Platform.TELEGRAM,
            )
            self.sent = []

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent.append({"chat_id": chat_id, "content": content})
            return SendResult(success=True, message_id="heartbeat-message")

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    adapter = _HeartbeatCaptureAdapter()
    append_to_transcript = MagicMock()
    runner.session_store.append_to_transcript = append_to_transcript
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": None,
            "silent_noop": True,
            "messages": [{"role": "user", "content": event.text}],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "api_calls": 1,
            "failed": False,
        }
    )

    async def _handle_heartbeat_event(adapter_event):
        return await runner._handle_message_with_agent(
            adapter_event, source, "agent:main:telegram:dm:12345:678", 1
        )

    adapter._message_handler = _handle_heartbeat_event
    await adapter._process_message_background(event, "agent:main:telegram:dm:12345:678")

    assert adapter.sent == []
    append_to_transcript.assert_not_called()


def test_gateway_alive_heartbeat_skips_a_running_session(monkeypatch, caplog):
    """A keepalive never enters the gateway's busy-message interrupt path."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._running_agents = {}
    running_agent = object()
    runner._running_agents[_heartbeat_event()["session_key"]] = running_agent

    from tools.runtime_heartbeat import runtime_heartbeat

    monkeypatch.setattr(
        runtime_heartbeat,
        "snapshot_active_targets",
        lambda caller_id=None: [{"target_id": "proc-heartbeat", "elapsed_s": 17}],
    )
    caplog.set_level(logging.INFO, logger="gateway.run")

    asyncio.run(runner._handle_heartbeat_event(_heartbeat_event()))

    adapter.handle_message.assert_not_awaited()
    assert any(
        record.levelno == logging.INFO
        and "ALIVE" in record.getMessage()
        and _heartbeat_event()["session_key"] in record.getMessage()
        for record in caplog.records
    )


def test_gateway_stuck_heartbeat_steers_a_running_session(monkeypatch):
    """A busy gateway agent receives a STUCK alert without a competing turn."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._running_agents = {}
    steers = []

    class _RunningAgent:
        def steer(self, text):
            steers.append(text)
            return True

    runner._running_agents[_heartbeat_event()["session_key"]] = _RunningAgent()

    from tools.runtime_heartbeat import runtime_heartbeat

    monkeypatch.setattr(
        runtime_heartbeat,
        "snapshot_active_targets",
        lambda caller_id=None: [{"target_id": "proc-heartbeat", "elapsed_s": 17}],
    )

    asyncio.run(runner._handle_heartbeat_event(_heartbeat_event(status="STUCK")))

    assert len(steers) == 1
    assert "stuck" in steers[0].lower()
    assert "intervene" in steers[0].lower()
    adapter.handle_message.assert_not_awaited()


def test_gateway_stuck_idle_heartbeat_disables_silent_noop(monkeypatch):
    """Only ALIVE warm probes may use the silent-noop contract."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._running_agents = {}

    from tools.runtime_heartbeat import runtime_heartbeat

    monkeypatch.setattr(
        runtime_heartbeat,
        "snapshot_active_targets",
        lambda caller_id=None: [{"target_id": "proc-heartbeat", "elapsed_s": 17}],
    )

    asyncio.run(runner._handle_heartbeat_event(_heartbeat_event(status="STUCK")))

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].metadata["allow_silent_noop"] is False


def test_gateway_foreign_heartbeat_is_not_injected(monkeypatch):
    """A heartbeat without a gateway-owned session is dropped before routing."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={})
    runner._running_agents = {}

    from tools.runtime_heartbeat import runtime_heartbeat

    monkeypatch.setattr(
        runtime_heartbeat,
        "snapshot_active_targets",
        lambda caller_id=None: [{"target_id": "proc-heartbeat", "elapsed_s": 17}],
    )

    asyncio.run(runner._handle_heartbeat_event(_heartbeat_event()))

    adapter.handle_message.assert_not_awaited()


def test_gateway_post_turn_drain_requeues_heartbeat_for_warm_watcher():
    """The post-turn watch drain cannot silently discard a pending heartbeat."""
    from gateway.run import _drain_gateway_watch_events

    isolated = queue.Queue()
    event = _heartbeat_event()
    isolated.put(event)

    assert _drain_gateway_watch_events(isolated) == []
    assert isolated.get_nowait() is event


def _stop_after_sleeps(monkeypatch, runner, count):
    sleep_calls = 0

    async def _bounded_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= count:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _bounded_sleep)


def test_duplicate_async_queue_replay_injects_once(monkeypatch, isolated_registry):
    """Byte-identical queue replays produce one turn in one gateway lifecycle."""
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(dict(_async_event()))
    isolated.put(dict(_async_event()))

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()


def test_unroutable_async_event_is_not_requeued_forever(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    event = _async_event("deleg_desktop_or_cli")
    event["session_key"] = "20260711_unparseable_ui_session"
    isolated.put(event)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_not_awaited()
    assert isolated.empty()


def test_concurrent_claims_share_the_same_narrow_delivery_seam():
    """Concurrent consumers in one runner cannot both enter the adapter."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_injection(_event):
        entered.set()
        await release.wait()

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_blocked_injection))
    runner = _runner(adapter)
    event = _async_event()
    text = "completion"

    async def _exercise():
        first = asyncio.create_task(runner._deliver_completion_notification(text, dict(event)))
        await entered.wait()
        second = asyncio.create_task(runner._deliver_completion_notification(text, dict(event)))
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(first, second)

    assert sorted(asyncio.run(_exercise()), key=str) == [None, True]
    adapter.handle_message.assert_awaited_once()


def test_long_gateway_delivery_renews_claim_until_turn_owns_it(monkeypatch):
    from tools import async_delegation as ad

    event = _async_event("deleg_long_delivery")
    _persist_pending_completion(event)
    entered = asyncio.Event()
    finish = asyncio.Event()
    renewals = []
    original_renew = ad.renew_completion_delivery

    async def held_injection(_event):
        entered.set()
        await finish.wait()

    def observed_renew(delegation_id, claim):
        renewed = original_renew(delegation_id, claim)
        renewals.append((delegation_id, claim, renewed))
        return renewed

    monkeypatch.setattr(ad, "_DELIVERY_CLAIM_LEASE_SECONDS", 0.2)
    monkeypatch.setattr(ad, "_DELIVERY_CLAIM_RENEW_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(ad, "renew_completion_delivery", observed_renew)
    runner = _runner(SimpleNamespace(handle_message=held_injection))

    async def exercise():
        task = asyncio.create_task(
            runner._deliver_completion_notification("completion", event)
        )
        await entered.wait()
        await asyncio.sleep(0.25)
        assert len(renewals) >= 2
        assert renewals[-1][2] is True
        assert ad.claim_event_delivery(event, "foreign") is None
        finish.set()
        assert await task is True

    asyncio.run(exercise())


def test_failed_long_gateway_delivery_stops_renewal_and_releases_claim(
    monkeypatch,
):
    from tools import async_delegation as ad

    event = _async_event("deleg_long_delivery_timeout")
    _persist_pending_completion(event)
    renewal_attempts = []

    def failed_renew(_evt, _claim):
        renewal_attempts.append(1)
        raise OSError("renew store unavailable")

    async def timeout(_event):
        await asyncio.sleep(0.03)
        raise asyncio.TimeoutError()

    monkeypatch.setattr(ad, "_DELIVERY_CLAIM_RENEW_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(ad, "renew_event_delivery", failed_renew)
    runner = _runner(SimpleNamespace(handle_message=timeout))

    assert asyncio.run(
        runner._deliver_completion_notification("completion", event)
    ) is False
    attempts_after_return = len(renewal_attempts)
    asyncio.run(asyncio.sleep(0.03))

    assert len(renewal_attempts) == attempts_after_return
    retry_claim = ad.claim_event_delivery(event, "retry")
    assert retry_claim
    ad.release_event_delivery(event, retry_claim)


def test_failed_async_injection_is_retried_without_schedule_time_ack(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_async_event())

    adapter = SimpleNamespace(
        handle_message=AsyncMock(side_effect=[RuntimeError("temporary"), None])
    )
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=3)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert adapter.handle_message.await_count == 2


def _persist_pending_completion(event):
    from tools import async_delegation

    async_delegation._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": event["session_key"],
        "origin_ui_session_id": "",
        "parent_session_id": event.get("parent_session_id"),
        "dispatched_at": event["dispatched_at"],
    })
    async_delegation._persist_completion(event, {
        "status": "completed",
        "summary": event["summary"],
    })


def test_gateway_schedule_only_transfers_claim_without_acknowledging():
    from tools import async_delegation as ad

    event = _async_event("deleg_schedule_only")
    _persist_pending_completion(event)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    assert asyncio.run(
        runner._deliver_completion_notification("completion", event)
    ) is True

    synth_event = adapter.handle_message.await_args.args[0]
    claims = synth_event.metadata["delivery_claims"]
    assert claims[0][0] is event
    durable = ad.get_durable_delegation(event["delegation_id"])
    assert durable["delivery_state"] == "pending"
    assert durable["delivery_attempts"] == 1


def test_ordinary_process_schedule_only_stays_claimed_until_real_turn():
    event = _completion_event(
        started_at=2000.0,
        session_id="proc-schedule-only",
    )
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    assert asyncio.run(
        runner._deliver_completion_notification("completion", event)
    ) is True
    assert asyncio.run(
        runner._deliver_completion_notification("completion", dict(event))
    ) is None

    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.metadata["delivery_claims"][0][0] is event
    adapter.handle_message.assert_awaited_once()


def test_ordinary_process_claim_completes_only_after_real_gateway_turn(
    isolated_registry,
):
    event = _completion_event(
        started_at=2001.0,
        session_id="proc-real-turn",
    )
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    assert asyncio.run(
        runner._deliver_completion_notification("completion", event)
    ) is True
    synth_event = adapter.handle_message.await_args.args[0]
    claims = synth_event.metadata["delivery_claims"]
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._run_agent_inner = AsyncMock(
        return_value={"api_calls": 1, "final_response": "handled"}
    )

    asyncio.run(
        runner._run_agent(
            "completion",
            "",
            [],
            synth_event.source,
            "session-1",
            delivery_claims=claims,
        )
    )

    assert not isolated_registry.claim_notification_delivery(event, "replay")


def test_busy_gateway_merge_preserves_transferred_claim():
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"
    )
    pending_event = MessageEvent(text="first", source=source)
    claimed_event = _async_event("deleg_busy_merge")
    incoming = MessageEvent(
        text="completion",
        source=source,
        internal=True,
        metadata={"delivery_claims": [(claimed_event, "claim-1")]},
    )
    pending = {"session": pending_event}

    merge_pending_message_event(
        pending, "session", incoming, merge_text=True
    )

    assert pending_event.metadata["delivery_claims"] == [
        (claimed_event, "claim-1")
    ]


@pytest.mark.asyncio
async def test_gateway_adapter_cancellation_releases_untransferred_claim():
    from tools import async_delegation as ad

    event_data = _async_event("deleg_adapter_cancel")
    _persist_pending_completion(event_data)
    claim = ad.claim_event_delivery(event_data, "gateway-scheduled")
    assert claim

    class _Adapter(BasePlatformAdapter):
        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True)

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    adapter = _Adapter(
        PlatformConfig(
            enabled=True, token="fake-token", typing_indicator=False
        ),
        Platform.TELEGRAM,
    )
    adapter._message_handler = AsyncMock(side_effect=asyncio.CancelledError())
    event = MessageEvent(
        text="completion",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="678",
        ),
        internal=True,
        metadata={"delivery_claims": [(event_data, claim)]},
    )
    session_key = event_data["session_key"]
    adapter._active_sessions[session_key] = asyncio.Event()

    with pytest.raises(asyncio.CancelledError):
        await adapter._process_message_background(event, session_key)

    durable = ad.get_durable_delegation(event_data["delegation_id"])
    assert durable["delivery_state"] == "pending"
    assert durable["delivery_attempts"] == 1


@pytest.mark.parametrize("exc", [RuntimeError("failed"), asyncio.CancelledError()])
def test_gateway_transferred_claim_released_when_turn_raises(exc):
    from tools import async_delegation as ad

    event = _async_event(f"deleg_{type(exc).__name__}")
    _persist_pending_completion(event)
    claim = ad.claim_event_delivery(event, "gateway-turn")
    assert claim

    runner = _runner(SimpleNamespace(handle_message=AsyncMock()))
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._run_agent_inner = AsyncMock(side_effect=exc)
    source = runner.session_store._entries[event["session_key"]].origin

    async def run():
        return await runner._run_agent(
            "completion",
            "",
            [],
            source,
            "session-1",
            delivery_claims=[(event, claim)],
        )

    with pytest.raises(type(exc)):
        asyncio.run(run())

    durable = ad.get_durable_delegation(event["delegation_id"])
    assert durable["delivery_state"] == "pending"
    assert durable["delivery_attempts"] == 1


def test_explicit_kill_returns_output_before_consuming_notification(monkeypatch):
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_kill_consumed",
        command="sleep 999",
        task_id="task",
        started_at=1.0,
        output_buffer="important terminal output\n",
        notify_on_complete=True,
    )
    session.process = MagicMock()
    session.process.pid = 4242
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "_terminate_host_pid", lambda *_a, **_kw: None)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(pr_module, "process_registry", registry)

    result = registry.kill_process(session.id)
    assert result["status"] == "killed"
    assert result["output"] == "important terminal output\n"
    assert registry.is_completion_consumed(session.id)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": session.id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    adapter.handle_message.assert_not_awaited()


def test_process_tool_redacts_explicit_kill_output(monkeypatch):
    from tools import process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_kill_redacted",
        command="printenv",
        task_id="task",
        started_at=1.0,
        output_buffer="PRIVATE_TOKEN=opaque-value\n",
        exited=True,
        exit_code=0,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)

    def _redact(result):
        assert result["output"] == "PRIVATE_TOKEN=opaque-value\n"
        result["output"] = "PRIVATE_TOKEN=<redacted>\n"
        return result

    monkeypatch.setattr(pr_module, "_redact_process_result", _redact)

    result = json.loads(pr_module._handle_process({
        "action": "kill",
        "session_id": session.id,
    }))
    assert result["output"] == "PRIVATE_TOKEN=<redacted>\n"


def test_autonomous_completion_redacts_real_command_and_output_secrets(monkeypatch):
    import agent.redact as redact_module
    import tools.process_registry as pr_module

    secret = "abc123randomopaquetokenvalue999"
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_autonomous_redaction",
        command=f"printenv MY_SERVICE_TOKEN={secret}",
        task_id="task",
        started_at=1234.5,
        output_buffer=f"MY_SERVICE_TOKEN={secret}\nHOME=/home/user\n",
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)
    monkeypatch.setattr(redact_module, "_REDACT_ENABLED", True)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": session.id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    delivered = adapter.handle_message.await_args.args[0]
    assert secret not in delivered.text
    assert "HOME=/home/user" in delivered.text
