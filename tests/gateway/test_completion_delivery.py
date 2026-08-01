"""Lifecycle-scoped gateway delivery regressions for terminal completions.

The gateway contract here is deliberately narrower than exactly-once: one live
GatewayRunner suppresses concurrent/replayed copies after successful adapter
injection, failed injection remains retryable, and durable async-delegation
state (when available) is acknowledged through its authoritative SQLite API.
"""

import asyncio
import json
import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
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
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries=origins or {},
    )
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
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


def _runtime_heartbeat_event(**overrides):
    event = {
        "type": "heartbeat",
        "target_id": "proc-heartbeat",
        "target_ids": ["proc-heartbeat"],
        "generations": [11],
        "generation": 11,
        "target_kind": "process",
        "session_id": "proc-heartbeat",
        "session_key": "agent:main:telegram:dm:12345:678",
        "provider": "openai",
        "cache_context": "openai-cache",
        "status": "ALIVE",
        "evidence": "output grew",
    }
    event.update(overrides)
    return event


@pytest.fixture
def current_heartbeat(monkeypatch):
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        lambda _event: True,
    )


def test_gateway_heartbeat_routes_to_exact_idle_owner(monkeypatch, current_heartbeat):
    event = _runtime_heartbeat_event()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    runner._running_agents = {}
    inject = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_inject_watch_notification", inject)

    asyncio.run(runner._handle_heartbeat_event(event))

    inject.assert_awaited_once()
    assert inject.await_args.kwargs == {
        "turn_origin": "heartbeat_warm",
        "allow_silent_noop": True,
        "heartbeat_event": event,
    }


def test_gateway_push_heartbeat_preserves_typed_event_metadata():
    event = _runtime_heartbeat_event()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})

    delivered = asyncio.run(
        runner._inject_watch_notification(
            "[HEARTBEAT] target remains ALIVE",
            event,
            turn_origin="heartbeat_warm",
            allow_silent_noop=True,
        )
    )

    assert delivered is True
    message_event = adapter.handle_message.await_args.args[0]
    assert message_event.internal is True
    assert message_event.metadata["turn_origin"] == "heartbeat_warm"
    assert message_event.metadata["allow_silent_noop"] is True
    assert message_event.metadata["heartbeat_event"] == event


def test_gateway_raw_api_heartbeat_never_self_posts_as_user_traffic(monkeypatch):
    adapter = SimpleNamespace(handle_message=AsyncMock(), supports_push=False)
    runner = _runner(adapter)
    runner.adapters = {Platform.API_SERVER: adapter}
    event = _runtime_heartbeat_event(session_key="opaque-api-session")
    wake = AsyncMock()
    monkeypatch.setattr("gateway.wake.deliver_wake", wake)

    delivered = asyncio.run(
        runner._inject_watch_notification(
            "[HEARTBEAT] target remains ALIVE",
            event,
            turn_origin="heartbeat_warm",
            allow_silent_noop=True,
        )
    )

    assert delivered is None
    wake.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


def test_gateway_alive_heartbeat_does_not_duplicate_busy_turn(
    monkeypatch, current_heartbeat
):
    event = _runtime_heartbeat_event()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    runner._running_agents = {event["session_key"]: object()}
    inject = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_inject_watch_notification", inject)

    asyncio.run(runner._handle_heartbeat_event(event))

    inject.assert_not_awaited()


@pytest.mark.parametrize("status", ["STUCK", "UNKNOWN"])
def test_gateway_unhealthy_heartbeat_is_directly_visible_without_model_turn(
    monkeypatch, current_heartbeat, status
):
    event = _runtime_heartbeat_event(status=status, evidence="no progress")
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    runner._running_agents = {event["session_key"]: object()}
    inject = AsyncMock()
    deliver = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_inject_watch_notification", inject)
    monkeypatch.setattr(runner, "_deliver_platform_notice", deliver)

    asyncio.run(runner._handle_heartbeat_event(event))

    inject.assert_not_awaited()
    deliver.assert_awaited_once()
    assert status in deliver.await_args.args[1]
    assert "no progress" in deliver.await_args.args[1]


def test_gateway_foreign_heartbeat_never_crosses_owner(
    monkeypatch, current_heartbeat
):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={})
    runner._running_agents = {}
    inject = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_inject_watch_notification", inject)

    asyncio.run(runner._handle_heartbeat_event(_runtime_heartbeat_event()))

    inject.assert_not_awaited()


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


def test_numeric_completion_gets_nudge_and_none_gets_no_turn(monkeypatch):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    injected = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_inject_watch_notification", injected)

    event = _completion_event(started_at=1.0)
    assert asyncio.run(runner._deliver_completion_notification("payload", event)) is True
    prompt = injected.await_args.args[0]
    assert prompt.startswith("payload")
    assert "If no user-visible action is needed, emit no response." in prompt

    none_event = _completion_event(started_at=2.0)
    none_event["exit_code"] = None
    assert asyncio.run(
        runner._deliver_completion_notification("not complete", none_event)
    ) is None
    assert injected.await_count == 1


def test_failed_async_injection_is_retried_and_only_success_is_acked(
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

    from tools import async_delegation

    acknowledgements = []
    monkeypatch.setattr(
        async_delegation,
        "complete_completion_delivery",
        lambda delegation_id, _claim_id: acknowledgements.append(delegation_id) or True,
        raising=False,
    )

    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert adapter.handle_message.await_count == 2
    assert acknowledgements == ["deleg_duplicate"]


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
    session.process.poll.return_value = -15
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
