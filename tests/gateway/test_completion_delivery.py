"""Gateway terminal-fence regressions for completion delivery.

One live GatewayRunner suppresses concurrent/replayed copies, failed injection
remains retryable, and push adapters retain durable claims until the provider
turn and transcript publication finish.
"""

import asyncio
import json
import queue
import threading
import time
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
    isolated = AsyncMock()
    monkeypatch.setattr(runner, "_run_isolated_heartbeat", isolated)

    async def run_and_drain():
        await runner._handle_heartbeat_event(event)
        await runner._heartbeat_warm_tasks[event["session_key"]]

    asyncio.run(run_and_drain())

    isolated.assert_awaited_once()
    assert isolated.await_args.args == (event["session_key"], event)
    assert not runner._is_session_running(event["session_key"])


def test_gateway_does_not_duplicate_runtime_owned_warm(
    monkeypatch, current_heartbeat
):
    event = _runtime_heartbeat_event(heartbeat_warm_owned=True)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    runner._running_agents = {}
    isolated = AsyncMock()
    monkeypatch.setattr(runner, "_run_isolated_heartbeat", isolated)

    asyncio.run(runner._handle_heartbeat_event(event))

    isolated.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_isolated_heartbeat_never_reads_live_history(monkeypatch):
    event = _runtime_heartbeat_event()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    calls = []

    class _Agent:
        @property
        def _session_messages(self):
            raise AssertionError("heartbeat read live conversation history")

        def run_conversation(self, message, **kwargs):
            calls.append((message, kwargs))

    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {event["session_key"]: (_Agent(), 0)}

    async def run_inline(callback):
        callback()

    monkeypatch.setattr(runner, "_run_in_executor_with_context", run_inline)

    await runner._run_isolated_heartbeat(event["session_key"], event)

    assert calls == [("", {
        "turn_origin": "heartbeat_warm",
        "heartbeat_event": event,
    })]


def test_gateway_raw_api_heartbeat_never_runs_or_self_posts(
    monkeypatch, current_heartbeat
):
    adapter = SimpleNamespace(handle_message=AsyncMock(), supports_push=False)
    event = _runtime_heartbeat_event(session_key="opaque-api-session")
    runner = _runner(adapter, origins={event["session_key"]: object()})
    runner.adapters = {Platform.API_SERVER: adapter}
    runner._running_agents = {}
    isolated = AsyncMock()
    monkeypatch.setattr(runner, "_run_isolated_heartbeat", isolated)

    asyncio.run(runner._handle_heartbeat_event(event))

    isolated.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


def test_gateway_alive_heartbeat_warms_independently_of_busy_turn(
    monkeypatch, current_heartbeat
):
    event = _runtime_heartbeat_event()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    ordinary_owner = object()
    runner._running_agents = {event["session_key"]: ordinary_owner}
    state = runner._session_state(event["session_key"])
    state.turn.agent = ordinary_owner
    isolated = AsyncMock()
    monkeypatch.setattr(runner, "_run_isolated_heartbeat", isolated)

    async def run_and_drain():
        await runner._handle_heartbeat_event(event)
        await runner._heartbeat_warm_tasks[event["session_key"]]

    asyncio.run(run_and_drain())

    isolated.assert_awaited_once()
    assert state.turn.agent is ordinary_owner


@pytest.mark.asyncio
async def test_gateway_heartbeat_never_reserves_or_mutates_busy_slot(
    monkeypatch, current_heartbeat
):
    event = _runtime_heartbeat_event()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    ordinary_owner = object()
    runner._running_agents = {event["session_key"]: ordinary_owner}
    state = runner._session_state(event["session_key"])
    state.turn.agent = ordinary_owner
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(*_args):
        started.set()
        await release.wait()

    monkeypatch.setattr(runner, "_run_isolated_heartbeat", blocked)
    await runner._handle_heartbeat_event(event)
    task = runner._heartbeat_warm_tasks[event["session_key"]]
    await asyncio.wait_for(started.wait(), timeout=1)

    assert state.turn.agent is ordinary_owner
    assert state.turn.heartbeat_owner is None
    release.set()
    await asyncio.wait_for(task, timeout=1)

    assert state.turn.agent is ordinary_owner
    assert state.turn.heartbeat_owner is None


@pytest.mark.asyncio
async def test_gateway_blocked_warm_does_not_block_later_unhealthy_notice(
    monkeypatch, current_heartbeat
):
    alive = _runtime_heartbeat_event()
    stuck = _runtime_heartbeat_event(status="STUCK", evidence="no progress")
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={alive["session_key"]: object()})
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(*_args):
        started.set()
        await release.wait()

    deliver = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_run_isolated_heartbeat", blocked)
    monkeypatch.setattr(runner, "_deliver_platform_notice", deliver)

    handler = asyncio.create_task(runner._handle_heartbeat_event(alive))
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        # The watcher awaits each handler serially. Therefore the ALIVE
        # handler itself must detach its tracked warm before a later event can
        # be processed.
        await asyncio.wait_for(asyncio.shield(handler), timeout=0.05)
        await runner._handle_heartbeat_event(stuck)
    finally:
        release.set()
        await asyncio.wait_for(handler, timeout=1)
        warm_task = runner._heartbeat_warm_tasks.get(alive["session_key"])
        if warm_task is not None:
            await asyncio.wait_for(warm_task, timeout=1)

    deliver.assert_awaited_once()
    assert "STUCK" in deliver.await_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["stale", "exception"])
async def test_gateway_heartbeat_releases_its_reservation_on_early_exit(
    monkeypatch, exit_kind
):
    event = _runtime_heartbeat_event()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    runner._running_agents = {}
    checks = iter((True, exit_kind != "stale"))
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        lambda _event: next(checks),
    )

    async def fail(*_args):
        raise RuntimeError("heartbeat failed")

    isolated = AsyncMock(side_effect=fail if exit_kind == "exception" else None)
    monkeypatch.setattr(runner, "_run_isolated_heartbeat", isolated)

    await runner._handle_heartbeat_event(event)
    task = getattr(runner, "_heartbeat_warm_tasks", {}).get(
        event["session_key"]
    )
    if task is not None:
        if exit_kind == "exception":
            with pytest.raises(RuntimeError, match="heartbeat failed"):
                await task
        else:
            await task
    if exit_kind == "stale":
        isolated.assert_not_awaited()

    assert not runner._is_session_running(event["session_key"])


@pytest.mark.asyncio
async def test_gateway_heartbeat_schedule_is_single_flight_and_cancel_cleans_up(
    monkeypatch,
):
    event = _runtime_heartbeat_event()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    started = asyncio.Event()
    calls = 0

    async def blocked(*_args):
        nonlocal calls
        calls += 1
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "_run_isolated_heartbeat", blocked)

    first = runner._schedule_isolated_heartbeat(event["session_key"], event)
    second = runner._schedule_isolated_heartbeat(event["session_key"], event)
    assert first is second
    await asyncio.wait_for(started.wait(), timeout=1)
    assert calls == 1
    assert first in runner._background_tasks

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.sleep(0)

    assert event["session_key"] not in runner._heartbeat_warm_tasks
    assert first not in runner._background_tasks


@pytest.mark.parametrize("status", ["STUCK", "UNKNOWN"])
def test_gateway_unhealthy_heartbeat_is_visible_and_warms_only_when_live(
    monkeypatch, current_heartbeat, status
):
    event = _runtime_heartbeat_event(status=status, evidence="no progress")
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    runner._running_agents = {}
    inject = AsyncMock()
    deliver = AsyncMock(return_value=True)
    isolated = AsyncMock()
    monkeypatch.setattr(runner, "_inject_watch_notification", inject)
    monkeypatch.setattr(runner, "_deliver_platform_notice", deliver)
    monkeypatch.setattr(runner, "_run_isolated_heartbeat", isolated)

    asyncio.run(runner._handle_heartbeat_event(event))

    inject.assert_not_awaited()
    deliver.assert_awaited_once()
    assert status in deliver.await_args.args[1]
    assert "no progress" in deliver.await_args.args[1]
    isolated.assert_not_awaited()


def test_gateway_unhealthy_heartbeat_revalidates_at_notice_boundary(monkeypatch):
    event = _runtime_heartbeat_event(status="STUCK", evidence="no progress")
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter, origins={event["session_key"]: object()})
    deliver = AsyncMock()
    checks = iter((True, False))
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        lambda _event: next(checks),
    )
    monkeypatch.setattr(runner, "_deliver_platform_notice", deliver)

    asyncio.run(runner._handle_heartbeat_event(event))

    deliver.assert_not_awaited()


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


def test_numeric_completion_gets_nudge_and_unknown_fails_open(monkeypatch):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    injected = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_inject_watch_notification", injected)

    event = _completion_event(started_at=1.0)
    assert asyncio.run(runner._deliver_completion_notification("payload", event)) is True
    prompt = injected.await_args.args[0]
    assert prompt.startswith("payload")
    assert "final assistant message must be literally empty (zero characters)" in prompt

    none_event = _completion_event(started_at=2.0)
    none_event["exit_code"] = None
    assert asyncio.run(
        runner._deliver_completion_notification("not complete", none_event)
    ) is True
    assert injected.await_count == 2
    assert injected.await_args.args[0] == "not complete"


def test_gateway_push_keeps_claim_until_turn_commit(isolated_registry):
    """Push-adapter acceptance is not the provider/history terminal fence."""
    from tools import async_delegation as ad

    adapter = SimpleNamespace(handle_message=AsyncMock(), supports_push=True)
    runner = _runner(adapter)
    event = _completion_event(started_at=2.5, session_id="proc-push-fence")

    assert asyncio.run(
        runner._deliver_completion_notification("payload", event)
    ) is True

    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "effect_started"
    delivered_event = adapter.handle_message.await_args.args[0]
    assert delivered_event.metadata["_completion_delivery_receipt"]

    asyncio.run(
        runner._finish_completion_delivery_receipt(
            delivered_event, "committed"
        )
    )
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "delivered"
    assert not isolated_registry.completion_event_should_deliver(event)


def test_legacy_gateway_completion_keeps_terminal_fence(isolated_registry):
    """Legacy events without stable IDs still wait for the push turn to finish."""
    adapter = SimpleNamespace(handle_message=AsyncMock(), supports_push=True)
    runner = _runner(adapter)
    event = _completion_event(started_at=2.6, session_id="proc-legacy")
    event.pop("started_at")

    assert asyncio.run(
        runner._deliver_completion_notification("payload", event)
    ) is True

    delivered_event = adapter.handle_message.await_args.args[0]
    assert delivered_event.metadata["_completion_delivery_receipt"]
    asyncio.run(
        runner._finish_completion_delivery_receipt(
            delivered_event, "committed"
        )
    )


def test_owner_observed_success_skips_gateway_turn(monkeypatch, isolated_registry):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    injected = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_inject_watch_notification", injected)
    monkeypatch.setattr(
        "tools.approval.get_current_session_key",
        lambda default="": "agent:main:telegram:dm:123",
    )
    event = _completion_event(started_at=3.0)
    isolated_registry._record_completion_observed(ProcessSession(
        id=event["session_id"],
        command=event["command"],
        session_key=event["session_key"],
        started_at=event["started_at"],
        exit_code=event["exit_code"],
        completion_reason=event["completion_reason"],
        output_buffer=event["output"],
        notify_on_complete=True,
    ))

    assert asyncio.run(
        runner._deliver_completion_notification("payload", event)
    ) is None
    injected.assert_not_awaited()

    from tools import async_delegation as ad

    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "delivered"
    restarted_queue = queue.Queue()
    assert ad.restore_undelivered_completions(restarted_queue) == 0
    assert restarted_queue.empty()


def test_failed_process_injection_releases_lifecycle_claim(monkeypatch):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    injected = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr(runner, "_inject_watch_notification", injected)
    event = _completion_event(started_at=4.0)

    assert asyncio.run(
        runner._deliver_completion_notification("payload", event)
    ) is False
    assert asyncio.run(
        runner._deliver_completion_notification("payload", event)
    ) is True
    assert injected.await_count == 2


def test_failed_completion_prompt_releases_lifecycle_claim_for_retry(monkeypatch):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    injected = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_inject_watch_notification", injected)
    monkeypatch.setattr(
        "tools.process_registry.completion_delivery_prompt",
        MagicMock(side_effect=[RuntimeError("judge failed"), "payload"]),
    )
    event = _completion_event(started_at=5.0)

    with pytest.raises(RuntimeError, match="judge failed"):
        asyncio.run(runner._deliver_completion_notification("payload", event))
    assert asyncio.run(
        runner._deliver_completion_notification("payload", event)
    ) is True
    injected.assert_awaited_once_with("payload", event)


def test_completion_judge_runs_off_loop_and_delivers_after_timeout(monkeypatch):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    injected = AsyncMock(return_value=True)
    monkeypatch.setattr(runner, "_inject_watch_notification", injected)
    entered = threading.Event()

    def timed_out_judge(_event, payload):
        entered.set()
        time.sleep(0.05)
        return payload

    monkeypatch.setattr(
        "tools.process_registry.completion_delivery_prompt", timed_out_judge
    )
    async def exercise():
        delivery = asyncio.create_task(
            runner._deliver_completion_notification(
                "payload", _completion_event(started_at=3.0)
            )
        )
        await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1)
        tick = asyncio.Event()
        asyncio.get_running_loop().call_later(0.01, tick.set)
        await asyncio.wait_for(tick.wait(), timeout=0.1)
        assert not delivery.done()
        return await delivery

    assert asyncio.run(exercise()) is True


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
    assert acknowledgements == []
    delivered_event = adapter.handle_message.await_args_list[-1].args[0]
    asyncio.run(
        runner._finish_completion_delivery_receipt(
            delivered_event, "committed"
        )
    )
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


def test_explicit_kill_returns_output_and_completion_still_fails_open(monkeypatch):
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

    adapter.handle_message.assert_awaited_once()


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
