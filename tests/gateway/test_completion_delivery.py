"""Gateway terminal-fence regressions for completion delivery.

One live GatewayRunner suppresses concurrent/replayed copies, failed injection
remains retryable, and push adapters retain durable claims until the provider
turn and transcript publication finish.
"""

import asyncio
import copy
import json
import queue
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, _drain_gateway_watch_events
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
        "termination_source": "",
        "output": "done\n",
    }


def _canonical_delegated_completion():
    return {
        **_completion_event(started_at=42.25, session_id="proc-collision"),
        "termination_source": "",
        "delegated_child": True,
    }


def _malformed_collision_copy(event):
    malformed = json.loads(json.dumps(event))
    malformed.pop("termination_source")
    malformed.update(
        command="distinct-malformed-command",
        output="distinct malformed output",
        error="distinct malformed error",
        _completion_delivery_token="serialized-token-canary",
        _completion_delivery_claim_id="serialized-claim-canary",
    )
    return malformed


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


class _InterleavingDrainQueue(queue.Queue):
    def __init__(self, producer_start, producer_done):
        super().__init__()
        self.producer_start = producer_start
        self.producer_done = producer_done
        self.legacy_gets = 0
        self.triggered = False

    def get_nowait(self):
        self.legacy_gets += 1
        return super().get_nowait()

    def release_during_selective_drain(self):
        if not self.legacy_gets and not self.triggered:
            self.triggered = True
            self.producer_start.set()
            assert self.producer_done.wait(2), "producer did not finish"

    def empty(self):
        with self.mutex:
            observed_empty = not self._qsize()
        if observed_empty and self.legacy_gets and not self.triggered:
            self.triggered = True
            self.producer_start.set()
            assert self.producer_done.wait(2), "producer did not finish"
            return True
        return observed_empty


class _DrainSignalEvent(dict):
    def __init__(self, completion_queue, **values):
        super().__init__(values)
        self.completion_queue = completion_queue

    def get(self, key, default=None):
        if key == "type":
            self.completion_queue.release_during_selective_drain()
        return super().get(key, default)


def test_gateway_watch_drain_preserves_foreign_fifo_during_concurrent_put():
    producer_start = threading.Event()
    producer_done = threading.Event()
    completion_queue = _InterleavingDrainQueue(producer_start, producer_done)
    first = _DrainSignalEvent(completion_queue, type="completion", seq=1)
    watch = _DrainSignalEvent(completion_queue, type="watch_match", seq="watch")
    second = _DrainSignalEvent(completion_queue, type="completion", seq=2)
    third = _DrainSignalEvent(completion_queue, type="completion", seq=3)
    for event in (first, watch, second):
        completion_queue.put(event)

    def produce():
        assert producer_start.wait(2), "watch drain did not reach producer boundary"
        completion_queue.put(third)
        producer_done.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        assert _drain_gateway_watch_events(completion_queue) == [watch]
        producer.join(2)
        assert not producer.is_alive()
        with completion_queue.mutex:
            remaining = list(completion_queue.queue)
        assert [event["seq"] for event in remaining] == [1, 2, 3]
    finally:
        producer_start.set()
        producer.join(2)


def test_gateway_completion_drain_preserves_watch_fifo_during_concurrent_put(
    monkeypatch, isolated_registry
):
    producer_start = threading.Event()
    producer_done = threading.Event()
    completion_queue = _InterleavingDrainQueue(producer_start, producer_done)
    first = _DrainSignalEvent(completion_queue, type="watch_match", seq=1)
    heartbeat = _DrainSignalEvent(completion_queue, type="heartbeat", seq="heartbeat")
    second = _DrainSignalEvent(completion_queue, type="watch_match", seq=2)
    third = _DrainSignalEvent(completion_queue, type="watch_match", seq=3)
    for event in (first, heartbeat, second):
        completion_queue.put(event)
    isolated_registry.completion_queue = completion_queue

    def produce():
        assert producer_start.wait(2), (
            "completion drain did not reach producer boundary"
        )
        completion_queue.put(third)
        producer_done.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    runner = _runner(SimpleNamespace())
    runner._handle_heartbeat_event = AsyncMock()
    _stop_after_sleeps(monkeypatch, runner, count=2)
    try:
        asyncio.run(runner._async_delegation_watcher(interval=0))
        producer.join(2)
        assert not producer.is_alive()
        runner._handle_heartbeat_event.assert_awaited_once_with(heartbeat)
        with completion_queue.mutex:
            remaining = list(completion_queue.queue)
        assert [event["seq"] for event in remaining] == [1, 2, 3]
    finally:
        producer_start.set()
        producer.join(2)


def test_gateway_retry_restores_selected_tail_before_concurrent_put(
    monkeypatch, isolated_registry
):
    producer_start = threading.Event()
    producer_done = threading.Event()
    completion_queue = _InterleavingDrainQueue(producer_start, producer_done)
    first = _DrainSignalEvent(
        completion_queue,
        **_completion_event(started_at=1.0, session_id="proc-fifo-1"),
    )
    second = _DrainSignalEvent(
        completion_queue,
        **_completion_event(started_at=2.0, session_id="proc-fifo-2"),
    )
    concurrent = _completion_event(started_at=3.0, session_id="proc-fifo-3")
    completion_queue.put(first)
    completion_queue.put(second)
    isolated_registry.completion_queue = completion_queue

    def produce():
        assert producer_start.wait(2), "watcher did not reach producer boundary"
        completion_queue.put(concurrent)
        producer_done.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    runner = _runner(SimpleNamespace())
    runner._deliver_completion_notification = AsyncMock(return_value=False)
    _stop_after_sleeps(monkeypatch, runner, count=2)
    try:
        asyncio.run(runner._async_delegation_watcher(interval=0))
        producer.join(2)
        assert not producer.is_alive()
        with completion_queue.mutex:
            remaining = list(completion_queue.queue)
        expected = [first, second, concurrent]
        assert remaining == expected
        assert all(actual is wanted for actual, wanted in zip(remaining, expected))
    finally:
        producer_start.set()
        producer.join(2)


def test_gateway_failed_owner_does_not_block_other_owner_progress(
    monkeypatch, isolated_registry
):
    producer_start = threading.Event()
    producer_done = threading.Event()
    completion_queue = _InterleavingDrainQueue(producer_start, producer_done)

    def routed(seq, owner):
        routed_event = _DrainSignalEvent(
            completion_queue,
            **_completion_event(
                started_at=float(seq), session_id=f"proc-{owner}-{seq}"
            ),
        )
        routed_event["session_key"] = owner
        routed_event["owner"] = owner
        routed_event["seq"] = seq
        return routed_event

    first = routed(1, "A")
    foreign_b = routed(1, "B")
    second = routed(2, "A")
    foreign_c = routed(1, "C")
    concurrent = _completion_event(started_at=3.0, session_id="proc-A-3")
    concurrent["session_key"] = "A"
    concurrent["owner"] = "A"
    concurrent["seq"] = 3
    for event in (first, foreign_b, second, foreign_c):
        completion_queue.put(event)
    isolated_registry.completion_queue = completion_queue

    def produce():
        assert producer_start.wait(2), "watcher did not reach producer boundary"
        completion_queue.put(concurrent)
        producer_done.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    observed = []
    runner = _runner(SimpleNamespace())

    async def deliver(_text, event, **_kwargs):
        observed.append((event["owner"], event["seq"]))
        return event is not first

    runner._deliver_completion_notification = AsyncMock(side_effect=deliver)
    _stop_after_sleeps(monkeypatch, runner, count=2)
    try:
        asyncio.run(runner._async_delegation_watcher(interval=0))
        producer.join(2)
        assert not producer.is_alive()
        assert observed == [("A", 1), ("B", 1), ("C", 1)]
        with completion_queue.mutex:
            remaining = list(completion_queue.queue)
        assert remaining == [first, second, concurrent]
    finally:
        producer_start.set()
        producer.join(2)


def test_gateway_ordinary_watcher_failure_has_one_retry(monkeypatch, isolated_registry):
    event = _completion_event(started_at=4.0, session_id="proc-single-retry")
    event["exit_code"] = 1
    isolated_registry.completion_queue.put(event)
    runner = _runner(SimpleNamespace())
    runner._inject_watch_notification = AsyncMock(return_value=False)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    with isolated_registry.completion_queue.mutex:
        retries = list(isolated_registry.completion_queue.queue)
    assert len(retries) == 1
    assert retries[0]["session_id"] == event["session_id"]
    assert retries[0]["started_at"] == event["started_at"]


def _committing_gateway_runner(monkeypatch, tmp_path, db):
    """Run accepted synthetic events through the real gateway history fence."""
    from tests.gateway.test_first_turn_session_meta_rebaseline import (
        SESSION_ID,
        SESSION_KEY,
        _bootstrap,
    )

    runner = _bootstrap(monkeypatch, tmp_path, db)
    runner.session_store._entries = {}
    setattr(runner, "_session_source_cache", {})
    committed_prompts = []

    async def run_agent(**kwargs):
        prompt = kwargs["message"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "terminal"},
        ]
        db.append_message(SESSION_ID, "user", prompt)
        db.append_message(SESSION_ID, "assistant", "terminal")
        committed_prompts.append(prompt)
        return {
            "final_response": "terminal",
            "messages": messages,
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "api_calls": 1,
            "completed": True,
            "completion_delivery_status": "committed",
        }

    runner._run_agent = AsyncMock(side_effect=run_agent)
    adapter = SimpleNamespace(supports_async_delivery=True, send=AsyncMock())

    async def handle_message(event):
        await runner._handle_message_with_agent(event, event.source, SESSION_KEY, 1)
        runner._running = False

    adapter.handle_message = AsyncMock(side_effect=handle_message)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running = True
    return runner, adapter, committed_prompts


async def _run_gateway_completion_queue(monkeypatch, runner, registry):
    """Drive both production gateway consumers that previously dropped retries."""
    assert _drain_gateway_watch_events(registry.completion_queue) == []
    assert not registry.completion_queue.empty()
    _stop_after_sleeps(monkeypatch, runner, count=2)
    await runner._async_delegation_watcher(interval=0)


def test_gateway_recovers_checkpointed_completion_through_history_commit(
    monkeypatch, tmp_path, isolated_registry
):
    """A failed initial persist survives restart and reaches a terminal gateway turn."""
    from hermes_state import SessionDB
    from tests.gateway.test_first_turn_session_meta_rebaseline import SESSION_ID
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    persist = ad.persist_event_delivery
    attempts = 0

    def fail_once(event):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("storage unavailable")
        return persist(event)

    monkeypatch.setattr(ad, "persist_event_delivery", fail_once)
    process = ProcessSession(
        id="proc-gateway-checkpoint",
        command="true",
        session_key="agent:main:telegram:dm:123",
        started_at=5.7,
        pid=999999999,
        watcher_platform="telegram",
        watcher_chat_id="123",
        exited=False,
        exit_code=1,
        output_buffer="failed after writing useful output\n",
        notify_on_complete=True,
    )
    isolated_registry._running[process.id] = process
    assert isolated_registry._write_checkpoint()
    process.exited = True
    isolated_registry._move_to_finished(process)
    assert isolated_registry.completion_queue.get_nowait()["session_id"] == process.id

    restarted = ProcessRegistry()
    assert restarted.recover_from_checkpoint() == 0
    monkeypatch.setattr(registry_module, "process_registry", restarted)
    restored = restarted.completion_queue.get_nowait()
    assert restored["session_id"] == process.id
    restarted.completion_queue.put(restored)

    db = SessionDB(db_path=tmp_path / "sessions.db")
    db.create_session(SESSION_ID, source="telegram")
    runner, adapter, committed_prompts = _committing_gateway_runner(
        monkeypatch, tmp_path, db
    )
    complete = ad.complete_event_delivery

    def complete_after_history(event, claim_id):
        assert committed_prompts
        assert any(
            row["content"] == committed_prompts[-1]
            for row in db.get_messages_as_conversation(SESSION_ID)
        )
        return complete(event, claim_id)

    monkeypatch.setattr(ad, "complete_event_delivery", complete_after_history)
    try:
        asyncio.run(_run_gateway_completion_queue(monkeypatch, runner, restarted))
        adapter.handle_message.assert_awaited_once()
        receipt = ad.get_durable_event_delivery(restored)
        assert receipt is not None
        assert receipt["delivery_state"] == "delivered"

        from tools.process_registry import format_process_notification

        replay_text = format_process_notification(restored)
        assert replay_text
        assert asyncio.run(
            runner._deliver_completion_notification(replay_text, dict(restored))
        ) is None
        adapter.handle_message.assert_awaited_once()

        after_restart = ProcessRegistry()
        assert after_restart.recover_from_checkpoint() == 0
        assert after_restart.completion_queue.empty()
    finally:
        db.close()


def test_gateway_committed_effect_ack_failure_never_requeues(
    monkeypatch, isolated_registry
):
    """A committed effect records ACK recovery instead of retrying its provider."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    event = _completion_event(
        started_at=5.8, session_id="proc-gateway-retained-claim"
    )
    event["exit_code"] = 1
    assert ad.persist_event_delivery(event)
    assert isolated_registry.claim_completion_delivery(event)
    claim = ad.claim_event_delivery(event, "gateway-test")
    assert claim
    release = MagicMock(side_effect=AssertionError("committed effect was released"))
    monkeypatch.setattr(
        ad,
        "complete_event_delivery",
        lambda *_args: (_ for _ in ()).throw(OSError("complete unavailable")),
    )
    monkeypatch.setattr(ad, "release_event_delivery", release)

    assert registry_module.finish_completion_event_delivery(
        event, claim, "committed", registry=isolated_registry
    )
    release.assert_not_called()
    assert isolated_registry.completion_queue.empty()
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "recovery_committed_ack_failed"

    restarted = ProcessRegistry()
    assert restarted.completion_queue.empty()


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


def test_genuine_local_token_transplant_keeps_distinct_malformed_event_visible(
    monkeypatch, isolated_registry
):
    """A token minted for producer A cannot claim producer B."""
    import tools.process_registry as registry_module

    effects = []

    async def inject(_text, event):
        effects.append(event["command"])
        return True

    runner = _runner(SimpleNamespace(handle_message=AsyncMock()))
    monkeypatch.setattr(runner, "_inject_watch_notification", inject)

    def malformed(name, started_at):
        event = isolated_registry._completion_event(
            ProcessSession(
                id=f"proc-{name}",
                command=f"{name}-command",
                session_key="agent:main:telegram:dm:123",
                started_at=started_at,
                output_buffer=f"{name}-output",
                exited=True,
                exit_code=1,
                completion_reason="exited",
                notify_on_complete=True,
            )
        )
        event.pop("session_id")
        event["error"] = f"{name}-error"
        return event

    first = malformed("first", 10.0)
    second = malformed("second", 20.0)
    first_token = first["_completion_delivery_token"]
    assert first_token != second["_completion_delivery_token"]
    second["_completion_delivery_token"] = first_token

    assert (
        asyncio.run(runner._deliver_completion_notification("visible", first)) is True
    )
    for duplicate in (dict(first), copy.copy(first), copy.deepcopy(first)):
        assert (
            asyncio.run(runner._deliver_completion_notification("visible", duplicate))
            is None
        )

    assert (
        asyncio.run(runner._deliver_completion_notification("visible", second)) is True
    )
    assert effects == ["first-command", "second-command"]
    assert registry_module.completion_delivery_prompt(second, "visible") is not None


def test_checkpoint_malformed_tuple_collision_delivers_once_and_stays_settled(
    monkeypatch, tmp_path, isolated_registry
):
    """A sanitized checkpoint event cannot inherit canonical success authority."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"auxiliary": {"completion_visibility": {"enabled": False}}},
    )
    effects = []

    async def inject(_text, event):
        effects.append(event["command"])
        return True

    runner = _runner(SimpleNamespace(handle_message=AsyncMock()))
    monkeypatch.setattr(runner, "_inject_watch_notification", inject)
    canonical = _canonical_delegated_completion()
    assert (
        asyncio.run(runner._deliver_completion_notification("canonical", canonical))
        is None
    )
    canonical_receipt = ad.get_durable_event_delivery(canonical)
    assert canonical_receipt and canonical_receipt["delivery_state"] == "delivered"

    malformed = _malformed_collision_copy(canonical)
    registry_module.CHECKPOINT_PATH.write_text(
        json.dumps([{"completion_event": malformed}]), encoding="utf-8"
    )
    restarted = ProcessRegistry()
    assert restarted.recover_from_checkpoint() == 0
    restored = restarted.completion_queue.get_nowait()
    assert not any(key.startswith("_completion_delivery_") for key in restored)
    assert registry_module.completion_delivery_prompt(restored, "visible") is not None
    monkeypatch.setattr(registry_module, "process_registry", restarted)

    assert (
        asyncio.run(runner._deliver_completion_notification("visible", restored))
        is True
    )
    assert effects == ["distinct-malformed-command"]
    malformed_receipt = ad.get_durable_event_delivery(restored)
    assert malformed_receipt and malformed_receipt["delivery_state"] == "delivered"
    assert malformed_receipt["delivery_id"] != canonical_receipt["delivery_id"]
    assert (
        asyncio.run(runner._deliver_completion_notification("visible", dict(restored)))
        is None
    )
    assert effects == ["distinct-malformed-command"]

    restored_after_restart = queue.Queue()
    assert ad.restore_undelivered_completions(restored_after_restart) == 0
    assert restored_after_restart.empty()


def test_sqlite_malformed_tuple_collision_delivers_once_and_stays_settled(
    monkeypatch, isolated_registry
):
    """Legacy tuple-keyed malformed SQLite traffic migrates before arbitration."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"auxiliary": {"completion_visibility": {"enabled": False}}},
    )
    canonical = _canonical_delegated_completion()
    malformed = _malformed_collision_copy(canonical)
    legacy_id = ad._ordinary_completion_legacy_delivery_id(canonical)
    assert legacy_id
    with ad._connect() as conn:
        conn.execute(
            """INSERT INTO ordinary_completion_deliveries
               (delivery_id, event_json, delivery_state, updated_at)
               VALUES (?, ?, 'pending', ?)""",
            (legacy_id, json.dumps(malformed), 1.0),
        )

    restarted = ProcessRegistry()
    restored = restarted.completion_queue.get_nowait()
    assert not any(key.startswith("_completion_delivery_") for key in restored)
    assert registry_module.completion_delivery_prompt(restored, "visible") is not None
    monkeypatch.setattr(registry_module, "process_registry", restarted)
    effects = []

    async def inject(_text, event):
        effects.append(event["command"])
        return True

    runner = _runner(SimpleNamespace(handle_message=AsyncMock()))
    monkeypatch.setattr(runner, "_inject_watch_notification", inject)
    assert (
        asyncio.run(runner._deliver_completion_notification("canonical", canonical))
        is None
    )
    canonical_receipt = ad.get_durable_event_delivery(canonical)
    assert canonical_receipt and canonical_receipt["delivery_state"] == "delivered"

    assert (
        asyncio.run(runner._deliver_completion_notification("visible", restored))
        is True
    )
    assert effects == ["distinct-malformed-command"]
    malformed_receipt = ad.get_durable_event_delivery(restored)
    assert malformed_receipt and malformed_receipt["delivery_state"] == "delivered"
    assert malformed_receipt["delivery_id"] != canonical_receipt["delivery_id"]
    assert (
        asyncio.run(runner._deliver_completion_notification("visible", dict(restored)))
        is None
    )
    assert effects == ["distinct-malformed-command"]

    restored_after_restart = queue.Queue()
    assert ad.restore_undelivered_completions(restored_after_restart) == 0
    assert restored_after_restart.empty()


def test_committed_gateway_effect_ack_failure_reconciles_without_provider_replay(
    monkeypatch, isolated_registry
):
    """A committed parent effect retries only its ACK, never the provider event."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    adapter = SimpleNamespace(handle_message=AsyncMock(), supports_push=True)
    runner = _runner(adapter)
    event = _completion_event(started_at=43.0, session_id="proc-committed-ack-failure")
    event["exit_code"] = 1

    assert (
        asyncio.run(runner._deliver_completion_notification("visible", event)) is True
    )
    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    receipt = ad.get_durable_event_delivery(event)
    assert receipt and receipt["delivery_state"] == "effect_started"

    complete_calls = 0

    def fail_complete(*_args):
        nonlocal complete_calls
        complete_calls += 1
        raise OSError("intentional committed ACK failure")

    real_complete = ad.complete_event_delivery
    real_mark_recovery = ad.mark_completion_delivery_recovery
    release = MagicMock(side_effect=AssertionError("committed effect was released"))
    monkeypatch.setattr(ad, "complete_event_delivery", fail_complete)
    monkeypatch.setattr(
        ad, "mark_completion_delivery_recovery", MagicMock(return_value=False)
    )
    monkeypatch.setattr(ad, "release_event_delivery", release)
    asyncio.run(runner._finish_completion_delivery_receipt(delivered, "committed"))

    assert complete_calls == 1
    release.assert_not_called()
    assert isolated_registry.completion_queue.empty()
    receipt = ad.get_durable_event_delivery(event)
    assert receipt and receipt["delivery_state"] == "effect_started"
    checkpoint_entries = json.loads(
        registry_module.CHECKPOINT_PATH.read_text(encoding="utf-8")
    )
    assert checkpoint_entries[0]["completion_ack"]["session_id"] == event["session_id"]

    monkeypatch.setattr(ad, "complete_event_delivery", real_complete)
    monkeypatch.setattr(ad, "mark_completion_delivery_recovery", real_mark_recovery)
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: False)
    restarted = ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", restarted)
    restart_adapter = SimpleNamespace(handle_message=AsyncMock(), supports_push=True)
    restart_runner = _runner(restart_adapter)
    assert (
        asyncio.run(
            restart_runner._deliver_completion_notification("visible", dict(event))
        )
        is None
    )
    restart_adapter.handle_message.assert_not_awaited()
    assert restarted.completion_queue.empty()
    receipt = ad.get_durable_event_delivery(event)
    assert receipt and receipt["delivery_state"] == "recovery_committed_ack_failed"


def test_busy_gateway_merge_finalizes_every_completion_receipt(
    monkeypatch, isolated_registry
):
    """One queued provider turn retains every completion's durable fence."""
    from gateway.platforms.base import merge_pending_message_event
    from tools import async_delegation as ad

    pending = {}

    async def queue_while_busy(event):
        merge_pending_message_event(pending, "busy", event, merge_text=True)

    adapter = SimpleNamespace(
        handle_message=AsyncMock(side_effect=queue_while_busy),
        supports_push=True,
    )
    runner = _runner(adapter)
    events = [
        _completion_event(started_at=2.51, session_id="proc-merge-one"),
        _completion_event(started_at=2.52, session_id="proc-merge-two"),
    ]

    assert asyncio.run(
        runner._deliver_completion_notification("completion-one", events[0])
    ) is True
    assert asyncio.run(
        runner._deliver_completion_notification("completion-two", events[1])
    ) is True

    merged = pending["busy"]
    assert merged.text.count("completion-one") == 1
    assert merged.text.count("completion-two") == 1
    asyncio.run(runner._finish_completion_delivery_receipt(merged, "committed"))

    for event in events:
        receipt = ad.get_durable_event_delivery(event)
        assert receipt is not None
        assert receipt["delivery_state"] == "delivered"
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: False)
    restarted_queue = queue.Queue()
    assert ad.restore_undelivered_completions(restarted_queue) == 0
    assert restarted_queue.empty()


@pytest.mark.asyncio
async def test_cancelled_gateway_completion_turn_releases_receipt(
    monkeypatch, tmp_path, isolated_registry
):
    """Task cancellation cannot abandon a gateway receipt under its live PID."""
    from hermes_state import SessionDB
    from tests.gateway.test_first_turn_session_meta_rebaseline import (
        SESSION_ID,
        SESSION_KEY,
        _bootstrap,
        _event,
        _source,
    )
    from tools import async_delegation as ad

    db = SessionDB(db_path=tmp_path / "sessions.db")
    db.create_session(SESSION_ID, source="telegram")
    runner = _bootstrap(monkeypatch, tmp_path, db)
    delivery_event = _completion_event(
        started_at=2.53, session_id="proc-cancelled-gateway"
    )
    delivery_event["exit_code"] = 1
    delivery_event["session_key"] = SESSION_KEY
    assert ad.persist_event_delivery(delivery_event)
    assert isolated_registry.claim_completion_delivery(delivery_event)
    claim = ad.claim_event_delivery(delivery_event, "gateway-test")
    assert claim
    event = _event()
    event.internal = True
    event.metadata = {
        "_completion_delivery_synthetic": True,
        "_completion_delivery_receipt": {
            "event": delivery_event,
            "claim_id": claim,
        },
    }
    runner._run_agent = AsyncMock(side_effect=asyncio.CancelledError())

    try:
        with pytest.raises(asyncio.CancelledError):
            await runner._handle_message_with_agent(event, _source(), SESSION_KEY, 1)

        receipt = ad.get_durable_event_delivery(delivery_event)
        assert receipt is not None
        assert receipt["delivery_state"] == "pending"
        assert not isolated_registry.completion_queue.empty()

        retry_runner, adapter, _committed = _committing_gateway_runner(
            monkeypatch, tmp_path, db
        )
        await _run_gateway_completion_queue(
            monkeypatch, retry_runner, isolated_registry
        )
        adapter.handle_message.assert_awaited_once()
        receipt = ad.get_durable_event_delivery(delivery_event)
        assert receipt is not None
        assert receipt["delivery_state"] == "delivered"
        assert isolated_registry.completion_queue.empty()
    finally:
        db.close()


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


def test_gateway_watcher_silences_delegated_success_and_commits_lifecycle(
    monkeypatch, isolated_registry
):
    """The live watcher must retain the canonical delegated-child marker."""
    import tools.process_registry as registry_module
    from tools import async_delegation as ad

    process = ProcessSession(
        id="proc-gateway-delegated-success",
        command="true",
        task_id="task",
        session_key="agent:main:telegram:dm:123",
        started_at=6.0,
        output_buffer="done\n",
        exited=True,
        exit_code=0,
        completion_reason="exited",
        notify_on_complete=True,
        delegated_child=True,
    )
    isolated_registry._finished[process.id] = process
    monkeypatch.setattr(registry_module, "process_registry", isolated_registry)

    async def _instant_sleep(*_args, **_kwargs):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    asyncio.run(runner._run_process_watcher({
        "session_id": process.id,
        "check_interval": 0,
        "session_key": process.session_key,
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    adapter.handle_message.assert_not_awaited()
    receipt = ad.get_durable_event_delivery(isolated_registry._completion_event(process))
    assert receipt is not None
    assert receipt["delivery_state"] == "delivered"


@pytest.mark.parametrize("missing_key", ["termination_source", "session_id"])
def test_gateway_watcher_delivers_incomplete_delegated_completion_and_preserves_recovery(
    monkeypatch, isolated_registry, missing_key
):
    """The real watcher must fail open when its canonical envelope is incomplete."""
    import tools.process_registry as registry_module
    from tools import async_delegation as ad

    process = ProcessSession(
        id=f"proc-gateway-incomplete-{missing_key}",
        command="true",
        task_id="task",
        session_key="agent:main:telegram:dm:123",
        started_at=6.1,
        output_buffer="done\n",
        exited=True,
        exit_code=0,
        completion_reason="exited",
        notify_on_complete=True,
        delegated_child=True,
    )
    isolated_registry._finished[process.id] = process
    canonical_event = isolated_registry._completion_event

    def incomplete_event(session):
        event = canonical_event(session)
        event.pop(missing_key)
        return event

    monkeypatch.setattr(isolated_registry, "_completion_event", incomplete_event)
    monkeypatch.setattr(registry_module, "process_registry", isolated_registry)

    async def _instant_sleep(*_args, **_kwargs):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    asyncio.run(runner._run_process_watcher({
        "session_id": process.id,
        "check_interval": 0,
        "session_key": process.session_key,
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    adapter.handle_message.assert_awaited_once()
    event = incomplete_event(process)
    receipt = ad.get_durable_event_delivery(event)
    if missing_key == "termination_source":
        assert receipt is not None
        assert receipt["delivery_state"] == "effect_started"
        delivered_event = adapter.handle_message.await_args.args[0]
        asyncio.run(runner._finish_completion_delivery_receipt(
            delivered_event, "committed"
        ))
        receipt = ad.get_durable_event_delivery(event)
        assert receipt is not None
        assert receipt["delivery_state"] == "delivered"
    else:
        assert receipt is None
    assert isolated_registry.completion_queue.empty()


def _run_malformed_producer_through_both_gateway_watchers(
    monkeypatch, isolated_registry, mutate_event
):
    """Return one producer event after both competing consumers settle it."""
    import tools.process_registry as registry_module

    process = ProcessSession(
        id="proc-gateway-malformed-producer",
        command="true",
        task_id="task",
        session_key="agent:main:telegram:dm:123",
        started_at=6.2,
        output_buffer="done\n",
        exited=True,
        exit_code=0,
        completion_reason="exited",
        notify_on_complete=True,
        delegated_child=True,
    )
    canonical_event = isolated_registry._completion_event

    def malformed_event(session):
        event = canonical_event(session)
        mutate_event(event)
        return event

    monkeypatch.setattr(isolated_registry, "_completion_event", malformed_event)
    monkeypatch.setattr(registry_module, "process_registry", isolated_registry)
    isolated_registry._running[process.id] = process
    isolated_registry._move_to_finished(process)
    queued_event = isolated_registry.completion_queue.get_nowait()
    isolated_registry.completion_queue.put(queued_event)

    async def _instant_sleep(*_args, **_kwargs):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    asyncio.run(runner._run_process_watcher({
        "session_id": process.id,
        "check_interval": 0,
        "session_key": process.session_key,
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))
    if adapter.handle_message.await_count:
        asyncio.run(runner._finish_completion_delivery_receipt(
            adapter.handle_message.await_args.args[0], "committed"
        ))

    asyncio.run(_run_gateway_completion_queue(
        monkeypatch, runner, isolated_registry
    ))
    return queued_event, adapter


@pytest.mark.parametrize(
    "started_at",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_nonfinite_completion_fails_open_once_without_durable_identity(
    monkeypatch, isolated_registry, started_at
):
    from tools import async_delegation as ad

    event, adapter = _run_malformed_producer_through_both_gateway_watchers(
        monkeypatch,
        isolated_registry,
        lambda candidate: candidate.__setitem__("started_at", started_at),
    )

    adapter.handle_message.assert_awaited_once()
    assert ad.get_durable_event_delivery(event) is None
    assert not isolated_registry.completion_event_should_deliver(event)
    assert isolated_registry.completion_queue.empty()
    restarted = ProcessRegistry()
    assert restarted.recover_from_checkpoint() == 0
    assert restarted.completion_queue.empty()


def test_missing_session_completion_fails_open_once_without_durable_identity(
    monkeypatch, isolated_registry
):
    from tools import async_delegation as ad

    event, adapter = _run_malformed_producer_through_both_gateway_watchers(
        monkeypatch,
        isolated_registry,
        lambda candidate: candidate.pop("session_id"),
    )

    adapter.handle_message.assert_awaited_once()
    assert ad.get_durable_event_delivery(event) is None
    assert not isolated_registry.completion_event_should_deliver(event)
    assert isolated_registry.completion_queue.empty()
    token = event["_completion_delivery_token"]
    delivered_event = adapter.handle_message.await_args.args[0]
    assert token not in delivered_event.text
    assert token not in str(delivered_event.metadata)
    restarted = ProcessRegistry()
    assert restarted.recover_from_checkpoint() == 0
    assert restarted.completion_queue.empty()


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


def test_gateway_async_effect_uses_shared_ack_recovery(monkeypatch, isolated_registry):
    from tools import async_delegation as ad

    event = _async_event("deleg-gateway-shared-settlement")
    ad._persist_dispatch(event)
    ad._persist_completion(event, {"status": "completed", "summary": "Found it"})
    runner = _runner(SimpleNamespace(handle_message=AsyncMock(), supports_push=False))
    monkeypatch.setattr(
        runner, "_inject_watch_notification", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        ad,
        "complete_completion_delivery",
        MagicMock(side_effect=OSError("intentional direct ACK failure")),
    )

    assert (
        asyncio.run(runner._deliver_completion_notification("payload", event)) is True
    )
    durable = ad.get_durable_delegation(event["delegation_id"])
    assert durable is not None
    assert durable["delivery_state"] == "recovery_committed_ack_failed"
    assert isolated_registry.completion_queue.empty()


@pytest.mark.parametrize("event_type", ["completion", "async_delegation"])
def test_gateway_api_self_post_uses_completion_fence(monkeypatch, event_type):
    from tools import async_delegation as ad

    adapter = SimpleNamespace(handle_message=AsyncMock(), supports_async_delivery=False)
    runner = _runner(adapter)
    runner.adapters = {Platform.API_SERVER: adapter}  # type: ignore[assignment]
    if event_type == "async_delegation":
        event = _async_event("deleg-api-self-post")
        event.update(session_key="raw-api-session", origin_session_id="raw-api-session")
        ad._persist_dispatch(event)
        ad._persist_completion(event, {"status": "completed", "summary": "Found it"})
    else:
        event = _completion_event(started_at=4.25, session_id="proc-api-self-post")
        event.update(session_key="raw-api-session", origin_session_id="raw-api-session")
        for field in ("platform", "chat_type", "chat_id"):
            event.pop(field, None)
        assert ad.persist_event_delivery(event)
    self_post = AsyncMock()
    monkeypatch.setattr("gateway.wake.deliver_wake", self_post)

    assert (
        asyncio.run(runner._deliver_completion_notification("payload", event)) is True
    )
    call = self_post.await_args
    assert call is not None
    assert call.kwargs["completion_delivery"] is True
    if event_type == "async_delegation":
        durable = ad.get_durable_delegation(event["delegation_id"])
    else:
        durable = ad.get_durable_event_delivery(event)
    assert durable is not None
    assert durable["delivery_state"] == "delivered"


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


@pytest.mark.parametrize(
    "failure_stage", ["claim", "prompt_release", "inject_release"]
)
def test_gateway_storage_failure_retries_once_in_same_process(
    monkeypatch, isolated_registry, failure_stage
):
    """Claim and rollback writes cannot strand an ordinary gateway completion."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    event = _completion_event(
        started_at=5.1, session_id=f"proc-gateway-{failure_stage}"
    )
    assert ad.persist_event_delivery(event)
    runner = _runner(SimpleNamespace(handle_message=AsyncMock()))
    effects = []
    inject_calls = 0

    async def inject(_text, _event):
        nonlocal inject_calls
        inject_calls += 1
        if failure_stage == "inject_release" and inject_calls == 1:
            return False
        effects.append(_event["session_id"])
        return True

    monkeypatch.setattr(runner, "_inject_watch_notification", inject)
    prompt_calls = 0

    def prompt(_event, text):
        nonlocal prompt_calls
        prompt_calls += 1
        if failure_stage == "prompt_release" and prompt_calls == 1:
            raise RuntimeError("prompt failed")
        return text

    monkeypatch.setattr(registry_module, "completion_delivery_prompt", prompt)
    claim = ad.claim_event_delivery
    claim_calls = 0

    def claim_once(*args):
        nonlocal claim_calls
        claim_calls += 1
        if failure_stage == "claim" and claim_calls == 1:
            raise OSError("claim storage unavailable")
        return claim(*args)

    monkeypatch.setattr(ad, "claim_event_delivery", claim_once)
    release = ad.release_event_delivery
    release_calls = 0

    def release_once(*args):
        nonlocal release_calls
        release_calls += 1
        if failure_stage.endswith("release") and release_calls == 1:
            raise OSError("release storage unavailable")
        return release(*args)

    monkeypatch.setattr(ad, "release_event_delivery", release_once)

    if failure_stage == "prompt_release":
        with pytest.raises(RuntimeError, match="prompt failed"):
            asyncio.run(runner._deliver_completion_notification("payload", event))
    else:
        expected = True if failure_stage == "claim" else False
        assert (
            asyncio.run(runner._deliver_completion_notification("payload", event))
            is expected
        )

    if failure_stage == "claim":
        assert effects == [event["session_id"]]
        assert isolated_registry.completion_queue.empty()
        receipt = ad.get_durable_event_delivery(event)
        assert receipt is not None
        assert receipt["delivery_state"] == "delivered"
    else:
        assert effects == []
        assert isolated_registry.completion_event_should_deliver(event)
        retry = isolated_registry.completion_queue.get_nowait()
        assert isolated_registry.completion_queue.empty()
        receipt = ad.get_durable_event_delivery(event)
        assert receipt is not None
        assert receipt["delivery_state"] == "effect_started"
        assert retry["_completion_delivery_retained_claim_id"] == receipt["delivery_claim"]

        assert (
            asyncio.run(runner._deliver_completion_notification("payload", retry))
            is True
        )
        assert effects == [event["session_id"]]
        receipt = ad.get_durable_event_delivery(event)
        assert receipt is not None
        assert receipt["delivery_state"] == "delivered"
        assert isolated_registry.completion_queue.empty()
        assert not isolated_registry.completion_event_should_deliver(event)

    assert asyncio.run(
        runner._deliver_completion_notification("payload", dict(event))
    ) is None
    assert effects == [event["session_id"]]


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


def test_gateway_presentation_projection_shares_producer_delivery_identity(
    monkeypatch, isolated_registry
):
    from tools import async_delegation as ad

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"auxiliary": {"completion_visibility": {"enabled": False}}},
    )
    process = ProcessSession(
        id="proc-honest-long-output",
        command="printf long-output",
        task_id="task",
        session_key="agent:main:telegram:dm:123",
        started_at=72.0,
        output_buffer="prefix\n" + ("x" * 2400),
        exited=True,
        exit_code=0,
        completion_reason="exited",
        termination_source="",
        notify_on_complete=True,
        delegated_child=False,
    )
    isolated_registry._running[process.id] = process
    isolated_registry._move_to_finished(process)
    queued_event = isolated_registry.completion_queue.get_nowait()
    isolated_registry.completion_queue.put(queued_event)
    queued_delivery_id = ad._ordinary_completion_delivery_id(queued_event)
    assert queued_delivery_id

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._load_background_notifications_mode = lambda: "off"

    async def instant_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    asyncio.run(
        runner._run_process_watcher({
            "session_id": process.id,
            "check_interval": 0,
            "session_key": process.session_key,
            "platform": "telegram",
            "chat_type": "dm",
            "chat_id": "123",
            "notify_on_complete": True,
        })
    )
    delivered = adapter.handle_message.await_args.args[0]
    receipt_event = delivered.metadata["_completion_delivery_receipt"]["event"]
    assert ad._ordinary_completion_delivery_id(receipt_event) == queued_delivery_id
    asyncio.run(runner._finish_completion_delivery_receipt(delivered, "committed"))

    asyncio.run(_run_gateway_completion_queue(monkeypatch, runner, isolated_registry))

    adapter.handle_message.assert_awaited_once()
    receipt = ad.get_durable_event_delivery(queued_event)
    assert receipt is not None
    assert receipt["delivery_state"] == "delivered"


def test_released_projection_retry_survives_delayed_producer(
    monkeypatch, isolated_registry
):
    from tools import async_delegation as ad

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"auxiliary": {"completion_visibility": {"enabled": False}}},
    )
    process = ProcessSession(
        id="proc-projection-retry-race",
        command="printf long-output",
        task_id="task",
        session_key="agent:main:telegram:dm:123",
        started_at=73.0,
        output_buffer="prefix\n" + ("x" * 2400),
        exited=True,
        exit_code=0,
        completion_reason="exited",
        termination_source="",
        notify_on_complete=True,
        delegated_child=False,
    )
    isolated_registry._running[process.id] = process
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._load_background_notifications_mode = lambda: "off"

    async def instant_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    asyncio.run(
        runner._run_process_watcher({
            "session_id": process.id,
            "check_interval": 0,
            "session_key": process.session_key,
            "platform": "telegram",
            "chat_type": "dm",
            "chat_id": "123",
            "notify_on_complete": True,
        })
    )
    first_delivery = adapter.handle_message.await_args.args[0]
    first_event = first_delivery.metadata["_completion_delivery_receipt"]["event"]
    canonical_event = isolated_registry._completion_event(process)
    assert ad._ordinary_completion_delivery_id(
        first_event
    ) == ad._ordinary_completion_delivery_id(canonical_event)

    asyncio.run(
        runner._finish_completion_delivery_receipt(first_delivery, "provider_failed")
    )
    receipt = ad.get_durable_event_delivery(canonical_event)
    assert receipt is not None
    assert receipt["delivery_state"] == "pending"
    assert isolated_registry.completion_queue.qsize() == 1

    isolated_registry._move_to_finished(process)
    sleep_calls = 0

    async def bounded_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", bounded_sleep)
    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert adapter.handle_message.await_count == 2
    receipt = ad.get_durable_event_delivery(canonical_event)
    assert receipt is not None
    assert receipt["delivery_state"] == "effect_started"
    assert isolated_registry.completion_queue.empty()
