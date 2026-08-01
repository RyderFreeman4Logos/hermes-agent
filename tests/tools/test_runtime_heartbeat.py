"""Behavior contract for per-target runtime KV-cache heartbeats."""

from __future__ import annotations

import concurrent.futures
import queue
import threading
from types import SimpleNamespace

import pytest


class FakeTimer:
    created = []

    def __init__(self, interval, callback):
        self.interval = interval
        self.callback = callback
        self.cancelled = False
        self.daemon = False
        type(self).created.append(self)

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


def _runtime():
    return {
        "warm_kv_timeout": {
            "default": 31,
            "providers": {
                "custom": 29,
                "custom:pm": 1700,
                "openai": 23,
                "openai-codex": 1700,
            },
        },
        "heartbeat": {
            "enabled": True,
            "mode": "per_target",
            "safety_ratio": 0.01,
            "min_interval_seconds": 2,
            "max_interval_seconds": 7,
        },
    }


def test_canonical_runtime_provider_identity_is_lossless(monkeypatch):
    from tools.runtime_heartbeat import canonical_runtime_provider_identity

    custom = SimpleNamespace(
        provider="custom",
        requested_provider="custom:pm",
        base_url="https://pm.invalid/v1",
        model="gpt-5.4-mini",
    )
    builtin = SimpleNamespace(
        provider="openai-codex",
        requested_provider="openai-codex",
        base_url="",
        model="gpt-5.4-mini",
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.canonical_custom_identity",
        lambda **_kwargs: "custom:pm",
    )

    assert canonical_runtime_provider_identity(custom) == "custom:pm"
    assert canonical_runtime_provider_identity(builtin) == "openai-codex"


def test_exact_1700_ignores_family_default_ratio_and_clamps():
    from tools.runtime_heartbeat import resolve_heartbeat_interval

    assert resolve_heartbeat_interval(_runtime(), "custom:pm") == 1700
    assert resolve_heartbeat_interval(_runtime(), "openai-codex") == 1700


def test_bound_custom_identity_reaches_tool_worker_with_exact_interval(monkeypatch):
    from tools.runtime_heartbeat import (
        bind_agent_provider,
        get_current_provider,
        preflight_current_heartbeat,
        reset_current_provider,
    )
    from tools.thread_context import propagate_context_to_thread

    agent = SimpleNamespace(
        provider="custom",
        requested_provider="custom:pm",
        base_url="https://pm.invalid/v1",
        model="gpt-5.4-mini",
    )
    monkeypatch.setattr("tools.runtime_heartbeat._runtime_config", _runtime)
    token = bind_agent_provider(agent)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            provider, interval = executor.submit(
                propagate_context_to_thread(
                    lambda: (get_current_provider(), preflight_current_heartbeat())
                )
            ).result(timeout=5)
    finally:
        reset_current_provider(token)

    assert provider == "custom:pm"
    assert interval == 1700


@pytest.mark.parametrize(
    "providers",
    [
        {"custom": 1700, "default": 1700},
        {"custom:pm": 0},
        {"custom:pm": True},
        {"custom:pm": "1700"},
    ],
)
def test_missing_or_invalid_exact_mapping_is_explicit(providers):
    from tools.runtime_heartbeat import (
        HeartbeatConfigError,
        resolve_heartbeat_interval,
    )

    runtime = _runtime()
    runtime["warm_kv_timeout"]["providers"] = providers
    with pytest.raises(HeartbeatConfigError, match="custom:pm"):
        resolve_heartbeat_interval(runtime, "custom:pm")


def test_ordinary_idle_coordinator_arms_nothing():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)

    assert manager.outstanding_for_caller("idle-owner") == []
    assert FakeTimer.created == []


def test_per_target_owner_isolation_and_cancel():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "output_size": 0, "cpu_seconds": 0.0}
    assert manager.arm(
        "a", caller_id="owner-a", kind="process", interval=1700, inspect=alive
    )
    assert manager.arm(
        "b", caller_id="owner-b", kind="process", interval=1700, inspect=alive
    )

    assert manager.cancel("a") is True
    assert manager.outstanding_for_caller("owner-a") == []
    assert manager.outstanding_for_caller("owner-b") == ["b"]
    assert FakeTimer.created[0].cancelled is True
    assert FakeTimer.created[1].cancelled is False


def test_alive_fires_once_and_rearms_from_checkin_at_exact_interval():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    snapshots = iter(
        [
            {"alive": True, "output_size": 0, "cpu_seconds": 0.0},
            {"alive": True, "output_size": 8, "cpu_seconds": 0.0},
        ]
    )
    assert manager.arm(
        "proc",
        caller_id="owner",
        kind="process",
        interval=1700,
        inspect=lambda: next(snapshots),
    )
    first = FakeTimer.created[-1]
    assert first.interval == 1700

    first.callback()
    event = events.get_nowait()
    assert event["status"] == "ALIVE"
    assert event["session_key"] == "owner"
    assert events.empty()
    assert len(FakeTimer.created) == 2
    assert FakeTimer.created[-1].interval == 1700
    assert manager.outstanding_for_caller("owner") == ["proc"]


def test_stuck_live_target_emits_once_and_rearms_exactly_once():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    assert manager.arm(
        "proc",
        caller_id="owner",
        kind="process",
        interval=1700,
        inspect=lambda: {
            "alive": True,
            "output_size": 0,
            "cpu_seconds": 0.0,
        },
    )

    FakeTimer.created[0].callback()

    assert events.get_nowait()["status"] == "STUCK"
    assert events.empty()
    assert len(FakeTimer.created) == 2
    assert FakeTimer.created[1].interval == 1700


def test_caller_reset_preserves_exact_interval_and_other_owner():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}
    manager.arm(
        "a", caller_id="owner-a", kind="delegation", interval=1700, inspect=alive
    )
    manager.arm(
        "b", caller_id="owner-b", kind="delegation", interval=1700, inspect=alive
    )
    old_a, old_b = FakeTimer.created

    assert manager.reset_for_caller("owner-a") == 1
    assert old_a.cancelled is True
    assert old_b.cancelled is False
    assert FakeTimer.created[-1].interval == 1700


def test_cancel_waits_for_inflight_publication_before_returning():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    class BarrierQueue:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.events = []

        def put(self, event):
            self.entered.set()
            assert self.release.wait(timeout=2)
            self.events.append(event)

    FakeTimer.created = []
    events = BarrierQueue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    snapshots = iter([
        {"alive": True, "output_size": 0, "cpu_seconds": 0.0},
        {"alive": True, "output_size": 1, "cpu_seconds": 0.0},
    ])
    assert manager.arm(
        "proc", caller_id="owner", kind="process", interval=1700,
        inspect=lambda: next(snapshots),
    )

    fire = threading.Thread(target=FakeTimer.created[0].callback)
    fire.start()
    assert events.entered.wait(timeout=2)
    cancelled = []
    cancel = threading.Thread(target=lambda: cancelled.append(manager.cancel("proc")))
    cancel.start()
    cancel.join(timeout=0.05)
    assert cancel.is_alive(), "cancel returned while heartbeat publication was in flight"

    events.release.set()
    fire.join(timeout=2)
    cancel.join(timeout=2)
    assert not fire.is_alive() and not cancel.is_alive()
    assert cancelled == [True]
    published_at_cancel_return = len(events.events)
    assert published_at_cancel_return == 1
    assert len(events.events) == published_at_cancel_return
    assert manager.outstanding_for_caller("owner") == []


def test_inspection_failure_publishes_visible_unknown_without_rearming():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    calls = 0

    def inspect():
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"alive": True, "progress": True}
        raise RuntimeError("backend inspection failed")

    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    assert manager.arm(
        "delegate", caller_id="owner", kind="delegation", interval=1700,
        inspect=inspect,
    )

    FakeTimer.created[0].callback()

    event = events.get_nowait()
    assert event["status"] == "UNKNOWN"
    assert "backend inspection failed" in event["evidence"]
    assert manager.outstanding_for_caller("owner") == []
    assert len(FakeTimer.created) == 1


def test_sandbox_process_never_samples_same_number_host_pid(monkeypatch):
    import psutil
    from tools.process_registry import ProcessSession, process_registry
    from tools.runtime_heartbeat import inspect_process

    session = ProcessSession(
        id="sandbox", command="work", pid=1234, pid_scope="sandbox"
    )
    monkeypatch.setattr(process_registry, "get", lambda _target_id: session)
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda _pid: pytest.fail("sandbox PID was sampled through host psutil"),
    )

    snapshot = inspect_process("sandbox")

    assert snapshot["alive"] is True
    assert snapshot["cpu_seconds"] == 0.0


def test_host_process_cpu_includes_identity_checked_descendants(monkeypatch):
    import psutil
    from tools.process_registry import ProcessSession, process_registry
    from tools.runtime_heartbeat import inspect_process

    session = ProcessSession(
        id="host", command="work", pid=100, pid_scope="host",
        host_start_time=10,
    )
    session.process = SimpleNamespace(pid=100)
    session._tracked_descendants[200] = 20
    monkeypatch.setattr(process_registry, "get", lambda _target_id: session)
    monkeypatch.setattr(process_registry, "_remember_local_descendants", lambda _s: None)
    monkeypatch.setattr(
        process_registry,
        "_safe_host_start_time",
        lambda pid: {100: 10, 200: 20}[pid],
    )
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: SimpleNamespace(
            cpu_times=lambda: SimpleNamespace(user={100: 1.0, 200: 4.0}[pid], system=0.5)
        ),
    )

    snapshot = inspect_process("host")

    assert snapshot["cpu_seconds"] == 6.0


def test_stalling_delegation_is_stuck_while_finalizing_is_alive(monkeypatch):
    from tools.runtime_heartbeat import RuntimeHeartbeat, inspect_delegation

    monkeypatch.setattr(
        "tools.async_delegation.list_async_delegations",
        lambda: [{"delegation_id": "d", "status": "stalling"}],
    )
    stalled = inspect_delegation("d")
    assert stalled["alive"] is True
    assert stalled["progress"] is False
    target = SimpleNamespace(kind="delegation", baseline={})
    assert RuntimeHeartbeat._assess(target, stalled)[0] == "STUCK"

    monkeypatch.setattr(
        "tools.async_delegation.list_async_delegations",
        lambda: [{"delegation_id": "d", "status": "finalizing"}],
    )
    finalizing = inspect_delegation("d")
    assert finalizing == {
        "alive": True,
        "progress": True,
        "evidence": "delegation finalizing",
    }
