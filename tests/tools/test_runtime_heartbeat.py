"""Unit tests for built-in per-target runtime heartbeats."""

from __future__ import annotations

import queue

from tools.runtime_heartbeat import (
    RuntimeHeartbeat,
    inspect_delegation,
    resolve_kv_cache_ttl,
)


class FakeTimer:
    created = []

    def __init__(self, interval, callback):
        self.interval = interval
        self.callback = callback
        self.daemon = False
        self.started = False
        self.cancelled = False
        type(self).created.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


def _runtime(*, enabled=True, reset_on_caller_activation=True):
    return {
        "kv_cache_ttl": {
            "default": 100,
            "providers": {"openai": 200, "custom:pm": 300},
        },
        "heartbeat": {
            "enabled": enabled,
            "mode": "per_target",
            "safety_ratio": 0.8,
            "min_interval_seconds": 10,
            "max_interval_seconds": 250,
            "reset_on_caller_activation": reset_on_caller_activation,
        },
    }


def _manager(runtime=None):
    FakeTimer.created = []
    return RuntimeHeartbeat(
        config_loader=lambda: _runtime() if runtime is None else runtime,
        event_queue=queue.Queue(),
        timer_factory=FakeTimer,
    )


def test_resolve_ttl_exact_then_canonical_family_then_default_then_documented_fallback():
    runtime = _runtime()
    assert resolve_kv_cache_ttl(runtime, "custom:pm") == 300  # exact
    assert resolve_kv_cache_ttl(runtime, "openai-codex") == 200  # family openai
    assert resolve_kv_cache_ttl(runtime, "unknown") == 100  # configured default
    assert resolve_kv_cache_ttl({}, "unknown") == 3300  # documented fallback


def test_arm_and_cancel_are_per_target_not_shared_per_caller():
    manager = _manager()
    checks = {"one": {"alive": True, "output_size": 0}, "two": {"alive": True, "output_size": 0}}
    assert manager.arm("one", caller_id="caller", kind="process", provider="openai", inspect=lambda: checks["one"])
    assert manager.arm("two", caller_id="caller", kind="process", provider="openai", inspect=lambda: checks["two"])

    assert manager.cancel("one") is True
    assert manager.outstanding_for_caller("caller") == ["two"]
    assert FakeTimer.created[0].cancelled is True
    assert FakeTimer.created[1].cancelled is False


def test_reset_on_caller_activation_reschedules_only_that_callers_outstanding_targets():
    manager = _manager()
    check = lambda: {"alive": True, "output_size": 0}
    manager.arm("a", caller_id="caller-a", kind="process", provider="openai", inspect=check)
    manager.arm("b", caller_id="caller-b", kind="process", provider="openai", inspect=check)
    old_a, old_b = FakeTimer.created

    assert manager.reset_for_caller("caller-a") == 1
    assert old_a.cancelled is True
    assert old_b.cancelled is False
    assert FakeTimer.created[-1].interval == 160  # openai 200 * safety ratio .8
    assert manager.outstanding_for_caller("caller-a") == ["a"]
    assert manager.outstanding_for_caller("caller-b") == ["b"]


def test_reset_disabled_by_config():
    manager = _manager(_runtime(reset_on_caller_activation=False))
    manager.arm(
        "proc-no-reset",
        caller_id="caller",
        kind="process",
        provider="openai",
        inspect=lambda: {"alive": True, "output_size": 0},
    )
    original_timer = FakeTimer.created[-1]

    assert manager.reset_for_caller("caller") == 0
    assert original_timer.cancelled is False
    assert manager.outstanding_for_caller("caller") == ["proc-no-reset"]


def test_process_heartbeat_requires_output_or_cpu_growth_not_just_a_live_pid():
    manager = _manager()
    snapshots = [
        {"alive": True, "output_size": 10, "cpu_seconds": 1.0},
        {"alive": True, "output_size": 10, "cpu_seconds": 1.0},
    ]
    manager.arm("proc-stuck", caller_id="caller", kind="process", provider="openai", inspect=lambda: snapshots.pop(0))
    FakeTimer.created[-1].callback()

    evt = manager._event_queue.get_nowait()
    assert evt["type"] == "heartbeat"
    assert evt["status"] == "STUCK"
    assert "no output or CPU progress" in evt["evidence"]
    assert manager.outstanding_for_caller("caller") == []


def test_active_delegation_without_granular_activity_remains_alive(monkeypatch):
    record = {}
    monkeypatch.setattr(
        "tools.async_delegation.list_async_delegations",
        lambda: [record],
    )

    for status in ("running", "finalizing"):
        target_id = f"delegation-{status}"
        record = {
            "delegation_id": target_id,
            "status": status,
            "dispatched_at": 100.0,
            "last_activity_at": 100.0,
        }
        manager = _manager()
        manager.arm(
            target_id,
            caller_id="caller",
            kind="delegation",
            provider="openai",
            inspect=lambda: inspect_delegation(target_id),
        )

        FakeTimer.created[-1].callback()

        assert manager._event_queue is not None
        evt = manager._event_queue.get_nowait()
        assert evt["status"] == "ALIVE"
        assert f"no granular activity tracking; status={status}" in evt["evidence"]
        assert manager.outstanding_for_caller("caller") == [target_id]


def test_alive_heartbeat_is_compact_and_rearms_that_target():
    manager = _manager()
    snapshots = [
        {"alive": True, "output_size": 0, "cpu_seconds": 0.0},
        {"alive": True, "output_size": 12, "cpu_seconds": 0.0},
    ]
    manager.arm("proc-alive", caller_id="caller", kind="process", provider="openai", inspect=lambda: snapshots.pop(0))
    initial_timer = FakeTimer.created[-1]
    initial_timer.callback()

    evt = manager._event_queue.get_nowait()
    assert evt == {
        "type": "heartbeat", "target_id": "proc-alive", "target_kind": "process",
        "session_id": "proc-alive", "session_key": "caller", "status": "ALIVE",
        "evidence": "output grew 0->12 bytes",
    }
    assert len(FakeTimer.created) == 2
    assert manager.outstanding_for_caller("caller") == ["proc-alive"]


def test_default_config_exposes_per_target_runtime_contract():
    from hermes_cli.config import DEFAULT_CONFIG

    runtime = DEFAULT_CONFIG["runtime"]
    assert runtime["kv_cache_ttl"]["providers"]["custom:z1"] == 1700
    assert runtime["heartbeat"]["mode"] == "per_target"
    assert runtime["heartbeat"]["reset_on_caller_activation"] is True
