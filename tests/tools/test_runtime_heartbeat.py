"""Behavior contract for per-target runtime KV-cache heartbeats."""

from __future__ import annotations

import concurrent.futures
import queue
import shlex
import sys
import threading
import time
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


def _wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    pause = threading.Event()
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        pause.wait(0.05)
    return None


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


def test_named_custom_openai_codex_uses_base_mapping_but_unknown_alias_fails_closed():
    from tools.runtime_heartbeat import HeartbeatConfigError, resolve_heartbeat_interval

    runtime = _runtime()
    runtime["warm_kv_timeout"]["providers"] = {"openai-codex": 1700}
    assert resolve_heartbeat_interval(runtime, "custom:openai-codex") == 1700

    runtime["warm_kv_timeout"]["providers"] = {"pm": 1700}
    with pytest.raises(HeartbeatConfigError, match="custom:pm"):
        resolve_heartbeat_interval(runtime, "custom:pm")


def test_bound_custom_identity_reaches_live_target_with_exact_interval(monkeypatch):
    from tools.runtime_heartbeat import (
        RuntimeHeartbeat,
        bind_agent_provider,
        canonical_runtime_cache_context_identity,
        get_current_cache_context,
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
        api_mode="chat_completions",
    )
    monkeypatch.setattr("tools.runtime_heartbeat._runtime_config", _runtime)
    events = queue.Queue()
    FakeTimer.created = []
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)

    def arm_from_worker():
        interval = preflight_current_heartbeat()
        assert manager.arm(
            "proc",
            caller_id="owner",
            kind="process",
            interval=interval,
            inspect=lambda: {"alive": True, "progress": True},
        )
        return get_current_provider(), get_current_cache_context(), interval

    token = bind_agent_provider(agent)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            provider, cache_context, interval = executor.submit(
                propagate_context_to_thread(arm_from_worker)
            ).result(timeout=5)
    finally:
        reset_current_provider(token)

    assert provider == "custom:pm"
    assert cache_context == canonical_runtime_cache_context_identity(agent)
    assert interval == 1700
    FakeTimer.created[0].callback()
    event = events.get_nowait()
    assert (event["provider"], event["cache_context"]) == (
        provider,
        cache_context,
    )


def test_inactive_target_stays_unarmed():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)

    assert manager.arm(
        "done",
        caller_id="owner",
        kind="process",
        interval=1700,
        inspect=lambda: {"alive": False},
    ) is False
    assert manager.outstanding_for_caller("owner") == []
    assert FakeTimer.created == []


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


def test_successful_idle_session_arms_fires_and_cancels_without_process(
    monkeypatch, caplog
):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    class ImmediateThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    class Owner:
        provider = "openai"
        requested_provider = "openai"
        base_url = "https://api.openai.invalid/v1"
        model = "model"
        api_mode = "chat_completions"
        platform = "tui"
        session_id = "idle-owner"
        _turn_received_provider_response = False

        def __init__(self):
            self.calls = []

        def run_conversation(self, message, **kwargs):
            self.calls.append((message, kwargs))
            return {
                "heartbeat_warm_status": "warmed",
                "heartbeat_warm_reason": "physical_success",
            }

    monkeypatch.setattr("tools.runtime_heartbeat._runtime_config", _runtime)
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_warm_capability",
        lambda _owner: ("eligible", "test"),
    )
    monkeypatch.setattr("tools.runtime_heartbeat.threading.Thread", ImmediateThread)
    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    owner = Owner()
    result = {
        "completed": True,
        "failed": False,
        "interrupted": False,
        "api_calls": 1,
    }

    assert manager.arm_session_after_turn(
        owner, result, caller_id="idle-owner"
    ) is True
    [snapshot] = manager.active_snapshots()
    assert snapshot["kind"] == "session"
    assert snapshot["interval_s"] == 23

    with caplog.at_level("INFO", logger="tools.runtime_heartbeat"):
        FakeTimer.created[0].callback()
    event = events.get_nowait()
    assert event["target_kind"] == "session"
    assert event["status"] == "ALIVE"
    assert event["heartbeat_warm_owned"] is True
    assert owner.calls == [
        ("", {"turn_origin": "heartbeat_warm", "heartbeat_event": event})
    ]
    assert any("phase=due" in record.message for record in caplog.records)

    assert manager.cancel_session(owner) is True
    assert FakeTimer.created[-1].cancelled is True
    assert manager.outstanding_for_caller("idle-owner") == []
    assert manager.arm_session_after_turn(
        owner, {**result, "completed": False}, caller_id="idle-owner"
    ) is False
    owner.platform = "subagent"
    assert manager.arm_session_after_turn(
        owner, result, caller_id="idle-owner"
    ) is False


def test_turn_compress_and_model_switch_manage_session_warm(monkeypatch):
    from run_agent import AIAgent
    from tools.runtime_heartbeat import runtime_heartbeat

    calls = []
    result = {"final_response": "done", "completed": True, "interrupted": False}
    agent = AIAgent.__new__(AIAgent)
    vars(agent).update(
        session_id="idle-owner",
        platform="cli",
        _parent_session_id=None,
        _session_db=None,
        _conversation_root_id=lambda: "idle-owner",
        _turn_received_provider_response=False,
    )

    def fake_turn(owner, *_args, **_kwargs):
        assert calls == [("cancel", owner)]
        owner._turn_received_provider_response = True
        return result

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_turn)
    monkeypatch.setattr(
        runtime_heartbeat,
        "cancel_session",
        lambda owner: calls.append(("cancel", owner)),
    )
    monkeypatch.setattr(
        runtime_heartbeat,
        "arm_session_after_turn",
        lambda owner, turn_result: calls.append(("arm", owner, turn_result)),
    )
    monkeypatch.setattr(
        "agent.conversation_compression.compress_context",
        lambda _owner, messages, system_message, **_kwargs: (
            messages,
            system_message,
        ),
    )
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.switch_model",
        lambda *_args, **_kwargs: "switched",
    )

    assert AIAgent.run_conversation(agent, "hello") is result
    assert AIAgent._compress_context(
        agent, [], "system", commit_fence=object()
    ) == ([], "system")
    assert AIAgent.switch_model(agent, "new-model", "openai") == "switched"
    assert calls == [
        ("cancel", agent),
        ("arm", agent, result),
        ("cancel", agent),
        ("cancel", agent),
    ]


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


def test_active_snapshots_are_complete_stable_and_clear_on_cancel(
    monkeypatch, caplog
):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    wall_clock = SimpleNamespace(now=1_700_000_000.25)
    monkeypatch.setattr("tools.runtime_heartbeat.time.time", lambda: wall_clock.now)
    caplog.set_level("INFO", logger="tools.runtime_heartbeat")
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}

    assert manager.arm(
        "delegate-b",
        caller_id="owner-b",
        kind="delegation",
        interval=3300,
        inspect=alive,
    )
    wall_clock.now += 2
    assert manager.arm(
        "proc-a",
        caller_id="owner-a",
        kind="process",
        interval=1700,
        inspect=alive,
    )

    assert manager.active_snapshots() == [
        {
            "caller_id": "owner-b",
            "interval_s": 3300,
            "kind": "delegation",
            "last_success_at": None,
            "started_at": 1_700_000_000.25,
            "target_id": "delegate-b",
        },
        {
            "caller_id": "owner-a",
            "interval_s": 1700,
            "kind": "process",
            "last_success_at": None,
            "started_at": 1_700_000_002.25,
            "target_id": "proc-a",
        },
    ]

    assert manager.cancel("delegate-b") is True
    assert manager.active_snapshots() == [
        {
            "caller_id": "owner-a",
            "interval_s": 1700,
            "kind": "process",
            "last_success_at": None,
            "started_at": 1_700_000_002.25,
            "target_id": "proc-a",
        }
    ]
    assert manager.cancel("proc-a") is True
    assert manager.active_snapshots() == []
    assert "phase=cancel" in caplog.text


def test_active_snapshot_cadence_resets_after_periodic_rearm(monkeypatch):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    clock = SimpleNamespace(monotonic=100.0, wall=1_700_000_000.0)
    monkeypatch.setattr(
        "tools.runtime_heartbeat.time.monotonic", lambda: clock.monotonic
    )
    monkeypatch.setattr("tools.runtime_heartbeat.time.time", lambda: clock.wall)
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)

    assert manager.arm(
        "delegate",
        caller_id="owner",
        kind="delegation",
        interval=1700,
        inspect=lambda: {"alive": True, "progress": True},
    )
    assert manager.active_snapshots()[0]["started_at"] == 1_700_000_000.0

    clock.monotonic += 1700
    clock.wall += 1700
    FakeTimer.created[0].callback()

    assert manager.active_snapshots()[0]["started_at"] == 1_700_001_700.0


def test_active_snapshot_cadence_tracks_accepted_provider_activity(monkeypatch):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    clock = SimpleNamespace(monotonic=80.0, wall=1_700_000_000.0)
    monkeypatch.setattr(
        "tools.runtime_heartbeat.time.monotonic", lambda: clock.monotonic
    )
    monkeypatch.setattr("tools.runtime_heartbeat.time.time", lambda: clock.wall)
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}

    assert manager.arm(
        "delegate",
        caller_id="owner",
        kind="delegation",
        interval=1700,
        inspect=alive,
        provider="openai",
        cache_context="cache-a",
    )
    clock.monotonic = 100.0
    clock.wall += 20.0
    assert manager.reset_for_caller(
        "owner",
        provider="openai",
        cache_context="cache-a",
        activity_at=90.0,
    ) == 1
    accepted = manager.active_snapshots()[0]["started_at"]
    assert accepted == 1_700_000_010.0

    clock.monotonic = 400.0
    clock.wall += 300.0
    assert manager.reset_for_caller(
        "owner",
        provider="openai",
        cache_context="cache-a",
        activity_at=0.0,
    ) == 0
    assert manager.active_snapshots()[0]["started_at"] == accepted


def test_validated_alive_checkin_records_content_free_group_success(monkeypatch):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    clock = SimpleNamespace(monotonic=100.0, wall=1_700_000_000.0)
    monkeypatch.setattr(
        "tools.runtime_heartbeat.time.monotonic", lambda: clock.monotonic
    )
    monkeypatch.setattr("tools.runtime_heartbeat.time.time", lambda: clock.wall)
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}

    for target_id in ("a", "b"):
        assert manager.arm(
            target_id,
            caller_id="owner",
            kind="delegation",
            interval=1700,
            inspect=alive,
        )
    assert manager.arm(
        "foreign",
        caller_id="other-owner",
        kind="delegation",
        interval=1700,
        inspect=alive,
    )

    FakeTimer.created[0].callback()
    event = events.get_nowait()
    clock.wall += 2.0

    assert manager.record_checkin_success(event) is True
    snapshots = {item["target_id"]: item for item in manager.active_snapshots()}
    assert snapshots["a"]["last_success_at"] == clock.wall
    assert snapshots["b"]["last_success_at"] == clock.wall
    assert snapshots["foreign"]["last_success_at"] is None


def test_non_alive_or_cancelled_checkin_is_not_recorded():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    assert manager.arm(
        "target",
        caller_id="owner",
        kind="delegation",
        interval=1700,
        inspect=lambda: {"alive": True, "progress": True},
    )
    FakeTimer.created[0].callback()
    event = events.get_nowait()

    assert manager.record_checkin_success({**event, "status": "STUCK"}) is False
    assert manager.active_snapshots()[0]["last_success_at"] is None
    assert manager.cancel("target") is True
    assert manager.record_checkin_success(event) is False


def test_alive_fires_once_and_rearms_from_checkin_at_exact_interval(caplog):
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
    with caplog.at_level("INFO", logger="tools.runtime_heartbeat"):
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
    assert "phase=arm" in caplog.text
    assert "phase=due" in caplog.text


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

    event = events.get_nowait()
    assert event["status"] == "STUCK"
    assert manager.is_event_current(event) is True
    assert events.empty()
    assert len(FakeTimer.created) == 2
    assert FakeTimer.created[1].interval == 1700


@pytest.mark.parametrize("end_state", ["terminal", "cancelled"])
def test_stuck_live_event_is_suppressed_after_terminal_or_cancel(end_state):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    alive = {"value": True}
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    manager.arm(
        "proc",
        caller_id="owner",
        kind="process",
        interval=1700,
        inspect=lambda: {
            "alive": alive["value"],
            "output_size": 0,
            "cpu_seconds": 0.0,
        },
    )
    FakeTimer.created[0].callback()
    event = events.get_nowait()
    assert event["status"] == "STUCK"

    if end_state == "terminal":
        alive["value"] = False
    else:
        assert manager.cancel("proc") is True

    assert manager.is_event_current(event) is False


def test_due_targets_for_one_owner_coalesce_to_one_warm_event():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}
    manager.arm(
        "a", caller_id="owner", kind="delegation", interval=1700, inspect=alive
    )
    manager.arm(
        "b", caller_id="owner", kind="delegation", interval=1700, inspect=alive
    )
    first_a, first_b = FakeTimer.created

    first_a.callback()
    first_b.callback()

    assert events.qsize() == 1
    assert set(manager.outstanding_for_caller("owner")) == {"a", "b"}
    assert len(FakeTimer.created) == 4


def test_same_visible_session_keeps_distinct_agent_owners_and_warms_each(monkeypatch):
    from tools.runtime_heartbeat import (
        RuntimeHeartbeat,
        bind_agent_provider,
        reset_current_provider,
    )

    class ImmediateThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    class Owner:
        provider = "openai"
        requested_provider = "openai"
        base_url = "https://api.openai.invalid/v1"
        model = "model"
        api_mode = "chat_completions"

        def __init__(self):
            self.calls = []

        def run_conversation(self, message, **kwargs):
            self.calls.append((message, kwargs))
            return {
                "silent_noop": True,
                "heartbeat_warm_status": "warmed",
                "heartbeat_warm_reason": "physical_success",
            }

    monkeypatch.setattr("tools.runtime_heartbeat.threading.Thread", ImmediateThread)
    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    owners = [Owner(), Owner()]
    alive = lambda: {"alive": True, "progress": True}

    for index, owner in enumerate(owners):
        token = bind_agent_provider(owner)
        try:
            assert manager.arm(
                f"target-{index}",
                caller_id="visible-root-session",
                kind="delegation",
                interval=1700,
                inspect=alive,
            )
        finally:
            reset_current_provider(token)

    first, second = FakeTimer.created
    first.callback()
    second.callback()

    published = [events.get_nowait(), events.get_nowait()]
    assert {event["target_id"] for event in published} == {
        "target-0",
        "target-1",
    }
    assert [len(owner.calls) for owner in owners] == [1, 1]
    assert owners[0].calls[0][1]["heartbeat_event"]["target_id"] == "target-0"
    assert owners[1].calls[0][1]["heartbeat_event"]["target_id"] == "target-1"
    assert all(event["heartbeat_warm_owned"] is True for event in published)


def test_provider_activity_resets_only_the_exact_agent_owner(monkeypatch):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    class Owner:
        pass

    FakeTimer.created = []
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("tools.runtime_heartbeat.time.monotonic", lambda: clock.now)
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)
    owners = [Owner(), Owner()]
    alive = lambda: {"alive": True, "progress": True}
    for index, owner in enumerate(owners):
        assert manager.arm(
            f"target-{index}",
            caller_id="visible-root-session",
            kind="delegation",
            interval=1700,
            inspect=alive,
            provider="openai",
            cache_context="cache",
            owner=owner,
        )
    first, second = FakeTimer.created
    clock.now = 200.0

    assert manager.reset_for_caller(
        "visible-root-session",
        provider="openai",
        cache_context="cache",
        owner=owners[0],
    ) == 1
    assert first.cancelled is True
    assert second.cancelled is False


def test_recent_matching_provider_activity_postpones_same_owner_due_without_warm(
    monkeypatch,
):
    from agent.conversation_loop import _record_runtime_provider_activity
    from tools.runtime_heartbeat import (
        RuntimeHeartbeat,
        canonical_runtime_cache_context_identity,
        canonical_runtime_provider_identity,
    )

    class ImmediateThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    class Owner:
        provider = "openai"
        requested_provider = "openai"
        base_url = "https://api.openai.invalid/v1"
        model = "model"
        api_mode = "chat_completions"

        def __init__(self):
            self.calls = []

        def run_conversation(self, message, **kwargs):
            self.calls.append((message, kwargs))
            return {
                "heartbeat_warm_status": "warmed",
                "heartbeat_warm_reason": "physical_success",
            }

    monkeypatch.setattr("tools.runtime_heartbeat.threading.Thread", ImmediateThread)
    FakeTimer.created = []
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("tools.runtime_heartbeat.time.monotonic", lambda: clock.now)
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    owner = Owner()
    sibling = Owner()
    assert manager.arm(
        "target-owner",
        caller_id="visible-root-session",
        kind="delegation",
        interval=1700,
        inspect=lambda: {"alive": True, "progress": True},
        provider=canonical_runtime_provider_identity(owner),
        cache_context=canonical_runtime_cache_context_identity(owner),
        owner=owner,
    )
    assert manager.arm(
        "target-sibling",
        caller_id="visible-root-session",
        kind="delegation",
        interval=1700,
        inspect=lambda: {"alive": True, "progress": True},
        provider=canonical_runtime_provider_identity(sibling),
        cache_context=canonical_runtime_cache_context_identity(sibling),
        owner=sibling,
    )
    owner_timer = FakeTimer.created[0]
    sibling_timer = FakeTimer.created[1]
    owner_generation = manager._targets["target-owner"].generation
    sibling_generation = manager._targets["target-sibling"].generation

    monkeypatch.setattr("tools.runtime_heartbeat.runtime_heartbeat", manager)
    clock.now = 200.0
    _record_runtime_provider_activity(
        owner,
        clock.now,
        caller_id="visible-root-session",
    )

    owner_target = manager._targets["target-owner"]
    sibling_target = manager._targets["target-sibling"]
    assert owner_timer.cancelled is True
    assert sibling_timer.cancelled is False
    assert owner_target.generation > owner_generation
    assert sibling_target.generation == sibling_generation
    assert owner_target.deadline == 1900.0
    owner_timer.callback()
    assert owner.calls == []
    assert sibling.calls == []
    assert events.empty()


def test_successful_owned_warm_outcome_survives_its_lease_rearm(monkeypatch):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    class ImmediateThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    class Owner:
        provider = "openai"
        requested_provider = "openai"
        base_url = "https://api.openai.invalid/v1"
        model = "model"
        api_mode = "chat_completions"

        def run_conversation(self, _message, **kwargs):
            clock.now += 1.0
            assert manager.reset_for_caller(
                "visible-root-session",
                provider="openai",
                cache_context="cache",
                activity_at=clock.now,
                owner=self,
            ) == 1
            return {
                "heartbeat_warm_status": "warmed",
                "heartbeat_warm_reason": "physical_success",
            }

    monkeypatch.setattr("tools.runtime_heartbeat.threading.Thread", ImmediateThread)
    FakeTimer.created = []
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("tools.runtime_heartbeat.time.monotonic", lambda: clock.now)
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    owner = Owner()
    assert manager.arm(
        "target",
        caller_id="visible-root-session",
        kind="delegation",
        interval=1700,
        inspect=lambda: {"alive": True, "progress": True},
        provider="openai",
        cache_context="cache",
        owner=owner,
    )

    FakeTimer.created[0].callback()

    event = events.get_nowait()
    target = manager._targets["target"]
    assert target.generation != event["generation"]
    assert event["heartbeat_warm_capability"] == "warmed"
    assert event["heartbeat_warm_reason"] == "physical_success"
    assert target.warm_capability == "warmed"
    assert target.warm_reason == "physical_success"


def test_alive_coalescing_does_not_suppress_unhealthy_group_event():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    manager.arm(
        "alive",
        caller_id="owner",
        kind="delegation",
        interval=1700,
        inspect=lambda: {"alive": True, "progress": True},
    )
    manager.arm(
        "stuck",
        caller_id="owner",
        kind="delegation",
        interval=1700,
        inspect=lambda: {"alive": True, "progress": False},
    )
    first_alive, first_stuck = FakeTimer.created

    first_alive.callback()
    first_stuck.callback()

    assert [events.get_nowait()["status"], events.get_nowait()["status"]] == [
        "ALIVE",
        "STUCK",
    ]


def test_group_event_replaced_when_representative_completes():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}
    manager.arm(
        "a", caller_id="owner", kind="delegation", interval=1700, inspect=alive
    )
    manager.arm(
        "b", caller_id="owner", kind="delegation", interval=1700, inspect=alive
    )
    first_a, first_b = FakeTimer.created
    first_a.callback()
    queued = events.get_nowait()
    first_b.callback()

    assert manager.cancel("a") is True
    replacement = events.get_nowait()

    assert queued["target_id"] == "a"
    assert manager.is_event_current(queued) is False
    assert replacement["target_id"] == "b"
    assert manager.is_event_current(replacement) is True
    assert events.empty()
    assert manager.outstanding_for_caller("owner") == ["b"]
    assert manager.cancel("b") is True
    assert manager.is_event_current(replacement) is False


def test_consumed_group_event_does_not_publish_late_replacement():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}
    manager.arm(
        "a", caller_id="owner", kind="delegation", interval=1700, inspect=alive
    )
    manager.arm(
        "b", caller_id="owner", kind="delegation", interval=1700, inspect=alive
    )
    first_a, first_b = FakeTimer.created
    first_a.callback()
    event = events.get_nowait()
    first_b.callback()

    assert manager.is_event_current(event, consume=True) is True
    assert manager.cancel("a") is True
    assert events.empty()


def test_unhealthy_event_is_stale_when_exact_target_completes():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    manager.arm(
        "stuck", caller_id="owner", kind="delegation", interval=1700,
        inspect=lambda: {"alive": True, "progress": False},
    )
    manager.arm(
        "alive", caller_id="owner", kind="delegation", interval=1700,
        inspect=lambda: {"alive": True, "progress": True},
    )
    FakeTimer.created[0].callback()
    event = events.get_nowait()

    assert event["status"] == "STUCK"
    assert manager.cancel("stuck") is True
    assert manager.is_event_current(event) is False
    assert manager.outstanding_for_caller("owner") == ["alive"]


def test_event_generation_is_stale_after_target_rearm():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    manager.arm(
        "target", caller_id="owner", kind="delegation", interval=1700,
        inspect=lambda: {"alive": True, "progress": False},
    )
    FakeTimer.created[0].callback()
    event = events.get_nowait()

    assert manager.is_event_current(event) is True
    assert manager.reset_for_caller("owner") == 1
    assert manager.is_event_current(event) is False


def test_reused_target_id_cannot_revive_old_generation():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}
    manager.arm(
        "target", caller_id="owner", kind="delegation", interval=1700,
        inspect=lambda: {"alive": True, "progress": False},
    )
    manager.arm(
        "sibling", caller_id="owner", kind="delegation", interval=1700,
        inspect=alive,
    )
    FakeTimer.created[0].callback()
    event = events.get_nowait()

    manager.arm(
        "target", caller_id="owner", kind="delegation", interval=1700,
        inspect=alive,
    )

    assert manager.is_event_current(event) is False


def test_same_owner_distinct_provider_cache_contexts_do_not_coalesce():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}
    manager.arm(
        "openai-target",
        caller_id="owner",
        kind="delegation",
        interval=1700,
        inspect=alive,
        provider="openai",
        cache_context="openai:model-a:https://one.invalid",
    )
    manager.arm(
        "anthropic-target",
        caller_id="owner",
        kind="delegation",
        interval=1700,
        inspect=alive,
        provider="anthropic",
        cache_context="anthropic:model-b:https://two.invalid",
    )

    first, second = FakeTimer.created
    first.callback()
    second.callback()
    published = [events.get_nowait(), events.get_nowait()]

    assert {
        (event["provider"], event["cache_context"]) for event in published
    } == {
        ("openai", "openai:model-a:https://one.invalid"),
        ("anthropic", "anthropic:model-b:https://two.invalid"),
    }
    assert all(len(event["target_ids"]) == 1 for event in published)


def test_event_is_revalidated_against_executing_agent_cache_context():
    from tools.runtime_heartbeat import (
        RuntimeHeartbeat,
        canonical_runtime_cache_context_identity,
        canonical_runtime_provider_identity,
    )

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    agent = SimpleNamespace(
        provider="openai",
        requested_provider="openai",
        model="model-a",
        base_url="https://api.example/v1/",
        api_mode="chat_completions",
    )
    manager.arm(
        "target",
        caller_id="owner",
        kind="process",
        interval=1700,
        inspect=lambda: {"alive": True, "progress": True},
        provider=canonical_runtime_provider_identity(agent),
        cache_context=canonical_runtime_cache_context_identity(agent),
    )
    FakeTimer.created[0].callback()
    event = events.get_nowait()

    assert manager.is_event_current(event, agent=agent) is True
    agent.model = "model-b"
    assert manager.is_event_current(event, agent=agent) is False


def test_queued_owner_event_is_rejected_after_target_completion(monkeypatch):
    from tools.async_delegation import claim_event_delivery
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)
    monkeypatch.setattr("tools.runtime_heartbeat.runtime_heartbeat", manager)
    manager.arm(
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
    event = events.get_nowait()
    assert claim_event_delivery(event, "test") == ""

    assert manager.cancel("proc") is True

    assert claim_event_delivery(event, "test") is None


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


def test_provider_activity_resets_only_exact_group_from_dispatch_time(monkeypatch):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    clock = SimpleNamespace(now=80.0)
    monkeypatch.setattr(
        "tools.runtime_heartbeat.time.monotonic", lambda: clock.now
    )
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}
    manager.arm(
        "a",
        caller_id="owner",
        kind="delegation",
        interval=1700,
        inspect=alive,
        provider="openai",
        cache_context="cache-a",
    )
    manager.arm(
        "b",
        caller_id="owner",
        kind="delegation",
        interval=1700,
        inspect=alive,
        provider="openai",
        cache_context="cache-b",
    )
    old_a, old_b = FakeTimer.created

    clock.now = 100.0
    assert manager.reset_for_caller(
        "owner",
        provider="openai",
        cache_context="cache-a",
        activity_at=90.0,
    ) == 1
    assert old_a.cancelled is True
    assert old_b.cancelled is False
    assert FakeTimer.created[-1].interval == 1690.0


def test_late_older_activity_is_noop_for_exact_mixed_interval_groups(monkeypatch):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "tools.runtime_heartbeat.time.monotonic", lambda: clock.now
    )
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}
    for target_id, interval in (("short", 1700), ("long", 3300)):
        manager.arm(
            target_id,
            caller_id="owner",
            kind="delegation",
            interval=interval,
            inspect=alive,
            provider="openai",
            cache_context="cache-a",
        )

    clock.now = 300.0
    assert manager.reset_for_caller(
        "owner",
        provider="openai",
        cache_context="cache-a",
        activity_at=250.0,
    ) == 2

    accepted = {}
    for target_id in ("short", "long"):
        target = manager._targets[target_id]
        group = manager._group_key(target)
        pending = {"target_id": target_id, "accepted": True}
        replacement = {"target_id": target_id, "replacement": True}
        manager._group_pending[group] = pending
        manager._group_replacements[group] = replacement
        accepted[target_id] = (
            target,
            target.timer,
            target.deadline,
            target.generation,
            manager._group_tokens[group],
            manager._group_next_emit[group],
            pending,
            replacement,
        )

    clock.now = 400.0
    assert manager.reset_for_caller(
        "owner",
        provider="openai",
        cache_context="cache-a",
        activity_at=200.0,
    ) == 0

    for target_id, state in accepted.items():
        target, timer, deadline, generation, token, next_emit, pending, replacement = state
        group = manager._group_key(target)
        assert manager._targets[target_id] is target
        assert target.timer is timer
        assert timer.cancelled is False
        assert target.deadline == deadline
        assert target.generation == generation
        assert manager._group_tokens[group] == token
        assert manager._group_next_emit[group] == next_emit
        assert manager._group_pending[group] is pending
        assert manager._group_replacements[group] is replacement


def test_late_activity_cannot_reset_reused_target_generation(monkeypatch):
    from tools.runtime_heartbeat import RuntimeHeartbeat

    FakeTimer.created = []
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "tools.runtime_heartbeat.time.monotonic", lambda: clock.now
    )
    manager = RuntimeHeartbeat(event_queue=queue.Queue(), timer_factory=FakeTimer)
    alive = lambda: {"alive": True, "progress": True}
    arm = lambda: manager.arm(
        "target",
        caller_id="owner",
        kind="delegation",
        interval=1700,
        inspect=alive,
        provider="openai",
        cache_context="cache-a",
    )
    assert arm() is True

    clock.now = 300.0
    assert manager.cancel("target") is True
    assert arm() is True
    target = manager._targets["target"]
    group = manager._group_key(target)
    pending = {"target_id": "target", "accepted": True}
    replacement = {"target_id": "target", "replacement": True}
    manager._group_next_emit[group] = target.deadline
    manager._group_pending[group] = pending
    manager._group_replacements[group] = replacement
    timer = target.timer
    deadline = target.deadline
    generation = target.generation
    token = manager._group_tokens[group]

    clock.now = 400.0
    assert manager.reset_for_caller(
        "owner",
        provider="openai",
        cache_context="cache-a",
        activity_at=150.0,
    ) == 0
    assert manager._targets["target"] is target
    assert target.timer is timer
    assert timer.cancelled is False
    assert target.deadline == deadline
    assert target.generation == generation
    assert manager._group_tokens[group] == token
    assert manager._group_next_emit[group] == deadline
    assert manager._group_pending[group] is pending
    assert manager._group_replacements[group] is replacement


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
    assert manager.outstanding_for_caller("owner") == ["delegate"]
    assert manager.is_event_current(event) is True
    assert manager.cancel("delegate") is True
    assert manager.is_event_current(event) is False
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
    session._tracked_descendants[300] = 30
    monkeypatch.setattr(process_registry, "get", lambda _target_id: session)
    monkeypatch.setattr(
        process_registry, "_remember_local_descendants", lambda _s, **_kwargs: None
    )
    monkeypatch.setattr(
        process_registry,
        "_safe_host_start_time",
        lambda pid: {100: 10, 200: 20, 300: 31}[pid],
    )
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: SimpleNamespace(
            cpu_times=lambda: SimpleNamespace(
                user={100: 1.0, 200: 4.0}[pid], system=0.5
            )
        ),
    )

    snapshot = inspect_process("host")

    assert snapshot["cpu_seconds"] == 6.0
    assert snapshot["cpu_by_identity"] == {(100, 10): 1.5, (200, 20): 4.5}

    monkeypatch.setattr(
        process_registry,
        "_safe_host_start_time",
        lambda pid: {100: 11, 200: 20, 300: 31}[pid],
    )
    assert inspect_process("host")["cpu_seconds"] == 0.0


def test_cpu_progress_survives_a_high_cpu_child_exiting():
    from tools.runtime_heartbeat import RuntimeHeartbeat

    target = SimpleNamespace(
        kind="process",
        baseline={
            "output_size": 0,
            "cpu_seconds": 5.1,
            "cpu_by_identity": {(100, 10): 0.1, (200, 20): 5.0},
        },
    )
    snapshot = {
        "alive": True,
        "output_size": 0,
        "cpu_seconds": 0.2,
        "cpu_by_identity": {(100, 10): 0.2},
    }

    status, evidence = RuntimeHeartbeat._assess(target, snapshot)

    assert status == "ALIVE"
    assert "CPU advanced" in evidence


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="local terminal subreaper lifecycle is Linux-only",
)
def test_linux_subreaper_descendant_cpu_keeps_heartbeat_alive(
    monkeypatch, tmp_path
):
    import psutil
    from tools import process_registry as process_registry_module
    from tools.process_registry import ProcessRegistry
    from tools.runtime_heartbeat import RuntimeHeartbeat, inspect_process

    registry = ProcessRegistry()
    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    monkeypatch.setattr(
        process_registry_module, "CHECKPOINT_PATH", tmp_path / "processes.json"
    )
    stop = tmp_path / "stop-worker"
    marker = f"hermes-heartbeat-worker:{stop}"
    worker_code = (
        "import pathlib,sys; marker=sys.argv[1]; stop=pathlib.Path(sys.argv[2]); "
        'exec("n=0\\nwhile not stop.exists():\\n for _ in range(100000): n += 1")'
    )
    expected_argv = [sys.executable, "-c", worker_code, marker, str(stop)]
    command = shlex.join(expected_argv)
    session = registry.spawn_local(command, cwd=str(tmp_path))
    events = queue.Queue()
    FakeTimer.created = []
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=FakeTimer)

    try:
        assert session._subreaper_managed is True

        def _worker_process():
            try:
                return next(
                    (
                        child
                        for child in psutil.Process(session.pid).children(recursive=True)
                        if child.cmdline() == expected_argv
                    ),
                    None,
                )
            except (psutil.Error, OSError):
                return None

        worker = _wait_until(_worker_process)
        assert worker is not None, "CPU worker did not enter the subreaper tree"
        worker_start = registry._safe_host_start_time(worker.pid)
        assert worker_start is not None

        assert manager.arm(
            session.id,
            caller_id="owner",
            kind="process",
            interval=1700,
            inspect=lambda: inspect_process(session.id),
        )
        baseline = dict(manager._targets[session.id].baseline)
        baseline_worker_cpu = baseline["cpu_by_identity"].get(
            (worker.pid, worker_start), 0.0
        )

        def _advanced_worker_cpu():
            if registry._safe_host_start_time(worker.pid) != worker_start:
                return None
            try:
                cpu = worker.cpu_times()
            except psutil.Error:
                return None
            total = float(cpu.user + cpu.system)
            return total if total > baseline_worker_cpu else None

        worker_cpu = _wait_until(_advanced_worker_cpu)
        assert worker_cpu is not None, "CPU worker made no measurable progress"
        assert registry._safe_host_start_time(worker.pid) == worker_start
        assert baseline["output_size"] == session.output_size == 0

        snapshot = inspect_process(session.id)
        FakeTimer.created[0].callback()
        event = events.get_nowait()

        assert event["status"] == "ALIVE"
        assert "CPU advanced" in event["evidence"]
        assert snapshot["cpu_seconds"] >= worker_cpu
    finally:
        manager.cancel(session.id)
        stop.touch()
        if not session._completion_event.wait(10):
            registry.kill_process(session.id)
        assert session._completion_event.wait(10)
        session._reader_thread.join(10)
        assert not session._reader_thread.is_alive()


def test_queued_delegation_is_alive(monkeypatch):
    from tools.runtime_heartbeat import inspect_delegation

    monkeypatch.setattr(
        "tools.async_delegation.list_async_delegations",
        lambda: [{"delegation_id": "d", "status": "queued"}],
    )

    assert inspect_delegation("d") == {
        "alive": True,
        "progress": True,
        "evidence": "delegation in progress; status=queued",
    }


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
