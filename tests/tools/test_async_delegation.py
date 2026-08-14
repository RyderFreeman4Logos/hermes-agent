"""Tests for async (background) delegation — tools/async_delegation.py.

Covers the dispatch handle, non-blocking behavior, completion-event delivery
onto the shared process_registry.completion_queue, the rich re-injection block
formatting, capacity rejection, and crash handling.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry, format_process_notification


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    # Give just-released workers a beat to finalize BEFORE draining, so their
    # completion events land now instead of leaking into the next test's
    # queue (worker threads push events asynchronously; a drain that races an
    # in-flight _finalize misses it).
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _drain_one(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.02)
    return None


def _drain_for(delegation_id, timeout=5.0):
    """Drain until the event for *delegation_id* appears (discarding others).

    Completion events are pushed asynchronously by worker threads, so a
    straggler from a PREVIOUS test can land after that test's teardown drain
    and leak into the current test's queue. Matching on delegation_id makes
    the assertion immune to that cross-test leak.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("delegation_id") == delegation_id:
                return evt
            continue
        time.sleep(0.02)
    return None


def test_active_for_session_counts_every_live_delegation_state():
    with ad._records_lock:
        ad._records.update(
            {
                "running": {
                    "status": "running",
                    "origin_ui_session_id": "desktop-sid",
                },
                "stalling": {
                    "status": "stalling",
                    "origin_ui_session_id": "desktop-sid",
                },
                "finalizing": {
                    "status": "finalizing",
                    "origin_ui_session_id": "desktop-sid",
                },
                "completed": {
                    "status": "completed",
                    "origin_ui_session_id": "desktop-sid",
                },
                "other-session": {
                    "status": "running",
                    "origin_ui_session_id": "other-sid",
                },
            }
        )

    assert ad.active_for_session("desktop-sid") == 3
    assert ad.active_for_session("other-sid") == 1
    assert ad.active_for_session("") == 0


def test_child_control_receipt_is_durable_and_generation_fenced(tmp_path, monkeypatch):
    """Queue/interrupt reservations survive retries without double delivery."""
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(ad, "_db_path", lambda: db_path)
    with ad._transaction() as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, state, dispatched_at, updated_at,
                children_json)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (
                "deleg-control",
                "parent",
                time.time(),
                time.time(),
                json.dumps([{"subagent_id": "child-1", "session_id": "db-child"}]),
            ),
        )

    first = ad.reserve_child_control("deleg-control", "child-1", "queue", "next")
    retry = ad.reserve_child_control("deleg-control", "child-1", "queue", "next")
    conflict = ad.reserve_child_control("deleg-control", "child-1", "interrupt", "replace")
    assert first["state"] == "accepted"
    assert first["status"] == "accepted"
    assert retry["generation"] == first["generation"]
    assert conflict["status"] == "pending"

    ad._reset_for_tests()
    claimed = ad.claim_child_control("deleg-control", "child-1")
    assert claimed["generation"] == first["generation"]
    assert ad.claim_child_control("deleg-control", "child-1") is None
    monkeypatch.setattr(ad, "_CONTROL_PROCESS_TOKEN", "restarted-process")
    reclaimed = ad.claim_child_control("deleg-control", "child-1")
    assert reclaimed is not None
    assert reclaimed["generation"] == first["generation"]
    assert ad.finish_child_control("deleg-control", "child-1", first["generation"])
    assert ad.reserve_child_control("deleg-control", "child-1", "queue", "next")["state"] == "delivered"


def test_dispatch_returns_immediately_without_blocking():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "done", "api_calls": 1,
                "duration_seconds": 0.1, "model": "m"}

    t0 = time.monotonic()
    res = ad.dispatch_async_delegation(
        goal="g", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=3,
    )
    elapsed = time.monotonic() - t0

    assert res["status"] == "dispatched"
    assert res["delegation_id"].startswith("deleg_")
    # Non-blocking invariant: dispatch returned while the runner is still
    # gated (active), so it cannot have waited on the gate. The active_count
    # check is the environment-independent proof; the generous wall-clock
    # bound is a loose sanity backstop, not the primary assertion (a loaded
    # CI runner can be slow but never anywhere near the runner's 5s gate).
    assert ad.active_count() == 1
    assert elapsed < 4.0, f"dispatch blocked {elapsed:.2f}s (gate is 5s)"
    gate.set()


def test_async_executor_workers_are_daemon_threads():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "done"}

    res = ad.dispatch_async_delegation(
        goal="daemon check", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=1,
    )
    assert res["status"] == "dispatched"

    deadline = time.monotonic() + 2
    worker = None
    while time.monotonic() < deadline:
        worker = next(
            (t for t in threading.enumerate() if t.name.startswith("async-delegate")),
            None,
        )
        if worker is not None:
            break
        time.sleep(0.02)
    assert worker is not None
    assert worker.daemon is True
    gate.set()
    assert _drain_one() is not None


def test_completion_event_lands_on_shared_queue_with_session_key():
    def runner():
        return {"status": "completed", "summary": "the result",
                "api_calls": 3, "duration_seconds": 2.0, "model": "test-model"}

    res = ad.dispatch_async_delegation(
        goal="compute X", context="some context", toolsets=["web", "file"],
        role="leaf", model="test-model", session_key="agent:main:cli:dm:local",
        parent_session_id="20260703_parent_sid",
        runner=runner, max_async_children=3,
    )
    assert res["status"] == "dispatched"

    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["summary"] == "the result"
    assert evt["session_key"] == "agent:main:cli:dm:local"
    assert evt["parent_session_id"] == "20260703_parent_sid"
    assert evt["delegation_id"] == res["delegation_id"]


def test_batch_worker_exit_after_tool_progress_emits_one_terminal_failure():
    """A lost continuation after tool progress must not orphan the batch."""
    progress_seen = threading.Event()

    def runner():
        progress_seen.set()
        raise SystemExit("continuation exited after successful tool progress")

    res = ad.dispatch_async_delegation_batch(
        goals=["recover the completed gate result"],
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner-session",
        parent_session_id="owner-parent",
        runner=runner,
        max_async_children=1,
    )
    assert res["status"] == "dispatched"
    assert progress_seen.wait(timeout=2)

    evt = _drain_for(res["delegation_id"])
    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["is_batch"] is True
    assert evt["status"] == "error"
    assert evt["error"] == (
        "SystemExit: continuation exited after successful tool progress"
    )

    # The worker-exit path is terminal exactly once.
    assert _drain_for(res["delegation_id"], timeout=0.2) is None


def test_rich_reinjection_block_is_self_contained():
    def runner():
        return {"status": "completed", "summary": "The answer is 42.",
                "api_calls": 7, "duration_seconds": 3.5, "model": "test-model"}

    ad.dispatch_async_delegation(
        goal="Compute the meaning of life",
        context="User is a philosopher. Respond tersely.",
        toolsets=["web"], role="leaf", model="test-model",
        session_key="", runner=runner, max_async_children=3,
    )
    evt = _drain_one()
    assert evt is not None
    text = format_process_notification(evt)
    assert text is not None
    for needle in [
        "ASYNC DELEGATION COMPLETE",
        "Compute the meaning of life",
        "User is a philosopher",
        "Toolsets: web",
        "The answer is 42.",
        "Status: completed",
        "API calls: 7",
    ]:
        assert needle in text, f"missing {needle!r}"


def test_dispatch_rejected_at_capacity():
    ev = threading.Event()

    def blocker():
        ev.wait(timeout=60)
        return {"status": "completed", "summary": "x"}

    for i in range(2):
        r = ad.dispatch_async_delegation(
            goal=f"task{i}", context=None, toolsets=None, role="leaf",
            model="m", session_key="", runner=blocker, max_async_children=2,
        )
        assert r["status"] == "dispatched"

    r3 = ad.dispatch_async_delegation(
        goal="task3", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=blocker, max_async_children=2,
    )
    assert r3["status"] == "rejected"
    assert "capacity reached" in r3["error"]
    ev.set()


def test_interrupt_all_signals_running_children():
    ev = threading.Event()
    interrupted = {"count": 0}
    # No short internal timeout: the blocker holds until interrupt_fn fires.
    # The old ev.wait(timeout=5) made this test a change-detector for CI
    # worker load — on a CPU-starved runner the 5s expired before
    # interrupt_all() ran, the record finalized, and interrupt_all() found
    # nothing running (n == 0). The pytest-level timeout is the real
    # runaway guard.

    def blocker():
        ev.wait(timeout=60)
        return {"status": "interrupted", "summary": None,
                "error": "cancelled"}

    def interrupt_fn():
        interrupted["count"] += 1
        ev.set()

    r = ad.dispatch_async_delegation(
        goal="long task", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=blocker,
        interrupt_fn=interrupt_fn, max_async_children=3,
    )
    n = ad.interrupt_all(reason="test")
    assert n == 1
    assert interrupted["count"] == 1
    # child still emits a completion event after interrupt. Match on THIS
    # delegation's id — straggler 'completed' events from a previous test's
    # workers can finalize after that test's teardown drain and leak into
    # this queue (observed on loaded CI workers).
    evt = _drain_for(r["delegation_id"])
    assert evt is not None
    assert evt["status"] == "interrupted"


def _fast_stale_monitor(monkeypatch, *, idle=0.15, in_tool=0.3, grace=0.15):
    """Shrink the stale-monitor cadence so tests run in milliseconds."""
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", idle)
    monkeypatch.setattr(ad, "_STALE_IN_TOOL_SECONDS", in_tool)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", grace)


def test_stalled_runner_is_interrupted_then_finalized(monkeypatch):
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()
    interrupted = {"count": 0}

    def stuck_runner():
        gate.wait(timeout=10)
        return {"status": "completed", "summary": "too late"}

    def interrupt_fn():
        interrupted["count"] += 1

    res = ad.dispatch_async_delegation(
        goal="stuck child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=stuck_runner,
        interrupt_fn=interrupt_fn, max_async_children=1,
        # Frozen progress token: the child never advances an API call.
        progress_fn=lambda: ((0, None), False),
    )
    assert res["status"] == "dispatched"

    evt = _drain_for(res["delegation_id"], timeout=5.0)
    try:
        assert evt is not None
        assert evt["type"] == "async_delegation"
        assert evt["status"] == "stalled"
        assert evt["delegation_id"] == res["delegation_id"]
        assert evt["api_calls"] == 0
        assert "stalled" in evt["error"]
        # Interrupt was requested BEFORE force-finalization (grace window).
        assert interrupted["count"] >= 1
        assert ad.active_count() == 0
    finally:
        gate.set()

    # If the ignored runner eventually returns, it must not enqueue a second
    # completion for a delegation the monitor already finalized.
    assert _drain_one(timeout=0.5) is None


def test_progressing_runner_is_never_stalled(monkeypatch):
    """A child that keeps advancing is left alone no matter how long it runs."""
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()
    ticks = {"n": 0}

    def slow_but_alive_runner():
        gate.wait(timeout=10)
        return {"status": "completed", "summary": "done", "api_calls": 7}

    def progress_fn():
        # Token advances on every sample — simulates a child making steady
        # API-call progress.
        ticks["n"] += 1
        return (ticks["n"], None), False

    res = ad.dispatch_async_delegation(
        goal="slow child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=slow_but_alive_runner,
        max_async_children=1, progress_fn=progress_fn,
    )
    assert res["status"] == "dispatched"

    # Run well past the (shrunk) idle threshold — several monitor sweeps.
    time.sleep(0.6)
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()

    gate.set()
    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "completed"
    assert evt["summary"] == "done"


def test_stalling_runner_that_honors_interrupt_keeps_its_result(monkeypatch):
    """Interrupt-responsive children finalize through the NORMAL path.

    The monitor's interrupt gives a wedged-looking child a grace window; if
    the runner returns during it, the real result (partial work, api_calls)
    is delivered instead of a synthetic stalled event.
    """
    _fast_stale_monitor(monkeypatch, grace=5.0)
    interrupted = threading.Event()

    def runner():
        # "Wedged" until interrupted, then unwinds and reports partial work.
        interrupted.wait(timeout=10)
        return {
            "status": "interrupted",
            "summary": "partial work saved",
            "api_calls": 3,
        }

    res = ad.dispatch_async_delegation(
        goal="responsive child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=runner,
        interrupt_fn=interrupted.set, max_async_children=1,
        progress_fn=lambda: ((3, None), False),
    )
    assert res["status"] == "dispatched"

    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "interrupted"
    assert evt["summary"] == "partial work saved"
    assert evt["api_calls"] == 3
    assert ad.active_count() == 0


def test_streaming_child_counts_as_alive(monkeypatch):
    """A child mid-stream (api_call_count frozen, last_activity_ts ticking)
    must never be stalled — streamed chunks tick _touch_activity, and the
    progress token includes that timestamp (same liveness signal as the
    compaction inactivity budget, PR #71508)."""
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()
    now = {"ts": 1000.0}

    def progress_fn():
        # api_call_count and current_tool frozen (long streaming response in
        # flight), but the activity timestamp advances with every chunk.
        now["ts"] += 1.0
        return ((1, None, now["ts"]),), False

    res = ad.dispatch_async_delegation(
        goal="streaming child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", max_async_children=1,
        runner=lambda: (gate.wait(timeout=10), {"status": "completed", "summary": "streamed"})[1],
        progress_fn=progress_fn,
    )
    assert res["status"] == "dispatched"

    time.sleep(0.6)  # several sweeps past the shrunk idle threshold
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()

    gate.set()
    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "completed"


def test_stalled_event_carries_structured_stall_metadata(monkeypatch):
    """The terminal stalled event must expose machine-readable stall context
    (#51690) — quiet duration, tripped threshold, phase, grace — mirroring
    the sync path's timeout_seconds/timed_out_after_seconds/timeout_phase."""
    _fast_stale_monitor(monkeypatch)
    gate = threading.Event()

    res = ad.dispatch_async_delegation(
        goal="stall metadata", context=None, toolsets=None, role="leaf",
        model="m", session_key="", max_async_children=1,
        runner=lambda: {} if gate.wait(timeout=10) else {},
        progress_fn=lambda: ((0, "terminal"), True),
    )
    assert res["status"] == "dispatched"

    evt = _drain_for(res["delegation_id"], timeout=5.0)
    try:
        assert evt is not None
        assert evt["status"] == "stalled"
        assert evt["stalled_after_quiet_seconds"] >= 0.3  # in-tool threshold
        assert evt["stall_threshold_seconds"] == ad._STALE_IN_TOOL_SECONDS
        assert evt["stall_phase"] == "in_tool"
        assert evt["stall_grace_seconds"] == ad._STALL_GRACE_SECONDS
    finally:
        gate.set()


def test_list_async_delegations_exposes_live_activity(monkeypatch):
    """list_async_delegations must expose per-child live activity sampled
    from progress_fn plus seconds_since_progress, for /agents UIs (#51690)."""
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.03)
    gate = threading.Event()
    base_ts = time.time() - 12.0

    res = ad.dispatch_async_delegation(
        goal="live listing", context=None, toolsets=None, role="leaf",
        model="m", session_key="", max_async_children=1,
        runner=lambda: {} if gate.wait(timeout=10) else {},
        progress_fn=lambda: (((3, "web_search", base_ts),), True),
    )
    try:
        time.sleep(0.1)  # let the monitor stamp _progress_ts at least once
        item = next(
            d for d in ad.list_async_delegations()
            if d["delegation_id"] == res["delegation_id"]
        )
        assert item["status"] == "running"
        assert item["in_tool"] is True
        assert "seconds_since_progress" in item
        (child,) = item["children_activity"]
        assert child["api_calls"] == 3
        assert child["current_tool"] == "web_search"
        assert 10.0 <= child["seconds_since_activity"] <= 20.0
        # Callables and private bookkeeping must never leak.
        assert "progress_fn" not in item
        assert "interrupt_fn" not in item
        assert not any(k.startswith("_") for k in item)
    finally:
        gate.set()


def test_in_tool_stall_uses_higher_threshold(monkeypatch):
    """A frozen child inside a tool gets the in-tool ceiling, not the idle one."""
    _fast_stale_monitor(monkeypatch, idle=0.1, in_tool=10.0, grace=0.1)
    gate = threading.Event()

    def runner():
        gate.wait(timeout=10)
        return {"status": "completed", "summary": "long tool finished"}

    res = ad.dispatch_async_delegation(
        goal="long tool child", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=runner, max_async_children=1,
        # Frozen token but in_tool=True — a legitimately slow terminal
        # command / web fetch. Must NOT be stalled at the idle threshold.
        progress_fn=lambda: ((1, "terminal"), True),
    )
    assert res["status"] == "dispatched"

    time.sleep(0.5)  # far past idle threshold, well under in-tool threshold
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()

    gate.set()
    evt = _drain_for(res["delegation_id"], timeout=5.0)
    assert evt is not None
    assert evt["status"] == "completed"


def test_real_process_restart_restores_owned_completion_once(tmp_path):
    """Real-import E2E: a fresh interpreter restores a prior process's result."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": repo}
    producer = r'''
import time
from tools import async_delegation as ad
r = ad.dispatch_async_delegation(
    goal="restart", context=None, toolsets=None, role="leaf", model="m",
    session_key="owner-session", parent_session_id="durable-parent",
    runner=lambda: {"status": "completed", "summary": "after restart"},
)
deadline = time.time() + 5
while ad.active_count() and time.time() < deadline:
    time.sleep(.01)
print(r["delegation_id"])
'''
    first = subprocess.run(
        [sys.executable, "-c", producer], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    delegation_id = first.stdout.strip().splitlines()[-1]

    consumer = r'''
import json
from tools.process_registry import process_registry
evt = process_registry.completion_queue.get_nowait()
print(json.dumps(evt, sort_keys=True))
'''
    second = subprocess.run(
        [sys.executable, "-c", consumer], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    evt = json.loads(second.stdout.strip().splitlines()[-1])
    assert evt["delegation_id"] == delegation_id
    assert evt["session_key"] == "owner-session"
    assert evt["parent_session_id"] == "durable-parent"
    assert evt["summary"] == "after restart"

    acker = f'''
from tools import async_delegation as ad
assert ad.mark_completion_delivered({delegation_id!r})
'''
    subprocess.run(
        [sys.executable, "-c", acker], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    probe = subprocess.run(
        [sys.executable, "-c", "from tools.process_registry import process_registry; print(process_registry.completion_queue.qsize())"],
        cwd=repo, env=env, text=True, capture_output=True, timeout=15, check=True,
    )
    assert probe.stdout.strip().splitlines()[-1] == "0"


# ---------------------------------------------------------------------------
# Integration: delegate_task(background=True) routing
# ---------------------------------------------------------------------------

def test_delegate_task_background_routes_async_and_does_not_block(monkeypatch):
    """delegate_task(background=True) returns a handle without running the
    child synchronously, and the child completes on the background thread.
    A single task is dispatched as a one-item background batch unit."""
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    gate = threading.Event()

    def slow_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=60)  # a sync impl would hang delegate_task here
        return {
            "task_index": 0, "status": "completed", "summary": f"done: {goal}",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        }

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    # monkeypatch (not `with`) so patches outlive delegate_task's return and
    # remain active while the background worker runs.
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", slow_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    out = dt.delegate_task(
        goal="the real task", context="ctx",
        background=True, parent_agent=parent,
    )

    import json
    parsed = json.loads(out)
    assert parsed["status"] == "dispatched"
    assert parsed["mode"] == "background"
    assert parsed["delegation_id"].startswith("deleg_")
    # Non-blocking invariant: delegate_task returned while the child is STILL
    # blocked on the closed gate, so no completion event exists yet.
    assert process_registry.completion_queue.empty()
    assert ad.active_count() == 1  # one background batch unit, not finished

    gate.set()
    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    # Single task rides the batch path → carries a 1-item results list.
    assert evt.get("is_batch") is True
    assert len(evt["results"]) == 1
    assert evt["results"][0]["summary"] == "done: the real task"
    text = format_process_notification(evt)
    assert text is not None
    assert "the real task" in text


def test_background_batch_registers_heartbeat_before_an_immediate_worker_finishes(monkeypatch):
    """A terminal batch must remove its heartbeat child before dispatch returns."""
    from unittest.mock import MagicMock
    import tools.delegate_tool as dt
    import tools.runtime_heartbeat as rh

    class ImmediateExecutor:
        def submit(self, fn):
            fn()

    class Timer:
        def __init__(self, _delay, _callback):
            self.cancelled = False

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    heartbeat = rh.RuntimeHeartbeat(
        config={"providers": {"custom:localrouter": 30}}, timer_factory=Timer
    )
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "heartbeat-parent"
    parent.provider = "custom"
    parent.requested_provider = "custom:localrouter"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    child = MagicMock()
    child._delegate_role = "leaf"

    monkeypatch.setattr(ad, "_get_executor", lambda _limit: ImmediateExecutor())
    monkeypatch.setattr(rh, "runtime_heartbeat", heartbeat)
    monkeypatch.setattr(dt, "_build_child_agent", lambda **_kwargs: child)
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda *_args, **_kwargs: {
            "model": "m", "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "command": None, "args": None,
        },
    )
    monkeypatch.setattr(
        dt,
        "_run_single_child",
        lambda *_args, **_kwargs: {
            "task_index": 0, "status": "completed", "summary": "done",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        },
    )

    assert json.loads(
        dt.delegate_task(goal="instant task", background=True, parent_agent=parent)
    )["status"] == "dispatched"
    assert heartbeat.timer_for(parent) is None


def test_background_batch_unwinds_heartbeat_when_submit_fails(monkeypatch):
    """Rejected scheduling must not retain pre-submit heartbeat ownership."""
    import tools.runtime_heartbeat as rh

    class FailingExecutor:
        def submit(self, _fn):
            raise RuntimeError("submit failed")

    heartbeat = rh.RuntimeHeartbeat(
        config={"providers": {"custom:localrouter": 30}}, timer_factory=_Timer
    )
    parent = type("Parent", (), {
        "provider": "custom",
        "requested_provider": "custom:localrouter",
        "session_id": "heartbeat-parent",
    })()
    monkeypatch.setattr(ad, "_get_executor", lambda _limit: FailingExecutor())

    result = ad.dispatch_async_delegation_batch(
        goals=["task"], context=None, toolsets=None, role="leaf", model="m",
        session_key="", parent_session_id="heartbeat-parent", runner=lambda: {},
        before_submit=lambda delegation_id: heartbeat.register_child(
            parent, "subagent", delegation_id
        ),
        on_submit_failure=lambda delegation_id: heartbeat.complete_child(
            parent, "subagent", delegation_id
        ),
    )

    assert result["status"] == "rejected"
    assert heartbeat.timer_for(parent) is None
    assert heartbeat._child_owners == {}
    assert not heartbeat._owners[id(parent)].children


class _Timer:
    def __init__(self, _delay, _callback):
        self.cancelled = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


def test_delegate_task_background_uses_live_tui_agent_session_id(monkeypatch):
    """TUI async delegation must route to the live/compressed agent id.

    Regression: delegate_task captured the stale approval/session context key
    after compression rotated parent_agent.session_id. The resulting completion
    was orphaned and could be consumed by an unrelated desktop session poller.
    """
    import json
    from unittest.mock import MagicMock
    import tools.delegate_tool as dt
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.approval import reset_current_session_key, set_current_session_key

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "post-compress-tip"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(
        dt,
        "_run_single_child",
        lambda *a, **k: {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "m",
            "exit_reason": "completed",
        },
    )

    approval_token = set_current_session_key("pre-compress-parent")
    session_tokens = set_session_vars(
        source="tui",
        session_key="pre-compress-parent",
        ui_session_id="origin-tab",
    )
    try:
        out = dt.delegate_task(goal="bg task", background=True, parent_agent=parent)
        assert json.loads(out)["status"] == "dispatched"
        evt = _drain_one()
    finally:
        reset_current_session_key(approval_token)
        clear_session_vars(session_tokens)

    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["session_key"] == "post-compress-tip"
    assert evt["origin_ui_session_id"] == "origin-tab"


def test_concurrent_dispatch_respects_capacity():
    """Two threads racing dispatch with cap=1 must yield exactly one accept
    (capacity check and record insert are atomic under the records lock)."""
    gate = threading.Event()

    def blocker():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "x"}

    results = []
    barrier = threading.Barrier(2)

    def racer():
        barrier.wait(timeout=5)
        results.append(
            ad.dispatch_async_delegation(
                goal="race", context=None, toolsets=None, role="leaf",
                model="m", session_key="", runner=blocker,
                max_async_children=1,
            )
        )

    threads = [threading.Thread(target=racer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    statuses = sorted(r["status"] for r in results)
    assert statuses == ["dispatched", "rejected"]
    gate.set()


# ---------------------------------------------------------------------------
# Gateway routing: session_key -> platform/chat_id, rich formatting, injection
# ---------------------------------------------------------------------------

def _make_async_evt(**over):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_x1",
        "session_key": "agent:main:telegram:dm:12345:678",
        "goal": "Investigate flaky test",
        "context": "repo /tmp/p",
        "toolsets": ["terminal"],
        "role": "leaf",
        "model": "m",
        "status": "completed",
        "summary": "Found the bug in test_foo",
        "api_calls": 4,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
    }
    evt.update(over)
    return evt


def test_gateway_formatter_renders_async_block():
    from gateway.run import _format_gateway_process_notification

    txt = _format_gateway_process_notification(_make_async_evt())
    assert txt is not None
    assert "ASYNC DELEGATION COMPLETE" in txt
    assert "Found the bug in test_foo" in txt
    assert "Investigate flaky test" in txt


def test_gateway_cli_origin_event_left_unrouted():
    """An empty session_key (CLI origin) is left without routing fields."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    evt = _make_async_evt(session_key="")
    runner._enrich_async_delegation_routing(evt)
    assert "platform" not in evt


# ---------------------------------------------------------------------------
# Issue 73 unit 3 — cross-process exclusive acceptance + restart delivery
# ---------------------------------------------------------------------------


def _seed_control_delegation(db_path, delegation_id="deleg-succ", child_id="child-1"):
    with ad._transaction() as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, state, dispatched_at, updated_at,
                children_json)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (
                delegation_id,
                "parent",
                time.time(),
                time.time(),
                json.dumps([{"subagent_id": child_id, "session_id": "db-child"}]),
            ),
        )


def test_restart_claim_works_after_recovery_marks_row_unknown(tmp_path, monkeypatch):
    """An accepted control is still deliverable after the owner died.

    recover_abandoned_delegations classifies a dead-owner running/finalizing
    row as ``unknown``. A previously accepted queue/interrupt control on that
    row must not be stranded: a fresh process must still be able to claim it
    and start exactly one next/replacement turn.
    """
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(ad, "_db_path", lambda: db_path)
    _seed_control_delegation(db_path)

    receipt = ad.reserve_child_control("deleg-succ", "child-1", "queue", "next")
    assert receipt["status"] == "accepted"

    # Owner process dies before delivering; recovery marks the row unknown.
    with ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET state='unknown' WHERE delegation_id=?",
            ("deleg-succ",),
        )
    # A fresh process takes over delivery.
    monkeypatch.setattr(ad, "_CONTROL_PROCESS_TOKEN", "restarted-process")

    claimed = ad.claim_child_control("deleg-succ", "child-1")
    assert claimed is not None
    assert claimed["generation"] == receipt["generation"]
    assert claimed["state"] == "running"

    # Exactly-once: a second claim in the same process delivers nothing.
    assert ad.claim_child_control("deleg-succ", "child-1") is None


_CAS_CHILD = r"""
import json, os, sys, time
from pathlib import Path
from tools import async_delegation as ad

db_path, delegation_id, child_id, action, message, result_file = sys.argv[1:7]
ad._db_path = lambda: Path(db_path)
# Synchronize so both processes try to reserve on the still-empty control row
# at the same time, reproducing the two-readers-see-empty race.
go = db_path + ".go"
deadline = time.monotonic() + 10
while not os.path.exists(go):
    if time.monotonic() > deadline:
        break
    time.sleep(0.005)
receipt = ad.reserve_child_control(delegation_id, child_id, action, message)
with open(result_file, "w") as fh:
    fh.write(json.dumps(receipt))
"""


def test_cross_process_reserve_reports_at_most_one_accepted(tmp_path, monkeypatch):
    """Two processes reserving the same child cannot both report acceptance.

    reserve_child_control's UPDATE is CAS-gated on the previously-read control
    blob, so even two separate SQLite connections that both read an empty
    control row can only let ONE commit accept. Without the CAS, both writes
    succeed (last-writer-wins) and both callers receive `accepted` — the
    mutually-exclusive queue/interrupt fence silently breaks across processes.
    """
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(ad, "_db_path", lambda: db_path)
    _seed_control_delegation(db_path)

    results = []
    procs = []
    for action in ("queue", "interrupt"):
        result_file = tmp_path / f"res-{action}.json"
        procs.append(
            subprocess.Popen(
                [
                    sys.executable, "-c", _CAS_CHILD,
                    os.fspath(db_path), "deleg-succ", "child-1",
                    action, "next", os.fspath(result_file),
                ],
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        )
        results.append(result_file)
    # Release both children simultaneously onto the empty control row.
    (tmp_path / "state.db.go").write_text("go")
    for proc in procs:
        proc.wait(timeout=30)
    receipts = []
    for result_file in results:
        receipts.append(json.loads(result_file.read_text()))

    statuses = [r.get("status") for r in receipts]
    assert statuses.count("accepted") <= 1, statuses
    assert statuses.count("queued") <= 1, statuses
    # The winner accepts; the loser never reports acceptance.
    assert len([s for s in statuses if s in ("accepted", "queued")]) <= 1, statuses


# Two processes both read the same accepted control and race to claim it.
# claim_child_control must CAS its UPDATE on the previously-read blob (like
# reserve), otherwise both processes see rowcount=1 and both start a next
# turn — the mutually-exclusive delivery fence silently breaks. Both processes
# share one claim token so only the CAS-race winner can own it; a shared
# go-barrier parks them until each has observed the accepted blob.
_CLAIM_CHILD = r"""
import json, os, sys, time
from pathlib import Path
from tools import async_delegation as ad

db_path, delegation_id, child_id, result_file = sys.argv[1:5]
ad._db_path = lambda: Path(db_path)
# A shared token: a running control claimed by ourselves returns None, so only
# the CAS-race winner can report a successful claim — the loser must too.
ad._CONTROL_PROCESS_TOKEN="***"
ready = db_path + ".claim-ready"
go = db_path + ".claim-go"
# Observe the still-accepted blob, then park so both race the write together.
with ad._transaction() as conn:
    conn.execute(
        "SELECT child_control_json FROM async_delegations WHERE delegation_id=?",
        (delegation_id,),
    ).fetchone()
with open(ready, "a") as fh:
    fh.write("x")
deadline = time.monotonic() + 10
while not os.path.exists(go):
    if time.monotonic() > deadline:
        break
    time.sleep(0.005)
claimed = ad.claim_child_control(delegation_id, child_id)
with open(result_file, "w") as fh:
    fh.write(json.dumps({"claimed": bool(claimed)}))
"""


def test_cross_process_claim_allows_at_most_one_winner(tmp_path, monkeypatch):
    """Two OS processes claiming the same accepted control: at most one wins.

    claim_child_control's UPDATE is CAS-gated on the previously-read control
    blob. Two separate SQLite connections that both read the same ``accepted``
    blob can still let only ONE commit the claim; without the CAS both writes
    succeed with rowcount=1 and both processes start a replacement turn.
    """
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(ad, "_db_path", lambda: db_path)
    _seed_control_delegation(db_path)
    receipt = ad.reserve_child_control("deleg-succ", "child-1", "queue", "next")
    assert receipt["status"] == "accepted"

    procs, result_files = [], []
    for _ in range(2):
        result_file = tmp_path / f"claim-{len(result_files)}.json"
        result_files.append(result_file)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable, "-c", _CLAIM_CHILD,
                    os.fspath(db_path), "deleg-succ", "child-1",
                    os.fspath(result_file),
                ],
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        )
    # Release both claims only after every process observed the accepted blob.
    ready_path = tmp_path / "state.db.claim-ready"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if ready_path.stat().st_size >= 2:
                break
        except OSError:
            pass
        time.sleep(0.01)
    (tmp_path / "state.db.claim-go").write_text("go")
    for proc in procs:
        proc.wait(timeout=30)
    results = [json.loads(r.read_text()) for r in result_files]
    winners = [r["claimed"] for r in results].count(True)
    assert winners <= 1, results


def test_recovery_retains_dead_owner_row_with_accepted_control(tmp_path, monkeypatch):
    """Recovery retains a dead-owner row that still has an accepted control.

    On the primary restart path the owning process dies while an accepted
    queue/interrupt control is still pending. recover_abandoned_delegations
    marks the row ``unknown``; it must ALSO retain it so the restart drain
    (which delivers children found only via the retained registry, then still
    fail-closes on foreign owners) starts exactly one next/replacement turn.
    Without the retain the accepted control strands on the unknown+unretained
    row and the drain skips it.
    """
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(ad, "_db_path", lambda: db_path)
    _seed_control_delegation(db_path)
    receipt = ad.reserve_child_control("deleg-succ", "child-1", "queue", "next")
    assert receipt["status"] == "accepted"

    # The row is running with no live owner pid -> recovery classifies it dead.
    with ad._transaction() as conn:
        before = conn.execute(
            "SELECT retained FROM async_delegations WHERE delegation_id=?",
            ("deleg-succ",),
        ).fetchone()[0]
    assert before == 0

    ad.recover_abandoned_delegations()

    with ad._transaction() as conn:
        row = conn.execute(
            "SELECT state, retained FROM async_delegations WHERE delegation_id=?",
            ("deleg-succ",),
        ).fetchone()
    assert row[0] == "unknown"
    assert row[1] == 1, "accepted control must be retained for restart drain"
    assert any(r[0] == "deleg-succ" for r in ad._retained_rows()), (
        "accepted control must be findable by the restart drain"
    )

