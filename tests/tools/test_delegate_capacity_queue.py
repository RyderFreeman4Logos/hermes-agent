"""Top-level delegate_task queues behind saturated worker capacity."""

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from run_agent import AIAgent
from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_async_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _dispatch_batch(
    runner, capacity, *, session_key="owner", parent_session_id=None, interrupt_fn=None
):
    return ad.dispatch_async_delegation_batch(
        goals=["capacity test"],
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key=session_key,
        parent_session_id=parent_session_id,
        runner=runner,
        interrupt_fn=interrupt_fn,
        max_async_children=capacity,
    )

def _dispatch_single(runner, capacity, *, session_key="owner", interrupt_fn=None):
    return ad.dispatch_async_delegation(
        goal="capacity test",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key=session_key,
        runner=runner,
        interrupt_fn=interrupt_fn,
        max_async_children=capacity,
    )


def _durable_count():
    conn = ad._connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM async_delegations").fetchone()[0]
    finally:
        conn.close()


def _observe_admission_wait(monkeypatch):
    waiting = threading.Event()
    original_wait = ad._admission_condition.wait

    def observed_wait(*args, **kwargs):
        waiting.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(ad._admission_condition, "wait", observed_wait)
    return waiting


@pytest.mark.parametrize(
    "dispatch_one",
    [
        pytest.param(_dispatch_single, id="single"),
        pytest.param(_dispatch_batch, id="batch"),
    ],
)
def test_registration_rejects_when_process_backlog_is_full(dispatch_one):
    release = threading.Event()
    running_started = threading.Event()
    queued_started = threading.Event()
    rejected_started = threading.Event()

    def running():
        running_started.set()
        assert release.wait(timeout=60)
        return {"results": []}

    running_dispatch = dispatch_one(running, 1, session_key="running-owner")
    assert running_started.wait(timeout=5)
    queued_dispatch = dispatch_one(
        lambda: queued_started.set() or {"results": []},
        1,
        session_key="queued-owner",
    )
    try:
        rejected = dispatch_one(
            lambda: rejected_started.set() or {"results": []},
            1,
            session_key="rejected-owner",
        )
        queued_started_before_release = queued_started.is_set()
        rejected_started_before_release = rejected_started.is_set()
        active_before_release = ad.active_count()
        durable_before_release = _durable_count()
    finally:
        release.set()

    dispatches = [running_dispatch, queued_dispatch, rejected]
    expected_ids = {
        result["delegation_id"]
        for result in dispatches
        if result["status"] == "dispatched"
    }
    completed_ids = {
        process_registry.completion_queue.get(timeout=5)["delegation_id"]
        for _ in expected_ids
    }
    assert completed_ids == expected_ids
    assert running_dispatch["status"] == "dispatched"
    assert queued_dispatch["status"] == "dispatched"
    assert rejected["status"] == "rejected"
    assert "capacity reached" in rejected["error"].lower()
    assert not queued_started_before_release
    assert not rejected_started_before_release
    assert active_before_release == 2
    assert durable_before_release == 2
    assert queued_started.is_set()
    assert not rejected_started.is_set()
    with pytest.raises(queue.Empty):
        process_registry.completion_queue.get(timeout=0.2)


@pytest.mark.parametrize(
    "dispatch_one",
    [
        pytest.param(_dispatch_single, id="single"),
        pytest.param(_dispatch_batch, id="batch"),
    ],
)
def test_persistence_failure_releases_backlog_reservation(monkeypatch, dispatch_one):
    runner_started = threading.Event()
    monkeypatch.setattr(
        ad, "_persist_dispatch", MagicMock(side_effect=RuntimeError("persist failed"))
    )

    with pytest.raises(RuntimeError, match="persist failed"):
        dispatch_one(lambda: runner_started.set(), 1)

    assert ad.active_count() == 0
    assert ad._pending_admission_ids == set()
    assert not runner_started.is_set()
    assert _durable_count() == 0


def test_backlog_admission_is_atomic_across_sessions():
    release = threading.Event()
    running_started = threading.Event()
    candidate_started = threading.Event()

    def running():
        running_started.set()
        assert release.wait(timeout=60)
        return {"results": []}

    running_dispatch = _dispatch_batch(running, 1, session_key="occupier")
    assert running_started.wait(timeout=5)

    callers_ready = threading.Barrier(3)

    def compete(dispatch_one, session_key):
        callers_ready.wait(timeout=5)
        return dispatch_one(
            lambda: candidate_started.set() or {"results": []},
            1,
            session_key=session_key,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            single = callers.submit(compete, _dispatch_single, "session-a")
            batch = callers.submit(compete, _dispatch_batch, "session-b")
            callers_ready.wait(timeout=5)
            candidates = [single.result(timeout=5), batch.result(timeout=5)]
        candidate_started_before_release = candidate_started.is_set()
        active_before_release = ad.active_count()
        durable_before_release = _durable_count()
    finally:
        release.set()

    accepted = [result for result in candidates if result["status"] == "dispatched"]
    expected_ids = {
        running_dispatch["delegation_id"],
        *(result["delegation_id"] for result in accepted),
    }
    completed_ids = {
        process_registry.completion_queue.get(timeout=5)["delegation_id"]
        for _ in expected_ids
    }
    assert completed_ids == expected_ids
    assert sorted(result["status"] for result in candidates) == [
        "dispatched",
        "rejected",
    ]
    assert not candidate_started_before_release
    assert active_before_release == 2
    assert durable_before_release == 2
    assert candidate_started.is_set()


def test_cap_growth_never_oversubscribes_executor_generations(monkeypatch):
    release = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def runner(started):
        def run():
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            started.set()
            assert release.wait(timeout=60)
            with active_lock:
                active -= 1
            return {"results": []}

        return run

    first = _dispatch_batch(runner(first_started), 1)
    assert first_started.wait(timeout=5)
    second = _dispatch_batch(runner(second_started), 2)
    assert second_started.wait(timeout=5)

    waiting = _observe_admission_wait(monkeypatch)
    third_started = threading.Event()
    third = _dispatch_batch(runner(third_started), 2)
    assert waiting.wait(timeout=5)

    assert not third_started.is_set()
    with active_lock:
        assert active == max_active == 2
    assert sorted(r["status"] for r in ad.list_async_delegations()) == [
        "queued",
        "running",
        "running",
    ]

    release.set()
    expected_ids = {
        first["delegation_id"],
        second["delegation_id"],
        third["delegation_id"],
    }
    completed_ids = {
        process_registry.completion_queue.get(timeout=5)["delegation_id"]
        for _ in expected_ids
    }
    assert completed_ids == expected_ids
    assert third_started.is_set()
    assert max_active == 2


def test_cap_decrease_waits_for_running_excess_to_finish(monkeypatch):
    first_started = threading.Event()
    second_started = threading.Event()
    first_release = threading.Event()
    second_release = threading.Event()
    active = 0
    active_lock = threading.Lock()

    def occupying_runner(started, release):
        def run():
            nonlocal active
            with active_lock:
                active += 1
            started.set()
            assert release.wait(timeout=60)
            with active_lock:
                active -= 1
            return {"results": []}

        return run

    first = _dispatch_batch(occupying_runner(first_started, first_release), 2)
    second = _dispatch_batch(occupying_runner(second_started, second_release), 2)
    assert first_started.wait(timeout=5)
    assert second_started.wait(timeout=5)

    waiting = _observe_admission_wait(monkeypatch)
    third_started = threading.Event()
    old_active_at_third_start = []

    def third_runner():
        with active_lock:
            old_active_at_third_start.append(active)
        third_started.set()
        return {"results": []}

    third = _dispatch_batch(third_runner, 1)
    first_release.set()
    assert waiting.wait(timeout=5)
    assert not third_started.is_set()

    second_release.set()
    assert third_started.wait(timeout=5)
    assert old_active_at_third_start == [0]
    expected_ids = {
        first["delegation_id"],
        second["delegation_id"],
        third["delegation_id"],
    }
    completed_ids = {
        process_registry.completion_queue.get(timeout=5)["delegation_id"]
        for _ in expected_ids
    }
    assert completed_ids == expected_ids


@pytest.mark.parametrize(
    "tool_args, expected_count, capacity",
    [
        pytest.param({"goal": "queued single"}, 1, 1, id="single"),
        pytest.param(
            {"tasks": [{"goal": "queued one"}, {"goal": "queued two"}]},
            2,
            2,
            id="batch",
        ),
    ],
)
def test_public_handler_queues_when_worker_capacity_is_saturated(
    monkeypatch, tool_args, expected_count, capacity
):
    """The model tool call returns before queued children can start."""
    occupied = threading.Barrier(capacity + 1)
    release_occupied = threading.Event()
    queued_started = threading.Event()
    release_queued = threading.Event()

    def occupying_runner():
        occupied.wait(timeout=5)
        assert release_occupied.wait(timeout=60)
        return {"results": [{"status": "completed", "summary": "occupied done"}]}

    occupied_ids = set()
    for index in range(capacity):
        first = ad.dispatch_async_delegation_batch(
            goals=[f"occupy worker {index}"],
            context=None,
            toolsets=None,
            role="leaf",
            model="m",
            session_key="occupier",
            runner=occupying_runner,
            max_async_children=capacity,
        )
        assert first["status"] == "dispatched"
        occupied_ids.add(first["delegation_id"])
    occupied.wait(timeout=5)

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "public-handler-session"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None

    def build_child(**kwargs):
        child = MagicMock()
        child.model = "m"
        child._delegate_role = "leaf"
        child._subagent_id = f"child-{kwargs['task_index']}"
        return child

    def queued_runner(task_index, goal, child=None, parent_agent=None, **kwargs):
        queued_started.set()
        assert release_queued.wait(timeout=60)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": f"done: {goal}",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "m",
        }

    credentials = {
        "model": "m",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build_child)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", queued_runner)
    monkeypatch.setattr(
        "tools.delegate_tool._get_max_concurrent_children", lambda: capacity
    )
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_delegation_credentials",
        lambda *args, **kwargs: credentials,
    )
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: True
    )

    with ThreadPoolExecutor(max_workers=1) as caller:
        call = caller.submit(AIAgent._dispatch_delegate_task, parent, tool_args)
        try:
            result = json.loads(call.result(timeout=2))
            assert result["status"] == "dispatched"
            assert result["count"] == expected_count
            assert not queued_started.is_set()
        finally:
            release_occupied.set()
            release_queued.set()

    assert queued_started.wait(timeout=5)
    event = process_registry.completion_queue.get(timeout=5)
    for _ in range(capacity):
        if event["delegation_id"] not in occupied_ids:
            break
        event = process_registry.completion_queue.get(timeout=5)
    assert event["delegation_id"] == result["delegation_id"]
    assert len(event["results"]) == expected_count


def test_queued_cancellation_waits_for_durable_registration_and_completes_once(
    monkeypatch,
):
    occupied = threading.Event()
    release_occupied = threading.Event()

    def occupying_runner():
        occupied.set()
        assert release_occupied.wait(timeout=60)
        return {"results": [{"status": "completed", "summary": "occupied done"}]}

    occupier = _dispatch_batch(occupying_runner, 1, session_key="occupier")
    assert occupied.wait(timeout=5)

    persist_entered = threading.Event()
    release_persist = threading.Event()
    persisted_under_records_lock = []
    original_persist_dispatch = ad._persist_dispatch

    def blocked_persist_dispatch(record):
        records_lock_was_free = []

        def probe_records_lock():
            acquired = ad._records_lock.acquire(blocking=False)
            records_lock_was_free.append(acquired)
            if acquired:
                ad._records_lock.release()

        probe = threading.Thread(target=probe_records_lock)
        probe.start()
        probe.join(timeout=5)
        assert not probe.is_alive()
        persisted_under_records_lock.append(not records_lock_was_free[0])
        persist_entered.set()
        assert release_persist.wait(timeout=60)
        original_persist_dispatch(record)

    created = []
    runner_started = threading.Event()
    lifecycle = []
    not_started = []
    original_dispatch_batch = ad.dispatch_async_delegation_batch

    def tracked_dispatch_batch(**kwargs):
        callback = kwargs["on_not_started"]

        def tracked_not_started(status):
            not_started.append(status)
            callback(status)

        kwargs["on_not_started"] = tracked_not_started
        return original_dispatch_batch(**kwargs)

    class Child:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.session_id = "queued-child-session"
            self._session_init_model_config = {}
            self.close_count = 0
            self.interrupt_count = 0
            created.append(self)

        def close(self):
            self.close_count += 1

        def interrupt(self, _reason):
            self.interrupt_count += 1

    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = None
    parent._client_kwargs = {}
    parent._current_turn_id = "queued-turn"
    parent._delegate_depth = 0
    parent._delegate_spinner = None
    parent._interrupt_requested = False
    parent._print_fn = None
    parent._session_db = None
    parent.api_key = None
    parent.api_mode = None
    parent.base_url = None
    parent.disabled_toolsets = []
    parent.enabled_toolsets = []
    parent.model = "m"
    parent.provider = None
    parent.reasoning_config = None
    parent.session_id = "queued-public-session"
    credentials = {
        "model": "m",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }

    monkeypatch.setattr("run_agent.AIAgent", Child)
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {})
    monkeypatch.setattr("tools.delegate_tool._get_max_concurrent_children", lambda: 1)
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_delegation_credentials",
        lambda *args, **kwargs: credentials,
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *args, **kwargs: runner_started.set(),
    )
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: True
    )
    monkeypatch.setattr(
        "hermes_cli.observability.observe_lifecycle", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: lifecycle.append((name, kwargs)) or [],
    )
    monkeypatch.setattr(ad, "dispatch_async_delegation_batch", tracked_dispatch_batch)
    monkeypatch.setattr(ad, "_persist_dispatch", blocked_persist_dispatch)

    try:
        with ThreadPoolExecutor(max_workers=2) as caller:
            call = caller.submit(
                AIAgent._dispatch_delegate_task,
                parent,
                {"goal": "cancel while queued"},
            )
            assert persist_entered.wait(timeout=5)

            cancel_started = threading.Event()

            def cancel_queued():
                cancel_started.set()
                return ad.interrupt_for_session(
                    parent_session_id=parent.session_id, reason="test"
                )

            cancel = caller.submit(cancel_queued)
            assert cancel_started.wait(timeout=5)
            # On the broken ordering persistence runs after publication, so
            # force cancellation to finish before releasing the INSERT.
            if not persisted_under_records_lock[0]:
                assert cancel.result(timeout=5) == 1
            release_persist.set()

            result = json.loads(call.result(timeout=5))
            assert cancel.result(timeout=5) == 1
            assert result["status"] == "dispatched"
            assert len(created) == 1
            child = created[0]
            assert parent._active_children == []
            assert not runner_started.is_set()
            assert child.interrupt_count == 1
            assert child.close_count == 1
            assert not_started == ["interrupted"]
            assert [name for name, _kwargs in lifecycle] == [
                "subagent_start",
                "subagent_stop",
            ]
            assert lifecycle[0][1]["child_session_id"] == child.session_id
            assert lifecycle[1][1]["child_session_id"] == child.session_id
            assert lifecycle[1][1]["child_status"] == "interrupted"
            assert ad.interrupt_for_session(parent_session_id=parent.session_id) == 0
            assert child.close_count == 1

        event = process_registry.completion_queue.get(timeout=5)
        assert event["delegation_id"] == result["delegation_id"]
        assert event["status"] == "interrupted"
        with pytest.raises(queue.Empty):
            process_registry.completion_queue.get(timeout=0.2)

        with ad._transaction() as conn:
            durable_rows = conn.execute(
                """SELECT state, delivery_state, completed_at, event_json
                   FROM async_delegations WHERE delegation_id=?""",
                (result["delegation_id"],),
            ).fetchall()
        assert len(durable_rows) == 1
        state, delivery_state, completed_at, event_json = durable_rows[0]
        assert state == "interrupted"
        assert delivery_state == "pending"
        assert completed_at is not None
        assert json.loads(event_json)["status"] == "interrupted"

        release_occupied.set()
        occupied_event = process_registry.completion_queue.get(timeout=5)
        assert occupied_event["delegation_id"] == occupier["delegation_id"]
        assert ad.mark_completion_delivered(result["delegation_id"])
        assert ad.mark_completion_delivered(occupier["delegation_id"])

        monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: False)
        assert ad.recover_abandoned_delegations() == 0
        restored = queue.Queue()
        assert ad.restore_undelivered_completions(restored) == 0
        assert restored.empty()
        with pytest.raises(queue.Empty):
            process_registry.completion_queue.get(timeout=0.2)

        durable = ad.get_durable_delegation(result["delegation_id"])
        assert durable is not None
        assert durable["state"] == "interrupted"
        assert durable["delivery_state"] == "delivered"
        assert not runner_started.is_set()
        assert child.close_count == 1
        assert [name for name, _kwargs in lifecycle] == [
            "subagent_start",
            "subagent_stop",
        ]
    finally:
        release_persist.set()
        release_occupied.set()


@pytest.mark.parametrize("cancel_scope", ["stop", "session"])
def test_cancel_queued_delegation_completes_once_without_starting_runner(
    monkeypatch, cancel_scope,
):
    first_started = threading.Event()
    second_started = threading.Event()
    release_occupied = threading.Event()
    queued_started = threading.Event()
    queued_interrupted = threading.Event()

    def occupying_runner(started):
        def run():
            started.set()
            assert release_occupied.wait(timeout=60)
            return {"results": []}

        return run

    first = _dispatch_batch(
        occupying_runner(first_started),
        1,
        session_key="other-owner",
        interrupt_fn=release_occupied.set,
    )
    assert first_started.wait(timeout=5)
    second = _dispatch_batch(
        occupying_runner(second_started),
        2,
        session_key="other-owner",
        interrupt_fn=release_occupied.set,
    )
    assert second_started.wait(timeout=5)

    waiting = _observe_admission_wait(monkeypatch)
    queued = _dispatch_batch(
        lambda: queued_started.set() or {"results": []},
        2,
        session_key="queued-owner",
        parent_session_id="queued-parent",
        interrupt_fn=queued_interrupted.set,
    )
    assert queued["status"] == "dispatched"
    assert waiting.wait(timeout=5)

    if cancel_scope == "stop":
        assert ad.interrupt_all(reason="/stop") == 3
    else:
        assert ad.interrupt_for_session(parent_session_id="queued-parent", reason="test") == 1
    assert queued_interrupted.is_set()
    assert not queued_started.is_set()

    release_occupied.set()
    expected_ids = {
        first["delegation_id"],
        second["delegation_id"],
        queued["delegation_id"],
    }
    events = {}
    for _ in expected_ids:
        event = process_registry.completion_queue.get(timeout=5)
        events[event["delegation_id"]] = event
    assert events.keys() == expected_ids
    assert events[queued["delegation_id"]]["status"] == "interrupted"
    with pytest.raises(queue.Empty):
        process_registry.completion_queue.get(timeout=0.2)
    assert not queued_started.is_set()


def test_public_handler_cleans_resources_after_schedule_rejection(monkeypatch):
    parent = MagicMock(
        _delegate_depth=0,
        session_id="schedule-failure-session",
        _interrupt_requested=False,
        _active_children=[],
        _active_children_lock=None,
    )
    children = []

    def build_child(**kwargs):
        child = MagicMock(
            model="m",
            _delegate_role="leaf",
            _subagent_id=f"child-{kwargs['task_index']}",
        )
        children.append(child)
        parent._active_children.append(child)
        return child

    credentials = {
        "model": "m",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build_child)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *args, **kwargs: pytest.fail("schedule failure ran child inline"),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_delegation_credentials",
        lambda *args, **kwargs: credentials,
    )
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: True
    )
    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        lambda **kwargs: {"status": "rejected", "error": "submit failed"},
    )
    monkeypatch.setattr(
        "tools.delegation_live_log.new_live_delegation_id",
        lambda: "deleg_rejected",
    )

    result = json.loads(
        AIAgent._dispatch_delegate_task(
            parent,
            {
                "tasks": [
                    {"goal": "first task goal"},
                    {"goal": "second task goal"},
                ]
            },
        )
    )

    assert result["status"] == "rejected"
    assert result["mode"] == "background"
    assert "submit failed" in result["error"]
    assert "not started" in result["note"]
    assert parent._active_children == []
    assert len(children) == 2
    for child in children:
        child.close.assert_called_once_with()

    from tools.delegation_live_log import live_transcript_root

    assert not (live_transcript_root() / "deleg_rejected").exists()
    assert ad.active_count() == 0
    assert _durable_count() == 0
