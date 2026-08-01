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


@pytest.mark.parametrize("cancel_scope", ["stop", "session"])
def test_cancel_queued_delegation_completes_once_without_starting_runner(
    cancel_scope,
):
    occupied = threading.Event()
    release_occupied = threading.Event()
    queued_started = threading.Event()
    queued_interrupted = threading.Event()

    def occupying_runner():
        occupied.set()
        assert release_occupied.wait(timeout=60)
        return {"results": [{"status": "completed", "summary": "occupied done"}]}

    first = ad.dispatch_async_delegation_batch(
        goals=["occupy worker"],
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="other-owner",
        runner=occupying_runner,
        interrupt_fn=release_occupied.set,
        max_async_children=1,
    )
    assert occupied.wait(timeout=5)

    queued = ad.dispatch_async_delegation_batch(
        goals=["cancel before admission"],
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="queued-owner",
        runner=lambda: queued_started.set() or {"results": []},
        interrupt_fn=queued_interrupted.set,
        max_async_children=1,
    )
    assert queued["status"] == "dispatched"

    if cancel_scope == "stop":
        assert ad.interrupt_all(reason="/stop") == 2
    else:
        assert ad.interrupt_for_session(session_key="queued-owner", reason="test") == 1
    assert queued_interrupted.is_set()
    event = process_registry.completion_queue.get(timeout=5)
    if cancel_scope == "stop":
        other = process_registry.completion_queue.get(timeout=5)
        events = {
            event["delegation_id"]: event,
            other["delegation_id"]: other,
        }
        assert events[queued["delegation_id"]]["status"] == "interrupted"
        assert first["delegation_id"] in events
    else:
        assert event["delegation_id"] == queued["delegation_id"]
        assert event["status"] == "interrupted"
    assert not queued_started.is_set()

    release_occupied.set()
    if cancel_scope == "session":
        first_event = process_registry.completion_queue.get(timeout=5)
        assert first_event["delegation_id"] == first["delegation_id"]
    with pytest.raises(queue.Empty):
        process_registry.completion_queue.get(timeout=0.2)
    assert not queued_started.is_set()


def test_public_handler_does_not_run_inline_after_schedule_failure(monkeypatch):
    parent = MagicMock(
        _delegate_depth=0,
        session_id="schedule-failure-session",
        _interrupt_requested=False,
        _active_children=[],
        _active_children_lock=None,
    )
    child = MagicMock(model="m", _delegate_role="leaf", _subagent_id="child-0")
    credentials = {
        "model": "m",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **kw: child)
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

    result = json.loads(
        AIAgent._dispatch_delegate_task(parent, {"goal": "must stay background"})
    )

    assert result == {"error": "submit failed"}
