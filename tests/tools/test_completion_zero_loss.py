"""Zero-loss, exactly-once background completion delivery."""

from __future__ import annotations

import queue
import sys
import threading

import pytest

from agent.delegation_context import delegated_child_context
from tools import async_delegation as delivery
from tools import delegate_tool
from tools import process_registry as registry_module
from tools.process_registry import ProcessRegistry


@pytest.fixture()
def isolated_delivery(monkeypatch, tmp_path):
    monkeypatch.setattr(delivery, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(registry_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    return tmp_path


def _event(*, owner: str = "owner-1", session_id: str = "proc_same", started_at: float = 1.0):
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": "route",
        "parent_session_id": owner,
        "started_at": started_at,
        "command": "pytest -q",
        "exit_code": 1,
        "completion_reason": "exited",
        "termination_source": "",
        "output": "1 failed",
    }


def _claim_and_complete(event: dict, consumer: str = "test") -> str:
    claim = delivery.claim_event_delivery(event, consumer)
    assert claim
    assert delivery.complete_event_delivery(event, claim)
    return claim


def test_duplicate_event_has_one_insertion_before_and_after_ack(isolated_delivery):
    event = _event()
    assert delivery.persist_event_delivery(event)
    assert delivery.persist_event_delivery(dict(event))

    claim = delivery.claim_event_delivery(event, "first")
    assert claim
    assert delivery.claim_event_delivery(dict(event), "duplicate-before-ack") is None
    assert delivery.complete_event_delivery(event, claim)
    assert delivery.claim_event_delivery(dict(event), "duplicate-after-ack") is None
    receipt = delivery.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "delivered"


def test_restart_and_wake_replay_unacknowledged_once(isolated_delivery, monkeypatch):
    event = _event(session_id="proc_restart")
    assert delivery.persist_event_delivery(event)
    abandoned_claim = delivery.claim_event_delivery(event, "old-runtime")
    assert abandoned_claim

    # Model a dead runtime generation. Reconstruction must reclaim its claim,
    # while an acknowledged replay must never be reconstructed again.
    monkeypatch.setattr("gateway.status.runtime_status_pid_is_live", lambda _record: False)
    restarted = ProcessRegistry()
    restored = restarted.completion_queue.get_nowait()
    assert restored["session_id"] == event["session_id"]
    _claim_and_complete(restored, "restarted-runtime")

    woken_again = ProcessRegistry()
    assert woken_again.completion_queue.empty()


def test_owner_generation_is_part_of_stable_identity(isolated_delivery):
    first = _event(owner="owner-generation-a")
    second = _event(owner="owner-generation-b")
    assert delivery.persist_event_delivery(first)
    assert delivery.persist_event_delivery(second)

    first_claim = delivery.claim_event_delivery(first, "owner-a")
    second_claim = delivery.claim_event_delivery(second, "owner-b")
    assert first_claim and second_claim and first_claim != second_claim
    assert delivery.complete_event_delivery(first, first_claim)
    assert delivery.complete_event_delivery(second, second_claim)


def test_restore_cap_leaves_overflow_pending_for_later_delivery(isolated_delivery):
    events = [_event(session_id=f"proc_cap_{index}", started_at=index + 1.0) for index in range(3)]
    for event in events:
        assert delivery.persist_event_delivery(event)

    first_batch: queue.Queue = queue.Queue()
    assert delivery.restore_undelivered_completions(first_batch, limit=2) == 2
    selected = [first_batch.get_nowait() for _ in range(2)]
    selected_ids = {event["session_id"] for event in selected}
    for event in selected:
        _claim_and_complete(event, "first-batch")

    pending = next(event for event in events if event["session_id"] not in selected_ids)
    receipt = delivery.get_durable_event_delivery(pending)
    assert receipt is not None
    assert receipt["delivery_state"] == "pending"

    second_batch: queue.Queue = queue.Queue()
    assert delivery.restore_undelivered_completions(second_batch, limit=2) == 1
    assert second_batch.get_nowait()["session_id"] == pending["session_id"]


def test_wake_restore_does_not_duplicate_an_event_already_queued(isolated_delivery):
    event = _event(session_id="proc_wake")
    assert delivery.persist_event_delivery(event)
    target: queue.Queue = queue.Queue()
    target.put(event)

    assert delivery.restore_undelivered_completions(target) == 0
    assert target.get_nowait()["session_id"] == "proc_wake"
    assert delivery.restore_undelivered_completions(target) == 1


def test_fast_parent_process_is_durable_before_public_reconstruction(isolated_delivery):
    registry = ProcessRegistry()
    session = registry.spawn_local(
        f"{sys.executable} -c 'print(\"done\")'",
        notify_on_complete=True,
        parent_session_id="parent-generation",
        session_key="route",
    )
    assert session._reader_thread is not None
    session._reader_thread.join(timeout=5)
    assert not session._reader_thread.is_alive()

    event = registry.completion_event(session.id)
    assert event is not None
    assert event["started_at"] == session.started_at
    assert event["parent_session_id"] == "parent-generation"
    receipt = delivery.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "pending"

    # Lose the in-memory queue, then reconstruct through the same public
    # ProcessRegistry startup path used by restart/wake recovery.
    restarted = ProcessRegistry()
    restored = restarted.completion_queue.get_nowait()
    assert restored["session_id"] == session.id
    _claim_and_complete(restored, "public-reconstruction")


def test_active_delegated_owner_receives_red_and_green_without_parent_wake(
    isolated_delivery,
):
    received: list[str] = []
    received_lock = threading.Lock()

    class Child:
        session_id = "child-generation"
        _parent_session_id = "parent-generation"

        def steer(self, text: str) -> bool:
            with received_lock:
                received.append(text)
            return True

    child = Child()
    subagent_id = "sa-zero-loss"
    delegate_tool._register_subagent(
        {
            "subagent_id": subagent_id,
            "agent": child,
            "owner_session_id": child._parent_session_id,
        }
    )
    registry = ProcessRegistry()
    try:
        sessions = []
        with delegated_child_context(
            child.session_id,
            owner_session_id=child._parent_session_id,
            subagent_id=subagent_id,
        ):
            for code in (1, 0):
                sessions.append(
                    registry.spawn_local(
                        f"{sys.executable} -c 'import sys; sys.exit({code})'",
                        notify_on_complete=True,
                        session_key="route",
                    )
                )
        for session in sessions:
            assert session._reader_thread is not None
            session._reader_thread.join(timeout=5)
            assert not session._reader_thread.is_alive()
    finally:
        delegate_tool._unregister_subagent(subagent_id, agent=child)

    assert len(received) == 2
    assert any("exit code 1" in text for text in received)
    assert any("exit code 0" in text for text in received)
    assert registry.completion_queue.empty()
    for session in sessions:
        event = registry.completion_event(session.id)
        assert event is not None
        receipt = delivery.get_durable_event_delivery(event)
        assert receipt is not None
        assert receipt["delivery_state"] == "child_local"


def test_inactive_or_ambiguous_child_failure_fails_open_once(isolated_delivery):
    registry = ProcessRegistry()
    with delegated_child_context(
        "child-gone",
        owner_session_id="parent-generation",
        subagent_id="sa-gone",
    ):
        session = registry.spawn_local(
            "printf 'pytest: 1 passed\\nPOST_FOREIGN_CWD: refused\\n'; exit 91",
            notify_on_complete=True,
            session_key="route",
        )
    assert session._reader_thread is not None
    session._reader_thread.join(timeout=5)
    event = registry.completion_queue.get_nowait()
    assert event["delegated_child"] is True
    assert event["delegated_subagent_id"] == "sa-gone"
    claim = delivery.claim_event_delivery(event, "parent")
    assert claim
    assert delivery.claim_event_delivery(dict(event), "duplicate") is None
    assert delivery.complete_event_delivery(event, claim)


def test_inactive_routine_child_tail_stays_child_local(isolated_delivery):
    registry = ProcessRegistry()
    with delegated_child_context(
        "child-gone",
        owner_session_id="parent-generation",
        subagent_id="sa-gone",
    ):
        session = registry.spawn_local(
            f"{sys.executable} -c 'print(\"done\")'",
            notify_on_complete=True,
            session_key="route",
        )
    assert session._reader_thread is not None
    session._reader_thread.join(timeout=5)
    assert not session._reader_thread.is_alive()

    event = registry.completion_event(session.id)
    assert event is not None
    assert registry.completion_queue.empty()
    receipt = delivery.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "child_local"
