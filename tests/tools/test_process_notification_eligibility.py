import json

import pytest

import tools.process_registry as pr
from gateway.session_context import set_session_vars, clear_session_vars
from gateway.session_finite import finite_turn_scope
from tools import process_registry_wait_policy as policy



@pytest.mark.parametrize("mode", ["off", "finite", "foreign", "unsupported"])
def test_notified_wait_retains_delivery_eligibility(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if mode == "off":
        (tmp_path / "config.yaml").write_text("display:\n  background_process_notifications: off\n")
    tokens = set_session_vars(session_key="caller", async_delivery=mode != "unsupported")
    session = pr.ProcessSession(
        id="review-notified", command="bounded-work", notify_on_complete=True,
        session_key="other" if mode == "foreign" else "caller", task_id="caller", owner_task_id="caller",
    )
    pr.process_registry._running[session.id] = session
    try:
        with finite_turn_scope(mode == "finite"):
            assert not (policy._can_deliver_completion() and policy._caller_receives_completion(session))
            result = json.loads(pr._handle_process({"action": "wait", "session_id": session.id, "timeout": 1}))
            assert result.get("wait_deferred") is not True, (mode, result)
            assert result["status"] == "timeout"
    finally:
        pr.process_registry._running.pop(session.id, None)
        clear_session_vars(tokens)


def test_unhanded_child_is_not_parent_delivery_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tokens = set_session_vars(session_key="parent")
    session = pr.ProcessSession(id="review-child", command="bounded-work", session_key="parent",
                                task_id="sa-child", owner_task_id="sa-child")
    try:
        assert policy._caller_receives_completion(session) is False
    finally:
        clear_session_vars(tokens)

def test_parent_poll_never_promises_suppressed_child_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tokens = set_session_vars(session_key="parent")
    session = pr.ProcessSession(id="review-unhanded-child", command="bounded-work", session_key="parent",
                                task_id="sa-child", owner_task_id="sa-child")
    pr.process_registry._running[session.id] = session
    try:
        with finite_turn_scope(False):
            for _ in range(2):
                result = json.loads(pr._handle_process({"action": "poll", "session_id": session.id}))
                assert result["status"] == "running"
            assert not session.notify_on_complete, result
            assert "Do NOT call" not in result.get("note", ""), result
    finally:
        pr.process_registry._running.pop(session.id, None)
        clear_session_vars(tokens)


@pytest.mark.parametrize("handoff", [False, True])
def test_eligible_wait_defers_and_retains_completion_receipt(tmp_path, monkeypatch, handoff):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pr.process_registry, "_write_checkpoint", lambda: None)
    tokens = set_session_vars(session_key="parent", async_delivery=True)
    owner = "sa-child" if handoff else "parent"
    session = pr.ProcessSession(id="eligible-handoff" if handoff else "eligible-parent",
                                command="bounded-work", session_key="parent", task_id=owner,
                                owner_task_id=owner, notify_on_complete=True)
    pr.process_registry._running[session.id] = session
    try:
        if handoff:
            assert pr.process_registry.transfer_ownership(
                session.id, from_owner=owner, to_owner="parent", to_task_id="parent",
                to_session_key="parent", note="continue owned work",
            ) is session
        with finite_turn_scope(False):
            result = json.loads(pr._handle_process({"action": "wait", "session_id": session.id, "timeout": 1}))
        assert result["wait_deferred"] is True
        assert not pr.process_registry.is_completion_consumed(session.id)
        pr.process_registry._finish_exited(session, 7)
        events = [event for event in list(pr.process_registry.completion_queue.queue)
                  if event.get("session_id") == session.id]
        assert len(events) == 1
        assert events[0]["exit_code"] == 7
        assert events[0]["owner_task_id"] == "parent"
    finally:
        pr.process_registry._running.pop(session.id, None)
        pr.process_registry._finished.pop(session.id, None)
        clear_session_vars(tokens)


def test_preview_denies_child_but_actual_malformed_terminal_fails_open():
    from tools.process_registry_notifications import (
        should_surface_notification, should_surface_notification_preview,
    )
    event = {"type": "completion", "session_id": "child", "owner_task_id": "sa-child"}
    assert not should_surface_notification_preview(event, surface_child=False)
    assert should_surface_notification(event, surface_child=False)
    event.update(command="bounded-work", started_at=1.0, exit_code=1,
                 completion_reason="exited", termination_source="")
    assert not should_surface_notification(event, surface_child=False)
    event["exit_code"] = "invalid"
    assert should_surface_notification(event, surface_child=False)
