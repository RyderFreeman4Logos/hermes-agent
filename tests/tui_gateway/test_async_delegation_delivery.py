"""Durable async-delegation completion delivery through the TUI poller rail."""

from __future__ import annotations

import os
import queue
import threading
from unittest.mock import patch

from tools import async_delegation as ad
from tui_gateway import server


def _persist_pending_with_missing_event_routing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._local_pending_requeue_at.clear()
    delegation_id = "deleg_pending_owner"
    record = {
        "delegation_id": delegation_id,
        "session_key": "parent-key",
        "origin_ui_session_id": "owner-tab",
        "parent_session_id": "parent-session-id",
        # Empty is correct for TUI/Desktop; it must not erase the other
        # durable return addresses during a replay.
        "origin_session_id": "",
        "dispatched_at": 1.0,
        "goal": "finish the background task",
        "provider": "custom:test-provider",
    }
    ad._persist_dispatch(record)
    # Simulate the residual #28 payload: the durable row has routing, but the
    # completion JSON lost every TUI selector and carries no api-server wake id.
    event = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "",
        "origin_ui_session_id": "",
        "origin_session_id": "",
        "status": "completed",
        "summary": "background task finished",
        "dispatched_at": 1.0,
        "completed_at": 2.0,
    }
    ad._persist_completion(event, {"status": "completed", "summary": "background task finished"})
    with ad._DB_LOCK, ad._connect() as conn:
        conn.execute(
            "UPDATE async_delegations SET owner_pid=? WHERE delegation_id=?",
            (os.getpid(), delegation_id),
        )
    return delegation_id


def test_owner_idle_requeue_backfills_routing_and_starts_parent_delivery_turn(tmp_path, monkeypatch):
    delegation_id = _persist_pending_with_missing_event_routing(tmp_path, monkeypatch)
    target_queue = queue.Queue()

    assert ad.requeue_local_pending_async_completions(target_queue) == 1
    event = target_queue.get_nowait()
    assert event["origin_session_id"] == ""
    assert event["session_key"] == "parent-key"
    assert event["origin_ui_session_id"] == "owner-tab"
    assert event["parent_session_id"] == "parent-session-id"

    owner = {
        "session_key": "parent-key",
        "history_lock": threading.Lock(),
        "running": True,
    }
    foreign = {
        "session_key": "foreign-key",
        "history_lock": threading.Lock(),
        "running": False,
    }
    assert server._session_owns_notification_event("owner-tab", owner, event) is True
    assert server._session_owns_notification_event("foreign-tab", foreign, event) is False

    submitted = []
    with patch.object(server, "_run_prompt_submit", side_effect=lambda *args, **kwargs: submitted.append((args, kwargs))):
        assert server._dispatch_idle_completion_batch(
            "delivery-rid",
            "owner-tab",
            owner,
            [(event, "async completion ready")],
            consumer="test-owner-idle-poller",
        ) is True

    assert submitted, "the owner poller did not start the parent delivery turn"
    durable = ad.get_durable_delegation(delegation_id)
    assert durable is not None
    assert durable["delivery_state"] == "delivered"
    assert durable["delivery_attempts"] >= 1


def test_pending_delivery_heartbeat_wakes_only_the_rightful_parent(tmp_path, monkeypatch):
    delegation_id = _persist_pending_with_missing_event_routing(tmp_path, monkeypatch)
    armed = []

    with patch("tools.runtime_heartbeat.runtime_heartbeat.outstanding_for_caller", return_value=[]), patch(
        "tools.runtime_heartbeat.runtime_heartbeat.arm",
        side_effect=lambda *args, **kwargs: armed.append((args, kwargs)) or True,
    ):
        assert ad.ensure_owner_pending_delivery_heartbeats() == 1

    assert armed[0][0] == (delegation_id,)
    assert armed[0][1]["caller_id"] == "parent-key"
    assert armed[0][1]["kind"] == "async_delivery"
    assert armed[0][1]["provider"] == "custom:test-provider"

    owner = {
        "session_key": "parent-key",
        "history_lock": threading.Lock(),
        "running": False,
    }
    foreign = {
        "session_key": "foreign-key",
        "history_lock": threading.Lock(),
        "running": False,
    }
    heartbeat = {
        "type": "heartbeat",
        "target_id": delegation_id,
        "target_kind": "async_delivery",
        "session_key": "parent-key",
        "status": "STUCK",
        "evidence": "async completion remains durable-pending after one warm-KV interval",
    }
    submitted = []
    with (
        patch.object(server, "_run_prompt_submit", side_effect=lambda *args, **kwargs: submitted.append((args, kwargs))),
        patch.object(server.runtime_heartbeat, "snapshot_active_targets", return_value=[]),
    ):
        server._handle_heartbeat_event("foreign-tab", foreign, heartbeat)
        server._handle_heartbeat_event("owner-tab", owner, heartbeat)

    assert len(submitted) == 1
    args, kwargs = submitted[0]
    assert args[1] == "owner-tab"
    assert kwargs["turn_origin"] == "heartbeat_warm"
    assert "durable-pending" in args[3]
