"""Deterministic regressions for the manual-compression completion fence."""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest

from tools.process_registry import process_registry
from tui_gateway import server


def _session(**extra):
    session = {
        "agent": SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "_finalized": False,
    }
    session.update(extra)
    return session


def _delegation_event():
    return {
        "type": "async_delegation",
        "delegation_id": "delegation-1",
        "session_key": "session-key",
        "origin_ui_session_id": "sid",
        "status": "completed",
        "summary": "delegated completion",
    }


def test_manual_compression_fence_is_generation_owned_and_busy_is_fail_closed():
    session = _session()

    first = server._begin_manual_compression_fence(session)
    assert isinstance(first, threading.Event)
    assert session["running"] is True
    with pytest.raises(RuntimeError, match="/interrupt"):
        server._begin_manual_compression_fence(session)

    server._finish_manual_compression_fence(session, threading.Event())
    assert session["_manual_compression_fence"] is first
    assert session["running"] is True

    server._finish_manual_compression_fence(session, first)
    assert first.is_set()
    assert "_manual_compression_fence" not in session
    assert session["running"] is False

    second = server._begin_manual_compression_fence(session)
    server._finish_manual_compression_fence(session, first)
    assert session["_manual_compression_fence"] is second
    assert session["running"] is True
    server._finish_manual_compression_fence(session, second)
    assert second.is_set()
    assert "_manual_compression_fence" not in session
    assert session["running"] is False


def test_poller_holds_dequeued_event_until_fence_release(monkeypatch):
    class ObservedQueue(queue.Queue):
        def __init__(self):
            super().__init__()
            self.dequeued = threading.Event()
            self.allow_return = threading.Event()
            self.requeued = threading.Event()

        def put(self, item, *args, **kwargs):
            if self.dequeued.is_set():
                self.requeued.set()
            return super().put(item, *args, **kwargs)

    isolated = ObservedQueue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    real_get_completion = process_registry.get_completion_for_owner

    def held_get(owns_event, *, timeout=None):
        event = real_get_completion(owns_event, timeout=timeout)
        isolated.dequeued.set()
        assert isolated.allow_return.wait(2)
        return event

    monkeypatch.setattr(process_registry, "get_completion_for_owner", held_get)
    monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server, "_notification_event_belongs_elsewhere", lambda *_args: False
    )
    monkeypatch.setattr(
        server, "_notification_event_requires_owner", lambda _event: False
    )

    delivered = []
    monkeypatch.setattr(
        server,
        "_submit_process_notification_turn",
        lambda _sid, _session, event, _text: delivered.append(event) or True,
    )

    session = _session()
    event = _delegation_event()
    isolated.put(event)
    stop = threading.Event()
    poller = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, "sid", session),
        daemon=True,
    )
    poller.start()
    fence = None
    try:
        assert isolated.dequeued.wait(2)
        fence = server._begin_manual_compression_fence(session)
        isolated.allow_return.set()
        assert not isolated.requeued.wait(0.2)
        assert delivered == []

        server._finish_manual_compression_fence(session, fence)
        stop.set()
        assert poller.join(2) is None
        assert delivered == [event]
        assert not poller.is_alive()
    finally:
        stop.set()
        isolated.allow_return.set()
        if fence is not None:
            server._finish_manual_compression_fence(session, fence)
        poller.join(2)


def test_owner_selection_preserves_foreign_fifo(monkeypatch):
    isolated = queue.Queue()
    foreign = {"type": "completion", "session_key": "foreign"}
    owner = {"type": "completion", "session_key": "owner"}
    owner_after = {"type": "async_delegation", "session_key": "owner"}
    for event in (foreign, owner, owner_after):
        isolated.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated)

    selected = process_registry.get_completion_for_owner(
        lambda event: event.get("session_key") == "owner", timeout=0.1
    )
    assert selected is owner
    assert isolated.get_nowait() is foreign
    assert isolated.get_nowait() is owner_after

    process_registry.requeue_completion_front(selected)
    assert isolated.get_nowait() is selected
    assert isolated.empty()
