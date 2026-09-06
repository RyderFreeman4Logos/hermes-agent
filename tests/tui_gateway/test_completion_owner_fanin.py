"""#62 fan-in timeout must ride #151 owner-FIFO dequeue."""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

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


def test_pending_live_child_fanin_uses_owner_get_timeout(monkeypatch):
    """Pending child tails extend get_completion_for_owner, never queue.get."""
    isolated = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    seen: list[float | None] = []

    def spy(_owns_event, *, timeout=None):
        seen.append(timeout)
        raise queue.Empty

    def boom(*_args, **_kwargs):
        raise AssertionError("poller used completion_queue.get")

    monkeypatch.setattr(process_registry, "get_completion_for_owner", spy)
    monkeypatch.setattr(isolated, "get", boom)
    monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_loop_tick", lambda *_args: None)
    if hasattr(server, "_is_live_delegate_child_completion"):
        monkeypatch.setattr(
            server, "_is_live_delegate_child_completion", lambda *_args: True
        )
    if hasattr(server, "_session_can_steer_completions"):
        monkeypatch.setattr(
            server, "_session_can_steer_completions", lambda _session: False
        )

    child = {"type": "completion", "session_id": "proc_child"}
    session = _session(_completion_pending=[child])
    stop = threading.Event()
    poller = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, "sid", session),
        daemon=True,
    )
    poller.start()
    try:
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            time.sleep(0.01)
        assert seen, "owner get was never called"
        assert all(t is not None and 0 <= t <= 2.0 for t in seen)
        assert len(seen) >= 2
    finally:
        stop.set()
        poller.join(2)
        assert not poller.is_alive()
