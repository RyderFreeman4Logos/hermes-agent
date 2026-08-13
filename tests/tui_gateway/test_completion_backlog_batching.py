"""Regression coverage for compression fences and completion FIFO."""

from __future__ import annotations

import threading

from tools.process_registry import ProcessRegistry
from tui_gateway import server


def _session() -> dict:
    return {"history_lock": threading.Lock(), "running": False}


def test_manual_compression_fence_release_is_generation_owned() -> None:
    """A stale finalizer cannot release a successor compression generation."""
    session = _session()

    first = server._begin_manual_compression_fence(session)
    server._finish_manual_compression_fence(session, first)

    second = server._begin_manual_compression_fence(session)
    server._finish_manual_compression_fence(session, first)

    assert session.get("_manual_compression_fence") is second
    assert session["running"] is True

    server._finish_manual_compression_fence(session, second)
    assert "_manual_compression_fence" not in session
    assert session["running"] is False


def test_owned_completion_requeue_keeps_owner_fifo_at_front() -> None:
    """A held owned event returns ahead of later owned completions."""
    registry = ProcessRegistry()
    foreign = {"owner": "foreign", "id": 0}
    first = {"owner": "self", "id": 1}
    second = {"owner": "self", "id": 2}
    registry.completion_queue.put(foreign)
    registry.completion_queue.put(first)
    registry.completion_queue.put(second)

    held = registry.get_completion_for_owner(
        lambda event: event["owner"] == "self", timeout=0
    )
    registry.requeue_completion_front(held)

    assert registry.completion_queue.get_nowait() is first
    assert registry.get_completion_for_owner(
        lambda event: event["owner"] == "self", timeout=0
    ) is second
    assert registry.completion_queue.get_nowait() is foreign
