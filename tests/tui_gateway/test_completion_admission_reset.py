"""Admission reset ordering for busy prompt corrections (#315)."""
from __future__ import annotations

import threading

import pytest

from agent.interrupt_control import InterruptControlMixin
from tui_gateway import server


class _ObservedLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.owner: int | None = None
        self.waiter_attempted = threading.Event()

    def __enter__(self):
        if threading.current_thread().name == "busy-correction":
            self.waiter_attempted.set()
        self._lock.acquire()
        self.owner = threading.get_ident()
        return self

    def __exit__(self, *_args) -> None:
        self.owner = None
        self._lock.release()

    def held_by_current_thread(self) -> bool:
        return self.owner == threading.get_ident()


class _Agent(InterruptControlMixin):
    def __init__(self) -> None:
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()
        self._pending_redirect = None
        self._pending_redirect_lock = threading.Lock()
        self._model_request_active = threading.Event()
        self._model_request_active.set()
        self._executing_tools = False
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_interrupt_reason = None
        self._hard_interrupt_requested = threading.Event()
        self._interrupt_thread_signal_pending = False
        self._execution_thread_id = None
        self._supports_active_turn_redirect = True
        self.api_mode = "chat_completions"


@pytest.mark.parametrize(
    ("mode", "status", "drain_name"),
    [("steer", "steered", "_drain_pending_steer"),
     ("interrupt", "redirected", "_drain_pending_redirect")],
)
def test_admission_reset_precedes_waiting_busy_correction(
    monkeypatch, mode: str, status: str, drain_name: str,
):
    history_lock = _ObservedLock()
    agent = _Agent()
    session = {
        "agent": agent,
        "session_key": "admission-reset-session",
        "history": [],
        "history_lock": history_lock,
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "inflight_turn": None,
    }
    admitted_inside_lock = threading.Event()
    release_admission = threading.Event()
    correction_done = threading.Event()
    original_start = server._start_inflight_turn
    original_clear = agent.clear_interrupt
    results: dict[str, object] = {}

    def gated_start(target_session, text):
        original_start(target_session, text)
        admitted_inside_lock.set()
        assert release_admission.wait(2), "admission release timed out"

    def ordered_clear(*args, **kwargs):
        if not history_lock.held_by_current_thread():
            assert correction_done.wait(2), "busy correction did not reach the unlocked clear"
        return original_clear(*args, **kwargs)

    def admit() -> None:
        try:
            results["admit"] = server._admit_prompt_turn(
                "sid", session, "start turn", None, None
            )
        except BaseException as exc:  # surfaced after both threads are joined
            results["admit_error"] = exc

    def correct() -> None:
        try:
            results["correction"] = server._handle_busy_submit(
                "rid", "sid", session, "keep this correction", None
            )
        except BaseException as exc:  # surfaced after both threads are joined
            results["correction_error"] = exc
        finally:
            correction_done.set()

    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: mode)
    monkeypatch.setattr(server, "_start_inflight_turn", gated_start)
    monkeypatch.setattr(agent, "clear_interrupt", ordered_clear)
    admit_thread = threading.Thread(target=admit, name="admission")
    correction_thread = threading.Thread(target=correct, name="busy-correction")
    try:
        admit_thread.start()
        assert admitted_inside_lock.wait(2), "admission never entered its history critical section"
        correction_thread.start()
        assert history_lock.waiter_attempted.wait(2), "busy correction never waited on admission"
        assert correction_thread.is_alive()
        release_admission.set()
        admit_thread.join(4)
        correction_thread.join(4)
        assert not admit_thread.is_alive(), "admission deadlocked"
        assert not correction_thread.is_alive(), "busy correction deadlocked"
        assert "admit_error" not in results
        assert "correction_error" not in results
        assert results["admit"] == ([], agent)
        assert results["correction"]["result"]["status"] == status
        drain = getattr(agent, drain_name)
        assert drain() == "keep this correction"
        assert drain() is None
        assert session.get("queued_prompt") is None
    finally:
        release_admission.set()
        correction_done.set()
        admit_thread.join(4)
        correction_thread.join(4)