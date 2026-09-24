"""Admission reset ordering for busy prompt corrections (#315)."""
from __future__ import annotations

import asyncio
import contextlib
import threading
import types

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
        self.session_id = "owner-agent"
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self.client = None
        self._session_messages = None


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _session(agent: _Agent) -> dict:
    return {
        "agent": agent,
        "session_key": "owner-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        "active_session_lease": types.SimpleNamespace(enabled=False, released=True),
    }


def _patch_inline_turn(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a, **_k: False)


@pytest.mark.parametrize(
    ("result_steer", "late_steer", "expected"),
    [
        ("result correction", None, "result correction"),
        (None, "late correction", "late correction"),
        ("result correction", "late correction", "result correction\nlate correction"),
        (None, None, None),
    ],
)
def test_turn_finally_hands_off_result_then_late_steer(
    monkeypatch, tmp_path, result_steer: str | None, late_steer: str | None,
    expected: str | None,
):
    _patch_inline_turn(monkeypatch, tmp_path)
    agent = _Agent()
    result = {"final_response": "done"}
    if result_steer is not None:
        result["pending_steer"] = result_steer
    agent.run_conversation = lambda *_a, **_k: result
    session = _session(agent)
    real_finish = server._finish_turn

    def finish_then_late(sid, target_session, state):
        real_finish(sid, target_session, state)
        if late_steer is not None:
            assert agent.steer(late_steer)

    monkeypatch.setattr(server, "_finish_turn", finish_then_late)
    assert server._run_prompt_submit("rid", "owner-ui", session, "prompt") is True

    queued = session.get("queued_prompt")
    assert (queued or {}).get("text") == expected
    assert agent._drain_pending_steer() is None


@pytest.mark.parametrize("mode", ["steer", "interrupt"])
@pytest.mark.parametrize("exit_kind", ["normal", "early", "error", "cancel"])
def test_finishing_turn_never_drains_successor_correction(
    monkeypatch, tmp_path, mode: str, exit_kind: str,
):
    _patch_inline_turn(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: mode)
    agent = _Agent()
    session = _session(agent)
    correction = f"successor {mode} correction"
    admitted: dict[str, object] = {}

    if exit_kind == "normal":
        agent.run_conversation = lambda *_a, **_k: {"final_response": "done"}
    elif exit_kind == "error":
        def fail(*_args, **_kwargs):
            raise RuntimeError("turn failed")
        agent.run_conversation = fail
    elif exit_kind == "cancel":
        def cancel(*_args, **_kwargs):
            raise asyncio.CancelledError()
        agent.run_conversation = cancel
    else:
        agent.run_conversation = lambda *_a, **_k: {"final_response": "unused"}
        monkeypatch.setattr(server, "_prepare_turn_input", lambda *_a, **_k: None)

    def admit_and_correct(*_args, **_kwargs):
        session["running"] = True
        assert server._admit_prompt_turn(
            "owner-ui", session, "successor", None, None, None, None
        ) == ([], agent)
        agent._executing_tools = mode == "interrupt"
        response = server._handle_busy_submit(
            "rid-b", "owner-ui", session, correction, None
        )
        admitted["status"] = response["result"]["status"]

    monkeypatch.setattr(server, "_emit_settled_session_info", admit_and_correct)
    with contextlib.suppress(asyncio.CancelledError):
        assert server._run_prompt_submit("rid-a", "owner-ui", session, "turn A") is True

    assert admitted["status"] in {"steered", "redirected"}
    assert session["running"] is True
    assert session.get("queued_prompt") is None
    assert agent._drain_pending_steer() == correction
    assert agent._drain_pending_steer() is None


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

    def gated_start(target_session, text, **kwargs):
        original_start(target_session, text, **kwargs)
        admitted_inside_lock.set()
        assert release_admission.wait(2), "admission release timed out"

    def ordered_clear(*args, **kwargs):
        if not history_lock.held_by_current_thread():
            assert correction_done.wait(2), "busy correction did not reach the unlocked clear"
        return original_clear(*args, **kwargs)

    def admit() -> None:
        try:
            results["admit"] = server._admit_prompt_turn(
                "sid", session, "start turn", None, None, None, None
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