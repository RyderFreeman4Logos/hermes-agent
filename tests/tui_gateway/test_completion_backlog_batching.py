"""Regressions for busy completion backlog batching (#61).

N process completions must not become N idle LLM turns. Steer acceptance is
not delivery; fallback is one bounded idle turn with per-event claims.
"""

from __future__ import annotations

import queue as queue_mod
import threading
import time
from types import SimpleNamespace

import pytest

from tools.process_registry import ProcessRegistry
from tui_gateway import server


def _completion_event(session_id: str, *, session_key: str = "session-key", exit_code: int = 0, output: str = "ok", **extra):
    event = {
        "type": "completion",
        "session_id": session_id,
        "session_key": session_key,
        "started_at": 1.0 + hash(session_id) % 1000 / 1000.0,
        "command": f"cmd-{session_id}",
        "exit_code": exit_code,
        "completion_reason": "exited",
        "termination_source": "",
        "output": output,
    }
    event.update(extra)
    return event


def _session(**extra):
    base = {
        "agent": SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "_finalized": False,
    }
    base.update(extra)
    return base


class _StopAfter:
    """Stop after *events* have been observed (queue get successes).

    The poller also probes ``is_set`` once per loop for liveness, so a
    simple check-counter is too brittle. Count only when we want the
    next loop entry to fail.
    """

    def __init__(self, loops: int):
        self._loops = loops
        self._checks = 0

    def is_set(self):
        # Called twice per iteration (while + _notification_poller_is_current).
        self._checks += 1
        # Allow ``loops`` full iterations (2 checks each), then stop.
        return self._checks > (self._loops * 2)


@pytest.fixture
def isolated_registry(monkeypatch):
    registry = ProcessRegistry()
    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        registry,
    )
    # Poller imports process_registry.process_registry at call time via module attr.
    import tools.process_registry as prm

    monkeypatch.setattr(prm, "process_registry", registry)
    monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _s: [])
    monkeypatch.setattr(server, "_notification_event_belongs_elsewhere", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_notification_event_requires_owner", lambda _e: True)
    monkeypatch.setattr(server, "_session_owns_notification_event", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_sync_runtime_heartbeat_status", lambda *a, **k: None)
    monkeypatch.setattr(server, "_NOTIFICATION_REQUEUE_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(server, "_COMPLETION_BACKLOG_COALESCE_WINDOW_SECONDS", 0.02)
    return registry


def test_busy_tool_chain_steers_once_and_skips_idle_duplicate(isolated_registry, monkeypatch):
    """Several completions during one long tool chain apply exactly once."""
    registry = isolated_registry
    steered: list[str] = []
    delivered: list[tuple[str, dict]] = []
    applied_messages: list = []

    class _Agent:
        def __init__(self):
            self._pending_steer = None
            self._pending_steer_lock = threading.Lock()

        def steer(self, text: str) -> bool:
            steered.append(text)
            with self._pending_steer_lock:
                if self._pending_steer:
                    self._pending_steer = self._pending_steer + "\n" + text
                else:
                    self._pending_steer = text
            return True

        def _drain_pending_steer(self):
            with self._pending_steer_lock:
                text = self._pending_steer
                self._pending_steer = None
            return text

        def _apply_pending_steer_to_tool_results(self, messages, num_tool_msgs):
            from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results

            apply_pending_steer_to_tool_results(self, messages, num_tool_msgs)
            applied_messages.extend(messages)

    agent = _Agent()
    sess = _session(agent=agent, running=True, session_key="session-key")
    events = [
        _completion_event(f"proc_chain_{i}", output=f"out-{i}")
        for i in range(1, 4)
    ]
    for evt in events:
        registry.completion_queue.put(evt)

    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **kwargs: delivered.append((text, kwargs)),
    )
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    # Drain while busy — steer path, no idle turns.
    server._notification_poller_loop(_StopAfter(3), "sid", sess)
    assert len(steered) == 3
    assert delivered == []
    assert registry.completion_queue.empty()
    assert len(sess.get("_completion_steer_pending") or []) == 3

    # Tool boundary applies the steers, but terminal publication still owns ACK.
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "tool-out"},
    ]
    agent._apply_pending_steer_to_tool_results(messages, 1)
    assert "out-1" in messages[-1]["content"]
    assert "out-3" in messages[-1]["content"]
    pending = sess.get("_completion_steer_pending") or []
    assert len(pending) == 3
    assert {item["state"] for item in pending} == {"applied"}
    from tools.async_delegation import get_durable_event_delivery

    for evt in events:
        receipt = get_durable_event_delivery(evt)
        assert receipt is not None
        assert receipt["delivery_state"] == "effect_started"

    server._finish_steered_completion_claims(sess, "committed")
    assert sess.get("_completion_steer_pending") in (None, [])
    for evt in events:
        receipt = get_durable_event_delivery(evt)
        assert receipt is not None
        assert receipt["delivery_state"] == "delivered"
        assert not registry.completion_event_should_deliver(evt)

    # Idle poll after the terminal fence must not open a duplicate turn.
    sess["running"] = False
    server._notification_poller_loop(_StopAfter(1), "sid", sess)
    assert delivered == []


def test_turn_end_before_apply_batches_fallback_once(isolated_registry, monkeypatch):
    """Active turn ends before pending steer drains → one bounded fallback turn."""
    registry = isolated_registry
    delivered: list[tuple[str, dict]] = []

    class _Agent:
        def __init__(self):
            self._pending_steer = None
            self._pending_steer_lock = threading.Lock()

        def steer(self, text: str) -> bool:
            with self._pending_steer_lock:
                if self._pending_steer:
                    self._pending_steer = self._pending_steer + "\n" + text
                else:
                    self._pending_steer = text
            return True

        def _drain_pending_steer(self):
            with self._pending_steer_lock:
                text = self._pending_steer
                self._pending_steer = None
            return text

    agent = _Agent()
    sess = _session(agent=agent, running=True)
    events = [_completion_event(f"proc_fb_{i}", output=f"fb-{i}") for i in range(1, 4)]
    for evt in events:
        registry.completion_queue.put(evt)

    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, session, text, **kwargs: (
            delivered.append((text, kwargs)),
            session.__setitem__("running", False),
        ),
    )

    server._notification_poller_loop(_StopAfter(3), "sid", sess)
    assert len(sess.get("_completion_steer_pending") or []) == 3
    assert delivered == []

    leftover = agent._drain_pending_steer()
    assert leftover
    sess["running"] = False
    server._finalize_steered_completion_fallback(sess, leftover)
    items = server._take_steered_completion_fallback(sess)
    assert len(items) == 3
    with sess["history_lock"]:
        sess["running"] = True
    server._dispatch_completion_batch("sid", sess, items, consumer="tui-test")
    assert len(delivered) == 1
    text, kwargs = delivered[0]
    assert "fb-1" in text and "fb-2" in text and "fb-3" in text
    assert kwargs.get("completion_delivery") is True
    for evt in events:
        assert not registry.completion_event_should_deliver(evt)


def test_idle_coalesce_overflow_is_lossless(isolated_registry, monkeypatch):
    registry = isolated_registry
    delivered: list[str] = []
    monkeypatch.setattr(server, "_COMPLETION_BACKLOG_MAX_ITEMS", 2)
    monkeypatch.setattr(server, "_COMPLETION_BACKLOG_MAX_CHARS", 10_000)
    monkeypatch.setattr(server, "_COMPLETION_BACKLOG_COALESCE_WINDOW_SECONDS", 0.05)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    def _deliver(_rid, _sid, session, text, **_kwargs):
        delivered.append(text)
        session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", _deliver)

    sess = _session(running=False)
    events = [
        _completion_event(f"proc_ov_{i}", output=f"o{i}", started_at=float(i))
        for i in range(1, 5)
    ]
    for evt in events:
        registry.completion_queue.put(evt)

    server._notification_poller_loop(_StopAfter(6), "sid", sess)
    # Two batches of 2 (or first batch 2 + second batch remaining).
    assert len(delivered) >= 2
    joined = "\n".join(delivered)
    for i in range(1, 5):
        assert f"proc_ov_{i}" in joined
    assert registry.completion_queue.empty()
    for evt in events:
        assert not registry.completion_event_should_deliver(evt)


def test_failure_visible_success_can_silent_complete(isolated_registry, monkeypatch):
    registry = isolated_registry
    delivered: list[tuple[str, dict]] = []
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, session, text, **kwargs: (
            delivered.append((text, kwargs)),
            session.__setitem__("running", False),
        ),
    )
    # Force success path silent via completion_delivery_prompt None.
    monkeypatch.setattr(
        "tools.process_registry.completion_delivery_prompt",
        lambda evt, payload: None if evt.get("exit_code") == 0 else payload,
    )

    ok = _completion_event("proc_ok", exit_code=0, output="fine")
    bad = _completion_event("proc_bad", exit_code=1, output="boom", started_at=2.0)
    registry.completion_queue.put(ok)
    registry.completion_queue.put(bad)
    sess = _session(running=False)
    server._notification_poller_loop(_StopAfter(4), "sid", sess)

    assert not registry.completion_event_should_deliver(ok)
    assert len(delivered) == 1
    assert "boom" in delivered[0][0] or "proc_bad" in delivered[0][0]
    assert "exit code 1" in delivered[0][0]


def test_heartbeat_path_unaffected_by_completion_batching(isolated_registry, monkeypatch):
    registry = isolated_registry
    heartbeats: list = []
    delivered: list = []
    monkeypatch.setattr(
        server,
        "_handle_heartbeat_event",
        lambda sid, session, evt: heartbeats.append(evt),
    )
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *a, **k: delivered.append(a),
    )
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    hb = {
        "type": "heartbeat",
        "session_key": "session-key",
        "target_id": "proc-hb",
        "status": "ALIVE",
        "evidence": "alive",
    }
    registry.completion_queue.put(hb)
    registry.completion_queue.put(_completion_event("proc_after_hb", started_at=3.0))
    sess = _session(running=False)
    server._notification_poller_loop(_StopAfter(4), "sid", sess)

    assert len(heartbeats) == 1
    assert heartbeats[0]["type"] == "heartbeat"
    assert len(delivered) == 1


def test_busy_pending_steer_does_not_block_later_queue_events(isolated_registry, monkeypatch):
    """A retained busy-steer event must not starve later completions."""
    registry = isolated_registry
    steered: list[str] = []

    class _Agent:
        def __init__(self):
            self._pending_steer = None
            self._pending_steer_lock = threading.Lock()

        def steer(self, text: str) -> bool:
            steered.append(text)
            with self._pending_steer_lock:
                self._pending_steer = (
                    (self._pending_steer + "\n" + text) if self._pending_steer else text
                )
            return True

        def _apply_pending_steer_to_tool_results(self, messages, num_tool_msgs):
            return None

    agent = _Agent()
    sess = _session(agent=agent, running=True)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no idle turn while busy")),
    )

    first = _completion_event("proc_first", output="first")
    second = _completion_event("proc_second", output="second", started_at=2.0)
    registry.completion_queue.put(first)
    registry.completion_queue.put(second)
    server._notification_poller_loop(_StopAfter(3), "sid", sess)

    assert len(steered) == 2
    assert registry.completion_queue.empty()
    pending = sess.get("_completion_steer_pending") or []
    assert {p["evt"]["session_id"] for p in pending} == {"proc_first", "proc_second"}
