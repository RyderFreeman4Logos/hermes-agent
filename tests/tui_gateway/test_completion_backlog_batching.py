"""Regressions for busy completion backlog batching (#61).

N process completions must not become N idle LLM turns. Steer acceptance is
not delivery; fallback is one bounded idle turn with per-event claims.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from tools.process_registry import ProcessRegistry
from tui_gateway import server

_REAL_NOTIFICATION_EVENT_BELONGS_ELSEWHERE = (
    server._notification_event_belongs_elsewhere
)
_REAL_NOTIFICATION_EVENT_REQUIRES_OWNER = server._notification_event_requires_owner
_REAL_SESSION_OWNS_NOTIFICATION_EVENT = server._session_owns_notification_event


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


def _delegation_event(delegation_id: str, *, summary: str) -> dict:
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "session-key",
        "origin_ui_session_id": "sid",
        "goal": "delegated task",
        "status": "completed",
        "summary": summary,
        "api_calls": 1,
        "duration_seconds": 1.0,
        "dispatched_at": 2.0,
        "completed_at": 3.0,
    }


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
def isolated_registry(monkeypatch, tmp_path):
    from tools import async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = ProcessRegistry()
    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        registry,
    )
    # Poller imports process_registry.process_registry at call time via module attr.
    import tools.process_registry as prm

    monkeypatch.setattr(prm, "process_registry", registry)
    monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _s: [])
    monkeypatch.setattr(
        server, "_notification_event_belongs_elsewhere", lambda *_a, **_k: False
    )
    monkeypatch.setattr(server, "_notification_event_requires_owner", lambda _e: True)
    monkeypatch.setattr(
        server, "_session_owns_notification_event", lambda *_a, **_k: True
    )
    monkeypatch.setattr(server, "_sync_runtime_heartbeat_status", lambda *a, **k: None)
    monkeypatch.setattr(server, "_NOTIFICATION_REQUEUE_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(server, "_COMPLETION_BACKLOG_COALESCE_WINDOW_SECONDS", 0.02)
    return registry


@pytest.mark.parametrize("count", [1, 2], ids=["single", "batch"])
def test_pure_async_batch_is_completion_delivery(count, monkeypatch):
    from agent.message_sanitization import COMPLETION_DELIVERY_INSTRUCTION
    from tools.process_registry import (
        completion_delivery_prompt,
        format_process_notification,
    )

    items = []
    for index in range(count):
        event = _delegation_event(f"deleg-{index}", summary=f"result-{index}")
        text = format_process_notification(event)
        assert text is not None
        items.append({
            "evt": event,
            "text": text,
            "model_text": completion_delivery_prompt(event, text),
        })

    prompt, is_delivery = server._compose_completion_batch_prompt(items)

    assert is_delivery is True
    assert prompt.endswith(COMPLETION_DELIVERY_INSTRUCTION)

    if count > 1:
        submitted = []
        monkeypatch.setattr(
            "tools.process_registry.claim_completion_event_delivery",
            lambda *_args, **_kwargs: "claim",
        )
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda _rid, _sid, _session, text, **kwargs: submitted.append(
                (text, kwargs["model_text"])
            ),
        )
        server._dispatch_completion_batch(
            "sid", _session(running=True), items, consumer="tui-test"
        )
        assert submitted == [(
            prompt.removesuffix(COMPLETION_DELIVERY_INSTRUCTION),
            prompt,
        )]


def test_busy_tool_chain_steers_once_and_skips_idle_duplicate(
    isolated_registry, monkeypatch
):
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


def test_idle_batch_transient_claim_failure_preserves_visible_fifo(
    isolated_registry, monkeypatch
):
    from tools import async_delegation as ad

    registry = isolated_registry
    events = [
        _completion_event(f"proc-claim-fifo-{seq}", started_at=float(seq))
        for seq in (1, 2, 3)
    ]
    registry.completion_queue.put(events[2])
    rendered = []

    def submit(_rid, _sid, session, text, **kwargs):
        rendered.append(text)
        kwargs["completion_delivery_callback"]("committed")
        session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    claim_calls = 0

    def transient_claim(_event, _consumer):
        nonlocal claim_calls
        claim_calls += 1
        if claim_calls == 1:
            raise OSError("claim storage unavailable")
        return ""

    monkeypatch.setattr(ad, "claim_event_delivery", transient_claim)
    session = _session(running=True)
    server._dispatch_completion_batch(
        "sid",
        session,
        [
            {"evt": events[0], "text": "owner-A-1"},
            {"evt": events[1], "text": "owner-A-2"},
        ],
        consumer="tui-test",
    )

    assert len(rendered) == 1
    assert rendered[0].index("owner-A-1") < rendered[0].index("owner-A-2")
    with registry.completion_queue.mutex:
        assert list(registry.completion_queue.queue) == [events[2]]


@pytest.mark.parametrize("failure_index", [0, 1])
def test_idle_batch_persistent_claim_failure_restores_untouched_tail(
    isolated_registry, monkeypatch, failure_index
):
    from tools import async_delegation as ad

    registry = isolated_registry
    events = [
        _completion_event(f"proc-claim-persistent-{seq}", started_at=float(seq))
        for seq in (1, 2, 3)
    ]
    registry.completion_queue.put(events[2])
    rendered = []

    def submit(_rid, _sid, session, text, **kwargs):
        rendered.append(text)
        kwargs["completion_delivery_callback"]("committed")
        session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", submit)

    def claim(event, _consumer):
        if event is events[failure_index]:
            raise OSError("claim storage unavailable")
        return ""

    monkeypatch.setattr(ad, "claim_event_delivery", claim)
    session = _session(running=True)
    server._dispatch_completion_batch(
        "sid",
        session,
        [
            {"evt": events[0], "text": "owner-A-1"},
            {"evt": events[1], "text": "owner-A-2"},
        ],
        consumer="tui-test",
    )

    if failure_index:
        assert len(rendered) == 1
        assert "owner-A-1" in rendered[0]
        expected = events[1:]
    else:
        assert rendered == []
        assert session["running"] is False
        expected = events
    with registry.completion_queue.mutex:
        remaining = list(registry.completion_queue.queue)
    assert remaining == expected
    assert all(actual is wanted for actual, wanted in zip(remaining, expected))


def test_idle_batch_provider_failure_restores_one_ordered_retry_unit(
    isolated_registry, monkeypatch
):
    from tools import async_delegation as ad

    registry = isolated_registry
    events = [
        _completion_event(f"proc-receipt-retry-{seq}", started_at=float(seq))
        for seq in (1, 2, 3)
    ]
    registry.completion_queue.put(events[2])
    monkeypatch.setattr(ad, "claim_event_delivery", lambda *_args: "")

    def submit(_rid, _sid, session, _text, **kwargs):
        kwargs["completion_delivery_callback"]("provider_failed")
        session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    server._dispatch_completion_batch(
        "sid",
        _session(running=True),
        [
            {"evt": events[0], "text": "owner-A-1"},
            {"evt": events[1], "text": "owner-A-2"},
        ],
        consumer="tui-test",
    )

    with registry.completion_queue.mutex:
        retries = list(registry.completion_queue.queue)
    assert [event["session_id"] for event in retries] == [
        event["session_id"] for event in events
    ]


def test_idle_batch_retries_transient_requeue_exception_without_drop(
    isolated_registry, monkeypatch
):
    from tools import async_delegation as ad

    registry = isolated_registry
    event = _completion_event("proc-requeue-outage", started_at=4.0)
    assert ad.persist_event_delivery(event)
    registry.completion_queue.put(event)
    selected = registry.completion_queue.get_nowait()
    requeue_calls = 0

    def fail_requeue(_events):
        nonlocal requeue_calls
        requeue_calls += 1
        raise OSError("requeue wrapper unavailable")

    monkeypatch.setattr(registry, "_requeue_completions_front", fail_requeue)

    def submit(_rid, _sid, session, _text, **kwargs):
        kwargs["completion_delivery_callback"]("provider_failed")
        session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    server._dispatch_completion_batch(
        "sid",
        _session(running=True),
        [{"evt": selected, "model_text": "completion", "text": "completion"}],
        consumer="tui-test",
    )

    assert requeue_calls == 3
    remaining = registry.completion_queue.get_nowait()
    assert remaining["session_id"] == event["session_id"]
    assert registry.completion_queue.empty()


def test_shutdown_formatter_error_restores_batch_before_concurrent_tail(
    isolated_registry, monkeypatch
):
    registry = isolated_registry
    malformed = _delegation_event("deleg-malformed-shutdown", summary="bad")
    malformed["is_batch"] = True
    malformed["results"] = [None]
    valid = _completion_event("proc-valid-shutdown", exit_code=1, started_at=2.0)
    concurrent = _completion_event(
        "proc-concurrent-shutdown", exit_code=1, started_at=3.0
    )
    registry.completion_queue.put(malformed)
    registry.completion_queue.put(valid)
    producer_start = threading.Event()
    producer_done = threading.Event()
    released = False

    def owns(_sid, _session, _event):
        nonlocal released
        if not released:
            released = True
            producer_start.set()
            assert producer_done.wait(2), "producer did not finish"
        return True

    monkeypatch.setattr(server, "_session_owns_notification_event", owns)

    def produce():
        assert producer_start.wait(2), "shutdown drain missed producer boundary"
        registry.completion_queue.put(concurrent)
        producer_done.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    stopped = threading.Event()
    stopped.set()
    try:
        server._notification_poller_loop(stopped, "sid", _session(running=False))
        producer.join(2)
        assert not producer.is_alive()
        with registry.completion_queue.mutex:
            remaining = list(registry.completion_queue.queue)
        expected = [malformed, valid, concurrent]
        assert remaining == expected
        assert all(actual is wanted for actual, wanted in zip(remaining, expected))
    finally:
        producer_start.set()
        producer.join(2)


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
    registry.completion_queue = queue.Queue(maxsize=2)
    heartbeats: list = []
    delivered: list = []
    monkeypatch.setattr(
        server,
        "_handle_heartbeat_event",
        lambda sid, session, evt: heartbeats.append(evt),
    )

    def submit(*args, **kwargs):
        delivered.append(args)
        kwargs["completion_delivery_callback"]("committed")

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
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
    assert registry.completion_queue.empty()


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


def test_manual_compression_fence_release_is_generation_owned():
    """A stale finalizer cannot release a successor compression generation."""
    session = _session()

    first = server._begin_manual_compression_fence(session)
    assert first is not None
    server._finish_manual_compression_fence(session, first)

    second = server._begin_manual_compression_fence(session)
    assert second is not first
    server._finish_manual_compression_fence(session, first)

    assert session.get("_manual_compression_fence") is second
    assert session["running"] is True

    server._finish_manual_compression_fence(session, second)
    assert "_manual_compression_fence" not in session
    assert session["running"] is False


@pytest.mark.parametrize(
    "outcome", ["success", "failure", "cancel", "noop", "lock_skip"]
)
def test_manual_compress_holds_completions_until_terminal_boundary(
    isolated_registry, monkeypatch, outcome
):
    """Manual compression owns the turn until every terminal path completes."""
    from agent import conversation_compression
    from tools import async_delegation as ad

    compression_entered = threading.Event()
    release_compression = threading.Event()
    terminal_entered = threading.Event()
    release_terminal = threading.Event()
    delivered = threading.Event()
    poller_waiting = threading.Event()
    event_dequeued = threading.Event()
    deliveries: list[tuple[str, dict]] = []

    get_completion_for_owner = isolated_registry.get_completion_for_owner

    def observed_get_completion_for_owner(*args, **kwargs):
        poller_waiting.set()
        item = get_completion_for_owner(*args, **kwargs)
        event_dequeued.set()
        return item

    monkeypatch.setattr(
        isolated_registry,
        "get_completion_for_owner",
        observed_get_completion_for_owner,
    )

    class _Agent:
        session_id = "session-key"
        api_mode = "codex_responses"
        tools: list = []
        _cached_system_prompt = ""
        _awaiting_cache_creation_usage = False
        _awaiting_cache_creation_base_input = 0

        def __init__(self):
            self.context_compressor = SimpleNamespace(
                _last_compression_telemetry=None,
                precomputed_token_count=None,
            )

        def _compress_context(self, messages, _system_prompt="", **_kwargs):
            compression_entered.set()
            assert release_compression.wait(5), "compression release timed out"
            if outcome == "failure":
                raise RuntimeError("forced compression failure")
            if outcome == "cancel":
                self.context_compressor._last_compression_telemetry = {
                    "commit_status": "aborted",
                    "failure_class": "commit_fence_cancelled",
                }
                return list(messages), ""
            if outcome == "lock_skip":
                self._compression_skipped_due_to_lock = "other-generation"
                return list(messages), ""
            self.context_compressor._last_compression_telemetry = {
                "commit_status": "committed"
            }
            return (
                [
                    {"role": "system", "content": "system"},
                    {"role": "assistant", "content": "compressed"},
                ],
                "",
            )

        def steer(self, _text):
            pytest.fail("a completion must not steer the manual compression turn")

    agent = _Agent()
    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    if outcome != "noop":
        history.extend([
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ])
    session = _session(agent=agent, history=history)
    monkeypatch.setitem(server._sessions, "sid", session)

    if outcome == "noop":
        record_noop = conversation_compression.record_compression_noop

        def held_noop(*args, **kwargs):
            compression_entered.set()
            assert release_compression.wait(5), "no-op release timed out"
            return record_noop(*args, **kwargs)

        monkeypatch.setattr(
            conversation_compression, "record_compression_noop", held_noop
        )

    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_a, **_k: None
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(
        "agent.interrupt_compat.request_hard_interrupt", lambda _agent: None
    )
    monkeypatch.setattr(
        "agent.model_metadata.estimate_request_tokens_rough", lambda *_a, **_k: 100
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *_a, **_k: 10_000
    )

    def status_update(_sid, kind, _text=None):
        if kind == "ready":
            terminal_entered.set()
            assert release_terminal.wait(5), "terminal release timed out"

    monkeypatch.setattr(server, "_status_update", status_update)

    def run_completion(_rid, _sid, sess, text, **kwargs):
        deliveries.append((text, kwargs))
        callback = kwargs.get("completion_delivery_callback")
        if callback is not None:
            callback("committed")
        with sess["history_lock"]:
            sess["running"] = False
        delivered.set()

    monkeypatch.setattr(server, "_run_prompt_submit", run_completion)
    monkeypatch.setattr(server, "_NOTIFICATION_QUEUE_WAIT_SECONDS", 0.01)

    process_event = _completion_event(
        f"proc-manual-compress-{outcome}", output="process-done"
    )
    delegation_event = _delegation_event(
        f"deleg-manual-compress-{outcome}", summary="subagent-done"
    )
    assert ad.persist_event_delivery(process_event)
    ad._persist_dispatch({
        "delegation_id": delegation_event["delegation_id"],
        "session_key": "session-key",
        "origin_ui_session_id": "sid",
        "parent_session_id": "session-key",
        "dispatched_at": 2.0,
    })
    ad._persist_completion(
        delegation_event, {"status": "completed", "summary": "subagent-done"}
    )

    command_result: dict = {}

    def run_compress():
        try:
            command_result["response"] = server._methods["session.compress"](
                "rid", {"session_id": "sid"}
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            command_result["exception"] = exc

    stop_poller = threading.Event()
    command_thread = threading.Thread(target=run_compress, daemon=True)
    poller_thread = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop_poller, "sid", session),
        daemon=True,
    )

    try:
        poller_thread.start()
        assert poller_waiting.wait(2)
        command_thread.start()
        assert compression_entered.wait(2), command_result
        first_fence = session.get("_manual_compression_fence")
        assert first_fence is not None
        if outcome == "failure":
            interrupt = server._methods["session.interrupt"](
                "stop", {"session_id": "sid"}
            )
            assert "error" not in interrupt
            assert session["running"] is True
            successor = server._methods["session.compress"](
                "second", {"session_id": "sid"}
            )
            assert successor.get("error", {}).get("code") == 4009
            assert session.get("_manual_compression_fence") is first_fence
        isolated_registry.completion_queue.put(process_event)
        isolated_registry.completion_queue.put(delegation_event)

        assert event_dequeued.wait(2)
        assert deliveries == []
        assert command_thread.is_alive()

        release_compression.set()
        assert terminal_entered.wait(2)
        with session["history_lock"]:
            assert session.get("_manual_compression_fence") is not None
        assert deliveries == []

        release_terminal.set()
        command_thread.join(2)
        assert not command_thread.is_alive()
        assert "exception" not in command_result
        assert delivered.wait(2)

        assert len(deliveries) == 1
        text, _kwargs = deliveries[0]
        assert text.count("process-done") == 1
        assert text.count("subagent-done") == 1
        assert text.index("process-done") < text.index("subagent-done")
        assert isolated_registry.completion_queue.empty()
        assert session["running"] is False
        assert "_manual_compression_fence" not in session
        assert "_manual_compression_fence_owner" not in session
        process_receipt = ad.get_durable_event_delivery(process_event)
        delegation_receipt = ad.get_durable_delegation(
            delegation_event["delegation_id"]
        )
        assert process_receipt is not None
        assert delegation_receipt is not None
        assert process_receipt["delivery_state"] == "delivered"
        assert delegation_receipt["delivery_state"] == "delivered"
        restarted = ProcessRegistry()
        assert restarted.completion_queue.empty()
    finally:
        release_compression.set()
        release_terminal.set()
        stop_poller.set()
        command_thread.join(2)
        poller_thread.join(2)


def test_foreign_poller_preserves_owner_fifo_and_unrelated_progress(
    isolated_registry, monkeypatch
):
    """A foreign poller cannot rotate an owner's durable completion FIFO."""
    from tools import async_delegation as ad

    registry = isolated_registry
    a_waiting = threading.Event()
    a_holds_first = threading.Event()

    get_completion_for_owner = registry.get_completion_for_owner

    def observed_get_completion_for_owner(*args, **kwargs):
        if threading.current_thread().name == "poller-A":
            a_waiting.set()
        item = get_completion_for_owner(*args, **kwargs)
        if (
            threading.current_thread().name == "poller-A"
            and item.get("session_id") == "proc-a-1"
        ):
            a_holds_first.set()
        return item

    monkeypatch.setattr(
        registry,
        "get_completion_for_owner",
        observed_get_completion_for_owner,
    )
    monkeypatch.setattr(
        server,
        "_notification_event_belongs_elsewhere",
        _REAL_NOTIFICATION_EVENT_BELONGS_ELSEWHERE,
    )
    monkeypatch.setattr(
        server,
        "_notification_event_requires_owner",
        _REAL_NOTIFICATION_EVENT_REQUIRES_OWNER,
    )
    monkeypatch.setattr(
        server,
        "_session_owns_notification_event",
        _REAL_SESSION_OWNS_NOTIFICATION_EVENT,
    )
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_NOTIFICATION_QUEUE_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(server, "_NOTIFICATION_REQUEUE_BACKOFF_SECONDS", 0.01)
    monkeypatch.setattr(server, "_COMPLETION_BACKLOG_COALESCE_WINDOW_SECONDS", 0.0)

    delivered: dict[str, list[str]] = {"A": [], "B": []}
    b_delivered = threading.Event()
    a_all_delivered = threading.Event()

    def run_completion(_rid, sid, session, text, **kwargs):
        delivered[sid].append(text)
        callback = kwargs.get("completion_delivery_callback")
        if callback is not None:
            callback("committed")
        with session["history_lock"]:
            session["running"] = False
        if sid == "B":
            b_delivered.set()
        elif len(delivered["A"]) == 3:
            a_all_delivered.set()

    monkeypatch.setattr(server, "_run_prompt_submit", run_completion)

    session_a = _session(session_key="conversation-A")
    session_b = _session(session_key="conversation-B")
    monkeypatch.setitem(server._sessions, "A", session_a)
    monkeypatch.setitem(server._sessions, "B", session_b)

    a_events = [
        _completion_event(
            f"proc-a-{position}",
            session_key="conversation-A",
            output=f"owner-{position}",
            origin_ui_session_id="A",
            started_at=float(position),
        )
        for position in (1, 2, 3)
    ]
    b_event = _completion_event(
        "proc-b",
        session_key="conversation-B",
        output="foreign-progress",
        origin_ui_session_id="B",
        started_at=10.0,
    )
    for event in (*a_events, b_event):
        assert ad.persist_event_delivery(event)

    stop_a = threading.Event()
    stop_b = threading.Event()
    stop_a_restarted = threading.Event()
    poller_a = threading.Thread(
        name="poller-A",
        target=server._notification_poller_loop,
        args=(stop_a, "A", session_a),
        daemon=True,
    )
    poller_b = threading.Thread(
        name="poller-B",
        target=server._notification_poller_loop,
        args=(stop_b, "B", session_b),
        daemon=True,
    )
    poller_a_restarted = threading.Thread(
        name="poller-A-restarted",
        target=server._notification_poller_loop,
        args=(stop_a_restarted, "A", session_a),
        daemon=True,
    )
    fence = None
    try:
        poller_a.start()
        assert a_waiting.wait(2)
        fence = server._begin_manual_compression_fence(session_a)
        registry.completion_queue.put(a_events[0])
        assert a_holds_first.wait(2)

        registry.completion_queue.put(a_events[1])
        registry.completion_queue.put(b_event)
        registry.completion_queue.put(a_events[2])
        poller_b.start()
        assert b_delivered.wait(3)
        assert delivered["A"] == []

        stop_b.set()
        poller_b.join(2)
        assert not poller_b.is_alive()
        stop_a.set()
        poller_a.join(2)
        assert not poller_a.is_alive()

        poller_a_restarted.start()
        server._finish_manual_compression_fence(session_a, fence)
        assert a_all_delivered.wait(3)
        assert len(delivered["A"]) == 3
        assert len(delivered["B"]) == 1
        owner_text = "\n".join(delivered["A"])
        assert all(owner_text.count(f"owner-{position}") == 1 for position in (1, 2, 3))
        positions = [owner_text.index(f"owner-{position}") for position in (1, 2, 3)]
        assert positions == sorted(positions)

        for event in (*a_events, b_event):
            receipt = ad.get_durable_event_delivery(event)
            assert receipt is not None
            assert receipt["delivery_state"] == "delivered"
        restarted = ProcessRegistry()
        assert restarted.completion_queue.empty()
    finally:
        stop_a.set()
        stop_b.set()
        stop_a_restarted.set()
        if fence is not None:
            server._finish_manual_compression_fence(session_a, fence)
        poller_a.join(2)
        poller_b.join(2)
        poller_a_restarted.join(2)


def test_shutdown_foreign_poller_preserves_fifo_with_concurrent_producer(
    isolated_registry, monkeypatch
):
    """A stopped foreign poller cannot append old completions after a new one."""
    from tools import async_delegation as ad

    producer_start = threading.Event()
    producer_done = threading.Event()

    class InterleavingQueue(queue.Queue):
        def __init__(self):
            super().__init__()
            self.legacy_gets = 0
            self.triggered = False

        def get_nowait(self):
            self.legacy_gets += 1
            return super().get_nowait()

        def release_during_selective_drain(self):
            if not self.legacy_gets and not self.triggered:
                self.triggered = True
                producer_start.set()
                assert producer_done.wait(2), "producer did not finish"

        def empty(self):
            with self.mutex:
                observed_empty = not self._qsize()
            if observed_empty and self.legacy_gets and not self.triggered:
                self.triggered = True
                producer_start.set()
                assert producer_done.wait(2), "producer did not finish"
                return True
            return observed_empty

    registry = isolated_registry
    interleaving_queue = InterleavingQueue()
    registry.completion_queue = interleaving_queue
    monkeypatch.setattr(
        server,
        "_notification_event_requires_owner",
        _REAL_NOTIFICATION_EVENT_REQUIRES_OWNER,
    )
    monkeypatch.setattr(
        server,
        "_session_owns_notification_event",
        _REAL_SESSION_OWNS_NOTIFICATION_EVENT,
    )

    real_belongs_elsewhere = _REAL_NOTIFICATION_EVENT_BELONGS_ELSEWHERE

    def belongs_elsewhere(sid, session, event):
        interleaving_queue.release_during_selective_drain()
        return real_belongs_elsewhere(sid, session, event)

    monkeypatch.setattr(
        server, "_notification_event_belongs_elsewhere", belongs_elsewhere
    )
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_NOTIFICATION_QUEUE_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(server, "_COMPLETION_BACKLOG_COALESCE_WINDOW_SECONDS", 0.0)

    session_a = _session(session_key="conversation-A")
    session_b = _session(session_key="conversation-B")
    monkeypatch.setitem(server._sessions, "A", session_a)
    monkeypatch.setitem(server._sessions, "B", session_b)
    fence = server._begin_manual_compression_fence(session_a)

    events = [
        _completion_event(
            f"proc-a-{position}",
            session_key="conversation-A",
            origin_ui_session_id="A",
            output=f"owner-{position}",
            started_at=float(position),
        )
        for position in (1, 2, 3)
    ]
    for event in events:
        assert ad.persist_event_delivery(event)
    for event in events[:2]:
        interleaving_queue.put(event)

    delivered: list[int] = []
    delivery_done = threading.Event()

    def run_completion(_rid, _sid, session, text, **kwargs):
        positions = [
            (text.index(f"owner-{position}"), position)
            for position in (1, 2, 3)
            if f"owner-{position}" in text
        ]
        delivered.extend(position for _index, position in sorted(positions))
        callback = kwargs.get("completion_delivery_callback")
        assert callback is not None
        callback("committed")
        with session["history_lock"]:
            session["running"] = False
        if len(delivered) == 3:
            delivery_done.set()

    monkeypatch.setattr(server, "_run_prompt_submit", run_completion)

    def produce():
        assert producer_start.wait(2), "shutdown drain did not reach producer boundary"
        interleaving_queue.put(events[2])
        producer_done.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    stop_b = threading.Event()
    stop_b.set()
    foreign = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop_b, "B", session_b),
        daemon=True,
    )
    poller_a = None
    stop_a = None
    foreign.start()
    try:
        foreign.join(2)
        assert not foreign.is_alive()
        producer.join(2)
        assert not producer.is_alive()
        with interleaving_queue.mutex:
            physical_order = [
                int(event["session_id"].rsplit("-", 1)[1])
                for event in interleaving_queue.queue
            ]

        stop_a = threading.Event()
        poller_a = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop_a, "A", session_a),
            daemon=True,
        )
        poller_a.start()
        server._finish_manual_compression_fence(session_a, fence)
        assert delivery_done.wait(3), delivered

        receipts = [ad.get_durable_event_delivery(event) for event in events]
        for receipt in receipts:
            assert receipt is not None
            assert receipt["delivery_state"] == "delivered"
            assert receipt["delivery_attempts"] == 1
        assert physical_order == [1, 2, 3] and delivered == [1, 2, 3], (
            f"shutdown drain reordered owner A: queue={physical_order}, "
            f"visible={delivered}"
        )
        restarted = ProcessRegistry()
        assert restarted.completion_queue.empty()
    finally:
        producer_start.set()
        producer.join(2)
        server._finish_manual_compression_fence(session_a, fence)
        stop_b.set()
        foreign.join(2)
        if poller_a is not None and stop_a is not None:
            stop_a.set()
            poller_a.join(2)
