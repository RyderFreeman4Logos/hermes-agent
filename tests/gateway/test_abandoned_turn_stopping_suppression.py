"""Public notification proof for abandoned kills that return ``stopping``."""

import asyncio
import shlex
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
import tools.process_registry as process_registry_module
import tui_gateway.session_notifications as tui_notifications
from tools.process_registry import ProcessRegistry


def _wait_until(predicate, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _spawn_late_pipe(
    registry, monkeypatch, tmp_path, *, task_id, marker, settlement_stage
):
    """Spawn a real pipe child and hold its reader before terminal settlement."""
    entered_finish = threading.Event()
    release_finish = threading.Event()
    if settlement_stage == "before-reader-ready":
        original_finish = registry._finish_reader

        def held_finish(*args, **kwargs):
            entered_finish.set()
            assert release_finish.wait(8), "test did not release the pipe reader"
            return original_finish(*args, **kwargs)

        monkeypatch.setattr(registry, "_finish_reader", held_finish)
    code = (
        "import signal, time\n"
        "def finish(_signum, _frame):\n"
        f"    print({marker!r}, flush=True)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, finish)\n"
        f"print({'READY-' + marker!r}, flush=True)\n"
        "while True: time.sleep(0.05)\n"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(process_registry_module, "save_completed_result", lambda _session: None)
    session = registry.spawn_local(
        command,
        cwd=str(tmp_path),
        task_id=task_id,
        session_key="test-session",
    )
    session.notify_on_complete = True
    assert _wait_until(lambda: f"READY-{marker}" in session.output_buffer)
    enable_completion_wait = lambda: None
    if settlement_stage == "before-reader-ready":
        monkeypatch.setattr(
            session._reader_settlement_ready, "wait", lambda timeout=None: False
        )
    else:
        original_completion_wait = session._completion_event.wait
        fast_timeout = [True]

        def controlled_completion_wait(timeout=None):
            if fast_timeout[0]:
                return False
            return original_completion_wait(timeout)

        def held_disposition_wait(timeout=None):
            entered_finish.set()
            return release_finish.wait(timeout=8)

        monkeypatch.setattr(session._completion_event, "wait", controlled_completion_wait)
        monkeypatch.setattr(
            session._kill_disposition_event, "wait", held_disposition_wait
        )
        enable_completion_wait = lambda: fast_timeout.__setitem__(0, False)
    return session, entered_finish, release_finish, enable_completion_wait


def _completion_event(registry, session):
    assert session._completion_event.wait(timeout=4), "late reader did not settle"
    assert _wait_until(lambda: not registry.completion_queue.empty())
    event = registry.completion_queue.get_nowait()
    assert event["session_id"] == session.id
    return event


async def _exercise_notification_consumers(monkeypatch, registry, session, event):
    tui_dispatches = []
    monkeypatch.setattr(
        tui_notifications, "_emit", lambda *_args, **_kwargs: None, raising=False
    )
    monkeypatch.setattr(tui_notifications, "_notif_claim_turn", lambda _session: True)
    monkeypatch.setattr(
        tui_notifications,
        "_notif_dispatch_event",
        lambda *_args: tui_dispatches.append(event),
    )
    assert tui_notifications._notif_handle_event(
        "test-live-session",
        {},
        event,
        set(),
        registry,
        lambda _event: "completion text",
        None,
        owned=True,
    )

    enqueue = AsyncMock(return_value=True)
    runner = SimpleNamespace(
        _load_background_notifications_mode=lambda: "concise",
        _build_process_completion_event=lambda *_args: event,
        _enqueue_process_completion_notification=enqueue,
    )
    await gateway_run.GatewayRunner._run_process_watcher(
        runner,
        {
            "session_id": session.id,
            "check_interval": 0,
            "platform": "test",
            "chat_id": "test-chat",
            "thread_id": "",
            "notify_on_complete": True,
        },
    )
    return tui_dispatches, enqueue


@pytest.mark.linux_only
@pytest.mark.parametrize(
    "settlement_stage", ["before-reader-ready", "after-reader-ready"]
)
def test_abandoned_timeout_stopping_suppresses_late_gateway_and_tui_turns(
    monkeypatch, tmp_path, settlement_stage
):
    registry = ProcessRegistry()
    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    monkeypatch.setattr(gateway_run, "_dump_wedged_turn_stacks", lambda _task_id: None)
    session, entered_finish, release_finish, enable_completion_wait = _spawn_late_pipe(
        registry,
        monkeypatch,
        tmp_path,
        task_id="abandoned-turn",
        marker="ABANDONED-LATE-41",
        settlement_stage=settlement_stage,
    )
    worker_done = threading.Event()
    timeout_fired = threading.Event()
    try:
        assert gateway_run._abandon_timed_out_gateway_turn(
            agent_holder=[SimpleNamespace(interrupt=lambda _reason: None)],
            task_id="abandoned-turn",
            process_baseline=frozenset(),
            worker_done=worker_done,
            timeout_fired=timeout_fired,
            cleanup_lock=threading.Lock(),
        )
        assert timeout_fired.is_set()
        assert entered_finish.wait(timeout=2)
        assert not session._completion_event.is_set()
        enable_completion_wait()
        release_finish.set()
        event = _completion_event(registry, session)

        tui_dispatches, gateway_enqueue = asyncio.run(
            _exercise_notification_consumers(monkeypatch, registry, session, event)
        )

        assert registry.is_completion_consumed(session.id)
        assert tui_dispatches == []
        gateway_enqueue.assert_not_awaited()
    finally:
        release_finish.set()
        session.process.wait(timeout=4)
        session._reader_thread.join(timeout=4)


@pytest.mark.linux_only
def test_normal_stopping_kill_keeps_late_gateway_and_tui_notification(
    monkeypatch, tmp_path
):
    registry = ProcessRegistry()
    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    session, entered_finish, release_finish, enable_completion_wait = _spawn_late_pipe(
        registry,
        monkeypatch,
        tmp_path,
        task_id="normal-turn",
        marker="NORMAL-LATE-41",
        settlement_stage="before-reader-ready",
    )
    try:
        result = registry.kill_process(
            session.id,
            source="normal.explicit.kill",
            consume_output=True,
        )
        assert result["status"] == "stopping"
        assert entered_finish.wait(timeout=2)
        assert not session._completion_event.is_set()
        enable_completion_wait()
        release_finish.set()
        event = _completion_event(registry, session)

        tui_dispatches, gateway_enqueue = asyncio.run(
            _exercise_notification_consumers(monkeypatch, registry, session, event)
        )

        assert not registry.is_completion_consumed(session.id)
        assert tui_dispatches == [event]
        gateway_enqueue.assert_awaited_once()
    finally:
        release_finish.set()
        session.process.wait(timeout=4)
        session._reader_thread.join(timeout=4)
