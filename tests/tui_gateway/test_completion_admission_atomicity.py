"""Completion admission is atomic with Stop and respects its owning profile's OFF policy."""
from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from agent import turn_context
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from tests.tui_gateway.test_completion_public_lifecycle import (
    _agent, _completion, _gateway_runtime, _response, _session, _user_rows,
)
from tools.process_registry import process_registry
from tui_gateway import server


@pytest.mark.parametrize("boundary", ["before_insert", "after_insert"])
def test_public_stop_and_core_completion_insert_are_one_boundary(tmp_path, monkeypatch, boundary):
    sid, session_id, event_id = "atomic-ui", "atomic-session", "atomic-completion"
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="tui", model="test/model")
    agent = _agent(db, session_id)
    agent._interruptible_api_call = lambda _kwargs: _response()
    session = _session(agent, session_id)
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    stop, retry_stop = threading.Event(), threading.Event()
    replies = []
    threads = []
    stale_appends = []
    first_worker = []
    original = getattr(turn_context, "append_message" if boundary == "after_insert" else "_stage_turn_user_message")
    original_append = turn_context.append_message

    def observe_append(messages, message, *args, **kwargs):
        if (first_worker and threading.current_thread() is first_worker[0]
                and event_id in str(message.get("content"))):
            stale_appends.append(message)
        return original_append(messages, message, *args, **kwargs)

    def gated(*args, **kwargs):
        value = original(*args, **kwargs)
        if not entered.is_set():
            first_worker.append(threading.current_thread())
            entered.set()
            assert release.wait(5), "test did not release core insertion"
        return value

    def interrupt():
        replies.append(server.handle_request({
            "id": "stop-at-core", "method": "session.interrupt", "params": {"session_id": sid},
        }))
        stopped.set()

    try:
        with _gateway_runtime(monkeypatch, tmp_path, session, sid) as events:
            monkeypatch.setattr(server, "_load_cfg", lambda: {
                "display": {"background_process_notifications": "all"},
            })
            name = "append_message" if boundary == "after_insert" else "_stage_turn_user_message"
            if boundary == "before_insert":
                monkeypatch.setattr(turn_context, "append_message", observe_append)
            monkeypatch.setattr(turn_context, name, gated)
            poller = threading.Thread(target=server._notification_poller_loop, args=(stop, sid, session))
            threads.append(poller)
            poller.start()
            events.put(_completion(event_id, session_id))
            assert entered.wait(5)
            stop.set()
            canceller = threading.Thread(target=interrupt)
            threads.append(canceller)
            canceller.start()
            # Stop is allowed to wait for an in-progress insertion. On the old
            # code it finishes here and reclaims an already-appended receipt.
            stopped.wait(2)
            release.set()
            canceller.join(5)
            assert not canceller.is_alive()
            assert replies[0]["result"]["status"] == "interrupted"
            poller.join(5)
            assert not poller.is_alive()
            worker = session.get("_run_thread")
            if worker is not None:
                worker.join(5)
                assert not worker.is_alive()
            if boundary == "before_insert":
                # The real post-turn safety net may already have retried in a
                # successor worker. Only the cancelled worker must insert zero.
                assert stale_appends == []
            assert agent._inflight_turn_id is None
            monkeypatch.setattr(turn_context, name, original)
            retry = threading.Thread(target=server._notification_poller_loop, args=(retry_stop, sid, session))
            threads.append(retry)
            retry.start()
            deadline = time.monotonic() + 5
            while not process_registry.is_completion_consumed(event_id) and time.monotonic() < deadline:
                time.sleep(0.01)
            retry_stop.set()
            retry.join(5)
            assert not retry.is_alive()
            worker = session.get("_run_thread")
            if worker is not None:
                worker.join(5)
                assert not worker.is_alive()
            assert sum(event_id in row for row in _user_rows(db, session_id)) == 1
            assert process_registry.is_completion_consumed(event_id)
    finally:
        release.set()
        stop.set()
        retry_stop.set()
        for thread in [*threads, session.get("_run_thread")]:
            if thread is not None:
                thread.join(5)
        db.close()
        process_registry._completion_consumed.discard(event_id)
        process_registry._poll_observed.discard(event_id)


@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
@pytest.mark.parametrize("setting", [False, "off"], ids=["bool-off", "string-off"])
def test_public_poller_off_uses_owner_profile_before_buffering(tmp_path, monkeypatch, busy, setting):
    homes = [tmp_path / "a", tmp_path / "b"]
    for home, value in zip(homes, [setting, "all"]):
        home.mkdir()
        (home / "config.yaml").write_text(yaml.safe_dump({
            "display": {"background_process_notifications": value},
        }), encoding="utf-8")
    submits, emitted = [], []
    monkeypatch.setenv("HERMES_HOME", str(homes[1]))
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: emitted.append(args))
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_loop_tick", lambda *args: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_heartbeat_tick", lambda *args: None)
    monkeypatch.setattr(server, "_notif_poll_kanban", lambda *args: None)
    monkeypatch.setattr(server, "_poll_bot_live_delivery_once", lambda *args: False)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *args, **kwargs: submits.append(args) or True)

    class OnePoll:
        calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 1

    for index, home in enumerate([homes[0], homes[1], homes[0]]):
        sid, event_id = f"policy-ui-{index}", f"policy-completion-{index}"
        session = {"agent": SimpleNamespace(steer=lambda text: True), "session_key": "",
                   "history": [], "history_lock": threading.Lock(), "running": busy,
                   "profile_home": str(home), "transport": None}
        isolated = queue.Queue()
        isolated.put(_completion(event_id, ""))
        monkeypatch.setattr(process_registry, "completion_queue", isolated)
        start = len(submits)
        try:
            server._notification_poller_loop(OnePoll(), sid, session)
            held = session.get("_completion_transfer") or session.get("_completion_pending") or []
            assert any(args[0] == "status.update" and args[1] == sid for args in emitted)
            if home == homes[0]:
                assert len(submits) == start
                assert not held
                assert isolated.empty()
                assert session.get("_completion_active_receipt") is None
            elif busy:
                assert [event["session_id"] for event in held] == [event_id]
            else:
                assert len(submits) == start + 1
            assert get_hermes_home() == homes[1]
        finally:
            process_registry._completion_consumed.discard(event_id)
