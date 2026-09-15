"""Public lifecycle witnesses for ordered completion ownership handoff."""
from __future__ import annotations

import queue
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent
from tools.process_registry import process_registry
from tui_gateway import server


def _response(content: str = "ack") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content, tool_calls=None, reasoning_content=None, reasoning=None,
            ),
            finish_reason="stop",
        )],
        model="test/model",
        usage=None,
    )


def _tool_response() -> SimpleNamespace:
    call = SimpleNamespace(
        id="call-boundary",
        type="function",
        function=SimpleNamespace(name="test_boundary", arguments="{}"),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="", tool_calls=[call], reasoning_content=None, reasoning=None,
            ),
            finish_reason="tool_calls",
        )],
        model="test/model",
        usage=None,
    )


def _agent(db: SessionDB, session_id: str) -> AIAgent:
    tool_defs = [{
        "type": "function",
        "function": {
            "name": "test_boundary",
            "description": "Provide a deterministic current-turn insertion boundary.",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=3,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            skip_background_review=True,
            session_db=db,
            session_id=session_id,
            platform="tui",
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "stable test prompt"
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._cleanup_task_resources = lambda *_a, **_k: None
    agent._save_trajectory = lambda *_a, **_k: None
    agent._invoke_tool = lambda *_a, **_k: "boundary complete"
    return agent


def _session(agent: AIAgent, session_id: str, *, running: bool = False) -> dict:
    ready = threading.Event()
    ready.set()
    return {
        "agent": agent,
        "agent_ready": ready,
        "session_key": session_id,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": running,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        "source": "tui",
    }


def _completion(event_id: str, session_id: str) -> dict:
    return {
        "type": "completion",
        "session_id": event_id,
        "session_key": session_id,
        "command": f"printf {event_id}",
        "exit_code": 0,
        "output": event_id,
    }


def _barrier(kind: str, event_id: str, session_id: str, sid: str) -> dict:
    event = {
        "type": kind,
        "session_id": event_id,
        "session_key": session_id,
        "origin_ui_session_id": sid,
        "command": f"watch {event_id}",
        "exit_code": 0,
        "output": event_id,
        "pattern": event_id,
    }
    if kind == "async_delegation":
        event.update(
            delegation_id=event_id,
            goal=event_id,
            status="completed",
            summary=f"completed {event_id}",
        )
    return event


def _user_rows(db: SessionDB, session_id: str) -> list[str]:
    return [
        str(row.get("content") or "")
        for row in db.get_messages_as_conversation(session_id)
        if row.get("role") == "user"
    ]


@contextmanager
def _gateway_runtime(monkeypatch, tmp_path, session: dict, sid: str, *, sync_caps: bool = False):
    isolated: queue.Queue = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_a: None)
    monkeypatch.setattr(server, "_persist_session_row_for_submit", lambda *_a: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a: True)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_a: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_a, **_k: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_a: None)
    monkeypatch.setattr(server, "_apply_pending_model_switch", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_agent_compression_with_config", lambda *_a: None)
    if not sync_caps:
        monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_a: None)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: False)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_record_turn_marker", lambda *_a, **_k: "marker")
    monkeypatch.setattr(server, "_retire_turn_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_loop_tick", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_maybe_fire_tui_heartbeat_tick", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_notif_poll_kanban", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_poll_bot_live_delivery_once", lambda *_a, **_k: False)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *_a, **_k: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    yield isolated


@pytest.mark.parametrize("first_state", ["core_inserted", "transfer_owned"])
@pytest.mark.parametrize(
    "barrier_kind", ["watch_match", "watch_idle", "watch_timeout", "async_delegation"]
)
def test_public_finalization_poller_retry_keeps_c1_barrier_c2_order(
    tmp_path, monkeypatch, first_state, barrier_kind
):
    """A live poller keeps C1/W/C2 as three durable turns in all finite variants."""
    sid = f"ui-{first_state}-{barrier_kind}"
    session_id = f"session-{first_state}-{barrier_kind}"
    c1_id, w_id, c2_id = f"c1-{sid}", f"w-{sid}", f"c2-{sid}"
    event_ids = (c1_id, c2_id)
    process_registry._completion_consumed.difference_update(event_ids)
    process_registry._poll_observed.difference_update(event_ids)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="tui", model="test/model")
    agent = _agent(db, session_id)
    session = _session(agent, session_id)
    first_call_started = threading.Event()
    release_first_call = threading.Event()
    second_call_started = threading.Event()
    release_second_call = threading.Event()
    calls = 0
    post_turn_calls = 0
    real_post_turn = server._run_post_turn_followups

    def model_call(_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_call_started.set()
            assert release_first_call.wait(4), "initial turn was not released"
            return _tool_response() if first_state == "core_inserted" else _response()
        if calls == 2 and first_state == "core_inserted":
            second_call_started.set()
            assert release_second_call.wait(4), "post-insertion turn was not released"
        return _response()

    agent._interruptible_api_call = model_call
    stop = threading.Event()
    poller = None
    try:
        with _gateway_runtime(monkeypatch, tmp_path, session, sid) as events:
            def ordered_post_turn(*args, **kwargs):
                nonlocal post_turn_calls
                post_turn_calls += 1
                if post_turn_calls == 1:
                    deadline = time.monotonic() + 6
                    while time.monotonic() < deadline:
                        if any(w_id in row for row in _user_rows(db, session_id)):
                            break
                        time.sleep(0.01)
                    assert any(w_id in row for row in _user_rows(db, session_id)), (
                        "poller did not retry the barrier before current-turn followups"
                    )
                return real_post_turn(*args, **kwargs)

            monkeypatch.setattr(server, "_run_post_turn_followups", ordered_post_turn)
            poller = threading.Thread(
                target=server._notification_poller_loop,
                args=(stop, sid, session),
                name=f"public-order-poller-{barrier_kind}",
            )
            poller.start()
            response = server.handle_request({
                "id": "initial-user",
                "method": "prompt.submit",
                "params": {"session_id": sid, "text": "initial user turn"},
            })
            assert response["result"]["status"] == "streaming"
            assert first_call_started.wait(4), "initial real agent turn did not reach the provider boundary"
            events.put(_completion(c1_id, session_id))
            events.put(_barrier(barrier_kind, w_id, session_id, sid))
            events.put(_completion(c2_id, session_id))

            deadline = time.monotonic() + 4
            while session.get("_completion_transfer_barrier") is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert session.get("_completion_transfer_barrier") is not None
            release_first_call.set()
            if first_state == "core_inserted":
                assert second_call_started.wait(4), "C1 did not reach the real tool-result insertion boundary"
                deadline = time.monotonic() + 4
                while not process_registry.is_completion_consumed(c1_id) and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert process_registry.is_completion_consumed(c1_id)
                assert session.get("_completion_transfer") in (None, [])
                release_second_call.set()

            deadline = time.monotonic() + 8
            rows: list[str] = []
            while time.monotonic() < deadline:
                rows = _user_rows(db, session_id)
                if all(any(item in row for row in rows) for item in (c1_id, w_id, c2_id)):
                    break
                time.sleep(0.02)
            matching = [row for row in rows if any(item in row for item in (c1_id, w_id, c2_id))]
            assert len(matching) == 3
            assert c1_id in matching[0]
            assert w_id in matching[1]
            assert c2_id in matching[2]
            assert all(sum(item in row for row in matching) == 1 for item in (c1_id, w_id, c2_id))
            assert all(process_registry.is_completion_consumed(item) for item in event_ids)
            assert session.get("_completion_active_receipt") is None
            assert session.get("_completion_pending") in (None, [])
            assert session.get("_completion_transfer") in (None, [])
    finally:
        stop.set()
        release_first_call.set()
        release_second_call.set()
        if poller is not None:
            poller.join(5)
            assert not poller.is_alive()
        db.close()
        process_registry._completion_consumed.difference_update(event_ids)
        process_registry._poll_observed.difference_update(event_ids)


def test_public_host_failure_falls_back_to_local_receipt_ingestion(tmp_path, monkeypatch):
    """A prior isolated host cannot capture a receipt created by later inline fallback."""
    sid, session_id, event_id = "ui-host-fallback", "session-host-fallback", "c-host-fallback"
    process_registry._completion_consumed.discard(event_id)
    process_registry._poll_observed.discard(event_id)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="tui", model="test/model")
    agent = _agent(db, session_id)
    session = _session(agent, session_id)
    session["agent"] = None
    inline_started = threading.Event()
    release_inline = threading.Event()

    class Supervisor:
        fail = False
        completion = None

        def submit_turn(self, _frame, *, on_complete):
            if self.fail:
                raise BrokenPipeError("synthetic host send failure")
            self.completion = on_complete

    supervisor = Supervisor()
    calls = 0

    def model_call(_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            inline_started.set()
            assert release_inline.wait(4), "inline fallback was not released"
        return _response()

    agent._interruptible_api_call = model_call
    stop = threading.Event()
    poller = None
    try:
        with _gateway_runtime(monkeypatch, tmp_path, session, sid) as events:
            monkeypatch.setattr(
                server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True}
            )
            monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *_a: supervisor)
            monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a: False)

            def start_inline_agent(_sid, current):
                current["agent"] = agent
                current["agent_ready"].set()

            monkeypatch.setattr(server, "_start_agent_build", start_inline_agent)
            poller = threading.Thread(target=server._notification_poller_loop, args=(stop, sid, session))
            poller.start()

            first = server.handle_request({
                "id": "host-success",
                "method": "prompt.submit",
                "params": {"session_id": sid, "text": "isolated turn"},
            })
            assert first["result"]["turn_isolation"] is True
            assert supervisor.completion is not None
            supervisor.completion({"type": "turn.end", "session_info_emitted": True})
            assert session["running"] is False
            assert session["_compute_host_active"] is True

            supervisor.fail = True
            second = server.handle_request({
                "id": "host-failure",
                "method": "prompt.submit",
                "params": {"session_id": sid, "text": "inline fallback turn"},
            })
            assert second["result"]["status"] == "streaming"
            assert inline_started.wait(4), "public fail-open route did not reach the inline agent"
            events.put(_completion(event_id, session_id))
            deadline = time.monotonic() + 4
            while not session.get("_completion_transfer") and time.monotonic() < deadline:
                time.sleep(0.01)
            assert [e["session_id"] for e in session.get("_completion_transfer") or []] == [event_id]
            release_inline.set()

            deadline = time.monotonic() + 8
            rows: list[str] = []
            while time.monotonic() < deadline:
                rows = _user_rows(db, session_id)
                if any(event_id in row for row in rows):
                    break
                time.sleep(0.02)
            assert sum(event_id in row for row in rows) == 1
            assert process_registry.is_completion_consumed(event_id)
            assert session.get("_completion_active_receipt") is None
            assert session.get("_completion_pending") in (None, [])
            assert session.get("_completion_transfer") in (None, [])
    finally:
        stop.set()
        release_inline.set()
        if poller is not None:
            poller.join(5)
            assert not poller.is_alive()
        db.close()
        process_registry._completion_consumed.discard(event_id)
        process_registry._poll_observed.discard(event_id)


def test_public_bot_capability_rebuild_binds_receipt_to_replacement(tmp_path, monkeypatch):
    """Real turn preparation binds the receipt after a Bot capability rebuild."""
    sid, session_id, event_id = "ui-bot-rebuild", "session-bot-rebuild", "c-bot-rebuild"
    process_registry._completion_consumed.discard(event_id)
    process_registry._poll_observed.discard(event_id)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="tui", model="test/model")
    original = _agent(db, session_id)
    replacement = _agent(db, session_id)
    original._session_title_hint = "Bot Chat"
    replacement._session_title_hint = "Bot Chat"
    replacement._interruptible_api_call = lambda _kwargs: _response()
    session = _session(original, session_id)
    session.update(bot_caps_seen="caps-old", profile_home=str(tmp_path / "profile"))
    inserted_before_consume: list[str] = []
    consumed_after_insert: list[list[str]] = []
    real_mark_consumed = server._mark_completion_events_consumed
    from agent import turn_context
    real_append_message = turn_context.append_message

    def append_message(messages, message):
        real_append_message(messages, message)
        if message.get("role") == "user":
            inserted_before_consume.append(str(message.get("content") or ""))

    def mark_consumed(events):
        consumed_after_insert.append(list(inserted_before_consume))
        real_mark_consumed(events)

    stop = threading.Event()
    poller = None
    try:
        with _gateway_runtime(monkeypatch, tmp_path, session, sid, sync_caps=True) as events:
            monkeypatch.setattr("tools.bot_mode_probe.capability_fingerprint", lambda _home: "caps-new")
            monkeypatch.setattr(server, "_config_model_target", lambda: ("test/model", "openai-compat"))
            monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
            monkeypatch.setattr(server, "_session_source", lambda _session: "tui")
            monkeypatch.setattr(server, "_make_agent", lambda *_a, **_k: replacement)
            monkeypatch.setattr(turn_context, "append_message", append_message)
            monkeypatch.setattr(server, "_mark_completion_events_consumed", mark_consumed)
            poller = threading.Thread(target=server._notification_poller_loop, args=(stop, sid, session))
            poller.start()
            events.put(_completion(event_id, session_id))

            deadline = time.monotonic() + 8
            rows: list[str] = []
            while time.monotonic() < deadline:
                rows = _user_rows(db, session_id)
                if process_registry.is_completion_consumed(event_id) and any(event_id in row for row in rows):
                    break
                time.sleep(0.02)
            assert session["agent"] is replacement
            assert sum(event_id in row for row in rows) == 1
            assert consumed_after_insert and event_id in consumed_after_insert[0][-1]
            assert session.get("_completion_active_receipt") is None
            assert session.get("_completion_pending") in (None, [])
            assert session.get("_completion_transfer") in (None, [])
    finally:
        stop.set()
        if poller is not None:
            poller.join(5)
            assert not poller.is_alive()
        db.close()
        process_registry._completion_consumed.discard(event_id)
        process_registry._poll_observed.discard(event_id)
