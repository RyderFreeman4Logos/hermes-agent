"""Current-turn completion handoff regressions (#315)."""
from __future__ import annotations

import json
import queue
import threading
from copy import deepcopy
from unittest.mock import patch

from agent.turn_iteration_prep import prepare_iteration
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.process_registry import process_registry
from tui_gateway import server


def _completion(session_id: str) -> dict:
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": "owner-session",
        "command": f"printf {session_id}",
        "exit_code": 0,
        "output": "done",
    }


def _agent(db: SessionDB | None = None) -> AIAgent:
    with patch("run_agent.AIAgent.__init__", return_value=None):
        agent = AIAgent.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._model_request_active = threading.Event()
    agent._executing_tools = True
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    agent._hard_interrupt_requested = threading.Event()
    agent._interrupt_thread_signal_pending = False
    agent._execution_thread_id = None
    agent.session_id = "turn-boundary-session"
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.client = None
    agent._session_messages = None
    agent.step_callback = None
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._adopt_nous_key_before_expiry = lambda: None
    agent.run_budget_seconds = None
    agent.budget_warning_ratio = None
    agent.iteration_budget = None
    agent.logger = None
    agent._sanitize_args_cursor = {}
    agent._sanitize_tool_call_arguments = lambda *_a, **_k: 0
    agent._session_db = db
    agent._session_db_created = db is not None
    agent._session_persist_lock = threading.RLock()
    agent._persist_disabled = False
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = agent.session_id
    agent._last_flushed_db_idx = 0
    agent._db_flush_scan_prefix = None
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    agent._pending_cli_user_message = None
    agent._incremental_persistence_failed = False
    agent._compression_adoption_failed = False
    agent._inflight_turn_id = None
    agent._inflight_turn_session_id = None
    return agent


def _session(agent: AIAgent, *, running: bool = True) -> dict:
    return {
        "agent": agent,
        "session_key": "owner-session",
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
    }


def _history(current_user: str = "U1") -> list[dict]:
    return [
        {"role": "user", "content": "U0"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "old-call",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old-call", "content": "T0"},
        {"role": "assistant", "content": "A0 final"},
        {"role": "user", "content": current_user},
    ]


def _rows(messages: list[dict]) -> list[tuple[str, object]]:
    return [(row.get("role"), row.get("content")) for row in messages]


def _clear(*event_ids: str) -> None:
    process_registry._completion_consumed.difference_update(event_ids)
    process_registry._poll_observed.difference_update(event_ids)


def test_late_busy_stage_reaches_one_idle_turn_without_new_activity(monkeypatch):
    event_id = "proc_late_turn_boundary"
    _clear(event_id)
    isolated: queue.Queue = queue.Queue()
    late_stage = threading.Event()
    release_stage = threading.Event()
    admitted = threading.Event()
    stop = threading.Event()
    turns: list[str] = []
    poller: threading.Thread | None = None
    try:
        monkeypatch.setattr(process_registry, "completion_queue", isolated)
        monkeypatch.setattr(server, "_get_db", lambda: None)
        monkeypatch.setattr(server, "_maybe_fire_tui_loop_tick", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_maybe_fire_tui_heartbeat_tick", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_notif_poll_kanban", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_poll_bot_live_delivery_once", lambda *_a, **_k: False)

        agent = _agent()
        session = _session(agent)
        sid = "late-turn-boundary-ui"
        server._sessions[sid] = session

        def emit(kind, _sid, payload=None):
            if kind == "status.update" and event_id in str(payload):
                late_stage.set()
                assert release_stage.wait(2), "late staging release timed out"

        def submit(_rid, _sid, _session, text, **_kwargs):
            turns.append(text)
            admitted.set()
            return True

        monkeypatch.setattr(server, "_emit", emit)
        monkeypatch.setattr(server, "_run_prompt_submit", submit)
        isolated.put(_completion(event_id))
        poller = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, sid, session),
            name="late-stage-poller",
        )
        poller.start()
        assert late_stage.wait(2), "poller never reached the real status boundary"

        with session["history_lock"]:
            session["running"] = False
        server._run_post_turn_followups("rid", sid, session, {}, None)
        assert session.get("_completion_transfer") in (None, [])
        release_stage.set()

        assert admitted.wait(2), "late staged transfer never reached idle admission"
        assert len(turns) == 1
        assert event_id in turns[0]
        assert process_registry.is_completion_consumed(event_id)
        assert isolated.empty()
    finally:
        stop.set()
        release_stage.set()
        if poller is not None:
            poller.join(4)
            assert not poller.is_alive()
        server._sessions.pop("late-turn-boundary-ui", None)
        _clear(event_id)


def test_context_refusal_runs_the_existing_post_turn_handoff(monkeypatch):
    event_id = "proc_context_refusal_boundary"
    _clear(event_id)
    try:
        agent = _agent()
        session = _session(agent)
        sid = "context-refusal-ui"
        server._sessions[sid] = session
        monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_record_turn_marker", lambda *_a, **_k: "marker")
        monkeypatch.setattr(server, "_prepare_turn_input", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_finish_turn", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_retire_turn_marker", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_clear_inflight_turn", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a, **_k: False)
        assert server._deliver_completions_via_steer(
            sid, session, [_completion(event_id)], set()
        )

        assert server._run_prompt_submit("rid", sid, session, "@/refused")
        session["_run_thread"].join(2)
        assert not session["_run_thread"].is_alive()
        assert session["running"] is False
        assert event_id in session["queued_prompt"]["text"]
        assert process_registry.is_completion_consumed(event_id)
        assert session.get("_completion_transfer") == []
    finally:
        server._sessions.pop("context-refusal-ui", None)
        _clear(event_id)


def test_pre_api_without_current_tool_boundary_preserves_durable_prefix(tmp_path, monkeypatch):
    event_id = "proc_no_current_tool"
    _clear(event_id)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("turn-boundary-session", source="test")
    try:
        agent = _agent(db)
        session = _session(agent)
        messages = _history()
        agent._persist_user_message_idx = 4
        agent._persist_session(messages, [])
        live_prefix = json.dumps(messages, sort_keys=False, separators=(",", ":"))
        durable_prefix = _rows(db.get_messages_as_conversation(agent.session_id))
        assert server._deliver_completions_via_steer(
            "owner-ui", session, [_completion(event_id)], set()
        )

        prepare_iteration(
            agent,
            messages=messages,
            api_call_count=1,
            user_message="U1",
            current_turn_user_idx=4,
        )
        agent._persist_session(messages, [])

        assert json.dumps(messages, sort_keys=False, separators=(",", ":")) == live_prefix
        assert _rows(db.get_messages_as_conversation(agent.session_id)) == durable_prefix
        assert [item["session_id"] for item in session["_completion_transfer"]] == [event_id]
        assert not process_registry.is_completion_consumed(event_id)
    finally:
        db.close()
        _clear(event_id)


def test_current_tool_boundary_persists_ordered_completions_and_user_steer(tmp_path):
    event_ids = ("proc_current_tool_a", "proc_current_tool_b")
    _clear(*event_ids)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("turn-boundary-session", source="test")
    try:
        agent = _agent(db)
        session = _session(agent)
        messages = _history()
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "current-call",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "current-call", "content": "T1"},
            ]
        )
        agent._persist_user_message_idx = 4
        agent._persist_session(messages, [])
        durable_prefix = deepcopy(_rows(db.get_messages_as_conversation(agent.session_id)))
        assert server._deliver_completions_via_steer(
            "owner-ui", session, [_completion(event_ids[0]), _completion(event_ids[1])], set()
        )
        assert agent.steer("simultaneous user steer")

        prepare_iteration(
            agent,
            messages=messages,
            api_call_count=2,
            user_message="U1",
            current_turn_user_idx=4,
        )
        agent._persist_session(messages, [])

        assert _rows(messages)[:-1] == durable_prefix
        assert messages[-1]["role"] == "user"
        content = str(messages[-1]["content"])
        assert content.index(event_ids[0]) < content.index(event_ids[1])
        assert "simultaneous user steer" in content
        assert _rows(db.get_messages_as_conversation(agent.session_id)) == _rows(messages)
        assert session.get("_completion_transfer") == []
        assert all(process_registry.is_completion_consumed(item) for item in event_ids)
    finally:
        db.close()
        _clear(*event_ids)
