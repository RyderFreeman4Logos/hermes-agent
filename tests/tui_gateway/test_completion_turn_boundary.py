"""Current-turn completion handoff regressions (#315)."""
from __future__ import annotations

import json
import queue
import threading
from copy import deepcopy
from unittest.mock import patch

from agent.turn_iteration_prep import prepare_iteration
from agent.turn_stop_gates import apply_stop_gates
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
        assert not process_registry.is_completion_consumed(event_id)
        assert isolated.empty()
    finally:
        stop.set()
        release_stage.set()
        if poller is not None:
            poller.join(4)
            assert not poller.is_alive()
        server._sessions.pop("late-turn-boundary-ui", None)
        _clear(event_id)


def test_context_refusal_preserves_staged_completion_without_replaying_user_prompt(tmp_path, monkeypatch):
    event_id = "proc_context_refusal_boundary"
    _clear(event_id)
    try:
        agent = _agent()
        agent._config_context_length = 1_000
        session = _session(agent)
        session["cwd"] = str(tmp_path)
        sid = "context-refusal-ui"
        server._sessions[sid] = session
        for name in ("first.txt", "second.txt"):
            (tmp_path / name).write_text("x" * 1_200, encoding="utf-8")
        monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_record_turn_marker", lambda *_a, **_k: "marker")
        monkeypatch.setattr(server, "_finish_turn", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_retire_turn_marker", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_clear_inflight_turn", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_a, **_k: None)
        real_drain = server._drain_queued_prompt
        drains = 0

        def drain_once(*args, **kwargs):
            nonlocal drains
            drains += 1
            return real_drain(*args, **kwargs) if drains == 1 else False

        monkeypatch.setattr(server, "_drain_queued_prompt", drain_once)
        assert server._deliver_completions_via_steer(
            sid, session, [_completion(event_id)], set()
        )
        with session["history_lock"]:
            server._enqueue_prompt(
                session, "Inspect @file:first.txt and @file:second.txt", None
            )
            session["running"] = False
        server._run_post_turn_followups("rid", sid, session, {}, None)
        session["_run_thread"].join(2)
        assert not session["_run_thread"].is_alive()
        assert session["running"] is False
        assert event_id in session["queued_prompt"]["text"]
        assert "@file:first.txt" not in session["queued_prompt"]["text"]
        assert "@file:second.txt" not in session["queued_prompt"]["text"]
        assert not process_registry.is_completion_consumed(event_id)
        assert session.get("_completion_transfer") == []
    finally:
        server._sessions.pop("context-refusal-ui", None)
        _clear(event_id)


def test_real_context_refusal_keeps_staged_completion_after_second_drain(tmp_path, monkeypatch):
    """A real @file refusal must not drop a completion staged behind the user turn.

    Both drains are the production function. The user turn runs first and returns
    from context refusal without its own followups; the second drain still has to
    start the staged completion.
    """
    event_id = "proc_real_context_refusal"
    _clear(event_id)
    drains: list[str] = []
    started: list[str] = []
    real_drain = server._drain_queued_prompt
    real_prepare = server._prepare_turn_input
    try:
        agent = _agent()
        agent.steer = lambda _text: True
        agent._config_context_length = 1_000
        agent.model = ""
        agent.base_url = ""
        agent.api_key = ""
        agent.provider = ""
        session = _session(agent, running=True)
        session["cwd"] = str(tmp_path)
        sid = "real-context-refusal-ui"
        server._sessions[sid] = session
        for name in ("first.txt", "second.txt"):
            (tmp_path / name).write_text("x" * 1_200, encoding="utf-8")

        def record_drain(rid, drain_sid, drain_session):
            head = (drain_session.get("queued_prompt") or {}).get("text") or ""
            drains.append(f"{head[:40]}|run={drain_session.get('running')}")
            started_turn = real_drain(rid, drain_sid, drain_session)
            drains.append(f"returned={started_turn}|run={drain_session.get('running')}")
            return started_turn

        def refuse_only_file_prompt(sid_arg, session_arg, st, text, images, **kwargs):
            if isinstance(text, str) and "@file:" in text:
                return real_prepare(sid_arg, session_arg, st, text, images, **kwargs)
            started.append(text)
            return ("prompt", "run", 80, None)

        monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_record_turn_marker", lambda *_a, **_k: "marker")
        monkeypatch.setattr(server, "_retire_turn_marker", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_drain_queued_prompt", record_drain)
        monkeypatch.setattr(server, "_prepare_turn_input", refuse_only_file_prompt)
        assert server._deliver_completions_via_steer(
            sid, session, [{**_completion(event_id), "session_id": event_id}], set()
        )
        with session["history_lock"]:
            session["running"] = False
            server._enqueue_prompt(
                session, "Inspect @file:first.txt and @file:second.txt", None
            )
        server._run_post_turn_followups("rid", sid, session, {}, None)
        thread = session.get("_run_thread")
        if thread is not None:
            thread.join(4)
            assert not thread.is_alive()

        assert len(drains) >= 2
        assert "@file:first.txt" in drains[0]
        assert started and event_id in started[0], "\n".join(drains)
        assert "@file:" not in started[0]
        assert session["running"] is False
        assert not process_registry.is_completion_consumed(event_id)
        assert session.get("_completion_transfer") in (None, [])
    finally:
        server._sessions.pop("real-context-refusal-ui", None)
        _clear(event_id)


def test_staged_completion_and_late_user_prompt_keep_separate_queue_entries(monkeypatch):
    event_id = "proc_completion_queue_interleaving"
    _clear(event_id)
    try:
        agent = _agent()
        session = _session(agent)
        sid = "completion-queue-interleaving-ui"
        server._sessions[sid] = session
        monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
        assert server._deliver_completions_via_steer(
            sid, session, [_completion(event_id)], set()
        )
        with session["history_lock"]:
            session["running"] = False
        drains = 0

        def enqueue_late_user(*_args, **_kwargs):
            nonlocal drains
            drains += 1
            if drains == 2:
                with session["history_lock"]:
                    server._enqueue_prompt(session, "late user prompt", None)
            return False

        monkeypatch.setattr(server, "_drain_queued_prompt", enqueue_late_user)
        server._run_post_turn_followups("rid", sid, session, {}, None)

        assert event_id in session["queued_prompt"]["text"]
        assert session["queued_prompt"]["text"].strip() != "late user prompt"
        assert [entry["text"] for entry in session["queued_prompts"]] == ["late user prompt"]
        assert not process_registry.is_completion_consumed(event_id)
    finally:
        server._sessions.pop("completion-queue-interleaving-ui", None)
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


def test_pre_api_after_persisted_interim_answer_retains_appendable_prefix(tmp_path):
    event_id = "proc_after_interim_answer"
    _clear(event_id)
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
        agent._emit_interim_assistant_message = lambda _message: None
        agent._interim_content_was_streamed = lambda _content: False
        with patch("agent.turn_stop_gates._verify_on_stop_nudge", return_value="verify N1"):
            verdict = apply_stop_gates(
                agent,
                {"role": "assistant", "content": "persisted A1"},
                final_response="persisted A1",
                messages=messages,
                conversation_history=[],
                pending_verification_response=None,
                pending_verification_response_previewed=False,
            )
        assert verdict.continue_turn
        live_prefix = json.dumps(messages, sort_keys=False, separators=(",", ":"))
        durable_prefix = _rows(db.get_messages_as_conversation(agent.session_id))
        assert durable_prefix[-1] == ("assistant", "persisted A1")
        assert server._deliver_completions_via_steer(
            "owner-ui", session, [_completion(event_id)], set()
        )
        assert agent.steer("genuine user steer")

        prepare_iteration(
            agent,
            messages=messages,
            api_call_count=2,
            user_message="U1",
            current_turn_user_idx=4,
        )
        agent._persist_session(messages, [])

        assert json.dumps(messages, sort_keys=False, separators=(",", ":")) == live_prefix
        assert _rows(db.get_messages_as_conversation(agent.session_id)) == durable_prefix
        assert [item["session_id"] for item in session["_completion_transfer"]] == [event_id]
        assert not process_registry.is_completion_consumed(event_id)
        assert agent._drain_pending_steer() == "genuine user steer"
        assert agent._drain_pending_steer() is None
    finally:
        db.close()
        _clear(event_id)


def test_prior_completion_keeps_order_until_next_valid_tool_boundary(tmp_path):
    event_ids = ("proc_prior_completion", "proc_later_completion")
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
                            "id": "first-call",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "first-call", "content": "T1"},
            ]
        )
        agent._persist_user_message_idx = 4
        agent._persist_session(messages, [])
        assert server._deliver_completions_via_steer(
            "owner-ui", session, [_completion(event_ids[0])], set()
        )
        prepare_iteration(
            agent, messages=messages, api_call_count=1,
            user_message="U1", current_turn_user_idx=4,
        )
        agent._persist_session(messages, [])
        assert process_registry.is_completion_consumed(event_ids[0])

        assert server._deliver_completions_via_steer(
            "owner-ui", session, [_completion(event_ids[1])], set()
        )
        assert agent.steer("genuine later steer")
        prior_prefix = json.dumps(messages, sort_keys=False, separators=(",", ":"))
        prepare_iteration(
            agent, messages=messages, api_call_count=2,
            user_message="U1", current_turn_user_idx=4,
        )

        assert json.dumps(messages, sort_keys=False, separators=(",", ":")) == prior_prefix
        assert [item["session_id"] for item in session["_completion_transfer"]] == [event_ids[1]]
        assert not process_registry.is_completion_consumed(event_ids[1])
        assert agent._pending_steer == "genuine later steer"

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "second-call",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "second-call", "content": "T2"},
            ]
        )
        agent._persist_session(messages, [])
        prepare_iteration(
            agent, messages=messages, api_call_count=3,
            user_message="U1", current_turn_user_idx=4,
        )
        agent._persist_session(messages, [])

        prior_rows = [i for i, row in enumerate(messages)
                      if row.get("role") == "user" and event_ids[0] in str(row.get("content"))]
        later_rows = [i for i, row in enumerate(messages)
                      if row.get("role") == "user" and event_ids[1] in str(row.get("content"))]
        assert len(prior_rows) == len(later_rows) == 1
        assert prior_rows[0] < later_rows[0]
        assert "genuine later steer" in str(messages[later_rows[0]]["content"])
        assert _rows(db.get_messages_as_conversation(agent.session_id)) == _rows(messages)
        assert session.get("_completion_transfer") == []
        assert process_registry.is_completion_consumed(event_ids[1])
    finally:
        db.close()
        _clear(*event_ids)


def test_queued_raw_completion_context_refs_stay_literal_until_core_ingest(tmp_path, monkeypatch):
    """Queued autonomous output is data, even when it resembles two @file directives."""
    event_id = "proc_literal_completion_refs"
    _clear(event_id)
    try:
        agent = _agent()
        agent._config_context_length = 1_000
        session = _session(agent, running=False)
        session["cwd"] = str(tmp_path)
        sid = "literal-completion-ui"
        server._sessions[sid] = session
        for name in ("first.txt", "second.txt"):
            (tmp_path / name).write_text("x" * 1_200, encoding="utf-8")
        captured = {}
        receipt = {"events": [_completion(event_id)]}
        session["_completion_active_receipt"] = receipt
        monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_record_turn_marker", lambda *_a, **_k: "marker")
        monkeypatch.setattr(server, "_finish_turn", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_retire_turn_marker", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_clear_inflight_turn", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_a, **_k: None)

        def stop_before_provider(_sid, _session, st, prompt, *_args):
            captured["prompt"] = prompt
            st.result = {"final_response": "unused"}

        monkeypatch.setattr(server, "_invoke_agent", stop_before_provider)
        assert server._run_prompt_submit(
            "rid", sid, session, "result @file:first.txt @file:second.txt",
            completion_receipt=receipt,
        )
        session["_run_thread"].join(2)
        assert not session["_run_thread"].is_alive()
        assert captured["prompt"] == "result @file:first.txt @file:second.txt"
        assert process_registry.is_completion_consumed(event_id) is False
        assert [event["session_id"] for event in session["_completion_pending"]] == [event_id]
    finally:
        server._sessions.pop("literal-completion-ui", None)
        _clear(event_id)
