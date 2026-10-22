"""Regression for #49225 — codex app-server turns must reach the session DB
exactly once.

The codex app-server runtime (``run_codex_app_server_turn``) is an early-return
path that bypasses ``conversation_loop`` and therefore never runs the loop's
per-step ``_persist_session()`` flushes. Before the fix, the projected
assistant/tool messages were persisted *nowhere* (state.db got only
session_meta rows), leaving ``session_search`` (FTS) and conversation-distill
blind to real gateway conversations.

The fix has the codex runtime flush its own projected messages via
``_flush_messages_to_session_db()`` (idempotent through the intrinsic
``_DB_PERSISTED_MARKER``) and return ``agent_persisted=True`` so the gateway
skips its own ``append_to_transcript`` DB write. This is critical: the inbound
user turn is already flushed at turn start (``turn_context._persist_session``),
and ``append_message`` is a raw INSERT with no dedup — a gateway re-write would
duplicate the user turn (#860 / #42039). This test locks in:

1. ``run_codex_app_server_turn`` flushes projected messages and returns
   ``agent_persisted=True``.
2. Exactly-once persistence: the already-flushed user turn is NOT re-written,
   and the new projected assistant message lands once.
3. The gateway resolution expression preserves standard-runtime behaviour.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.codex_runtime import run_codex_app_server_turn
from hermes_state import SessionDB
from run_agent import AIAgent

def _make_turn():
    return SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[{"role": "assistant", "content": "CODEX_ASSISTANT"}],
        tool_iterations=0,
        final_text="CODEX_ASSISTANT",
        should_retire=False,
    )

def _make_agent(session_db=None, session_id="sess-codex"):
    agent = MagicMock()
    # Pre-seed the session so run_codex_app_server_turn skips the spawn block.
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.return_value = _make_turn()
    agent._codex_session_prompt = None  # seeded session: no recorded composition to compare
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = session_db
    agent._session_db_created = True
    agent.session_id = session_id
    return agent

def test_codex_success_flushes_and_reports_persisted():
    """Codex success turn must self-persist and return agent_persisted=True."""
    agent = _make_agent(session_db=None)  # no DB -> flush is a no-op, still True
    result = run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
        effective_task_id="task-1",
    )
    assert result["completed"] is True
    assert isinstance(result["messages"][-1]["timestamp"], float)
    # With the agent as sole persister, the gateway must SKIP its DB write.
    assert result["agent_persisted"] is True
    agent._reset_stream_delivery_tracking.assert_called_once_with()

@pytest.mark.parametrize(
    ("usage", "expected_counts"),
    [
        (
            {"inputTokens": 30, "cachedInputTokens": 12, "outputTokens": 5, "totalTokens": 35},
            {"cache_read": 12, "prompt_total": 30},
        ),
        (None, {}),
        ({"inputTokens": 30, "cachedInputTokens": 40, "outputTokens": 5}, {}),
    ],
)
def test_codex_success_persists_reply_end_on_its_projected_assistant_row(
    monkeypatch, tmp_path, usage, expected_counts,
):
    from agent.context_compressor import ContextCompressor
    from agent.usage_pricing import CostResult

    monkeypatch.setattr(
        ContextCompressor, "_is_synthetic_compression_user_turn",
        classmethod(lambda _cls, _message: False),
    )
    monkeypatch.setattr("agent.codex_runtime.time.monotonic", lambda: 103.25)
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *_args, **_kwargs: CostResult(None, "unknown", "none", "n/a"),
    )
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    sid = "sess-codex-footer"
    db.create_session(session_id=sid, source="telegram", model="codex")
    try:
        agent: Any = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent._session_db_created = True
        agent._reply_started_monotonic = 100.0
        agent._reply_usage = None
        agent._reply_usage_complete = True
        agent._sync_external_memory_for_turn = lambda **_kwargs: None
        agent._codex_session = MagicMock()
        turn = _make_turn()
        turn.token_usage_last = usage
        agent._codex_session.run_turn.return_value = turn
        messages = [{"role": "user", "content": "USER_TURN"}]
        agent._flush_messages_to_session_db(messages)

        from agent import codex_runtime

        published_reply_ends = []
        original_store = codex_runtime._store_codex_thread_id

        def check_reply_end_before_thread_publish(target_agent, thread_id):
            row = next(row for row in db.get_messages(sid) if row["role"] == "assistant")
            published_reply_ends.append((row.get("display_metadata") or {}).get("reply_end"))
            original_store(target_agent, thread_id)

        monkeypatch.setattr(codex_runtime, "_store_codex_thread_id", check_reply_end_before_thread_publish)
        result = run_codex_app_server_turn(
            agent,
            user_message="USER_TURN",
            original_user_message="USER_TURN",
            messages=messages,
            effective_task_id="task-1",
        )

        assert result["completed"] is True
        expected = {"elapsed_s": 3.25, **expected_counts}
        assert messages[-1]["display_metadata"]["reply_end"] == expected
        assert published_reply_ends == [expected]
        db.close()
        db = SessionDB(db_path)
        assistant_row = next(row for row in db.get_messages(sid) if row["role"] == "assistant")
        assert assistant_row["display_metadata"]["reply_end"] == expected
    finally:
        db.close()


@pytest.mark.parametrize("interrupted,error", [(True, None), (False, "turn failed")])
def test_codex_interrupted_or_error_turn_does_not_attach_reply_end(interrupted, error):
    agent = _make_agent(session_db=None)
    agent._reply_started_monotonic = 100.0
    turn = _make_turn()
    turn.projected_messages = [{"role": "assistant", "content": "partial"}]
    turn.final_text = "partial"
    turn.interrupted = interrupted
    turn.error = error
    agent._codex_session.run_turn.return_value = turn
    if interrupted:
        agent._interrupt_requested = True
        agent._interrupt_message = "stop"

        def clear_interrupt():
            agent._interrupt_requested = False
            agent._interrupt_message = None

        agent.clear_interrupt.side_effect = clear_interrupt
    messages = [{"role": "user", "content": "question"}]

    result = run_codex_app_server_turn(
        agent,
        user_message="question",
        original_user_message="question",
        messages=messages,
        effective_task_id="task-1",
    )

    assert result["completed"] is False
    assert "display_metadata" not in messages[-1]


def test_codex_user_interrupt_is_reported_and_cleared():
    agent = _make_agent(session_db=None)
    turn = _make_turn()
    turn.interrupted = True
    turn.final_text = ""
    agent._codex_session.run_turn.return_value = turn
    agent._interrupt_requested = True
    agent._interrupt_message = "new correction"

    def clear_interrupt():
        agent._interrupt_requested = False
        agent._interrupt_message = None

    agent.clear_interrupt.side_effect = clear_interrupt
    result = run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
        effective_task_id="task-1",
    )

    assert result["interrupted"] is True
    assert result["interrupt_message"] == "new correction"
    agent.clear_interrupt.assert_called_once_with()
    assert agent._interrupt_requested is False

def test_codex_turn_persists_each_message_exactly_once():
    """The user turn (flushed at turn start) must not be duplicated; the
    projected assistant message must land once.  Uses a real SessionDB and the
    real AIAgent._flush_messages_to_session_db to prove no #860/#42039
    duplicate-write regression on the codex path."""
    tmp = tempfile.mkdtemp(prefix="codex_persist_")
    db = None
    try:
        db = SessionDB(Path(tmp) / "state.db")
        sid = "sess-codex-once"
        db.create_session(session_id=sid, source="telegram", model="codex")

        # Real agent bound to this DB/session, minimal construction.
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent._session_db_created = True
        agent._codex_session = MagicMock()
        agent._codex_session.run_turn.return_value = _make_turn()
        agent.tool_progress_callback = None

        # Model the real flow: the inbound user turn is flushed at turn start
        # (turn_context._persist_session) on the SAME `messages` list the codex
        # path later reuses. That flush stamps _DB_PERSISTED_MARKER on the user
        # dict, so the codex-path flush skips it — no duplicate.
        user_msg = {"role": "user", "content": "USER_TURN"}
        messages = [user_msg]
        agent._flush_messages_to_session_db(messages)  # turn-start flush

        result = run_codex_app_server_turn(
            agent,
            user_message="USER_TURN",
            original_user_message="USER_TURN",
            messages=messages,
            effective_task_id="task-1",
        )
        assert result["agent_persisted"] is True

        rows = db.get_messages(sid, include_inactive=True)
        contents = [r["content"] for r in rows]
        # Exactly one user turn, exactly one assistant turn — no duplicates.
        assert contents.count("USER_TURN") == 1, contents
        assert contents.count("CODEX_ASSISTANT") == 1, contents
        assistant_row = next(
            row for row in rows if row["content"] == "CODEX_ASSISTANT"
        )
        assert isinstance(assistant_row["timestamp"], float)
        # session_search can now see the codex conversation.
        hits = {r["session_id"] for r in db.search_messages("CODEX_ASSISTANT")}
        assert sid in hits
    finally:
        import shutil

        if db is not None:
            db.close()
        shutil.rmtree(tmp, ignore_errors=True)
