"""Test: the context engine is notified of a compression-boundary rollover.

When _compress_context rotates session_id (compression split), the active
context engine receives on_session_start(new_sid, boundary_reason="compression",
old_session_id=<old>). This lets plugin engines (e.g. hermes-lcm) preserve
DAG lineage across the split instead of treating it as a fresh /new.

See hermes-lcm#68: after Hermes compresses and mints a new physical session,
LCM was losing continuity (compression_count: 1, store_messages: 0,
dag_nodes: 0). With boundary_reason="compression" plugins can distinguish
this from a real user-initiated /new.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression import (
    finalize_context_engine_compression_notification,
)

class TestCompressionBoundaryHook:
    def _make_agent(self, session_db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )
            # ROTATION fallback — pin in_place=False regardless of default (#38763).
            agent.compression_in_place = False
            return agent

    def test_on_session_start_called_with_compression_boundary(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)

            # Stub the context compressor: we only need to observe the hook.
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "tail question"},
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            # Avoid the summary-error warning path
            compressor._last_summary_error = None
            # MagicMock auto-creates truthy attrs; explicitly clear the abort
            # flag so the post-compress abort branch in
            # conversation_compression.py does not short-circuit before the
            # session-id rotation we are asserting on.
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor

            original_sid = agent.session_id
            messages = [
                {"role": "user", "content": f"m{i}"} for i in range(10)
            ]

            agent._compress_context(messages, "sys", approx_tokens=10_000)

            # Session_id rotated
            assert agent.session_id != original_sid, \
                "compression should rotate session_id when session_db is set"

            # Hook fired with boundary_reason="compression" and old_session_id
            calls = [
                c for c in compressor.on_session_start.call_args_list
            ]
            assert calls, "on_session_start was never called on the context engine"
            # Find the compression boundary call (there may be others from init)
            comp_calls = [
                c for c in calls
                if c.kwargs.get("boundary_reason") == "compression"
            ]
            assert comp_calls, (
                f"Expected an on_session_start call with "
                f"boundary_reason='compression', got {calls!r}"
            )
            call = comp_calls[-1]
            # Positional new session_id
            assert call.args and call.args[0] == agent.session_id, \
                f"Expected new session_id as first positional arg, got {call!r}"
            assert call.kwargs.get("old_session_id") == original_sid, \
                f"Expected old_session_id={original_sid!r}, got {call.kwargs!r}"
            assert len(comp_calls) == 1
            assert agent._awaiting_cache_usage_after_compression is True

    def test_automatic_notification_follows_core_persistence(self):
        from hermes_state import SessionDB

        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "summary"}
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            compressor.on_session_start.side_effect = (
                lambda *_args, **kwargs: events.append(
                    kwargs.get("boundary_reason")
                )
            )
            agent.context_compressor = compressor
            original_publish = db.publish_compression_child

            def _record_publish(*args, **kwargs):
                result = original_publish(*args, **kwargs)
                events.append("persist")
                return result

            with patch.object(
                db, "publish_compression_child", side_effect=_record_publish
            ):
                agent._compress_context(
                    [{"role": "user", "content": "request"}],
                    "sys",
                    approx_tokens=100,
                )

            assert events == ["persist", "compression"]

    def test_failure_before_persistence_does_not_notify(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.side_effect = RuntimeError("synthetic compression failure")
            agent.context_compressor = compressor

            with pytest.raises(RuntimeError, match="synthetic compression failure"):
                agent._compress_context(
                    [{"role": "user", "content": "request"}],
                    "sys",
                    approx_tokens=100,
                )

            compressor.on_session_start.assert_not_called()
            assert agent._awaiting_cache_usage_after_compression is False

    def test_failed_persist_does_not_arm_cache_bound_latch(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.return_value = [
                {"role": "user", "content": "summary"}
            ]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False
            compressor._last_compression_made_progress = True
            agent.context_compressor = compressor

            with patch.object(
                db, "publish_compression_child", side_effect=RuntimeError("persist failed")
            ):
                agent._compress_context(
                    [{"role": "user", "content": "m" * 400}],
                    "sys",
                    approx_tokens=100,
                )

            assert agent._awaiting_cache_usage_after_compression is False


    def test_no_progress_does_not_notify(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)
            compressor = MagicMock()
            compressor.compress.side_effect = lambda messages, **_kwargs: messages
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor
            messages = [{"role": "user", "content": "request"}]

            returned, _ = agent._compress_context(
                messages,
                "sys",
                approx_tokens=100,
            )

            assert returned is messages
            compressor.on_session_start.assert_not_called()
            assert agent._awaiting_cache_usage_after_compression is False


    def test_no_hook_when_no_session_db(self):
        """Without session_db, session_id does not rotate and the hook is not fired."""
        from run_agent import AIAgent
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=None,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )

        compressor = MagicMock()
        compressor.compress.return_value = [{"role": "user", "content": "x"}]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        agent.context_compressor = compressor

        original_sid = agent.session_id
        agent._compress_context([{"role": "user", "content": "m"}], "sys", approx_tokens=100)

        # No DB => no rotation => no compression-boundary hook
        assert agent.session_id == original_sid
        comp_calls = [
            c for c in compressor.on_session_start.call_args_list
            if c.kwargs.get("boundary_reason") == "compression"
        ]
        assert not comp_calls, (
            f"No compression hook should fire without session_db rotation, "
            f"got {comp_calls!r}"
        )

    def test_hook_failure_does_not_break_compression(self):
        """If the context engine raises from on_session_start, compression still completes."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db)

            compressor = MagicMock()
            compressor.compress.return_value = [{"role": "user", "content": "summary"}]
            compressor.compression_count = 1
            compressor.last_prompt_tokens = 0
            compressor.last_completion_tokens = 0
            compressor._last_summary_error = None
            compressor._last_compress_aborted = False

            # Raise only on the compression-boundary call, not on earlier calls.
            def _raise_on_compression(*args, **kwargs):
                if kwargs.get("boundary_reason") == "compression":
                    raise RuntimeError("plugin exploded")
                return None
            compressor.on_session_start.side_effect = _raise_on_compression
            agent.context_compressor = compressor

            original_sid = agent.session_id

            # Must not raise. Input must be large enough that the fake
            # compressor's one-message summary is a genuine shrink — the
            # no-growth commit guard refuses to rotate on transcript growth.
            compressed, _prompt = agent._compress_context(
                [{"role": "user", "content": "m" * 400}], "sys", approx_tokens=100
            )
            assert compressed
            assert agent.session_id != original_sid


_BOUNDARY_CASES = [
    (mode, outcome)
    for mode in ("rotation", "in-place", "memory-rotation", "memory-in-place")
    for outcome in (
        "success", "success-without-progress-flag", "summary-error", "abort",
        "noop-copy", "noop-identity", "empty", "fence-cancel",
        *(("persist-error", "growth", "notify-error") if not mode.startswith("memory") else ()),
        *(("prompt-error",) if mode == "in-place" else ()),
    )
]


@pytest.mark.parametrize("mode,outcome", _BOUNDARY_CASES)
@pytest.mark.parametrize("pending", [False, True], ids=["fresh", "already-pending"])
def test_public_compression_boundary_outcome_matrix(tmp_path, monkeypatch, mode, outcome, pending):
    """Admission follows the committed boundary, not the summary progress hint."""
    import copy
    from contextlib import ExitStack

    import agent.conversation_compression as compression
    from hermes_state import SessionDB

    assert Path(compression.__file__).resolve() == Path(__file__).resolve().parents[2] / "agent/conversation_compression.py"
    memory_only = mode.startswith("memory")
    in_place = mode.endswith("in-place")
    with ExitStack() as stack:
        db = None if memory_only else SessionDB(db_path=tmp_path / "boundary.db")
        if db is not None:
            stack.callback(db.close)
        agent = TestCompressionBoundaryHook()._make_agent(db)
        agent.compression_in_place = in_place
        agent._awaiting_cache_usage_after_compression = pending
        messages = [{"role": "user", "content": "synthetic " * 100}]
        original_sid = agent.session_id
        if db is not None:
            agent._ensure_db_session()
            db.append_message(original_sid, "user", messages[0]["content"])
            # This entire input is the durable history, not an unflushed turn.
            agent._persist_user_message_idx = len(messages)
        before_rows = db.get_messages(original_sid) if db is not None else None
        fence = compression.CompressionCommitFence()
        compressor = MagicMock()
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_aux_model_failure_model = None
        compressor._last_compress_aborted = outcome == "abort"
        # Deliberately inconsistent hints: semantic no-op never commits,
        # and a real publication need not have a plugin progress flag.
        compressor._last_compression_made_progress = outcome != "success-without-progress-flag"
        compressor.compress.return_value = [{"role": "user", "content": "summary"}]
        if outcome == "summary-error":
            compressor.compress.side_effect = RuntimeError("synthetic summary failure")
        elif outcome in {"noop-copy", "noop-identity"}:
            compressor.compress.side_effect = lambda rows, **kwargs: (
                copy.deepcopy(rows) if outcome == "noop-copy" else rows
            )
        elif outcome == "empty":
            compressor.compress.return_value = []
        elif outcome == "growth":
            compressor.compress.return_value = [{"role": "user", "content": "synthetic " * 1000}]
            monkeypatch.setattr("agent.context_compressor.salvage_grown_transcript", lambda *a, **k: None)
        elif outcome == "fence-cancel":
            def cancel(rows, **kwargs):
                assert fence.cancel_before_commit()
                return [{"role": "user", "content": "summary"}]
            compressor.compress.side_effect = cancel
        agent.context_compressor = compressor
        persist = None
        if db is not None:
            method = "archive_and_compact" if in_place else "publish_compression_child"
            persist = stack.enter_context(patch.object(db, method, wraps=getattr(db, method)))
            if outcome == "persist-error":
                persist.side_effect = RuntimeError("synthetic persist failure")
            if outcome == "prompt-error":
                stack.enter_context(patch.object(db, "update_system_prompt", side_effect=RuntimeError("synthetic prompt failure")))
        before_contents = [row["content"] for row in messages]
        returned = messages
        boundary_seen = []
        if outcome == "notify-error":
            def fail_notification(*args, **kwargs):
                if kwargs.get("boundary_reason") == "compression":
                    boundary_seen.append(agent._awaiting_cache_usage_after_compression)
                    raise RuntimeError("synthetic notification failure")
            compressor.on_session_start.side_effect = fail_notification
        if outcome == "summary-error":
            with pytest.raises(RuntimeError, match="synthetic summary failure"):
                agent._compress_context(messages, "sys", approx_tokens=1000, commit_fence=fence)
        else:
            returned, _ = agent._compress_context(messages, "sys", approx_tokens=1000, commit_fence=fence)
        assert compressor.compress.call_count == 1
        if persist is not None:
            import sqlite3

            # Read committed rows through a different connection, including
            # prompt-update failure after the archive transaction succeeded.
            with sqlite3.connect(tmp_path / "boundary.db") as reader:
                active = reader.execute(
                    "SELECT content FROM messages WHERE session_id=? AND active=1 ORDER BY id",
                    (agent.session_id,),
                ).fetchall()
                archived = reader.execute(
                    "SELECT count(*) FROM messages WHERE session_id=? AND active=0 AND compacted=1",
                    (original_sid,),
                ).fetchone()[0]
            published = active == [("summary",)]
            assert persist.call_count == int(published or outcome == "persist-error")
            if published:
                assert [row["content"] for row in returned] == ["summary"]
                assert (agent.session_id != original_sid) == (not in_place)
                assert agent._last_compaction_in_place is in_place
                assert archived == int(in_place)
            else:
                assert [row[0] for row in active] == before_contents
                assert db.get_messages(original_sid) == before_rows
        else:
            assert agent._session_db is None
            assert agent.session_id == original_sid
            published = [row["content"] for row in returned] != before_contents
        assert agent._awaiting_cache_usage_after_compression is (pending or published)
        if outcome == "notify-error":
            assert boundary_seen == [True]
        notifications = [c for c in compressor.on_session_start.call_args_list
                         if c.kwargs.get("boundary_reason") == "compression"]
        # Notification requires prompt bookkeeping too; cache attribution does not.
        assert len(notifications) == int(published and not memory_only and outcome != "prompt-error")


class TestSessionCompressEvent:
    """The session:compress event_callback fires after a compression split."""

    def _make_agent(self, session_db, event_callback=None):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
                event_callback=event_callback,
            )
            # ROTATION fallback — pin in_place=False regardless of default (#38763).
            agent.compression_in_place = False
            return agent

    def _stub_compressor(self):
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        return compressor

    def test_event_emitted_on_compression(self):
        from hermes_state import SessionDB

        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(
                db, event_callback=lambda et, ctx: events.append((et, ctx))
            )
            original_sid = agent.session_id
            agent.context_compressor = self._stub_compressor()

            agent._compress_context(
                [{"role": "user", "content": f"m{i}"} for i in range(10)],
                "sys",
                approx_tokens=10_000,
            )

            compress_events = [e for e in events if e[0] == "session:compress"]
            assert compress_events, f"session:compress not emitted, got {events!r}"
            _, ctx = compress_events[-1]
            assert ctx["session_id"] == agent.session_id
            assert ctx["old_session_id"] == original_sid
            assert ctx["compression_count"] == 1

    def test_no_callback_is_safe(self):
        """Compression must work when no event_callback is wired."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db, event_callback=None)
            agent.context_compressor = self._stub_compressor()
            compressed, _ = agent._compress_context(
                [{"role": "user", "content": "m"}], "sys", approx_tokens=100
            )
            assert compressed

