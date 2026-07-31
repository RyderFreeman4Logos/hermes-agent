"""Unit tests for the TurnContext/TurnRunner seam extracted from
``GatewayRunner._run_agent_inner`` (gateway/turn_context.py + gateway/run.py).

The extraction contract: the closure bodies moved onto ``TurnRunner`` methods
byte-identically (modulo local -> ctx.field rewrites), with every closed-over
local carried as a ``TurnContext`` field. These tests pin the seam's wiring —
shared mutable containers, no-queue early returns — not the progress behavior
itself (that's covered by test_run_progress_topics.py et al.).
"""

import asyncio
import os
import queue as queue_mod
from types import SimpleNamespace

import pytest

from gateway.turn_context import TurnContext


def _make_runner(ctx):
    from gateway.run import TurnRunner

    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return None

    return TurnRunner(_StubGatewayRunner(), ctx)


class TestTurnContext:
    def test_defaults_are_independent_containers(self):
        a, b = TurnContext(), TurnContext()
        a.last_progress_msg[0] = "x"
        a.repeat_count[0] = 3
        a._cleanup_msg_ids.append("1")
        assert b.last_progress_msg == [None]
        assert b.repeat_count == [0]
        assert b._cleanup_msg_ids == []

    def test_shared_containers_visible_to_outer_scope(self):
        # The outer body and the runner share the SAME list objects, so
        # mutation through the ctx is visible to locals captured elsewhere.
        last_progress_msg = [None]
        ctx = TurnContext(last_progress_msg=last_progress_msg)
        ctx.last_progress_msg[0] = "🔍 web_search"
        assert last_progress_msg[0] == "🔍 web_search"


class TestTurnRunner:
    def test_methods_exist_and_bind(self):
        from gateway.run import TurnRunner

        ctx = TurnContext()
        runner = _make_runner(ctx)
        assert callable(runner.progress_callback)
        assert asyncio.iscoroutinefunction(TurnRunner.send_progress_messages)
        assert runner._ctx is ctx

    def test_send_progress_messages_no_queue_returns(self):
        ctx = TurnContext(progress_queue=None)
        runner = _make_runner(ctx)
        assert asyncio.run(runner.send_progress_messages()) is None

    def test_send_progress_messages_no_adapter_returns(self):
        ctx = TurnContext(progress_queue=queue_mod.Queue())
        runner = _make_runner(ctx)  # stub adapter resolver returns None
        assert asyncio.run(runner.send_progress_messages()) is None

    def test_gateway_turn_start_claims_only_its_pending_completion(
        self, monkeypatch, tmp_path
    ):
        from gateway.run import TurnRunner
        from tools import async_delegation as ad

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        delegation_id = "deleg-gateway-turn-start"
        session_key = "agent:main:telegram:dm:123"
        ad._persist_dispatch(
            {
                "delegation_id": delegation_id,
                "session_key": session_key,
                "origin_ui_session_id": "",
                "parent_session_id": "gateway-session-id",
                "origin_session_id": "",
                "dispatched_at": 1.0,
                "goal": "gateway child",
            }
        )
        ad._persist_completion(
            {
                "type": "async_delegation",
                "delegation_id": delegation_id,
                "session_key": session_key,
                "status": "completed",
                "summary": "gateway child done",
                "dispatched_at": 1.0,
                "completed_at": 2.0,
            },
            {"summary": "gateway child done", "status": "completed"},
        )
        with ad._DB_LOCK, ad._connect() as conn:
            conn.execute(
                "UPDATE async_delegations SET owner_pid=? WHERE delegation_id=?",
                (os.getpid(), delegation_id),
            )

        class _Gateway:
            _session_db = SimpleNamespace(
                _db=SimpleNamespace(resolve_resume_session_id=lambda value: value)
            )

        ctx = TurnContext(
            session_key=session_key,
            session_id="gateway-session-id",
        )
        runner = TurnRunner(_Gateway(), ctx)
        claimed = runner._claim_turn_start_async_completions(
            SimpleNamespace(session_id="gateway-session-id")
        )

        assert len(claimed) == 1
        assert claimed[0][0]["delegation_id"] == delegation_id
        assert "gateway child done" in claimed[0][2]
        assert ad.get_durable_delegation(delegation_id)["delivery_attempts"] >= 1
        ad.complete_event_delivery(claimed[0][0], claimed[0][1])
