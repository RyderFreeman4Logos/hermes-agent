from __future__ import annotations

import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import CompressionSessionBusyError, SessionDB
from run_agent import AIAgent


def _agent(db, session_id: str) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = session_id
    agent._session_db = db
    agent._session_db_created = True
    agent._session_persist_lock = threading.RLock()
    agent._persist_disabled = False
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._pending_cli_user_message = None
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_session_id = session_id
    agent._flushed_db_message_ids = set()
    agent._db_flush_scan_prefix = None
    agent._active_compression_lock_holder = None
    agent._interrupt_requested = False
    agent._memory_manager = None
    agent._gateway_session_key = "route"
    agent.context_compressor = MagicMock()
    agent.platform = "tui"
    return agent


def _flush_in_thread(agent: AIAgent, messages: list[dict]):
    result: dict[str, object] = {}
    done = threading.Event()

    def _run() -> None:
        result["value"] = agent._flush_messages_to_session_db(messages, [])
        done.set()

    thread = threading.Thread(target=_run)
    thread.start()
    return thread, done, result


def test_active_lease_waits_then_release_retries_parent(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="cli")
    assert db.try_acquire_compression_lock("parent", "holder", ttl_seconds=60)
    agent = _agent(db, "parent")
    messages = [{"role": "user", "content": "healthy turn"}]

    thread, done, result = _flush_in_thread(agent, messages)
    assert not done.wait(timeout=0.15)
    assert db.get_messages("parent") == []

    db.release_compression_lock("parent", "holder")
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result["value"] is True
    assert [m["content"] for m in db.get_messages("parent")] == ["healthy turn"]


def test_expired_lease_without_publication_retries_parent(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="cli")
    assert db.try_acquire_compression_lock("parent", "dead-holder", ttl_seconds=0.2)
    agent = _agent(db, "parent")
    messages = [{"role": "user", "content": "surviving turn"}]

    thread, done, result = _flush_in_thread(agent, messages)
    assert not done.wait(timeout=0.1)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result["value"] is True
    assert agent.session_id == "parent"
    assert [m["content"] for m in db.get_messages("parent")] == ["surviving turn"]


def test_publication_while_waiting_adopts_child_and_preserves_order(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="tui")
    assert db.try_acquire_compression_lock("parent", "holder", ttl_seconds=60)
    agent = _agent(db, "parent")
    messages = [
        {"role": "assistant", "content": "tool request"},
        {"role": "tool", "content": "tool result", "tool_name": "terminal"},
    ]

    thread, done, result = _flush_in_thread(agent, messages)
    assert not done.wait(timeout=0.15)
    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="tui",
        messages=[{"role": "user", "content": "compressed handoff"}],
        compression_lock_holder="holder",
    )
    db.release_compression_lock("parent", "holder")
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result["value"] is True
    assert agent.session_id == "child"
    assert [m["content"] for m in db.get_messages("parent")] == []
    assert [m["content"] for m in db.get_messages("child")] == [
        "compressed handoff",
        "tool request",
        "tool result",
    ]
    assert all(message.get("_db_persisted") for message in messages)

    from tui_gateway import server

    session = {"agent": agent, "session_key": "parent"}
    with patch.object(server, "_transfer_active_session_slot", return_value=True), patch.object(
        server, "_restart_slash_worker"
    ):
        server._sync_session_key_after_compress("ui", session)
    assert session["session_key"] == "child"


@pytest.mark.parametrize("child_count", [0, 2])
def test_closed_parent_without_unique_child_fails_closed(tmp_path, child_count: int) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="cli")
    db.end_session("parent", "compression")
    for index in range(child_count):
        child = f"child-{index}"
        db.create_session(child, source="cli", parent_session_id="parent")
        db.append_message(child, "user", f"handoff-{index}")
    agent = _agent(db, "parent")

    assert agent._flush_messages_to_session_db(
        [{"role": "assistant", "content": "must not fork"}], []
    ) is False
    assert agent.session_id == "parent"
    assert db.get_messages("parent") == []


def test_genuine_db_error_fails_promptly() -> None:
    db = SimpleNamespace()
    db.append_message = MagicMock(side_effect=sqlite3.OperationalError("disk I/O error"))
    agent = _agent(db, "parent")

    assert agent._flush_messages_to_session_db(
        [{"role": "user", "content": "turn"}], []
    ) is False
    db.append_message.assert_called_once()


def test_interrupt_stops_wait_without_spin_or_duplicate_rows(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="cli")
    assert db.try_acquire_compression_lock("parent", "holder", ttl_seconds=60)
    real_append = db.append_message
    attempts = 0

    def _counted_append(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return real_append(*args, **kwargs)

    db.append_message = _counted_append
    agent = _agent(db, "parent")
    messages = [{"role": "user", "content": "interrupted turn"}]
    thread, done, result = _flush_in_thread(agent, messages)
    assert not done.wait(timeout=0.15)

    agent._interrupt_requested = True
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert result["value"] is None
    assert attempts < 10
    assert db.get_messages("parent") == []

    db.release_compression_lock("parent", "holder")
    agent._interrupt_requested = False
    assert agent._flush_messages_to_session_db(messages, []) is True
    assert [m["content"] for m in db.get_messages("parent")] == ["interrupted turn"]


def test_busy_exception_is_the_only_retryable_storage_error() -> None:
    db = SimpleNamespace()
    db.append_message = MagicMock(
        side_effect=[CompressionSessionBusyError("busy"), sqlite3.OperationalError("broken")]
    )
    agent = _agent(db, "parent")

    assert agent._flush_messages_to_session_db(
        [{"role": "user", "content": "turn"}], []
    ) is False
    assert db.append_message.call_count == 2
