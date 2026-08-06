from __future__ import annotations

import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_state import CompressionSessionBusyError, SessionDB
from run_agent import AIAgent


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _agent(session_db, session_id: str = "parent") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = session_id
    agent._session_db = session_db
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
        try:
            result["value"] = agent._flush_messages_to_session_db(messages, [])
        except BaseException as exc:  # surface thread failures to the test
            result["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_run)
    thread.start()
    return thread, done, result


def _observe_busy(db: SessionDB, monkeypatch) -> threading.Event:
    busy = threading.Event()
    real_append = db.append_messages_batch

    def _append(*args, **kwargs):
        try:
            return real_append(*args, **kwargs)
        except CompressionSessionBusyError:
            busy.set()
            raise

    monkeypatch.setattr(db, "append_messages_batch", _append)
    return busy


def test_active_lease_waits_for_successor_then_persists_on_child(
    db: SessionDB, monkeypatch
) -> None:
    db.create_session("parent", source="tui")
    assert db.try_acquire_compression_lock("parent", "holder", ttl_seconds=60)
    busy = _observe_busy(db, monkeypatch)
    agent = _agent(db)
    messages = [{"role": "assistant", "content": "completed turn"}]

    thread, done, result = _flush_in_thread(agent, messages)
    assert busy.wait(timeout=2)
    assert not done.is_set()

    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="tui",
        messages=[{"role": "user", "content": "compressed handoff"}],
        compression_lock_holder="holder",
    )
    thread.join(timeout=2)
    db.release_compression_lock("parent", "holder")

    assert not thread.is_alive()
    assert "error" not in result
    assert result["value"] is True
    assert agent.session_id == "child"
    assert db.get_messages("parent") == []
    assert [m["content"] for m in db.get_messages("child")] == [
        "compressed handoff",
        "completed turn",
    ]


def test_expired_lease_without_successor_retries_live_parent(
    db: SessionDB, monkeypatch
) -> None:
    db.create_session("parent", source="cli")
    assert db.try_acquire_compression_lock("parent", "holder", ttl_seconds=0.15)
    busy = _observe_busy(db, monkeypatch)
    agent = _agent(db)
    messages = [{"role": "user", "content": "parent remains authoritative"}]

    thread, _done, result = _flush_in_thread(agent, messages)
    assert busy.wait(timeout=2)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert "error" not in result
    assert result["value"] is True
    assert agent.session_id == "parent"
    assert [m["content"] for m in db.get_messages("parent")] == [
        "parent remains authoritative"
    ]


def test_successor_publication_race_adoption_is_idempotent(
    db: SessionDB, monkeypatch
) -> None:
    db.create_session("parent", source="tui")
    assert db.try_acquire_compression_lock("parent", "holder", ttl_seconds=60)
    real_append = db.append_messages_batch
    published = False

    def _publish_between_checks(*args, **kwargs):
        nonlocal published
        try:
            return real_append(*args, **kwargs)
        except CompressionSessionBusyError:
            if not published:
                published = True
                db.publish_compression_child(
                    parent_session_id="parent",
                    child_session_id="child",
                    source="tui",
                    messages=[{"role": "user", "content": "handoff"}],
                    compression_lock_holder="holder",
                )
            raise

    monkeypatch.setattr(db, "append_messages_batch", _publish_between_checks)
    agent = _agent(db)
    messages = [{"role": "assistant", "content": "race-safe turn"}]

    assert agent._flush_messages_to_session_db(messages, []) is True
    assert agent._flush_messages_to_session_db(messages, []) is True
    db.release_compression_lock("parent", "holder")

    assert agent.session_id == "child"
    assert db.get_messages("parent") == []
    assert [m["content"] for m in db.get_messages("child")] == [
        "handoff",
        "race-safe turn",
    ]


def test_interrupt_stops_compression_wait(db: SessionDB, monkeypatch) -> None:
    db.create_session("parent", source="cli")
    assert db.try_acquire_compression_lock("parent", "holder", ttl_seconds=60)
    busy = _observe_busy(db, monkeypatch)
    agent = _agent(db)
    messages = [{"role": "user", "content": "interrupted turn"}]

    thread, _done, result = _flush_in_thread(agent, messages)
    assert busy.wait(timeout=2)
    agent._interrupt_requested = True
    thread.join(timeout=1)
    db.release_compression_lock("parent", "holder")

    assert not thread.is_alive()
    assert "error" not in result
    assert result["value"] is None
    assert db.get_messages("parent") == []
    assert not messages[0].get("_db_persisted")


def test_active_lease_without_successor_times_out_fatally() -> None:
    session_db = SimpleNamespace(
        append_messages_batch=MagicMock(
            side_effect=CompressionSessionBusyError("busy")
        ),
        _TRANSCRIPT_WRITE_PATIENCE_S=0.12,
    )
    agent = _agent(session_db)
    started = time.monotonic()

    assert (
        agent._flush_messages_to_session_db(
            [{"role": "user", "content": "must remain uncommitted"}], []
        )
        is False
    )
    elapsed = time.monotonic() - started

    assert elapsed >= 0.08
    assert 1 < session_db.append_messages_batch.call_count < 10


@pytest.mark.parametrize("child_count", [0, 2])
def test_closed_parent_without_authoritative_successor_fails_closed(
    db: SessionDB, child_count: int
) -> None:
    db.create_session("parent", source="cli")
    db.end_session("parent", "compression")
    for index in range(child_count):
        db.create_session(f"child-{index}", source="cli", parent_session_id="parent")
        db.append_message(f"child-{index}", "user", f"handoff-{index}")
    agent = _agent(db)

    assert (
        agent._flush_messages_to_session_db(
            [{"role": "assistant", "content": "must not guess"}], []
        )
        is False
    )
    assert agent.session_id == "parent"
    assert db.get_messages("parent") == []


def test_real_storage_error_keeps_exact_fatal_safeguard() -> None:
    session_db = SimpleNamespace()
    session_db.append_messages_batch = MagicMock(
        side_effect=sqlite3.OperationalError("disk I/O error")
    )
    agent = _agent(session_db)

    assert (
        agent._flush_messages_to_session_db([{"role": "user", "content": "turn"}], [])
        is False
    )
    session_db.append_messages_batch.assert_called_once()
    assert AIAgent._format_turn_completion_explanation(
        "session_persistence_failed"
        ) == (
            "⚠️ No reply: the turn was stopped because session storage could not be "
            "written (the transcript would have been lost on restart). Check the state "
            "database health (`hermes doctor`), then send your message again."
        )
