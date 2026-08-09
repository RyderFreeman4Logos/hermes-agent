"""#66/#78: completion ACK follows provider and durable publication."""

from __future__ import annotations

import queue
import threading
import types

import pytest

from hermes_state import SessionDB
from tui_gateway import server


class _InlineThread:
    def __init__(self, target=None, daemon=None, **_kwargs):
        self._target = target

    def start(self):
        if self._target:
            self._target()

    def is_alive(self):
        return False


class _HeldThread(_InlineThread):
    def start(self):
        pass

    def run(self):
        if self._target:
            self._target()


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: False)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: None)


def _session(agent):
    return {
        "agent": agent,
        "session_key": "active",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
    }


def test_ordinary_idle_dispatch_acks_after_provider_and_history_commit(
    turn_env, monkeypatch
):
    """Starting the daemon turn is not an ordinary completion ACK."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "restore_undelivered_completions", lambda _queue: 0)
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    monkeypatch.setattr(server.threading, "Thread", _HeldThread)
    provider_finished = []
    acknowledgements = []
    prompt = "[ordinary completion event]"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "terminal"},
    ]

    def run_conversation(message, **_kwargs):
        assert message == prompt
        provider_finished.append(True)
        return {
            "final_response": "terminal",
            "messages": messages,
            "completion_delivery_status": "committed",
        }

    agent = types.SimpleNamespace(
        model="test",
        provider="test",
        clear_interrupt=lambda: None,
        run_conversation=run_conversation,
    )
    session = _session(agent)
    original_complete = registry.complete_completion_delivery

    def complete(evt):
        acknowledgements.append((list(provider_finished), list(session["history"])))
        original_complete(evt)

    monkeypatch.setattr(registry, "complete_completion_delivery", complete)
    event = {
        "type": "completion",
        "session_id": "proc-ordinary",
        "session_key": "active",
        "started_at": 1.0,
    }

    server._dispatch_completion_batch(
        "sid",
        session,
        [{"evt": event, "model_text": prompt, "text": prompt}],
        consumer="tui-test",
    )

    assert acknowledgements == []
    session["_run_thread"].run()
    assert acknowledgements == [([True], messages)]


def test_completion_history_conflict_gets_explicit_recovery_outcome(turn_env):
    """A durable suffix cannot be called delivered when live history rejects it."""
    outcomes = []
    session = _session(None)

    def run_conversation(message, **_kwargs):
        with session["history_lock"]:
            session["history_version"] += 1
        return {
            "final_response": "terminal",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "terminal"},
            ],
            "completion_delivery_status": "committed",
        }

    session["agent"] = types.SimpleNamespace(
        model="test",
        provider="test",
        clear_interrupt=lambda: None,
        run_conversation=run_conversation,
    )

    server._run_prompt_submit(
        "rid",
        "sid",
        session,
        "[ordinary completion event]",
        completion_delivery=True,
        completion_delivery_callback=outcomes.append,
    )

    assert outcomes == ["history_conflict"]
    assert session["history"] == []


def test_committed_failure_suffix_is_a_terminal_delivery(turn_env):
    """A durable failed/effect suffix is an explicit non-replay disposition."""
    outcomes = []
    messages = [
        {"role": "user", "content": "[ordinary completion event]"},
        {"role": "assistant", "content": "effect failed durably"},
    ]

    def run_conversation(_message, **_kwargs):
        return {
            "final_response": "",
            "messages": messages,
            "failed": True,
            "partial": True,
            "error": "provider failed after effect",
            "completion_delivery_status": "committed",
        }

    session = _session(
        types.SimpleNamespace(
            model="test",
            provider="test",
            clear_interrupt=lambda: None,
            run_conversation=run_conversation,
        )
    )

    server._run_prompt_submit(
        "rid",
        "sid",
        session,
        "[ordinary completion event]",
        completion_delivery=True,
        completion_delivery_callback=outcomes.append,
    )

    assert outcomes == ["committed"]
    assert session["history"] == messages


@pytest.mark.parametrize("rotate", [False, True], ids=["active", "rotated"])
def test_completion_ack_waits_for_terminal_active_suffix_cas(
    turn_env, tmp_path, rotate
):
    """Real SessionDB: a compacted marker is never acknowledged early."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="active", source="tui", model="test")
    provider_finished = []
    acknowledgements = []
    prompt = "[completion event]"

    def run_conversation(message, **_kwargs):
        assert message == prompt
        db.append_message("active", role="user", content=prompt)
        provider_finished.append(True)
        if rotate:
            db.archive_and_compact("active", [])
        return {"final_response": "terminal", "messages": []}

    agent = types.SimpleNamespace(
        session_id="active",
        _session_db=db,
        model="test",
        provider="test",
        clear_interrupt=lambda: None,
        run_conversation=run_conversation,
    )
    try:
        server._run_prompt_submit(
            "rid",
            "sid",
            _session(agent),
            prompt,
            display_kind="async_delegation_complete",
            display_metadata={"completion_delivery_status": "effect_started"},
            completion_delivery_callback=acknowledgements.append,
        )

        assert provider_finished == [True]
        assert acknowledgements == (
            ["committed"] if not rotate else ["missing_active_marker"]
        )
        rows = db.get_messages_as_conversation("active")
        assert len(rows) == (0 if rotate else 1)
        if not rotate:
            assert (
                rows[0]["display_metadata"]["completion_delivery_status"]
                == "effect_started"
            )
    finally:
        db.close()


def test_ack_after_suffix_commit_survives_repeated_wakeups(
    turn_env, tmp_path, monkeypatch
):
    """After the commit/ACK boundary, a second wakeup cannot rerun the effect."""
    from tools import async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="active", source="tui", model="test")
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg-after-ack",
        "session_key": "active",
        "dispatched_at": 1.0,
        "completed_at": 2.0,
        "summary": "done",
    }
    ad._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": "active",
        "origin_ui_session_id": "sid",
        "parent_session_id": "active",
        "dispatched_at": 1.0,
    })
    ad._persist_completion(event, {"status": "completed", "summary": "done"})
    calls = []

    def run_conversation(message, **_kwargs):
        calls.append(message)
        db.append_message("active", role="user", content=message)
        return {"final_response": "terminal", "messages": []}

    agent = types.SimpleNamespace(
        session_id="active",
        _session_db=db,
        model="test",
        provider="test",
        clear_interrupt=lambda: None,
        run_conversation=run_conversation,
    )
    try:
        claim = "claim-1"
        assert ad.claim_completion_delivery(event["delegation_id"], claim)
        server._run_prompt_submit(
            "rid",
            "sid",
            _session(agent),
            "[completion event]",
            display_kind="async_delegation_complete",
            completion_delivery_callback=lambda outcome: (
                server._finish_completion_claim(event, claim, outcome)
            ),
        )
        assert (
            ad.get_durable_delegation(event["delegation_id"])["delivery_state"]
            == "delivered"
        )
        assert not ad.claim_completion_delivery(event["delegation_id"], "claim-2")
        assert calls == ["[completion event]"]
    finally:
        db.close()


def test_startup_restore_makes_effect_started_completion_retryable(
    tmp_path, monkeypatch
):
    """A crash before suffix/CAS acknowledgement restores the durable claim."""
    from tools import async_delegation as ad

    db_path = tmp_path / "state.db"
    monkeypatch.setattr(ad, "_db_path", lambda: db_path)
    db = SessionDB(db_path)
    db.create_session(session_id="active", source="tui", model="test")
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg-pre-ack-crash",
        "session_key": "active",
        "dispatched_at": 1.0,
        "completed_at": 2.0,
        "summary": "done",
    }
    try:
        ad._persist_dispatch({
            "delegation_id": event["delegation_id"],
            "session_key": "active",
            "origin_ui_session_id": "sid",
            "parent_session_id": "active",
            "dispatched_at": 1.0,
        })
        ad._persist_completion(event, {"status": "completed", "summary": "done"})
        assert ad.claim_completion_delivery(event["delegation_id"], "crashed-claim")
        durable = ad.get_durable_delegation(event["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "effect_started"

        restored_events = queue.Queue()
        assert ad.restore_undelivered_completions(restored_events) == 1
        assert restored_events.get_nowait()["restored"] is True
        durable = ad.get_durable_delegation(event["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "pending"
        assert ad.claim_completion_delivery(event["delegation_id"], "retry-claim")
        assert ad.complete_completion_delivery(event["delegation_id"], "retry-claim")
        durable = ad.get_durable_delegation(event["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "delivered"
    finally:
        db.close()


def test_recovery_marker_prevents_completion_replay_but_cached_agent_still_runs(
    turn_env, tmp_path, monkeypatch
):
    """A lost active marker gets one durable recovery state, never a zero-API loop."""
    from tools import async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="active", source="tui", model="test")
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg-crash-boundary",
        "session_key": "active",
        "dispatched_at": 1.0,
        "completed_at": 2.0,
        "summary": "done",
    }
    ad._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": "active",
        "origin_ui_session_id": "sid",
        "parent_session_id": "active",
        "dispatched_at": 1.0,
    })
    ad._persist_completion(event, {"status": "completed", "summary": "done"})
    calls = []
    prompt = "[completion event]"

    def run_conversation(message, **_kwargs):
        calls.append(message)
        if message == prompt:
            db.append_message("active", role="user", content=prompt)
            # Crash boundary: the completion effect ran, then compression
            # removed the active row before the suffix CAS can finish.
            db.archive_and_compact("active", [])
        return {"final_response": "terminal", "messages": []}

    agent = types.SimpleNamespace(
        session_id="active",
        _session_db=db,
        model="test",
        provider="test",
        clear_interrupt=lambda: None,
        run_conversation=run_conversation,
    )
    try:
        claim = "claim-1"
        assert ad.claim_completion_delivery(event["delegation_id"], claim)
        assert (
            ad.get_durable_delegation(event["delegation_id"])["delivery_state"]
            == "effect_started"
        )

        server._run_prompt_submit(
            "rid-1",
            "sid",
            _session(agent),
            prompt,
            display_kind="async_delegation_complete",
            completion_delivery_callback=lambda outcome: (
                server._finish_completion_claim(event, claim, outcome)
            ),
        )

        state = ad.get_durable_delegation(event["delegation_id"])["delivery_state"]
        assert calls == [prompt]
        assert state == "recovery_missing_active_marker"
        assert not ad.claim_completion_delivery(event["delegation_id"], "claim-2")

        # Reusing the same cached AIAgent for the next real prompt still makes
        # a provider turn; the failed completion did not latch a zero-API error.
        server._run_prompt_submit("rid-2", "sid", _session(agent), "real follow-up")
        assert calls == [prompt, "real follow-up"]
    finally:
        db.close()
