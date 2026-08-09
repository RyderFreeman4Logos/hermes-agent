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


def test_completion_preflight_rejection_releases_claim(turn_env, monkeypatch):
    """A TUI completion rejected before provider startup remains retryable."""
    monkeypatch.setattr(
        "agent.context_references.preprocess_context_references",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            blocked=True,
            warnings=["blocked completion context"],
            message="",
        ),
    )
    agent = types.SimpleNamespace(
        model="test",
        provider="test",
        base_url="",
        api_key="",
        clear_interrupt=lambda: None,
        run_conversation=lambda *_args, **_kwargs: pytest.fail(
            "provider must not run after preflight rejection"
        ),
    )
    outcomes = []

    server._run_prompt_submit(
        "rid",
        "sid",
        _session(agent),
        "@blocked",
        completion_delivery=True,
        completion_delivery_callback=outcomes.append,
    )

    assert outcomes == ["provider_failed"]


def test_no_effect_provider_failure_releases_ordinary_claim(turn_env, monkeypatch):
    """A failed provider turn with no durable suffix is never acknowledged."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "restore_undelivered_completions", lambda _queue: 0)
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    event = {
        "type": "completion",
        "session_id": "proc-provider-failure",
        "session_key": "active",
        "started_at": 1.0,
    }
    session = _session(types.SimpleNamespace(
        model="test",
        provider="test",
        clear_interrupt=lambda: None,
        run_conversation=lambda *_args, **_kwargs: {
            "final_response": "",
            "messages": [],
            "failed": True,
            "partial": True,
            "error": "provider unavailable",
            "completion_delivery_status": "dropped",
        },
    ))

    server._dispatch_completion_batch(
        "sid",
        session,
        [{"evt": event, "model_text": "[completion event]", "text": "event"}],
        consumer="tui-test",
    )

    assert registry.completion_event_should_deliver(event)
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] in {"pending", "dropped"}
    assert session["history"] == []


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


def test_history_conflict_commits_ordinary_recovery_before_ack(
    turn_env, monkeypatch, tmp_path
):
    """An ordinary claim gets a durable recovery receipt before suppression."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    event = {
        "type": "completion",
        "session_id": "proc-history-conflict",
        "session_key": "active",
        "started_at": 2.0,
    }
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

    server._dispatch_completion_batch(
        "sid",
        session,
        [{"evt": event, "model_text": "[completion event]", "text": "event"}],
        consumer="tui-test",
    )

    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "recovery_history_conflict"
    assert not registry.completion_event_should_deliver(event)


def test_failed_ordinary_recovery_transition_leaves_claim_for_retry(
    monkeypatch, tmp_path
):
    """A failed recovery write cannot become a process-registry ACK."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    event = {
        "type": "completion",
        "session_id": "proc-recovery-write-failed",
        "session_key": "active",
        "started_at": 3.0,
    }
    assert registry.claim_completion_delivery(event)
    claim = ad.claim_event_delivery(event, "tui-test")
    assert claim
    monkeypatch.setattr(ad, "mark_completion_delivery_recovery", lambda *_args: False)

    server._finish_completion_claim(event, claim, "history_conflict")

    assert registry.completion_event_should_deliver(event)
    assert registry.completion_queue.get_nowait() == event
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "pending"


def test_visibility_noop_requires_durable_ack(monkeypatch, tmp_path):
    """A suppressed completion is retryable when its durable ACK fails."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    monkeypatch.setattr(
        server, "_compose_completion_batch_prompt", lambda _items: ("", True)
    )
    monkeypatch.setattr(ad, "complete_event_delivery", lambda *_args: False)
    event = {
        "type": "completion",
        "session_id": "proc-visibility-noop",
        "session_key": "active",
        "started_at": 3.5,
    }

    server._dispatch_completion_batch(
        "sid",
        _session(None),
        [{"evt": event, "model_text": "ignored", "text": "ignored"}],
        consumer="tui-test",
    )

    assert registry.completion_event_should_deliver(event)
    assert registry.completion_queue.get_nowait() == event
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "pending"


def test_ordinary_completion_claim_restores_after_restart(monkeypatch, tmp_path):
    """A restart requeues an unfinished ordinary completion from state.db."""
    import queue

    from tools import async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    event = {
        "type": "completion",
        "session_id": "proc-restart",
        "session_key": "active",
        "started_at": 4.0,
    }
    assert ad.persist_event_delivery(event)
    assert ad.claim_event_delivery(event, "dead-consumer")
    restarted_queue = queue.Queue()

    # A sibling surface may initialize its registry while this claim is live;
    # it must not steal or replay the in-flight provider effect.
    assert ad.restore_undelivered_completions(restarted_queue) == 0
    assert restarted_queue.empty()
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: False)

    assert ad.restore_undelivered_completions(restarted_queue) == 1
    restored = restarted_queue.get_nowait()
    assert {key: restored[key] for key in event} == event
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "pending"
    assert receipt["delivery_claim"] is None


def test_process_completion_is_durable_before_checkpoint_and_queue(
    monkeypatch, tmp_path
):
    """The producer writes the ordinary receipt before retiring its checkpoint."""
    from tools import async_delegation as ad
    from tools.process_registry import ProcessRegistry, ProcessSession

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = ProcessRegistry()
    order = []
    persist = ad.persist_event_delivery

    def record_persist(event):
        order.append("persist")
        return persist(event)

    monkeypatch.setattr(ad, "persist_event_delivery", record_persist)
    monkeypatch.setattr(
        registry, "_write_checkpoint", lambda: order.append("checkpoint")
    )
    process = ProcessSession(
        id="proc-durable-producer",
        command="true",
        session_key="active",
        started_at=5.0,
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._running[process.id] = process

    registry._move_to_finished(process)

    assert order == ["persist", "checkpoint"]
    event = registry.completion_queue.get_nowait()
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "pending"


def test_process_completion_persist_failure_keeps_restart_checkpoint(
    monkeypatch,
):
    """A failed receipt write cannot erase the producer's restart source."""
    from tools import async_delegation as ad
    from tools.process_registry import ProcessRegistry, ProcessSession

    monkeypatch.setattr(ad, "restore_undelivered_completions", lambda _queue: 0)
    registry = ProcessRegistry()
    checkpoints = []
    monkeypatch.setattr(
        ad,
        "persist_event_delivery",
        lambda _event: (_ for _ in ()).throw(OSError("storage unavailable")),
    )
    monkeypatch.setattr(
        registry, "_write_checkpoint", lambda: checkpoints.append(True)
    )
    process = ProcessSession(
        id="proc-persist-failure",
        command="true",
        session_key="active",
        started_at=5.5,
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._running[process.id] = process

    registry._move_to_finished(process)

    assert checkpoints == []
    assert registry.completion_queue.get_nowait()["session_id"] == process.id


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
