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


def test_tui_post_turn_visibility_suppression_is_durable(
    turn_env, monkeypatch, tmp_path
):
    """The TUI post-turn drain records an explicit no-op disposition."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    monkeypatch.setattr(
        registry_module, "completion_delivery_prompt", lambda _evt, _text: None
    )
    event = {
        "type": "completion",
        "session_id": "proc-tui-post-turn-suppressed",
        "session_key": "active",
        "started_at": 3.6,
        "command": "true",
        "exit_code": 0,
        "completion_reason": "exited",
        "termination_source": "",
        "output": "done",
    }
    assert ad.persist_event_delivery(event)
    registry.completion_queue.put(event)
    agent = types.SimpleNamespace(
        model="test",
        provider="test",
        clear_interrupt=lambda: None,
        run_conversation=lambda message, **_kwargs: {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "ok"},
            ],
        },
    )

    server._run_prompt_submit("rid", "sid", _session(agent), "ordinary user turn")

    assert registry.completion_queue.empty()
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "recovery_visibility_suppressed"


def test_ordinary_completion_claim_parks_after_dead_owner_restart(
    monkeypatch, tmp_path
):
    """A dead effect owner is parked instead of replaying an ambiguous effect."""
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

    assert ad.restore_undelivered_completions(restarted_queue) == 0
    assert restarted_queue.empty()
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "recovery_effect_started"
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


def test_process_completion_persist_failure_recovers_after_restart(
    monkeypatch, tmp_path
):
    """A failed initial receipt write materializes from the terminal checkpoint."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module
    from tools.process_registry import ProcessRegistry, ProcessSession

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(
        registry_module, "CHECKPOINT_PATH", tmp_path / "processes.json"
    )
    registry = ProcessRegistry()
    persist = ad.persist_event_delivery
    attempts = 0

    def fail_once(event):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("storage unavailable")
        return persist(event)

    monkeypatch.setattr(ad, "persist_event_delivery", fail_once)
    process = ProcessSession(
        id="proc-persist-failure",
        command="true",
        session_key="active",
        started_at=5.5,
        pid=999999999,
        exited=False,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._running[process.id] = process
    assert registry._write_checkpoint()
    process.exited = True

    registry._move_to_finished(process)

    assert registry.completion_queue.get_nowait()["session_id"] == process.id
    restarted = ProcessRegistry()
    assert restarted.recover_from_checkpoint() == 0
    restored = restarted.completion_queue.get_nowait()
    assert restored["session_id"] == process.id
    assert restarted.completion_queue.empty()
    receipt = ad.get_durable_event_delivery(restored)
    assert receipt is not None
    assert receipt["delivery_state"] == "pending"


def test_committed_effect_complete_exception_never_requeues_provider(
    monkeypatch, tmp_path
):
    """A visible effect becomes ACK-only even when durable completion fails."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    event = {
        "type": "completion",
        "session_id": "proc-committed-ack-only",
        "session_key": "active",
        "started_at": 5.6,
    }
    assert registry.claim_completion_delivery(event)
    claim = ad.claim_event_delivery(event, "tui-test")
    assert claim
    monkeypatch.setattr(
        ad,
        "complete_event_delivery",
        lambda *_args: (_ for _ in ()).throw(OSError("complete unavailable")),
    )
    release = ad.release_event_delivery
    releases = []

    def record_release(*args):
        releases.append(args)
        return release(*args)

    monkeypatch.setattr(ad, "release_event_delivery", record_release)

    assert registry_module.finish_completion_event_delivery(
        event, claim, "committed", registry=registry
    )
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "recovery_committed_ack"
    assert releases == []
    assert registry.completion_queue.empty()
    restored = queue.Queue()
    assert ad.restore_undelivered_completions(restored) == 0
    assert restored.empty()


@pytest.mark.parametrize("event_type", ["completion", "async_delegation"])
def test_dual_committed_ack_failure_is_nonterminal_and_never_replays(
    monkeypatch, tmp_path, event_type
):
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = registry_module.ProcessRegistry()
    if event_type == "async_delegation":
        event = {
            "type": event_type,
            "delegation_id": "deleg-dual-ack",
            "session_key": "active",
            "origin_ui_session_id": "active",
            "parent_session_id": "parent",
            "dispatched_at": 5.61,
            "completed_at": 5.62,
            "status": "completed",
            "summary": "done",
        }
        ad._persist_dispatch(event)
        ad._persist_completion(event, {"status": "completed", "summary": "done"})
    else:
        event = {
            "type": event_type,
            "session_id": "proc-dual-ack",
            "session_key": "active",
            "started_at": 5.61,
            "command": "true",
            "exit_code": 0,
            "completion_reason": "exited",
            "termination_source": "",
            "output": "done",
        }
        assert ad.persist_event_delivery(event)

    assert registry.claim_completion_delivery(event)
    claim = ad.claim_event_delivery(event, "tui-test")
    assert claim
    monkeypatch.setattr(ad, "complete_event_delivery", lambda *_args: False)
    monkeypatch.setattr(ad, "mark_completion_delivery_recovery", lambda *_args: False)

    assert not registry_module.finish_completion_event_delivery(
        event, claim, "committed", registry=registry
    )
    before = (
        ad.get_durable_delegation("deleg-dual-ack")
        if event_type == "async_delegation"
        else ad.get_durable_event_delivery(event)
    )
    assert before is not None
    assert before["delivery_state"] == "effect_started"

    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: False)
    restored = queue.Queue()
    assert ad.restore_undelivered_completions(restored) == 0
    assert restored.empty()
    after = (
        ad.get_durable_delegation("deleg-dual-ack")
        if event_type == "async_delegation"
        else ad.get_durable_event_delivery(event)
    )
    assert after is not None
    assert after["delivery_state"] == "recovery_effect_started"


@pytest.mark.parametrize("surface", ["busy", "idle"])
@pytest.mark.parametrize("storage_failure", ["claim", "release"])
def test_tui_storage_failure_retries_once_in_same_process(
    monkeypatch, tmp_path, surface, storage_failure
):
    """Busy-steer and idle-batch paths preserve claims across SQLite failures."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    event = {
        "type": "completion",
        "session_id": f"proc-tui-{surface}-{storage_failure}",
        "session_key": "active",
        "started_at": 5.7,
    }
    assert ad.persist_event_delivery(event)
    effects = []
    steer_calls = 0

    def steer(text):
        nonlocal steer_calls
        steer_calls += 1
        if storage_failure == "release" and steer_calls == 1:
            return False
        effects.append(text)
        return True

    session = _session(types.SimpleNamespace(steer=steer))
    submit_calls = 0

    def submit(_rid, _sid, _session, text, **kwargs):
        nonlocal submit_calls
        submit_calls += 1
        if storage_failure == "release" and submit_calls == 1:
            raise RuntimeError("submit failed")
        effects.append(text)
        kwargs["completion_delivery_callback"]("committed")

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    claim = ad.claim_event_delivery
    claim_calls = 0

    def claim_once(*args):
        nonlocal claim_calls
        claim_calls += 1
        if storage_failure == "claim" and claim_calls == 1:
            raise OSError("claim storage unavailable")
        return claim(*args)

    monkeypatch.setattr(ad, "claim_event_delivery", claim_once)
    release = ad.release_event_delivery
    release_calls = 0

    def release_once(*args):
        nonlocal release_calls
        release_calls += 1
        if storage_failure == "release" and release_calls == 1:
            raise OSError("release storage unavailable")
        return release(*args)

    monkeypatch.setattr(ad, "release_event_delivery", release_once)

    item = {"evt": event, "model_text": "completion", "text": "completion"}
    if surface == "busy":
        assert server._try_steer_busy_completion(
            session, event, "completion", "completion"
        ) is True
    else:
        server._dispatch_completion_batch("sid", session, [item], consumer="tui-test")

    if storage_failure == "claim":
        assert effects == ["completion"]
        assert registry.completion_queue.empty()
        if surface == "busy":
            pending = session["_completion_steer_pending"][-1]
            assert server._finish_completion_claim(
                pending["evt"], pending["claim"], "committed"
            )
    else:
        assert effects == []
        assert registry.completion_event_should_deliver(event)
        retry = registry.completion_queue.get_nowait()
        assert registry.completion_queue.empty()
        receipt = ad.get_durable_event_delivery(event)
        assert receipt is not None
        assert receipt["delivery_state"] == "effect_started"
        assert retry["_completion_delivery_retained_claim_id"] == receipt["delivery_claim"]

        session["running"] = True
        if surface == "busy":
            assert (
                server._try_steer_busy_completion(
                    session, retry, "completion", "completion"
                )
                is True
            )
            pending = session["_completion_steer_pending"][-1]
            assert server._finish_completion_claim(
                pending["evt"], pending["claim"], "committed"
            )
        else:
            server._dispatch_completion_batch(
                "sid", session, [{**item, "evt": retry}], consumer="tui-test"
            )

    assert effects == ["completion"]
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "delivered"
    assert registry.completion_queue.empty()
    assert not registry.completion_event_should_deliver(event)

    session["running"] = True
    if surface == "busy":
        assert server._try_steer_busy_completion(
            session, dict(event), "completion", "completion"
        ) is True
    else:
        server._dispatch_completion_batch(
            "sid", session, [dict(item)], consumer="tui-test"
        )
    assert effects == ["completion"]


def test_tui_settlement_exception_restores_retained_copy_under_capacity(
    monkeypatch, tmp_path
):
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "bounded-copy.db")
    registry = registry_module.ProcessRegistry()
    registry.completion_queue = queue.Queue(maxsize=1)
    monkeypatch.setattr(registry_module, "process_registry", registry)
    event = {
        "type": "completion",
        "session_id": "proc-tui-bounded-copy",
        "session_key": "active",
        "started_at": 5.71,
        "command": "true",
        "exit_code": 0,
        "completion_reason": "exited",
        "termination_source": "",
        "output": "done",
    }
    assert ad.persist_event_delivery(event)
    registry.completion_queue.put(event)
    assert registry.get_completion_for_owner(lambda _event: True, timeout=0.5) is event

    monkeypatch.setattr(
        server,
        "_finish_completion_claim",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("settlement failed")),
    )
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)

    def submit(_rid, _sid, _session, _text, **kwargs):
        kwargs["completion_delivery_callback"]("provider_failed")

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    server._dispatch_completion_batch(
        "sid",
        _session(types.SimpleNamespace()),
        [{"evt": event, "model_text": "completion", "text": "completion"}],
        consumer="tui-test",
    )

    retries = registry.drain_matching_completions(lambda _event: True)
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "effect_started"
    assert len(retries) == 1
    assert (
        retries[0]["_completion_delivery_retained_claim_id"]
        == receipt["delivery_claim"]
    )


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


def test_startup_restore_parks_effect_started_completion(tmp_path, monkeypatch):
    """An ambiguous effect boundary is parked instead of replaying provider work."""
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
        assert ad.restore_undelivered_completions(restored_events) == 0
        assert restored_events.empty()
        durable = ad.get_durable_delegation(event["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "recovery_effect_started"
        assert not ad.claim_completion_delivery(event["delegation_id"], "retry-claim")
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
