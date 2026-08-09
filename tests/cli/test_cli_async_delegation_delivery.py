"""Regression coverage for CLI async-delegation completion ownership."""

import queue
import types

import pytest

import cli as cli_module
from cli import HermesCLI


def test_cli_completion_drain_uses_visible_session_identity(monkeypatch):
    """A CLI window must not claim another window's restored completion."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()

    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_visible",
        "session_key": "visible-session",
    }
    calls = []

    class FakeRegistry:
        completion_queue = queue.Queue()

        def drain_notifications(self, *, session_key="", owns_event=None):
            calls.append((session_key, owns_event(event)))
            return [(event, "completion payload")]

        def claim_completion_delivery(self, _event):
            return True

        def release_completion_delivery(self, _event):
            pass

    claimed = []
    completed = []

    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        FakeRegistry(),
    )
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda evt, consumer: claimed.append((evt, consumer)) or "claim-token",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )

    cli._drain_process_notifications("cli-idle")

    assert calls == [("visible-session", True)]
    pending = cli._pending_input.get_nowait()
    assert pending == "completion payload"
    assert isinstance(pending, cli_module._CompletionDeliveryMessage)
    assert pending.event == event
    assert pending.claim_id == "claim-token"
    assert claimed == [(event, "cli-idle")]
    assert completed == []


@pytest.mark.parametrize("storage_failure", ["claim", "release"])
def test_cli_storage_failure_retries_once_in_same_process(
    monkeypatch, tmp_path, storage_failure
):
    """Classic CLI drain preserves an ordinary envelope across SQLite failures."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._session_db = None
    cli._pending_input = queue.Queue()
    event = {
        "type": "completion",
        "session_id": f"proc-cli-{storage_failure}",
        "session_key": "visible-session",
        "started_at": 8.0,
        "command": "true",
        "exit_code": 0,
        "completion_reason": "exited",
        "termination_source": "",
        "output": "done",
    }
    assert ad.persist_event_delivery(event)
    registry.completion_queue.put(event)
    effects = []
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

    cli._drain_process_notifications("cli-idle")
    if storage_failure == "release":
        first = cli._pending_input.get_nowait()
        monkeypatch.setattr(cli, "_chat", lambda *_args, **_kwargs: None)
        assert cli.chat(
            first,
            completion_delivery=True,
            completion_event=first.event,
            completion_claim=first.claim_id,
        ) is None
    assert cli._pending_input.empty()
    assert effects == []
    assert registry.completion_event_should_deliver(event)
    retry = registry.completion_queue.get_nowait()
    assert registry.completion_queue.empty()
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == (
        "pending" if storage_failure == "claim" else "effect_started"
    )
    if storage_failure == "release":
        assert retry["_completion_delivery_retained_claim_id"] == receipt["delivery_claim"]

    registry.completion_queue.put(retry)
    cli._drain_process_notifications("cli-idle")
    pending = cli._pending_input.get_nowait()

    def committed(message, **kwargs):
        effects.append(message)
        kwargs["completion_delivery_callback"]("committed")
        return "done"

    monkeypatch.setattr(cli, "_chat", committed)
    assert cli.chat(
        pending,
        completion_delivery=True,
        completion_event=pending.event,
        completion_claim=pending.claim_id,
    ) == "done"
    assert len(effects) == 1
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "delivered"
    assert registry.completion_queue.empty()
    assert not registry.completion_event_should_deliver(event)

    registry.completion_queue.put(dict(event))
    cli._drain_process_notifications("cli-idle")
    assert cli._pending_input.empty()
    assert len(effects) == 1


def test_cli_completion_ownership_rejects_foreign_session():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._session_db = None

    assert not cli._owns_process_notification(
        {"type": "async_delegation", "session_key": "foreign-session"}
    )


def test_cli_completion_ownership_accepts_compression_lineage():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"

    class FakeSessionDB:
        def resolve_resume_session_id(self, session_id):
            assert session_id == "pre-compression-session"
            return "visible-session"

    cli._session_db = FakeSessionDB()

    assert cli._owns_process_notification(
        {
            "type": "async_delegation",
            "session_key": "pre-compression-session",
        }
    )


def test_cli_visibility_suppression_records_durable_disposition(
    monkeypatch, tmp_path
):
    """A hidden CLI completion is terminally dispositioned, not merely drained."""
    from tools import async_delegation as ad
    import tools.process_registry as registry_module

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, "process_registry", registry)
    monkeypatch.setattr(
        registry_module, "completion_delivery_prompt", lambda _evt, _text: None
    )
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    cli._session_db = None
    event = {
        "type": "completion",
        "session_id": "proc-cli-suppressed",
        "session_key": "visible-session",
        "started_at": 7.0,
        "command": "true",
        "exit_code": 0,
        "completion_reason": "exited",
        "termination_source": "",
        "output": "done",
    }
    assert ad.persist_event_delivery(event)
    registry.completion_queue.put(event)

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.empty()
    assert registry.completion_queue.empty()
    receipt = ad.get_durable_event_delivery(event)
    assert receipt is not None
    assert receipt["delivery_state"] == "recovery_visibility_suppressed"


def test_cli_numeric_completion_queues_model_nudge_and_nonterminal_fails_open(
    monkeypatch,
):
    from tools.process_registry import ProcessRegistry

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    cli._session_db = None
    registry = ProcessRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda _event, _consumer: "claim-token",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery", lambda *_args: None
    )

    registry.completion_queue.put(
        {
            "type": "completion",
            "session_id": "proc-done",
            "session_key": "visible-session",
            "command": "echo done",
            "exit_code": 0,
            "output": "done",
        }
    )
    registry.completion_queue.put(
        {
            "type": "completion",
            "session_id": "proc-running",
            "session_key": "visible-session",
            "command": "sleep 1",
            "exit_code": None,
            "output": "still running",
        }
    )

    cli._drain_process_notifications("cli-idle")

    prompt = cli._pending_input.get_nowait()
    assert isinstance(prompt, cli_module._CompletionDeliveryMessage)
    assert "Background process proc-done completed normally" in prompt
    assert "must be literally empty (zero characters)" in prompt
    nonterminal = cli._pending_input.get_nowait()
    assert isinstance(nonterminal, cli_module._CompletionDeliveryMessage)
    assert "Background process proc-running exited (exit code None)" in nonterminal
    assert "Command: sleep 1" in nonterminal
    assert "Output:\nstill running" in nonterminal
    assert cli._pending_input.empty()


def test_cli_heartbeat_routes_only_to_owner_as_silent_warm_turn(monkeypatch):
    from tools.process_registry import ProcessRegistry

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    cli._session_db = None
    calls = []
    cli.agent = types.SimpleNamespace(
        run_conversation=lambda message, **kwargs: calls.append((message, kwargs))
    )

    class _ImmediateThread:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr("cli.threading.Thread", _ImmediateThread)
    registry = ProcessRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda _event, _consumer: "claim-token",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery", lambda *_args: None
    )
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        lambda _event: True,
    )
    event = {
        "type": "heartbeat",
        "target_id": "proc-heartbeat",
        "target_ids": ["proc-heartbeat"],
        "generations": [9],
        "generation": 9,
        "target_kind": "process",
        "session_id": "proc-heartbeat",
        "session_key": "visible-session",
        "provider": "openai",
        "cache_context": "openai-cache",
        "status": "ALIVE",
        "evidence": "output grew",
    }
    registry.completion_queue.put(event)

    cli._drain_process_notifications("cli-idle")

    assert calls == [("", {
        "turn_origin": "heartbeat_warm",
        "heartbeat_event": event,
    })]
    assert cli._pending_input.empty()


def test_cli_does_not_duplicate_runtime_owned_warm(monkeypatch):
    from tools.process_registry import ProcessRegistry

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    cli._session_db = None
    calls = []
    cli.agent = types.SimpleNamespace(
        run_conversation=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    class _ImmediateThread:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr("cli.threading.Thread", _ImmediateThread)
    registry = ProcessRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda _event, _consumer: "claim-token",
    )
    completed = []
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda *args: completed.append(args),
    )
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        lambda _event: True,
    )
    event = {
        "type": "heartbeat",
        "target_id": "proc-heartbeat",
        "session_key": "visible-session",
        "status": "ALIVE",
        "heartbeat_warm_owned": True,
    }
    registry.completion_queue.put(event)

    cli._drain_process_notifications("cli-idle")

    assert calls == []
    assert completed == [(event, "claim-token")]


def test_cli_unhealthy_heartbeat_is_printed_without_agent_turn(monkeypatch):
    from tools.process_registry import ProcessRegistry

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    cli._session_db = None
    registry = ProcessRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda _event, _consumer: "claim-token",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery", lambda *_args: None
    )
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        lambda _event: True,
    )
    printed = []
    monkeypatch.setattr("cli._cprint", lambda message, **_kwargs: printed.append(message))
    registry.completion_queue.put(
        {
            "type": "heartbeat",
            "target_id": "proc-heartbeat",
            "session_key": "visible-session",
            "status": "STUCK",
            "evidence": "no progress",
        }
    )

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.empty()
    assert len(printed) == 1
    assert "STUCK" in printed[0]
    assert "no progress" in printed[0]


def test_cli_stale_unhealthy_heartbeat_is_not_printed_or_run(monkeypatch):
    from tools.process_registry import ProcessRegistry

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    cli._session_db = None
    calls = []
    cli.agent = types.SimpleNamespace(
        run_conversation=lambda message, **kwargs: calls.append((message, kwargs))
    )
    registry = ProcessRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda _event, _consumer: "claim-token",
    )
    completed = []
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda *args: completed.append(args),
    )
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        lambda _event: False,
    )
    printed = []
    monkeypatch.setattr("cli._cprint", lambda message, **_kwargs: printed.append(message))
    event = {
        "type": "heartbeat",
        "target_id": "proc-heartbeat",
        "session_key": "visible-session",
        "status": "STUCK",
        "evidence": "old observation",
    }
    registry.completion_queue.put(event)

    cli._drain_process_notifications("cli-idle")

    assert printed == []
    assert calls == []
    assert cli._pending_input.empty()
    assert completed == [(event, "claim-token")]
