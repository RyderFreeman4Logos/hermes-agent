"""Regression coverage for CLI async-delegation completion ownership."""

import queue

from cli import HermesCLI, _CompletionDeliveryMessage, _HeartbeatWarmMessage


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
        def drain_notifications(self, *, session_key="", owns_event=None):
            calls.append((session_key, owns_event(event)))
            return [(event, "completion payload")]

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
    assert cli._pending_input.get_nowait() == "completion payload"
    assert claimed == [(event, "cli-idle")]
    assert completed == [(event, "claim-token")]


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


def test_cli_numeric_completion_queues_model_nudge_and_none_queues_nothing(
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
    assert isinstance(prompt, _CompletionDeliveryMessage)
    assert "Background process proc-done completed normally" in prompt
    assert "If no user-visible action is needed, emit no response." in prompt
    assert cli._pending_input.empty()


def test_cli_heartbeat_routes_only_to_owner_as_silent_warm_turn(monkeypatch):
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
            "type": "heartbeat",
            "target_id": "proc-heartbeat",
            "target_kind": "process",
            "session_id": "proc-heartbeat",
            "session_key": "visible-session",
            "status": "ALIVE",
            "evidence": "output grew",
        }
    )

    cli._drain_process_notifications("cli-idle")

    prompt = cli._pending_input.get_nowait()
    assert isinstance(prompt, _HeartbeatWarmMessage)
    assert "proc-heartbeat" in prompt
    assert cli._pending_input.empty()
