"""Regression coverage for CLI async-delegation completion ownership."""

import queue
import time
from unittest.mock import MagicMock

import pytest

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
        def drain_notifications(self, *, session_key="", owns_event=None):
            calls.append((session_key, owns_event(event)))
            return [(event, "completion payload")]

    claimed = []
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
        lambda *_args: pytest.fail("queued input completed delivery"),
    )
    monkeypatch.setattr(
        "tools.async_delegation.start_event_delivery_renewal",
        lambda _claims: __import__("threading").Event(),
    )

    cli._drain_process_notifications("cli-idle")

    assert calls == [("visible-session", True)]
    pending = cli._pending_input.get_nowait()
    assert pending.text == "completion payload"
    assert pending.event is event
    assert pending.claim == "claim-token"
    assert pending.renewal_stop is not None
    assert claimed == [(event, "cli-idle")]
    pending.renewal_stop.set()


def test_cli_claim_renews_while_queued_past_lease(tmp_path, monkeypatch):
    from tools import async_delegation as ad

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ad, "_DELIVERY_CLAIM_LEASE_SECONDS", 0.05)
    monkeypatch.setattr(ad, "_DELIVERY_CLAIM_RENEW_INTERVAL_SECONDS", 0.01)
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg-cli-backlog",
        "session_key": "visible-session",
        "status": "completed",
        "summary": "done",
        "dispatched_at": 1000.0,
        "completed_at": 1001.0,
    }
    ad._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": event["session_key"],
        "origin_ui_session_id": "",
        "parent_session_id": None,
        "dispatched_at": event["dispatched_at"],
    })
    ad._persist_completion(event, {"status": "completed", "summary": "done"})

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [(event, "completion payload")]

    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    cli._drain_process_notifications("cli-idle")
    pending = cli._pending_input.get_nowait()

    time.sleep(0.12)
    assert ad.claim_event_delivery(event, "foreign-after-lease") is None

    pending.renewal_stop.set()
    cli._settle_delivery_claims([(pending.event, pending.claim)])
    retry = ad.claim_event_delivery(event, "retry")
    assert retry
    ad.release_event_delivery(event, retry)


def test_cli_enqueue_failure_stops_renewal_and_releases(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"

    class BrokenQueue:
        def put(self, _item):
            raise RuntimeError("queue closed")

    event = {"type": "completion", "session_id": "proc-cli-enqueue"}

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [(event, "completion payload")]

    renewal_stop = MagicMock()
    settled = []
    cli._pending_input = BrokenQueue()
    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery", lambda *_args: "claim"
    )
    monkeypatch.setattr(
        "tools.async_delegation.start_event_delivery_renewal",
        lambda _claims: renewal_stop,
    )
    monkeypatch.setattr(
        cli,
        "_settle_delivery_claims",
        lambda claims, result=None: settled.append((list(claims), result)),
    )

    with pytest.raises(RuntimeError, match="queue closed"):
        cli._drain_process_notifications("cli-idle")

    renewal_stop.set.assert_called_once_with()
    assert settled == [([(event, "claim")], None)]


def test_cli_renewal_start_failure_releases_claim(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = MagicMock()
    event = {"type": "completion", "session_id": "proc-cli-schedule"}

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [(event, "completion payload")]

    settled = []
    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery", lambda *_args: "claim"
    )
    monkeypatch.setattr(
        "tools.async_delegation.start_event_delivery_renewal",
        lambda _claims: (_ for _ in ()).throw(RuntimeError("thread unavailable")),
    )
    monkeypatch.setattr(
        cli,
        "_settle_delivery_claims",
        lambda claims, result=None: settled.append((list(claims), result)),
    )

    with pytest.raises(RuntimeError, match="thread unavailable"):
        cli._drain_process_notifications("cli-idle")

    cli._pending_input.put.assert_not_called()
    assert settled == [([(event, "claim")], None)]


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
