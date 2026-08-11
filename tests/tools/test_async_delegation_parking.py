"""An undeliverable delegation completion must stop replaying on every boot.

Restart survival for background `delegate_task` works by persisting completions
and re-enqueuing the pending ones at registry startup
(`restore_undelivered_completions`). `_MAX_DELIVERY_ATTEMPTS` is meant to stop a
completion that can never be delivered from replaying forever — but the cap is
only consulted by `release_completion_delivery`, i.e. only on the path where a
consumer successfully *claimed* the row and then failed.

A completion whose origin session is permanently gone (dead owner pid) never
gets that far: the consumer's ownership gate drops it BEFORE claiming, so
`delivery_attempts` never advances and `delivery_state` stays `'pending'`.
`restore_undelivered_completions` re-enqueues it on the next boot, it is dropped
again, and the cycle repeats for the life of the install — re-logging a WARNING
each time.
"""

from contextlib import contextmanager
import json
import time

import pytest


@pytest.fixture
def delegation_db(tmp_path, monkeypatch):
    """Point the durable delegation store at a temp HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    return ad


def _insert_pending(
    ad, delegation_id: str, *, attempts: int, event_json: str | None = None
) -> None:
    """Insert a durable completion in the pending, undelivered state."""
    now = time.time()
    event = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "telegram:dead-session",
        "origin_ui_session_id": "",
        "origin_session_id": "",
        "parent_session_id": None,
        "goal": "a task whose origin session is gone",
        "status": "success",
        "summary": "done",
    }
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id, state,
                dispatched_at, completed_at, updated_at, event_json,
                delivery_state, delivery_attempts)
               VALUES (?, ?, '', 'success', ?, ?, ?, ?, 'pending', ?)""",
            (
                delegation_id,
                "telegram:dead-session",
                now,
                now,
                now,
                event_json if event_json is not None else json.dumps(event),
                attempts,
            ),
        )


def _delivery_row(ad, delegation_id: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            "SELECT delivery_state, delivery_attempts FROM async_delegations "
            "WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()


def _insert_ordinary(ad, delivery_id: str, event_json: str, updated_at: float) -> None:
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            """INSERT INTO ordinary_completion_deliveries
               (delivery_id, event_json, delivery_state, updated_at)
               VALUES (?, ?, 'pending', ?)""",
            (delivery_id, event_json, updated_at),
        )


def _ordinary_state(ad, delivery_id: str) -> str:
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            "SELECT delivery_state FROM ordinary_completion_deliveries "
            "WHERE delivery_id=?",
            (delivery_id,),
        ).fetchone()[0]


def _stored_event_json(ad, table: str, id_column: str, row_id: str) -> str:
    assert table in {"async_delegations", "ordinary_completion_deliveries"}
    assert id_column in {"delegation_id", "delivery_id"}
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            f"SELECT event_json FROM {table} WHERE {id_column}=?", (row_id,)
        ).fetchone()[0]


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_exhausted_completion_is_parked_instead_of_replayed(delegation_db):
    """A row past the attempt budget must not be re-enqueued again."""
    ad = delegation_db
    _insert_pending(ad, "d-exhausted", attempts=ad._MAX_DELIVERY_ATTEMPTS)

    queue = _Queue()
    restored = ad.restore_undelivered_completions(queue)

    assert queue.items == [], "an exhausted completion was replayed to the queue"
    assert restored == 0, "an exhausted completion was counted as restored"


def test_exhausted_completion_reaches_a_terminal_state(delegation_db):
    """Parking must be durable, or the next boot repeats the same work."""
    ad = delegation_db
    _insert_pending(ad, "d-exhausted", attempts=ad._MAX_DELIVERY_ATTEMPTS)

    ad.restore_undelivered_completions(_Queue())

    state, _attempts = _delivery_row(ad, "d-exhausted")
    assert state != "pending", (
        "row left 'pending' — it will be re-enqueued and re-dropped every boot"
    )
    # 'parked' (not 'delivered'): the result was never handed to anyone, so
    # claiming delivery would be dishonest. The row stays queryable for audit.
    assert state == "parked"


def test_parking_is_idempotent_across_repeated_restarts(delegation_db):
    """The second boot must be a no-op, proving the cycle is actually broken."""
    ad = delegation_db
    _insert_pending(ad, "d-exhausted", attempts=ad._MAX_DELIVERY_ATTEMPTS)

    first = _Queue()
    ad.restore_undelivered_completions(first)
    second = _Queue()
    ad.restore_undelivered_completions(second)

    assert first.items == []
    assert second.items == []


def test_healthy_completion_is_still_restored(delegation_db):
    """Control: restart survival itself must not regress.

    A completion below the budget is exactly what the persist + auto-redispatch
    machinery exists to recover, so it must still be re-enqueued and stamped
    ``restored=True``.
    """
    ad = delegation_db
    _insert_pending(ad, "d-healthy", attempts=0)

    queue = _Queue()
    restored = ad.restore_undelivered_completions(queue)

    assert restored == 1
    assert len(queue.items) == 1
    assert queue.items[0]["delegation_id"] == "d-healthy"
    assert queue.items[0]["restored"] is True
    state, _ = _delivery_row(ad, "d-healthy")
    assert state == "pending"


def test_boundary_one_attempt_below_budget_is_still_restored(delegation_db):
    """Off-by-one guard: the cap must not retire a row that has budget left."""
    ad = delegation_db
    _insert_pending(ad, "d-nearly", attempts=ad._MAX_DELIVERY_ATTEMPTS - 1)

    queue = _Queue()
    restored = ad.restore_undelivered_completions(queue)

    assert restored == 1
    assert queue.items[0]["delegation_id"] == "d-nearly"


def test_mixed_batch_parks_only_the_exhausted_row(delegation_db):
    """One poisoned row must not suppress recovery of healthy siblings."""
    ad = delegation_db
    _insert_pending(ad, "d-exhausted", attempts=ad._MAX_DELIVERY_ATTEMPTS)
    _insert_pending(ad, "d-healthy", attempts=1)

    queue = _Queue()
    restored = ad.restore_undelivered_completions(queue)

    assert restored == 1
    assert [item["delegation_id"] for item in queue.items] == ["d-healthy"]
    assert _delivery_row(ad, "d-exhausted")[0] == "parked"
    assert _delivery_row(ad, "d-healthy")[0] == "pending"


@pytest.mark.parametrize(
    "bad_event_json",
    [
        '{"_completion_delivery_token":"private-async-token"',
        "[]",
        json.dumps({"type": "completion"}),
    ],
    ids=["malformed-json", "wrong-shape", "wrong-type"],
)
def test_corrupt_async_row_is_parked_without_blocking_valid_sibling(
    delegation_db, bad_event_json, caplog
):
    ad = delegation_db
    _insert_pending(ad, "d-corrupt", attempts=0, event_json=bad_event_json)
    _insert_pending(ad, "d-valid", attempts=0)

    queue = _Queue()
    assert ad.restore_undelivered_completions(queue) == 1
    assert [item["delegation_id"] for item in queue.items] == ["d-valid"]
    assert _delivery_row(ad, "d-corrupt")[0] == "parked_corrupt"
    parked_payload = _stored_event_json(
        ad, "async_delegations", "delegation_id", "d-corrupt"
    )
    assert json.loads(parked_payload)["corrupt_delivery_row"] == "d-corrupt"
    assert "_completion_delivery_" not in parked_payload

    caplog.clear()
    ad.restore_undelivered_completions(_Queue())
    assert not [
        record
        for record in caplog.records
        if "corrupt async completion row d-corrupt" in record.getMessage()
    ]


@pytest.mark.parametrize(
    "bad_event_json",
    [
        '{"_completion_delivery_binding":"private-ordinary-binding"',
        "[]",
        json.dumps({"type": "async_delegation"}),
    ],
    ids=["malformed-json", "wrong-shape", "wrong-type"],
)
def test_corrupt_ordinary_row_is_local_and_enqueue_follows_commit(
    delegation_db, monkeypatch, bad_event_json
):
    ad = delegation_db
    corrupt_id = "ordinary-corrupt"
    events = [
        {
            "type": "completion",
            "session_id": f"proc-valid-{index}",
            "session_key": "owner",
            "started_at": float(index),
            "command": f"command-{index}",
            "exit_code": 0,
            "completion_reason": "exited",
            "termination_source": "",
        }
        for index in (1, 2)
    ]
    _insert_ordinary(ad, corrupt_id, bad_event_json, 0.0)
    for event in events:
        delivery_id = ad._ordinary_completion_delivery_id(event)
        assert delivery_id is not None
        _insert_ordinary(ad, delivery_id, json.dumps(event), event["started_at"])

    real_transaction = ad._transaction
    transaction_active = False

    @contextmanager
    def tracked_transaction():
        nonlocal transaction_active
        with real_transaction() as conn:
            transaction_active = True
            try:
                yield conn
            finally:
                transaction_active = False

    monkeypatch.setattr(ad, "_transaction", tracked_transaction)

    class CommitAwareQueue(_Queue):
        def put(self, item):
            assert not transaction_active, "event escaped before SQLite commit"
            super().put(item)

    queue = CommitAwareQueue()
    assert ad.restore_undelivered_completions(queue) == 2
    assert [item["session_id"] for item in queue.items] == [
        "proc-valid-1",
        "proc-valid-2",
    ]
    assert _ordinary_state(ad, corrupt_id) == "parked_corrupt"
    parked_payload = _stored_event_json(
        ad,
        "ordinary_completion_deliveries",
        "delivery_id",
        corrupt_id,
    )
    assert json.loads(parked_payload)["corrupt_delivery_row"] == corrupt_id
    assert "_completion_delivery_" not in parked_payload


def test_ownership_drop_advances_the_delivery_attempt_counter(delegation_db):
    """The counter must advance on the pre-claim drop path too.

    `claim_completion_delivery` is the only site that bumps `delivery_attempts`,
    but an unownable completion is dropped BEFORE it is claimed. Without a
    counter bump on that path the budget is never consumed and the parking
    threshold above is unreachable — the fix would be dead code.
    """
    ad = delegation_db
    _insert_pending(ad, "d-unowned", attempts=0)

    ad._note_delivery_attempt("d-unowned")

    _state, attempts = _delivery_row(ad, "d-unowned")
    assert attempts == 1


def test_repeated_ownership_drops_eventually_reach_the_parking_threshold(
    delegation_db,
):
    """End-to-end: the drop path alone must be able to retire a poisoned row."""
    ad = delegation_db
    _insert_pending(ad, "d-unowned", attempts=0)

    for _ in range(ad._MAX_DELIVERY_ATTEMPTS):
        ad._note_delivery_attempt("d-unowned")

    queue = _Queue()
    ad.restore_undelivered_completions(queue)

    assert queue.items == []
    assert _delivery_row(ad, "d-unowned")[0] == "parked"


def _ordinary_event(name: str, started_at: float) -> dict:
    return {
        "type": "completion",
        "session_id": f"proc-{name}",
        "session_key": "owner",
        "started_at": started_at,
        "command": f"command-{name}",
        "exit_code": 1,
        "completion_reason": "exited",
        "termination_source": "",
        "output": name,
    }


def test_wrong_ordinary_row_identity_is_parked_without_rekeying(delegation_db):
    ad = delegation_db
    event = _ordinary_event("wrong-row", 11.0)
    normalized_id = ad._ordinary_completion_delivery_id(event)
    assert normalized_id
    _insert_ordinary(ad, "wrong-row-id", json.dumps(event), 1.0)

    queue = _Queue()
    assert ad.restore_undelivered_completions(queue) == 0

    assert queue.items == []
    assert _ordinary_state(ad, "wrong-row-id") == "parked_corrupt"
    with ad._DB_LOCK, ad._transaction() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM ordinary_completion_deliveries WHERE delivery_id=?",
                (normalized_id,),
            ).fetchone()
            is None
        )


def test_swapped_ordinary_identities_preserve_two_once_only_effects(delegation_db):
    ad = delegation_db
    events = [
        _ordinary_event("swapped-a", 12.0),
        _ordinary_event("swapped-b", 13.0),
    ]
    ids = [ad._ordinary_completion_delivery_id(event) for event in events]
    assert all(ids) and ids[0] != ids[1]
    _insert_ordinary(ad, ids[0], json.dumps(events[1]), 1.0)
    _insert_ordinary(ad, ids[1], json.dumps(events[0]), 2.0)

    queue = _Queue()
    assert ad.restore_undelivered_completions(queue) == 2
    effects = []
    for event in queue.items:
        claim = ad.claim_event_delivery(event, "swapped-test")
        assert claim
        effects.append(event["session_id"])
        assert ad.complete_event_delivery(event, claim)

    assert sorted(effects) == sorted(event["session_id"] for event in events)
    restarted = _Queue()
    assert ad.restore_undelivered_completions(restarted) == 0
    assert restarted.items == []


def test_wrong_type_scalar_fields_park_only_their_rows(delegation_db):
    ad = delegation_db
    async_events = [
        {
            "type": "async_delegation",
            "delegation_id": f"deleg-bad-{name}",
            "session_key": "owner",
            "status": "completed",
            "summary": name,
        }
        for name in ("attempts", "owner-pid", "owner-start")
    ]
    ordinary_events = [
        _ordinary_event("bad-attempts", 13.5),
        _ordinary_event("bad-owner-pid", 14.0),
        _ordinary_event("bad-owner-start", 15.0),
    ]
    ordinary_ids = [
        ad._ordinary_completion_delivery_id(event) for event in ordinary_events
    ]
    valid = _ordinary_event("valid-scalar-neighbor", 16.0)
    valid_id = ad._ordinary_completion_delivery_id(valid)
    assert all(ordinary_ids) and valid_id
    with ad._connect() as conn:
        for event, attempts, owner_pid, owner_started_at in (
            (async_events[0], "not-an-integer", None, None),
            (async_events[1], 0, "not-a-pid", 1),
            (async_events[2], 0, 999999999, "not-a-start-time"),
        ):
            conn.execute(
                """INSERT INTO async_delegations
                   (delegation_id, origin_session, origin_ui_session_id, state,
                    dispatched_at, completed_at, updated_at, event_json,
                    delivery_state, delivery_attempts, delivery_claim,
                    owner_pid, owner_started_at)
                   VALUES (?, 'owner', '', 'completed', 1, 2, 2, ?, ?, ?,
                           'claim', ?, ?)""",
                (
                    event["delegation_id"],
                    json.dumps(event),
                    "pending" if owner_pid is None else "effect_started",
                    attempts,
                    owner_pid,
                    owner_started_at,
                ),
            )
        conn.execute(
            """INSERT INTO ordinary_completion_deliveries
               (delivery_id, event_json, delivery_state, updated_at,
                delivery_attempts)
               VALUES (?, ?, 'pending', 2, ?)""",
            (ordinary_ids[0], json.dumps(ordinary_events[0]), "not-an-integer"),
        )
        for event, delivery_id, owner_pid, owner_started_at in (
            (ordinary_events[1], ordinary_ids[1], "not-a-pid", 1),
            (ordinary_events[2], ordinary_ids[2], 999999999, "not-a-start-time"),
        ):
            conn.execute(
                """INSERT INTO ordinary_completion_deliveries
                   (delivery_id, event_json, delivery_state, updated_at,
                    delivery_claim, delivery_owner_pid,
                    delivery_owner_started_at)
                   VALUES (?, ?, 'effect_started', 2, 'claim', ?, ?)""",
                (delivery_id, json.dumps(event), owner_pid, owner_started_at),
            )
        conn.execute(
            """INSERT INTO ordinary_completion_deliveries
               (delivery_id, event_json, delivery_state, updated_at)
               VALUES (?, ?, 'pending', 3)""",
            (valid_id, json.dumps(valid)),
        )

    queue = _Queue()
    assert ad.restore_undelivered_completions(queue) == 1
    assert [event["session_id"] for event in queue.items] == [valid["session_id"]]
    with ad._connect() as conn:
        assert {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT delegation_id, delivery_state FROM async_delegations"
            )
        } == {event["delegation_id"]: "parked_corrupt" for event in async_events}
        assert {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT delivery_id, delivery_state "
                "FROM ordinary_completion_deliveries WHERE delivery_id != ?",
                (valid_id,),
            )
        } == {delivery_id: "parked_corrupt" for delivery_id in ordinary_ids}


def test_malformed_abandoned_task_is_parked_without_blocking_neighbor(delegation_db):
    ad = delegation_db
    valid = _ordinary_event("valid-task-neighbor", 17.0)
    valid_id = ad._ordinary_completion_delivery_id(valid)
    assert valid_id
    with ad._connect() as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id, state,
                dispatched_at, updated_at, task_json, delivery_state,
                owner_pid)
               VALUES ('deleg-bad-task', 'owner', '', 'running', 1, 1,
                       '{not-json', 'pending', 999999999)"""
        )
        conn.execute(
            """INSERT INTO ordinary_completion_deliveries
               (delivery_id, event_json, delivery_state, updated_at)
               VALUES (?, ?, 'pending', 2)""",
            (valid_id, json.dumps(valid)),
        )

    queue = _Queue()
    assert ad.restore_undelivered_completions(queue) == 1
    assert [event["session_id"] for event in queue.items] == [valid["session_id"]]
    with ad._connect() as conn:
        state, payload = conn.execute(
            "SELECT delivery_state, event_json FROM async_delegations "
            "WHERE delegation_id='deleg-bad-task'"
        ).fetchone()
    assert state == "parked_corrupt"
    assert json.loads(payload)["corrupt_delivery_row"] == "deleg-bad-task"
