"""Focused unit-2 coverage for retained child ownership and resume."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ad, "_current_owner_profile", lambda: "default", raising=False)
    ad._reset_for_tests()
    yield tmp_path
    ad._reset_for_tests()


def _insert_delegation(delegation_id="deleg-owner", parent_session_id="owner"):
    now = 1_700_000_000.0
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, origin_session_id)
               VALUES (?, '', '', ?, 'running', ?, ?, 'pending', 0, '')""",
            (delegation_id, parent_session_id, now, now),
        )


def _session_db(tmp_path):
    from hermes_state import SessionDB

    return SessionDB(Path(tmp_path) / "state.db")


def _seed_sessions(tmp_path):
    db = _session_db(tmp_path)
    db.create_session("owner", source="cli", profile_name="default")
    db.create_session("foreign", source="cli", profile_name="default")
    db.create_session(
        "child",
        source="delegate",
        parent_session_id="owner",
        model="test/model",
        model_config={"_delegate_from": "owner"},
        profile_name="default",
    )
    db.append_message("child", "user", "original goal")
    db.append_message("child", "assistant", "original answer")
    return db


def _dispatch_and_retain():
    _insert_delegation()
    ad.record_dispatched_children(
        "deleg-owner",
        [
            {
                "subagent_id": "sa-owner",
                "session_id": "child",
                "model": "test/model",
                "provider": "test-provider",
                "goal": "original goal",
            }
        ],
    )
    ad.retain_completed_delegation("deleg-owner")


def test_retained_child_registry_survives_restart_and_scopes_by_owner(tmp_path):
    db = _seed_sessions(tmp_path)
    db.close()
    _dispatch_and_retain()

    own = ad.list_retained_children(owner_session_id="owner")
    assert [entry["child_session_id"] for entry in own] == ["child"]
    assert ad.list_retained_children(owner_session_id="foreign") == []

    # The registry is durable, not just an in-memory active-child snapshot.
    ad._reset_for_tests()
    assert [entry["child_session_id"] for entry in ad.list_retained_children(owner_session_id="owner")] == [
        "child"
    ]


def test_compression_lineage_authority_allows_continuation_and_rejects_foreign(tmp_path):
    db = _session_db(tmp_path)
    db.create_session("root", source="cli", profile_name="default")
    db.end_session("root", "compression")
    db.create_session("continuation", source="cli", parent_session_id="root", profile_name="default")
    db.create_session("branch", source="cli", parent_session_id="root", profile_name="default", model_config={"_branched_from": "root"})
    db.create_session("delegate", source="delegate", parent_session_id="root", profile_name="default", model_config={"_delegate_from": "root"})
    db.create_session("stranger", source="cli", profile_name="default")
    db.close()

    assert ad._lineage_root("root") == "root"
    assert ad._lineage_root("continuation") == "root"
    assert ad._lineage_root("branch") is None
    assert ad._lineage_root("delegate") is None
    assert ad._lineage_root("stranger") == "stranger"


def test_retained_registry_is_not_reconstructed_from_a_foreign_handle(tmp_path):
    db = _seed_sessions(tmp_path)
    db.close()
    _dispatch_and_retain()

    # Missing and foreign handles must be indistinguishable to the caller.
    assert ad.find_retained_child("missing", owner_session_id="owner") is None
    assert ad.find_retained_child("sa-owner", owner_session_id="foreign") is None


def test_completed_event_retains_the_dispatched_child_manifest(tmp_path):
    db = _seed_sessions(tmp_path)
    db.close()
    _insert_delegation()
    ad.record_dispatched_children(
        "deleg-owner", [{"subagent_id": "sa-owner", "session_id": "child"}]
    )
    ad._persist_completion(
        {"delegation_id": "deleg-owner", "status": "completed"},
        {"status": "completed", "summary": "done"},
    )

    assert [entry["child_id"] for entry in ad.list_retained_children("owner")] == [
        "sa-owner"
    ]


def test_retained_child_follow_up_claim_is_single_use_until_released(tmp_path):
    db = _seed_sessions(tmp_path)
    db.close()
    _dispatch_and_retain()
    entry = ad.find_retained_child("sa-owner", owner_session_id="owner")
    assert entry is not None

    first = ad.claim_retained_child(entry)
    assert first
    assert ad.claim_retained_child(entry) is None
    ad.release_retained_child(entry, first)
    assert ad.claim_retained_child(entry)


def test_schema_upgrade_keeps_legacy_async_rows(tmp_path):
    path = tmp_path / "state.db"
    raw = sqlite3.connect(path)
    raw.execute(
        """CREATE TABLE async_delegations (
            delegation_id TEXT PRIMARY KEY,
            origin_session TEXT NOT NULL,
            origin_ui_session_id TEXT NOT NULL DEFAULT '',
            parent_session_id TEXT,
            state TEXT NOT NULL,
            dispatched_at REAL NOT NULL,
            completed_at REAL,
            updated_at REAL NOT NULL,
            event_json TEXT,
            result_json TEXT,
            delivery_state TEXT NOT NULL DEFAULT 'pending',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            delivered_at REAL
        )"""
    )
    raw.commit()
    raw.close()

    conn = ad._connect()
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(async_delegations)")}
    finally:
        conn.close()
    assert {"retained", "children_json", "owner_profile"} <= columns


def test_delegate_task_exposes_completed_child_follow_up():
    import inspect
    import tools.delegate_tool as dt

    assert "follow_up" in inspect.signature(dt.delegate_task).parameters


def test_completed_follow_up_resumes_the_child_session(tmp_path, monkeypatch):
    import tools.delegate_tool as dt

    db = _seed_sessions(tmp_path)
    db.close()
    _dispatch_and_retain()

    captured = {}

    def fake_build(**kwargs):
        captured["build"] = kwargs
        child = MagicMock()
        child.session_id = kwargs["resume_session_id"]
        return child

    def fake_run(task_index, goal, child, parent_agent, **kwargs):
        captured["goal"] = goal
        captured["history"] = getattr(child, "_delegate_resume_history", None)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "follow-up answer",
            "api_calls": 1,
            "duration_seconds": 0.1,
        }

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_build)
    monkeypatch.setattr(dt, "_run_single_child", fake_run)
    monkeypatch.setattr(dt, "_finalize_child_results", lambda *args, **kwargs: None)

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "owner"
    parent._session_db = _session_db(tmp_path)
    try:
        output = json.loads(
            dt.delegate_task(
                goal="one more thing",
                follow_up="sa-owner",
                parent_agent=parent,
            )
        )
    finally:
        parent._session_db.close()

    assert output["mode"] == "follow_up"
    assert output["results"][0]["summary"] == "follow-up answer"
    assert captured["build"]["resume_session_id"] == "child"
    assert [message["role"] for message in captured["history"]] == ["user", "assistant"]
    assert captured["goal"] == "one more thing"


def test_foreign_completed_follow_up_has_same_error_as_missing(tmp_path):
    import tools.delegate_tool as dt

    db = _seed_sessions(tmp_path)
    db.close()
    _dispatch_and_retain()

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "foreign"
    parent._session_db = _session_db(tmp_path)
    try:
        missing = json.loads(
            dt.delegate_task(
                goal="hello", follow_up="missing", parent_agent=parent
            )
        )
        foreign = json.loads(
            dt.delegate_task(
                goal="hello", follow_up="sa-owner", parent_agent=parent
            )
        )
    finally:
        parent._session_db.close()

    assert missing["error"] == foreign["error"]


def test_completed_follow_up_does_not_bypass_running_child_ownership(monkeypatch):
    import tools.delegate_tool as dt

    live = MagicMock()
    live.session_id = "live-child"
    dt._register_subagent(
        {
            "subagent_id": "sa-live",
            "agent": live,
            "accepting_steer": True,
        }
    )
    monkeypatch.setattr(
        dt,
        "steer_subagent",
        lambda *args, **kwargs: pytest.fail("follow-up must not steer running children"),
    )
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "owner"
    try:
        output = json.loads(
            dt.delegate_task(
                goal="hello", follow_up="sa-live", parent_agent=parent
            )
        )
    finally:
        dt._unregister_subagent("sa-live", agent=live)
    assert "error" in output
