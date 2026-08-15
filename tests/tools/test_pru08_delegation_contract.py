import queue
import threading
from types import SimpleNamespace

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _isolated_async_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _dispatch(runner):
    return ad.dispatch_async_delegation(
        goal="capacity",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner",
        runner=runner,
        max_async_children=1,
    )


def test_capacity_retains_one_bounded_queued_delegation():
    release = threading.Event()
    running_started = threading.Event()
    queued_started = threading.Event()
    rejected_started = threading.Event()

    def running():
        running_started.set()
        assert release.wait(timeout=10)
        return {"status": "completed", "summary": "running"}

    first = _dispatch(running)
    assert running_started.wait(timeout=5)
    second = _dispatch(
        lambda: queued_started.set()
        or {"status": "completed", "summary": "queued"}
    )
    third = _dispatch(
        lambda: rejected_started.set()
        or {"status": "completed", "summary": "rejected"}
    )

    try:
        assert first["status"] == "dispatched"
        assert second["status"] == "dispatched"
        assert third["status"] == "rejected"
        assert not queued_started.is_set()
        assert not rejected_started.is_set()
        assert sorted(item["status"] for item in ad.list_async_delegations()) == [
            "queued",
            "running",
        ]
    finally:
        release.set()

    assert queued_started.wait(timeout=5)
    completed = {
        process_registry.completion_queue.get(timeout=5)["delegation_id"]
        for _ in range(2)
    }
    assert completed == {first["delegation_id"], second["delegation_id"]}
    with pytest.raises(queue.Empty):
        process_registry.completion_queue.get(timeout=0.1)


def test_workspace_declaration_wins_but_markdown_bait_does_not(tmp_path):
    from tools.delegate_tool import _resolve_workspace_hint

    parent = tmp_path / "parent"
    requested = tmp_path / "requested"
    bait = tmp_path / "bait"
    for path in (parent, requested, bait):
        path.mkdir()
    agent = SimpleNamespace(_current_task_id=None, cwd=str(parent))

    assert _resolve_workspace_hint(
        agent, f"workspace: {requested}", None
    ) == str(requested)
    assert _resolve_workspace_hint(
        agent,
        f"```text\nworkspace: {bait}\n```\n",
        None,
    ) == str(parent)
    assert _resolve_workspace_hint(
        agent,
        f"- workspace: {bait}\n",
        None,
    ) == str(parent)


def test_delegated_command_policy_rejects_bare_cargo_only_when_configured():
    from tools import terminal_tool

    task_id = "policy-child"
    terminal_tool.register_task_env_overrides(
        task_id,
        {"command_policy": {"just_only_cargo": True}},
    )
    try:
        blocked = terminal_tool._delegate_command_policy_violation(
            task_id, "cargo test"
        )
        allowed = terminal_tool._delegate_command_policy_violation(
            task_id, "just test"
        )
        ordinary = terminal_tool._delegate_command_policy_violation(
            "ordinary-task", "cargo test"
        )
    finally:
        terminal_tool.clear_task_env_overrides(task_id)

    assert blocked == {
        "error": (
            "Blocked by delegation.command_policy: bare Cargo is prohibited; "
            "use the repository Just recipe."
        ),
        "rule": "just_only_cargo",
        "policy_source": "delegation.command_policy",
    }
    assert allowed is None
    assert ordinary is None


def test_stall_grace_does_not_free_capacity_before_worker_returns():
    delegation_id = "delegate-stalled"
    with ad._records_lock:
        ad._records[delegation_id] = {
            "delegation_id": delegation_id,
            "status": "stalling",
            "dispatched_at": 1.0,
            "_stall_quiet_seconds": 20.0,
            "_stall_threshold_seconds": 10.0,
            "_stall_in_tool": True,
        }

    ad._finalize_stalled(delegation_id)
    assert ad.active_count() == 1
    assert ad.list_async_delegations()[0]["status"] == "stalling"

    ad._finalize(
        delegation_id,
        {"status": "completed", "summary": "late"},
        "completed",
    )
    evt = process_registry.completion_queue.get(timeout=1)
    assert evt["status"] == "stalled"
    assert evt["exit_reason"] == "stalled"
    assert ad.active_count() == 0
