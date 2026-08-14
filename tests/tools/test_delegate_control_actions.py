"""delegate_task(action=...) — model-facing live orchestration of subagents.

Covers the control plane added to delegate_task: action='list' /
'steer' / 'stop' resolve against the module-level _active_subagents
registry, scoped by the _delegate_parent_ref ownership chain so a
conversation can only control its own spawn tree. Also pins the two
integration contracts: control actions are synchronous (never
backgrounded) and never consume the per-turn subagent spawn cap.
"""

import json
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

from tools.delegate_tool import (
    _handle_control_action,
    _is_descendant_of,
    _run_pending_child_controls,
    _register_subagent,
    _unregister_subagent,
    delegate_task,
    interrupt_subagent_with_replacement,
    queue_subagent,
)


class _StubChild:
    """Weakref-able stand-in for a live child AIAgent."""

    def __init__(self, parent=None, accept_steer: bool = True):
        self.steered: list[str] = []
        self.accept_steer = accept_steer
        self._live_transcript_path = "/tmp/live/task-0.log"
        if parent is not None:
            self._delegate_parent_ref = weakref.ref(parent)

    def steer(self, text: str) -> bool:
        if not self.accept_steer:
            return False
        self.steered.append(text)
        return True


class _StubParent:
    pass


def _register(sid: str, child, **extra) -> None:
    record = {
        "subagent_id": sid,
        "parent_id": None,
        "depth": 0,
        "goal": "test goal",
        "model": "test-model",
        "started_at": 1000.0,
        "status": "running",
        "tool_count": 0,
        "agent": child,
    }
    record.update(extra)
    _register_subagent(record)


# ---------------------------------------------------------------------------
# Ownership chain
# ---------------------------------------------------------------------------


def test_direct_child_is_descendant():
    parent = _StubParent()
    child = _StubChild(parent)
    assert _is_descendant_of(child, parent) is True


def test_grandchild_is_descendant():
    parent = _StubParent()
    mid = _StubChild(parent)
    grandchild = _StubChild(mid)
    assert _is_descendant_of(grandchild, parent) is True


def test_foreign_agent_is_not_descendant():
    parent = _StubParent()
    other_parent = _StubParent()
    foreign = _StubChild(other_parent)
    assert _is_descendant_of(foreign, parent) is False


def test_missing_ref_is_not_descendant():
    parent = _StubParent()
    orphan = _StubChild()  # no parent ref
    assert _is_descendant_of(orphan, parent) is False
    assert _is_descendant_of(None, parent) is False


def test_dead_parent_ref_is_not_descendant():
    parent = _StubParent()
    child = _StubChild(parent)
    del parent
    import gc

    gc.collect()
    assert _is_descendant_of(child, _StubParent()) is False


# ---------------------------------------------------------------------------
# action='list'
# ---------------------------------------------------------------------------


def test_list_shows_only_own_children():
    parent = _StubParent()
    mine = _StubChild(parent)
    foreign = _StubChild(_StubParent())
    _register("sid-ctl-list-1", mine)
    _register("sid-ctl-list-2", foreign)
    try:
        out = json.loads(_handle_control_action("list", None, None, parent))
        assert out["count"] == 1
        entry = out["subagents"][0]
        assert entry["subagent_id"] == "sid-ctl-list-1"
        assert entry["goal"] == "test goal"
        assert entry["accepting_steer"] is True
        assert entry["live_transcript"] == "/tmp/live/task-0.log"
        # Internal fields must not leak
        assert "agent" not in entry
        assert "owner_transport" not in entry
    finally:
        _unregister_subagent("sid-ctl-list-1")
        _unregister_subagent("sid-ctl-list-2")


def test_list_empty_registry_has_note():
    out = json.loads(_handle_control_action("list", None, None, _StubParent()))
    assert out["count"] == 0
    assert "note" in out


# ---------------------------------------------------------------------------
# action='steer'
# ---------------------------------------------------------------------------


def test_steer_reaches_owned_child():
    parent = _StubParent()
    child = _StubChild(parent)
    _register("sid-ctl-steer-1", child)
    try:
        out = json.loads(
            _handle_control_action("steer", "sid-ctl-steer-1", "focus on X", parent)
        )
        assert out["status"] == "queued"
        assert child.steered == ["focus on X"]
    finally:
        _unregister_subagent("sid-ctl-steer-1")


def test_steer_foreign_child_is_refused():
    parent = _StubParent()
    foreign = _StubChild(_StubParent())
    _register("sid-ctl-steer-2", foreign)
    try:
        out = _handle_control_action("steer", "sid-ctl-steer-2", "hijack", parent)
        assert "No live subagent" in out
        assert foreign.steered == []
    finally:
        _unregister_subagent("sid-ctl-steer-2")


def test_steer_requires_message():
    parent = _StubParent()
    child = _StubChild(parent)
    _register("sid-ctl-steer-3", child)
    try:
        out = _handle_control_action("steer", "sid-ctl-steer-3", "   ", parent)
        assert "requires a non-empty 'message'" in out
    finally:
        _unregister_subagent("sid-ctl-steer-3")


def test_steer_requires_subagent_id():
    out = _handle_control_action("steer", "", "text", _StubParent())
    assert "requires subagent_id" in out


def test_steer_closed_acceptance_is_refused():
    parent = _StubParent()
    child = _StubChild(parent)
    _register("sid-ctl-steer-4", child, accepting_steer=False)
    try:
        out = _handle_control_action("steer", "sid-ctl-steer-4", "late", parent)
        assert "no longer accepting" in out
        assert child.steered == []
    finally:
        _unregister_subagent("sid-ctl-steer-4")


# ---------------------------------------------------------------------------
# action='stop'
# ---------------------------------------------------------------------------


def test_stop_interrupts_owned_child(monkeypatch):
    import tools.delegate_tool as dt

    parent = _StubParent()
    child = _StubChild(parent)
    _register("sid-ctl-stop-1", child)
    interrupted = []
    monkeypatch.setattr(
        dt, "request_hard_interrupt", lambda agent, reason: interrupted.append(agent) or True
    )
    try:
        out = json.loads(
            _handle_control_action("stop", "sid-ctl-stop-1", None, parent)
        )
        assert out["status"] == "interrupt_requested"
        assert interrupted == [child]
    finally:
        _unregister_subagent("sid-ctl-stop-1")


def test_stop_foreign_child_is_refused(monkeypatch):
    import tools.delegate_tool as dt

    parent = _StubParent()
    foreign = _StubChild(_StubParent())
    _register("sid-ctl-stop-2", foreign)
    interrupted = []
    monkeypatch.setattr(
        dt, "request_hard_interrupt", lambda agent, reason: interrupted.append(agent) or True
    )
    try:
        out = _handle_control_action("stop", "sid-ctl-stop-2", None, parent)
        assert "No live subagent" in out
        assert interrupted == []
    finally:
        _unregister_subagent("sid-ctl-stop-2")


def test_stop_unknown_id_mentions_completion_path():
    out = _handle_control_action("stop", "sid-gone", None, _StubParent())
    assert "No live subagent" in out
    assert "completion message" in out


# ---------------------------------------------------------------------------
# delegate_task() entrypoint routing
# ---------------------------------------------------------------------------


def test_delegate_task_routes_control_action_before_spawn_machinery():
    """action='list' must return synchronously without touching spawn paths
    (no goal/tasks required, no pause gate, no depth checks)."""
    parent = _StubParent()
    out = json.loads(delegate_task(action="list", parent_agent=parent))
    assert out["action"] == "list"


def test_delegate_task_control_action_bypasses_spawn_pause():
    from tools.delegate_tool import set_spawn_paused

    parent = _StubParent()
    set_spawn_paused(True)
    try:
        out = json.loads(delegate_task(action="list", parent_agent=parent))
        assert out["action"] == "list"
    finally:
        set_spawn_paused(False)


def test_delegate_task_unknown_action_is_an_error():
    out = delegate_task(action="pause", goal="g", parent_agent=_StubParent())
    assert "Unknown action" in out


def test_delegate_task_spawn_action_still_validates_goal():
    out = delegate_task(action="spawn", parent_agent=_StubParent())
    assert "Provide either 'goal'" in out


def test_delegate_task_requires_parent_agent_for_control():
    out = delegate_task(action="list", parent_agent=None)
    assert "requires a parent agent" in out


def test_empty_tasks_array_with_goal_is_single_task_not_batch_error():
    """Small models emit tasks=[] alongside goal; that must not trip the
    'Batch mode requires at least 2 tasks' gate (observed live with
    gpt-5.4-mini on Nous Portal)."""
    out = delegate_task(tasks=[], goal="", parent_agent=_StubParent())
    # Falls through to the single-goal validation, not the batch gate.
    assert "Provide either 'goal'" in out
    assert "at least 2 tasks" not in out


# ---------------------------------------------------------------------------
# Guardrail: control actions never consume the spawn cap
# ---------------------------------------------------------------------------


def test_spawn_count_zero_for_control_actions():
    from agent.tool_guardrails import _subagent_spawn_count

    assert _subagent_spawn_count({"action": "list"}) == 0
    assert _subagent_spawn_count({"action": "steer", "subagent_id": "x"}) == 0
    assert _subagent_spawn_count({"action": "stop", "subagent_id": "x"}) == 0
    # Spawn shapes unchanged
    assert _subagent_spawn_count({"goal": "g"}) == 1
    assert _subagent_spawn_count({"action": "spawn", "goal": "g"}) == 1
    assert _subagent_spawn_count({"tasks": [{"goal": "a"}, {"goal": "b"}]}) == 2


def test_control_action_not_blocked_at_spawn_cap():
    """Once the cap is hit, steer/stop must STILL work — that's when the
    user most needs to rein children in."""
    from agent.tool_guardrails import (
        LoopCapConfig,
        ToolCallGuardrailConfig,
        ToolCallGuardrailController,
    )

    cfg = ToolCallGuardrailConfig(loop_caps=LoopCapConfig(max_subagents=1))
    ctl = ToolCallGuardrailController(cfg)
    # Exhaust the cap with a spawn
    assert ctl.before_call("delegate_task", {"goal": "a"}).action == "allow"
    # A second spawn is blocked
    assert ctl.before_call("delegate_task", {"goal": "b"}).action == "block"
    # Control actions still pass on a fresh controller after cap exhaustion
    ctl2 = ToolCallGuardrailController(cfg)
    assert ctl2.before_call("delegate_task", {"goal": "a"}).action == "allow"
    assert (
        ctl2.before_call(
            "delegate_task", {"action": "stop", "subagent_id": "x"}
        ).action
        == "allow"
    )
    assert (
        ctl2.before_call("delegate_task", {"action": "list"}).action == "allow"
    )
    # And spawns remain blocked afterwards — the control call didn't reset it
    assert ctl2.before_call("delegate_task", {"goal": "c"}).action == "block"


# ---------------------------------------------------------------------------
# Running-child queue and interrupt-with-replacement
# ---------------------------------------------------------------------------


def test_queue_and_interrupt_are_serialized_per_child(monkeypatch):
    """Only one mutually-exclusive delivery mode can reserve a child."""
    import tools.async_delegation as ad
    import tools.delegate_tool as dt

    parent = _StubParent()
    child = _StubChild(parent)
    sid = "sid-ctl-delivery-race"
    _register(sid, child, delegation_id="deleg-race")
    monkeypatch.setattr(
        ad,
        "reserve_child_control",
        lambda *args, **kwargs: {"status": "accepted", "generation": 1},
    )
    monkeypatch.setattr(dt, "request_hard_interrupt", lambda *args: True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(queue_subagent, sid, "next turn"),
                pool.submit(interrupt_subagent_with_replacement, sid, "replace"),
            ]
            results = [future.result() for future in futures]
        accepted = [item for item in results if item.get("status") in {"queued", "accepted"}]
        pending = [item for item in results if item.get("status") == "pending"]
        assert len(accepted) == 1
        assert len(pending) == 1
    finally:
        _unregister_subagent(sid)


def test_queue_subagent_uses_durable_acceptance_without_touching_current_child(
    tmp_path, monkeypatch
):
    import tools.async_delegation as ad

    db_path = tmp_path / "state.db"
    monkeypatch.setattr(ad, "_db_path", lambda: db_path)
    with ad._transaction() as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, state, dispatched_at, updated_at,
                children_json)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (
                "deleg-queue-live",
                "parent",
                time.time(),
                time.time(),
                json.dumps([{"subagent_id": "sid-queue-live", "session_id": "db-child"}]),
            ),
        )
    parent = _StubParent()
    child = _StubChild(parent)
    _register("sid-queue-live", child, delegation_id="deleg-queue-live")
    try:
        receipt = queue_subagent("sid-queue-live", "next turn")
        assert receipt["status"] == "queued"
        assert child.steered == []
    finally:
        _unregister_subagent("sid-queue-live")


def test_pending_child_control_runs_one_turn_on_same_session(monkeypatch):
    """A queued or replacement turn resumes the retained SessionDB child."""
    import tools.async_delegation as ad
    import tools.delegate_tool as dt

    parent = _StubParent()
    entry = {"child_id": "sid-ctl-same-session", "child_session_id": "db-child"}
    claimed = [{"action": "interrupt", "message": "replace it", "generation": 4}]
    retained = []
    delivered = []
    turns = []

    monkeypatch.setattr(
        ad,
        "claim_child_control",
        lambda *args, **kwargs: claimed.pop(0) if claimed else None,
    )
    monkeypatch.setattr(ad, "retain_completed_delegation", lambda *args, **kwargs: retained.append(args))
    monkeypatch.setattr(ad, "find_retained_child", lambda *args, **kwargs: entry)
    monkeypatch.setattr(ad, "claim_retained_child", lambda *args, **kwargs: "resume-claim")
    monkeypatch.setattr(ad, "release_retained_child", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dt,
        "_run_retained_child_turn",
        lambda found, message, owner, **kwargs: turns.append((found, message))
        or {"status": "completed", "summary": "replacement"},
    )
    monkeypatch.setattr(
        ad,
        "finish_child_control",
        lambda *args, **kwargs: delivered.append((args, kwargs)),
    )

    results = _run_pending_child_controls(
        "deleg-same-session", "sid-ctl-same-session", parent,
        current_result={"status": "interrupted"},
    )

    assert len(results) == 1
    assert turns == [(entry, "replace it")]
    assert retained
    assert delivered


def test_queue_and_interrupt_actions_are_control_only():
    from agent.tool_guardrails import _subagent_spawn_count

    assert _subagent_spawn_count({"action": "queue", "subagent_id": "x", "message": "m"}) == 0
    assert _subagent_spawn_count({"action": "interrupt", "subagent_id": "x", "message": "m"}) == 0


def test_foreign_parent_cannot_queue_or_interrupt_replacement(monkeypatch):
    import tools.delegate_tool as dt

    owner = _StubParent()
    foreign = _StubParent()
    child = _StubChild(owner)
    sid = "sid-ctl-foreign-delivery"
    _register(sid, child, delegation_id="deleg-foreign")
    monkeypatch.setattr(dt, "request_hard_interrupt", lambda *args: True)
    try:
        assert "No live subagent" in _handle_control_action(
            "queue", sid, "leak", foreign
        )
        assert "No live subagent" in _handle_control_action(
            "interrupt", sid, "leak", foreign
        )
    finally:
        _unregister_subagent(sid)


def test_restart_drain_delivers_each_accepted_control_exactly_once(monkeypatch):
    """After a restart, accepted controls start exactly one turn each.

    When the owning process exits, recovery classifies its delegation row
    ``unknown`` but an accepted queue/interrupt control is still pending. The
    rebooted session's next delegate_task drains every accepted control it
    owns as exactly one retained-child turn — they must not be stranded.
    """
    import tools.delegate_tool as dt
    import tools.async_delegation as ad

    parent = _StubParent()
    parent.session_id = "owner-session"
    entry = {"child_id": "child-restart", "child_session_id": "db-child"}
    claimed = [
        {"action": "queue", "message": "resume after restart", "generation": 7},
        {"action": "queue", "message": "resume again", "generation": 8},
    ]
    delivered = []
    turns = []

    monkeypatch.setattr(
        ad,
        "list_pending_child_controls",
        lambda **kw: [("deleg-restart", "child-restart")],
    )
    monkeypatch.setattr(ad, "find_retained_child", lambda *a, **k: entry)
    monkeypatch.setattr(
        ad,
        "claim_child_control",
        lambda *a, **k: (claimed.pop(0) if claimed else None),
    )
    monkeypatch.setattr(ad, "retain_completed_delegation", lambda *a, **k: None)
    monkeypatch.setattr(ad, "claim_retained_child", lambda *a, **k: "resume-claim")
    monkeypatch.setattr(ad, "release_retained_child", lambda *a, **k: None)
    monkeypatch.setattr(
        dt,
        "_run_retained_child_turn",
        lambda found, message, owner, **kwargs: turns.append((found, message))
        or {"status": "completed", "summary": "delivered"},
    )
    monkeypatch.setattr(
        ad, "finish_child_control", lambda *a, **k: delivered.append(a)
    )

    results = dt._drain_restarted_child_controls(parent)

    assert len(results) == 2
    assert turns == [
        (entry, "resume after restart"),
        (entry, "resume again"),
    ]
    assert len(delivered) == 2


def test_restart_drain_skips_foreign_children(monkeypatch):
    """A drained parent only delivers controls for children it owns."""
    import tools.delegate_tool as dt
    import tools.async_delegation as ad

    parent = _StubParent()
    parent.session_id = "owner-session"
    run = []
    monkeypatch.setattr(
        ad,
        "list_pending_child_controls",
        lambda **kw: [("deleg-foreign", "child-other")],
    )
    monkeypatch.setattr(ad, "find_retained_child", lambda *a, **k: None)
    monkeypatch.setattr(
        dt,
        "_run_pending_child_controls",
        lambda *a, **k: run.append(a) or [{"status": "completed"}],
    )

    assert dt._drain_restarted_child_controls(parent) == []
    assert run == []
