"""Secure parent control of live native delegate children."""

import json
import threading
from typing import Any, cast
from unittest.mock import MagicMock

from agent.subagent_lifecycle import bind_subagent_parent
from tools.delegate_tool import _register_subagent, _unregister_subagent
from tools.registry import registry


class _Transport:
    def write(self, _obj: dict) -> bool:
        return True

    def close(self) -> None:
        return None


class _Child:
    def __init__(self) -> None:
        self.interrupts: list[str] = []

    def hard_interrupt(self, message: str) -> None:
        self.interrupts.append(message)


def _call_interrupt(params: dict, *, transport, session_record) -> dict:
    import tui_gateway.server as srv

    session_id = params["session_id"]
    srv._sessions[session_id] = session_record
    try:
        result = srv.dispatch(
            {"id": 1, "method": "subagent.interrupt", "params": params},
            transport=transport,
        )
        assert result is not None
        return result
    finally:
        srv._sessions.pop(session_id, None)


def _control(parent, args: dict) -> dict:
    from model_tools import handle_function_call

    with bind_subagent_parent(parent):
        raw = handle_function_call(
            "delegate_control",
            args,
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )
    return json.loads(raw)


def test_interrupt_requires_the_commissioning_session_transport_and_generation():
    owner_transport = _Transport()
    owner_record = {
        "session_key": "owner-session",
        "history": [],
        "transport": owner_transport,
    }
    recycled_record = {
        "session_key": "owner-session",
        "history": [],
        "transport": owner_transport,
    }
    foreign_transport = _Transport()
    foreign_record = {
        "session_key": "foreign-session",
        "history": [],
        "transport": foreign_transport,
    }
    child = _Child()
    _register_subagent(
        {
            "subagent_id": "sid-interrupt-generation",
            "status": "running",
            "agent": child,
            "owner_session_id": "owner-session",
            "owner_transport": owner_transport,
            "owner_session_record": owner_record,
        }
    )
    try:
        owned = _call_interrupt(
            {
                "session_id": "owner-session",
                "subagent_id": "sid-interrupt-generation",
            },
            transport=owner_transport,
            session_record=owner_record,
        )
        assert owned["result"]["found"] is True
        assert len(child.interrupts) == 1

        for session_id, transport, record in (
            ("foreign-session", foreign_transport, foreign_record),
            ("owner-session", _Transport(), owner_record),
            ("owner-session", owner_transport, recycled_record),
        ):
            rejected = _call_interrupt(
                {
                    "session_id": session_id,
                    "subagent_id": "sid-interrupt-generation",
                },
                transport=transport,
                session_record=record,
            )
            assert rejected["result"]["found"] is False
        assert len(child.interrupts) == 1
    finally:
        _unregister_subagent("sid-interrupt-generation", agent=child)


def test_delegate_control_is_deferred_in_the_delegation_toolset():
    from toolsets import TOOLSETS
    from tools.delegate_tool import _blocked_toolsets_for_role

    entry = registry.get_entry("delegate_control")
    assert entry is not None
    assert entry.toolset == "delegation"
    delegation_tools = TOOLSETS["delegation"].get("tools")
    assert isinstance(delegation_tools, list)
    assert {"delegate_task", "delegate_control"} <= set(delegation_tools)
    assert "delegation" in _blocked_toolsets_for_role("leaf")
    assert "delegation" not in _blocked_toolsets_for_role("orchestrator")


def test_model_list_returns_only_the_exact_parents_live_children():
    owner = object()
    foreign_owner = object()
    own_child = _Child()
    foreign_child = _Child()
    long_goal = "g" * 500
    records = (
        {
            "subagent_id": "sid-owned-complete-handle",
            "status": "running",
            "started_at": 10.0,
            "goal": long_goal,
            "role": "leaf",
            "depth": 0,
            "tool_count": 7,
            "agent": own_child,
            "owner_agent": owner,
        },
        {
            "subagent_id": "sid-foreign",
            "status": "running",
            "started_at": 11.0,
            "goal": "secret foreign goal",
            "role": "leaf",
            "depth": 0,
            "agent": foreign_child,
            "owner_agent": foreign_owner,
        },
    )
    for record in records:
        _register_subagent(record)
    try:
        result = _control(owner, {"action": "list"})
        assert result["status"] == "ok"
        assert result["children"] == [
            {
                "subagent_id": "sid-owned-complete-handle",
                "status": "running",
                "started_at": 10.0,
                "goal_preview": long_goal[:200],
                "role": "leaf",
                "depth": 0,
                "tool_count": 7,
            }
        ]
        assert result["next_cursor"] is None
        assert "foreign" not in repr(result)
        assert "owner_agent" not in repr(result)
        assert "agent" not in result["children"][0]
    finally:
        _unregister_subagent("sid-owned-complete-handle", agent=own_child)
        _unregister_subagent("sid-foreign", agent=foreign_child)


def test_model_list_is_bounded_and_paginated():
    owner = object()
    children = []
    for index in range(21):
        child = _Child()
        children.append(child)
        _register_subagent(
            {
                "subagent_id": f"sid-page-{index:02d}",
                "status": "running",
                "started_at": float(index),
                "goal": f"goal {index}",
                "role": "leaf",
                "depth": 0,
                "agent": child,
                "owner_agent": owner,
            }
        )
    try:
        first = _control(owner, {"action": "list"})
        assert len(first["children"]) == 20
        assert first["next_cursor"] == 20
        second = _control(owner, {"action": "list", "cursor": 20})
        assert [child["subagent_id"] for child in second["children"]] == [
            "sid-page-20"
        ]
        assert second["next_cursor"] is None
    finally:
        for index, child in enumerate(children):
            _unregister_subagent(f"sid-page-{index:02d}", agent=child)


def test_run_single_child_binds_the_exact_parent_and_role():
    from tools.delegate_tool import _run_single_child

    running = threading.Event()
    release = threading.Event()
    parent = MagicMock()
    child = MagicMock()
    child._subagent_id = "sid-real-registration"
    child._delegate_depth = 1
    child._delegate_role = "orchestrator"
    child.model = "test-model"

    def run_conversation(**_kwargs):
        running.set()
        assert release.wait(5)
        return {
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }

    child.run_conversation.side_effect = run_conversation
    runner = threading.Thread(
        target=lambda: _run_single_child(
            0,
            "real registration",
            child=child,
            parent_agent=parent,
        )
    )
    runner.start()
    assert running.wait(5)
    try:
        result = _control(parent, {"action": "list"})
        assert [item["subagent_id"] for item in result["children"]] == [
            "sid-real-registration"
        ]
        assert result["children"][0]["role"] == "orchestrator"
    finally:
        release.set()
        runner.join(5)
    assert not runner.is_alive()


def test_status_hides_foreign_and_unknown_children_alike():
    owner = object()
    foreign_owner = object()
    own_child = _Child()
    foreign_child = _Child()
    _register_subagent(
        {
            "subagent_id": "sid-status-owned",
            "status": "running",
            "started_at": 12.0,
            "goal": "owned goal",
            "role": "leaf",
            "depth": 0,
            "agent": own_child,
            "owner_agent": owner,
        }
    )
    _register_subagent(
        {
            "subagent_id": "sid-status-foreign",
            "status": "running",
            "started_at": 13.0,
            "goal": "foreign goal",
            "role": "leaf",
            "depth": 0,
            "agent": foreign_child,
            "owner_agent": foreign_owner,
        }
    )
    try:
        owned = _control(
            owner,
            {"action": "status", "subagent_id": "sid-status-owned"},
        )
        assert owned == {
            "status": "ok",
            "child": {
                "subagent_id": "sid-status-owned",
                "status": "running",
                "started_at": 12.0,
                "goal_preview": "owned goal",
                "role": "leaf",
                "depth": 0,
                "tool_count": 0,
            },
        }
        for subagent_id in ("sid-status-foreign", "sid-status-missing"):
            assert _control(
                owner,
                {"action": "status", "subagent_id": subagent_id},
            ) == {"status": "not_found", "subagent_id": subagent_id}
    finally:
        _unregister_subagent("sid-status-owned", agent=own_child)
        _unregister_subagent("sid-status-foreign", agent=foreign_child)


def test_cancel_and_kill_are_cooperative_and_owner_scoped():
    owner = object()
    foreign_owner = object()
    children = {action: _Child() for action in ("cancel", "kill")}
    foreign_child = _Child()
    for action, child in children.items():
        _register_subagent(
            {
                "subagent_id": f"sid-{action}-owned",
                "status": "running",
                "started_at": 20.0,
                "goal": action,
                "role": "leaf",
                "depth": 0,
                "agent": child,
                "owner_agent": owner,
            }
        )
    _register_subagent(
        {
            "subagent_id": "sid-cancel-foreign",
            "status": "running",
            "agent": foreign_child,
            "owner_agent": foreign_owner,
        }
    )
    try:
        for action, child in children.items():
            result = _control(
                owner,
                {"action": action, "subagent_id": f"sid-{action}-owned"},
            )
            assert result == {
                "status": "accepted",
                "action": action,
                "subagent_id": f"sid-{action}-owned",
                "cooperative": True,
            }
            assert len(child.interrupts) == 1
            status = _control(
                owner,
                {"action": "status", "subagent_id": f"sid-{action}-owned"},
            )
            assert status["child"]["status"] == "cancelling"

        for subagent_id in ("sid-cancel-foreign", "sid-cancel-missing"):
            assert _control(
                owner,
                {"action": "cancel", "subagent_id": subagent_id},
            ) == {"status": "not_found", "subagent_id": subagent_id}
        assert foreign_child.interrupts == []
    finally:
        for action, child in children.items():
            _unregister_subagent(f"sid-{action}-owned", agent=child)
        _unregister_subagent("sid-cancel-foreign", agent=foreign_child)


def test_model_control_rejects_a_recycled_gateway_session_generation():
    import tui_gateway.server as srv
    from gateway.session_context import clear_session_vars, set_session_vars
    from tui_gateway.transport import bind_transport, reset_transport

    owner = object()
    owner_transport = _Transport()
    owner_record = {
        "session_key": "ui-owner",
        "history": [],
        "transport": owner_transport,
    }
    recycled_record = {
        "session_key": "ui-owner",
        "history": [],
        "transport": owner_transport,
    }
    child = _Child()
    _register_subagent(
        {
            "subagent_id": "sid-model-generation",
            "status": "running",
            "started_at": 30.0,
            "goal": "generation-bound",
            "role": "leaf",
            "depth": 0,
            "agent": child,
            "owner_agent": owner,
            "owner_session_id": "ui-owner",
            "owner_transport": owner_transport,
            "owner_session_record": owner_record,
        }
    )
    srv._sessions["ui-owner"] = owner_record
    session_tokens = set_session_vars(
        session_key="durable-parent",
        session_id="durable-parent",
        ui_session_id="ui-owner",
    )
    transport_token = bind_transport(cast(Any, owner_transport))
    record_token = srv._current_runtime_session_record.set(owner_record)
    try:
        assert _control(owner, {"action": "list"})["children"][0][
            "subagent_id"
        ] == "sid-model-generation"
        srv._sessions["ui-owner"] = recycled_record
        assert _control(owner, {"action": "list"}) == {"status": "rejected"}
    finally:
        srv._current_runtime_session_record.reset(record_token)
        reset_transport(transport_token)
        clear_session_vars(session_tokens)
        srv._sessions.pop("ui-owner", None)
        _unregister_subagent("sid-model-generation", agent=child)


def test_model_control_requires_a_parent_and_bounded_string_handles():
    assert _control(None, {"action": "list"}) == {"status": "rejected"}
    owner = object()
    for subagent_id in ("x" * 129, {"not": "a handle"}):
        assert _control(
            owner,
            {"action": "status", "subagent_id": subagent_id},
        ) == {"status": "rejected"}
