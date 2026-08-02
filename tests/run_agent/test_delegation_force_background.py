"""Model-facing delegation must never silently fall back inline."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.config import DEFAULT_CONFIG
from run_agent import AIAgent


@pytest.mark.parametrize(
    ("configured", "depth", "expected_background", "expected_force"),
    [
        pytest.param(False, 0, True, True, id="top-level-always-guaranteed"),
        pytest.param(False, 1, False, False, id="nested-default-sync"),
        pytest.param(True, 1, True, True, id="configured-nested-guarantee"),
    ],
)
def test_model_dispatch_resolves_no_inline_guarantee_at_every_depth(
    configured, depth, expected_background, expected_force
):
    agent = AIAgent.__new__(AIAgent)
    setattr(agent, "_delegate_depth", depth)

    with (
        patch("tools.delegate_tool._load_config", return_value={"force_background": configured}),
        patch("tools.delegate_tool.delegate_task", return_value="dispatched") as delegate,
    ):
        result = AIAgent._dispatch_delegate_task(
            agent,
            {"goal": "test", "background": not expected_background},
        )

    assert result == "dispatched"
    assert delegate.call_args.kwargs["background"] is expected_background
    assert delegate.call_args.kwargs["force_background"] is expected_force


@pytest.mark.parametrize(
    ("depth", "configured", "expected"),
    [
        (0, False, True),
        (1, False, False),
        (1, True, True),
    ],
)
def test_registry_fallback_resolves_force_background(monkeypatch, depth, configured, expected):
    import tools.delegate_tool as delegate_tool

    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {"force_background": configured},
    )
    parent = SimpleNamespace(_delegate_depth=depth)

    assert delegate_tool._model_force_background_value(parent) is expected


def test_force_background_defaults_to_false():
    assert DEFAULT_CONFIG["delegation"]["force_background"] is False


@pytest.mark.parametrize(
    ("value", "configured", "expected"),
    [
        pytest.param(None, True, True, id="config-default"),
        pytest.param(False, True, False, id="explicit-opt-out"),
        pytest.param(True, False, True, id="explicit-opt-in"),
        pytest.param("false", True, False, id="truthy-string-normalization"),
    ],
)
def test_force_background_resolution(value, configured, expected):
    from tools.delegate_tool import _resolve_force_background

    with patch(
        "tools.delegate_tool._load_config",
        return_value={"force_background": configured},
    ):
        assert _resolve_force_background(value) is expected


def _delegate_dependencies(monkeypatch, *, async_delivery, dispatch_result=None):
    import tools.delegate_tool as delegate_tool

    parent = MagicMock(
        _delegate_depth=0,
        session_id="parent-session",
        _interrupt_requested=False,
        _active_children=[],
        _active_children_lock=None,
    )
    child = MagicMock(model="m", _delegate_role="leaf", _subagent_id="child-1")
    parent._active_children = [child]
    child_result = {
        "task_index": 0,
        "status": "completed",
        "summary": "done",
        "api_calls": 1,
        "duration_seconds": 0.1,
        "model": "m",
        "exit_reason": "completed",
    }

    def _run_child(*_args, **_kwargs):
        child.close.assert_not_called()
        assert child in parent._active_children
        parent._active_children.remove(child)
        return dict(child_result)

    run_child = MagicMock(side_effect=_run_child)
    resolved_dispatch_result = dispatch_result or {
        "status": "dispatched",
        "delegation_id": "delegation-1",
    }

    def _dispatch(**kwargs):
        if resolved_dispatch_result.get("status") == "rejected":
            kwargs["on_not_started"]("error")
        return dict(resolved_dispatch_result)

    dispatch = MagicMock(side_effect=_dispatch)
    credentials = {
        "model": "m",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }

    monkeypatch.setattr(delegate_tool, "_build_child_agent", lambda **_kwargs: child)
    monkeypatch.setattr(delegate_tool, "_run_single_child", run_child)
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        lambda *_args, **_kwargs: credentials,
    )
    monkeypatch.setattr(
        "tools.delegation_live_log.create_live_transcripts",
        lambda *_args, **_kwargs: (None, [], []),
    )
    monkeypatch.setattr(
        "tools.async_delegation._current_origin_session_id",
        lambda: "",
    )
    probe = MagicMock()
    if isinstance(async_delivery, Exception):
        probe.side_effect = async_delivery
    else:
        probe.return_value = async_delivery
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", probe)
    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        dispatch,
    )
    return delegate_tool, parent, run_child, dispatch


@pytest.mark.parametrize(
    "async_delivery",
    [False, RuntimeError("probe failed")],
    ids=["unsupported", "probe-error"],
)
def test_forced_dispatch_rejects_before_child_runs_without_completion_route(
    monkeypatch, async_delivery
):
    delegate_tool, parent, run_child, dispatch = _delegate_dependencies(
        monkeypatch,
        async_delivery=async_delivery,
    )

    result = json.loads(
        delegate_tool.delegate_task(
            goal="complete this",
            background=False,
            force_background=True,
            parent_agent=parent,
        )
    )

    assert result["status"] == "rejected"
    assert result["mode"] == "background"
    assert "not started" in result["note"]
    run_child.assert_not_called()
    dispatch.assert_not_called()
    assert parent._active_children == []


def test_forced_dispatch_rejects_full_async_pool_without_inline_fallback(monkeypatch):
    delegate_tool, parent, run_child, _ = _delegate_dependencies(
        monkeypatch,
        async_delivery=True,
        dispatch_result={"status": "rejected", "error": "pool full"},
    )

    result = json.loads(
        delegate_tool.delegate_task(
            goal="complete this",
            background=True,
            force_background=True,
            parent_agent=parent,
        )
    )

    assert result["status"] == "rejected"
    assert result["mode"] == "background"
    assert "pool full" in result["error"]
    assert "not started" in result["note"]
    run_child.assert_not_called()
    assert parent._active_children == []


def test_direct_python_caller_can_explicitly_keep_pool_fallback(monkeypatch):
    delegate_tool, parent, run_child, dispatch = _delegate_dependencies(
        monkeypatch,
        async_delivery=True,
        dispatch_result={"status": "rejected", "error": "pool full"},
    )

    result = json.loads(
        delegate_tool.delegate_task(
            goal="safe internal prerequisite",
            background=True,
            force_background=False,
            parent_agent=parent,
        )
    )

    assert result["results"][0]["status"] == "completed"
    assert "SYNCHRONOUSLY" in result["note"]
    run_child.assert_called_once()
    dispatch.assert_called_once()
    assert parent._active_children == []


def test_direct_python_caller_can_explicitly_keep_sync_fallback(monkeypatch):
    delegate_tool, parent, run_child, dispatch = _delegate_dependencies(
        monkeypatch,
        async_delivery=False,
    )

    result = json.loads(
        delegate_tool.delegate_task(
            goal="safe internal prerequisite",
            background=True,
            force_background=False,
            parent_agent=parent,
        )
    )

    assert result["results"][0]["status"] == "completed"
    assert "SYNCHRONOUSLY" in result["note"]
    run_child.assert_called_once()
    dispatch.assert_not_called()
    assert parent._active_children == []
