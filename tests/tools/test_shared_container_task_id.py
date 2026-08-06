"""Regression tests for delegated terminal environment ownership."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools import terminal_tool


@pytest.fixture(autouse=True)
def _clean_terminal_state():
    before = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    yield
    terminal_tool.cleanup_vm("default", force_remove=True)
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before)


def test_regular_and_delegated_tasks_share_default_environment_key():
    assert terminal_tool._resolve_container_task_id(None) == "default"
    assert terminal_tool._resolve_container_task_id("") == "default"
    assert terminal_tool._resolve_container_task_id("parent-turn") == "default"
    assert terminal_tool._resolve_container_task_id("sa-0-child") == "default"


def test_cwd_only_override_does_not_request_isolation():
    terminal_tool.register_task_env_overrides(
        "acp-session-abc", {"cwd": "/home/user/project"}
    )
    assert terminal_tool._resolve_container_task_id("acp-session-abc") == "default"


def test_backend_override_keeps_its_task_id():
    terminal_tool.register_task_env_overrides(
        "bench-env", {"env_type": "docker", "cwd": "/work"}
    )
    assert terminal_tool._resolve_container_task_id("bench-env") == "bench-env"


def test_delegation_lifecycle_keeps_shared_environment_until_parent_close(
    monkeypatch, tmp_path
):
    from run_agent import AIAgent
    from tools.delegate_tool import _run_single_child
    from tools.environments.local import LocalEnvironment

    created_for = []

    class FakeDockerEnvironment(LocalEnvironment):
        def __init__(self, *, task_id, cwd, timeout, persistent_filesystem, **_kwargs):
            created_for.append(task_id)
            super().__init__(cwd=cwd, timeout=timeout)
            self._persistent = persistent_filesystem

    config = {
        "env_type": "docker",
        "docker_image": "test:latest",
        "cwd": str(tmp_path),
        "timeout": 10,
        "lifetime_seconds": 3600,
        "container_persistent": True,
    }
    monkeypatch.setattr(terminal_tool, "_DockerEnvironment", FakeDockerEnvironment)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda _cc: None)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    parent_owner = SimpleNamespace(
        session_id="parent-session",
        platform="cli",
        _current_task_id="parent-turn",
        _end_session_on_close=False,
    )
    child_owner = SimpleNamespace(
        session_id="child-session",
        platform="subagent",
        _current_task_id=None,
        _end_session_on_close=False,
    )
    parent = MagicMock()
    parent.session_id = parent_owner.session_id
    parent._current_task_id = parent_owner._current_task_id
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    child = MagicMock()
    child.session_id = child_owner.session_id
    child._subagent_id = "sa-0-child"
    child._delegate_saved_tool_names = []
    child._credential_pool = None

    initial = json.loads(
        terminal_tool.terminal_tool(
            command="export DELEGATION_MARKER=shared",
            task_id=parent._current_task_id,
        )
    )
    assert initial["exit_code"] == 0
    shared_env = terminal_tool.get_active_env(parent._current_task_id)

    def run_conversation(*, task_id, **_kwargs):
        child_owner._current_task_id = task_id
        result = json.loads(
            terminal_tool.terminal_tool(
                command='printf %s "$DELEGATION_MARKER"', task_id=task_id
            )
        )
        return {
            "final_response": result["output"],
            "completed": True,
            "api_calls": 0,
            "messages": [],
        }

    child.run_conversation.side_effect = run_conversation
    child.close.side_effect = lambda: AIAgent.close(child_owner)
    parent._active_children.append(child)

    result = _run_single_child(0, "use the shared environment", child, parent)

    assert result["status"] == "completed"
    assert result["summary"] == "shared"
    assert created_for == ["default"]
    assert terminal_tool.get_active_env("sa-0-child") is shared_env
    AIAgent.close(parent_owner)
    assert terminal_tool.get_active_env(parent._current_task_id) is None
