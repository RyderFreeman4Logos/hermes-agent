"""
Regression tests for terminal task_id mapping.

The top-level agent maps an absent task ID to ``"default"``. Every non-empty
task ID identifies separate logical shell and file state, whether or not the
backend reuses one physical container.
"""

import json

import pytest

from tools import terminal_tool
from tools.file_tools import (
    _get_file_ops,
    clear_file_ops_cache,
    read_file_tool,
    write_file_tool,
)
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_overrides():
    """Ensure no stray overrides from other tests leak in."""
    before = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before)


def test_none_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_empty_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id("") == "default"


def test_cwd_only_override_keeps_own_id():
    terminal_tool.register_task_env_overrides(
        "acp-session-abc", {"cwd": "/home/user/project"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("acp-session-abc")
            == "acp-session-abc"
        )
    finally:
        terminal_tool.clear_task_env_overrides("acp-session-abc")


def test_env_type_override_keeps_own_id():
    """env_type is an isolation key — must trigger per-task container."""
    terminal_tool.register_task_env_overrides(
        "bench-env", {"env_type": "sandbox", "cwd": "/work"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("bench-env")
            == "bench-env"
        )
    finally:
        terminal_tool.clear_task_env_overrides("bench-env")


def test_delegated_tasks_keep_separate_terminal_and_file_state(
    tmp_path, monkeypatch
):
    child_a = "subagent-0-child-a"
    child_b = "subagent-1-child-b"
    parent_dir = tmp_path / "parent"
    child_a_dir = tmp_path / "child-a"
    child_b_dir = tmp_path / "child-b"
    for path in (parent_dir, child_a_dir, child_b_dir):
        path.mkdir()

    config = {
        "env_type": "local",
        "cwd": str(parent_dir),
        "timeout": 10,
        "lifetime_seconds": 3600,
        "local_persistent": False,
    }
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    terminal_tool.register_task_env_overrides(child_a, {"cwd": str(child_a_dir)})
    terminal_tool.register_task_env_overrides(child_b, {"cwd": str(child_b_dir)})

    def run(command, task_id, **kwargs):
        result = json.loads(
            terminal_tool.terminal_tool(command=command, task_id=task_id, **kwargs)
        )
        assert result["exit_code"] == 0, result
        return result

    try:
        run("export CHILD_MARKER=PARENT", None)
        parent_output = run('printf %s "$CHILD_MARKER"', "default")["output"]

        run("export CHILD_MARKER=FILE_A", child_a)
        run("export CHILD_MARKER=FILE_B", child_b)
        write_a = json.loads(write_file_tool("marker.txt", "FILE_A", task_id=child_a))
        write_b = json.loads(write_file_tool("marker.txt", "FILE_B", task_id=child_b))
        assert not write_a.get("error"), write_a
        assert not write_b.get("error"), write_b

        child_b_file = json.loads(read_file_tool("marker.txt", task_id=child_b))[
            "content"
        ]
        child_a_export = run('printf %s "$CHILD_MARKER"', child_a)["output"]
        child_b_export = run('printf %s "$CHILD_MARKER"', child_b)["output"]

        background = run("printf child-b", child_b, background=True)
        process = process_registry.get(background["session_id"])
        assert process is not None
        process_registry.wait(process.id, timeout=2)

        assert {
            "parent_none_and_default_reuse": terminal_tool.get_active_env(None)
            is terminal_tool.get_active_env("default"),
            "child_environments_differ": terminal_tool.get_active_env(child_a)
            is not terminal_tool.get_active_env(child_b),
            "child_file_ops_differ": _get_file_ops(child_a)
            is not _get_file_ops(child_b),
            "child_file_paths_differ": write_a["resolved_path"]
            == str(child_a_dir / "marker.txt")
            and write_b["resolved_path"] == str(child_b_dir / "marker.txt"),
            "child_b_reads_own_file": "FILE_B" in child_b_file
            and "FILE_A" not in child_b_file,
            "child_a_export": child_a_export,
            "child_b_export": child_b_export,
            "parent_export": parent_output,
            "process_task_id": process.task_id,
        } == {
            "parent_none_and_default_reuse": True,
            "child_environments_differ": True,
            "child_file_ops_differ": True,
            "child_file_paths_differ": True,
            "child_b_reads_own_file": True,
            "child_a_export": "FILE_A",
            "child_b_export": "FILE_B",
            "parent_export": "PARENT",
            "process_task_id": child_b,
        }
    finally:
        for task_id in (child_a, child_b, "default"):
            terminal_tool.cleanup_vm(task_id)
            terminal_tool.clear_task_env_overrides(task_id)
        clear_file_ops_cache()
