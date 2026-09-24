"""Executor path: high-volume result plus subdirectory hints stay within the insertion cap."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.tool_executor import _ToolCallRef, _commit_tool_result
from tools.budget_config import DEFAULT_BUDGET
from tools.environments.local import LocalEnvironment
from tools.tool_result_storage import (
    _INSERTION_COMPACT_THRESHOLD_CHARS,
    extract_persisted_path,
)


def test_commit_tool_result_bounds_result_plus_two_subdir_hints(tmp_path, monkeypatch):
    """25k high-volume result + two ~32k hints must not land as an ~89k history row."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    root = tmp_path / "project"
    parent = root / "parent"
    deep = parent / "deep"
    deep.mkdir(parents=True)
    (root / "AGENTS.md").write_text("root-already-loaded\n", encoding="utf-8")
    (parent / "AGENTS.md").write_text("parent-rules\n" + ("P" * 32_000), encoding="utf-8")
    (deep / "AGENTS.md").write_text("deep-rules\n" + ("D" * 32_000), encoding="utf-8")
    target = deep / "file.py"
    target.write_text("print('ok')\n", encoding="utf-8")

    raw_result = "RAW-RESULT-START\n" + ("x" * 25_000) + "\nRAW-RESULT-END\n"
    tracker = SubdirectoryHintTracker(working_dir=str(root))
    hints = tracker.check_tool_call("terminal", {"command": f"cat {target}"})
    assert hints is not None
    assert "deep-rules" in hints and "parent-rules" in hints
    assert "D" * 20 in hints and "P" * 20 in hints

    agent = SimpleNamespace(
        verbose_logging=False,
        tool_progress_callback=None,
        _subdirectory_hints=SubdirectoryHintTracker(working_dir=str(root)),
        _tool_guardrails=SimpleNamespace(record_persisted_result=MagicMock()),
        _current_tool=None,
        _incremental_persistence_failed=False,
        _last_persistence_error_cause=None,
    )
    agent._touch_activity = lambda *_args, **_kwargs: None
    agent._tool_result_content_for_active_model = lambda _name, result: result
    agent._flush_messages_to_session_db = MagicMock(return_value=True)

    messages: list[dict] = []
    ref = _ToolCallRef(
        name="terminal",
        args={"command": f"cat {target}"},
        task_id="default",
        call_id="call-suffix-budget",
        trace=[],
    )
    env = LocalEnvironment.__new__(LocalEnvironment)

    with patch("agent.tool_executor.get_active_env", return_value=env):
        committed = _commit_tool_result(
            agent,
            messages,
            ref,
            raw_result,
            budget=DEFAULT_BUDGET,
            tool_duration=0.01,
            is_error=False,
            blocked=False,
            effect_disposition=None,
            observed=False,
        )

    assert committed is not None
    persisted_result, display_result, _risk = committed
    assert display_result == raw_result
    history = messages[-1]["content"]
    assert history == persisted_result
    assert len(history) <= _INSERTION_COMPACT_THRESHOLD_CHARS
    assert "deep-rules" in history and "parent-rules" in history
    assert "D" * 20 in history and "P" * 20 in history
    path = extract_persisted_path(history)
    assert path is not None
    assert Path(path).read_text(encoding="utf-8") == raw_result
