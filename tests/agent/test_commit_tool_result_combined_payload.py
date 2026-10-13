"""Production-seam combined payload: 32K insertion, 8K/16K hints, history_suffix reservation."""

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


def test_commit_tool_result_reserves_history_suffix_and_recovers_raw_result(
    tmp_path: Path, monkeypatch,
) -> None:
    """20–31K high-volume result + 16K nearest-first hints stay within 32K history."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    root = tmp_path / "project"
    parent = root / "parent"
    deep = parent / "deep"
    deep.mkdir(parents=True)
    (root / "AGENTS.md").write_text("root-already-loaded\n", encoding="utf-8")
    (parent / "AGENTS.md").write_text("parent-rules\n" + ("P" * 9_000), encoding="utf-8")
    (deep / "AGENTS.md").write_text("deep-rules\n" + ("D" * 9_000), encoding="utf-8")
    target = deep / "file.py"
    target.write_text("print('ok')\n", encoding="utf-8")

    raw_result = "RAW-RESULT-START\n" + ("x" * 25_000) + "\nRAW-RESULT-END\n"
    assert 20_000 <= len(raw_result) <= 31_000

    tracker = SubdirectoryHintTracker(working_dir=str(root))
    expected_hints = tracker.check_tool_call("terminal", {"command": f"cat {target}"})
    assert expected_hints is not None
    assert len(expected_hints) <= 16_000
    assert expected_hints.index("deep-rules") < expected_hints.index("parent-rules")
    assert "truncated" in expected_hints.lower()

    agent = SimpleNamespace(
        verbose_logging=False,
        tool_progress_callback=None,
        _subdirectory_hints=SubdirectoryHintTracker(working_dir=str(root)),
        _tool_guardrails=SimpleNamespace(record_persisted_result=MagicMock()),
        _current_tool=None,
    )
    agent._touch_activity = lambda *_args, **_kwargs: None
    agent._tool_result_content_for_active_model = lambda _name, result: result
    agent._flush_messages_to_session_db = MagicMock(return_value=True)

    messages: list[dict] = []
    ref = _ToolCallRef(
        name="terminal",
        args={"command": f"cat {target}"},
        task_id="default",
        call_id="call-combined-payload",
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
    assert messages and messages[-1]["role"] == "tool"
    history = messages[-1]["content"]
    assert history == persisted_result
    assert len(history) <= _INSERTION_COMPACT_THRESHOLD_CHARS
    assert expected_hints in history
    assert history.index("deep-rules") < history.index("parent-rules")
    assert history.endswith(expected_hints) or expected_hints in history

    path = extract_persisted_path(history)
    assert path is not None
    recovered = Path(path).read_text(encoding="utf-8")
    assert recovered == raw_result
    agent._flush_messages_to_session_db.assert_called_once_with(messages)
