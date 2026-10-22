"""Offline issue #324 regression: abandoned memory writes lose caller identity."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.tool_executor import execute_tool_calls_sequential
from run_agent import AIAgent
from tools.memory_tool_store import MemoryStore


class _AdmissionProvider(MemoryProvider):
    """Offline provider fixture with idempotency and conflict admission rules."""

    def __init__(self, admission_db: Path) -> None:
        self.calls: list[dict] = []
        self.conflicts: list[tuple[str, str]] = []
        self.identity_errors: list[str] = []
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self.first_finished = threading.Event()
        self._lock = threading.Lock()
        self._db = sqlite3.connect(admission_db, check_same_thread=False)
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute(
            "CREATE TABLE admissions (operation_key TEXT PRIMARY KEY, action TEXT, target TEXT, content TEXT)"
        )
        self._db.commit()

    @property
    def name(self) -> str:
        return "offline-admission"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self) -> None:
        pass

    def on_memory_write(self, action, target, content, metadata=None, execution_context=None):
        with self._lock:
            index = len(self.calls)
            self.calls.append({
                "action": action,
                "target": target,
                "content": content,
                "metadata": dict(metadata or {}),
                "context": execution_context,
            })

        def hold_first_call() -> None:
            if index == 0:
                self.first_entered.set()
                try:
                    assert self.release_first.wait(timeout=5), "first provider call was not released"
                finally:
                    self.first_finished.set()

        key = getattr(execution_context, "operation_key", None)
        if not key:
            hold_first_call()
            self.identity_errors.append(f"provider call {index} had no caller key")
            return
        with self._lock:
            previous = self._db.execute(
                "SELECT action, target, content FROM admissions WHERE operation_key = ?", (key,)
            ).fetchone()
            if previous is None:
                self._db.execute(
                    "INSERT INTO admissions VALUES (?, ?, ?, ?)", (key, action, target, content)
                )
                self._db.commit()
            elif previous != (action, target, content):
                self.conflicts.append((key, content))
        hold_first_call()

    def admissions(self) -> list[tuple[str, str, str, str]]:
        with self._lock:
            return self._db.execute(
                "SELECT operation_key, action, target, content FROM admissions ORDER BY rowid"
            ).fetchall()

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _make_agent(tmp_path: Path) -> AIAgent:
    definitions = [{
        "type": "function",
        "function": {
            "name": "memory",
            "description": "offline memory",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        patch("model_tools.get_tool_definitions", return_value=definitions),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="offline-test-key",
            base_url="https://offline.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._append_guardrail_observation = MagicMock(
        side_effect=lambda _name, _args, result, **_kwargs: result
    )
    agent._record_file_mutation_result = MagicMock()
    agent._subdirectory_hints.check_tool_call = MagicMock(return_value="")
    agent._tool_result_content_for_active_model = MagicMock(
        side_effect=lambda _name, result: result
    )
    return agent


def _call(call_id: str, operation_key: str, content: str = "same text"):
    args = {
        "action": "add",
        "target": "memory",
        "content": content,
        "operation_key": operation_key,
    }
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="memory", arguments=json.dumps(args)),
    )


def _run(agent, call):
    messages: list[dict] = []
    execute_tool_calls_sequential(agent, SimpleNamespace(tool_calls=[call]), messages, "offline-task")
    return messages


def test_abandoned_retry_preserves_operation_identity_and_context(tmp_path, monkeypatch):
    """Real sequential executor: local admission happens before timeout, then retry is one mutation."""
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "0.2")
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path / "memories")

    agent = _make_agent(tmp_path)
    store = MemoryStore(memory_char_limit=500, user_char_limit=300)
    manager = MemoryManager()
    provider = _AdmissionProvider(tmp_path / "provider-admission.sqlite")
    manager.add_provider(provider)
    agent._memory_store = store
    agent._memory_manager = manager
    original_notify = manager.notify_memory_tool_write
    original_args: list[dict] = []

    def capture_original_args(tool_result, tool_args, **kwargs):
        original_args.append(dict(tool_args))
        return original_notify(tool_result, tool_args, **kwargs)

    manager.notify_memory_tool_write = capture_original_args

    first_call = _call("call-timeout", "op-1")
    first_messages: list[dict] = []
    expected_keys = ["op-1", "op-1", "op-2", "op-conflict", "op-conflict"]
    final_admissions = []
    try:
        execute_tool_calls_sequential(
            agent, SimpleNamespace(tool_calls=[first_call]), first_messages, "offline-task"
        )
        assert provider.first_entered.wait(timeout=2), "provider was not reached after local admission"
        assert not provider.release_first.is_set(), "provider was released before the timeout result"
        assert provider.admissions() == [("op-1", "add", "memory", "same text")]
        assert (tmp_path / "memories" / "MEMORY.md").read_text(encoding="utf-8").count("same text") == 1
        assert "timed out" in first_messages[0]["content"]
        assert first_messages[0]["effect_disposition"] == "unknown"
        assert "op-1" not in first_messages[0]["content"]

        retry_messages = _run(agent, _call("call-retry", "op-1"))
        assert json.loads(retry_messages[0]["content"])["success"] is True

        distinct_messages = _run(agent, _call("call-distinct", "op-2"))
        assert json.loads(distinct_messages[0]["content"])["success"] is True

        _run(agent, _call("call-conflict-a", "op-conflict", "first intent"))
        _run(agent, _call("call-conflict-b", "op-conflict", "conflicting intent"))
    finally:
        provider.release_first.set()
        assert provider.first_finished.wait(timeout=2)
        manager.shutdown_all()
        final_admissions = provider.admissions()
        provider.close()

    contexts = [call["context"] for call in provider.calls]
    first_context = contexts[0] if contexts else None
    cancel_event = getattr(first_context, "cancel_event", None)
    deadline = getattr(first_context, "deadline", None)
    observed = {
        "original_args": [args["operation_key"] for args in original_args],
        "provider_metadata": [call["metadata"].get("operation_key") for call in provider.calls],
        "provider_context": [getattr(context, "operation_key", None) for context in contexts],
        "cancelled_after_timeout": bool(cancel_event and cancel_event.is_set()),
        "deadline_expired_after_timeout": bool(deadline is not None and deadline <= time.monotonic()),
        "admissions": final_admissions,
        "conflicts": provider.conflicts,
        "identity_errors": provider.identity_errors,
    }
    assert observed == {
        "original_args": expected_keys,
        "provider_metadata": expected_keys,
        "provider_context": expected_keys,
        "cancelled_after_timeout": True,
        "deadline_expired_after_timeout": True,
        "admissions": [
            ("op-1", "add", "memory", "same text"),
            ("op-2", "add", "memory", "same text"),
            ("op-conflict", "add", "memory", "first intent"),
        ],
        "conflicts": [("op-conflict", "conflicting intent")],
        "identity_errors": [],
    }, "caller identity was present in original args but lost at provider boundary"
