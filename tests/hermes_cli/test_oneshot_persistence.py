import logging
import sqlite3
import sys
import time
import types

import pytest

import hermes_state
from hermes_cli import oneshot
from hermes_state import SessionDB


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_persistence_budget_is_shared_across_writes(monkeypatch):
    clock = [100.0]
    patience = []

    class FakeSessionDB:
        _WRITE_PATIENCE_S = 20.0

        def __init__(self):
            self.db_path = "state.db"
            self._execute_write(lambda _conn: None, patience_s=60.0)

        def _execute_write(self, fn, patience_s=None):
            patience.append(patience_s)
            clock[0] += 4.0

    monkeypatch.setattr(oneshot.time, "monotonic", lambda: clock[0])
    monkeypatch.setitem(
        sys.modules,
        "hermes_state",
        _module("hermes_state", SessionDB=FakeSessionDB),
    )

    db = oneshot._create_session_db_for_oneshot()
    db._execute_write(lambda _conn: None, patience_s=60.0)

    assert patience == [10.0, 6.0]


def test_locked_state_db_fails_within_oneshot_budget(monkeypatch, tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(oneshot, "_PERSISTENCE_BUDGET_S", 0.2)

    holder = sqlite3.connect(db_path, timeout=1.0, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="one-shot budget.*No model call"):
            oneshot._create_session_db_for_oneshot()
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    assert time.monotonic() - started < 6.0


def test_stateless_oneshot_never_constructs_session_db(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt):
            captured["prompt"] = prompt
            captured["persist_disabled"] = self._persist_disabled
            return {"final_response": "ok"}

        def shutdown_memory_provider(self, *_args):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        oneshot,
        "_create_session_db_for_oneshot",
        lambda: pytest.fail("stateless mode opened state.db"),
    )
    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module("hermes_cli.models", detect_provider_for_model=lambda *_a, **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_k: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "requested_provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_a, **_k: {"session_search", "terminal"},
        ),
    )

    text, _result = oneshot._run_agent("hello", no_session_persistence=True)

    assert text == "ok"
    assert captured["session_db"] is None
    assert captured["disabled_toolsets"] == ["session_search"]
    assert captured["persist_disabled"] is True


def test_failed_result_without_response_is_visible_on_stderr(monkeypatch, capsys):
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_a, **_k: (
            "",
            {
                "failed": True,
                "turn_exit_reason": "session_persistence_failed",
            },
        ),
    )
    try:
        assert oneshot.run_oneshot("hello") == 2
    finally:
        logging.disable(logging.NOTSET)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "session persistence failed" in captured.err


def test_pretool_persistence_failure_suppresses_stdout(monkeypatch, capsys):
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_a, **_k: (
            "No reply: session storage could not be written.",
            {
                "failed": True,
                "turn_exit_reason": "session_persistence_failed",
            },
        ),
    )
    try:
        assert oneshot.run_oneshot("hello") == 2
    finally:
        logging.disable(logging.NOTSET)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "session persistence failed" in captured.err
