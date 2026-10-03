"""Owner-scope, ordered checkpoint, drain semantics, and native lineage.

Public behavior only. Each test fails on the pre-repair tree for the named class.
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest


def test_explicit_off_blocks_agent_notify_before_completion(tmp_path, monkeypatch):
    """Owning-profile OFF admits no completion turn, even with agent_notify."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from tools.process_registry import ProcessSession
    import tools.process_registry as pr

    home = tmp_path / "owner"
    (home / "profiles" / "owner").mkdir(parents=True)
    (home / "profiles" / "owner" / "config.yaml").write_text(
        "display:\n  background_process_notifications: off\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BACKGROUND_NOTIFICATIONS", "off")
    registry = pr.ProcessRegistry()
    monkeypatch.setattr(pr, "process_registry", registry)
    session = ProcessSession(
        id="proc_off", command="echo off", task_id="t", started_at=time.time(),
        output_buffer="x\n", exited=True, exit_code=1, notify_on_complete=True,
        parent_session_id="owner-db",
    )
    registry._finished[session.id] = session
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(handle_message=AsyncMock())}
    enqueued = []

    async def _enqueue(text, evt):
        enqueued.append(evt)
        return True

    runner._enqueue_process_completion_notification = _enqueue
    runner._completion_event_scope = lambda _evt: __import__("contextlib").nullcontext()
    runner._load_background_notifications_mode = staticmethod(lambda: "off")

    async def _go():
        await runner._run_process_watcher({
            "session_id": session.id, "check_interval": 0, "platform": "telegram",
            "chat_id": "1", "notify_on_complete": True, "parent_session_id": "stale",
        })

    asyncio.run(_go())
    assert enqueued == []


def test_completion_uses_post_handoff_session_not_armed_snapshot(tmp_path, monkeypatch):
    import asyncio
    from gateway.run import GatewayRunner
    import tools.process_registry as pr
    from tools.process_registry import ProcessSession

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    registry = pr.ProcessRegistry()
    monkeypatch.setattr(pr, "process_registry", registry)
    session = ProcessSession(
        id="proc_live", command="echo live", task_id="parent", started_at=time.time(),
        output_buffer="ok\n", exited=True, exit_code=0, parent_session_id="parent-db",
    )
    registry._finished[session.id] = session
    runner = object.__new__(GatewayRunner)
    seen = {}

    async def _enqueue(_text, evt):
        seen.update(evt)
        return True

    runner._enqueue_process_completion_notification = _enqueue
    runner._load_background_notifications_mode = lambda: "concise"
    runner._completion_event_scope = lambda _evt: __import__("contextlib").nullcontext()
    runner._launching_turn_active = lambda *_a, **_k: __import__("asyncio").sleep(0, result=False)

    async def _go():
        await runner._run_process_watcher({
            "session_id": session.id, "check_interval": 0, "notify_on_complete": True,
            "parent_session_id": "child-db", "platform": "", "chat_id": "",
        })

    asyncio.run(_go())
    assert seen["parent_session_id"] == "parent-db"


def test_delayed_checkpoint_does_not_overwrite_newer_owner(tmp_path, monkeypatch):
    import tools.process_registry as pr
    from tools.process_registry import ProcessSession

    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(pr, "CHECKPOINT_PATH", checkpoint)
    registry = pr.ProcessRegistry()
    session = ProcessSession(
        id="proc_gen", command="echo gen", task_id="child", started_at=time.time(),
        owner_task_id="sa-middle",
    )
    registry._running[session.id] = session
    real = registry._write_checkpoint
    started, release = threading.Event(), threading.Event()

    def slow(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(registry, "_write_checkpoint", slow)
    worker = threading.Thread(target=registry._write_checkpoint, daemon=True)
    worker.start()
    assert started.wait(5)
    monkeypatch.setattr(registry, "_write_checkpoint", real)
    session.owner_task_id = "grand"
    registry._write_checkpoint()
    release.set()
    worker.join(5)
    stored = json.loads(checkpoint.read_text())
    assert stored[0]["owner_task_id"] == "grand"


def test_drain_keeps_child_error_and_handoff():
    from tools.process_registry import ProcessRegistry
    registry = ProcessRegistry()
    for evt in (
        {"type": "completion", "session_id": "proc_err", "owner_task_id": "sa-child",
         "command": "boom", "exit_code": 1, "completion_reason": "exited",
         "delegated_child": True, "started_at": 1.0, "output": "nope"},
        {"type": "completion", "session_id": "proc_hand", "owner_task_id": "sa-middle",
         "command": "keep", "exit_code": 0, "completion_reason": "exited",
         "delegated_child": True, "handoff_note": "watch this", "started_at": 1.0, "output": "ok"},
    ):
        registry.completion_queue.put(evt)
    rows = registry.drain_notifications()
    assert len(rows) == 2
    assert "nope" in rows[0][1]
    assert "watch this" in rows[1][1]


def test_native_spawn_stamps_only_in_process_child(tmp_path, monkeypatch):
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER, delegated_child_context
    from tools.process_registry import ProcessRegistry

    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, str(tmp_path))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_SESSION", raising=False)
    registry = ProcessRegistry()
    forged = registry._spawn_env({})
    assert "HERMES_DELEGATED_CHILD_SESSION" not in forged
    script = textwrap.dedent(
        """
        import os, sys
        sys.path.insert(0, os.environ["SRC"])
        from agent.delegation_context import delegated_child_context
        from tools.process_registry import ProcessRegistry
        with delegated_child_context("child-session"):
            env = ProcessRegistry._spawn_env({})
        print(env.get("HERMES_DELEGATED_CHILD_SESSION", ""))
        """
    )
    child = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "SRC": os.getcwd(), "HERMES_HOME": str(tmp_path)},
        cwd=os.getcwd(), capture_output=True, text=True, check=False,
    )
    # In-process stamp uses the session ContextVar set by delegated_child_context.
    with delegated_child_context("child-session"):
        stamped = registry._spawn_env({})
    assert stamped.get("HERMES_DELEGATED_CHILD_SESSION")
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip()
    outside = registry._spawn_env({})
    # No in-process ContextVar means no new stamp is minted.
    assert outside.get("HERMES_DELEGATED_CHILD_SESSION") != "child-session"
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_SESSION", "forged")
    again = registry._spawn_env({})
    assert again.get("HERMES_DELEGATED_CHILD_SESSION") != "child-session"
