"""Tests for notify_on_complete background process feature.

Covers:
  - ProcessSession.notify_on_complete field
  - ProcessRegistry.completion_queue population on _move_to_finished()
  - Checkpoint persistence of notify_on_complete
  - Terminal tool schema includes notify_on_complete
  - Terminal tool handler passes notify_on_complete through
"""

import json
import os
import time
import pytest
from unittest.mock import MagicMock, patch

from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


def _make_session(
    sid="proc_test_notify",
    command="echo hello",
    task_id="t1",
    exited=False,
    exit_code=None,
    output="",
    notify_on_complete=False,
) -> ProcessSession:
    s = ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=time.time(),
        exited=exited,
        exit_code=exit_code,
        output_buffer=output,
        notify_on_complete=notify_on_complete,
    )
    return s


# =========================================================================
# ProcessSession field
# =========================================================================

class TestProcessSessionField:
    def test_default_false(self):
        s = ProcessSession(id="proc_1", command="echo hi")
        assert s.notify_on_complete is False

    def test_set_true(self):
        s = ProcessSession(id="proc_1", command="echo hi", notify_on_complete=True)
        assert s.notify_on_complete is True


# =========================================================================
# Completion queue
# =========================================================================

class TestCompletionQueue:
    def test_queue_exists(self, registry):
        assert hasattr(registry, "completion_queue")
        assert registry.completion_queue.empty()


    def test_move_to_finished_idempotent_no_duplicate(self, registry):
        """Calling _move_to_finished twice must NOT enqueue two notifications.

        Regression test: kill_process() and the reader thread can both call
        _move_to_finished() for the same session, producing duplicate
        [SYSTEM: Background process ...] messages.
        """
        s = _make_session(notify_on_complete=True, output="done", exit_code=-15)
        s.exited = True
        s.exit_code = -15
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)  # first call — should enqueue
            s.exit_code = 143  # reader thread updates exit code
            registry._move_to_finished(s)  # second call — should be no-op

        assert registry.completion_queue.qsize() == 1
        completion = registry.completion_queue.get_nowait()
        assert completion["exit_code"] == -15  # from the first (kill) call


    def test_output_truncated_to_2000(self, registry):
        """Long output is truncated to last 2000 chars."""
        long_output = "x" * 5000
        s = _make_session(
            notify_on_complete=True,
            output=long_output,
        )
        s.exited = True
        s.exit_code = 0
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)

        completion = registry.completion_queue.get_nowait()
        assert len(completion["output"]) == 2000

    def test_multiple_completions_queued(self, registry):
        """Multiple notify processes all push to the same queue."""
        for i in range(3):
            s = _make_session(
                sid=f"proc_{i}",
                notify_on_complete=True,
                output=f"output_{i}",
            )
            s.exited = True
            s.exit_code = 0
            registry._running[s.id] = s
            with patch.object(registry, "_write_checkpoint"):
                registry._move_to_finished(s)

        completions = []
        while not registry.completion_queue.empty():
            completions.append(registry.completion_queue.get_nowait())
        assert len(completions) == 3
        ids = {c["session_id"] for c in completions}
        assert ids == {"proc_0", "proc_1", "proc_2"}

    def test_spawn_local_queues_notify_when_reader_finishes_immediately(
        self, registry, monkeypatch
    ):
        """A spawn-advertised notification survives an immediate reader exit."""
        import tools.process_registry as pr

        class ImmediateThread:
            def __init__(self, *, target, args, **_kwargs):
                self.target = target
                self.args = args

            def start(self):
                self.target(*self.args)

        def finish(session):
            session.exited = True
            session.exit_code = 0
            session.completion_reason = "exited"
            registry._move_to_finished(session)

        monkeypatch.setattr(pr.subprocess, "Popen", lambda *_args, **_kwargs: MagicMock(pid=123))
        monkeypatch.setattr(pr.threading, "Thread", ImmediateThread)
        monkeypatch.setattr(registry, "_reader_loop", finish)
        monkeypatch.setattr(registry, "_safe_host_start_time", lambda _pid: None)
        monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

        session = registry.spawn_local("true", cwd="/tmp", notify_on_complete=True)

        assert registry.completion_queue.get_nowait()["session_id"] == session.id


# =========================================================================
# Checkpoint persistence
# =========================================================================

class TestCheckpointNotify:
    def test_checkpoint_includes_notify(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            s = _make_session(notify_on_complete=True)
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads((tmp_path / "procs.json").read_text())
            assert len(data) == 1
            assert data[0]["notify_on_complete"] is True


    def test_recover_defaults_false(self, registry, tmp_path):
        """Old checkpoint entries without the field default to False."""
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            s = registry.get("proc_live")
            assert s.notify_on_complete is False


# =========================================================================
# Terminal tool schema
# =========================================================================

class TestTerminalSchema:
    def test_schema_has_notify_on_complete(self):
        from tools.terminal_tool import TERMINAL_SCHEMA
        props = TERMINAL_SCHEMA["parameters"]["properties"]
        assert "notify_on_complete" in props
        assert props["notify_on_complete"]["type"] == "boolean"
        assert props["notify_on_complete"]["default"] is False

    def test_handler_passes_notify(self):
        """_handle_terminal passes notify_on_complete to terminal_tool."""
        from tools.terminal_tool import _handle_terminal
        with patch("tools.terminal_tool.terminal_tool", return_value='{"ok":true}') as mock_tt:
            _handle_terminal(
                {"command": "echo hi", "background": True, "notify_on_complete": True},
                task_id="t1",
            )
            _, kwargs = mock_tt.call_args
            assert kwargs["notify_on_complete"] is True


# =========================================================================
# Code execution blocked params
# =========================================================================

class TestCodeExecutionBlocked:
    def test_notify_on_complete_blocked_in_sandbox(self):
        from tools.code_execution_tool import _TERMINAL_BLOCKED_PARAMS
        assert "notify_on_complete" in _TERMINAL_BLOCKED_PARAMS


# =========================================================================
# Completion consumed suppression
# =========================================================================

class TestCompletionConsumed:
    """Test that wait/log consume completion notifications while poll stays read-only."""

    def test_wait_marks_completion_consumed(self, registry):
        """wait() returning exited status marks session as consumed."""
        s = _make_session(sid="proc_wait", notify_on_complete=True, output="done")
        s.exited = True
        s.exit_code = 0
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)

        # Notification is in the queue
        assert not registry.completion_queue.empty()
        assert not registry.is_completion_consumed("proc_wait")

        # Agent calls wait() — gets the result directly
        result = registry.wait("proc_wait", timeout=1)
        assert result["status"] == "exited"

        # Now the completion is marked as consumed
        assert registry.is_completion_consumed("proc_wait")


    def test_poll_observed_does_not_suppress_gateway_watcher(self, registry):
        """The gateway/tui watcher gate (is_completion_consumed) must stay False
        after a read-only poll, so the autonomous delivery turn still fires
        even though the CLI drain was deduped (#10156)."""
        s = _make_session(sid="proc_gw", notify_on_complete=True, output="done")
        s.exited = True
        s.exit_code = 0
        registry._finished[s.id] = s

        registry.poll("proc_gw")
        # CLI-side dedup signal present...
        assert "proc_gw" in registry._poll_observed
        # ...but the gateway watcher gate is untouched, so it still delivers.
        assert not registry.is_completion_consumed("proc_gw")

    def test_running_poll_does_not_mark_poll_observed(self, registry):
        """poll() on a still-running process must not record _poll_observed."""
        s = _make_session(sid="proc_run2", notify_on_complete=True, output="partial")
        registry._running[s.id] = s

        registry.poll("proc_run2")
        assert "proc_run2" not in registry._poll_observed

    def test_wait_and_log_still_skip_cli_drain(self, registry):
        """wait()/read_log() consume the output, so the CLI drain skips their
        completions via _completion_consumed (the original #8228 contract)."""
        for sid, action in (("proc_w", "wait"), ("proc_l", "log")):
            s = _make_session(sid=sid, notify_on_complete=True, output="done")
            s.exited = True
            s.exit_code = 0
            registry._running[s.id] = s
            with patch.object(registry, "_write_checkpoint"):
                registry._move_to_finished(s)
            if action == "wait":
                registry.wait(sid, timeout=1)
            else:
                registry.read_log(sid)
            assert registry.is_completion_consumed(sid)
        assert registry.drain_notifications() == []


# ---------------------------------------------------------------------------
# Silent-background-process hint
#
# background=True without notify_on_complete=True OR watch_patterns runs
# the process silently — the agent has no way to learn it finished short
# of calling process(action="poll") explicitly. The tool result must
# include a "hint" field that nudges the agent toward
# notify_on_complete=True for bounded tasks. May 2026 PR #31231 incident:
# bg CI poller exited green, agent never noticed, user had to surface it.
# ---------------------------------------------------------------------------


def _silent_bg_base_config(tmp_path):
    return {
        "env_type": "local",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "cwd": str(tmp_path),
        "timeout": 30,
        "auto_background_timeout_threshold": 200,
    }


def _silent_bg_harness(monkeypatch, tmp_path):
    """Common test fixture: patch enough of terminal_tool to spawn a fake
    background process and capture the JSON result the agent sees."""
    import tools.terminal_tool as terminal_tool_module
    from tools import process_registry as process_registry_module
    from types import SimpleNamespace

    config = _silent_bg_base_config(tmp_path)
    spawn_kwargs = []
    dummy_env = SimpleNamespace(
        env={},
        execute=MagicMock(
            return_value={"output": "done", "exit_code": 0, "error": None}
        ),
    )
    def fake_spawn_local(**kwargs):
        spawn_kwargs.append(kwargs)
        return SimpleNamespace(
            id="proc_silent_test",
            pid=4242,
            notify_on_complete=False,
            watcher_platform="",
            watcher_chat_id="",
            watcher_user_id="",
            watcher_user_name="",
            watcher_thread_id="",
            watcher_message_id="",
            watcher_interval=0,
        )

    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool_module, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setattr(process_registry_module.process_registry, "spawn_local", fake_spawn_local)
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)
    terminal_tool_module._test_env = dummy_env
    setattr(terminal_tool_module, "_test_spawn_kwargs", spawn_kwargs)
    return terminal_tool_module


def _call_terminal_handler(monkeypatch, tmp_path, **args):
    from gateway import session_context

    tt = _silent_bg_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: True)
    try:
        result = json.loads(tt._handle_terminal({"command": "printf done", **args}))
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)
    return tt, result


def test_omitted_background_and_timeout_auto_promote_with_notify(monkeypatch, tmp_path):
    _tt, result = _call_terminal_handler(monkeypatch, tmp_path)

    assert result["session_id"] == "proc_silent_test"
    assert result["notify_on_complete"] is True


@pytest.mark.parametrize(
    "args",
    [{}, {"background": True, "notify_on_complete": True}],
)
def test_notify_is_armed_at_spawn(monkeypatch, tmp_path, args):
    tt, _result = _call_terminal_handler(monkeypatch, tmp_path, **args)

    assert getattr(tt, "_test_spawn_kwargs")[-1]["notify_on_complete"] is True


def test_omitted_background_long_timeout_auto_promotes_before_cap(monkeypatch, tmp_path):
    _tt, result = _call_terminal_handler(monkeypatch, tmp_path, timeout=7200)

    assert result["session_id"] == "proc_silent_test"
    assert result["notify_on_complete"] is True
    assert "Foreground timeout" not in str(result.get("error"))


def test_auto_promotion_forces_explicit_notify_false_to_true(monkeypatch, tmp_path):
    _tt, result = _call_terminal_handler(
        monkeypatch, tmp_path, timeout=7200, notify_on_complete=False
    )

    assert result["session_id"] == "proc_silent_test"
    assert result["notify_on_complete"] is True


def test_explicit_background_false_long_timeout_still_rejected(monkeypatch, tmp_path):
    _tt, result = _call_terminal_handler(
        monkeypatch, tmp_path, timeout=7200, background=False
    )

    assert "Foreground timeout 7200s exceeds the maximum" in result["error"]


def test_omitted_background_short_explicit_timeout_stays_foreground(monkeypatch, tmp_path):
    tt, result = _call_terminal_handler(monkeypatch, tmp_path, timeout=30)

    assert result["error"] is None
    tt._test_env.execute.assert_called_once()


def test_background_without_notify_emits_silent_process_hint(monkeypatch, tmp_path):
    """The footgun case (May 2026 PR #31231): bg=True alone runs silently
    and the agent has no signal it finished. Tool must nudge."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    try:
        result = json.loads(
            tt.terminal_tool(
                command="while true; do gh pr checks 999; sleep 30; done",
                background=True,
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert result["session_id"] == "proc_silent_test"
    hint = result.get("hint", "")
    assert hint, "Silent background process must include a hint field"
    assert "notify_on_complete" in hint, (
        "Hint must name the corrective flag so the agent can self-correct"
    )
    assert "silent" in hint.lower() or "no way to learn" in hint.lower(), (
        "Hint must explain the failure mode, not just suggest the fix"
    )


def test_background_notify_works_without_runtime_heartbeat_or_hint(monkeypatch, tmp_path):
    """Completion notification does not depend on runtime-heartbeat setup."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    try:
        result = json.loads(
            tt.terminal_tool(
                command="pytest tests/",
                background=True,
                notify_on_complete=True,
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert "hint" not in result, (
        f"Correct usage must not emit a hint, got: {result.get('hint')!r}"
    )
    assert result.get("notify_on_complete") is True


def test_foreground_command_does_not_emit_hint(monkeypatch, tmp_path):
    """Hint only applies to background processes — foreground returns its
    result synchronously and the agent always sees the outcome."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)

    # Foreground path doesn't go through spawn_local. Patch the local-env
    # exec method to short-circuit to a clean exit so the test doesn't
    # actually shell out.
    from types import SimpleNamespace
    dummy_env = SimpleNamespace(
        env={},
        execute=lambda *a, **kw: {"output": "done", "exit_code": 0, "error": None},
    )
    monkeypatch.setitem(tt._active_environments, "default", dummy_env)

    try:
        result = json.loads(
            tt.terminal_tool(
                command="echo hello",
                background=False,
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert "hint" not in result, (
        f"Foreground commands must not emit the background-silence hint, got: {result.get('hint')!r}"
    )


# ---------------------------------------------------------------------------
# Homebrewed-CI-watcher hint
#
# Background processes whose command looks like a hand-rolled CI poller
# (`gh pr view` / `gh pr checks` combined with jq/awk on stdout) get an
# additional hint pointing at the canonical green-ci-policy snippet. The
# homebrew shape has burned us repeatedly (May 2026 PRs #31329, #31448,
# #31695, #31709, #31745, #32264, #33131) with stdout buffering, jq null
# keys, conclusion-vs-status confusion, and TTY-only banner grepping —
# none of which the canonical snippets suffer from. Fire on every detection;
# false positives are cheap (~one read).
# ---------------------------------------------------------------------------


def test_non_ci_background_command_does_not_emit_homebrew_hint(monkeypatch, tmp_path):
    """A long-running task that happens to use awk for unrelated reasons
    must not be mistaken for a CI poller — the gating signal is the
    combination of `gh pr ...` AND a stdout parser."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    try:
        result = json.loads(
            tt.terminal_tool(
                command="cat /var/log/syslog | awk '/error/ {print}' > /tmp/errs.log",
                background=True,
                notify_on_complete=True,
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert "hint" not in result, (
        f"Non-CI command using awk must not be flagged as homebrew CI poller, got: {result.get('hint')!r}"
    )
