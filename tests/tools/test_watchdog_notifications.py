"""Regression coverage for pure watchdog terminal commands."""

import json
from types import SimpleNamespace

import tools.terminal_tool as terminal_tool


def _watchdog_harness(monkeypatch, tmp_path):
    config = {
        "env_type": "local",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "cwd": str(tmp_path),
        "timeout": 30,
    }
    proc = SimpleNamespace(
        id="proc_watchdog_test",
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
    dummy_env = SimpleNamespace(env={})

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setattr(
        "tools.process_registry.process_registry.spawn_local",
        lambda **_kwargs: proc,
    )
    monkeypatch.setitem(terminal_tool._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool._last_activity, "default", 0.0)
    return proc


def test_pure_watchdog_commands_are_detected_without_matching_real_work():
    assert terminal_tool._is_pure_watchdog_command(
        "/home/obj/.hermes/skills/watchdog/resolve-checkin.sh"
    )
    assert terminal_tool._is_pure_watchdog_command(
        "sleep $CHECKIN; echo 'HEARTBEAT after TTL'"
    )
    assert terminal_tool._is_pure_watchdog_command(
        "while true; do echo HEARTBEAT; sleep ${CHECKIN}; done"
    )
    assert not terminal_tool._is_pure_watchdog_command(
        "CHECKIN=$(resolve-checkin.sh); pytest tests/tools"
    )
    # A path segment named "git" is not the git command, so it remains a
    # resolver-only watchdog command.
    assert terminal_tool._is_pure_watchdog_command(
        "/tmp/git/resolve-checkin.sh"
    )


def test_pure_watchdog_background_forces_notify_off_and_returns_guidance(monkeypatch, tmp_path):
    proc = _watchdog_harness(monkeypatch, tmp_path)
    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                command="sleep $CHECKIN; echo 'HEARTBEAT after TTL'",
                background=True,
                notify_on_complete=True,
                watch_patterns=["HEARTBEAT"],
            )
        )
    finally:
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._last_activity.pop("default", None)

    assert proc.notify_on_complete is False
    assert result["session_id"] == "proc_watchdog_test"
    assert "notify_on_complete" not in result
    assert "watch_patterns" not in result
    assert "WD/TTL sleep" in result["watchdog_notice"]
    assert "wait for real task completion" in result["watchdog_notice"]


def test_terminal_description_discourages_resolve_checkin_loops():
    assert "DO NOT use resolve-checkin in a loop" in terminal_tool.TERMINAL_TOOL_DESCRIPTION
    assert "DO NOT notify on pure TTL sleep heartbeats" in terminal_tool.TERMINAL_TOOL_DESCRIPTION
