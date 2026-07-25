"""Regression coverage for the resolve-checkin idle-spin breaker."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_definitions(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _terminal_call(command: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name="terminal",
            arguments=json.dumps({"command": command}),
        ),
    )


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_definitions("terminal")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        instance.client = MagicMock()
        return instance


def test_third_identical_resolve_checkin_skips_terminal_handler(agent):
    """Two identical interval resolutions are enough; the third must be local."""
    command = "/home/obj/.hermes/skills/watchdog/resolve-checkin.sh"
    calls = []

    def fake_handle(name, args, task_id, **kwargs):
        calls.append((name, args, task_id, kwargs["tool_call_id"]))
        return json.dumps({"output": "CHECKIN=270", "exit_code": 0, "error": None})

    messages = []
    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        for index in range(3):
            agent._execute_tool_calls(
                SimpleNamespace(
                    content="",
                    tool_calls=[_terminal_call(command, f"resolver-{index}")],
                ),
                messages,
                "task-1",
            )

    assert len(calls) == 2, "The third identical resolver call must not reach terminal"
    assert "CHECKIN=270" in messages[-1]["content"]
    assert "DO NOT call resolve-checkin again" in messages[-1]["content"]
    assert "built-in runtime heartbeat" in messages[-1]["content"]


def test_real_terminal_progress_resets_resolve_checkin_loop_state(agent):
    """A non-resolver terminal command is progress, so a later resolver may run."""
    resolver = "/home/obj/.hermes/skills/watchdog/resolve-checkin.sh"
    calls = []

    def fake_handle(name, args, task_id, **kwargs):
        calls.append(args["command"])
        output = "CHECKIN=270" if "resolve-checkin" in args["command"] else "progress made"
        return json.dumps({"output": output, "exit_code": 0, "error": None})

    messages = []
    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        for index, command in enumerate((resolver, resolver, "echo progress", resolver)):
            agent._execute_tool_calls(
                SimpleNamespace(
                    content="",
                    tool_calls=[_terminal_call(command, f"call-{index}")],
                ),
                messages,
                "task-1",
            )

    assert calls == [resolver, resolver, "echo progress", resolver]


def test_resolve_checkin_fingerprint_includes_normalized_command_arguments():
    """Resolvers for different targets must not share one idle-spin streak."""
    first = "/skills/watchdog/resolve-checkin.sh --target worker-a --pid 101"
    equivalent_spacing = "  /skills/watchdog/resolve-checkin.sh   --target worker-a --pid 101  "
    second = "/skills/watchdog/resolve-checkin.sh --target worker-b --pid 202"
    fingerprint = getattr(AIAgent, "_resolve_checkin_fingerprint")

    first_fingerprint = fingerprint("terminal", {"command": first})
    assert first_fingerprint == fingerprint(
        "terminal", {"command": equivalent_spacing}
    )
    assert first_fingerprint != fingerprint(
        "terminal", {"command": second}
    )
