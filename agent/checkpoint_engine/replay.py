"""Replay contracts for checkpoint validation.

Replay is an execution boundary, not a transcript lookup.  Missing adapters
fail closed instead of treating historical assistant prose as a tool result.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


class ReplayUnavailable(RuntimeError):
    pass


class ReplayAdapter:
    def __init__(self, tool_runners: Mapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]) -> None:
        self._runners = dict(tool_runners)

    def replay(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or ():
                if not isinstance(call, Mapping):
                    raise ReplayUnavailable("malformed tool call")
                function = call.get("function") or {}
                name = function.get("name") if isinstance(function, Mapping) else call.get("name")
                runner = self._runners.get(str(name))
                if runner is None:
                    raise ReplayUnavailable(f"no replay adapter for {name}")
                result = dict(runner(call))
                if not result.get("status"):
                    raise ReplayUnavailable(f"adapter {name} returned no status")
                output.append({"role": "tool", "tool_call_id": str(call.get("id", "")), **result})
        return output
