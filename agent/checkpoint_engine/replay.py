"""Fail-closed replay adapters for checkpoint continuation modes."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence


class ReplayUnavailable(RuntimeError):
    """A requested replay path has no real execution adapter."""


class ReplayAdapter:
    def __init__(self, tools: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None) -> None:
        self.tools = dict(tools or {})

    def replay(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        post_context = kwargs.get("post_context")
        if post_context:
            raise ReplayUnavailable("post-context fallback is not replay")
        results: list[dict[str, Any]] = []
        for message in messages:
            for call in message.get("tool_calls", ()):
                function = call.get("function", {})
                name = function.get("name")
                if name not in self.tools:
                    raise ReplayUnavailable(f"no real replay adapter for {name}")
                receipt = self.tools[name](call)
                result = dict(receipt) if isinstance(receipt, Mapping) else {"content": str(receipt)}
                result.update({"role": "tool", "tool_call_id": call.get("id")})
                results.append(result)
        return results


class CheckpointReplayAdapter(ReplayAdapter):
    pass


class LeanReplayAdapter(ReplayAdapter):
    pass


class RecordedOriginalAdapter(ReplayAdapter):
    pass


class ContinuationReplayAdapter(ReplayAdapter):
    def __init__(
        self,
        tools: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None,
        *,
        continuation: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        super().__init__(tools)
        self.continuation = continuation

    def replay(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        replayed = super().replay(messages, **kwargs)
        context = kwargs.get("continuation_context")
        if self.continuation is None or context is None:
            raise ReplayUnavailable("no frozen continuation adapter")
        return replayed + [dict(message) for message in self.continuation(context)]


class ReplayRunner:
    def __init__(self, adapter: ReplayAdapter) -> None:
        self.adapter = adapter

    def run(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        return self.adapter.replay(messages, **kwargs)
