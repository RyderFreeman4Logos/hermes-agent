"""Regression test for /reload-mcp refreshing cached agent tool lists.

Before this fix, the gateway's _execute_mcp_reload reconnected MCP servers
and updated the global _servers registry, but cached AIAgent instances kept
their original tools list. Users had to run /new (discarding conversation
history) for the agent to pick up the new tools.

This test exercises _execute_mcp_reload directly with mocked MCP discovery
and asserts that every cached agent's `tools` and `valid_tool_names`
attributes are overwritten with the freshly-discovered tool set.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event() -> MessageEvent:
    return MessageEvent(text="/reload-mcp", source=_make_source(), message_id="m1")


def _make_runner_with_cached_agents(num_agents: int = 2):
    """Build a bare GatewayRunner with `num_agents` fake cached agents."""
    import threading

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )

    # Session store stub — _execute_mcp_reload writes a transcript message
    # at the end; tests don't care about that side effect.
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.append_to_transcript = MagicMock()

    # Build N fake cached agents with stale `tools` + `valid_tool_names`.
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    for i in range(num_agents):
        stale_tool = {
            "type": "function",
            "function": {"name": f"stale_tool_{i}", "description": "old"},
        }
        agent = SimpleNamespace(
            tools=[stale_tool],
            valid_tool_names={f"stale_tool_{i}"},
            enabled_toolsets=None,
            disabled_toolsets=None,
        )
        runner._agent_cache[f"session-{i}"] = (agent, f"sig-{i}")

    return runner


@pytest.mark.asyncio
async def test_reload_mcp_refreshes_cached_agent_tools():
    """After /reload-mcp succeeds, every cached agent gets its tool list
    replaced with the freshly-discovered set."""
    runner = _make_runner_with_cached_agents(num_agents=3)

    # Snapshot the stale state so we can assert it changed.
    pre_reload_tools = {
        key: list(entry[0].tools) for key, entry in runner._agent_cache.items()
    }

    # Fresh tools that get_tool_definitions() will return after the reload.
    fresh_tool_defs = [
        {
            "type": "function",
            "function": {"name": "HassTurnOn", "description": "Turns on a device"},
        },
        {
            "type": "function",
            "function": {"name": "HassTurnOff", "description": "Turns off a device"},
        },
    ]

    with (
        patch("tools.mcp_tool.shutdown_mcp_servers"),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=["HassTurnOn", "HassTurnOff"]),
        patch.dict("tools.mcp_tool._servers", {"homeassistant": object()}, clear=True),
        patch("model_tools.get_tool_definitions", return_value=fresh_tool_defs),
    ):
        result = await runner._execute_mcp_reload(_make_event())

    # The reload itself returned a status string (not an exception).
    assert isinstance(result, str)

    # Every cached agent has fresh tools and the matching valid_tool_names.
    expected_names = {"HassTurnOn", "HassTurnOff"}
    for key, (agent, _sig) in runner._agent_cache.items():
        assert agent.tools == fresh_tool_defs, (
            f"Agent {key} kept stale tools: {agent.tools} != {fresh_tool_defs}"
        )
        assert agent.valid_tool_names == expected_names, (
            f"Agent {key} kept stale valid_tool_names: {agent.valid_tool_names}"
        )
        # Sanity check that the swap actually changed something.
        assert agent.tools != pre_reload_tools[key]


@pytest.mark.asyncio
async def test_reload_mcp_handles_empty_agent_cache():
    """Reload with no cached agents (e.g. fresh gateway) must not raise."""
    runner = _make_runner_with_cached_agents(num_agents=0)
    assert len(runner._agent_cache) == 0

    with (
        patch("tools.mcp_tool.shutdown_mcp_servers"),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
        patch.dict("tools.mcp_tool._servers", {}, clear=True),
        patch("model_tools.get_tool_definitions", return_value=[]),
    ):
        result = await runner._execute_mcp_reload(_make_event())

    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_reload_mcp_preserves_per_agent_toolset_overrides():
    """If a cached agent was built with enabled_toolsets=["safe"], the
    refresh must pass that same list to get_tool_definitions so the agent
    doesn't silently gain disabled tools after a reload."""
    runner = _make_runner_with_cached_agents(num_agents=1)
    # Override the toolsets on the cached agent.
    agent, _sig = runner._agent_cache["session-0"]
    agent.enabled_toolsets = ["safe"]
    agent.disabled_toolsets = ["terminal"]

    captured_calls = []

    def _capture_get_tool_definitions(**kwargs):
        captured_calls.append(kwargs)
        return [{"type": "function", "function": {"name": "refreshed"}}]

    with (
        patch("tools.mcp_tool.shutdown_mcp_servers"),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=["refreshed"]),
        patch.dict("tools.mcp_tool._servers", {"homeassistant": object()}, clear=True),
        patch("model_tools.get_tool_definitions", side_effect=_capture_get_tool_definitions),
    ):
        await runner._execute_mcp_reload(_make_event())

    assert captured_calls, "get_tool_definitions was never called to refresh the cache"
    assert captured_calls[0]["enabled_toolsets"] == ["safe"]
    assert captured_calls[0]["disabled_toolsets"] == ["terminal"]


@pytest.mark.parametrize("warning_handler_raises", [False, True])
@pytest.mark.asyncio
async def test_reload_mcp_continues_after_later_cached_agent_refresh_fails(
    warning_handler_raises,
):
    runner = _make_runner_with_cached_agents(num_agents=2)
    first = runner._agent_cache["session-0"][0]
    second = runner._agent_cache["session-1"][0]
    first._tool_snapshot_generation = 0
    second._tool_snapshot_generation = 0
    fresh_tools = [
        {"type": "function", "function": {"name": "fresh_tool", "description": "new"}}
    ]
    refresh_calls = 0
    warning_seen = threading.Event()

    class RefreshWarningHandler(logging.Handler):
        def emit(self, record):
            if record.getMessage() != "Cached agent MCP tool refresh failed":
                return
            warning_seen.set()
            if warning_handler_raises:
                raise RuntimeError("diagnostic handler failure")

    def _build_tools(**_kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 2:
            raise RuntimeError("pre-publication refresh failure marker")
        return fresh_tools

    gateway_logger = logging.getLogger("gateway.run")
    handler = RefreshWarningHandler()
    gateway_logger.addHandler(handler)
    try:
        with (
            patch("tools.mcp_tool.shutdown_mcp_servers"),
            patch("tools.mcp_tool.discover_mcp_tools", return_value=["fresh_tool"]),
            patch.dict("tools.mcp_tool._servers", {"fresh-server": object()}, clear=True),
            patch("model_tools.get_tool_definitions", side_effect=_build_tools),
            patch("gateway.run.t", side_effect=lambda key, **_kwargs: key),
        ):
            result = await runner._execute_mcp_reload(_make_event())
    finally:
        gateway_logger.removeHandler(handler)

    assert result != "gateway.reload_mcp.failed"
    assert first.valid_tool_names == {"fresh_tool"}
    assert first._tool_snapshot_generation == 1
    assert second.valid_tool_names == {"stale_tool_1"}
    assert second._tool_snapshot_generation == 0
    assert warning_seen.is_set()


@pytest.mark.asyncio
async def test_reload_lock_repeated_cancellation_releases_eventual_acquisition():
    class TrackingLock:
        def __init__(self):
            self._lock = threading.Lock()
            self.waiter_entered = threading.Event()
            self.waiter_acquired = threading.Event()
            self.release_called = threading.Event()

        def acquire(self):
            self.waiter_entered.set()
            acquired = self._lock.acquire()
            self.waiter_acquired.set()
            return acquired

        def release(self):
            self.release_called.set()
            self._lock.release()

        def locked(self):
            return self._lock.locked()

    runner = _make_runner_with_cached_agents(num_agents=0)
    lock = TrackingLock()
    lock._lock.acquire()
    loop = asyncio.get_running_loop()

    with patch("tools.mcp_tool._mcp_reload_lock", lock):
        task = asyncio.create_task(runner._execute_mcp_reload(_make_event()))
        assert await loop.run_in_executor(None, lock.waiter_entered.wait, 2)
        task.cancel()
        cancellation_processed = loop.create_future()
        loop.call_soon(cancellation_processed.set_result, None)
        await cancellation_processed
        released_other_owner = lock.release_called.is_set()
        task.cancel()
        lock._lock.release()
        assert await loop.run_in_executor(None, lock.waiter_acquired.wait, 2)
        assert await loop.run_in_executor(None, lock.release_called.wait, 2)
        with pytest.raises(asyncio.CancelledError):
            await task

    assert not released_other_owner
    assert lock.release_called.is_set()
    assert not lock.locked()
    assert lock._lock.acquire(blocking=False)
    lock._lock.release()
