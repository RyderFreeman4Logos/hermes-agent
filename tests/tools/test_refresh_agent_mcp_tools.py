"""Tests for the shared MCP agent-tool refresh helper and discovery-wait bound.

``refresh_agent_mcp_tools`` is the single rebuild path used by the TUI
``reload.mcp`` RPC, the gateway reload, and the late-binding refresh thread —
so a slow MCP server that connects after the agent's one-time tool snapshot is
picked up everywhere identically.  These assert the *contracts* those callers
rely on (name-based diff, in-place mutation, agent-scoped filtering) rather than
freezing any particular tool list.
"""

import threading
import types

import pytest

from tools import mcp_tool


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _agent(tool_names, *, enabled=None, disabled=None):
    a = types.SimpleNamespace()
    a.tools = [_tool(n) for n in tool_names]
    a.valid_tool_names = set(tool_names)
    a.enabled_toolsets = enabled
    a.disabled_toolsets = disabled
    a._tool_snapshot_generation = 0
    return a


def test_refresh_adds_late_landing_tools(monkeypatch):
    """A server that registers after build → its tools land in the snapshot."""
    agent = _agent(["read_file", "terminal"])

    new_defs = [_tool(n) for n in ("read_file", "terminal", "mcp_granola_get_account_info")]
    monkeypatch.setattr(mcp_tool, "get_tool_definitions", lambda **kw: new_defs, raising=False)
    # get_tool_definitions is imported inside the helper from model_tools, so patch there too.
    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: new_defs)

    added = mcp_tool.refresh_agent_mcp_tools(agent)

    assert added == {"mcp_granola_get_account_info"}
    assert "mcp_granola_get_account_info" in agent.valid_tool_names
    assert len(agent.tools) == 3


def test_public_refresh_reinjection_failure_logs_are_static(monkeypatch):
    from unittest.mock import MagicMock, patch

    import model_tools
    from tools import bot_mode_dm

    marker = "reinject-credential-exception-marker"
    agent = _agent(["old_tool"])
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: (_ for _ in ()).throw(RuntimeError(marker))
    )
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: (_ for _ in ()).throw(RuntimeError(marker))
    )
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_kwargs: [_tool("new_tool")])
    monkeypatch.setattr(
        bot_mode_dm,
        "ensure_message_agent_tool",
        lambda _agent: (_ for _ in ()).throw(RuntimeError(marker)),
    )
    logger = MagicMock()

    with patch.object(mcp_tool, "logger", logger):
        assert mcp_tool.refresh_agent_mcp_tools(agent) == {"new_tool"}

    assert agent.valid_tool_names == {"new_tool"}
    assert agent._tool_snapshot_generation == 1
    calls = [
        call
        for method in (logger.debug, logger.info, logger.warning, logger.error)
        for call in method.call_args_list
    ]
    assert len(calls) == 3
    assert all(len(call.args) == 1 and call.kwargs == {} for call in calls)
    assert marker not in repr(calls)


def test_refresh_preserves_memory_provider_and_context_engine_tools(monkeypatch):
    """B1 regression: a rebuild must NOT drop post-build-injected tools.

    get_tool_definitions() returns only the registry-derived tools. agent_init
    appends memory-provider tools (mem0/honcho/…) and context-engine tools
    (lcm_*) directly onto agent.tools AFTER that. A naive
    `agent.tools = get_tool_definitions()` would silently delete them on every
    refresh. The helper must re-inject them.
    """
    # Agent already carries: a built-in, a memory-provider tool, a context tool.
    agent = _agent(["read_file", "memory_search", "lcm_grep"])

    # Provider exposes its schemas; context compressor exposes lcm_*.
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "memory_search", "description": "", "parameters": {}}
        ]
    )
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [
            {"name": "lcm_grep", "description": "", "parameters": {}}
        ]
    )
    agent._context_engine_tool_names = {"lcm_grep"}

    import model_tools
    # The registry now ALSO has a newly-connected MCP tool, but does NOT contain
    # the memory/context tools (they're never in get_tool_definitions output).
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("mcp_new_server_tool")],
    )

    added = mcp_tool.refresh_agent_mcp_tools(agent)

    # The new MCP tool landed AND the injected families survived.
    assert "mcp_new_server_tool" in agent.valid_tool_names
    assert "memory_search" in agent.valid_tool_names   # not clobbered
    assert "lcm_grep" in agent.valid_tool_names         # not clobbered
    assert added == {"mcp_new_server_tool"}


def test_refresh_does_not_reinject_disabled_memory_provider_tools(monkeypatch):
    """A refresh removes stale provider tools when memory becomes disabled."""
    agent = _agent(
        ["read_file", "memory_search"],
        enabled=["all"],
        disabled=["memory"],
    )
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "memory_search", "description": "", "parameters": {}}
        ]
    )

    import model_tools
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kw: [_tool("read_file")],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert "memory_search" not in agent.valid_tool_names
    assert all(t["function"]["name"] != "memory_search" for t in agent.tools)


def test_refresh_respects_context_engine_toolset_gate(monkeypatch):
    """#5544: context-engine tools must NOT be re-injected on a restricted
    toolset. A platform with enabled_toolsets that excludes context_engine
    must not get lcm_* leaked back in by a refresh."""
    agent = _agent(["read_file"], enabled=["coding"])  # context_engine NOT enabled
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [{"name": "lcm_grep", "description": "", "parameters": {}}]
    )
    agent._context_engine_tool_names = set()

    import model_tools
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("mcp_new_tool")],
    )

    mcp_tool.refresh_agent_mcp_tools(agent)

    assert "mcp_new_tool" in agent.valid_tool_names  # MCP tool still lands
    assert "lcm_grep" not in agent.valid_tool_names   # gated out (#5544)


def test_refreshed_tool_is_callable_through_valid_tool_names_guard(monkeypatch):
    """The whole point: a late tool, once refreshed, passes the name guard the
    run loop uses to accept/reject tool calls (agent.valid_tool_names)."""
    agent = _agent(["read_file"])

    import model_tools
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("mcp_granola_list_meetings")],
    )

    # Before refresh the run loop would reject the call ("Tool does not exist").
    assert "mcp_granola_list_meetings" not in agent.valid_tool_names

    mcp_tool.refresh_agent_mcp_tools(agent)

    # After refresh the same guard accepts it AND it's in the tools= payload.
    assert "mcp_granola_list_meetings" in agent.valid_tool_names
    assert any(t["function"]["name"] == "mcp_granola_list_meetings" for t in agent.tools)


def test_refresh_is_thread_safe_under_concurrent_calls(monkeypatch):
    """Concurrent refreshes keep tools / valid_tool_names coherent.

    The registry alternates between two DIFFERENT tool sets every call, so the
    write path (publish) runs repeatedly rather than short-circuiting on the
    no-change early return — this actually exercises the lock. The invariant:
    a reader of ``valid_tool_names`` must always match ``agent.tools``, and the
    final published pair must be one of the two valid sets (never a mix).
    """
    agent = _agent(["a"])

    import itertools
    set_a = [_tool("a"), _tool("b")]
    set_b = [_tool("a"), _tool("c")]
    flip = itertools.cycle([set_a, set_b])
    flip_lock = threading.Lock()

    def _gtd(**kw):
        with flip_lock:
            return list(next(flip))

    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions", _gtd)

    errors = []

    def _worker():
        try:
            for _ in range(50):
                mcp_tool.refresh_agent_mcp_tools(agent)
                # Coherence invariant: the name set must match the tool list
                # at every observation, never a torn cross-attribute state.
                names = {t["function"]["name"] for t in agent.tools}
                assert agent.valid_tool_names == names
                assert names in ({"a", "b"}, {"a", "c"})
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert agent.valid_tool_names in ({"a", "b"}, {"a", "c"})


def test_registry_change_during_slow_build_retries_without_stale_publish(monkeypatch):
    from tools.registry import registry
    import model_tools

    agent = _agent(["old"])
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    start_generation = registry._generation

    def _build(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=5)
            return [_tool("stale")]
        return [_tool("fresh")]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _build)
    worker = threading.Thread(target=mcp_tool.refresh_agent_mcp_tools, args=(agent,))
    worker.start()
    assert entered.wait(timeout=2)
    with registry._lock:
        registry._generation += 1
    release.set()
    worker.join(timeout=5)
    with registry._lock:
        registry._generation = start_generation

    assert not worker.is_alive()
    assert agent.valid_tool_names == {"fresh"}
    assert agent._tool_snapshot_generation == 1


def test_agent_epoch_fences_aba_builder(monkeypatch):
    import model_tools

    agent = _agent(["old"])
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def _build(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=5)
            return [_tool("stale")]
        return [_tool("winner")]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _build)
    worker = threading.Thread(target=mcp_tool.refresh_agent_mcp_tools, args=(agent,))
    worker.start()
    assert entered.wait(timeout=2)
    with mcp_tool._agent_tools_lock:
        agent.tools = [_tool("winner")]
        agent.valid_tool_names = {"winner"}
        agent._tool_snapshot_generation += 2
    release.set()
    worker.join(timeout=5)

    assert agent.valid_tool_names == {"winner"}
    assert agent._tool_snapshot_generation == 3


def test_same_name_schema_change_commits_and_invalidates(monkeypatch):
    import model_tools

    agent = _agent(["same"])
    agent.tools[0]["function"]["description"] = "old"
    invalidations = []
    agent._invalidate_system_prompt = lambda: invalidations.append(True)
    changed = _tool("same")
    changed["function"]["description"] = "new"
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_kw: [changed])

    added = mcp_tool.refresh_agent_mcp_tools(agent)

    assert added == set()
    assert agent.tools[0]["function"]["description"] == "new"
    assert invalidations == [True]
    assert agent._tool_snapshot_generation == 1


def test_invalidator_failure_does_not_fail_after_snapshot_publication(monkeypatch):
    import model_tools

    agent = _agent(["old"])
    agent._invalidate_system_prompt = lambda: (_ for _ in ()).throw(
        RuntimeError("secret invalidation failure")
    )
    monkeypatch.setattr(
        model_tools, "get_tool_definitions", lambda **_kw: [_tool("new")]
    )

    assert mcp_tool.refresh_agent_mcp_tools(agent, raise_on_exhaustion=True) == {"new"}
    assert agent.valid_tool_names == {"new"}
    assert agent._tool_snapshot_generation == 1


@pytest.mark.parametrize("failing_tail", ["history", "success_output"])
def test_cli_post_refresh_tails_cannot_report_failure(monkeypatch, failing_tail):
    import model_tools
    from cli import HermesCLI

    agent = _agent(["old"], enabled=[])
    cli = object.__new__(HermesCLI)
    cli.agent = agent
    cli.enabled_toolsets = []
    cli._command_running = True
    printed = []

    class _History(list):
        def append(self, item):
            if failing_tail == "history":
                raise RuntimeError("conversation tail unavailable")
            super().append(item)

    def _print(message, *_args, **_kwargs):
        if failing_tail == "success_output" and "Agent updated" in str(message):
            raise BrokenPipeError("stdout unavailable")
        printed.append(str(message))

    cli.conversation_history = _History()
    monkeypatch.setattr("builtins.print", _print)
    monkeypatch.setattr(mcp_tool, "shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", lambda: [])
    monkeypatch.setattr(mcp_tool, "_servers", {})
    monkeypatch.setattr(mcp_tool, "_lock", threading.Lock())
    monkeypatch.setattr(model_tools, "_last_resolved_tool_names", ["old"])
    monkeypatch.setattr(
        model_tools, "get_tool_definitions", lambda **_kwargs: [_tool("new")]
    )

    cli._reload_mcp()

    assert agent.valid_tool_names == {"new"}
    assert agent._tool_snapshot_generation == 1
    assert model_tools._last_resolved_tool_names == ["new"]
    assert not any("MCP reload failed" in line for line in printed)


def test_retry_exhaustion_publishes_nothing_and_keeps_compatibility_global(monkeypatch):
    from tools.registry import registry
    import model_tools

    agent = _agent(["winner"])
    model_tools._last_resolved_tool_names = ["winner"]
    start_generation = registry._generation

    def _always_stale(**_kwargs):
        with registry._lock:
            registry._generation += 1
        return [_tool("loser")]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _always_stale)
    try:
        assert mcp_tool.refresh_agent_mcp_tools(agent) == set()
    finally:
        with registry._lock:
            registry._generation = start_generation

    assert agent.valid_tool_names == {"winner"}
    assert agent._tool_snapshot_generation == 0
    assert model_tools._last_resolved_tool_names == ["winner"]


# ── discovery-wait bound (mcp_discovery_timeout config) ──────────────────────


def test_resolve_discovery_timeout_explicit_wins(monkeypatch):
    from hermes_cli import mcp_startup

    assert mcp_startup._resolve_discovery_timeout(2.5) == 2.5


def test_wait_returns_instantly_when_no_discovery_thread(monkeypatch):
    """The common case (no MCP / discovery done) pays ~0s regardless of bound."""
    import time
    from hermes_cli import mcp_startup

    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_discovery_timeout": 999.0})

    t0 = time.time()
    mcp_startup.wait_for_mcp_discovery()
    assert time.time() - t0 < 0.2  # never blocks on the bound when nothing's pending
