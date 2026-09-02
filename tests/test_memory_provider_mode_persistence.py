import json
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_manager import MemoryManager

from acp_adapter.session import SessionManager, _merge_session_memory_provider_mode
from agent.memory_provider import (
    normalize_memory_provider_mode,
    persisted_memory_provider_mode,
)
from hermes_cli.cli_commands_mixin import CLICommandsMixin, _rebind_memory_provider_mode
from tools.memory_tool import get_memory_provider_mode
from tui_gateway import server


class DurableDB:
    def __init__(self):
        self.rows = {}
        self.updates = []

    def get_session(self, session_id):
        return self.rows.get(session_id)

    def get_session_title(self, session_id):
        return (self.rows.get(session_id) or {}).get("title")

    def get_next_title_in_lineage(self, _title):
        return "synthetic branch"

    def create_session(self, session_id, source, model=None, model_config=None, **_kwargs):
        self.rows.setdefault(
            session_id,
            {
                "id": session_id,
                "source": source,
                "model": model,
                "model_config": json.dumps(model_config or {}),
            },
        )
        return session_id

    def update_session_meta(self, session_id, model_config_json, model=None):
        row = self.rows[session_id]
        row["model_config"] = model_config_json
        if model is not None:
            row["model"] = model
        self.updates.append((session_id, json.loads(model_config_json), model))

    def append_messages_batch(self, *_args, **_kwargs):
        return None

    def replace_messages(self, *_args, **_kwargs):
        return None

    def set_session_title(self, session_id, title):
        self.rows[session_id]["title"] = title

    def end_session(self, *_args, **_kwargs):
        return None


def _stored_config(db, session_id):
    return json.loads(db.rows[session_id]["model_config"])


class FakeAgent:
    def __init__(self, mode):
        self.model = "synthetic-model"
        self.provider = "synthetic-provider"
        self.base_url = "https://synthetic.invalid/v1"
        self.api_mode = "chat_completions"
        self._memory_provider_mode = mode
        self._session_init_model_config = {"memory_provider_mode": mode}
        self._session_db = None
        self.session_id = "synthetic-session"
        self.session_start = None
        self._memory_manager = None

    def reset_session_state(self):
        return None


class RecordingRuntimeProvider:
    name = "synthetic-runtime"

    def __init__(self):
        self.calls = []
        self.initialized = []
        self.ended = []
        self.shutdown_calls = 0
        self.switched = []

    def is_available(self):
        return True

    def get_tool_schemas(self):
        return []

    def initialize(self, **kwargs):
        self.initialized.append(kwargs)

    def authoritative_memory_write(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return {"success": True, "operation_id": "synthetic-op"}

    def on_session_end(self, messages):
        self.ended.append(messages)

    def on_session_switch(self, new_session_id, **kwargs):
        self.switched.append((new_session_id, kwargs))

    def shutdown(self):
        self.shutdown_calls += 1


def _build_runtime_agent(ambient_mode, target_mode):
    provider = RecordingRuntimeProvider()
    cfg = {
        "memory": {
            "provider": provider.name,
            "provider_mode": ambient_mode,
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
        "agent": {},
    }
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("plugins.memory.load_memory_provider", return_value=provider),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            enabled_toolsets=["memory"],
            memory_provider_mode_override=target_mode,
        )
    return agent, provider


@pytest.mark.parametrize(
    ("ambient_mode", "target_mode", "has_memory_tool"),
    [("hybrid", "authoritative", True), ("authoritative", "hybrid", False)],
)
def test_aiagent_persisted_mode_owns_memory_tool_schema_and_dispatch(
    ambient_mode, target_mode, has_memory_tool
):
    agent, provider = _build_runtime_agent(ambient_mode, target_mode)
    schemas = {
        tool["function"]["name"]: tool["function"] for tool in agent.tools
    }

    assert ("memory" in schemas) is has_memory_tool
    assert ("memory" in agent.valid_tool_names) is has_memory_tool
    if target_mode == "authoritative":
        assert schemas["memory"]["parameters"]["properties"]["target"]["enum"] == [
            "memory",
            "user",
        ]
        result = json.loads(
            agent._invoke_tool(
                "memory",
                {"action": "add", "target": "memory", "content": "synthetic"},
                "task",
                skip_tool_request_middleware=True,
            )
        )
        assert result["success"] is True
        assert provider.calls and len(provider.calls) == 1
        assert agent._memory_store is None
    else:
        assert provider.calls == []


def test_gateway_cold_build_uses_persisted_mode_for_real_tool_surface():
    provider = RecordingRuntimeProvider()
    cfg = {
        "memory": {
            "provider": provider.name,
            "provider_mode": "hybrid",
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
        "agent": {},
    }
    runtime = {
        "provider": "synthetic",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key-1234567890",
        "api_mode": "chat_completions",
    }
    with (
        patch("tui_gateway.server._load_cfg", return_value=cfg),
        patch("tui_gateway.server._resolve_startup_runtime", return_value=("synthetic-model", "synthetic")),
        patch(
            "tui_gateway.server._resolve_runtime_with_fallback",
            return_value=SimpleNamespace(runtime=runtime, used_fallback=False),
        ),
        patch("tui_gateway.server._parse_tui_skills_env", return_value=[]),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=["memory"]),
        patch("tui_gateway.server._load_provider_routing", return_value={}),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("hermes_cli.config.resolve_ephemeral_system_prompt_from_config", return_value=""),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("plugins.memory.load_memory_provider", return_value=provider),
    ):
        agent = server._make_agent(
            "synthetic-sid",
            "synthetic-key",
            memory_provider_mode_override="authoritative",
            platform_override="tui",
        )

    assert "memory" in agent.valid_tool_names
    result = json.loads(
        agent._invoke_tool(
            "memory",
            {"action": "add", "target": "memory", "content": "synthetic"},
            "task",
            skip_tool_request_middleware=True,
        )
    )
    assert result["success"] is True
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    ("ambient_mode", "target_mode", "has_memory_tool"),
    [("hybrid", "authoritative", True), ("authoritative", "hybrid", False)],
)
def test_mid_chat_rebind_rebuilds_complete_memory_runtime(
    ambient_mode, target_mode, has_memory_tool
):
    agent, old_provider = _build_runtime_agent(ambient_mode, ambient_mode)
    new_provider = RecordingRuntimeProvider()

    with patch("plugins.memory.load_memory_provider", return_value=new_provider):
        assert _rebind_memory_provider_mode(
            agent,
            target_mode,
            target_session_id="target-session",
            messages=[{"role": "user", "content": "old-session"}],
        ) is True
    release, ticket, old_manager = agent._pending_memory_retirement
    release.set()
    assert old_manager.wait_for_retirement(ticket, timeout=5)

    assert ("memory" in agent.valid_tool_names) is has_memory_tool
    assert ({tool["function"]["name"] for tool in agent.tools} >= {"memory"}) is has_memory_tool
    assert agent._memory_provider_mode == target_mode
    assert agent._memory_manager.provider_mode == target_mode
    assert old_provider.shutdown_calls == 1
    assert old_provider.ended == [[{"role": "user", "content": "old-session"}]]
    assert new_provider.initialized[0]["session_id"] == "target-session"
    assert new_provider.switched == []
    if target_mode == "authoritative":
        result = json.loads(
            agent._invoke_tool(
                "memory",
                {"action": "add", "target": "memory", "content": "synthetic"},
                "task",
                skip_tool_request_middleware=True,
            )
        )
        assert result["success"] is True
        assert new_provider.calls and old_provider.calls == []
    else:
        assert new_provider.calls == []


def test_mid_chat_rebind_failure_rolls_back_full_runtime_and_new_resources():
    agent, old_provider = _build_runtime_agent("hybrid", "hybrid")
    new_provider = RecordingRuntimeProvider()
    old = {
        "manager": agent._memory_manager,
        "store": agent._memory_store,
        "tools": agent.tools,
        "names": agent.valid_tool_names,
        "mode": agent._memory_provider_mode,
        "config": agent._memory_config,
        "init_config": agent._session_init_model_config,
    }

    with (
        patch("plugins.memory.load_memory_provider", return_value=new_provider),
        patch("model_tools.get_tool_definitions", side_effect=RuntimeError("snapshot-failed")),
    ):
        assert _rebind_memory_provider_mode(agent, "authoritative") is False

    assert agent._memory_manager is old["manager"]
    assert agent._memory_store is old["store"]
    assert agent.tools is old["tools"]
    assert agent.valid_tool_names is old["names"]
    assert agent._memory_provider_mode == old["mode"]
    assert agent._memory_config is old["config"]
    assert agent._session_init_model_config is old["init_config"]
    assert old_provider.shutdown_calls == 0
    assert new_provider.shutdown_calls == 1


def test_rebind_has_no_fallible_postcommit_seam():
    agent, old_provider = _build_runtime_agent("hybrid", "hybrid")
    old_manager = agent._memory_manager
    new_provider = RecordingRuntimeProvider()
    agent._invalidate_system_prompt = MagicMock(
        side_effect=RuntimeError("post-commit invalidation failed")
    )

    with patch("plugins.memory.load_memory_provider", return_value=new_provider):
        assert _rebind_memory_provider_mode(
            agent,
            "authoritative",
            target_session_id="target-session",
            messages=[{"role": "user", "content": "outgoing"}],
        )

    agent._invalidate_system_prompt.assert_not_called()
    assert agent._memory_manager is not old_manager
    assert agent._memory_manager.get_provider(new_provider.name) is new_provider
    assert new_provider.shutdown_calls == 0
    release, ticket, retirement_manager = agent._pending_memory_retirement
    release.set()
    assert retirement_manager is old_manager
    assert retirement_manager.wait_for_retirement(ticket, timeout=5)
    assert old_provider.shutdown_calls == 1
    agent._memory_manager.shutdown_all()


def test_retirement_admission_failure_cancels_rebind_publication():
    agent, old_provider = _build_runtime_agent("hybrid", "hybrid")
    old_manager = agent._memory_manager
    old_tools = agent.tools
    rejected = Future()
    rejected.set_exception(RuntimeError("memory retirement admission failed"))
    new_provider = RecordingRuntimeProvider()

    with (
        patch("plugins.memory.load_memory_provider", return_value=new_provider),
        patch.object(
            old_manager,
            "commit_session_boundary_async",
            return_value=rejected,
        ),
    ):
        assert not _rebind_memory_provider_mode(
            agent,
            "authoritative",
            target_session_id="target-session",
            messages=[{"role": "user", "content": "outgoing"}],
        )

    assert agent._memory_manager is old_manager
    assert agent.tools is old_tools
    assert agent._memory_provider_mode == "hybrid"
    assert old_provider.shutdown_calls == 0
    assert new_provider.shutdown_calls == 1


def test_memory_builder_does_not_construct_builtin_store():
    from agent.agent_init import build_memory_subsystem
    import tools.memory_tool as memory_tool

    agent = SimpleNamespace(
        enabled_toolsets=["memory"],
        disabled_toolsets=[],
        _memory_store=None,
        _memory_enabled=True,
        _user_profile_enabled=False,
        platform="tui",
    )
    with patch.object(
        memory_tool,
        "MemoryStore",
        side_effect=AssertionError("built-in store belongs to upstream init"),
    ):
        state = build_memory_subsystem(
            agent,
            {
                "memory": {
                    "memory_enabled": True,
                    "user_profile_enabled": False,
                }
            },
            skip_memory=True,
            provider_mode="hybrid",
            session_id="target-session",
        )

    assert state["store"] is None
    assert state["memory_enabled"] is True
    assert state["user_profile_enabled"] is False


def test_partial_memory_builder_failure_shuts_unregistered_provider():
    from agent.agent_init import build_memory_subsystem

    provider = RecordingRuntimeProvider()

    def _explode():
        raise RuntimeError("availability failed")

    provider.is_available = _explode
    agent = SimpleNamespace(
        enabled_toolsets=[],
        disabled_toolsets=[],
        platform="tui",
    )
    with (
        patch("plugins.memory.load_memory_provider", return_value=provider),
        pytest.raises(RuntimeError, match="availability failed"),
    ):
        build_memory_subsystem(
            agent,
            {"memory": {"provider": provider.name}},
            skip_memory=False,
            provider_mode="hybrid",
            session_id="target-session",
        )

    assert provider.shutdown_calls == 1


@pytest.mark.parametrize(
    ("old_mode", "target_mode", "target_name"),
    [
        ("hybrid", "authoritative", "memory"),
        ("authoritative", "hybrid", "plain"),
    ],
)
def test_stale_refresh_cannot_overwrite_memory_rebind(
    monkeypatch, old_mode, target_mode, target_name
):
    from tools import mcp_tool
    import model_tools

    def _definition(name):
        return {
            "type": "function",
            "function": {"name": name, "description": name, "parameters": {}},
        }

    old_name = "plain" if old_mode == "hybrid" else "memory"
    old_manager = MemoryManager(provider_mode=old_mode)
    new_manager = MemoryManager(provider_mode=target_mode)
    agent = SimpleNamespace(
        _tool_snapshot_generation=0,
        _memory_manager=old_manager,
        _memory_store=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        _memory_provider_mode=old_mode,
        _memory_config={},
        _session_init_model_config={"memory_provider_mode": old_mode},
        _skip_memory=False,
        enabled_toolsets=None,
        disabled_toolsets=None,
        quiet_mode=True,
        tools=[_definition(old_name)],
        valid_tool_names={old_name},
        _context_engine_tool_names=set(),
        context_compressor=None,
        session_id="outgoing",
    )
    entered = threading.Event()
    release = threading.Event()
    refresh_calls = 0

    def _definitions(**_kwargs):
        nonlocal refresh_calls
        if threading.current_thread().name == "stale-refresh":
            refresh_calls += 1
            if refresh_calls == 1:
                entered.set()
                assert release.wait(timeout=5)
                return [_definition(old_name)]
        return [_definition(target_name)]

    monkeypatch.setattr(model_tools, "get_tool_definitions", _definitions)
    monkeypatch.setattr(
        "agent.agent_init.build_memory_subsystem",
        lambda *_args, **_kwargs: {
            "manager": new_manager,
            "store": None,
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
    )
    worker = threading.Thread(
        target=mcp_tool.refresh_agent_mcp_tools,
        args=(agent,),
        name="stale-refresh",
    )
    worker.start()
    assert entered.wait(timeout=2)
    assert _rebind_memory_provider_mode(
        agent,
        target_mode,
        target_session_id="target",
        messages=[{"role": "user", "content": "outgoing"}],
    )
    release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert agent._memory_manager is new_manager
    assert agent._memory_provider_mode == target_mode
    assert agent.valid_tool_names == {target_name}
    assert {tool["function"]["name"] for tool in agent.tools} == {target_name}
    assert agent._tool_snapshot_generation == 2
    retirement_release, retirement_ticket, retirement_manager = (
        agent._pending_memory_retirement
    )
    retirement_release.set()
    assert retirement_manager is old_manager
    assert retirement_manager.wait_for_retirement(retirement_ticket, timeout=5)
    new_manager.shutdown_all()


@pytest.mark.parametrize("mode", [[], {}, False, 0, 1, None, "", " authoritative", "unknown"])
def test_persisted_provider_mode_normalizes_to_hybrid(mode):
    assert normalize_memory_provider_mode(mode) == "hybrid"
    assert get_memory_provider_mode({"provider_mode": mode}) == "hybrid"

    agent = FakeAgent(mode)
    session = {"agent": agent, "resume_runtime_overrides": {}}
    assert server._session_memory_provider_mode(session) == "hybrid"

    row = {
        "model": "synthetic-model",
        "model_config": json.dumps(
            {"memory_provider_mode": mode, "unrelated": {"keep": ["value"]}}
        ),
    }
    overrides = server._stored_session_runtime_overrides(row)
    assert overrides["memory_provider_mode_override"] == "hybrid"

    existing = {"memory_provider_mode": mode, "unrelated": {"keep": ["value"]}}
    config = server._runtime_model_config(agent, existing)
    assert config["memory_provider_mode"] == "hybrid"
    assert config["unrelated"] == existing["unrelated"]
    assert config["unrelated"] is not existing["unrelated"]
    config["unrelated"]["keep"].append("new")
    assert existing["unrelated"]["keep"] == ["value"]

    merged = _merge_session_memory_provider_mode(existing, agent)
    assert merged["memory_provider_mode"] == "hybrid"
    assert merged["unrelated"] is not existing["unrelated"]


@pytest.mark.parametrize(
    "model_config",
    ["{broken", json.dumps({"memory_provider_mode": "unknown"}), json.dumps([])],
)
def test_malformed_session_mode_fails_safe_to_hybrid(model_config):
    assert persisted_memory_provider_mode({"model_config": model_config}) == "hybrid"


def test_acp_restore_uses_shared_malformed_mode_fallback():
    db = MagicMock()
    db.get_session.return_value = {
        "id": "acp-session",
        "source": "acp",
        "model_config": "{broken",
    }
    db.get_messages_as_conversation.return_value = []
    manager = SessionManager(db=db)
    restored_agent = FakeAgent("hybrid")
    manager._make_agent = MagicMock(return_value=restored_agent)

    restored = manager._restore("acp-session")

    assert restored is not None
    assert manager._make_agent.call_args.kwargs["memory_provider_mode_override"] == "hybrid"


@pytest.mark.parametrize("mode", [[], {}, False, 0, 1, "", " authoritative", "unknown"])
def test_aiagent_invalid_persisted_mode_defaults_hybrid(mode):
    cfg = {
        "memory": {"provider": "", "provider_mode": "authoritative"},
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            memory_provider_mode_override=mode,
        )

    assert agent._memory_provider_mode == "hybrid"
    assert agent._session_init_model_config["memory_provider_mode"] == "hybrid"


def test_runtime_model_config_persists_frozen_mode():
    agent = FakeAgent("authoritative")

    config = server._runtime_model_config(agent)

    assert config["memory_provider_mode"] == "authoritative"


def test_stored_session_runtime_overrides_restores_frozen_mode():
    overrides = server._stored_session_runtime_overrides(
        {
            "model": "synthetic-model",
            "model_config": json.dumps(
                {"memory_provider_mode": "authoritative"}
            ),
        }
    )

    assert overrides["memory_provider_mode_override"] == "authoritative"


@pytest.mark.parametrize(
    ("ambient_mode", "target_mode"),
    [("hybrid", "authoritative"), ("authoritative", "hybrid")],
)
def test_classic_cli_mid_chat_resume_requests_complete_frozen_mode_rebind(
    ambient_mode, target_mode
):
    manager = SimpleNamespace(
        provider_mode=ambient_mode,
        on_session_switch=MagicMock(),
        commit_session_boundary_async=MagicMock(),
    )
    agent = FakeAgent(ambient_mode)
    agent._memory_manager = manager
    agent._flush_messages_to_session_db = MagicMock()
    agent._invalidate_system_prompt = MagicMock()

    row = {
        "id": "target-session",
        "title": "target",
        "model_config": json.dumps({"memory_provider_mode": target_mode}),
    }
    db = MagicMock()
    db.get_session.return_value = row
    db.resolve_resume_session_id.return_value = "target-session"
    db.get_resume_conversations.return_value = (
        [{"role": "user", "content": "target history"}],
        [{"role": "user", "content": "target history"}],
    )
    cli = CLICommandsMixin()
    cli.agent = agent
    cli.session_id = "ambient-session"
    cli.conversation_history = [{"role": "user", "content": "ambient history"}]
    cli._resume_display_history = []
    cli._session_db = db
    cli._pending_resume_sessions = None
    cli._pending_title = None
    cli._resumed = False
    cli._display_resumed_history = MagicMock()
    cli._restore_session_cwd = MagicMock()
    cli._restore_session_yolo = MagicMock()
    cli._restore_session_model = MagicMock()
    ordering = []
    manager.commit_session_boundary_async.side_effect = (
        lambda *_args, **_kwargs: ordering.append("boundary")
    )

    with (
        patch(
            "hermes_cli.main._resolve_session_by_name_or_id",
            return_value="target-session",
        ),
        patch("cli._sync_process_session_id"),
        patch(
            "hermes_cli.cli_commands_mixin._rebind_memory_provider_mode",
            side_effect=lambda *_args, **_kwargs: ordering.append("rebind") or True,
        ) as rebind,
    ):
        cli._handle_resume_command("/resume target-session")

    assert cli.session_id == "target-session"
    rebind.assert_called_once_with(
        agent,
        target_mode,
        [{"role": "user", "content": "ambient history"}],
        target_session_id="target-session",
    )
    manager.commit_session_boundary_async.assert_called_once_with(
        [{"role": "user", "content": "ambient history"}],
        new_session_id="target-session",
        parent_session_id="ambient-session",
        reset=False,
        reason="resume",
    )
    manager.on_session_switch.assert_not_called()
    assert ordering == ["rebind", "boundary"]


@pytest.mark.parametrize("seam", ["sync", "reset", "invalidate"])
def test_classic_cli_resume_settles_published_retirement_before_failure_escapes(seam):
    agent, old_provider = _build_runtime_agent("hybrid", "hybrid")
    old_manager = agent._memory_manager
    new_provider = RecordingRuntimeProvider()
    agent._flush_messages_to_session_db = MagicMock()
    agent.reset_session_state = MagicMock(
        side_effect=RuntimeError("reset failed") if seam == "reset" else None
    )
    agent._invalidate_system_prompt = MagicMock(
        side_effect=RuntimeError("invalidate failed") if seam == "invalidate" else None
    )
    db = MagicMock()
    db.get_session.return_value = {
        "id": "target-session",
        "model_config": json.dumps({"memory_provider_mode": "authoritative"}),
    }
    db.resolve_resume_session_id.return_value = "target-session"
    db.get_resume_conversations.return_value = (
        [{"role": "user", "content": "target"}],
        [],
    )
    cli = CLICommandsMixin()
    cli.agent = agent
    cli.session_id = "ambient-session"
    cli.conversation_history = [{"role": "user", "content": "ambient"}]
    cli._session_db = db
    cli._pending_resume_sessions = None
    cli._pending_title = None
    cli._resumed = False
    cli._display_resumed_history = MagicMock()
    cli._restore_session_cwd = MagicMock()
    cli._restore_session_yolo = MagicMock()
    cli._restore_session_model = MagicMock()

    try:
        with (
            patch(
                "hermes_cli.main._resolve_session_by_name_or_id",
                return_value="target-session",
            ),
            patch(
                "cli._sync_process_session_id",
                side_effect=RuntimeError("sync failed") if seam == "sync" else None,
            ),
            patch("plugins.memory.load_memory_provider", return_value=new_provider),
            pytest.raises(RuntimeError, match=f"{seam} failed"),
        ):
            cli._handle_resume_command("/resume target-session")

        pending = getattr(agent, "_pending_memory_retirement", None)
        release_was_set = pending is None or pending[0].is_set()
    finally:
        pending = getattr(agent, "_pending_memory_retirement", None)
        if pending is not None:
            pending[0].set()
            pending[2].wait_for_retirement(pending[1], timeout=5)

    assert release_was_set
    assert agent._pending_memory_retirement is None
    assert agent._memory_manager is not old_manager
    assert new_provider.shutdown_calls == 0
    assert old_provider.shutdown_calls == 1
    agent._memory_manager.shutdown_all()


def test_classic_cli_resume_rebinds_after_memory_init_fallback():
    cfg = {
        "memory": {
            "provider": RecordingRuntimeProvider.name,
            "provider_mode": "hybrid",
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
        "agent": {},
    }
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "agent.agent_init.build_memory_subsystem",
            side_effect=RuntimeError("provider init failed"),
        ),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            enabled_toolsets=["memory"],
            memory_provider_mode_override="hybrid",
        )
    assert agent._memory_manager is None

    new_provider = RecordingRuntimeProvider()
    agent._flush_messages_to_session_db = MagicMock()
    agent.reset_session_state = MagicMock()
    agent._invalidate_system_prompt = MagicMock()
    db = MagicMock()
    db.get_session.return_value = {
        "id": "target-session",
        "model_config": json.dumps({"memory_provider_mode": "authoritative"}),
    }
    db.resolve_resume_session_id.return_value = "target-session"
    db.get_resume_conversations.return_value = (
        [{"role": "user", "content": "target"}],
        [],
    )
    cli = CLICommandsMixin()
    cli.agent = agent
    cli.session_id = "ambient-session"
    cli.conversation_history = [{"role": "user", "content": "ambient"}]
    cli._session_db = db
    cli._pending_resume_sessions = None
    cli._pending_title = None
    cli._resumed = False
    cli._display_resumed_history = MagicMock()
    cli._restore_session_cwd = MagicMock()
    cli._restore_session_yolo = MagicMock()
    cli._restore_session_model = MagicMock()

    with (
        patch(
            "hermes_cli.main._resolve_session_by_name_or_id",
            return_value="target-session",
        ),
        patch("cli._sync_process_session_id"),
        patch("plugins.memory.load_memory_provider", return_value=new_provider),
    ):
        try:
            cli._handle_resume_command("/resume target-session")
            assert cli.session_id == "target-session"
            assert cli.conversation_history == [{"role": "user", "content": "target"}]
            assert agent._memory_manager.get_provider(new_provider.name) is new_provider
            assert agent._pending_memory_retirement is None
            assert new_provider.shutdown_calls == 0
        finally:
            if agent._memory_manager is not None:
                agent._memory_manager.shutdown_all()


def test_classic_cli_resume_mode_rebind_failure_keeps_old_session():
    agent = FakeAgent("hybrid")
    agent._flush_messages_to_session_db = MagicMock()
    row = {
        "id": "target-session",
        "model_config": json.dumps({"memory_provider_mode": "authoritative"}),
    }
    db = MagicMock()
    db.get_session.return_value = row
    db.resolve_resume_session_id.return_value = "target-session"
    db.get_resume_conversations.return_value = ([{"role": "user", "content": "target"}], [])
    cli = CLICommandsMixin()
    cli.agent = agent
    cli.session_id = "ambient-session"
    cli.conversation_history = [{"role": "user", "content": "ambient"}]
    cli._session_db = db
    cli._pending_resume_sessions = None
    cli._pending_title = None
    cli._resumed = False
    cli._display_resumed_history = MagicMock()
    cli._restore_session_cwd = MagicMock()
    cli._restore_session_yolo = MagicMock()
    cli._restore_session_model = MagicMock()

    with (
        patch(
            "hermes_cli.main._resolve_session_by_name_or_id",
            return_value="target-session",
        ),
        patch("cli._sync_process_session_id"),
        patch("agent.memory_manager.MemoryManager", side_effect=RuntimeError("sentinel")),
    ):
        cli._handle_resume_command("/resume target-session")

    assert cli.session_id == "ambient-session"
    assert cli.conversation_history == [{"role": "user", "content": "ambient"}]
    assert agent.session_id == "synthetic-session"
    assert agent._memory_provider_mode == "hybrid"
    db.end_session.assert_not_called()


def test_tui_turn_frame_carries_frozen_mode():
    session = {
        "agent": FakeAgent("authoritative"),
        "session_key": "synthetic-tui",
        "history": [],
        "history_lock": threading.Lock(),
        "cols": 80,
    }

    frame = server._compute_host_turn_frame("rid", "sid", session, "prompt")

    assert frame["memory_provider_mode_override"] == "authoritative"


def test_tui_branch_carries_frozen_mode_and_lineage(monkeypatch):
    db = DurableDB()
    parent_key = "synthetic-tui-parent"
    db.create_session(parent_key, source="tui", model_config={"keep": "marker"})
    session = {
        "agent": FakeAgent("authoritative"),
        "session_key": parent_key,
        "history": [{"role": "user", "content": "synthetic prompt"}],
        "display_history_prefix": [],
        "history_lock": threading.Lock(),
        "profile_home": None,
        "cols": 80,
        "source": "tui",
        "cwd": "/synthetic",
    }
    make_calls = []
    branch_key = "synthetic-tui-branch"

    def make_agent(*_args, **kwargs):
        make_calls.append(dict(kwargs))
        agent = FakeAgent(kwargs.get("memory_provider_mode_override") or "hybrid")
        agent._session_db = db
        return agent

    def init_session(sid, key, agent, history, **_kwargs):
        server._sessions[sid] = {
            "agent": agent,
            "session_key": key,
            "history": history,
        }

    @contextmanager
    def session_db(_session):
        yield db

    monkeypatch.setattr(server, "_sess", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_session_db", session_db)
    monkeypatch.setattr(server, "_new_session_key", lambda: branch_key)
    monkeypatch.setattr(server.uuid, "uuid4", lambda: SimpleNamespace(hex="12345678"))
    monkeypatch.setattr(server, "_session_source", lambda _session: "tui")
    monkeypatch.setattr(server, "_session_cwd", lambda _session: "/synthetic")
    monkeypatch.setattr(server, "_resolve_model", lambda: "synthetic-model")
    monkeypatch.setattr(server, "_set_session_context", lambda _key: None)
    monkeypatch.setattr(server, "_clear_session_context", lambda _token: None)
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_init_session", init_session)
    monkeypatch.setattr(server, "_transfer_db_to_agent", lambda *_args: False)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_history_to_messages", lambda history: history)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"memory": {"provider_mode": "hybrid"}})
    server._sessions.clear()

    response = server._methods["session.branch"]("synthetic-rid", {"session_id": "parent"})

    assert "error" not in response
    config = _stored_config(db, branch_key)
    assert config["memory_provider_mode"] == "authoritative"
    assert config["_branched_from"] == parent_key
    assert make_calls[-1]["memory_provider_mode_override"] == "authoritative"


@pytest.mark.parametrize(
    "mode",
    ["authoritative", [], {}, False, 0, 1, None, "", " authoritative", "unknown"],
)
def test_cli_branch_carries_frozen_mode_in_new_row(monkeypatch, mode):
    db = DurableDB()
    parent_key = "synthetic-cli-parent"
    db.create_session(parent_key, source="cli", model_config={"keep": "marker"})
    agent = FakeAgent(mode)
    agent.session_id = parent_key
    cli = CLICommandsMixin.__new__(CLICommandsMixin)
    cli.conversation_history = [{"role": "user", "content": "synthetic prompt"}]
    cli._session_db = db
    cli.session_id = parent_key
    cli.model = "synthetic-model"
    cli.max_turns = 4
    cli.reasoning_config = {"effort": "low"}
    cli.agent = agent
    cli._pending_title = None
    cli._resumed = False
    cli._transfer_session_yolo = lambda *_args: None

    monkeypatch.setattr("cli._sync_process_session_id", lambda _session_id: None)
    monkeypatch.setattr("cli._cprint", lambda *_args: None)

    cli._handle_branch_command("/branch synthetic branch")

    rows = [row for key, row in db.rows.items() if key != parent_key]
    assert len(rows) == 1
    config = json.loads(rows[0]["model_config"])
    expected_mode = "authoritative" if mode == "authoritative" else "hybrid"
    assert config["memory_provider_mode"] == expected_mode
    assert config["_branched_from"] == parent_key
