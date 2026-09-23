import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

from acp_adapter.session import SessionManager, SessionState
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from tui_gateway.compute_host import ComputeHost
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

    def replace_messages(self, *_args, **_kwargs):
        return None

    def append_messages_batch(self, *_args, **_kwargs):
        return None

    def set_session_title(self, session_id, title):
        self.rows[session_id]["title"] = title

    def set_auto_title(self, session_id, title, *, source):
        self.rows[session_id]["title"] = title

    def end_session(self, *_args, **_kwargs):
        return None

    def get_messages_as_conversation(self, *_args, **_kwargs):
        return []


class FakeAgent:
    def __init__(self, mode):
        self.model = "synthetic-model"
        self.provider = "synthetic-provider"
        self.base_url = "https://synthetic.invalid/v1"
        self.api_mode = "chat_completions"
        self._memory_provider_mode = mode
        self._session_init_model_config = {
            "memory_provider_mode": mode,
        }
        self._session_db = None
        self._session_db_created = False


class ModeManager(SessionManager):
    def __init__(self, db, live_mode="authoritative"):
        super().__init__(db=db)
        self.live_mode = live_mode
        self.agent_calls = []

    def _make_agent(self, **kwargs):
        self.agent_calls.append(dict(kwargs))
        mode = kwargs.get("memory_provider_mode_override") or self.live_mode
        return FakeAgent(mode)


def _stored_config(db, session_id):
    return json.loads(db.rows[session_id]["model_config"])


def test_acp_create_freezes_mode_and_restore_uses_stored_mode():
    db = DurableDB()
    manager = ModeManager(db, live_mode="authoritative")

    state = manager.create_session(cwd="/synthetic/workspace")
    # Official S keeps empty editor probes ephemeral; freeze still must persist
    # once the session has a real transcript.
    state.history = [{"role": "user", "content": "synthetic"}]
    manager._persist(state)
    assert _stored_config(db, state.session_id)["memory_provider_mode"] == "authoritative"

    manager.live_mode = "hybrid"
    manager._sessions.clear()
    restored = manager.get_session(state.session_id)

    assert restored is not None
    assert restored.agent._memory_provider_mode == "authoritative"
    assert manager.agent_calls[-1]["memory_provider_mode_override"] == "authoritative"


def test_acp_update_merges_mode_without_dropping_session_metadata():
    db = DurableDB()
    manager = ModeManager(db)
    session_id = "synthetic-acp-update"
    db.create_session(
        session_id,
        source="acp",
        model="old-model",
        model_config={
            "cwd": "/synthetic/old",
            "provider": "old-provider",
            "base_url": "https://old.invalid/v1",
            "api_mode": "responses",
            "keep": "synthetic-marker",
        },
    )
    state = SessionState(
        session_id=session_id,
        agent=FakeAgent("authoritative"),
        cwd="/synthetic/new",
        model="new-model",
    )
    state.agent.provider = "new-provider"
    state.agent.base_url = "https://new.invalid/v1"
    state.agent.api_mode = "chat_completions"

    manager._persist(state)
    config = _stored_config(db, session_id)

    assert config == {
        "cwd": "/synthetic/new",
        "provider": "new-provider",
        "base_url": "https://new.invalid/v1",
        "api_mode": "chat_completions",
        "keep": "synthetic-marker",
        "memory_provider_mode": "authoritative",
    }


def test_acp_fork_carries_original_frozen_mode():
    db = DurableDB()
    manager = ModeManager(db, live_mode="authoritative")
    original = manager.create_session(cwd="/synthetic/original")
    original.history = [{"role": "user", "content": "synthetic"}]
    manager._persist(original)

    manager.live_mode = "hybrid"
    fork = manager.fork_session(original.session_id, cwd="/synthetic/fork")

    assert fork is not None
    assert _stored_config(db, fork.session_id)["memory_provider_mode"] == "authoritative"
    assert manager.agent_calls[-1]["memory_provider_mode_override"] == "authoritative"


def test_tui_initial_row_contains_resolved_mode_before_agent_build(monkeypatch):
    db = DurableDB()
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "synthetic-model")
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"memory": {"provider_mode": "authoritative"}},
    )

    server._ensure_session_db_row({"session_key": "synthetic-tui"})

    config = _stored_config(db, "synthetic-tui")
    assert config["memory_provider_mode"] == "authoritative"
    assert (
        server._stored_session_runtime_overrides(db.rows["synthetic-tui"])[
            "memory_provider_mode_override"
        ]
        == "authoritative"
    )


def test_tui_reset_persists_new_agent_mode(monkeypatch):
    db = DurableDB()
    db.create_session(
        "synthetic-tui-reset",
        source="tui",
        model="synthetic-model",
        model_config={"memory_provider_mode": "authoritative", "keep": "marker"},
    )
    old_agent = FakeAgent("authoritative")
    new_agent = FakeAgent("hybrid")
    new_agent._session_db = db
    monkeypatch.setattr(server, "_set_session_context", lambda _key, cwd=None: None)
    monkeypatch.setattr(server, "_clear_session_context", lambda _token: None)
    monkeypatch.setattr(server, "_make_agent", lambda *_args, **_kwargs: new_agent)
    monkeypatch.setattr(server, "_config_model_target", lambda: None)
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: False)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "summary")
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)

    session = {
        "session_key": "synthetic-tui-reset",
        "agent": old_agent,
        "history": [],
        "history_lock": threading.Lock(),
    }

    server._reset_session_agent("synthetic-sid", session)

    config = _stored_config(db, "synthetic-tui-reset")
    assert config["memory_provider_mode"] == "hybrid"
    assert config["keep"] == "marker"
    assert session["agent"] is new_agent


def test_tui_branch_carries_frozen_mode_and_lineage(monkeypatch):
    db = DurableDB()
    parent_key = "synthetic-tui-parent"
    db.create_session(parent_key, source="tui", model_config={"keep": "marker"})
    parent_agent = FakeAgent("authoritative")
    session = {
        "agent": parent_agent,
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
    monkeypatch.setattr(server, "_set_session_context", lambda _key, cwd=None: None)
    monkeypatch.setattr(server, "_clear_session_context", lambda _token: None)
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_init_session", init_session)
    monkeypatch.setattr(server, "_transfer_db_to_agent", lambda *_args: False)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_history_to_messages", lambda history: history)
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"memory": {"provider_mode": "hybrid"}},
    )
    server._sessions.clear()

    response = server._methods["session.branch"]("synthetic-rid", {"session_id": "parent"})

    assert "error" not in response
    config = _stored_config(db, branch_key)
    assert config["memory_provider_mode"] == "authoritative"
    assert config["_branched_from"] == parent_key
    assert make_calls[-1]["memory_provider_mode_override"] == "authoritative"


def test_cli_branch_carries_frozen_mode_in_new_row(monkeypatch):
    db = DurableDB()
    parent_key = "synthetic-cli-parent"
    db.create_session(parent_key, source="cli", model_config={"keep": "marker"})
    agent = FakeAgent("authoritative")
    agent.session_id = parent_key
    agent.session_start = None
    agent.reset_session_state = lambda: None
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

    cli._handle_branch_command("/branch synthetic branch")

    rows = [row for key, row in db.rows.items() if key != parent_key]
    assert len(rows) == 1
    config = json.loads(rows[0]["model_config"])
    assert config["memory_provider_mode"] == "authoritative"
    assert config["_branched_from"] == parent_key


def test_compute_host_rebuild_passes_and_persists_frozen_mode(monkeypatch):
    db = DurableDB()
    key = "synthetic-host-session"
    db.create_session(key, source="tui", model_config={"memory_provider_mode": "authoritative"})

    class ChildServer:
        def __init__(self):
            self._sessions = {}
            self.calls = []
            self._persist_live_session_runtime = server._persist_live_session_runtime

        def _make_agent(self, *_args, **kwargs):
            self.calls.append(dict(kwargs))
            agent = FakeAgent(kwargs.get("memory_provider_mode_override") or "hybrid")
            agent._session_db = db
            return agent

        @staticmethod
        def _transfer_db_to_agent(*_args):
            return False

        def _init_session(self, sid, session_key, agent, history, **_kwargs):
            self._sessions[sid] = {
                "agent": agent,
                "session_key": session_key,
                "history": history,
            }

    child = ChildServer()
    @contextmanager
    def session_db(_session):
        yield db

    monkeypatch.setattr(server, "_session_db", session_db)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"memory": {"provider_mode": "hybrid"}})
    serving_session = {
        "session_key": key,
        "history": [],
        "history_lock": threading.Lock(),
        "cols": 80,
        "attached_images": [],
        "profile_home": None,
    }
    frame = server._compute_host_turn_frame(
        "synthetic-rid", "synthetic-host-sid", serving_session, "synthetic prompt"
    )
    assert frame["memory_provider_mode_override"] == "authoritative"
    host = ComputeHost(stdout=SimpleNamespace(write=lambda *_args: None, flush=lambda: None), heartbeat_secs=0)
    try:
        session = host._ensure_server_session(child, frame)
    finally:
        host.close()

    assert child.calls[-1]["memory_provider_mode_override"] == "authoritative"
    assert session["agent"]._memory_provider_mode == "authoritative"
    assert _stored_config(db, key)["memory_provider_mode"] == "authoritative"


def test_cli_init_agent_resume_keeps_frozen_mode_after_config_change(monkeypatch):
    """Fresh CLI rebuild on --resume must keep mode A after live config becomes B."""
    import types

    import cli as cli_mod

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._memory_provider_mode = kwargs.get("memory_provider_mode_override") or "hybrid"
            self._session_db_created = False

        def _ensure_db_session(self):
            return None

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = True
    cli.conversation_history = [{"role": "user", "content": "synthetic"}]
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    cli._memory_provider_mode_override = None
    cli._restore_session_memory_mode(
        {"model_config": json.dumps({"memory_provider_mode": "authoritative"})}
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kw: None,
    )
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("cli._prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr("cli.ChatConsole", lambda: types.SimpleNamespace(print=lambda *_a, **_k: None))

    assert cli._init_agent() is True
    assert captured["memory_provider_mode_override"] == "authoritative"
    assert cli.agent._memory_provider_mode == "authoritative"


def test_cli_midchat_resume_uses_shared_state_restore():
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    src = CLICommandsMixin._handle_resume_command.__code__.co_names
    assert "_restore_session_state" in src


def test_cli_memory_restore_updates_manager_and_initial_snapshot(monkeypatch):
    from agent.memory_manager import MemoryManager
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    agent = FakeAgent("hybrid")
    agent._memory_manager = MemoryManager(provider_mode="hybrid")
    agent.enabled_toolsets = ["memory"]
    agent.disabled_toolsets = []
    agent.quiet_mode = True
    agent._memory_enabled = True
    agent._user_profile_enabled = True
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kwargs: [])
    cli = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
    cli.agent = agent
    cli._memory_provider_mode_override = None

    cli._restore_session_memory_mode(
        {"model_config": json.dumps({"memory_provider_mode": "authoritative"})}
    )

    assert agent._memory_provider_mode == "authoritative"
    assert agent._memory_manager.provider_mode == "authoritative"
    assert agent._session_init_model_config["memory_provider_mode"] == "authoritative"


def test_tui_new_row_reads_selected_profile_memory_mode(monkeypatch, tmp_path):
    from hermes_constants import get_hermes_home

    launch_home = tmp_path / "launch"
    selected_home = tmp_path / "selected"
    launch_home.mkdir()
    selected_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_resolve_model", lambda: "synthetic-model")
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "memory": {
                "provider_mode": (
                    "hybrid" if get_hermes_home() == selected_home else "authoritative"
                )
            }
        },
    )

    _, config = server._workdir_row_model_config(
        {"profile_home": str(selected_home)}
    )

    assert config["memory_provider_mode"] == "hybrid"


def test_deferred_resume_keeps_memory_mode_without_llm_provider():
    current = {
        "resume_session_id": "synthetic-session",
        "resume_runtime_overrides": {"memory_provider_mode_override": "hybrid"},
    }
    kwargs = server._deferred_build_agent_kwargs(current, object())
    assert kwargs["memory_provider_mode_override"] == "hybrid"


def test_seeded_branch_live_record_keeps_parent_memory_mode(monkeypatch):
    db = DurableDB()
    db.create_session(
        "parent",
        source="tui",
        model_config={"memory_provider_mode": "authoritative"},
    )
    record = {"cwd": "/synthetic", "profile_home": None}

    @contextmanager
    def session_db(_record):
        yield db

    monkeypatch.setattr(server, "_session_db", session_db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "synthetic-model")
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    server._seed_branch_row(
        record,
        "child",
        "parent",
        [{"role": "user", "content": "synthetic"}],
        "tui",
        None,
    )

    assert record["resume_runtime_overrides"] == {
        "memory_provider_mode_override": "authoritative"
    }


def test_gateway_fresh_agent_uses_durable_memory_mode(monkeypatch):
    from gateway.run_turn_runner import TurnRunner

    db = DurableDB()
    db.create_session(
        "stored-session",
        source="gateway",
        model_config={"memory_provider_mode": "hybrid"},
    )
    captured = {}

    def make_agent(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    source = SimpleNamespace(
        user_id="u", user_id_alt=None, user_name="user", chat_id="c",
        chat_name="chat", chat_type="direct", thread_id=None,
    )
    ctx = SimpleNamespace(
        AIAgent=make_agent, user_config={}, enabled_toolsets=["memory"],
        disabled_toolsets=[], session_id="stored-session", source=source,
        session_key="gateway-key",
    )
    runner = SimpleNamespace(
        _prefill_messages=None, _service_tier=None,
        _session_db=SimpleNamespace(_db=db),
        _refresh_fallback_model=lambda: None,
    )
    monkeypatch.setattr("gateway.run._checkpoint_agent_kwargs", lambda _cfg: {})

    TurnRunner(runner, ctx)._build_fresh_agent(
        {"model": "synthetic", "runtime": {}}, "gateway", None, 4, {}, {}, False
    )

    assert captured["memory_provider_mode_override"] == "hybrid"


def test_background_agent_kwargs_carry_frozen_memory_mode(monkeypatch):
    from tui_gateway import server

    agent = FakeAgent("authoritative")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"memory": {"provider_mode": "hybrid"}})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "synthetic-model")
    monkeypatch.setattr(server, "_load_reasoning_config", lambda *_a, **_kw: {})
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_cfg_max_turns", lambda *_a, **_kw: 25)
    kwargs = server._background_agent_kwargs(agent, "task-id")
    assert kwargs["memory_provider_mode_override"] == "authoritative"


class _PublicBoundaryProvider:
    name = "synthetic-memory-provider"

    def __init__(self, sink):
        self.sink = sink

    def get_tool_schemas(self):
        return []

    def authoritative_memory_write(self, request, **_kwargs):
        self.sink.append(("authoritative", request["content"]))
        return json.dumps({"success": True, "operation_id": "synthetic"})

    def on_memory_write(self, _action, _target, content, **_kwargs):
        self.sink.append(("hybrid", content))


class _PublicBoundaryAgent:
    model = "synthetic-model"
    provider = "synthetic-provider"
    tools = []

    def __init__(self, session_id, mode, store, manager, session_db=None):
        self.session_id = session_id
        self._memory_provider_mode = mode
        self._session_init_model_config = {"memory_provider_mode": mode}
        self._memory_store = store
        self._memory_manager = manager
        self._session_db = session_db

    def clear_interrupt(self):
        return None

    def close(self):
        return None

    def _build_memory_write_metadata(self, **kwargs):
        return kwargs

    def run_conversation(self, prompt, conversation_history=None, **_kwargs):
        messages = [
            *(conversation_history or []),
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "synthetic answer"},
        ]
        if self._session_db is not None:
            self._session_db.append_message(self.session_id, role="user", content=str(prompt))
            self._session_db.append_message(
                self.session_id, role="assistant", content="synthetic answer"
            )
        return {"final_response": "synthetic answer", "messages": messages}


def _install_public_tui_runtime(monkeypatch, tmp_path, build_modes, write_sink):
    from agent.memory_manager import MemoryManager
    from hermes_constants import get_hermes_home
    from tests.tui_gateway.test_tui_gateway_server import _configure_immediate_prompt_run
    from tools.memory_tool import MemoryStore

    launch_home = tmp_path / "hermes"
    selected_home = launch_home / "profiles" / "B"
    selected_home.mkdir(parents=True)
    launch_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: launch_home / "profiles" / name,
    )

    def config():
        selected = get_hermes_home() == selected_home
        mode = "hybrid" if selected else "authoritative"
        return {
            "model": {"default": "synthetic-model"},
            "memory": {
                "provider_mode": mode,
                "memory_enabled": True,
                "user_profile_enabled": True,
            },
        }

    monkeypatch.setattr(server, "_load_cfg", config)
    monkeypatch.setattr(server, "_resolve_model", lambda: "synthetic-model")
    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_load_cfg", config)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_announce_built_agent", lambda *_args: None)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *_args: False)
    monkeypatch.setattr(server, "_start_session_services", lambda *_args: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_args: None)
    monkeypatch.setattr(server, "_transfer_db_to_agent", lambda agent, db: setattr(agent, "_session_db", db) or True)
    monkeypatch.setattr("tui_gateway.entry.ensure_mcp_discovery_started", lambda: None)

    def make_agent(_sid, key, session_db=None, memory_provider_mode_override=None, **_kwargs):
        mode = memory_provider_mode_override or config()["memory"]["provider_mode"]
        store = MemoryStore()
        store.load_from_disk()
        manager = MemoryManager(provider_mode=mode)
        manager.add_provider(_PublicBoundaryProvider(write_sink))
        agent = _PublicBoundaryAgent(key, mode, store, manager, session_db)
        build_modes.append((key, mode))
        return agent

    monkeypatch.setattr(server, "_make_agent", make_agent)
    server._sessions.clear()
    return launch_home, selected_home


def test_public_tui_profile_row_resume_and_core_write(monkeypatch, tmp_path):
    """Create, first submit, close, and eager resume stay within selected profile B."""
    from agent.inline_tool_executors import InlineToolContext, _memory
    from hermes_state import SessionDB

    builds, writes = [], []
    _launch, selected = _install_public_tui_runtime(monkeypatch, tmp_path, builds, writes)
    try:
        created = server.handle_request({
            "id": "create", "method": "session.create", "params": {"profile": "B"}
        })
        runtime_id = created["result"]["session_id"]
        stored_id = created["result"]["stored_session_id"]
        submitted = server.handle_request({
            "id": "turn", "method": "prompt.submit",
            "params": {"session_id": runtime_id, "text": "synthetic question"},
        })
        assert submitted["result"]["status"] == "streaming"
        live = server._sessions[runtime_id]["agent"]
        assert live._memory_provider_mode == "hybrid"
        with SessionDB(selected / "state.db") as db:
            row = db.get_session(stored_id)
            assert json.loads(row["model_config"])["memory_provider_mode"] == "hybrid"

        closed = server.handle_request({
            "id": "close", "method": "session.close", "params": {"session_id": runtime_id}
        })
        assert closed["result"]["closed"] is True
        resumed = server.handle_request({
            "id": "resume", "method": "session.resume",
            "params": {"session_id": stored_id, "profile": "B", "eager_build": True},
        })
        assert "error" not in resumed, resumed
        resumed_agent = server._sessions[resumed["result"]["session_id"]]["agent"]
        result = json.loads(_memory(
            resumed_agent,
            {"action": "add", "target": "memory", "content": "profile-B fact"},
            InlineToolContext(effective_task_id="tui", tool_call_id="memory"),
        ))
        assert result["success"] is True
        assert resumed_agent._memory_provider_mode == "hybrid"
        assert writes == [("hybrid", "profile-B fact")]
    finally:
        server._sessions.clear()


def test_public_tui_deferred_and_eager_resume_keep_memory_mode(monkeypatch, tmp_path):
    """LLM-provider absence must not remove a durable memory-mode override."""
    from hermes_state import SessionDB

    builds, writes = [], []
    launch, _selected = _install_public_tui_runtime(monkeypatch, tmp_path, builds, writes)
    db = SessionDB(launch / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    db.create_session(
        "durable-hybrid", source="tui", model="synthetic-model",
        model_config={"memory_provider_mode": "hybrid"},
    )
    db.append_message("durable-hybrid", role="user", content="stored question")
    db.append_message("durable-hybrid", role="assistant", content="stored answer")
    try:
        deferred = server.handle_request({
            "id": "deferred", "method": "session.resume",
            "params": {"session_id": "durable-hybrid"},
        })
        deferred_sid = deferred["result"]["session_id"]
        server._start_agent_build(deferred_sid, server._sessions[deferred_sid])
        assert server._sessions[deferred_sid]["agent"]._memory_provider_mode == "hybrid"
        server.handle_request({
            "id": "close", "method": "session.close", "params": {"session_id": deferred_sid}
        })

        eager = server.handle_request({
            "id": "eager", "method": "session.resume",
            "params": {"session_id": "durable-hybrid", "eager_build": True},
        })
        assert "error" not in eager, eager
        eager_sid = eager["result"]["session_id"]
        assert server._sessions[eager_sid]["agent"]._memory_provider_mode == "hybrid"
        assert [mode for key, mode in builds if key == "durable-hybrid"] == ["hybrid", "hybrid"]
    finally:
        server._sessions.clear()
        db.close()


def test_public_tui_seeded_create_and_direct_branch_keep_parent_mode(monkeypatch, tmp_path):
    """Both public branch forms must bind their first agent to the durable parent mode."""
    from hermes_state import SessionDB

    builds, writes = [], []
    launch, _selected = _install_public_tui_runtime(monkeypatch, tmp_path, builds, writes)
    db = SessionDB(launch / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    db.create_session(
        "branch-parent", source="tui", model="synthetic-model",
        model_config={"memory_provider_mode": "hybrid"},
    )
    db.append_message("branch-parent", role="user", content="parent question")
    db.append_message("branch-parent", role="assistant", content="parent answer")
    try:
        parent = server.handle_request({
            "id": "parent", "method": "session.resume",
            "params": {"session_id": "branch-parent", "eager_build": True},
        })
        assert "error" not in parent, parent
        parent_sid = parent["result"]["session_id"]

        seeded = server.handle_request({
            "id": "seeded", "method": "session.create",
            "params": {
                "parent_session_id": "branch-parent",
                "messages": [{"role": "user", "content": "seeded question"}],
            },
        })
        seeded_sid = seeded["result"]["session_id"]
        server._start_agent_build(seeded_sid, server._sessions[seeded_sid])
        seeded_key = seeded["result"]["stored_session_id"]
        assert server._sessions[seeded_sid]["agent"]._memory_provider_mode == "hybrid"
        assert json.loads(db.get_session(seeded_key)["model_config"])["memory_provider_mode"] == "hybrid"

        direct = server.handle_request({
            "id": "direct", "method": "session.branch",
            "params": {"session_id": parent_sid, "name": "direct child"},
        })
        direct_sid = direct["result"]["session_id"]
        direct_key = direct["result"]["stored_session_id"]
        assert server._sessions[direct_sid]["agent"]._memory_provider_mode == "hybrid"
        assert json.loads(db.get_session(direct_key)["model_config"])["memory_provider_mode"] == "hybrid"
    finally:
        server._sessions.clear()
        db.close()
