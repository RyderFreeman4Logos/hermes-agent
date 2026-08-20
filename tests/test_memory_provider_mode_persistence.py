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
    monkeypatch.setattr(server, "_set_session_context", lambda _key: None)
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
    monkeypatch.setattr(server, "_set_session_context", lambda _key: None)
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
