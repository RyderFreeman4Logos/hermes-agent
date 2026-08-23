import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acp_adapter.session import _merge_session_memory_provider_mode
from agent.memory_provider import normalize_memory_provider_mode
from hermes_cli.cli_commands_mixin import CLICommandsMixin
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
