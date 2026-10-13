"""Public cold-session approval routes for authoritative memory writes."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest


_PROVIDER_SOURCE = '''
import json
from pathlib import Path

from agent.memory_provider import MemoryProvider


class SyntheticApprovalProvider(MemoryProvider):
    @property
    def name(self):
        return "synthetic_approval"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.home = Path(kwargs["hermes_home"])
        event = {
            "session_id": session_id,
            "platform": kwargs.get("platform"),
            "user_id": kwargs.get("user_id"),
        }
        with (self.home / "approval-init.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\\n")

    def get_tool_schemas(self):
        return []

    def authoritative_memory_write(self, request, **kwargs):
        with (self.home / "approval-writes.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(request, sort_keys=True) + "\\n")
        return json.dumps({"success": True, "operation_id": "synthetic-operation"})
'''


@pytest.fixture()
def authoritative_profile(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    plugin = home / "plugins" / "synthetic_approval"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(_PROVIDER_SOURCE, encoding="utf-8")
    (home / "config.yaml").write_text(
        "memory:\n"
        "  provider: synthetic_approval\n"
        "  provider_mode: authoritative\n"
        "  write_approval: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _stage_authoritative(content: str) -> str:
    from tools import write_approval as wa

    record = wa.stage_write(
        wa.MEMORY,
        {
            "action": "add",
            "target": "memory",
            "content": content,
            "memory_provider_mode": "authoritative",
        },
        summary="synthetic authoritative approval",
        origin="foreground",
    )
    return record["id"]


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_authoritative_write(home: Path, content: str, *, platform: str, session_id: str) -> None:
    writes = _jsonl(home / "approval-writes.jsonl")
    assert writes[-1] == {"action": "add", "content": content, "target": "memory"}
    initialized = _jsonl(home / "approval-init.jsonl")
    assert initialized[-1]["platform"] == platform
    assert initialized[-1]["session_id"] == session_id
    assert not (home / "memories" / "MEMORY.md").exists(), (
        "authoritative approval must never fall back to the Markdown store"
    )


def test_cold_cli_memory_approve_resolves_authoritative_provider(authoritative_profile, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from tools import write_approval as wa

    pending_id = _stage_authoritative("cold cli fact")
    cli = CLICommandsMixin.__new__(CLICommandsMixin)
    cli.agent = None
    cli.session_id = "cli-session"

    cli._handle_memory_command(f"/memory approve {pending_id}")

    output = capsys.readouterr().out
    assert "Approved 1" in output
    assert wa.get_pending(wa.MEMORY, pending_id) is None
    _assert_authoritative_write(
        authoritative_profile, "cold cli fact", platform="cli", session_id="cli-session"
    )


@pytest.mark.asyncio
async def test_cold_messaging_memory_approve_resolves_authoritative_provider(
    authoritative_profile, monkeypatch
):
    import hermes_state
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore
    from hermes_state import AsyncSessionDB
    from tools import write_approval as wa

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", authoritative_profile / "state.db")
    store = SessionStore(sessions_dir=authoritative_profile, config=GatewayConfig())
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="synthetic-user",
        chat_id="synthetic-chat",
        chat_type="dm",
    )
    entry = store.get_or_create_session(source)
    pending_id = _stage_authoritative("cold messaging fact")
    runner = object.__new__(GatewayRunner)
    runner.config = {}
    runner.adapters = {}
    runner.session_store = store
    runner._session_db = AsyncSessionDB(store._db)
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()

    output = await runner._handle_memory_command(
        MessageEvent(
            text=f"/memory approve {pending_id}",
            source=source,
            message_id="synthetic-message",
        )
    )

    assert "Approved 1" in output
    assert wa.get_pending(wa.MEMORY, pending_id) is None
    _assert_authoritative_write(
        authoritative_profile,
        "cold messaging fact",
        platform="telegram",
        session_id=entry.session_id,
    )


def test_tui_jsonrpc_slash_exec_cold_approval_uses_profile_provider(authoritative_profile):
    from hermes_state import SessionDB
    from tools import write_approval as wa
    from tui_gateway import server

    session_key = "tui-stored-session"
    runtime_id = "tui-runtime-session"
    db = SessionDB(authoritative_profile / "state.db")
    db.create_session(
        session_key,
        source="tui",
        model="synthetic-model",
        model_config={"memory_provider_mode": "authoritative"},
    )
    pending_id = _stage_authoritative("cold tui fact")
    session = {
        "session_key": session_key,
        "agent": None,
        "slash_worker": None,
        "profile_home": str(authoritative_profile),
    }
    server._sessions[runtime_id] = session
    try:
        response = server.handle_request(
            {
                "id": "approval",
                "method": "slash.exec",
                "params": {
                    "session_id": runtime_id,
                    "command": f"/memory approve {pending_id}",
                },
            }
        )
    finally:
        worker = session.get("slash_worker")
        if worker is not None:
            worker.close()
        server._sessions.pop(runtime_id, None)
        db.close()

    assert "error" not in response, response
    assert "Approved 1" in response["result"]["output"]
    assert wa.get_pending(wa.MEMORY, pending_id) is None
    _assert_authoritative_write(
        authoritative_profile,
        "cold tui fact",
        platform="cli",
        session_id=session_key,
    )
