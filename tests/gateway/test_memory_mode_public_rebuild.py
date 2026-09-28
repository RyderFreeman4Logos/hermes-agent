"""Public messaging-turn coverage for durable memory-mode reconstruction."""

import asyncio
import json
import sys
import types
from types import SimpleNamespace

from agent.inline_tool_executors import InlineToolContext, _memory
from agent.memory_manager import MemoryManager
from gateway.config import Platform
from gateway.session import SessionSource
from hermes_state import SessionDB
from tools.memory_tool import MemoryStore

from tests.gateway.test_compression_failure_session_sync import (
    SESSION_KEY,
    _SessionStore,
    _runner,
)


class _Provider:
    name = "synthetic-provider"

    def __init__(self, sink):
        self.sink = sink

    def get_tool_schemas(self):
        return []

    def authoritative_memory_write(self, request, **_kwargs):
        self.sink.append(("authoritative", request["content"]))
        return json.dumps({"success": True, "operation_id": "synthetic"})

    def on_memory_write(self, _action, _target, content, **_kwargs):
        self.sink.append(("hybrid", content))


class _GatewayMemoryAgent:
    created = []
    writes = []

    def __init__(self, **kwargs):
        self.session_id = kwargs["session_id"]
        self.model = kwargs["model"]
        self.provider = kwargs.get("provider")
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")
        self.api_mode = kwargs.get("api_mode")
        self.tools = []
        self._memory_provider_mode = kwargs.get("memory_provider_mode_override") or "authoritative"
        self._session_init_model_config = {
            "memory_provider_mode": self._memory_provider_mode,
        }
        self._memory_store = MemoryStore()
        self._memory_store.load_from_disk()
        self._memory_manager = MemoryManager(provider_mode=self._memory_provider_mode)
        self._memory_manager.add_provider(_Provider(type(self).writes))
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=200000)
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        type(self).created.append(self)

    def _build_memory_write_metadata(self, **kwargs):
        return kwargs

    def clear_interrupt(self):
        return None

    def interrupt(self, *_args, **_kwargs):
        return None

    def release_clients(self):
        return None

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **_kwargs):
        result = json.loads(_memory(
            self,
            {"action": "add", "target": "memory", "content": f"{user_message} fact"},
            InlineToolContext(effective_task_id=task_id or "gateway", tool_call_id="memory"),
        ))
        assert result["success"] is True
        response = f"answered {user_message}"
        return {
            "final_response": response,
            "messages": [
                *(conversation_history or []),
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response},
            ],
            "api_calls": 0,
        }


def _run_turn(runner, source, session_id, message):
    return asyncio.run(runner._run_agent(
        message=message,
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key=SESSION_KEY,
    ))


def test_public_gateway_two_turns_rebuild_from_durable_memory_mode(monkeypatch, tmp_path):
    """A cache eviction must not let the changed profile own the second turn."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("tools.write_approval.write_approval_enabled", lambda _subsystem: False)
    _GatewayMemoryAgent.created = []
    _GatewayMemoryAgent.writes = []
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _GatewayMemoryAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5.4")
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools", lambda *_args, **_kwargs: {"core"}
    )

    session_id = "durable-gateway-hybrid"
    db = SessionDB(tmp_path / "state.db")
    db.create_session(
        session_id, source="telegram", model="synthetic-model",
        model_config={"memory_provider_mode": "hybrid"},
    )
    store = _SessionStore()
    store.entry.session_id = session_id
    runner = _runner(store)
    runner._session_db = SimpleNamespace(_db=db)
    runner._spawn_release_thread = lambda target, args, _name, **_kwargs: target(*args)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="synthetic-chat",
        chat_type="dm",
        user_id="synthetic-user",
    )
    try:
        first = _run_turn(runner, source, session_id, "first")
        assert first["final_response"] == "answered first"
        assert SESSION_KEY in runner._agent_cache
        first_agent = _GatewayMemoryAgent.created[-1]

        runner._evict_cached_agent(SESSION_KEY)
        assert SESSION_KEY not in runner._agent_cache

        second = _run_turn(runner, source, session_id, "second")
        assert second["final_response"] == "answered second"
        assert _GatewayMemoryAgent.created[-1] is not first_agent
        assert [agent._memory_provider_mode for agent in _GatewayMemoryAgent.created] == [
            "hybrid", "hybrid"
        ]
        assert _GatewayMemoryAgent.writes == [
            ("hybrid", "first fact"),
            ("hybrid", "second fact"),
        ]
    finally:
        db.close()
