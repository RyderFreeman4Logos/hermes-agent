import base64
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource


_ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6L2ioAAAAASUVORK5CYII="
)


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="sent-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class CaptureQueuedNativeImageAgent:
    calls = []

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls.append(message)
        return {
            "final_response": f"done-{len(type(self).calls)}",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_kw: "native"
    return runner


@pytest.mark.asyncio
async def test_queued_followup_uses_pending_event_session_key_for_native_images(monkeypatch, tmp_path):
    CaptureQueuedNativeImageAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)

    image_path = tmp_path / "queued-image.png"
    image_path.write_bytes(_ONE_BY_ONE_PNG)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )
    pending_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )

    adapter._pending_messages["agent:main:telegram:group:-1001"] = MessageEvent(
        text="describe this",
        message_type=MessageType.PHOTO,
        source=pending_source,
        media_urls=[str(image_path)],
        media_types=["image/png"],
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-native-image-followup",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "done-2"
    assert len(CaptureQueuedNativeImageAgent.calls) == 2
    queued_message = CaptureQueuedNativeImageAgent.calls[1]
    assert isinstance(queued_message, list)
    assert queued_message[0]["type"] == "text"
    assert queued_message[0]["text"].startswith("describe this")
    assert any(part.get("type") == "image_url" for part in queued_message)


@pytest.mark.asyncio
async def test_recursive_queued_followup_transfers_delivery_claim(monkeypatch, tmp_path):
    CaptureQueuedNativeImageAgent.calls = []
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="123", chat_type="dm"
    )
    claimed_event = {"type": "completion", "session_id": "proc-queued-claim"}
    renewal_stop = MagicMock()
    completed = []
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda event, claim: completed.append((event, claim)),
    )
    adapter._pending_messages["agent:main:telegram:dm:123"] = MessageEvent(
        text="completion plus user text",
        source=source,
        internal=True,
        metadata={
            "delivery_claims": [(claimed_event, "claim-1")],
            "delivery_renewal_stops": [renewal_stop],
        },
    )

    result = await runner._run_agent(
        message="first turn",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-queued-claim",
        session_key="agent:main:telegram:dm:123",
    )

    assert result["final_response"] == "done-2"
    assert completed == [(claimed_event, "claim-1")]
    renewal_stop.set.assert_called_once_with()


@pytest.mark.asyncio
async def test_recursion_cap_requeues_before_staged_fifo_head_with_metadata(
    monkeypatch, tmp_path
):
    CaptureQueuedNativeImageAgent.calls = []
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
    session_key = "agent:main:telegram:dm:123"
    claim = ({"type": "completion", "session_id": "first"}, "claim-first")
    renewal_stop = MagicMock()
    first = MessageEvent(
        text="first",
        source=source,
        internal=True,
        metadata={
            "delivery_claims": [claim],
            "delivery_renewal_stops": [renewal_stop],
            "marker": object(),
        },
    )
    second = MessageEvent(text="second", source=source, metadata={"marker": object()})
    runner._enqueue_fifo(session_key, first, adapter)
    runner._enqueue_fifo(session_key, second, adapter)

    await runner._run_agent(
        message="current",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-recursion-cap",
        session_key=session_key,
        _interrupt_depth=runner._MAX_INTERRUPT_DEPTH,
    )

    assert adapter._pending_messages[session_key] is first
    assert runner._queued_events[session_key] == [second]
    assert first.metadata["delivery_claims"] == [claim]
    assert first.metadata["delivery_renewal_stops"] == [renewal_stop]


@pytest.mark.asyncio
async def test_shutdown_after_dequeue_stops_renewal_and_releases_exact_claim(
    monkeypatch, tmp_path
):
    CaptureQueuedNativeImageAgent.calls = []
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    released = []
    monkeypatch.setattr(
        "tools.async_delegation.release_event_delivery",
        lambda event, claim: released.append((event, claim)),
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    runner._draining = True
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
    session_key = "agent:main:telegram:dm:123"
    claimed_event = {"type": "completion", "session_id": "abandoned"}
    renewal_stop = MagicMock()
    adapter._pending_messages[session_key] = MessageEvent(
        text="completion",
        source=source,
        internal=True,
        metadata={
            "delivery_claims": [(claimed_event, "claim-abandoned")],
            "delivery_renewal_stops": [renewal_stop],
        },
    )

    await runner._run_agent(
        message="current",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-shutdown",
        session_key=session_key,
    )

    renewal_stop.set.assert_called_once_with()
    assert released == [(claimed_event, "claim-abandoned")]


@pytest.mark.asyncio
async def test_shutdown_claim_cleanup_continues_after_each_failure(monkeypatch, tmp_path):
    CaptureQueuedNativeImageAgent.calls = []
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    releases = []

    def release(event, claim):
        releases.append((event, claim))
        if claim == "claim-first":
            raise RuntimeError("release failed")

    monkeypatch.setattr("tools.async_delegation.release_event_delivery", release)

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    runner._draining = True
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
    session_key = "agent:main:telegram:dm:123"
    first_event = {"type": "completion", "session_id": "first"}
    second_event = {"type": "completion", "session_id": "second"}
    first_stop = MagicMock()
    first_stop.set.side_effect = RuntimeError("stop failed")
    second_stop = MagicMock()
    adapter._pending_messages[session_key] = MessageEvent(
        text="completions",
        source=source,
        internal=True,
        metadata={
            "delivery_claims": [
                (first_event, "claim-first"),
                (second_event, "claim-second"),
            ],
            "delivery_renewal_stops": [first_stop, second_stop],
        },
    )

    await runner._run_agent(
        message="current",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-shutdown-failure",
        session_key=session_key,
    )

    first_stop.set.assert_called_once_with()
    second_stop.set.assert_called_once_with()
    assert releases == [
        (first_event, "claim-first"),
        (second_event, "claim-second"),
    ]
