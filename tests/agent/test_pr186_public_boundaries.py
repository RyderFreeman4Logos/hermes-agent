"""Connected public regressions for PR #186 spill and stream boundaries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.codex_runtime import make_codex_app_server_event_bridge
from agent.stream_payload_bound import DEFAULT_STREAM_PAYLOAD_BOUND_BYTES
from agent.transports.codex_app_server_session import CodexAppServerSession
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.file_operations import ShellFileOperations
from tools.tool_result_storage import extract_persisted_path


@pytest.fixture(autouse=True)
def _disable_background_title(monkeypatch):
    monkeypatch.setattr(
        "agent.title_generator.maybe_auto_title", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "agent.retry_utils.jittered_backoff", lambda *_args, **_kwargs: 0.0
    )
    monkeypatch.setattr(
        "agent.turn_recovery.interruptible_backoff_sleep",
        lambda *_args, **_kwargs: False,
    )


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "synthetic test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _agent(tmp_path: Path, monkeypatch, *, api_mode: str = "chat_completions") -> AIAgent:
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **_kwargs: _tool_defs("terminal", "custom_json"),
    )
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.model_metadata.fetch_model_metadata", lambda *_a, **_k: {})
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://synthetic.invalid/v1",
            provider="openai",
            api_mode=api_mode,
            model="test/model",
            quiet_mode=True,
            max_iterations=3,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._cleanup_task_resources = lambda _task_id: None
    return agent


def _attach_db(agent: AIAgent, db_path: Path, session_id: str) -> SessionDB:
    db = SessionDB(db_path=db_path)
    db.create_session(session_id, "test", model="test/model")
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False
    return db


def _tool_call(name: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _response(*, text: str = "", tool_calls=None, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        model="test/model",
        usage=None,
    )


class _RemoteSync:
    def __init__(self, host_home: Path, remote_home: Path, *, readable: bool) -> None:
        self.host_home = host_home
        self.remote_home = remote_home
        self.readable = readable

    def sync(self, *, force: bool = False) -> None:
        del force
        if not self.readable:
            return
        source = self.host_home / "cache" / "spillover"
        target = self.remote_home / ".hermes" / "cache" / "spillover"
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)


class _RemoteEnvironment:
    """Disposable remote namespace backed by a real local shell and filesystem."""

    def __init__(self, root: Path, host_home: Path, *, readable: bool) -> None:
        self.root = root
        self.home = root / "home"
        self.home.mkdir(parents=True)
        self.cwd = str(root)
        self._sync_manager = _RemoteSync(host_home, self.home, readable=readable)

    def get_temp_dir(self) -> str:
        return str(self.root / "tmp")

    def execute(self, command: str, cwd=None, timeout=None, stdin_data=None) -> dict:
        run_cwd = Path(cwd or self.cwd)
        if not run_cwd.is_dir():
            run_cwd = self.root
        env = dict(os.environ)
        env["HOME"] = str(self.home)
        completed = subprocess.run(
            ["bash", "-c", command],
            cwd=run_cwd,
            env=env,
            input=stdin_data,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "output": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }


@pytest.mark.parametrize(
    "case,tool_name,raw_size,host_failure,remote_readable",
    [
        ("B1-named", "terminal", 40_000, False, True),
        ("B1-json", "custom_json", 110_000, False, True),
        ("B2-host-failure", "terminal", 40_000, True, True),
        ("B3-mapping-failure", "terminal", 40_000, False, False),
    ],
)
def test_public_turn_spill_survives_db_reload_and_remote_recreation(
    tmp_path,
    monkeypatch,
    case,
    tool_name,
    raw_size,
    host_failure,
    remote_readable,
):
    """B1-B3: public turn, first durable row, cold DB and remote readback."""
    agent = _agent(tmp_path, monkeypatch)
    db_path = tmp_path / "state.db"
    session_id = f"pr186-{case}"
    db = _attach_db(agent, db_path, session_id)
    host_home = Path(os.environ["HERMES_HOME"])
    monkeypatch.setenv("TERMINAL_ENV", "fixture186")
    first_remote = _RemoteEnvironment(tmp_path / "remote-one", host_home, readable=remote_readable)
    monkeypatch.setattr(
        "agent.terminal_env_registry.provider_flag",
        lambda _backend, key, default=None: (
            str(first_remote.home / ".hermes") if key == "cache_path_base" else default
        ),
    )
    raw = (
        json.dumps({"items": ["B1_JSON_TOKEN", "x" * (raw_size - 40)]})
        if tool_name == "custom_json"
        else "B_PUBLIC_TOKEN\n" + ("x" * (raw_size - 16)) + "\n"
    )
    call_id = f"call-{case}"
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call(tool_name, call_id)], finish_reason="tool_calls"),
        _response(text="done"),
    ]
    if host_failure:
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("block", encoding="utf-8")
        monkeypatch.setattr(
            "tools.tool_result_storage.get_spillover_dir",
            lambda: blocker / "spillover",
        )

    with (
        patch("model_tools.handle_function_call", return_value=raw),
        patch("agent.tool_executor.get_active_env", return_value=first_remote),
        patch.object(agent, "_spawn_background_review", return_value=None),
    ):
        result = agent.run_conversation("produce the synthetic result")
    assert result["completed"] is True
    db.close()

    reloaded = SessionDB(db_path=db_path)
    try:
        rows = reloaded.get_messages_as_conversation(session_id)
    finally:
        reloaded.close()
    stored = next(row["content"] for row in rows if row.get("role") == "tool")
    persisted_path = extract_persisted_path(stored)

    if case.startswith("B1-"):
        assert persisted_path is not None
        canonical = host_home / "cache" / "spillover" / f"{call_id}.txt"
        assert canonical.read_text(encoding="utf-8") == raw
        remote_root = first_remote.root
        shutil.rmtree(remote_root)
        recreated = _RemoteEnvironment(remote_root, host_home, readable=True)
        recreated._sync_manager.sync(force=True)
        recovered = ShellFileOperations(recreated).read_file_raw(persisted_path)
        assert recovered.error is None
        assert recovered.content == raw
    else:
        assert persisted_path is None
        assert stored == raw
        if case.startswith("B3-"):
            canonical = host_home / "cache" / "spillover" / f"{call_id}.txt"
            assert canonical.read_text(encoding="utf-8") == raw


def _chat_chunk(content=None, tool_calls=None, finish_reason=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content=None,
                    reasoning=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        model="test/model",
        usage=None,
    )


def _tool_delta(*, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=0,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _ClosableStream:
    def __init__(self, events, *, final_response=None):
        self.events = list(events)
        self.final_response = final_response
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


def _run_public_chat_overflow(tmp_path, monkeypatch, case: str):
    agent = _agent(tmp_path, monkeypatch)
    db = _attach_db(agent, tmp_path / "state.db", f"pr186-{case}")
    agent._disable_streaming = False
    agent.stream_delta_callback = lambda _text: None
    agent._execute_tool_calls = MagicMock()
    attempts = {"count": 0}

    if case == "C2-suppressed":
        events = [
            _chat_chunk(tool_calls=[_tool_delta(call_id="c2", name="terminal", arguments="{}")]),
            _chat_chunk(content="x" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES + 1)),
            _chat_chunk(finish_reason="tool_calls"),
        ]
        retries = "0"
    else:
        accepted = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES - (8 if case == "C5-retry-notice" else 0)

        def events_for_attempt():
            yield _chat_chunk(content="x" * accepted)
            yield _chat_chunk(tool_calls=[_tool_delta(call_id="c5", name="terminal", arguments='{')])
            raise httpx.RemoteProtocolError("synthetic transport drop")

        events = events_for_attempt()
        retries = "1" if case == "C5-retry-notice" else "0"

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: attempts.__setitem__("count", attempts["count"] + 1)
                or iter(events)
            )
        ),
        close=lambda: None,
    )
    monkeypatch.setenv("HERMES_STREAM_RETRIES", retries)
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **_kwargs: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *_a, **_k: None)
    monkeypatch.setattr(agent, "_abort_request_openai_client", lambda *_a, **_k: None)
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("stream the synthetic response")
    db.close()
    return agent, result, attempts["count"]


@pytest.mark.parametrize(
    "case",
    ["C2-suppressed", "C5-retry-notice", "C5-partial-tool-warning"],
)
def test_public_chat_overflow_stops_before_dispatch_retry_or_partial_warning(
    tmp_path, monkeypatch, case
):
    """C2/C5: producer-to-recorder overflow is terminal at the public turn."""
    agent, result, attempts = _run_public_chat_overflow(tmp_path, monkeypatch, case)
    assert attempts == 1
    agent._execute_tool_calls.assert_not_called()
    assert result["completed"] is False
    assert "Streamed assistant payload exceeded 262144 bytes" in result["final_response"]
    assert len(result["final_response"].encode("utf-8")) < 1024


class _AppServerClient:
    def __init__(self, notifications, server_requests=()):
        self.notifications = list(notifications)
        self.server_requests = list(server_requests)
        self.requests = []
        self.responses = []
        self.closed = False

    def initialize(self, **_kwargs):
        return {}

    def request(self, method, params=None, timeout=30):
        del timeout
        self.requests.append((method, params or {}))
        if method == "thread/start":
            return {"thread": {"id": "thread-186"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-186"}}
        return {}

    def take_notification(self, timeout=0):
        del timeout
        return self.notifications.pop(0) if self.notifications else None

    def take_server_request(self, timeout=0):
        del timeout
        return self.server_requests.pop(0) if self.server_requests else None

    def respond(self, request_id, result):
        self.responses.append((request_id, result))

    def respond_error(self, request_id, code, message, data=None):
        self.responses.append((request_id, {"code": code, "message": message, "data": data}))

    def is_alive(self):
        return not self.closed

    def stderr_tail(self, _n=20):
        return []

    def close(self):
        self.closed = True


@pytest.mark.parametrize("pending_approval", [False, True])
def test_public_app_server_overflow_persists_projection_and_skips_later_work(
    tmp_path, monkeypatch, pending_approval
):
    """C3: public app-server turn uses the real bridge, recorder, projector and DB."""
    agent = _agent(tmp_path, monkeypatch, api_mode="codex_app_server")
    db_path = tmp_path / "state.db"
    db = _attach_db(agent, db_path, f"pr186-c3-{pending_approval}")
    agent.stream_delta_callback = lambda _text: None
    notifications = [
        {
            "method": "item/completed",
            "params": {
                "threadId": "thread-186",
                "turnId": "turn-186",
                "item": {"type": "agentMessage", "id": "accepted", "text": "accepted projection"},
            },
        },
        {
            "method": "item/agentMessage/delta",
            "params": {
                "threadId": "thread-186",
                "turnId": "turn-186",
                "delta": "x" * DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
            },
        },
        {
            "method": "item/agentMessage/delta",
            "params": {"threadId": "thread-186", "turnId": "turn-186", "delta": "y"},
        },
        {
            "method": "turn/completed",
            "params": {"threadId": "thread-186", "turn": {"id": "turn-186", "status": "completed"}},
        },
    ]
    server_requests = (
        [
            {
                "id": "approval-after-overflow",
                "method": "item/commandExecution/requestApproval",
                "params": {"command": "pwd", "cwd": "/tmp"},
            }
        ]
        if pending_approval
        else []
    )
    client = _AppServerClient(notifications, server_requests)
    agent._codex_session = CodexAppServerSession(
        cwd=str(tmp_path),
        client_factory=lambda **_kwargs: client,
        on_event=make_codex_app_server_event_bridge(agent),
    )
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("app-server synthetic overflow")
    db.close()

    assert result["completed"] is False
    assert "262145 bytes" in result["error"]
    assert [method for method, _params in client.requests].count("turn/interrupt") == 1
    assert client.responses == []
    assert agent._codex_session is None
    assert [note["method"] for note in client.notifications] == ["turn/completed"]
    reloaded = SessionDB(db_path=db_path)
    try:
        contents = [row["content"] for row in reloaded.get_messages_as_conversation(agent.session_id)]
    finally:
        reloaded.close()
    assert contents.count("accepted projection") == 1
    explanations = [content for content in contents if "262145 bytes" in str(content)]
    assert len(explanations) == 1


def test_public_codex_relay_final_response_cannot_mask_recorder_overflow(
    tmp_path, monkeypatch
):
    """C4: populated Relay final_response never overrides the typed overflow."""
    agent = _agent(tmp_path, monkeypatch, api_mode="codex_responses")
    db = _attach_db(agent, tmp_path / "state.db", "pr186-c4")
    agent.stream_delta_callback = lambda _text: None
    completed = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="completed",
                content=[SimpleNamespace(type="output_text", text="must not win")],
            )
        ],
        usage=None,
        status="completed",
        model="test/model",
    )
    stream = _ClosableStream(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="x" * DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
            ),
            SimpleNamespace(type="response.output_text.delta", delta="y"),
            SimpleNamespace(type="response.completed", response=completed),
        ],
        final_response=completed,
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_kwargs: stream),
        close=lambda: None,
    )
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **_kwargs: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *_a, **_k: None)
    monkeypatch.setattr(agent, "_abort_request_openai_client", lambda *_a, **_k: None)
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("codex synthetic overflow")
    db.close()

    assert result["completed"] is False
    assert result["final_response"] != "must not win"
    assert "262145 bytes" in result["final_response"]
    assert stream.closed is True
