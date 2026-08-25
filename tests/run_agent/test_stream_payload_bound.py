"""#119/#195: truncate oversize streams and retain a resumable partial."""

from unittest.mock import patch


class _QueuedCodexClient:
    def __init__(self, notifications, server_requests=()):
        self._notifications = list(notifications)
        self._server_requests = list(server_requests)
        self.requests = []
        self.closed = False

    def initialize(self, **kwargs):
        return {}

    def request(self, method, params=None, timeout=30.0):
        self.requests.append((method, params or {}))
        if method == "thread/start":
            return {"thread": {"id": "thread-test"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-test"}}
        return {}

    def take_notification(self, timeout=0.0):
        if self._notifications:
            return self._notifications.pop(0)
        return None

    def take_server_request(self, timeout=0.0):
        if self._server_requests:
            return self._server_requests.pop(0)
        return None

    def is_alive(self):
        return not self.closed

    def stderr_tail(self, count=20):
        return []

    def respond_error(self, request_id, code, message, data=None):
        pass

    def close(self):
        self.closed = True


def _bound_notifications():
    return [
        {
            "method": "turn/started",
            "params": {
                "threadId": "thread-test",
                "turn": {"id": "turn-test"},
            },
        },
        {
            "method": "item/reasoning/delta",
            "params": {"delta": "r" * 90_000},
        },
        {
            "method": "item/agentMessage/delta",
            "params": {"delta": "v" * 50_000},
        },
        {
            "method": "item/completed",
            "params": {
                "threadId": "thread-test",
                "turnId": "turn-test",
                "item": {
                    "type": "agentMessage",
                    "id": "must-not-project",
                    "text": "must-not-append",
                },
            },
        },
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-test",
                "turn": {"id": "turn-test", "status": "completed"},
            },
        },
    ]


def test_live_writer_overflow_truncates_persists_status_and_continuation():
    from agent.stream_payload_bound import (
        DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
        persist_interrupted_stream_partial,
    )
    from run_agent import AIAgent

    assert DEFAULT_STREAM_PAYLOAD_BOUND_BYTES < 148_197

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._claim_stream_writer()
    agent.stream_delta_callback = lambda _t: None
    agent._stream_callback = None

    agent._record_streamed_assistant_text(
        "x" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES + 1)
    )
    assert len(agent._current_streamed_assistant_text.encode("utf-8")) == (
        DEFAULT_STREAM_PAYLOAD_BOUND_BYTES
    )

    messages = [{"role": "user", "content": "go"}]
    first = persist_interrupted_stream_partial(agent, messages, elapsed=596.8)
    assert messages[-1]["content"] == first
    assert messages[-1].get("status") == "stream_payload_limit"
    assert "continue" in first.lower()

    orphan = "y" * 148_197
    agent._current_streamed_assistant_text = orphan

    # Rapid second interrupt (<10s) with a tiny/empty payload must not leave
    # the 148k orphan as the last transcript message.
    agent._current_streamed_assistant_text = ""
    persist_interrupted_stream_partial(agent, messages, elapsed=0.1)
    last = messages[-1]
    assert last.get("role") == "assistant"
    assert orphan not in (last.get("content") or "")
    assert last.get("status") == "stream_payload_limit"


def test_reasoning_then_visible_stream_bytes_are_monotonic():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._claim_stream_writer()
    agent._fire_reasoning_delta("r" * 90_000)

    agent._record_streamed_assistant_text("v" * 50_000)

    assert agent._current_streamed_payload_bytes == 128 * 1024
    assert len(agent._current_streamed_assistant_text.encode("utf-8")) == 41_072
    assert agent._stream_payload_limit_error.status == "stream_payload_limit"


def test_codex_app_server_bridge_truncates_aggregate_stream_bound():
    from agent.codex_runtime import make_codex_app_server_event_bridge
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._claim_stream_writer()
    bridge = make_codex_app_server_event_bridge(agent)
    bridge({"method": "item/reasoning/delta", "params": {"delta": "r" * 90_000}})

    limit = bridge({"method": "item/agentMessage/delta", "params": {"delta": "v" * 50_000}})

    assert limit.status == "stream_payload_limit"
    assert agent._current_streamed_payload_bytes == 128 * 1024
    assert len(agent._current_streamed_assistant_text.encode("utf-8")) == 41_072


def test_codex_app_server_bound_persists_typed_partial_without_abort():
    from agent.codex_runtime import (
        make_codex_app_server_event_bridge,
        run_codex_app_server_turn,
    )
    from agent.transports.codex_app_server_session import CodexAppServerSession
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._claim_stream_writer()
    flushes = []

    def fake_flush(flushed_messages, conversation_history=None):
        flushes.append(list(flushed_messages))
        return True

    setattr(agent, "_session_db", object())
    setattr(agent, "_flush_messages_to_session_db", fake_flush)

    client = _QueuedCodexClient(_bound_notifications())
    agent._codex_session = CodexAppServerSession(
        cwd="/tmp",
        client_factory=lambda **_kwargs: client,
        on_event=make_codex_app_server_event_bridge(agent),
    )
    messages = [{"role": "user", "content": "go"}]
    result = run_codex_app_server_turn(
        agent,
        user_message="go",
        original_user_message="go",
        messages=messages,
        effective_task_id="task",
    )

    assert result["completed"] is False
    assert result["partial"] is True
    assert result["interrupted"] is False
    assert result["error"] is None
    assert result["status"] == "stream_payload_limit"
    assert messages[-1]["status"] == "stream_payload_limit"
    assert "continue" in messages[-1]["content"].lower()
    assert len(agent._current_streamed_assistant_text.encode("utf-8")) == 41_072
    assert flushes == [messages]

    follow_up_notifications = _bound_notifications()
    follow_up_notifications.pop(1)
    follow_up_notifications[1]["params"]["delta"] = "continued"
    follow_up_notifications[2]["params"]["item"]["text"] = "continued"
    follow_up_client = _QueuedCodexClient(follow_up_notifications)
    agent._codex_session = CodexAppServerSession(
        cwd="/tmp",
        client_factory=lambda **_kwargs: follow_up_client,
        on_event=make_codex_app_server_event_bridge(agent),
    )
    messages.append({"role": "user", "content": "continue"})

    follow_up = run_codex_app_server_turn(
        agent,
        user_message="continue",
        original_user_message="continue",
        messages=messages,
        effective_task_id="task",
    )

    assert follow_up["completed"] is True
    assert follow_up.get("status") != "stream_payload_limit"
    assert follow_up["final_response"] == "continued"
    assert agent._stream_payload_limit_error is None


def test_codex_app_server_approval_drain_returns_stream_limit():
    from agent.codex_runtime import make_codex_app_server_event_bridge
    from agent.transports.codex_app_server_session import CodexAppServerSession
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._claim_stream_writer()
    client = _QueuedCodexClient(
        _bound_notifications()[1:],
        server_requests=[
            {
                "id": "approval",
                "method": "unknown/request",
                "params": {},
            }
        ],
    )
    session = CodexAppServerSession(
        cwd="/tmp",
        client_factory=lambda **_kwargs: client,
        on_event=make_codex_app_server_event_bridge(agent),
    )

    turn = session.run_turn("go", turn_timeout=1.0, notification_poll_timeout=0.01)
    assert turn.status == "stream_payload_limit"
    assert turn.interrupted is False


def test_codex_app_server_compact_thread_returns_stream_limit():
    from agent.codex_runtime import make_codex_app_server_event_bridge
    from agent.transports.codex_app_server_session import CodexAppServerSession
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._claim_stream_writer()
    client = _QueuedCodexClient(_bound_notifications())
    session = CodexAppServerSession(
        cwd="/tmp",
        client_factory=lambda **_kwargs: client,
        on_event=make_codex_app_server_event_bridge(agent),
    )

    turn = session.compact_thread(turn_timeout=1.0, notification_poll_timeout=0.01)
    assert turn.status == "stream_payload_limit"
    assert turn.interrupted is False


def test_overflow_retains_partial_with_typed_stream_payload_limit():
    """#195: overflow keeps the partial, stamps stream_payload_limit, continues."""
    from agent.stream_payload_bound import (
        StreamPayloadBoundExceeded,
        persist_interrupted_stream_partial,
        resolve_stream_payload_bounds,
    )
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from run_agent import AIAgent

    limits = (DEFAULT_CONFIG.get("agent") or {}).get("stream_payload_limit") or {}
    assert limits.get("assistant_bytes") == 128 * 1024
    assert limits.get("reasoning_bytes") == 128 * 1024
    assistant_bound, reasoning_bound = resolve_stream_payload_bounds()
    assert assistant_bound == limits["assistant_bytes"]
    assert reasoning_bound == limits["reasoning_bytes"]

    err = StreamPayloadBoundExceeded(200, 64)
    assert err.status == "stream_payload_limit"

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._claim_stream_writer()
    partial = "kept-partial"
    agent._current_streamed_assistant_text = partial
    messages = [{"role": "user", "content": "go"}]
    first = persist_interrupted_stream_partial(
        agent, messages, elapsed=1.0, exceeded=True, size=200, bound=64
    )
    assert partial in first
    last = messages[-1]
    assert last.get("role") == "assistant"
    assert last.get("content") == first
    assert last.get("status") == "stream_payload_limit"
    assert "continue" in first.lower()


def test_reasoning_bound_is_independent_of_assistant_bound(monkeypatch):
    """#195: reasoning vs final overflow use separate configurable byte caps."""
    from run_agent import AIAgent

    monkeypatch.setattr(
        "agent.stream_payload_bound.resolve_stream_payload_bounds",
        lambda: (20, 80),
    )
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._claim_stream_writer()
    agent._fire_reasoning_delta("r" * 50)
    assert agent._current_streamed_payload_bytes == 50
    assistant_limit = agent._record_streamed_assistant_text("v" * 21)
    assert assistant_limit.status == "stream_payload_limit"
    assert agent._current_streamed_assistant_text == "v" * 20

    agent._reset_stream_delivery_tracking()
    agent._fire_reasoning_delta("r" * 50)
    reasoning_limit = agent._fire_reasoning_delta("r" * 31)
    assert reasoning_limit.status == "stream_payload_limit"
    assert agent._current_streamed_reasoning_bytes == 80


def test_stream_payload_limit_explainer_is_not_an_aborted_turn():
    from run_agent import AIAgent

    text = AIAgent._format_turn_completion_explanation("stream_payload_limit")
    assert "continue" in text.lower()
    assert "aborted" not in text.lower()


def test_conversation_loop_stream_limit_is_partial_not_interrupted():
    from agent.stream_payload_bound import DEFAULT_STREAM_PAYLOAD_BOUND_BYTES
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    setattr(agent, "_disable_streaming", True)

    def overflow(_api_kwargs):
        error = agent._record_streamed_assistant_text(
            "x" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES + 1)
        )
        assert error is not None
        raise error

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=overflow),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("go")

    assert result["turn_exit_reason"] == "stream_payload_limit"
    assert result["interrupted"] is False
    assert result["messages"][-1]["status"] == "stream_payload_limit"
    assert "continue" in result["final_response"].lower()
