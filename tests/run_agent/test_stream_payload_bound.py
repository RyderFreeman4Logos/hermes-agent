"""#119: abort an oversize streamed turn; do not persist a 148k interrupt orphan."""

import pytest


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


def test_oversize_stream_aborts_and_interrupt_does_not_leave_orphan():
    from agent.stream_payload_bound import (
        DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
        StreamPayloadBoundExceeded,
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

    with pytest.raises(StreamPayloadBoundExceeded, match="exceeded"):
        agent._record_streamed_assistant_text(
            "x" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES + 1)
        )

    orphan = "y" * 148_197
    agent._current_streamed_assistant_text = orphan
    messages = [{"role": "user", "content": "go"}]
    first = persist_interrupted_stream_partial(agent, messages, elapsed=596.8)
    assert "exceeded" in first.lower()
    assert orphan not in first
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == first
    assert len(first.encode("utf-8")) < DEFAULT_STREAM_PAYLOAD_BOUND_BYTES

    # Rapid second interrupt (<10s) with a tiny/empty payload must not leave
    # the 148k orphan as the last transcript message.
    agent._current_streamed_assistant_text = ""
    persist_interrupted_stream_partial(agent, messages, elapsed=0.1)
    last = messages[-1]
    assert last.get("role") == "assistant"
    assert orphan not in (last.get("content") or "")
    assert len((last.get("content") or "").encode("utf-8")) < 1024


def test_reasoning_then_visible_stream_bytes_are_monotonic():
    from agent.stream_payload_bound import StreamPayloadBoundExceeded
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

    with pytest.raises(StreamPayloadBoundExceeded, match="exceeded"):
        agent._record_streamed_assistant_text("v" * 50_000)

    assert agent._current_streamed_payload_bytes == 90_000
    assert agent._current_streamed_assistant_text == ""


def test_codex_app_server_bridge_propagates_aggregate_stream_bound():
    from agent.codex_runtime import make_codex_app_server_event_bridge
    from agent.stream_payload_bound import StreamPayloadBoundExceeded
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

    with pytest.raises(StreamPayloadBoundExceeded, match="exceeded"):
        bridge({"method": "item/agentMessage/delta", "params": {"delta": "v" * 50_000}})

    assert agent._current_streamed_payload_bytes == 90_000
    assert agent._current_streamed_assistant_text == ""


def test_codex_app_server_bound_aborts_before_projection_or_persistence():
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
    assert "exceeded" in result["error"]
    assert messages == [{"role": "user", "content": "go"}]
    assert flushes == []


def test_codex_app_server_approval_drain_propagates_stream_bound():
    from agent.codex_runtime import make_codex_app_server_event_bridge
    from agent.stream_payload_bound import StreamPayloadBoundExceeded
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

    with pytest.raises(StreamPayloadBoundExceeded, match="exceeded"):
        session.run_turn("go", turn_timeout=1.0, notification_poll_timeout=0.01)


def test_codex_app_server_compact_thread_propagates_stream_bound():
    from agent.codex_runtime import make_codex_app_server_event_bridge
    from agent.stream_payload_bound import StreamPayloadBoundExceeded
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

    with pytest.raises(StreamPayloadBoundExceeded, match="exceeded"):
        session.compact_thread(turn_timeout=1.0, notification_poll_timeout=0.01)
