"""Hostile convergence matrix for bounded streamed delivery and partials."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.stream_payload_bound import (
    DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
    StreamPayloadBoundExceeded,
    streamed_payload_bytes,
)
from agent.transports.codex_app_server_session import TurnResult
from tests.agent.transports.test_codex_app_server_session import FakeClient, make_session
from tests.run_agent.test_codex_app_server_integration import _make_codex_agent
from tests.run_agent.test_streaming import _make_stream_chunk, _make_tool_call_delta


def _agent(*, display=None):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://stub.invalid/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        stream_delta_callback=display,
    )
    agent._interrupt_requested = False
    return agent


@pytest.mark.parametrize("payload", ["xx", "é"], ids=["ascii", "utf8"])
@pytest.mark.parametrize(
    "sink",
    ["display", "tts", "both", "all-fail", "plugin-only", "no-consumer"],
)
def test_stream_payload_bound_precedes_every_sink(monkeypatch, sink, payload):
    display, tts, plugin = [], [], []

    def observe(target):
        def callback(text):
            target.append(text)
            if sink == "all-fail":
                raise RuntimeError("sink failed")

        return callback

    agent = _agent(
        display=observe(display) if sink in {"display", "both", "all-fail"} else None
    )
    agent._stream_callback = (
        observe(tts) if sink in {"tts", "both", "all-fail"} else None
    )
    monkeypatch.setattr(
        "agent.plugin_stream_hooks.enqueue_plugin_stream_hook",
        lambda hook, **data: plugin.append(data["delta"])
        if sink == "plugin-only" and hook == "on_stream_delta"
        else None,
    )
    agent._current_streamed_payload_bytes = (
        DEFAULT_STREAM_PAYLOAD_BOUND_BYTES - streamed_payload_bytes(payload) + 1
    )

    with pytest.raises(StreamPayloadBoundExceeded):
        agent._fire_stream_delta(payload)

    assert display == []
    assert tts == []
    assert plugin == []
    assert agent._current_streamed_assistant_text == ""


def test_scrubber_flush_tail_is_admitted_before_egress():
    displayed = []
    agent = _agent(display=displayed.append)
    agent._current_streamed_payload_bytes = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES

    assert agent._fire_stream_delta("<") is False
    with pytest.raises(StreamPayloadBoundExceeded):
        agent._reset_stream_delivery_tracking()

    assert displayed == []


@pytest.mark.parametrize(
    "delta",
    [
        {"text": "x" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES + 1)},
        {"reasoningContent": {"text": "é" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES // 2 + 1)}},
    ],
    ids=["text", "reasoning"],
)
def test_bedrock_no_consumer_payload_is_bounded(delta):
    pytest.importorskip("botocore.exceptions")
    agent = _agent()
    agent.api_mode = "bedrock_converse"
    agent.reasoning_callback = None
    client = MagicMock()
    client.converse_stream.return_value = {
        "stream": iter([{"contentBlockDelta": {"delta": delta}}])
    }

    with patch(
        "agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client
    ):
        with pytest.raises(StreamPayloadBoundExceeded):
            agent._interruptible_streaming_api_call(
                {"modelId": "test/model", "messages": []}
            )


@pytest.mark.parametrize("owner", ["normal", "approval-drain", "compact"])
def test_stream_payload_bound_escapes_app_server_event_owners(owner):
    overflow = StreamPayloadBoundExceeded(DEFAULT_STREAM_PAYLOAD_BOUND_BYTES + 1)

    def on_event(note):
        if note.get("method") == "item/agentMessage/delta":
            raise overflow

    client = FakeClient()
    if owner == "compact":
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1"},
        )
        client.queue_notification(
            "item/agentMessage/delta",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            delta="x",
        )
        session = make_session(client, on_event=on_event)
        with pytest.raises(StreamPayloadBoundExceeded) as caught:
            session.compact_thread(turn_timeout=1.0)
    else:
        client.queue_notification(
            "item/agentMessage/delta",
            threadId="t",
            turnId="tu1",
            delta="x",
        )
        if owner == "approval-drain":
            client.queue_server_request(
                "item/commandExecution/requestApproval",
                request_id="approval-1",
                command="pwd",
                cwd="/tmp",
            )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        session = make_session(
            client,
            on_event=on_event,
            approval_callback=lambda *_args, **_kwargs: "once",
        )
        with pytest.raises(StreamPayloadBoundExceeded) as caught:
            session.run_turn("hi", turn_timeout=1.0)

    assert caught.value is overflow


def test_app_server_runtime_treats_payload_bound_as_terminal(monkeypatch):
    overflow = StreamPayloadBoundExceeded(DEFAULT_STREAM_PAYLOAD_BOUND_BYTES + 1)

    def raise_bound(self, user_input, **kwargs):
        raise overflow

    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerSession.run_turn",
        raise_bound,
    )
    agent = _make_codex_agent()

    with pytest.raises(StreamPayloadBoundExceeded) as caught:
        agent.run_conversation("hi")

    assert caught.value is overflow
    assert agent._codex_session is None


def _chat_agent(monkeypatch, stream_factory, *, display=None, retries="0"):
    agent = _agent(display=display)
    agent.api_mode = "chat_completions"
    client = MagicMock()
    client.chat.completions.create.side_effect = stream_factory
    agent._create_request_openai_client = lambda *args, **kwargs: client
    agent._close_request_openai_client = lambda *args, **kwargs: None
    monkeypatch.setenv("HERMES_STREAM_RETRIES", retries)
    return agent, client


def test_chat_clean_eof_partial_uses_scrubbed_delivery(monkeypatch):
    hidden = "<memory-context>\n" + "secret" * 50_000 + "\n</memory-context>\n\n"
    raw = hidden + "Visible answer"
    displayed = []
    agent, client = _chat_agent(
        monkeypatch,
        lambda *_args, **_kwargs: iter([_make_stream_chunk(content=raw)]),
        display=displayed.append,
    )

    response = agent._interruptible_streaming_api_call({})
    partial = response.choices[0].message.content

    assert client.chat.completions.create.call_count == 1
    assert "".join(displayed).strip() == "Visible answer"
    assert partial.strip() == "Visible answer"
    assert "secret" not in partial
    assert streamed_payload_bytes(partial) <= DEFAULT_STREAM_PAYLOAD_BOUND_BYTES


def _mid_tool_stream(text, *, fail=True):
    import httpx

    yield _make_stream_chunk(content=text)
    yield _make_stream_chunk(
        tool_calls=[
            _make_tool_call_delta(index=0, tc_id="call-1", name="write_file")
        ]
    )
    yield _make_stream_chunk(
        tool_calls=[_make_tool_call_delta(index=0, arguments='{"path":')]
    )
    if fail:
        raise httpx.RemoteProtocolError("peer closed connection")
    yield _make_stream_chunk(finish_reason="tool_calls")


def test_chat_reconnect_status_bound_is_terminal(monkeypatch):
    attempts = 0

    def streams(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return _mid_tool_stream("first" if attempts == 1 else "retry", fail=attempts == 1)

    agent, _client = _chat_agent(monkeypatch, streams, retries="1")
    calls = []
    overflow = StreamPayloadBoundExceeded(DEFAULT_STREAM_PAYLOAD_BOUND_BYTES + 1)

    def generated_projection_bound(text):
        calls.append(text)
        if len(calls) == 2:
            raise overflow
        return True

    agent._fire_stream_delta = generated_projection_bound

    with pytest.raises(StreamPayloadBoundExceeded) as caught:
        agent._interruptible_streaming_api_call({})

    assert caught.value is overflow
    assert attempts == 1


def test_chat_partial_warning_bound_is_terminal(monkeypatch):
    warning_headroom = 80
    displayed = []
    text = "x" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES - warning_headroom)
    agent, client = _chat_agent(
        monkeypatch,
        lambda *_args, **_kwargs: _mid_tool_stream(text),
        display=displayed.append,
    )

    with pytest.raises(StreamPayloadBoundExceeded):
        agent._interruptible_streaming_api_call({})

    assert client.chat.completions.create.call_count == 1
    assert displayed == [text]


def test_chat_delivered_error_partial_bounds_authoritative_text(monkeypatch):
    import httpx

    agents = []

    def stream(*_args, **_kwargs):
        yield _make_stream_chunk(content="ok")
        agents[0]._current_streamed_assistant_text = "é" * (
            DEFAULT_STREAM_PAYLOAD_BOUND_BYTES // 2 + 1
        )
        raise httpx.RemoteProtocolError("peer closed connection")

    agent, _client = _chat_agent(monkeypatch, stream, display=lambda _text: None)
    agents.append(agent)

    with pytest.raises(StreamPayloadBoundExceeded):
        agent._interruptible_streaming_api_call({})


@pytest.mark.parametrize(
    ("content", "reasoning"),
    [
        ("é" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES // 2 + 1), None),
        (None, "é" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES // 2 + 1)),
        (
            "é" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES // 4),
            "é" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES // 4 + 1),
        ),
    ],
    ids=["assistant-only", "reasoning-only", "combined"],
)
def test_partial_stream_stub_bounds_each_utf8_payload(content, reasoning):
    from agent.chat_completion_helpers import _build_partial_stream_stub

    with pytest.raises(StreamPayloadBoundExceeded):
        _build_partial_stream_stub(
            "assistant", content, reasoning, "test/model", None
        )


def test_chat_no_delivery_retry_discards_attempt_payload_state(monkeypatch):
    import httpx

    attempts = 0
    first = "é" * ((DEFAULT_STREAM_PAYLOAD_BOUND_BYTES - 2) // 2) + "<"

    def streams(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            def failed():
                yield _make_stream_chunk(content=first)
                raise httpx.RemoteProtocolError("peer closed connection")

            return failed()
        return iter(
            [
                _make_stream_chunk(content="retryé"),
                _make_stream_chunk(finish_reason="stop"),
            ]
        )

    def fail(_text):
        raise RuntimeError("no physical delivery")

    agent, _client = _chat_agent(
        monkeypatch, streams, display=fail, retries="1"
    )
    agent._stream_callback = fail

    response = agent._interruptible_streaming_api_call({})

    assert attempts == 2
    assert response.choices[0].message.content == "retryé"
    assert agent._current_streamed_payload_bytes == streamed_payload_bytes("retryé")
    assert agent._current_streamed_assistant_text == ""


class _CodexDropStream:
    def __init__(self, text):
        self.text = text

    def __iter__(self):
        import httpx

        yield SimpleNamespace(type="response.output_text.delta", delta=self.text)
        raise httpx.RemoteProtocolError("peer closed connection")

    def close(self):
        return None


def test_codex_transport_drop_partial_uses_scrubbed_delivery():
    hidden = "<memory-context>\n" + "secret" * 50_000 + "\n</memory-context>\n\n"
    raw = hidden + "Visible answer"
    displayed = []
    agent = _agent(display=displayed.append)
    agent.api_mode = "codex_responses"
    client = MagicMock()
    client.responses.create.return_value = _CodexDropStream(raw)

    response = agent._run_codex_stream({}, client=client)

    assert client.responses.create.call_count == 1
    assert "".join(displayed).strip() == "Visible answer"
    assert response.output_text.strip() == "Visible answer"
    assert "secret" not in response.output_text
    assert streamed_payload_bytes(response.output_text) <= DEFAULT_STREAM_PAYLOAD_BOUND_BYTES


def test_codex_no_delivery_retry_discards_attempt_payload_state():
    attempts = 0
    first = "é" * ((DEFAULT_STREAM_PAYLOAD_BOUND_BYTES - 2) // 2) + "<"

    class Stream:
        def __init__(self, text, fail):
            self.text = text
            self.fail = fail

        def __iter__(self):
            import httpx

            yield SimpleNamespace(type="response.output_text.delta", delta=self.text)
            if self.fail:
                raise httpx.RemoteProtocolError("peer closed connection")
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", id="r2", usage=None),
            )

        def close(self):
            return None

    def create(**_kwargs):
        nonlocal attempts
        attempts += 1
        return Stream(first if attempts == 1 else "retryé", attempts == 1)

    def fail(_text):
        raise RuntimeError("no physical delivery")

    agent = _agent(display=fail)
    agent.api_mode = "codex_responses"
    agent._stream_callback = fail
    client = MagicMock()
    client.responses.create.side_effect = create

    response = agent._run_codex_stream({}, client=client)

    assert attempts == 2
    assert response.output_text == "retryé"
    assert agent._current_streamed_payload_bytes == streamed_payload_bytes("retryé")
    assert agent._current_streamed_assistant_text == ""


@pytest.mark.parametrize(
    ("sink", "live_delta", "has_stream", "has_interim", "physically_delivered"),
    [
        ("interim-only", False, False, True, True),
        ("stream-only", False, True, False, True),
        ("both", False, True, True, True),
        ("plugin-only", False, False, False, False),
        ("no-consumer", False, False, False, False),
        ("live-and-completed", True, True, True, True),
    ],
)
def test_app_server_completed_partial_sink_matrix(
    monkeypatch, sink, live_delta, has_stream, has_interim, physically_delivered
):
    from agent.codex_runtime import make_codex_app_server_event_bridge

    hidden = "HOSTILE-COMPLETED-SECRET"
    raw = f"<memory-context>\n{hidden * 128}\n</memory-context>\n\nVisible"
    streamed, interim, plugin, persisted = [], [], [], []
    agent = _make_codex_agent(
        stream_delta_callback=streamed.append if has_stream else None,
        interim_assistant_callback=(
            (lambda text, **_kwargs: interim.append(text)) if has_interim else None
        ),
    )
    monkeypatch.setattr(
        "agent.plugin_stream_hooks.enqueue_plugin_stream_hook",
        lambda hook, **payload: (
            plugin.append((hook, str(payload))) if sink == "plugin-only" else None
        ),
    )
    session_db = MagicMock()
    session_db.get_session.return_value = None
    setattr(agent, "_session_db", session_db)
    agent._flush_messages_to_session_db = (
        lambda messages, *_args, **_kwargs: persisted.append(str(messages)) or True
    )

    client = FakeClient()
    if live_delta:
        client.queue_notification(
            "item/agentMessage/delta",
            threadId="t",
            turnId="tu1",
            delta=raw,
        )
    client.queue_notification(
        "item/completed",
        threadId="t",
        turnId="tu1",
        item={"type": "agentMessage", "id": "m1", "text": raw},
    )
    client.queue_notification(
        "turn/completed",
        threadId="t",
        turn={
            "id": "tu1",
            "status": "failed",
            "error": {"message": "transport dropped"},
        },
    )
    setattr(
        agent,
        "_codex_session",
        make_session(client, on_event=make_codex_app_server_event_bridge(agent)),
    )

    result = agent.run_conversation("hi")

    assert streamed == (["Visible"] if has_stream else [])
    assert interim == (["Visible"] if has_interim and physically_delivered else [])
    assert result["final_response"] == ("Visible" if physically_delivered else "")
    assert result["partial"] is True
    assert bool(plugin) is (sink == "plugin-only")
    assert any("Visible" in payload for _hook, payload in plugin) is (
        sink == "plugin-only"
    )
    assert any("Visible" in payload for payload in persisted) is physically_delivered
    surfaces = [
        *streamed,
        *interim,
        *(payload for _hook, payload in plugin),
        result["final_response"],
        str(result["messages"]),
        *persisted,
    ]
    assert all(hidden not in surface for surface in surfaces)
    assert streamed_payload_bytes(result["final_response"]) <= DEFAULT_STREAM_PAYLOAD_BOUND_BYTES


def test_app_server_resets_stream_state_before_every_turn(monkeypatch):
    agent = _make_codex_agent()
    observed = []

    def turn(self, user_input, **kwargs):
        observed.append(
            (
                agent._current_streamed_assistant_text,
                agent._current_streamed_payload_bytes,
                agent._stream_think_scrubber._buf,
                agent._stream_context_scrubber._buf,
            )
        )
        agent._current_streamed_assistant_text = "stale"
        agent._current_streamed_payload_bytes = 5
        agent._stream_think_scrubber.feed("<")
        agent._stream_context_scrubber.feed("<")
        return TurnResult(
            final_text="ok",
            projected_messages=[{"role": "assistant", "content": "ok"}],
            turn_id="turn-1",
            thread_id="thread-1",
        )

    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerSession.run_turn",
        turn,
    )

    agent.run_conversation("one")
    agent.run_conversation("two")

    assert observed == [("", 0, "", ""), ("", 0, "", "")]
