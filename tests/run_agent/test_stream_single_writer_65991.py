"""Regression tests for the streaming single-writer invariant (#65991).

A retry that supersedes a still-live SSE stream must fence the old stream out
of the delta sink; otherwise both streams write into the same turn and the
persisted transcript is two coherent responses interleaved token-by-token.

These tests exercise the real ``AIAgent`` guard helpers and the streaming
consume-loop, asserting that exactly one writer ever reaches the turn.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _chunk(content=None, tool_calls=None, finish_reason=None, model=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=None)


def test_stream_callbacks_mark_visible_categories_and_callback_time(monkeypatch):
    from agent import physical_attempt_diagnostics as diagnostics

    agent = _make_agent()
    events = []

    def begin(category):
        events.extend((("visible", category), ("begin", category)))
        return category

    monkeypatch.setattr(diagnostics, "begin_callback", begin)
    monkeypatch.setattr(
        diagnostics, "end_callback", lambda marker: events.append(("end", marker))
    )
    setattr(
        agent,
        "stream_delta_callback",
        lambda _text: events.append(("delivered", "text")),
    )
    setattr(
        agent,
        "reasoning_callback",
        lambda _text: events.append(("delivered", "reasoning")),
    )
    setattr(
        agent,
        "tool_gen_callback",
        lambda _name: events.append(("delivered", "tool")),
    )
    setattr(
        agent,
        "interim_assistant_callback",
        lambda _text, **_kwargs: events.append(("delivered", "text")),
    )

    agent._fire_stream_delta("answer")
    agent._fire_reasoning_delta("thinking")
    agent._fire_tool_gen_started("browser")
    agent._fire_streamed_codex_commentary("commentary")

    assert events == [
        ("visible", "text"), ("begin", "text"),
        ("delivered", "text"), ("end", "text"),
        ("visible", "reasoning"), ("begin", "reasoning"),
        ("delivered", "reasoning"), ("end", "reasoning"),
        ("visible", "tool"), ("begin", "tool"),
        ("delivered", "tool"), ("end", "tool"),
        ("visible", "text"), ("begin", "text"),
        ("delivered", "text"), ("end", "text"),
    ]


def test_tool_call_content_uses_callback_and_transport_diagnostics(monkeypatch):
    from agent import physical_attempt_diagnostics as diagnostics
    from tui_gateway import server

    attempt = diagnostics.Attempt(
        "tool-call-content-attempt",
        streamed=True,
        started_ns=diagnostics.time.monotonic_ns(),
    )
    records = []
    frames = []
    tts = []

    monkeypatch.setattr(diagnostics, "start_attempt", lambda *_args, **_kwargs: attempt)
    monkeypatch.setattr(diagnostics, "_append", records.append)

    class StdioTransport:
        def write(self, obj):
            frames.append(obj)
            return True

    monkeypatch.setattr(server, "current_transport", lambda: None)
    monkeypatch.setattr(server, "_stdio_transport", StdioTransport())

    tool_call = SimpleNamespace(
        index=0,
        id="call-1",
        function=SimpleNamespace(name="terminal", arguments='{"command":"pwd"}'),
    )
    chunks = [
        _chunk(tool_calls=[tool_call]),
        _chunk(content="<think>suppressed tool-turn content</think>"),
        _chunk(finish_reason="tool_calls", model="test-model"),
    ]
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks)
    agent = _make_agent()
    setattr(
        agent,
        "stream_delta_callback",
        lambda text: server.write_json({"delta": text}),
    )
    setattr(agent, "_stream_callback", tts.append)
    setattr(agent, "tool_gen_callback", None)
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda *_a, **_k: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *_a, **_k: None)

    agent._interruptible_streaming_api_call({})

    terminal = records[-1]
    stage = terminal["stage_latency"]
    assert terminal["attempt_digest"] == attempt.digest
    assert frames == [{"delta": "<think>suppressed tool-turn content</think>"}]
    assert tts == []
    assert stage["first_visible_category"] == "text"
    assert stage["visible_event_count"] == 1
    assert stage["callbacks"]["text"]["count"] == 1
    assert stage["transports"]["stdio"]["count"] == 1


class TestSingleWriterSink:
    def test_superseded_writer_deltas_are_dropped(self):
        """A stale writer (older token, other thread) is fenced; only the
        newest writer reaches the callbacks and the accumulated turn text."""
        agent = _make_agent()
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        a_claimed = threading.Event()
        b_claimed = threading.Event()

        def writer_a():
            agent._claim_stream_writer()  # token 1
            a_claimed.set()
            b_claimed.wait(timeout=2)  # let B supersede us first
            # We are now stale — every sink call must be a no-op.
            agent._fire_stream_delta("A-should-drop")
            agent._fire_reasoning_delta("A-reason-drop")
            agent._record_streamed_assistant_text("A-record-drop")

        def writer_b():
            a_claimed.wait(timeout=2)
            agent._claim_stream_writer()  # token 2 — supersedes A
            b_claimed.set()

        tb = threading.Thread(target=writer_b)
        ta = threading.Thread(target=writer_a)
        tb.start()
        ta.start()
        ta.join(timeout=3)
        tb.join(timeout=3)

        assert delivered == [], "a superseded stream must not deliver any deltas"
        assert "A-record-drop" not in (agent._current_streamed_assistant_text or "")
        assert agent._stream_writer_dropped >= 1

    def test_current_writer_is_never_fenced(self):
        """The active writer always delivers — the guard can only drop a
        stream that a *newer* claim has superseded."""
        agent = _make_agent()
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        agent._claim_stream_writer()
        agent._fire_stream_delta("hello ")
        agent._fire_stream_delta("world")

        assert "".join(delivered) == "hello world"
        assert agent._stream_writer_dropped == 0

    def test_non_claiming_thread_is_not_a_writer(self):
        """A thread that never claimed (a non-streaming delta caller) is never
        treated as a stale writer, even after other attempts have claimed."""
        agent = _make_agent()
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        # Some other thread runs a couple of stream attempts and bumps the token.
        def other():
            agent._claim_stream_writer()
            agent._claim_stream_writer()

        t = threading.Thread(target=other)
        t.start()
        t.join(timeout=3)

        # This (main) thread never claimed → not superseded → delivers.
        assert agent._stream_writer_superseded() is False
        agent._fire_stream_delta("plain")
        assert delivered == ["plain"]


class TestSingleWriterLoop:
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_consume_loop_stops_when_superseded_mid_stream(self, _close, mock_create):
        """The real streaming loop bails out the moment a newer attempt claims
        the sink, so a superseded stream cannot interleave into the turn."""
        agent = _make_agent()
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        def stream_gen():
            yield _chunk(content="first")
            # A concurrent retry supersedes this stream between chunks.
            agent._claim_stream_writer()
            yield _chunk(content="-stale-tail", finish_reason="stop", model="m")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = stream_gen()
        mock_create.return_value = mock_client

        agent._interruptible_streaming_api_call({})

        assert "".join(delivered) == "first"
        assert "-stale-tail" not in "".join(delivered)

    def test_chat_parser_failure_closes_managed_stream(self):
        agent = _make_agent()
        managed_stream = MagicMock()
        managed_stream.__iter__.return_value = iter([object()])
        managed_stream.final_response = None

        with patch(
            "agent.relay_llm.stream",
            return_value=managed_stream,
        ):
            with pytest.raises(AttributeError):
                agent._interruptible_streaming_api_call({})

        managed_stream.close.assert_called_once()


class TestCodexSingleWriter:
    """The codex_responses path claims the sink and stops when superseded,
    matching the chat_completions/anthropic/bedrock parity added in salvage."""

    def _codex_event(self, event_type, **fields):
        return SimpleNamespace(type=event_type, **fields)

    def test_codex_stream_claims_writer_and_stops_when_superseded(self):
        from agent.codex_runtime import run_codex_stream

        agent = _make_agent()
        agent.api_mode = "codex_responses"
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        def event_gen():
            yield self._codex_event(
                "response.output_text.delta", delta="first", item_id="i1",
            )
            # A concurrent retry supersedes this stream between events.
            agent._claim_stream_writer()
            yield self._codex_event(
                "response.output_text.delta", delta="-stale-tail", item_id="i1",
            )
            yield self._codex_event(
                "response.completed",
                response=SimpleNamespace(
                    id="r1", status="completed", output=[], usage=None,
                ),
            )

        mock_client = MagicMock()
        mock_client.responses.create.return_value = event_gen()

        run_codex_stream(agent, {"model": "gpt-5.3-codex"}, client=mock_client)

        assert "".join(delivered) == "first"
        assert "-stale-tail" not in "".join(delivered)


    def test_codex_interrupt_closes_stream_without_draining_provider(self):
        from agent.codex_runtime import run_codex_stream

        agent = _make_agent()
        agent.api_mode = "codex_responses"
        produced = []
        stream_closed = threading.Event()

        def interrupt_after_first_delta(_text):
            agent._interrupt_requested = True

        agent.stream_delta_callback = interrupt_after_first_delta
        agent._stream_callback = None

        def event_gen():
            try:
                produced.append("first")
                yield self._codex_event(
                    "response.output_text.delta",
                    delta="first",
                    item_id="i1",
                )
                produced.append("lookahead")
                yield self._codex_event(
                    "response.output_text.delta",
                    delta="-unused",
                    item_id="i1",
                )
                produced.append("terminal")
                yield self._codex_event(
                    "response.completed",
                    response=SimpleNamespace(
                        id="r1",
                        status="completed",
                        output=[],
                        usage=None,
                    ),
                )
            finally:
                stream_closed.set()

        mock_client = MagicMock()
        mock_client.responses.create.return_value = event_gen()

        run_codex_stream(agent, {"model": "gpt-5.3-codex"}, client=mock_client)

        assert produced == ["first", "lookahead"]
        assert stream_closed.is_set()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
