"""#119: abort an oversize streamed turn; do not persist a 148k interrupt orphan."""

import pytest


def test_oversize_stream_aborts_and_interrupt_does_not_leave_orphan():
    from agent.stream_payload_bound import (
        DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
        StreamPayloadBoundExceeded,
        persist_interrupted_stream_partial,
    )
    from run_agent import AIAgent

    def make_agent():
        agent = AIAgent.__new__(AIAgent)
        agent._claim_stream_writer()
        agent.stream_delta_callback = lambda _t: None
        agent._stream_callback = None
        return agent

    assert DEFAULT_STREAM_PAYLOAD_BOUND_BYTES == 256 * 1024

    agent = make_agent()

    with pytest.raises(StreamPayloadBoundExceeded, match="exceeded"):
        agent._record_streamed_assistant_text(
            "x" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES + 1)
        )

    orphan = "y" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES + 1)
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


def test_default_stream_payload_limit_retains_131090_byte_assistant_delta():
    from run_agent import AIAgent

    payload = "x" * 131_090
    agent = AIAgent.__new__(AIAgent)
    agent._claim_stream_writer()

    assert agent._record_streamed_assistant_text(payload) is None
    assert agent._current_streamed_assistant_text == payload


def test_reasoning_and_assistant_share_utf8_payload_bound():
    from agent.stream_payload_bound import (
        DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
        StreamPayloadBoundExceeded,
    )
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._claim_stream_writer()
    agent.reasoning_callback = None

    agent._fire_reasoning_delta("é" * (DEFAULT_STREAM_PAYLOAD_BOUND_BYTES // 2))
    with pytest.raises(StreamPayloadBoundExceeded, match="262145"):
        agent._record_streamed_assistant_text("x")
