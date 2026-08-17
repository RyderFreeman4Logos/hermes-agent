"""#119: abort an oversize streamed turn; do not persist a 148k interrupt orphan."""

import pytest


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
