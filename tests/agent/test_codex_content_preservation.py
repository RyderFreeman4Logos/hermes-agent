"""Refusal provenance must not truncate replay, auxiliary or display text (#161)."""

from types import SimpleNamespace as NS

import pytest

from agent.auxiliary_client import _parse_codex_final_response
from agent.codex_responses_adapter import _chat_messages_to_responses_input, _normalize_codex_response
from agent.codex_runtime import _consume_codex_event_stream
from agent.history_commentary import _commentary_items
from agent.stream_delivery import StreamDeliveryMixin


@pytest.mark.parametrize("shape", [
    "forward", "reverse", "forward_unterminated", "reverse_unterminated",
    "mixed_done", "two_done", "mixed_tool", "commentary", "plain", "sole", "empty_text",
])
def test_refusal_text_survives_all_consumers(shape):
    text, refusal = "Hello. ", "But no more."
    parts = [NS(type="output_text", text=text), NS(type="refusal", refusal=refusal)]
    if shape.startswith("reverse"):
        parts.reverse()
    elif shape == "plain":
        parts = parts[:1]
    elif shape == "sole":
        parts = parts[1:]
    elif shape == "empty_text":
        parts[0].text = ""
    expected = "".join(getattr(part, "text", getattr(part, "refusal", "")) for part in parts)

    def message(content, item_id, phase=None):
        return NS(type="message", role="assistant", status="completed", id=item_id, content=content, phase=phase)

    def done(item, index):
        return NS(type="response.output_item.done", item=item, output_index=index)

    events = []
    if shape == "two_done":
        events = [done(message([part], f"msg_{i}"), i) for i, part in enumerate(parts)]
    elif shape in {"mixed_done", "mixed_tool", "commentary"}:
        events = [done(message(parts, "msg_0", "commentary" if shape == "commentary" else None), 0)]
        if shape == "mixed_tool":
            events.append(done(NS(type="function_call", id="fc_1", call_id="call_1", name="read_file",
                                  arguments='{"path":"a"}', status="completed"), 1))
        elif shape == "commentary":
            events.append(done(message([NS(type="output_text", text="Final")], "msg_1", "final_answer"), 1))
    else:
        events = [NS(type=f"response.{part.type}.delta", delta=getattr(part, "text", getattr(part, "refusal", "")),
                     item_id="msg_0", output_index=0, content_index=i) for i, part in enumerate(parts)]
    if not shape.endswith("unterminated"):
        # Terminal output duplicates the deltas; it must not duplicate assembled content.
        events.append(NS(type="response.completed", response=NS(status="completed", output=[message(parts, "msg_0")])))
    response = _consume_codex_event_stream(events, model="fixture")
    aux, aux_calls, _ = _parse_codex_final_response(response)
    assert "".join(aux) == expected + ("Final" if shape == "commentary" else "")
    assistant, finish = _normalize_codex_response(response)
    history = {"role": "assistant", "content": assistant.content,
               "codex_message_items": assistant.codex_message_items,
               "tool_calls": [{"id": tc.id, "type": "function", "function": vars(tc.function)} for tc in assistant.tool_calls]}
    wire = _chat_messages_to_responses_input([history])
    replay = [item["content"] if isinstance(item.get("content"), str)
              else "".join(part["text"] for part in item.get("content", []))
              for item in wire if item.get("role") == "assistant"]
    # Per-item trimming is intentional; never demand a synthetic inter-item space.
    expected_replay = [part.strip() for part in (text, refusal)] if shape == "two_done" else [expected.strip()]
    if shape == "commentary":
        expected_replay.append("Final")
    assert replay == expected_replay
    assert assistant.content == ("Final" if shape == "commentary" else "\n".join(expected_replay))
    calls = [item for item in wire if item.get("type") == "function_call"]
    assert len(calls) == len(aux_calls) == (1 if shape == "mixed_tool" else 0)
    assert finish == ("tool_calls" if shape == "mixed_tool" else "stop")
    if calls:
        assert calls[0]["name"] == "read_file" and calls[0]["arguments"] == '{"path":"a"}'
    display = StreamDeliveryMixin()
    display._strip_think_blocks = lambda value: value
    expected_commentary = [expected.strip()] if shape == "commentary" else []
    assert _commentary_items(history) == expected_commentary
    assert display._extract_codex_interim_visible_parts(history) == expected_commentary
