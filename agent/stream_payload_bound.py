"""Abort a turn whose streamed assistant payload exceeds a declared bound.

A grok thinking/stream loop can emit 100k+ chars over several minutes, then
leave that orphan as the last assistant row when the user mashes interrupt
(#119). Bound the live payload in one place so every stream writer and the
interrupt persist path share the same check.
"""

from __future__ import annotations

from typing import Any

from agent.message_metadata import append_message

# Retain a useful long-form answer before the bounded overflow path engages.
DEFAULT_STREAM_PAYLOAD_BOUND_BYTES = 256 * 1024


class StreamPayloadBoundExceeded(RuntimeError):
    """Raised when streamed assistant/reasoning text exceeds the declared bound."""

    def __init__(self, size: int, bound: int = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES) -> None:
        self.size = int(size)
        self.bound = int(bound)
        super().__init__(
            f"Streamed assistant payload exceeded {self.bound} bytes "
            f"({self.size} bytes)."
        )


def streamed_payload_bytes(text: str) -> int:
    return len((text or "").encode("utf-8"))


def accumulate_stream_text(
    existing: str,
    extra: str,
    *,
    bound: int = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
) -> str:
    """Append ``extra`` and raise if the UTF-8 payload would exceed ``bound``."""
    if not extra:
        return existing or ""
    combined = (existing or "") + extra
    size = streamed_payload_bytes(combined)
    if size > bound:
        raise StreamPayloadBoundExceeded(size, bound)
    return combined


def stream_payload_error_text(
    size: int,
    bound: int = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
) -> str:
    return (
        f"Streamed assistant payload exceeded {bound} bytes "
        f"({size} bytes). Turn aborted."
    )


def persist_interrupted_stream_partial(
    agent: Any,
    messages: list,
    *,
    elapsed: float = 0.0,
    bound: int = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
    exceeded: bool = False,
    size: int = 0,
) -> str:
    """Persist interrupt text without leaving an oversize orphan last message."""
    if bound is None:
        bound = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES
    strip = getattr(agent, "_strip_think_blocks", None)
    raw = getattr(agent, "_current_streamed_assistant_text", "") or ""
    visible = str(strip(raw) if callable(strip) else raw).strip()
    visible_size = streamed_payload_bytes(visible)
    if exceeded or visible_size > bound:
        text = stream_payload_error_text(
            int(size) or visible_size,
            bound,
        )
        try:
            agent._current_streamed_assistant_text = ""
        except Exception:
            pass
    elif visible:
        text = visible
    else:
        last = messages[-1] if messages else None
        if isinstance(last, dict) and last.get("role") == "assistant":
            last_text = last.get("content") or ""
            last_size = streamed_payload_bytes(last_text)
            if last_size > bound:
                text = stream_payload_error_text(last_size, bound)
                last["content"] = text
                return text
            return last_text
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        return f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}{elapsed:.1f}s elapsed)."

    last = messages[-1] if messages else None
    if isinstance(last, dict) and last.get("role") == "assistant":
        last["content"] = text
    else:
        append_message(messages, {"role": "assistant", "content": text})
    return text
