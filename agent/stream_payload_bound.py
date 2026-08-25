"""Bound streamed assistant/reasoning payloads without dropping the turn.

A grok thinking/stream loop can emit 100k+ chars over several minutes, then
leave that orphan as the last assistant row when the user mashes interrupt
(#119). Bound the live payload in one place so every stream writer and the
interrupt persist path share the same check.

#195: the bound is configurable (reasoning vs final separately). Overflow
keeps the retained partial, stamps typed ``stream_payload_limit``, and leaves
a continue path instead of replacing the turn with an error-only abort.
"""

from __future__ import annotations

from typing import Any

from agent.message_metadata import append_message

# Below the 148197-char orphan observed in session 20260624_143447_f60046.
DEFAULT_STREAM_PAYLOAD_BOUND_BYTES = 128 * 1024
STREAM_PAYLOAD_LIMIT_STATUS = "stream_payload_limit"


class StreamPayloadBoundExceeded(RuntimeError):
    """Raised when streamed assistant/reasoning text exceeds the declared bound."""

    status = STREAM_PAYLOAD_LIMIT_STATUS

    def __init__(self, size: int, bound: int = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES) -> None:
        self.size = int(size)
        self.bound = int(bound)
        self.status = STREAM_PAYLOAD_LIMIT_STATUS
        super().__init__(
            f"Streamed assistant payload exceeded {self.bound} bytes "
            f"({self.size} bytes)."
        )


def streamed_payload_bytes(text: str) -> int:
    return len((text or "").encode("utf-8"))


def utf8_truncate(text: str, bound: int) -> str:
    raw = (text or "").encode("utf-8")
    if bound <= 0:
        return ""
    if len(raw) <= bound:
        return text or ""
    return raw[:bound].decode("utf-8", errors="ignore")


def resolve_stream_payload_bounds() -> tuple[int, int]:
    """Return (assistant_bytes, reasoning_bytes) from config, else the code default."""
    assistant = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES
    reasoning = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES
    try:
        from hermes_cli.config import load_config

        limits = ((load_config() or {}).get("agent") or {}).get("stream_payload_limit") or {}
        assistant = int(limits.get("assistant_bytes") or assistant)
        reasoning = int(limits.get("reasoning_bytes") or reasoning)
    except Exception:
        pass
    if assistant <= 0:
        assistant = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES
    if reasoning <= 0:
        reasoning = DEFAULT_STREAM_PAYLOAD_BOUND_BYTES
    return assistant, reasoning


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
        f"({size} bytes). Send continue to resume from the retained partial."
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
        bound, _ = resolve_stream_payload_bounds()
    strip = getattr(agent, "_strip_think_blocks", None)
    raw = getattr(agent, "_current_streamed_assistant_text", "") or ""
    visible = str(strip(raw) if callable(strip) else raw).strip()
    visible_size = streamed_payload_bytes(visible)
    status = None
    if exceeded or visible_size > bound:
        kept = utf8_truncate(visible, bound)
        footer = stream_payload_error_text(int(size) or visible_size, bound)
        text = f"{kept}\n\n{footer}" if kept else footer
        status = STREAM_PAYLOAD_LIMIT_STATUS
        try:
            agent._current_streamed_assistant_text = kept
        except Exception:
            pass
    elif visible:
        text = visible
    else:
        last = messages[-1] if messages else None
        if isinstance(last, dict) and last.get("role") == "assistant":
            last_text = last.get("content") or ""
            if last.get("status") == STREAM_PAYLOAD_LIMIT_STATUS:
                return last_text
            last_size = streamed_payload_bytes(last_text)
            if last_size > bound:
                kept = utf8_truncate(str(last_text), bound)
                footer = stream_payload_error_text(last_size, bound)
                last["content"] = f"{kept}\n\n{footer}" if kept else footer
                last["status"] = STREAM_PAYLOAD_LIMIT_STATUS
                return last["content"]
            return last_text
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        return f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}{elapsed:.1f}s elapsed)."

    last = messages[-1] if messages else None
    if isinstance(last, dict) and last.get("role") == "assistant":
        last["content"] = text
        if status:
            last["status"] = status
    else:
        row = {"role": "assistant", "content": text}
        if status:
            row["status"] = status
        append_message(messages, row)
    return text
