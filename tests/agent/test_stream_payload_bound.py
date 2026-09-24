"""Abort an oversize streamed turn; compact high-volume tool results before history."""

import pytest

from agent.stream_payload_bound import (
    DEFAULT_STREAM_PAYLOAD_BOUND_BYTES,
    StreamPayloadBoundExceeded,
    persist_interrupted_stream_partial,
)
from tools.tool_result_storage import (
    _INSERTION_COMPACT_THRESHOLD_CHARS,
    extract_persisted_path,
    maybe_persist_tool_result,
)


class _Agent:
    def __init__(self):
        self._streamed_assistant_text_parts = []
        self._current_streamed_payload_bytes = 0
        self._stream_writer_generation = 1
        self._current_stream_writer_generation = 1
        self.verbose_logging = False

    def _stream_writer_superseded(self):
        return self._current_stream_writer_generation != self._stream_writer_generation

    def _record_streamed_assistant_text(self, text: str) -> None:
        from agent.stream_delivery import StreamDeliveryMixin

        StreamDeliveryMixin._record_streamed_assistant_text(self, text)

    @property
    def _current_streamed_assistant_text(self):
        return "".join(self._streamed_assistant_text_parts)

    @_current_streamed_assistant_text.setter
    def _current_streamed_assistant_text(self, value: str) -> None:
        from agent.stream_delivery import StreamDeliveryMixin

        StreamDeliveryMixin._current_streamed_assistant_text.fset(self, value)

    def _strip_think_blocks(self, text: str) -> str:
        return text

    def _vprint(self, *_args, **_kwargs):
        return None


def test_oversize_stream_aborts_and_interrupt_does_not_leave_orphan():
    assert DEFAULT_STREAM_PAYLOAD_BOUND_BYTES == 256 * 1024
    agent = _Agent()
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
    assert messages[-1]["content"] == first
    assert len(first.encode("utf-8")) < DEFAULT_STREAM_PAYLOAD_BOUND_BYTES

    agent._current_streamed_assistant_text = ""
    persist_interrupted_stream_partial(agent, messages, elapsed=0.1)
    last = messages[-1]
    assert orphan not in (last.get("content") or "")
    assert len((last.get("content") or "").encode("utf-8")) < 1024


def test_under_bound_delta_is_kept():
    agent = _Agent()
    payload = "x" * 131_090
    agent._record_streamed_assistant_text(payload)
    assert agent._current_streamed_assistant_text == payload


def test_high_volume_result_spills_before_32k(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "tools.tool_result_storage._is_host_side_env", lambda _env: True,
    )
    raw = "RAW-START\n" + ("x" * 25_000) + "\nRAW-END\n"
    suffix = "\nHINT " + ("h" * 8_000)
    stored = maybe_persist_tool_result(
        raw,
        "execute_code",
        "call-1",
        env=object(),
        history_suffix=suffix,
    )
    assert len(stored) <= _INSERTION_COMPACT_THRESHOLD_CHARS
    assert suffix in stored
    path = extract_persisted_path(stored)
    assert path is not None
    from pathlib import Path

    assert Path(path).read_text(encoding="utf-8") == raw
