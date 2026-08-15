"""Focused bounded-stream contracts for the TUI stdio transport."""

from __future__ import annotations

import threading
import time

from tui_gateway import transport


class _BlockingTransport:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.frames: list[dict] = []

    def write(self, obj: dict) -> bool:
        self.started.set()
        assert self.release.wait(2), "test sink was not released"
        self.frames.append(obj)
        return True

    def close(self) -> None:
        return None


def _event(kind: str, text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": kind, "payload": {"text": text}},
    }


def test_stream_deltas_stay_nonblocking_bounded_and_precede_completion() -> None:
    sink = _BlockingTransport()
    writer = transport.BufferedStreamWriter(
        sink, max_pending_deltas=3, coalesce_s=0.001, close_timeout_s=1
    )
    try:
        assert writer.write(_event("message.delta", "first"))
        assert sink.started.wait(1)

        started = time.monotonic()
        for index in range(20):
            assert writer.write(_event("message.delta", f"d{index}"))
        assert time.monotonic() - started < 0.5
        assert writer.write(_event("message.complete", "done"))
    finally:
        sink.release.set()
        writer.close()

    assert [frame["params"]["payload"]["text"] for frame in sink.frames] == [
        "first",
        "d17",
        "d18",
        "d19",
        "done",
    ]
