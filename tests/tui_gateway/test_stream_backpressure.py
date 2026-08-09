"""Focused contracts for the TUI-only buffered stdio transport."""

from __future__ import annotations

import io
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
        if not self.release.wait(10):
            raise TimeoutError("test sink was not released")
        self.frames.append(obj)
        return True

    def close(self) -> None:
        return None


def _event(kind: str, text: str = "") -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": kind,
            "session_id": "test-session",
            "payload": {"text": text},
        },
    }


def _wait_for_claims(writer, count: int) -> bool:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with writer._pending_lock:
            if writer._control_claimed == count:
                return True
        time.sleep(0.005)
    return False


def test_stream_delta_producer_stays_nonblocking_when_sink_blocks() -> None:
    sink = _BlockingTransport()
    writer = transport.BufferedStreamWriter(
        sink,
        coalesce_s=0.01,
        close_timeout_s=2,
    )
    producer = None
    result: list[bool] = []
    done = threading.Event()
    try:
        assert writer.write(_event("message.delta", "first"))
        assert sink.started.wait(2), "writer never reached the blocked sink"

        def push_delta() -> None:
            result.append(writer.write(_event("message.delta", "second")))
            done.set()

        producer = threading.Thread(target=push_delta, daemon=True)
        producer.start()
        assert done.wait(2), "streaming producer blocked behind stdio"
        assert result == [True]
    finally:
        sink.release.set()
        if producer is not None:
            producer.join(timeout=2)
        writer.close()


def test_waiting_completion_is_not_dropped_or_overtaken_by_later_delta() -> None:
    sink = _BlockingTransport()
    writer = transport.BufferedStreamWriter(
        sink,
        queue_maxsize=1,
        coalesce_s=0.01,
        close_timeout_s=2,
    )
    completion_result: list[bool] = []
    completion = None
    try:
        assert writer.write(_event("tool.start", "A"))
        assert sink.started.wait(2), "writer never reached the blocked sink"
        assert writer.write(_event("tool.start", "B"))

        completion = threading.Thread(
            target=lambda: completion_result.append(
                writer.write(_event("message.complete", "done"))
            ),
            daemon=True,
        )
        completion.start()
        assert _wait_for_claims(writer, 3), (
            "completion never reached the full-queue ordering barrier"
        )
        assert writer.write(_event("message.delta", "late"))
    finally:
        sink.release.set()
        if completion is not None:
            completion.join(timeout=2)
        writer.close()

    assert completion is not None and not completion.is_alive()
    assert completion_result == [True]
    assert [frame["params"]["payload"]["text"] for frame in sink.frames] == [
        "A",
        "B",
        "done",
        "late",
    ]


def test_delta_overflow_is_bounded_and_completion_survives() -> None:
    sink = _BlockingTransport()
    writer = transport.BufferedStreamWriter(
        sink,
        max_pending_deltas=3,
        coalesce_s=0.01,
        close_timeout_s=2,
    )
    try:
        assert writer.write(_event("message.delta", "first"))
        assert sink.started.wait(2), "writer never reached the blocked sink"
        for index in range(20):
            assert writer.write(_event("message.delta", f"d{index}"))
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


def test_close_returns_while_sink_is_blocked_then_writer_exits_after_recovery() -> None:
    sink = _BlockingTransport()
    writer = transport.BufferedStreamWriter(
        sink,
        queue_maxsize=1,
        coalesce_s=0.01,
        close_timeout_s=0.05,
    )
    close_done = threading.Event()
    closer = threading.Thread(
        target=lambda: (writer.close(), close_done.set()),
        daemon=True,
    )
    try:
        assert writer.write(_event("tool.start", "first"))
        assert sink.started.wait(2), "writer never reached the blocked sink"
        assert writer.write(_event("tool.start", "second"))

        closer.start()
        assert close_done.wait(2), "close blocked on the wedged sink"
        assert writer._thread.daemon and writer._thread.is_alive()
    finally:
        sink.release.set()
        closer.join(timeout=2)
        writer._thread.join(timeout=2)

    assert not writer._thread.is_alive()
    assert [frame["params"]["payload"]["text"] for frame in sink.frames] == [
        "first",
        "second",
    ]


def test_entry_main_scopes_buffered_writer_to_tui_stdio(monkeypatch) -> None:
    from hermes_cli import model_switch
    from tui_gateway import entry, server

    original = server._stdio_transport
    created: list[transport.BufferedStreamWriter] = []
    writes = []

    def make_writer(inner):
        writer = transport.BufferedStreamWriter(inner, close_timeout_s=2)
        created.append(writer)
        return writer

    monkeypatch.setattr(entry, "BufferedStreamWriter", make_writer, raising=False)
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(model_switch, "prewarm_picker_cache_async", lambda: None)
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *args: False)
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(
        entry,
        "write_json",
        lambda _obj: writes.append(server._stdio_transport) or True,
    )

    restored = False
    closed = False
    try:
        entry.main()
        restored = server._stdio_transport is original
        closed = len(created) == 1 and not created[0]._thread.is_alive()
    finally:
        for writer in created:
            writer.close()
        server._stdio_transport = original

    assert len(created) == 1
    assert writes == [created[0]]
    assert restored
    assert closed
