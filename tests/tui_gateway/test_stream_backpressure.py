"""Focused contracts for the TUI-only buffered stdio transport."""

from __future__ import annotations

import errno
import io
import threading
import time

import pytest

from tui_gateway import server, transport


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


class _RecordingTransport:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def write(self, obj: dict) -> bool:
        self.frames.append(obj)
        return True

    def close(self) -> None:
        return None


class _FailingTransport:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.called = threading.Event()

    def write(self, _obj: dict) -> bool:
        self.called.set()
        if self.error is not None:
            raise self.error
        return False

    def close(self) -> None:
        return None


class _FailAfterReadyTransport(_FailingTransport):
    def __init__(self, error: BaseException | None = None) -> None:
        super().__init__(error)
        self.writes = 0

    def write(self, obj: dict) -> bool:
        self.writes += 1
        if self.writes == 1:
            return True
        return super().write(obj)


class _RaisingStream:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.called = threading.Event()

    def write(self, _line: str) -> None:
        self.called.set()
        raise self.error

    def flush(self) -> None:
        return None


class _HeldOpenStdin:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def readline(self) -> str:
        self.entered.set()
        assert self.release.wait(10), "held-open stdin was never released"
        return ""


def _event(kind: str, text: str = "", **payload) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": kind,
            "session_id": "test-session",
            "payload": {"text": text, **payload},
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


def test_control_push_deadline_latches_dead_transport_without_reordering() -> None:
    sink = _BlockingTransport()
    writer = transport.BufferedStreamWriter(
        sink,
        queue_maxsize=1,
        control_push_timeout_s=0.05,
        coalesce_s=0.01,
        close_timeout_s=2,
    )
    try:
        assert writer.write(_event("tool.start", "A"))
        assert sink.started.wait(2), "writer never reached the blocked sink"
        assert writer.write(_event("tool.start", "B"))

        started = time.monotonic()
        assert writer.write(_event("message.complete", "C")) is False
        assert time.monotonic() - started < 0.5
        assert writer._closed is True
        assert server._transport_is_dead(writer)
        assert isinstance(writer.failure, transport.ControlQueueTimeoutError)
        assert writer.write(_event("message.delta", "late")) is False
    finally:
        sink.release.set()
        writer.close()

    assert [frame["params"]["payload"]["text"] for frame in sink.frames] == [
        "A",
        "B",
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


def test_reasoning_overflow_bounds_payload_bytes_and_marks_resync() -> None:
    sink = _BlockingTransport()
    payload_ceiling = 1024
    pending_count_ceiling = 16
    writer = transport.BufferedStreamWriter(
        sink,
        queue_maxsize=1,
        max_pending_deltas=pending_count_ceiling,
        max_pending_delta_bytes=payload_ceiling,
        control_push_timeout_s=0.05,
        coalesce_s=0.01,
        close_timeout_s=2,
    )
    parts = [str(index % 10) * 256 for index in range(32)]
    results: list[bool] = []
    producer = None
    try:
        assert writer.write(_event("status.update", "blocked"))
        assert sink.started.wait(2), "writer never reached the blocked sink"

        def push_reasoning() -> None:
            results.extend(
                writer.write(_event("reasoning.delta", part)) for part in parts
            )

        producer = threading.Thread(target=push_reasoning, daemon=True)
        producer.start()
        producer.join(timeout=2)
        assert not producer.is_alive(), "reasoning producer blocked on overflow"
        assert results == [True] * len(parts)
        with writer._pending_lock:
            assert len(writer._pending) <= pending_count_ceiling
            retained_payload_bytes = sum(
                len(str(frame["params"]["payload"].get("text") or "").encode("utf-8"))
                for frame in writer._pending
            )
        assert retained_payload_bytes <= payload_ceiling
    finally:
        sink.release.set()
        if producer is not None:
            producer.join(timeout=2)
        writer.close()

    reasoning = [
        frame for frame in sink.frames if frame["params"]["type"] == "reasoning.delta"
    ]
    assert any(frame["params"]["payload"].get("resync") for frame in reasoning)


def test_reasoning_survives_overflow_on_both_sides_of_tool_boundary() -> None:
    sink = _BlockingTransport()
    writer = transport.BufferedStreamWriter(
        sink,
        queue_maxsize=8,
        max_pending_deltas=3,
        coalesce_s=0.01,
        close_timeout_s=2,
    )
    before = [f"before-{index}" for index in range(5)]
    after = [f"after-{index}" for index in range(5)]
    try:
        assert writer.write(_event("status.update", "blocked"))
        assert sink.started.wait(2), "writer never reached the blocked sink"
        for text in before:
            assert writer.write(_event("reasoning.delta", text))
        assert writer.write(_event("tool.start", "tool"))
        for text in after:
            assert writer.write(_event("thinking.delta", text))
        assert writer.write(
            _event("message.complete", "done", reasoning="".join(after))
        )
    finally:
        sink.release.set()
        writer.close()

    types = [frame["params"]["type"] for frame in sink.frames]
    tool_index = types.index("tool.start")
    before_frames = sink.frames[1:tool_index]
    after_frames = sink.frames[tool_index + 1 : -1]
    assert all(
        frame["params"]["type"] in {"reasoning.delta", "thinking.delta"}
        for frame in before_frames + after_frames
    )
    assert any(frame["params"]["payload"].get("resync") for frame in before_frames)
    assert any(frame["params"]["payload"].get("resync") for frame in after_frames)
    assert sink.frames[-1]["params"]["payload"]["reasoning"] == "".join(after)


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


def test_inner_false_latches_transport_dead() -> None:
    inner = _FailingTransport()
    writer = transport.BufferedStreamWriter(inner, coalesce_s=0.01)
    original = server._stdio_transport
    server._stdio_transport = writer
    try:
        assert server.write_json({"jsonrpc": "2.0", "id": 1, "result": {}})
        assert inner.called.wait(2)
        writer._thread.join(timeout=2)

        assert not writer._thread.is_alive()
        assert server._transport_is_dead(writer)
        assert isinstance(writer.failure, BrokenPipeError)
        assert server.write_json({"jsonrpc": "2.0", "id": 2, "result": {}}) is False
    finally:
        server._stdio_transport = original
        writer.close()


def test_peer_gone_exception_latches_clean_disconnect() -> None:
    failure = BrokenPipeError("peer closed")
    inner = _FailingTransport(failure)
    writer = transport.BufferedStreamWriter(inner, coalesce_s=0.01)
    original = server._stdio_transport
    server._stdio_transport = writer
    try:
        assert server.write_json({"jsonrpc": "2.0", "id": 1, "result": {}})
        assert inner.called.wait(2)
        writer._thread.join(timeout=2)

        assert not writer._thread.is_alive()
        assert server._transport_is_dead(writer)
        assert writer.failure is failure
        assert server.write_json({"jsonrpc": "2.0", "id": 2, "result": {}}) is False
    finally:
        server._stdio_transport = original
        writer.close()


def test_non_peer_error_is_retained_and_raised_to_next_caller() -> None:
    failure = OSError(errno.ENOSPC, "disk full")
    inner = _FailingTransport(failure)
    writer = transport.BufferedStreamWriter(inner, coalesce_s=0.01)
    original = server._stdio_transport
    server._stdio_transport = writer
    try:
        assert server.write_json({"jsonrpc": "2.0", "id": 1, "result": {}})
        assert inner.called.wait(2)
        writer._thread.join(timeout=2)

        assert not writer._thread.is_alive()
        assert server._transport_is_dead(writer)
        assert writer.failure is failure
        with pytest.raises(OSError) as raised:
            server.write_json({"jsonrpc": "2.0", "id": 2, "result": {}})
        assert raised.value is failure
    finally:
        server._stdio_transport = original
        writer.close()


def _prepare_entry_main(monkeypatch, entry, stdin=None) -> None:
    from hermes_cli import model_switch

    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(model_switch, "prewarm_picker_cache_async", lambda: None)
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *args: False)
    monkeypatch.setattr(entry.sys, "stdin", stdin or io.StringIO(""))


def _run_entry_with_held_stdin(
    monkeypatch, entry, inner, *, fail_after_ready: bool = True
) -> list[BaseException]:
    original = server._stdio_transport
    stdin = _HeldOpenStdin()
    created: list[transport.BufferedStreamWriter] = []

    def make_writer(previous):
        writer = transport.BufferedStreamWriter(
            previous,
            control_push_timeout_s=0.05,
            coalesce_s=0.01,
            close_timeout_s=0.05,
        )
        created.append(writer)
        return writer

    monkeypatch.setattr(entry, "BufferedStreamWriter", make_writer, raising=False)
    _prepare_entry_main(monkeypatch, entry, stdin)
    server._stdio_transport = inner
    failures: list[BaseException] = []
    done = threading.Event()

    def run() -> None:
        try:
            entry.main()
        except BaseException as exc:
            failures.append(exc)
        finally:
            done.set()

    runner = threading.Thread(target=run, daemon=True)
    try:
        runner.start()
        if fail_after_ready:
            assert stdin.entered.wait(2), "entry never began waiting on held-open stdin"
            assert server.write_json(_event("reasoning.delta", "trigger failure"))
            assert inner.called.wait(2), "writer never reached the failing sink"
        assert done.wait(2), (
            "entry.main did not observe dead stdout while stdin stayed open"
        )
        assert not stdin.release.is_set(), (
            "stdin was released before bounded observation"
        )
        assert server._stdio_transport is inner
    finally:
        stdin.release.set()
        sink_release = getattr(inner, "release", None)
        if isinstance(sink_release, threading.Event):
            sink_release.set()
        runner.join(timeout=2)
        for writer in created:
            writer.close()
        server._stdio_transport = original

    assert not runner.is_alive()
    return failures


def test_entry_main_scopes_buffered_writer_to_tui_stdio(monkeypatch) -> None:
    from tui_gateway import entry

    original = server._stdio_transport
    inner = _RecordingTransport()
    server._stdio_transport = inner
    created: list[transport.BufferedStreamWriter] = []

    def make_writer(inner):
        writer = transport.BufferedStreamWriter(inner, close_timeout_s=2)
        created.append(writer)
        return writer

    monkeypatch.setattr(entry, "BufferedStreamWriter", make_writer, raising=False)
    _prepare_entry_main(monkeypatch, entry)

    restored = False
    closed = False
    try:
        entry.main()
        restored = server._stdio_transport is inner
        closed = len(created) == 1 and not created[0]._thread.is_alive()
    finally:
        for writer in created:
            writer.close()
        server._stdio_transport = original

    assert len(created) == 1
    assert inner.frames[0]["params"]["type"] == "gateway.ready"
    assert restored
    assert closed


def test_entry_main_surfaces_non_peer_writer_failure(monkeypatch) -> None:
    from tui_gateway import entry

    original = server._stdio_transport
    failure = OSError(errno.ENOSPC, "disk full")
    stream = _RaisingStream(failure)
    previous = transport.StdioTransport(lambda: stream, threading.Lock())
    server._stdio_transport = previous
    _prepare_entry_main(monkeypatch, entry)

    try:
        with pytest.raises(OSError) as raised:
            entry.main()
        assert server._stdio_transport is previous
    finally:
        server._stdio_transport = original

    assert raised.value is failure
    assert stream.called.is_set()


def test_entry_main_handles_peer_gone_stdio_as_clean_disconnect(monkeypatch) -> None:
    from tui_gateway import entry

    original = server._stdio_transport
    stream = _RaisingStream(BrokenPipeError("peer closed"))
    previous = transport.StdioTransport(lambda: stream, threading.Lock())
    server._stdio_transport = previous
    _prepare_entry_main(monkeypatch, entry)

    try:
        entry.main()
        assert server._stdio_transport is previous
    finally:
        server._stdio_transport = original

    assert stream.called.is_set()


def test_entry_main_wakes_with_held_stdin_when_inner_returns_false(monkeypatch) -> None:
    from tui_gateway import entry

    failures = _run_entry_with_held_stdin(
        monkeypatch,
        entry,
        _FailAfterReadyTransport(),
    )

    assert failures == []


def test_entry_main_wakes_with_held_stdin_on_peer_exception(monkeypatch) -> None:
    from tui_gateway import entry

    failures = _run_entry_with_held_stdin(
        monkeypatch,
        entry,
        _FailAfterReadyTransport(BrokenPipeError("peer closed")),
    )

    assert failures == []


def test_entry_main_wakes_with_held_stdin_on_non_peer_exception(monkeypatch) -> None:
    from tui_gateway import entry

    failure = OSError(errno.ENOSPC, "disk full")
    failures = _run_entry_with_held_stdin(
        monkeypatch,
        entry,
        _FailAfterReadyTransport(failure),
    )

    assert failures == [failure]


def test_entry_main_wakes_with_held_stdin_when_first_control_write_blocks(
    monkeypatch,
) -> None:
    from tui_gateway import entry

    failures = _run_entry_with_held_stdin(
        monkeypatch,
        entry,
        _BlockingTransport(),
        fail_after_ready=False,
    )

    assert failures == []
