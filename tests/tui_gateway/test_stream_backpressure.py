"""Focused bounded-stream contracts for the TUI stdio transport."""

from __future__ import annotations

import io
import json
import queue
import threading
import time
import types

from tui_gateway import server, transport


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


class _RecordingTransport:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def write(self, obj: dict) -> bool:
        self.frames.append(obj)
        return True

    def close(self) -> None:
        return None


class _FailAfterReadyTransport:
    def __init__(self) -> None:
        self.writes = 0
        self.failed = threading.Event()

    def write(self, _obj: dict) -> bool:
        self.writes += 1
        if self.writes == 1:
            return True
        self.failed.set()
        return False

    def close(self) -> None:
        return None


class _HeldOpenStdin:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def readline(self) -> str:
        self.entered.set()
        assert self.release.wait(2), "test stdin was not released"
        return ""


class _BurstStdin:
    def __init__(self, count: int = 20) -> None:
        self.count = count
        self.read_count = 0
        self._lock = threading.Lock()

    def readline(self) -> str:
        with self._lock:
            self.read_count += 1
            index = self.read_count
        if index > self.count:
            return ""
        return json.dumps({"jsonrpc": "2.0", "id": index, "method": "test"}) + "\n"


def _event(kind: str, text: str = "", **payload: object) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": kind,
            "session_id": "test-session",
            "payload": {"text": text, **payload},
        },
    }


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(0.005)
    return predicate()


def _prepare_entry(monkeypatch, stdin: object) -> None:
    from hermes_cli import model_switch
    from tui_gateway import entry

    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *_args: False)
    monkeypatch.setattr(entry, "_log_exit", lambda _reason: None)
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(model_switch, "prewarm_picker_cache_async", lambda: None)
    monkeypatch.setattr(entry.sys, "stdin", stdin)


def test_message_deltas_stay_nonblocking_bounded_and_precede_completion() -> None:
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


def test_reasoning_backpressure_is_bounded_and_lossless_across_control() -> None:
    sink = _BlockingTransport()
    byte_cap = 700
    writer = transport.BufferedStreamWriter(
        sink,
        max_pending_deltas=4,
        max_pending_bytes=byte_cap,
        control_push_timeout_s=0.5,
        coalesce_s=0.001,
        close_timeout_s=1,
    )
    before = ["界" * 400, *(f"before-{index}-界" * 4 for index in range(12))]
    after = [f"after-{index}-界" * 4 for index in range(12)]
    results: list[bool] = []

    def produce() -> None:
        results.extend(writer.write(_event("reasoning.delta", part)) for part in before)
        results.append(writer.write(_event("tool.start", "tool")))
        results.extend(writer.write(_event("thinking.delta", part)) for part in after)
        results.append(writer.write(_event("message.complete", "done")))

    producer = threading.Thread(target=produce, daemon=True)
    observed_pending = 0
    try:
        assert writer.write(_event("status.update", "blocked"))
        assert sink.started.wait(1)
        producer.start()
        assert _wait_until(lambda: writer.pending_bytes >= byte_cap // 2)
        observed_pending = writer.pending_bytes
        assert producer.is_alive(), "reasoning overflow must apply bounded backpressure"
    finally:
        sink.release.set()
        producer.join(timeout=2)
        writer.close()

    assert observed_pending <= byte_cap
    assert not producer.is_alive()
    assert all(results)
    types = [frame["params"]["type"] for frame in sink.frames]
    tool_index = types.index("tool.start")
    before_frames = sink.frames[1:tool_index]
    after_frames = sink.frames[tool_index + 1 : -1]
    assert "".join(frame["params"]["payload"]["text"] for frame in before_frames) == "".join(before)
    assert "".join(frame["params"]["payload"]["text"] for frame in after_frames) == "".join(after)


def test_total_pending_utf8_bytes_are_bounded_across_deltas_and_controls() -> None:
    byte_cap = 1_024

    def retained_at_scale(scale: int) -> int:
        sink = _BlockingTransport()
        writer = transport.BufferedStreamWriter(
            sink,
            max_pending_bytes=byte_cap,
            coalesce_s=0.001,
            close_timeout_s=1,
        )
        try:
            assert writer.write(_event("status.update", "blocked"))
            assert sink.started.wait(1)
            for _ in range(8):
                assert writer.write(_event("message.delta", "界" * scale))
            assert writer.write(_event("message.complete", "终" * 80))
            return writer.pending_bytes
        finally:
            sink.release.set()
            writer.close()

    small = retained_at_scale(8)
    large = retained_at_scale(8_192)
    assert 0 < small <= byte_cap
    assert 0 < large <= byte_cap


def test_control_timeout_latches_lifecycle_compatible_dead_state() -> None:
    sink = _BlockingTransport()
    writer = transport.BufferedStreamWriter(
        sink,
        queue_maxsize=1,
        max_pending_bytes=16_384,
        control_push_timeout_s=0.03,
        coalesce_s=0.001,
        close_timeout_s=0.05,
    )
    try:
        assert writer.write(_event("tool.start", "A"))
        assert sink.started.wait(1)
        assert writer.write(_event("tool.start", "B"))
        assert writer.write(_event("message.complete", "C")) is False

        assert writer._closed is True
        assert writer.wait_closed(0)
        assert server._transport_is_dead(writer)
        assert isinstance(writer.failure, transport.ControlQueueTimeoutError)
        writer.raise_if_failed()
    finally:
        sink.release.set()
        writer.close()


def test_entry_bounds_stdin_read_ahead_while_dispatch_is_held(monkeypatch) -> None:
    from tui_gateway import entry

    stdin = _BurstStdin()
    _prepare_entry(monkeypatch, stdin)
    created: list[queue.Queue] = []
    queue_created = threading.Event()
    real_queue = queue.Queue

    class _CapturingQueue(real_queue):
        def __init__(self, maxsize: int = 0) -> None:
            super().__init__(maxsize=maxsize)
            created.append(self)
            queue_created.set()

    monkeypatch.setattr(
        entry,
        "queue",
        types.SimpleNamespace(Queue=_CapturingQueue, Empty=queue.Empty, Full=queue.Full),
    )
    dispatch_started = threading.Event()
    dispatch_release = threading.Event()

    def held_dispatch(_request: dict) -> None:
        if not dispatch_started.is_set():
            dispatch_started.set()
            assert dispatch_release.wait(2), "test dispatch was not released"
        return None

    monkeypatch.setattr(entry, "dispatch", held_dispatch)
    original = server._stdio_transport
    server._stdio_transport = _RecordingTransport()
    failures: list[BaseException] = []
    runner = threading.Thread(
        target=lambda: _capture_failure(entry.main, failures), daemon=True
    )
    try:
        runner.start()
        assert queue_created.wait(1)
        assert dispatch_started.wait(1)
        stdin_queue = created[0]
        assert stdin_queue.maxsize == 1
        assert _wait_until(lambda: stdin_queue.full())
        assert stdin.read_count <= 3
    finally:
        dispatch_release.set()
        runner.join(timeout=2)
        server._stdio_transport = original

    assert not runner.is_alive()
    assert failures == []


def _capture_failure(callback, failures: list[BaseException]) -> None:
    try:
        callback()
    except BaseException as exc:
        failures.append(exc)


def test_entry_writer_failure_wakes_held_stdin(monkeypatch) -> None:
    from tui_gateway import entry

    stdin = _HeldOpenStdin()
    _prepare_entry(monkeypatch, stdin)
    inner = _FailAfterReadyTransport()
    original = server._stdio_transport
    server._stdio_transport = inner
    failures: list[BaseException] = []
    done = threading.Event()

    def run() -> None:
        _capture_failure(entry.main, failures)
        done.set()

    runner = threading.Thread(target=run, daemon=True)
    try:
        runner.start()
        assert stdin.entered.wait(1)
        assert server.write_json(_event("reasoning.delta", "trigger failure"))
        assert inner.failed.wait(1)
        assert done.wait(1), "writer failure did not wake held-open stdin"
        assert not stdin.release.is_set()
    finally:
        stdin.release.set()
        runner.join(timeout=2)
        server._stdio_transport = original

    assert not runner.is_alive()
    assert failures == []


def test_entry_initial_ready_write_has_finite_physical_deadline(monkeypatch) -> None:
    from tui_gateway import entry

    stdin = _HeldOpenStdin()
    _prepare_entry(monkeypatch, stdin)
    created: list[transport.BufferedStreamWriter] = []

    def make_writer(inner):
        writer = transport.BufferedStreamWriter(
            inner,
            control_push_timeout_s=0.03,
            coalesce_s=0.001,
            close_timeout_s=0.03,
        )
        created.append(writer)
        return writer

    monkeypatch.setattr(entry, "BufferedStreamWriter", make_writer)
    sink = _BlockingTransport()
    original = server._stdio_transport
    server._stdio_transport = sink
    failures: list[BaseException] = []
    done = threading.Event()

    def run() -> None:
        _capture_failure(entry.main, failures)
        done.set()

    runner = threading.Thread(target=run, daemon=True)
    try:
        runner.start()
        assert sink.started.wait(1)
        assert done.wait(1), "entry stayed live behind an unacknowledged ready frame"
        assert not sink.release.is_set()
        assert not stdin.entered.is_set()
        assert len(created) == 1
        assert isinstance(created[0].failure, transport.ControlQueueTimeoutError)
    finally:
        sink.release.set()
        stdin.release.set()
        runner.join(timeout=2)
        for writer in created:
            writer.close()
        server._stdio_transport = original

    assert not runner.is_alive()
    assert failures == []
