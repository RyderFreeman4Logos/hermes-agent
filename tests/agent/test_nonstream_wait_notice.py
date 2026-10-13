"""Wait status describes request-local silence, never active generation."""

from types import SimpleNamespace
import threading

import pytest

from agent import chat_completion_helpers as h
from agent.chat_completion_nonstream import _NonStreamRequest
from agent.chat_completion_wait_notice import WaitNoticeState


def _request():
    request = _NonStreamRequest.__new__(_NonStreamRequest)
    notices, touches = [], []
    request.agent = SimpleNamespace(
        _emit_wait_notice=notices.append,
        _touch_activity=touches.append,
        _interrupt_requested=False,
        _codex_stream_last_event_ts=9999.0,  # Another request cannot hide this one's stall.
    )
    request.api_kwargs = {"model": "test-model"}
    request.call_start = 1000.0
    request.wd = SimpleNamespace(
        codex=True, stale_timeout=600.0, ttfb_enabled=True, ttfb_timeout=120.0,
        idle_enabled=True, idle_timeout=180.0, idle_requires_progress=False, progress_timeout=0.0,
    )
    request.codex_watchdog_state = SimpleNamespace(
        lock=threading.Lock(), last_event_ts=None, last_progress_ts=None,
        retry_started_ts=None,
    )
    request.wait_notice_started_ts = None
    request.wait_notice = WaitNoticeState()
    request.result = {"error": None, "response": None}
    return request, notices, touches


@pytest.mark.parametrize(
    "event,progress,retry,expected",
    [
        (59.0, 59.0, None, None),  # Reasoning/text/tool arguments still arriving.
        (59.0, None, None, None),  # Lifecycle traffic is not transport silence either.
        (0.0, 0.0, None, "provider stream active; 60s without stream events"),
        (0.0, None, None, "provider stream active; 60s without stream events"),
        (1.0, None, None, None),  # 59 seconds of silence is still quiet.
        (None, None, None, "60s waiting for the first provider event"),
        (10.0, 10.0, 59.0, None),  # Internal reconnect gets a fresh first-event wait.
        (0.0, 0.0, 0.0, "60s waiting for the first provider event after reconnect"),
        (0.0, 0.0, 1.0, None),
    ],
)
def test_wait_notice_tracks_current_attempt_silence(event, progress, retry, expected):
    request, notices, touches = _request()
    state = request.codex_watchdog_state
    state.last_event_ts = None if event is None else request.call_start + event
    state.last_progress_ts = None if progress is None else request.call_start + progress
    state.retry_started_ts = None if retry is None else request.call_start + retry
    request._emit_wait_notice(30.0)
    request._emit_wait_notice(59.0)
    assert notices == []
    assert touches
    if event is None and retry is None:
        assert "receiving" not in touches[-1]
    request._emit_wait_notice(60.0)
    if expected is None:
        assert notices == []
        assert touches, "Quiet heartbeat must survive suppressing a visible warning"
    else:
        assert len(notices) == 1
        assert expected in notices[0]
        assert "auto-reconnect:" in notices[0]


def test_resumed_events_clear_only_this_requests_wait_notice(monkeypatch):
    request, notices, _ = _request()
    sentinel = object()
    ticks = [0]

    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return ticks[0] < 204

        def join(self, timeout):
            ticks[0] += 1
            if ticks[0] >= 201:
                request.codex_watchdog_state.last_event_ts = 1000.0 + ticks[0] * 0.3
            if ticks[0] == 204:
                request.result["response"] = sentinel

    monkeypatch.setattr(h.threading, "Thread", Worker)
    monkeypatch.setattr(h.time, "time", lambda: 1000.0 + ticks[0] * 0.3)
    assert request.run() is sentinel
    assert len(notices) == 2
    assert "waiting for the first provider event" in notices[0]
    # Nonempty thinking.delta payloads enter TUI reasoning history.
    assert notices[1] == ""


def test_codex_overflow_worker_preserves_error_and_attempt_origin(monkeypatch):
    """The overflow handoff must keep the four-value watchdog contract."""
    from agent.stream_payload_bound import StreamPayloadBoundExceeded

    request, _, _ = _request()
    request.call_start = 1000.0
    request.wd.ttfb_timeout = 1.0
    state = request.codex_watchdog_state
    assert state is not None
    state.retry_started_ts = None
    request.clients = h._RequestClientRegistry(request.agent)
    request.codex_token = None
    request.codex_retired = False
    request.agent._interrupt_requested = False
    overflow = StreamPayloadBoundExceeded(262_145)

    def dispatch(_agent, _kwargs, *, make_client):
        state.retry_started_ts = 1200.0
        raise overflow

    monkeypatch.setattr(h, "_dispatch_nonstreaming_api_request", dispatch)

    class Worker:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()

        def is_alive(self):
            return True

        def join(self, timeout):
            pass

    monkeypatch.setattr(h.threading, "Thread", Worker)
    monkeypatch.setattr(h.time, "time", lambda: 1202.0)
    elapsed = []
    monkeypatch.setattr(request, "_ttfb_kill", lambda seconds: elapsed.append(seconds))

    with pytest.raises(StreamPayloadBoundExceeded) as excinfo:
        request.run()
    assert excinfo.value is overflow
    assert elapsed == [2.0]
    assert request._codex_watchdog_snapshot()[-1] == 1200.0

    # A worker error must be re-raised before the interrupt path can turn it into
    # InterruptedError; the following generic path still exercises the 4-unpack.
    request.result["error"] = overflow  # type: ignore[reportArgumentType]
    with pytest.raises(StreamPayloadBoundExceeded) as excinfo:
        request._interrupt(2.0)
    assert excinfo.value is overflow

    monkeypatch.setattr(h, "_record_interrupted_provider_wait", lambda *args, **kwargs: None)
    monkeypatch.setattr(request, "_abort_request", lambda _reason: None)
    monkeypatch.setattr(h, "_join_worker_for_relay_teardown", lambda *args, **kwargs: None)
    request.result["error"] = RuntimeError("transport")  # type: ignore[reportArgumentType]
    with pytest.raises(InterruptedError):
        request._interrupt(2.0)
