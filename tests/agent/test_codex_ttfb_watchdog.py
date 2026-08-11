"""Regression tests for the Codex time-to-first-byte (TTFB) watchdog.

The chatgpt.com/backend-api/codex endpoint has an intermittent failure mode
where it accepts the connection but never emits a single stream event. The
watchdog in ``interruptible_api_call`` aborts such a connection, marks
acceptance ambiguous, and retains ownership of a resistant physical worker so
no blind replay can start while it remains alive.

The "bytes flowing" signal is ``agent._codex_stream_last_event_ts``, set on
*any* event by ``codex_runtime.run_codex_stream`` — so reasoning-only or
tool-call-only turns (which emit no output-text deltas) are not mistaken for a
stall.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

_CODEX_7200_CONFIG = """\
providers:
  openai-codex:
    request_timeout_seconds: 3600
    models:
      gpt-5.5:
        timeout_seconds: 7200
"""


def _make_codex_agent(tmp_path, monkeypatch, config="{}\n"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    # The watchdog is gated on the codex_responses api_mode; assert/force it so
    # the test is robust to detection-logic changes elsewhere.
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    # Keep the wall-clock stale timeout high so any early kill is unambiguously
    # the TTFB path, not the stale-call path.
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 60.0
    )
    return agent


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now


def _install_clocked_worker(monkeypatch, helpers, clock, ready, *, step, on_poll):
    """Run the real worker while each main-thread poll advances fake time."""
    real_thread = helpers.threading.Thread
    workers = []

    class ClockedThread:
        def __init__(self, *, target, daemon):
            self._thread = real_thread(target=target, daemon=daemon)
            workers.append(self)

        def start(self):
            self._thread.start()
            assert ready.wait(1), "Codex worker did not reach the fake stream"

        def is_alive(self):
            return self._thread.is_alive()

        def join(self, timeout=None):
            if timeout == 0.3:
                clock.now += step
                on_poll(clock.now)
            self._thread.join(0.05)

        def wait(self):
            self._thread.join(1)

    monkeypatch.setattr(helpers.time, "time", clock.time)
    monkeypatch.setattr(helpers.threading, "Thread", ClockedThread)
    return workers


def _prepare_clocked_call(agent, monkeypatch, clock, *, first_event=False):
    closes = []
    ready = threading.Event()
    release = threading.Event()
    complete = {"value": False}
    sentinel = SimpleNamespace(ok=True)
    dummy_client = SimpleNamespace()

    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **_k: dummy_client
    )
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda _client, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda _client, reason=None: closes.append(reason),
    )

    def fake_stream(_api_kwargs, client=None, on_first_delta=None):
        if first_event:
            agent._codex_stream_last_event_ts = clock.time()
        ready.set()
        release.wait()
        if complete["value"]:
            return sentinel
        raise ConnectionError("incomplete chunked read")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)
    return SimpleNamespace(
        closes=closes,
        ready=ready,
        release=release,
        complete=complete,
        sentinel=sentinel,
    )


def test_codex_owner_reservation_blocks_before_worker_start():
    from agent import chat_completion_helpers as h

    agent = SimpleNamespace()
    first = {"attempt": None, "cancel_event": threading.Event()}
    owner = h._reserve_codex_request_owner(agent, first)
    owner["thread"] = SimpleNamespace(is_alive=lambda: False)
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h._reserve_codex_request_owner(
                agent,
                {"attempt": None, "cancel_event": threading.Event()},
            )
        assert getattr(
            excinfo.value,
            "_hermes_pre_dispatch_retained_owner",
            False,
        ) is True
        assert getattr(agent, "_codex_request_owner") is owner
    finally:
        h._release_codex_request_owner(agent, owner)


def test_retained_codex_owner_blocks_non_codex_route_switch(
    tmp_path, monkeypatch
):
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    owner = h._reserve_codex_request_owner(
        agent, {"attempt": None, "cancel_event": threading.Event()}
    )
    owner.update(thread=SimpleNamespace(is_alive=lambda: True), started=True)
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    dispatched = []
    monkeypatch.setattr(
        h,
        "_dispatch_nonstreaming_api_request",
        lambda *_args, **_kwargs: dispatched.append(True),
    )

    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "fallback", "messages": []})
        assert getattr(
            excinfo.value, "_hermes_pre_dispatch_retained_owner", False
        ) is True
        assert dispatched == []
        assert getattr(agent, "_codex_request_owner") is owner
    finally:
        h._release_codex_request_owner(agent, owner)


def test_retained_codex_owner_blocks_direct_route_before_early_return(
    tmp_path, monkeypatch
):
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    owner = h._reserve_codex_request_owner(
        agent, {"attempt": None, "cancel_event": threading.Event()}
    )
    owner.update(thread=SimpleNamespace(is_alive=lambda: True), started=True)
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent.platform = "cron"
    dispatched = []
    monkeypatch.setattr(
        h,
        "direct_api_call",
        lambda *_args, **_kwargs: dispatched.append(True),
    )

    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "fallback", "messages": []})
        assert getattr(
            excinfo.value, "_hermes_pre_dispatch_retained_owner", False
        ) is True
        assert dispatched == []
        assert getattr(agent, "_codex_request_owner") is owner
    finally:
        h._release_codex_request_owner(agent, owner)


def test_ttfb_includes_silent_hang_hint_for_gpt_5_5(tmp_path, monkeypatch):
    """The no-first-byte watchdog should surface the same actionable hint as the
    stale-call timeout path when the model matches the silent-hang heuristic."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.4")

    closes: list = []
    statuses: list[str] = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **k: dummy_client
    )
    monkeypatch.setattr(agent, "_buffer_status", lambda msg: statuses.append(msg))
    monkeypatch.setattr(agent, "_emit_status", lambda msg: statuses.append(msg))
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        deadline = time.time() + 30
        while (
            time.time() < deadline
            and not stop["flag"]
            and not agent._interrupt_requested
        ):
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        message = str(excinfo.value)
        assert "gpt-5.4" in message
        assert "gpt-5.3-codex" in message
        assert "gpt-5.4-codex" in message
        assert "codex_ttfb_kill" in closes
        assert statuses, "expected a user-facing watchdog status"
        assert any("gpt-5.4" in s and "gpt-5.3-codex" in s for s in statuses)
    finally:
        stop["flag"] = True


def test_ttfb_does_not_kill_when_events_flow(tmp_path, monkeypatch):
    """Once a stream event has arrived, a generation that runs past the TTFB
    cutoff is NOT killed by the watchdog — it completes normally."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.4")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **k: dummy_client
    )
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        # Bytes flowing: mark stream activity right away, then keep generating
        # past the 0.4s TTFB cutoff before returning a real response.
        agent._codex_stream_last_event_ts = time.time()
        if on_first_delta:
            on_first_delta()
        time.sleep(0.9)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
    assert resp is sentinel
    assert "codex_ttfb_kill" not in closes


@pytest.mark.parametrize(
    "stale_timeout",
    [float("inf"), float("-inf"), float("nan")],
)
def test_wait_notice_omits_safety_timeout_when_all_deadlines_are_non_finite(
    stale_timeout,
):
    """A disabled watchdog must not advertise a future safety timeout."""
    from agent import chat_completion_helpers as h

    recovery = h._codex_wait_notice_recovery(
        stale_timeout=stale_timeout,
        ttfb_enabled=False,
        ttfb_timeout=float("nan"),
        last_event_ts=None,
        call_start=100.0,
        idle_enabled=False,
        idle_timeout=float("nan"),
        elapsed=30.0,
    )

    assert recovery == ""


def test_moa_heartbeat_survives_infinite_stale_timeout(monkeypatch):
    """The full 100-poll MoA heartbeat must leave a healthy call running."""
    from agent import chat_completion_helpers as h

    notices: list[str] = []
    response = SimpleNamespace(ok=True)
    agent = SimpleNamespace(
        platform="desktop",
        api_mode="chat_completions",
        provider="moa",
        _consecutive_stale_streams=0,
        _interrupt_requested=False,
        _compute_non_stream_stale_timeout=lambda _kwargs: float("inf"),
        _touch_activity=lambda _message: None,
        _emit_wait_notice=notices.append,
    )

    class HeartbeatThread:
        """Keep the synthetic worker alive through one heartbeat."""

        def __init__(self, *, target, daemon):
            self._polls = 0
            self._target = target

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            self._polls += 1
            if self._polls == 101:
                self._target()
                return False
            return True

    monkeypatch.setattr(h.threading, "Thread", HeartbeatThread)
    monkeypatch.setattr(
        h,
        "_dispatch_nonstreaming_api_request",
        lambda *_args, **_kwargs: response,
    )

    result = h.interruptible_api_call(agent, {"model": "openai-xai-wide"})

    assert result is response
    assert len(notices) == 1
    assert "waiting on openai-xai-wide" in notices[0]
    assert "auto-reconnect" not in notices[0]


def test_wait_notice_formatting_error_does_not_abort_request(monkeypatch):
    """Status construction is fail-open even if its formatter breaks."""
    from agent import chat_completion_helpers as h

    response = SimpleNamespace(ok=True)
    agent = SimpleNamespace(
        platform="desktop",
        api_mode="chat_completions",
        provider="moa",
        _consecutive_stale_streams=0,
        _interrupt_requested=False,
        _compute_non_stream_stale_timeout=lambda _kwargs: float("inf"),
        _touch_activity=lambda _message: None,
        _emit_wait_notice=lambda _message: None,
    )

    class HeartbeatThread:
        def __init__(self, *, target, daemon):
            self._polls = 0
            self._target = target

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            self._polls += 1
            if self._polls == 101:
                self._target()
                return False
            return True

    monkeypatch.setattr(h.threading, "Thread", HeartbeatThread)
    monkeypatch.setattr(
        h,
        "_dispatch_nonstreaming_api_request",
        lambda *_args, **_kwargs: response,
    )
    monkeypatch.setattr(
        h,
        "_codex_wait_notice_recovery",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad display state")),
    )

    result = h.interruptible_api_call(agent, {"model": "openai-xai-wide"})

    assert result is response


@pytest.mark.parametrize(
    "mode,step,expected_now,timeout_class,close_reason,first_event",
    [
        ("none", 121.0, 121.0, "ttfb_timeout", "codex_ttfb_kill", False),
        ("none", 13.0, 13.0, "sse_idle_timeout", "codex_stream_idle_kill", True),
        # #64507: keepalives cannot hide a request with no semantic progress.
        (
            "keepalive",
            600.0,
            1800.0,
            "no_progress_timeout",
            "codex_no_progress_timeout",
            False,
        ),
        (
            "semantic",
            600.0,
            7200.0,
            "total_request_timeout",
            "codex_total_request_timeout",
            False,
        ),
    ],
    ids=["ttfb", "sse-idle", "no-progress", "configured-total"],
)
def test_codex_timeout_classes_use_fake_clock_and_retry_path(
    tmp_path,
    monkeypatch,
    mode,
    step,
    expected_now,
    timeout_class,
    close_reason,
    first_event,
):
    from agent import chat_completion_helpers as h
    from agent import physical_attempt_diagnostics as diagnostics
    from agent.error_classifier import classify_api_error

    lifecycle = []
    monkeypatch.setattr(
        diagnostics,
        "record_ambiguity",
        lambda attempt, **metadata: lifecycle.append(("ambiguity", attempt, metadata)),
    )
    monkeypatch.setattr(
        diagnostics,
        "record_reconciliation",
        lambda attempt, **metadata: lifecycle.append(
            ("reconciliation", attempt, metadata)
        ),
    )

    agent = _make_codex_agent(
        tmp_path,
        monkeypatch,
        config=_CODEX_7200_CONFIG,
    )
    assert agent._resolved_api_call_timeout() == 7200.0
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *_a: 1200.0)
    clock = _FakeClock()
    call = _prepare_clocked_call(agent, monkeypatch, clock, first_event=first_event)
    attempt = object()
    fake_stream = agent._run_codex_stream

    def linked_stream(api_kwargs, **kwargs):
        lifecycle_state = api_kwargs.get(diagnostics._INTERNAL_LIFECYCLE_KEY)
        if lifecycle_state is not None:
            lifecycle_state["attempt"] = attempt
        return fake_stream(api_kwargs, **kwargs)

    monkeypatch.setattr(agent, "_run_codex_stream", linked_stream)

    def activity(now):
        if mode in {"keepalive", "semantic"}:
            setattr(agent, "_codex_stream_last_event_ts", now)
        if mode == "semantic":
            setattr(agent, "_codex_stream_last_progress_ts", now)

    real_thread = h.threading.Thread
    workers = _install_clocked_worker(
        monkeypatch, h, clock, call.ready, step=step, on_poll=activity
    )
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        timeout_error = excinfo.value
        assert clock.now == expected_now
        assert f"timeout_class={timeout_class}" in str(timeout_error)
        assert getattr(
            timeout_error,
            "_hermes_ambiguous_provider_acceptance",
            False,
        ) is True
        assert call.closes.count(close_reason) == 1
        assert classify_api_error(timeout_error).retryable is True
        owner = getattr(agent, "_codex_request_owner")
        assert owner["thread"].is_alive()
        assert owner["lifecycle"]["cancel_event"].is_set()
        with pytest.raises(TimeoutError) as blocked:
            h.interruptible_api_call(
                agent,
                {"model": "gpt-5.5", "input": "must not dispatch"},
            )
        assert getattr(
            blocked.value,
            "_hermes_pre_dispatch_retained_owner",
            False,
        ) is True
        assert len(workers) == 1
        assert getattr(agent, "_codex_request_owner") is owner
        assert lifecycle == [
            (
                "ambiguity",
                attempt,
                {"failure_class": timeout_class},
            ),
            (
                "reconciliation",
                attempt,
                {"action": "retain"},
            ),
            (
                "reconciliation",
                attempt,
                {"action": "wait"},
            ),
        ]
    finally:
        call.release.set()
        for worker in workers:
            worker.wait()

    # Feed the exact helper-generated object through the conversation loop:
    # ambiguity must terminate locally rather than replaying the billed request.
    monkeypatch.setattr(h.threading, "Thread", real_thread)
    from agent import conversation_loop
    import run_agent

    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **_kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(agent, "tools", [], raising=False)
    monkeypatch.setattr(agent, "valid_tool_names", set(), raising=False)
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda task_id: None)
    monkeypatch.setattr(
        agent,
        "_persist_session",
        lambda messages, conversation_history=None: None,
    )
    monkeypatch.setattr(
        agent,
        "_save_trajectory",
        lambda messages, user_query, completed: None,
    )
    calls = []

    def _raise_same_timeout(_kwargs):
        calls.append(None)
        raise timeout_error

    monkeypatch.setattr(agent, "_disable_streaming", True, raising=False)
    monkeypatch.setattr(agent, "_interruptible_api_call", _raise_same_timeout)
    monkeypatch.setattr(conversation_loop, "jittered_backoff", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(conversation_loop.time, "sleep", lambda _seconds: None)
    result = agent.run_conversation("one potentially billed action")

    # The original ambiguous attempt plus up to three fresh checkpoint
    # continuations are allowed. These are new reconciliation requests, not
    # transport replays of the originally accepted request.
    assert len(calls) == 4
    assert result["ambiguous_provider_attempt"] is True
    assert result.get("turn_exit_reason") == "ambiguous_provider_attempt"
    assert (
        result.get("failure_class") == timeout_class
        or result.get("timeout_class") == timeout_class
    )
    assert result["messages"][-1]["role"] == "assistant"


def test_semantic_progress_survives_1200_and_1500_seconds(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(
        tmp_path,
        monkeypatch,
        config=_CODEX_7200_CONFIG,
    )
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *_a: 1200.0)
    clock = _FakeClock()
    call = _prepare_clocked_call(agent, monkeypatch, clock)

    def semantic_progress(now):
        setattr(agent, "_codex_stream_last_event_ts", now)
        setattr(agent, "_codex_stream_last_progress_ts", now)
        if now >= 2400.0:
            call.complete["value"] = True
            call.release.set()

    workers = _install_clocked_worker(
        monkeypatch, h, clock, call.ready, step=600.0, on_poll=semantic_progress
    )
    try:
        response = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        assert response is call.sentinel
        assert clock.now == 2400.0
        assert not any(reason and "timeout" in reason for reason in call.closes)
    finally:
        call.release.set()
        for worker in workers:
            worker.wait()


def test_broken_codex_transport_surfaces_to_bounded_recovery(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h
    from agent.error_classifier import classify_api_error

    agent = _make_codex_agent(tmp_path, monkeypatch)
    transport_error = ConnectionError("incomplete chunked read")
    dummy_client = SimpleNamespace()
    closes = []
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **_k: dummy_client
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda _client, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_run_codex_stream",
        lambda *_a, **_k: (_ for _ in ()).throw(transport_error),
    )

    with pytest.raises(ConnectionError) as excinfo:
        h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})

    assert excinfo.value is transport_error
    assert closes.count("request_error_cleanup") == 1
    assert classify_api_error(excinfo.value).retryable is True


def test_codex_parser_counts_only_semantic_sse_progress():
    from agent.codex_runtime import _consume_codex_event_stream

    progress = []
    response = _consume_codex_event_stream(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.in_progress"),
            SimpleNamespace(type="response.output_text.delta", delta="answer"),
            SimpleNamespace(type="response.reasoning_text.delta", delta="thinking"),
            SimpleNamespace(type="response.function_call_arguments.delta", delta="{}"),
            SimpleNamespace(
                type="response.custom_tool_call_input.delta", delta="search query"
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed"),
            ),
        ],
        model="gpt-5.5",
        on_progress=lambda: progress.append(True),
    )

    assert response.status == "completed"
    assert len(progress) == 4


def test_outer_watchdog_timeout_copies_partial_tool_flag(tmp_path, monkeypatch):
    """Watchdog settle must terminate recovery when stream saw a tool-call event."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    clock = _FakeClock()
    call = _prepare_clocked_call(agent, monkeypatch, clock, first_event=False)

    def mark_partial(now):
        # Simulate a function_call event observed by the stream worker, then
        # silence that triggers the outer TTFB watchdog.
        agent._codex_stream_partial_tool_call = True
        if now >= 121.0:
            call.complete["value"] = False

    workers = _install_clocked_worker(
        monkeypatch, h, clock, call.ready, step=121.0, on_poll=mark_partial
    )
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        timeout_error = excinfo.value
        assert getattr(timeout_error, "_hermes_ambiguous_provider_acceptance", False)
        assert getattr(timeout_error, "_hermes_ambiguous_partial_tool_call", False)
        assert getattr(timeout_error, "timeout_class", None) == "ttfb_timeout"
    finally:
        call.release.set()
        for worker in workers:
            worker.wait()


def test_partial_tool_watchdog_timeout_does_not_continue(monkeypatch, tmp_path):
    """Conversation loop must not continue after partial-tool ambiguity."""
    from agent import conversation_loop
    import run_agent

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **_kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(agent, "tools", [], raising=False)
    monkeypatch.setattr(agent, "valid_tool_names", set(), raising=False)
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda task_id: None)
    monkeypatch.setattr(
        agent, "_persist_session", lambda messages, conversation_history=None: None
    )
    monkeypatch.setattr(
        agent, "_save_trajectory", lambda messages, user_query, completed: None
    )
    monkeypatch.setattr(conversation_loop, "jittered_backoff", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(conversation_loop.time, "sleep", lambda _seconds: None)

    calls = []
    timeout_error = TimeoutError("timeout_class=ttfb_timeout; partial tool boundary")
    timeout_error._hermes_ambiguous_provider_acceptance = True
    timeout_error._hermes_ambiguous_partial_tool_call = True
    timeout_error.timeout_class = "ttfb_timeout"
    timeout_error.failure_class = "ttfb_timeout"

    def _raise_timeout(_kwargs):
        calls.append(None)
        raise timeout_error

    monkeypatch.setattr(agent, "_disable_streaming", True, raising=False)
    monkeypatch.setattr(agent, "_interruptible_api_call", _raise_timeout)
    result = agent.run_conversation("must not continue after partial tool")
    assert len(calls) == 1
    assert result["ambiguous_provider_attempt"] is True
    assert result.get("turn_exit_reason") == "ambiguous_provider_attempt"
    assert result.get("failure_class") == "ttfb_timeout"


def test_large_request_ttfb_scale_is_not_capped_by_implicit_default(monkeypatch, caplog):
    """#92 / upstream #69228: implicit max must not undo adaptive scale-up."""
    from agent import chat_completion_helpers as h

    monkeypatch.delenv("HERMES_CODEX_TTFB_MAX_SECONDS", raising=False)
    monkeypatch.delenv("HERMES_CODEX_TTFB_STRICT", raising=False)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "120")
    monkeypatch.setattr(
        h, "estimate_request_context_tokens", lambda _payload: 150_000
    )
    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    with caplog.at_level("INFO"):
        (
            _enabled,
            _openai,
            est,
            ttfb_enabled,
            ttfb_timeout,
            _idle_enabled,
            idle_timeout,
        ) = h._resolve_codex_stream_watchdogs(agent, {"input": "x" * 100})

    assert est == 150_000
    assert ttfb_enabled is True
    assert ttfb_timeout == 180.0
    assert idle_timeout == 180.0
    assert not any("Capping openai-codex no-byte TTFB" in r.message for r in caplog.records)


def test_explicit_ttfb_max_still_caps_adaptive_scale(monkeypatch):
    from agent import chat_completion_helpers as h

    monkeypatch.setenv("HERMES_CODEX_TTFB_MAX_SECONDS", "150")
    monkeypatch.delenv("HERMES_CODEX_TTFB_STRICT", raising=False)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "120")
    monkeypatch.setattr(
        h, "estimate_request_context_tokens", lambda _payload: 150_000
    )
    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    (
        _enabled,
        _openai,
        _est,
        ttfb_enabled,
        ttfb_timeout,
        _idle_enabled,
        _idle_timeout,
    ) = h._resolve_codex_stream_watchdogs(agent, {"input": "x"}, log_adjustments=False)
    assert ttfb_enabled is True
    assert ttfb_timeout == 150.0
