"""Tests for MemoryManager.commit_session_boundary_async.

The /new session boundary must deliver on_session_end (old-session
extraction) strictly BEFORE on_session_switch (provider rebinding to the
new session), without blocking the caller. Both hooks run as one task on
the manager's single serialized background worker.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Provider that records hook invocations with thread identity."""

    def __init__(self, end_delay: float = 0.0):
        self.calls: List[tuple] = []
        self._end_delay = end_delay
        self._caller_thread_ids: List[int] = []

    # Required ABC surface (minimal no-ops)
    @property
    def name(self) -> str:
        return "recorder"

    def is_available(self) -> bool:
        return True

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def initialize(self, agent: Any = None, **kwargs) -> bool:  # type: ignore[override]
        return True

    def build_system_prompt(self) -> str:  # type: ignore[override]
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None:  # type: ignore[override]
        self.calls.append(("sync_turn", kwargs.get("session_id", "")))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._end_delay:
            time.sleep(self._end_delay)
        self._caller_thread_ids.append(threading.get_ident())
        self.calls.append(("end", list(messages)))

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self.calls.append(("switch", new_session_id, kwargs.get("reset")))

    def shutdown(self) -> None:
        self.calls.append(("shutdown",))


def _make_manager(provider: _RecordingProvider) -> MemoryManager:
    mm = MemoryManager()
    mm._providers.append(provider)  # bypass add_provider validation for the stub
    return mm


def test_boundary_commit_delivers_end_strictly_before_switch():
    """Even with a slow (LLM-like) extraction, switch waits for end."""
    provider = _RecordingProvider(end_delay=0.15)
    mm = _make_manager(provider)

    msgs = [{"role": "user", "content": "old turn"}]
    mm.commit_session_boundary_async(
        msgs, new_session_id="new-sid", parent_session_id="old-sid"
    )
    # DETERMINISTIC non-blocking witness — replaces `assert elapsed < 0.1`.
    #
    # The old form timed `commit_session_boundary_async` and required it under
    # 100ms, which makes the scheduler part of the assertion: thread startup
    # alone can exceed that on a loaded box, flipping the inequality with
    # nothing wrong in the code under test.
    #
    # The real contract is that the caller returns WITHOUT waiting for the slow
    # extraction. Assert it directly: the background `on_session_end` sleeps
    # 0.15s before recording anything, so if the caller had blocked on it, the
    # provider would already have recorded the "end" call by the time we get
    # here. An empty call list is a positive witness that /new was not gated.
    assert provider.calls == [], (
        "commit_session_boundary_async blocked on the slow extraction: "
        f"provider already recorded {provider.calls} before the caller returned"
    )

    assert mm.flush_pending(timeout=30)

    kinds = [c[0] for c in provider.calls]
    assert kinds == ["end", "switch"], f"ordering violated: {provider.calls}"
    assert provider.calls[0] == ("end", msgs)
    assert provider.calls[1] == ("switch", "new-sid", True)
    # And it genuinely ran off the caller's thread.
    assert provider._caller_thread_ids[0] != threading.get_ident()




def test_boundary_commit_switch_still_fires_when_end_raises():
    """A failing provider extraction must not strand providers on the old sid."""

    class _ExplodingEndProvider(_RecordingProvider):
        def on_session_end(self, messages):  # type: ignore[override]
            raise RuntimeError("provider extraction blew up")

    provider = _ExplodingEndProvider()
    mm = _make_manager(provider)

    mm.commit_session_boundary_async([{"role": "user", "content": "x"}], new_session_id="new-sid")
    assert mm.flush_pending(timeout=5)

    assert ("switch", "new-sid", True) in provider.calls


def test_resume_boundary_uses_reset_false():
    provider = _RecordingProvider()
    mm = _make_manager(provider)

    mm.commit_session_boundary_async(
        [], new_session_id="target", reset=False, reason="resume"
    )
    assert mm.flush_pending(timeout=5)

    assert provider.calls == [("end", []), ("switch", "target", False)]


def test_retirement_cuts_admission_and_orders_write_end_shutdown():
    started = threading.Event()
    release_write = threading.Event()
    release_retirement = threading.Event()

    class _BlockingProvider(_RecordingProvider):
        def sync_turn(self, user_content, assistant_content, **kwargs):
            if user_content == "first":
                started.set()
                assert release_write.wait(timeout=5)
            self.calls.append(("write", user_content))

    provider = _BlockingProvider()
    mm = _make_manager(provider)
    mm.sync_all("first", "response")
    assert started.wait(timeout=2)
    mm.sync_all("second", "response")

    ticket = mm.commit_session_boundary_async(
        [{"role": "user", "content": "outgoing"}],
        new_session_id="",
        retire=True,
        release_event=release_retirement,
    )
    mm.sync_all("late", "response")
    release_write.set()
    release_retirement.set()
    assert ticket.result(timeout=5) is None

    assert provider.calls == [
        ("write", "first"),
        ("write", "second"),
        ("end", [{"role": "user", "content": "outgoing"}]),
        ("shutdown",),
    ]


def test_retirement_shutdown_runs_when_end_raises():
    class _ExplodingEndProvider(_RecordingProvider):
        def on_session_end(self, messages):
            self.calls.append(("end", list(messages)))
            raise RuntimeError("boom")

    provider = _ExplodingEndProvider()
    mm = _make_manager(provider)
    release = threading.Event()
    ticket = mm.commit_session_boundary_async(
        [], new_session_id="", retire=True, release_event=release
    )
    release.set()
    assert ticket.result(timeout=5) is None
    assert provider.calls == [("end", []), ("shutdown",)]


class _ShutdownFailureProvider(_RecordingProvider):
    def __init__(self, name: str, calls: list, failure: BaseException | None = None):
        super().__init__()
        self._name = name
        self._all_calls = calls
        self._failure = failure

    @property
    def name(self) -> str:
        return self._name

    def shutdown(self) -> None:
        self._all_calls.append(self._name)
        if self._failure is not None:
            raise self._failure


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_shutdown_attempts_every_provider_after_ordinary_failure(failure_index):
    calls = []
    providers = [
        _ShutdownFailureProvider(
            str(index),
            calls,
            RuntimeError("ordinary shutdown failure") if index == failure_index else None,
        )
        for index in range(3)
    ]
    mm = MemoryManager()
    mm._providers = providers

    mm._shutdown_providers()

    assert calls == ["2", "1", "0"]


@pytest.mark.parametrize("failure_index", [0, 1, 2])
@pytest.mark.parametrize(
    "failure",
    [KeyboardInterrupt(), SystemExit(2), asyncio.CancelledError()],
    ids=["keyboard-interrupt", "system-exit", "cancelled-error"],
)
def test_shutdown_attempts_every_provider_then_reraises_baseexception(
    failure_index, failure
):
    calls = []
    providers = [
        _ShutdownFailureProvider(
            str(index), calls, failure if index == failure_index else None
        )
        for index in range(3)
    ]
    mm = MemoryManager()
    mm._providers = providers

    with pytest.raises(type(failure)) as raised:
        mm._shutdown_providers()

    assert raised.value is failure
    assert calls == ["2", "1", "0"]


def test_shutdown_reraises_first_baseexception_after_all_attempts():
    calls = []
    failures = [KeyboardInterrupt(), SystemExit(2), asyncio.CancelledError()]
    mm = MemoryManager()
    mm._providers = [
        _ShutdownFailureProvider(str(index), calls, failure)
        for index, failure in enumerate(failures)
    ]

    with pytest.raises(asyncio.CancelledError) as raised:
        mm._shutdown_providers()

    assert raised.value is failures[2]
    assert calls == ["2", "1", "0"]


def test_retirement_baseexception_reports_drained_after_sibling_cleanup():
    calls = []
    failure = KeyboardInterrupt()
    mm = MemoryManager()
    mm._providers = [
        _ShutdownFailureProvider("first", calls),
        _ShutdownFailureProvider("fatal", calls, failure),
        _ShutdownFailureProvider("last", calls),
    ]
    release = threading.Event()
    ticket = mm.commit_session_boundary_async(
        [], new_session_id="", retire=True, release_event=release
    )
    release.set()

    with pytest.raises(KeyboardInterrupt) as raised:
        ticket.result(timeout=5)

    assert raised.value is failure
    assert calls == ["last", "fatal", "first"]
    assert mm.shutdown_drain_state == {
        "status": "drained",
        "abandoned_writes": 0,
        "abandoned_prefetches": 0,
        "active_tasks": 0,
    }


