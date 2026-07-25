"""Regression coverage for transient SQLite compression-lock acquisition failures."""

from __future__ import annotations

from agent import conversation_compression


class _BusyThenAvailableLockDB:
    """Lock API that mimics SQLite busy failures swallowed as ``False``."""

    def __init__(self) -> None:
        self.acquire_calls = 0

    def try_acquire_compression_lock(
        self, _session_id: str, _holder: str, *, ttl_seconds: float
    ) -> bool:
        assert ttl_seconds == 300.0
        self.acquire_calls += 1
        return self.acquire_calls >= 3

    def get_compression_lock_holder(self, _session_id: str):
        return None


class _LiveHolderLockDB:
    """A failed acquire with a verified owner must not trigger backoff."""

    def __init__(self) -> None:
        self.acquire_calls = 0

    def try_acquire_compression_lock(
        self, _session_id: str, _holder: str, *, ttl_seconds: float
    ) -> bool:
        assert ttl_seconds == 300.0
        self.acquire_calls += 1
        return False

    def get_compression_lock_holder(self, _session_id: str):
        return "pid=123:tid=456:agent=abc:nonce=def"


class _AlwaysBusyLockDB:
    def __init__(self) -> None:
        self.acquire_calls = 0

    def try_acquire_compression_lock(
        self, _session_id: str, _holder: str, *, ttl_seconds: float
    ) -> bool:
        assert ttl_seconds == 300.0
        self.acquire_calls += 1
        return False

    def get_compression_lock_holder(self, _session_id: str):
        return None


def test_acquire_retries_on_busy_then_succeeds() -> None:
    db = _BusyThenAvailableLockDB()
    slept: list[float] = []

    acquired, live_holder = conversation_compression._try_acquire_compression_lock_with_retry(
        db,
        "session-busy",
        "candidate-holder",
        ttl_seconds=300.0,
        sleep=slept.append,
        jitter=lambda _low, _high: 0.02,
    )

    assert acquired is True
    assert live_holder is None
    assert db.acquire_calls == 3
    assert slept == [0.02, 0.02]


def test_acquire_no_spin_when_live_holder() -> None:
    db = _LiveHolderLockDB()
    slept: list[float] = []

    acquired, live_holder = conversation_compression._try_acquire_compression_lock_with_retry(
        db,
        "session-held",
        "candidate-holder",
        ttl_seconds=300.0,
        sleep=slept.append,
        jitter=lambda _low, _high: 0.02,
    )

    assert acquired is False
    assert live_holder == "pid=123:tid=456:agent=abc:nonce=def"
    assert db.acquire_calls == 1
    assert slept == []


def test_acquire_retry_respects_wall_budget() -> None:
    db = _AlwaysBusyLockDB()
    clock = [0.0]
    slept: list[float] = []

    def _sleep(delay: float) -> None:
        slept.append(delay)
        clock[0] += delay

    acquired, live_holder = conversation_compression._try_acquire_compression_lock_with_retry(
        db,
        "session-busy",
        "candidate-holder",
        ttl_seconds=300.0,
        retry_budget_seconds=0.025,
        sleep=_sleep,
        jitter=lambda _low, _high: 0.150,
        clock=lambda: clock[0],
    )

    assert acquired is False
    assert live_holder is None
    assert db.acquire_calls == 2
    assert slept == [0.025]
