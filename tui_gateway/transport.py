"""Transport abstraction for the tui_gateway JSON-RPC server.

Historically the gateway wrote every JSON frame directly to real stdout.  This
module decouples the I/O sink from the handler logic so the same dispatcher
can be driven over stdio (``tui_gateway.entry``) or WebSocket
(``tui_gateway.ws``) without duplicating code.

A :class:`Transport` is anything that can accept a JSON-serialisable dict and
forward it to its peer.  The active transport for the current request is
tracked in a :class:`contextvars.ContextVar` so handlers — including those
dispatched onto the worker pool — route their writes to the right peer.

Backward compatibility
----------------------
``tui_gateway.server.write_json`` still works without any transport bound.
When nothing is on the contextvar and no session-level transport is found,
it falls back to the module-level :class:`StdioTransport`, which wraps the
original ``_real_stdout`` + ``_stdout_lock`` pair.  Tests that monkey-patch
``server._real_stdout`` continue to work because the stdio transport resolves
the stream lazily through a callback.
"""

from __future__ import annotations

import contextvars
import errno
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# Errno values that mean "the peer is gone" rather than "the host has a
# real I/O problem".  Anything outside this set re-raises so it surfaces
# in the crash log instead of looking like a clean disconnect.
_PEER_GONE_ERRNOS = frozenset({
    errno.EPIPE,        # write to closed pipe (POSIX)
    errno.ECONNRESET,   # peer reset the connection
    errno.EBADF,        # fd closed under us
    errno.ESHUTDOWN,    # transport endpoint shut down
    getattr(errno, "WSAECONNRESET", -1),  # win32 mapping (no-op on POSIX)
    getattr(errno, "WSAESHUTDOWN", -1),
} - {-1})

logger = logging.getLogger(__name__)


def _is_peer_gone(exc: BaseException) -> bool:
    if isinstance(exc, BrokenPipeError):
        return True
    if isinstance(exc, ValueError):
        return not isinstance(exc, UnicodeEncodeError) and "closed file" in str(exc)
    return isinstance(exc, OSError) and exc.errno in _PEER_GONE_ERRNOS


# Optional knob: when true, StdioTransport does not call ``stream.flush``
# after writing.  Use this on environments where a half-closed pipe (TUI
# Node parent quit while the gateway is still emitting events) makes
# flush block long enough to starve the rest of the worker pool.
#
# IMPORTANT: Python text stdout is fully buffered when attached to a
# pipe (the TUI case), so this knob ONLY makes sense when the gateway
# is launched with ``-u`` or ``PYTHONUNBUFFERED=1``.  Without one of
# those, JSON-RPC frames will accumulate in the buffer and the TUI
# will hang waiting for ``gateway.ready``.  Default stays off so the
# existing flush-after-write behaviour is unchanged.
_DISABLE_FLUSH = (os.environ.get("HERMES_TUI_GATEWAY_NO_FLUSH", "") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@runtime_checkable
class Transport(Protocol):
    """Minimal interface every transport implements."""

    def write(self, obj: dict) -> bool:
        """Emit one JSON frame. Return ``False`` when the peer is gone."""

    def close(self) -> None:
        """Release any resources owned by this transport."""


_current_transport: contextvars.ContextVar[Optional[Transport]] = (
    contextvars.ContextVar(
        "hermes_gateway_transport",
        default=None,
    )
)


def current_transport() -> Optional[Transport]:
    """Return the transport bound for the current request, if any."""
    return _current_transport.get()


def bind_transport(transport: Optional[Transport]):
    """Bind *transport* for the current context. Returns a token for :func:`reset_transport`."""
    return _current_transport.set(transport)


def reset_transport(token) -> None:
    """Restore the transport binding captured by :func:`bind_transport`."""
    _current_transport.reset(token)


class StdioTransport:
    """Writes JSON frames to a stream (usually ``sys.stdout``).

    The stream is resolved via a callable so runtime monkey-patches of the
    underlying stream continue to work — this preserves the behaviour the
    existing test suite relies on (``monkeypatch.setattr(server, "_real_stdout", ...)``).
    """

    __slots__ = ("_stream_getter", "_lock")

    def __init__(self, stream_getter: Callable[[], Any], lock: threading.Lock) -> None:
        self._stream_getter = stream_getter
        self._lock = lock

    def write(self, obj: dict) -> bool:
        """Return ``True`` on success, ``False`` ONLY when the peer is gone.

        Returning ``False`` is the dispatcher's "broken stdout pipe" signal
        — ``entry.py`` calls ``sys.exit(0)`` when ``write_json`` reports
        ``False``.  So programming errors (non-JSON-safe payloads, encoding
        misconfig, unexpected ValueErrors, host I/O bugs like ENOSPC) MUST
        NOT return ``False``, otherwise a real bug looks like a clean
        disconnect and is harder to diagnose.  Those re-raise so the
        existing crash-log infrastructure records the traceback.

        Peer-gone branches:
          * ``BrokenPipeError``
          * ``ValueError("...closed file...")``
          * ``OSError`` whose errno is in :data:`_PEER_GONE_ERRNOS`
            (EPIPE / ECONNRESET / EBADF / ESHUTDOWN; plus WSA mappings
            on Windows).  Other OSError errnos (ENOSPC, EACCES, ...) are
            real host problems and re-raise.
        """
        # Serialization is OUTSIDE the lock so a large payload can't
        # block other threads emitting their own frames.  A non-JSON-safe
        # payload is a programming error: re-raise so the crash log
        # captures it instead of silently exiting via the False path.
        line = json.dumps(obj, ensure_ascii=False) + "\n"

        with self._lock:
            stream = self._stream_getter()
            try:
                stream.write(line)
            except BrokenPipeError:
                return False
            except ValueError as e:
                # ValueError("I/O operation on closed file") is the
                # ONLY ValueError that means "peer gone".  Anything
                # else — including UnicodeEncodeError, which is a
                # ValueError subclass for misconfigured locales —
                # is a real bug; re-raise so it surfaces in the crash log.
                if isinstance(e, UnicodeEncodeError) or "closed file" not in str(e):
                    raise
                return False
            except OSError as e:
                if e.errno not in _PEER_GONE_ERRNOS:
                    raise
                logger.debug("StdioTransport write peer gone: %s", e)
                return False

            # A flush that *raises* with a peer-gone errno means the
            # dispatcher should exit cleanly.  A flush that *hangs* on
            # a half-closed pipe holds the lock until it returns — see
            # ``_DISABLE_FLUSH`` for the "skip flush entirely" escape
            # hatch.
            if not _DISABLE_FLUSH:
                try:
                    stream.flush()
                except BrokenPipeError:
                    return False
                except ValueError as e:
                    if isinstance(e, UnicodeEncodeError) or "closed file" not in str(e):
                        raise
                    return False
                except OSError as e:
                    if e.errno not in _PEER_GONE_ERRNOS:
                        raise
                    logger.debug("StdioTransport flush peer gone: %s", e)
                    return False

        return True

    def close(self) -> None:
        return None


_STREAMING_EVENT_TYPES = frozenset({
    "message.delta",
    "reasoning.delta",
    "thinking.delta",
})
# A full local pipe for 100ms is already far beyond the 30fps drain cadence.
_STREAM_CONTROL_PUSH_TIMEOUT_S = 0.1


class ControlQueueTimeoutError(TimeoutError):
    """The bounded control queue could not accept a canonical frame."""


class BufferedStreamWriter:
    """Keep streaming producers off a blocking transport sink."""

    def __init__(
        self,
        inner: Transport,
        *,
        queue_maxsize: int = 256,
        max_pending_deltas: int = 512,
        control_push_timeout_s: float = _STREAM_CONTROL_PUSH_TIMEOUT_S,
        coalesce_s: float = 0.033,
        close_timeout_s: float = 1.0,
    ) -> None:
        self._inner = inner
        self._queue: queue.Queue[list[dict]] = queue.Queue(maxsize=queue_maxsize)
        self._pending: deque[dict] = deque(maxlen=max_pending_deltas)
        self._pending_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._control_claimed = 0
        self._closed = False
        self._failure: BaseException | None = None
        self._control_push_timeout_s = max(0.0, control_push_timeout_s)
        self._coalesce_s = coalesce_s
        self._close_timeout_s = close_timeout_s
        self._thread = threading.Thread(
            target=self._drain,
            name="tui-stdio-writer",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _is_streaming_frame(obj: dict) -> bool:
        params = obj.get("params") if isinstance(obj, dict) else None
        return isinstance(params, dict) and params.get("type") in _STREAMING_EVENT_TYPES

    @staticmethod
    def _is_droppable_frame(obj: dict) -> bool:
        params = obj.get("params") if isinstance(obj, dict) else None
        return isinstance(params, dict) and params.get("type") == "message.delta"

    @staticmethod
    def _resync_reasoning_frames(frames: list[dict]) -> list[dict]:
        """Coalesce undelivered reasoning per session without losing chronology."""
        merged: dict[str, dict] = {}
        order: list[str] = []
        for frame in frames:
            params = frame.get("params") or {}
            session_id = str(params.get("session_id") or "")
            payload = params.get("payload") or {}
            if session_id not in merged:
                resync = dict(frame)
                resync_params = dict(params)
                resync_payload = dict(payload)
                resync_params["type"] = "reasoning.delta"
                resync_payload.update({"text": "", "resync": True})
                resync_params["payload"] = resync_payload
                resync["params"] = resync_params
                merged[session_id] = resync
                order.append(session_id)
            target = merged[session_id]["params"]["payload"]
            target["text"] += str(payload.get("text") or "")
            if payload.get("verbose"):
                target["verbose"] = True
        return [merged[session_id] for session_id in order]

    @property
    def failure(self) -> BaseException | None:
        with self._pending_lock:
            return self._failure

    def _latch_closed(self, failure: BaseException | None = None) -> None:
        with self._pending_lock:
            if failure is not None and self._failure is None:
                self._failure = failure
            if failure is not None:
                self._pending.clear()
            self._closed = True

    def _expire_control_push(self, *, claimed: bool = False) -> bool:
        timeout = ControlQueueTimeoutError(
            f"control queue remained full for {self._control_push_timeout_s:.3f}s"
        )
        with self._pending_lock:
            if claimed:
                self._control_claimed -= 1
            already_closed = self._closed
            if self._failure is None:
                self._failure = timeout
            failure = self._failure
            self._pending.clear()
            self._closed = True
        if not already_closed:
            logger.warning("stdio stream writer wedged: %s", timeout)
        if failure is not timeout:
            self.raise_if_failed()
        return False

    def raise_if_failed(self) -> None:
        failure = self.failure
        if (
            failure is not None
            and not isinstance(failure, ControlQueueTimeoutError)
            and not _is_peer_gone(failure)
        ):
            raise failure

    def write(self, obj: dict) -> bool:
        if self._closed:
            self.raise_if_failed()
            return False
        if self._is_streaming_frame(obj):
            with self._pending_lock:
                if not self._closed:
                    pending_limit = self._pending.maxlen
                    if pending_limit is None or len(self._pending) < pending_limit:
                        self._pending.append(obj)
                        return True
                    # Final-answer deltas recover from message.complete. Reasoning
                    # does not have a per-tool terminal replacement, so compact all
                    # undelivered reasoning into explicit resync frames instead of
                    # dropping it or sending the producer through control backpressure.
                    # ponytail: bounded overflow-only scan; index it only if the
                    # fixed 512-frame ceiling ever shows up in profiles.
                    for index, pending in enumerate(self._pending):
                        if self._is_droppable_frame(pending):
                            del self._pending[index]
                            self._pending.append(obj)
                            return True
                    if self._is_droppable_frame(obj):
                        return True
                    resync = self._resync_reasoning_frames([*self._pending, obj])
                    self._pending.clear()
                    self._pending.extend(resync)
                    return True
            if self._closed:
                self.raise_if_failed()
                return False

        # Serialize controls without blocking delta producers. A claimed control
        # keeps later deltas pending until that control reaches the sink.
        deadline = time.monotonic() + self._control_push_timeout_s
        if not self._control_lock.acquire(timeout=self._control_push_timeout_s):
            return self._expire_control_push()
        try:
            with self._pending_lock:
                if self._closed:
                    batch = None
                else:
                    batch = list(self._pending)
                    self._pending.clear()
                    batch.append(obj)
                    self._control_claimed += 1
            if batch is None:
                self.raise_if_failed()
                return False
            try:
                self._queue.put(batch, timeout=max(0.0, deadline - time.monotonic()))
            except queue.Full:
                return self._expire_control_push(claimed=True)
            if self._closed:
                self.raise_if_failed()
                return False
            return True
        finally:
            self._control_lock.release()

    def _write_batch(self, batch: list[dict]) -> bool:
        for obj in batch:
            try:
                if self._inner.write(obj):
                    continue
            except Exception as exc:
                if _is_peer_gone(exc):
                    logger.debug("stdio stream writer peer gone: %s", exc)
                    self._latch_closed(exc)
                    return False
                self._latch_closed(exc)
                logger.exception("stdio stream writer inner write failed")
                return False
            self._latch_closed(BrokenPipeError("inner transport closed"))
            return False
        return True

    def _drain(self) -> None:
        try:
            while True:
                try:
                    batch = self._queue.get(timeout=self._coalesce_s)
                except queue.Empty:
                    batch = None
                if batch is not None:
                    if not self._write_batch(batch):
                        return
                    with self._pending_lock:
                        self._control_claimed -= 1

                with self._pending_lock:
                    if self._control_claimed:
                        pending = []
                    else:
                        pending = list(self._pending)
                        self._pending.clear()
                    done = (
                        self._closed
                        and not self._control_claimed
                        and self._queue.empty()
                        and not pending
                    )
                if pending and not self._write_batch(pending):
                    return
                if done:
                    return
        finally:
            self._latch_closed()

    def close(self) -> None:
        self._latch_closed()
        self._thread.join(timeout=self._close_timeout_s)


class TeeTransport:
    """Mirrors writes to one primary plus N best-effort secondaries.

    The primary's return value (and exceptions) determine the result —
    secondaries swallow failures so a wedged sidecar never stalls the
    main IO path.  Used by the PTY child so every dispatcher emit lands
    on stdio (Ink) AND on a back-WS feeding the dashboard sidebar.
    """

    __slots__ = ("_primary", "_secondaries")

    def __init__(self, primary: "Transport", *secondaries: "Transport") -> None:
        self._primary = primary
        self._secondaries = secondaries

    def write(self, obj: dict) -> bool:
        # Primary first so a slow sidecar (WS publisher) never delays Ink/stdio.
        ok = self._primary.write(obj)
        for sec in self._secondaries:
            try:
                sec.write(obj)
            except Exception:
                pass
        return ok

    def close(self) -> None:
        try:
            self._primary.close()
        finally:
            for sec in self._secondaries:
                try:
                    sec.close()
                except Exception:
                    pass
