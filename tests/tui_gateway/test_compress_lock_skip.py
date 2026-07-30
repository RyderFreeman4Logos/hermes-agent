"""Tests for TUI gateway /compress lock-hold signalling.

Covers the shared _compress_session_history choke point plus ALL THREE
in-process manual-compress consumers (session.compress RPC, the
command.dispatch compress branch, and the slash.exec mirror via
_mirror_slash_side_effects) — each must surface the lock-skip reason
instead of the misleading "No changes from compression" no-op text or a
generic "compress failed" error.
"""
import threading
from unittest.mock import MagicMock, patch

import pytest


def _make_history():
    return [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]


def _make_lock_skip_agent(signal):
    """Agent double whose _compress_context no-ops and sets the lock signal."""
    agent = MagicMock()
    agent._cached_system_prompt = ""
    agent.tools = None
    agent.session_id = "sess-lock"
    # Deferred-notify contract: no pending notification exists.
    agent._pending_context_engine_compression_notification = None

    def _fake_compress(msgs=None, *_args, **_kwargs):
        agent._compression_skipped_due_to_lock = signal
        return (list(msgs) if msgs is not None else _make_history(), "")

    agent._compress_context.side_effect = _fake_compress
    return agent


def _make_session(agent, history):
    return {
        "agent": agent,
        "history_lock": threading.Lock(),
        "history": history,
        "history_version": 1,
        "running": False,
        "session_key": "sess-lock",
    }


class _TurnWorker:
    def __init__(self, alive: bool):
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def test_compress_session_history_raises_on_lock_skip():
    """When _compression_skipped_due_to_lock is set on the agent,
    _compress_session_history must raise CompressionLockHeld with
    the holder string so callers can surface a clear message."""
    from tui_gateway.server import _compress_session_history, CompressionLockHeld

    history = _make_history()
    agent = _make_lock_skip_agent("pid=99999:tid=1:agent=1:nonce=abc")
    session = _make_session(agent, history)

    with (
        patch(
            "agent.model_metadata.estimate_request_tokens_rough", return_value=100
        ),
        pytest.raises(CompressionLockHeld) as exc_info,
    ):
        _compress_session_history(session)

    assert exc_info.value.holder == "pid=99999:tid=1:agent=1:nonce=abc"
    # The history must be untouched by the lock-skip.
    assert session["history"] == history
    assert session["history_version"] == 1


# ── Consumer 1: session.compress RPC ───────────────────────────────────


def test_session_compress_rpc_returns_lock_held_payload():
    from tui_gateway import server

    agent = _make_lock_skip_agent("pid=4242")
    session = _make_session(agent, _make_history())
    sid = "sid-lock-rpc"
    server._sessions[sid] = session
    try:
        with (
            patch.object(server, "_sess", return_value=(session, None)),
            patch.object(server, "_session_uses_compute_host", return_value=False),
            patch.object(server, "_status_update"),
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=100,
            ),
        ):
            r = server._methods["session.compress"]("r1", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    result = r["result"]
    assert result["lock_held"] is True
    assert result["compressed"] is False
    assert "Compression already in progress" in result["message"]
    assert "pid=4242" in result["message"]
    assert "No changes from compression" not in result["message"]


def test_stale_running_marker_allows_session_compress():
    """A dead worker must not leave manual compression permanently bricked."""
    from tui_gateway import server

    agent = _make_lock_skip_agent("pid=stale")
    session = _make_session(agent, _make_history())
    session["running"] = True
    session["_run_thread"] = _TurnWorker(alive=False)
    sid = "sid-stale-compress"
    server._sessions[sid] = session
    try:
        with (
            patch.object(server, "_sess", return_value=(session, None)),
            patch.object(server, "_session_uses_compute_host", return_value=False),
            patch.object(server, "_status_update"),
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=100,
            ),
        ):
            response = server._methods["session.compress"]("stale", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert response["result"]["lock_held"] is True
    assert session["running"] is False
    assert session["inflight_turn"] is None


def test_live_running_turn_still_blocks_compress_with_real_interrupt_help():
    from tui_gateway import server

    session = _make_session(_make_lock_skip_agent("pid=live"), _make_history())
    session["running"] = True
    session["_run_thread"] = _TurnWorker(alive=True)
    sid = "sid-live-compress"
    server._sessions[sid] = session
    try:
        with (
            patch.object(server, "_sess", return_value=(session, None)),
            patch.object(server, "_session_uses_compute_host", return_value=False),
        ):
            response = server._methods["session.compress"]("live", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert response["error"]["code"] == 4009
    assert response["error"]["message"] == (
        "session busy — press Escape or Ctrl+C, or run /interrupt before /compress"
    )
    assert session["running"] is True


@pytest.mark.parametrize("route", ["command.dispatch", "mirror"])
def test_stale_running_marker_allows_every_tui_compress_route(route):
    """The slash worker and live mirror must not reintroduce the stale block."""
    from tui_gateway import server

    agent = _make_lock_skip_agent(f"pid={route}")
    session = _make_session(agent, _make_history())
    session["running"] = True
    session["_run_thread"] = _TurnWorker(alive=False)
    sid = f"sid-stale-{route}"
    server._sessions[sid] = session
    try:
        with (
            patch.object(server, "_session_uses_compute_host", return_value=False),
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=100,
            ),
        ):
            if route == "command.dispatch":
                response = server._methods["command.dispatch"](
                    "stale-dispatch",
                    {"name": "compress", "arg": "", "session_id": sid},
                )
                assert response["result"]["type"] == "exec"
            else:
                response = server._mirror_slash_side_effects(sid, session, "/compress")
                assert "Compression already in progress" in response
    finally:
        server._sessions.pop(sid, None)

    assert session["running"] is False


# ── Consumer 2: command.dispatch compress branch ───────────────────────


def test_command_dispatch_compress_reports_lock_skip_as_output_not_error():
    """CompressionLockHeld must NOT fall through to the generic
    'compress failed' error handler — it's a clean no-op with feedback."""
    from tui_gateway import server

    agent = _make_lock_skip_agent("pid=5150")
    session = _make_session(agent, _make_history())
    sid = "sid-lock-dispatch"
    server._sessions[sid] = session
    try:
        with (
            patch.object(server, "_session_uses_compute_host", return_value=False),
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=100,
            ),
        ):
            r = server._methods["command.dispatch"](
                "r2", {"name": "compress", "arg": "", "session_id": sid}
            )
    finally:
        server._sessions.pop(sid, None)

    assert "error" not in r, f"lock-skip surfaced as error: {r.get('error')}"
    output = r["result"]["output"]
    assert r["result"]["type"] == "exec"
    assert "Compression already in progress" in output
    assert "pid=5150" in output
    assert "compress failed" not in output
    assert "No changes from compression" not in output


# ── Consumer 3: slash.exec mirror (_mirror_slash_side_effects) ─────────


def test_mirror_slash_side_effects_reports_lock_skip():
    from tui_gateway import server

    agent = _make_lock_skip_agent("pid=6161")
    session = _make_session(agent, _make_history())
    sid = "sid-lock-mirror"
    server._sessions[sid] = session
    try:
        with (
            patch.object(server, "_sync_session_key_after_compress"),
            patch.object(server, "_emit"),
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=100,
            ),
        ):
            output = server._mirror_slash_side_effects(sid, session, "/compress")
    finally:
        server._sessions.pop(sid, None)

    assert "Compression already in progress" in output
    assert "pid=6161" in output
    assert "No changes from compression" not in output
    assert "live session sync failed" not in output


