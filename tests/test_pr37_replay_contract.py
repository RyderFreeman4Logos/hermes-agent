"""Replay-contract regressions for PR #37 on the carried 3340 base."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

from hermes_state import SessionDB
import tui_gateway.server as server


def _session(agent, history, history_version):
    return {
        "agent": agent,
        "session_key": "session-key",
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": history_version,
    }


def test_outer_history_race_restores_cache_attribution_flags():
    original = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]
    current = [*original, {"role": "user", "content": "typed during compression"}]
    compressor = SimpleNamespace(awaiting_real_usage_after_compression=False)
    agent = SimpleNamespace(
        _awaiting_cache_usage_after_compression=False,
        _compression_skipped_due_to_lock=False,
        context_compressor=compressor,
    )

    def compress(*_args, **_kwargs):
        compressor.awaiting_real_usage_after_compression = True
        agent._awaiting_cache_usage_after_compression = True
        return ([{"role": "user", "content": "summary"}], "")

    agent._compress_context = compress
    session = _session(agent, current, 2)

    with patch.object(server, "_get_usage", return_value={}):
        removed, _usage = server._compress_session_history(
            session,
            approx_tokens=100,
            before_messages=original,
            history_version=1,
        )

    assert removed == 0
    assert session["history"] == current
    assert compressor.awaiting_real_usage_after_compression is False
    assert agent._awaiting_cache_usage_after_compression is False


def test_compression_lock_uses_bounded_non_fts_write(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    calls = []

    def execute_write(_fn, *, patience_s=None, recover_fts_errors=True):
        calls.append((patience_s, recover_fts_errors))
        return False, None

    monkeypatch.setattr(db, "_execute_write", execute_write)
    assert db.try_acquire_compression_lock("replay-session", "holder") is False
    assert calls == [(db._COMPRESSION_LOCK_ACQUIRE_PATIENCE_S, False)]
    db.close()
