"""A legacy zero-arg ACP factory must not drop session identity.

When a stored session carries an authoritative memory-provider mode, a factory
that rejects the new keyword still has to be built for that session id and cwd.
Falling through to a bare factory() call starts a different session.
"""
from types import SimpleNamespace

from acp_adapter.session import SessionManager
from hermes_state import SessionDB


def test_legacy_factory_keeps_identity_when_mode_override_is_rejected(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    calls = []

    def legacy_factory(*_args, **kwargs):
        calls.append(dict(kwargs))
        if "memory_provider_mode_override" in kwargs and "session_id" not in kwargs:
            raise TypeError("legacy factory rejects the mode keyword alone")
        return SimpleNamespace(model="fixture")

    manager = SessionManager(db=db, agent_factory=legacy_factory)
    session_id = "legacy-session"
    db.create_session(
        session_id,
        source="acp",
        model="fixture",
        model_config={"cwd": "/workspace/kept", "memory_provider_mode": "authoritative"},
    )

    state = manager.get_session(session_id)

    assert state is not None
    assert state.session_id == session_id
    assert state.cwd == "/workspace/kept"
    assert calls == [{
        "session_id": session_id,
        "cwd": "/workspace/kept",
        "model": "fixture",
        "memory_provider_mode_override": "authoritative",
    }]
    db.close()
