"""Per-session /model overrides must survive gateway restarts (#3659 salvage).

``GatewayRunner._session_model_overrides`` is in-memory, so before persistence
a gateway restart silently reverted every session to the global default model.
The non-secret parts (model/provider/base_url) are now written through to the
session store (``SessionEntry.model_override`` in sessions.json) and lazily
rehydrated on first use after a restart, with credentials re-resolved through
the normal runtime provider resolution.

Covers:
  - the override survives a simulated restart (a second SessionStore instance
    reading the same sessions dir, and a fresh runner rehydrating from it)
  - /new (SessionStore.reset_session) clears the persisted override so a
    restart cannot resurrect it
  - api_key is NEVER serialized to sessions.json
"""
import json
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import (
    SessionEntry,
    SessionSource,
    SessionStore,
    sanitize_model_override,
)

OVERRIDE = {
    "model": "gpt-5o",
    "provider": "openai",
    "api_key": "sk-SUPER-SECRET-do-not-persist",
    "base_url": "https://api.openai.example/v1",
    "api_mode": "responses",
}


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    """Build SessionStores over a shared sessions dir, without SQLite."""

    def _raise():
        raise RuntimeError("SQLite disabled in test")

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)

    def _make() -> SessionStore:
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        assert store._db is None
        return store

    return _make


def _sessions_json(tmp_path) -> str:
    return (tmp_path / "sessions.json").read_text(encoding="utf-8")


def test_override_persists_and_survives_restart(store_factory, tmp_path):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key

    store.set_model_override(session_key, OVERRIDE)

    # Simulated restart: a brand-new store instance reads the same dir.
    store2 = store_factory()
    persisted = store2.get_model_override(session_key)
    assert persisted == {
        "model": "gpt-5o",
        "provider": "openai",
        "base_url": "https://api.openai.example/v1",
    }


def test_override_save_failure_restores_memory_and_disk(
    store_factory, tmp_path, monkeypatch
):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    store.set_model_override(entry.session_key, OVERRIDE)
    real_save = store._save
    calls = 0

    def fail_after_first_write():
        nonlocal calls
        calls += 1
        real_save()
        if calls == 1:
            raise OSError("injected post-write mirror failure")

    monkeypatch.setattr(store, "_save", fail_after_first_write)
    with pytest.raises(OSError, match="post-write mirror failure"):
        store.set_model_override(
            entry.session_key,
            {"model": "replacement", "provider": "anthropic"},
        )

    assert store.get_model_override(entry.session_key) == sanitize_model_override(OVERRIDE)
    assert store_factory().get_model_override(entry.session_key) == sanitize_model_override(
        OVERRIDE
    )


def _make_runner(store):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = store
    return runner


def test_runner_rehydrates_override_after_restart(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_model_override(session_key, OVERRIDE)

    # Simulated restart: fresh store + fresh runner with an empty in-memory
    # override map, credentials re-resolved via runtime provider resolution.
    runner = _make_runner(store_factory())
    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value={
            "api_key": "sk-fresh-from-keychain",
            "api_mode": "responses",
            "base_url": "https://api.openai.example/v1",
            "provider": "openai",
        },
    ):
        runner._rehydrate_session_model_override(session_key)

    override = runner._session_model_overrides[session_key]
    assert override["model"] == "gpt-5o"
    assert override["provider"] == "openai"
    assert override["base_url"] == "https://api.openai.example/v1"
    # Credentials come from live resolution, never from disk.
    assert override["api_key"] == "sk-fresh-from-keychain"
    assert override["api_mode"] == "responses"


def test_sanitize_model_override():
    assert sanitize_model_override(None) is None
    assert sanitize_model_override({}) is None
    assert sanitize_model_override({"api_key": "sk-x", "api_mode": "chat"}) is None
    assert sanitize_model_override(OVERRIDE) == {
        "model": "gpt-5o",
        "provider": "openai",
        "base_url": "https://api.openai.example/v1",
    }


def test_primary_db_write_failure_keeps_memory_and_legacy_mirror(tmp_path, monkeypatch):
    """state.db primary failure must not advance memory or sessions.json."""
    from hermes_state import SessionDB

    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    assert store._db is not None
    entry = store.get_or_create_session(_make_source())
    store.set_model_override(entry.session_key, OVERRIDE)

    calls = 0

    def fail_primary(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise OSError("injected primary routing write failure")

    monkeypatch.setattr(store._db, "replace_gateway_routing_entries", fail_primary)
    with pytest.raises(OSError, match="primary routing write failure"):
        store.set_model_override(
            entry.session_key,
            {"model": "replacement", "provider": "anthropic"},
        )

    assert store.get_model_override(entry.session_key) == sanitize_model_override(OVERRIDE)
    assert calls == 1
    raw = json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))
    assert raw[entry.session_key]["model_override"]["model"] == "gpt-5o"
    restarted = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    assert restarted.get_model_override(entry.session_key) == sanitize_model_override(
        OVERRIDE
    )


def test_primary_db_success_survives_legacy_mirror_failure(tmp_path, monkeypatch):
    """Legacy mirror failure after state.db success must keep the new override."""
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    entry = store.get_or_create_session(_make_source())
    store.set_model_override(entry.session_key, OVERRIDE)
    real_json = store._save_sessions_json

    def fail_mirror(data):
        real_json(data)
        raise OSError("injected legacy mirror failure")

    monkeypatch.setattr(store, "_save_sessions_json", fail_mirror)
    store.set_model_override(
        entry.session_key,
        {"model": "replacement", "provider": "anthropic"},
    )

    assert store.get_model_override(entry.session_key) == {
        "model": "replacement",
        "provider": "anthropic",
    }
    restarted = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    assert restarted.get_model_override(entry.session_key) == {
        "model": "replacement",
        "provider": "anthropic",
    }


def test_post_commit_db_error_commits_override_forward(tmp_path, monkeypatch):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    entry = store.get_or_create_session(_make_source())
    store.set_model_override(entry.session_key, OVERRIDE)
    real_replace = store._db.replace_gateway_routing_entries
    calls = 0

    def commit_then_raise(*args, **kwargs):
        nonlocal calls
        calls += 1
        real_replace(*args, **kwargs)
        raise OSError("injected post-commit primary failure")

    monkeypatch.setattr(store._db, "replace_gateway_routing_entries", commit_then_raise)
    target = {"model": "replacement", "provider": "anthropic"}

    store.set_model_override(entry.session_key, target)

    assert calls == 1
    assert store.get_model_override(entry.session_key) == target
    raw = json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))
    assert raw[entry.session_key]["model_override"] == target
    assert SessionStore(
        sessions_dir=tmp_path, config=GatewayConfig()
    ).get_model_override(entry.session_key) == target


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("committed", [True, False], ids=["target", "preimage"])
def test_authority_baseexception_converges_memory_db_and_legacy_then_rethrows(
    tmp_path,
    monkeypatch,
    interrupt_type,
    committed,
):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    store._write_sessions_json = True
    entry = store.get_or_create_session(_make_source())
    store.set_model_override(entry.session_key, OVERRIDE)
    real_replace = store._db.replace_gateway_routing_entries
    interrupt = interrupt_type("injected authority interruption")
    calls = 0

    def interrupted_replace(*args, **kwargs):
        nonlocal calls
        calls += 1
        if committed:
            real_replace(*args, **kwargs)
        raise interrupt

    monkeypatch.setattr(
        store._db,
        "replace_gateway_routing_entries",
        interrupted_replace,
    )
    target = {"model": "replacement", "provider": "anthropic"}

    with pytest.raises(interrupt_type) as caught:
        store.set_model_override(entry.session_key, target)

    assert caught.value is interrupt
    assert calls == 1
    expected = target if committed else sanitize_model_override(OVERRIDE)
    assert store.get_model_override(entry.session_key) == expected
    raw = json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))
    assert raw[entry.session_key]["model_override"] == expected
    assert SessionStore(
        sessions_dir=tmp_path,
        config=GatewayConfig(),
    ).get_model_override(entry.session_key) == expected


@pytest.mark.parametrize("readback", ["unavailable", "third-state"])
def test_indeterminate_db_write_is_explicit_and_does_not_compensate(
    tmp_path, monkeypatch, readback
):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    entry = store.get_or_create_session(_make_source())
    store.set_model_override(entry.session_key, OVERRIDE)
    real_load = store._db.load_gateway_routing_entries
    reads = 0
    writes = 0

    def fail_before_commit(*_args, **_kwargs):
        nonlocal writes
        writes += 1
        raise OSError("injected primary routing write failure")

    def uncertain_readback(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            return real_load(*args, **kwargs)
        if readback == "unavailable":
            raise OSError("injected authority readback failure")
        return {"third-party": "{}"}

    monkeypatch.setattr(store._db, "replace_gateway_routing_entries", fail_before_commit)
    monkeypatch.setattr(store._db, "load_gateway_routing_entries", uncertain_readback)

    with pytest.raises(RuntimeError, match="indeterminate"):
        store.set_model_override(
            entry.session_key,
            {"model": "replacement", "provider": "anthropic"},
        )

    assert writes == 1
    assert store.get_model_override(entry.session_key) == sanitize_model_override(OVERRIDE)
