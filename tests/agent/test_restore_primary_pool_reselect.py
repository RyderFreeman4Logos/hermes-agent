# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""Test that _restore_primary_runtime re-selects from the credential pool
instead of using a stale snapshot key.

Bug: when a credential pool entry is revoked/marked-exhausted during a turn,
_restore_primary_runtime restores the original (now-stale) api_key from the
construction-time snapshot. The next turn immediately hits the same error,
exhausting remaining entries and falling through to cross-provider fallback.
"""

import time
from unittest.mock import MagicMock


from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    PooledCredential,
)


def _make_entry(
    label: str,
    access_token: str,
    *,
    source: str = "device_code",
    priority: int = 0,
    last_status: str | None = None,
    last_status_at: float | None = None,
) -> dict:
    return {
        "id": label,
        "label": label,
        "provider": "openai-codex",
        "auth_type": AUTH_TYPE_OAUTH,
        "source": source,
        "priority": priority,
        "access_token": access_token,
        "refresh_token": f"rt-{label}",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "last_status": last_status,
        "last_status_at": last_status_at,
    }


def _build_mock_pool(entries: list[dict], *, strategy: str = "round_robin"):
    """Build a mock CredentialPool with the given entries."""
    from agent.credential_pool import CredentialPool

    pool = CredentialPool(
        provider="openai-codex",
        entries=[PooledCredential.from_dict("openai-codex", e) for e in entries],
    )
    pool._strategy = strategy
    return pool


class TestRestorePrimaryPoolReselect:
    """_restore_primary_runtime should re-select from the credential pool."""

    def _make_agent(self, pool):
        """Create a minimal AIAgent with the given credential pool."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.model = "gpt-5.5"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent.api_mode = "codex_responses"
        agent.api_key = "original-key-entry-1"
        agent._client_kwargs = {
            "api_key": "original-key-entry-1",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }
        agent._credential_pool = pool
        agent._fallback_activated = True
        agent._fallback_index = 1
        agent._rate_limited_until = 0
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False
        agent.context_compressor = MagicMock()
        agent.context_compressor.update_model = MagicMock()

        # Snapshot the original state
        agent._primary_runtime = {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "api_key": "original-key-entry-1",
            "client_kwargs": {
                "api_key": "original-key-entry-1",
                "base_url": "https://chatgpt.com/backend-api/codex",
            },
            "use_prompt_caching": False,
            "use_native_cache_layout": False,
            "compressor_model": "gpt-5.5",
            "compressor_base_url": "https://chatgpt.com/backend-api/codex",
            "compressor_api_key": "original-key-entry-1",
            "compressor_provider": "openai-codex",
            "compressor_context_length": 128000,
            "compressor_threshold_tokens": 0.8,
        }

        # Mock client creation methods
        agent._create_openai_client = MagicMock(return_value=MagicMock())
        agent._apply_client_headers_for_base_url = MagicMock()
        agent._replace_primary_openai_client = MagicMock(return_value=True)

        return agent


    def test_restore_uses_freshest_available_entry(self):
        """When multiple entries are available, restore should select the pool's best pick."""
        entries = [
            _make_entry("entry-1", "key-1", priority=0,
                         last_status="exhausted", last_status_at=time.time() + 3600),
            _make_entry("entry-2", "key-2", priority=1),
            _make_entry("entry-3", "key-3", priority=2),
        ]
        pool = _build_mock_pool(entries)

        agent = self._make_agent(pool)
        result = agent._restore_primary_runtime()

        assert result is True
        # entry-1 is exhausted, so pool should select entry-2
        assert agent.api_key == "key-2"
        assert agent._client_kwargs["api_key"] == "key-2"





    def test_restore_updates_base_url_from_pool_entry(self):
        """If pool entry has a different base_url, restore should update it."""
        entries = [
            {
                **_make_entry("entry-1", "key-1", priority=0),
                "base_url": "https://custom-endpoint.example.com/v1",
            },
        ]
        pool = _build_mock_pool(entries)

        agent = self._make_agent(pool)
        result = agent._restore_primary_runtime()

        assert result is True
        assert "custom-endpoint.example.com" in agent.base_url
        assert "custom-endpoint.example.com" in agent._client_kwargs["base_url"]


_CODEX_URL = "https://chatgpt.com/backend-api/codex"
_STALE = "fixture-stale-snapshot"
_FRESH = "fixture-refreshed-approved"


def _codex_entry(entry_id, token, *, status="ok", reset_at=None, refresh="fixture-refresh"):
    return PooledCredential.from_dict(
        "openai-codex",
        {
            "id": entry_id,
            "label": entry_id,
            "provider": "openai-codex",
            "auth_type": AUTH_TYPE_OAUTH,
            "source": "device_code",
            "priority": 0,
            "access_token": token,
            "refresh_token": refresh,
            "base_url": _CODEX_URL,
            "last_status": status,
            "last_status_at": time.time() if status == "exhausted" else None,
            "last_error_code": 429 if status == "exhausted" else None,
            "last_error_reason": "usage_limit_reached" if status == "exhausted" else None,
            "last_error_reset_at": reset_at,
        },
    )


def _exhausted_codex_agent(pool, *, snapshot_token=_STALE):
    """Public restore caller: real pool, snapshot client, no live auth/quota/model I/O."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.model = "gpt-5.5"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_mode = "chat_completions"
    agent.api_key = "fallback-key"
    agent._client_kwargs = {"api_key": "fallback-key", "base_url": agent.base_url}
    agent._credential_pool = pool
    agent._credential_pool_entry_id = None
    agent._delegation_fixed_api_key = False
    agent._fallback_activated = True
    agent._fallback_index = 1
    agent._rate_limited_until = 0
    agent._restore_wait_logged = False
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._provider_fallback_active = False
    agent.context_compressor = MagicMock()
    agent._primary_runtime = {
        "model": "gpt-5.5",
        "provider": "openai-codex",
        "requested_provider": "openai-codex",
        "base_url": _CODEX_URL,
        "api_mode": "codex_responses",
        "api_key": snapshot_token,
        "client_kwargs": {"api_key": snapshot_token, "base_url": _CODEX_URL},
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "compressor_model": "gpt-5.5",
        "compressor_base_url": _CODEX_URL,
        "compressor_api_key": snapshot_token,
        "compressor_provider": "openai-codex",
        "compressor_context_length": 128000,
    }
    built = []

    def _build(kwargs, *, reason, shared):
        client = MagicMock()
        client.api_key = kwargs["api_key"]
        built.append(kwargs["api_key"])
        agent.client = client
        return client

    agent._create_openai_client = _build
    agent._built_client_tokens = built
    from agent.client_lifecycle import ClientLifecycleMixin
    agent._swap_credential = ClientLifecycleMixin._swap_credential.__get__(agent)
    agent._replace_primary_openai_client = lambda **k: (_build(agent._client_kwargs, reason="credential_rotation", shared=True) or True)
    agent._apply_client_headers_for_base_url = lambda *a, **k: None
    agent._reapply_route_client_config = lambda *a, **k: None
    return agent


def _patch_probe(monkeypatch, *, fresh=None, open_=True):
    """Stub only the auth-refresh and quota-transport boundaries. Tokens are fixtures."""
    calls = {"refresh": 0, "probe": 0}

    def _pure(token, refresh_token, **_kwargs):
        calls["refresh"] += 1
        if fresh is None:
            return None
        return {"access_token": fresh[0], "refresh_token": fresh[1], "last_refresh": "fixture"}

    def _probe(token, *, base_url=None):
        calls["probe"] += 1
        calls["probed_token"] = token
        return open_

    monkeypatch.setattr("hermes_cli.auth_codex.refresh_codex_oauth_pure", _pure)
    monkeypatch.setattr("hermes_cli.auth._probe_codex_quota_restored", _probe)
    monkeypatch.setattr("hermes_cli.auth_codex._codex_access_token_is_expiring", lambda token, skew: fresh is not None)
    monkeypatch.setattr(
        "agent.credential_pool.CredentialPool._sync_device_code_entry_to_auth_store",
        lambda self, entry: None,
    )
    monkeypatch.setattr("agent.credential_pool.CredentialPool._persist", lambda self, **k: None)
    monkeypatch.setattr("agent.credential_pool.CredentialPool._sync_entry_from_auth_store", lambda self, entry: entry)
    monkeypatch.setattr("agent.credential_pool.CredentialPool._resync_stale_entry", lambda self, entry: entry)
    return calls


def test_refresh_before_probe_restores_the_approved_entry(monkeypatch):
    """A positive probe that rotated the token must restore that entry, not the snapshot."""
    from agent.credential_pool import CredentialPool

    future = time.time() + 6 * 86400
    entry = _codex_entry("owned-1", "fixture-expired", status="exhausted", reset_at=future)
    pool = CredentialPool("openai-codex", [entry])
    pool._current_id = entry.id
    calls = _patch_probe(monkeypatch, fresh=(_FRESH, "fixture-refresh-2"), open_=True)
    agent = _exhausted_codex_agent(pool)

    assert agent._restore_primary_runtime() is True
    assert calls["refresh"] == 1
    assert calls["probe"] == 1
    assert calls["probed_token"] == _FRESH
    live = pool.current()
    assert live is not None and live.id == "owned-1"
    assert live.access_token == _FRESH
    assert live.last_status == "ok"
    assert agent.api_key == _FRESH
    assert agent._client_kwargs["api_key"] == _FRESH
    assert agent.client.api_key == _FRESH
    assert agent._credential_pool_entry_id == "owned-1"
    assert agent._built_client_tokens[-1] == _FRESH


def test_ordinary_post_429_reopen_adopts_the_approved_entry(monkeypatch):
    """mark_exhausted_and_rotate clears current; the next restore still adopts the probed entry."""
    from agent.credential_pool import CredentialPool

    entry = _codex_entry("owned-1", "fixture-live", status="ok")
    pool = CredentialPool("openai-codex", [entry])
    pool._current_id = entry.id
    rotated = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "usage_limit_reached", "message": "quota", "reset_at": time.time() + 6 * 86400},
        credential_id=entry.id,
        api_key_hint="fixture-live",
    )
    assert rotated is None
    assert pool.current() is None
    calls = _patch_probe(monkeypatch, fresh=(_FRESH, "fixture-refresh-2"), open_=True)
    agent = _exhausted_codex_agent(pool)

    assert agent._restore_primary_runtime() is True
    assert calls["probe"] == 1
    assert calls["probed_token"] == _FRESH
    assert agent.api_key == _FRESH
    assert agent.client.api_key == _FRESH
    assert agent._credential_pool_entry_id == "owned-1"
    assert pool.current().id == "owned-1"
    assert pool.current().last_status == "ok"


def test_closed_probe_after_refresh_does_not_restore(monkeypatch):
    from agent.credential_pool import CredentialPool

    future = time.time() + 6 * 86400
    entry = _codex_entry("owned-1", "fixture-expired", status="exhausted", reset_at=future)
    pool = CredentialPool("openai-codex", [entry])
    pool._current_id = entry.id
    calls = _patch_probe(monkeypatch, fresh=(_FRESH, "fixture-refresh-2"), open_=False)
    agent = _exhausted_codex_agent(pool)

    assert agent._restore_primary_runtime() is False
    assert agent._fallback_activated is True
    assert agent.provider == "openrouter"
    assert calls["probe"] == 1
    assert agent._credential_pool_entry_id is None
    assert pool.current().last_status == "exhausted"


def test_wrong_provider_pool_is_not_probed(monkeypatch):
    from agent.credential_pool import CredentialPool

    entry = _codex_entry("owned-1", "fixture-expired", status="exhausted", reset_at=time.time() + 86400)
    pool = CredentialPool("openrouter", [entry])
    calls = _patch_probe(monkeypatch, fresh=(_FRESH, "fixture-refresh-2"), open_=True)
    agent = _exhausted_codex_agent(pool)

    assert agent._restore_primary_runtime() is True
    assert calls["probe"] == 0
    assert calls["refresh"] == 0
    assert agent._credential_pool_entry_id is None


def test_delegated_fixed_key_does_not_adopt_pool(monkeypatch):
    from agent.credential_pool import CredentialPool

    entry = _codex_entry("owned-1", "fixture-expired", status="exhausted", reset_at=time.time() + 86400)
    pool = CredentialPool("openai-codex", [entry])
    pool._current_id = entry.id
    calls = _patch_probe(monkeypatch, fresh=(_FRESH, "fixture-refresh-2"), open_=True)
    agent = _exhausted_codex_agent(pool)
    agent._delegation_fixed_api_key = True

    assert agent._restore_primary_runtime() is True
    assert calls["probe"] == 0
    assert agent.api_key == _STALE
    assert agent._credential_pool is None
    assert agent._credential_pool_entry_id is None


def test_prefetched_pool_is_loaded_once_then_adopts_refreshed_entry(monkeypatch):
    """A cross-provider attached pool is not the primary pool: load it once, then adopt."""
    from agent.credential_pool import CredentialPool, load_pool

    future = time.time() + 6 * 86400
    entry = _codex_entry("owned-1", "fixture-expired", status="exhausted", reset_at=future)
    primary = CredentialPool("openai-codex", [entry])
    attached = CredentialPool("openrouter", [_codex_entry("other", "fixture-other")])
    calls = _patch_probe(monkeypatch, fresh=(_FRESH, "fixture-refresh-2"), open_=True)
    loads = {"n": 0}

    def _load(key):
        loads["n"] += 1
        return primary

    monkeypatch.setattr("agent.credential_pool.load_pool", _load)
    agent = _exhausted_codex_agent(attached, snapshot_token=_STALE)

    assert agent._restore_primary_runtime() is True
    assert loads["n"] == 1
    assert calls["probe"] == 1
    assert agent.api_key == _FRESH
    assert agent.client.api_key == _FRESH
    assert agent._credential_pool_entry_id == "owned-1"
    assert agent._credential_pool is primary
    assert load_pool is not None
