"""Saved routes must replace live recovery authority, not just CLI fields."""

import json

import pytest

import cli
import hermes_cli.config as cfg
import hermes_cli.runtime_provider as rp
from agent import credential_pool as cp
from run_agent import AIAgent


OLD_URL = "https://ambient.example.test/v1"
SAVED_URL = "https://saved.example.test/v1"


def _pool(provider, name, endpoint):
    return cp.CredentialPool(provider, [
        cp.PooledCredential(
            provider=provider, id=f"{name}-{i}", label=f"{name}-{i}",
            auth_type="api_key", priority=i, source="manual",
            access_token=f"synthetic-{name}-{i}", base_url=endpoint,
        ) for i in range(2)
    ])


@pytest.fixture
def route_setup(monkeypatch):
    config = {"providers": {"lab": {
        "api": SAVED_URL, "api_key": "synthetic-config-key",
    }}}
    monkeypatch.setattr(cfg, "load_config", lambda: config)
    monkeypatch.setattr(cfg, "load_config_readonly", lambda: config)
    monkeypatch.setattr(rp, "load_config", lambda: config)
    monkeypatch.setattr(cp, "_load_config_safe", lambda: config)
    # Pool storage and client construction are I/O seams; selection, alias
    # resolution, switch, entry binding, admission, rotation and swap are real.
    monkeypatch.setattr(cp.CredentialPool, "_persist", lambda *a, **k: None)
    monkeypatch.setattr(AIAgent, "_create_openai_client", lambda *a, **k: object())
    monkeypatch.setattr(AIAgent, "_replace_primary_openai_client", lambda *a, **k: None)
    monkeypatch.setattr(AIAgent, "_apply_client_headers_for_base_url", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.timeouts.get_provider_request_timeout", lambda *a, **k: None)

    def make(identity, replacement, live=True):
        provider = "custom:lab" if identity != "provider" else "custom:old"
        model = "saved-model" if identity == "same" else "ambient-model"
        old_pool = _pool(provider, "old", OLD_URL)
        resolved_pool = _pool("custom:lab", "saved", SAVED_URL) if replacement else None
        # An empty storage pool means actual resolution falls back to config.
        storage_pool = resolved_pool or cp.CredentialPool("custom:lab", [])
        monkeypatch.setattr(rp, "load_pool", lambda *a, **k: storage_pool)
        # A second generic reload must NOT replace the resolved pool/None.
        monkeypatch.setattr(cp, "load_pool", lambda *a, **k: old_pool)
        agent = AIAgent.__new__(AIAgent)
        agent.__dict__.update(
            model=model, provider=provider, requested_provider=provider,
            api_key="synthetic-old-0", api_mode="chat_completions",
            client=object(), _client_kwargs={"api_key": "synthetic-old-0", "base_url": OLD_URL},
            _credential_pool=old_pool, _credential_pool_entry_id="old-0",
            _anthropic_client=None, _anthropic_api_key="", _anthropic_base_url=None,
            _is_anthropic_oauth=False, _config_context_length=None,
            _primary_runtime={}, _cached_system_prompt=None, context_compressor=None,
            _fallback_chain=[], _fallback_model=None, _fallback_index=0,
            _fallback_activated=False, _session_db=None, _transport_cache={},
        )
        agent.base_url = OLD_URL  # Use the real setter and its derived fields.
        obj = cli.HermesCLI.__new__(cli.HermesCLI)
        obj.__dict__.update(
            model=model, provider=provider, requested_provider=provider,
            base_url=OLD_URL, api_key=agent.api_key, api_mode=agent.api_mode,
            agent=agent if live else None, _credential_pool=old_pool,
            _explicit_model_override=False, _console_print=lambda *a, **k: None,
        )
        row = {"model": "saved-model", "model_config": json.dumps({
            "gateway_runtime": {"provider": "custom:lab", "api_mode": "chat_completions"},
        })}
        return obj, agent, old_pool, resolved_pool, row

    return make


@pytest.mark.parametrize("identity,replacement,live", [
    ("same", False, True), ("same", True, True),
    ("model", False, True), ("model", True, True),
    ("provider", False, True), ("provider", True, True),
    ("same", False, False), ("provider", True, False),
])
def test_resume_hands_off_pool_before_real_recovery(route_setup, identity, replacement, live):
    obj, agent, old_pool, resolved_pool, row = route_setup(identity, replacement, live)
    assert agent.base_url == OLD_URL
    obj._restore_session_model(row)
    assert obj._credential_pool is resolved_pool
    assert obj.base_url == SAVED_URL
    if not live:
        assert obj.agent is None
        return

    assert agent.base_url == SAVED_URL
    assert agent.api_key == obj.api_key
    binding_before_recovery = agent._credential_pool_entry_id
    # Check recovery before checking pool identity: RED must witness the real
    # old-endpoint rotation, not just an attribute mismatch.
    recovered, _ = agent._recover_with_credential_pool(status_code=402, has_retried_429=False)
    assert agent.base_url == SAVED_URL, "recovery regained ambient endpoint authority"
    assert agent._credential_pool is resolved_pool
    assert binding_before_recovery == ("saved-0" if replacement else None)
    assert recovered is replacement
    assert all(entry.last_status is None for entry in old_pool._entries)
    assert old_pool._unmatched_rotation_streak == 0
    if replacement:
        assert cp.credential_pool_matches_provider(resolved_pool, agent.provider, base_url=agent.base_url)
        assert agent._credential_pool_entry_id == "saved-1"
        assert agent.api_key == "synthetic-saved-1"
        assert resolved_pool._entries[0].last_status == "exhausted"
    else:
        assert agent._credential_pool_entry_id is None
        assert agent.api_key == "synthetic-config-key"


@pytest.mark.parametrize("replacement", [False, True])
def test_resume_failed_client_build_restores_old_binding_and_detaches(route_setup, monkeypatch, replacement):
    obj, agent, old_pool, resolved_pool, row = route_setup("same", replacement)
    old_client = agent.client
    old_kwargs = dict(agent._client_kwargs)
    seen = []

    def fail_build(*a, **k):
        seen.append(agent._credential_pool is resolved_pool)
        raise RuntimeError("synthetic client failure")

    monkeypatch.setattr(agent, "_create_openai_client", fail_build)
    with pytest.raises(ValueError, match="Cannot restore saved model route"):
        obj._restore_session_model(row)
    assert seen == [True]
    assert obj.agent is None
    assert obj._credential_pool is resolved_pool
    assert obj.base_url == SAVED_URL
    assert agent.base_url == OLD_URL
    assert agent.api_key == "synthetic-old-0"
    assert agent._credential_pool is old_pool
    assert agent._credential_pool_entry_id == "old-0"
    assert agent.client is old_client
    assert agent._client_kwargs == old_kwargs
