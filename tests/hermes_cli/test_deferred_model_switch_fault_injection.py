"""Stage-by-stage crash consistency for deferred model-switch apply.

A log line is not enough: after a failure injected after any apply stage,
client, model, compressor limits, overrides, UI, and the session row must
describe one restored route, and the next request must use that route.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli.model_switch import (
    ModelSwitchResult,
    apply_model_switch_after_compression,
    get_model_switch_after_compression,
    schedule_model_switch_after_compression,
)

STAGES = (
    "compress_success",
    "provider_resolve",
    "client_construct",
    "compressor_update",
    "overrides",
    "db",
    "frontend",
)


class _Client:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        payload = {"client": self.name, **kwargs}
        self.calls.append(payload)
        return payload


class _Compressor:
    def __init__(self, *, model: str, provider: str, context_length: int):
        self.model = model
        self.provider = provider
        self.base_url = "https://old.example/v1"
        self.api_key = "old-key"
        self.api_mode = "chat_completions"
        self.context_length = context_length
        self.threshold_tokens = context_length // 2

    def update_model(self, **kwargs):
        self.model = kwargs.get("model", self.model)
        self.provider = kwargs.get("provider", self.provider)
        self.base_url = kwargs.get("base_url", self.base_url)
        self.api_key = kwargs.get("api_key", self.api_key)
        self.api_mode = kwargs.get("api_mode", self.api_mode)
        if kwargs.get("context_length") is not None:
            self.context_length = kwargs["context_length"]
            self.threshold_tokens = self.context_length // 2


class _SessionDB:
    def __init__(self):
        self.row = {
            "model": "old-model",
            "model_config": json.dumps({"max_iterations": 7, "service_tier": "default"}),
            "system_prompt": "old prompt",
            "billing_provider": "old-provider",
            "billing_base_url": "https://old.example/v1",
            "billing_mode": "chat_completions",
        }

    def get_session(self, _session_id):
        return dict(self.row)

    def update_session_meta(self, _session_id, model_config, model=None):
        self.row["model_config"] = model_config
        if model is not None:
            self.row["model"] = model

    def update_system_prompt(self, _session_id, prompt):
        self.row["system_prompt"] = prompt

    def update_session_billing_route(
        self, _session_id, *, provider, base_url, billing_mode=None
    ):
        self.row.update(
            billing_provider=provider,
            billing_base_url=base_url,
            billing_mode=billing_mode,
        )


def _route(agent, db):
    stored = json.loads(db.row["model_config"]) if db.row["model_config"] else {}
    return {
        "model": agent.model,
        "provider": agent.provider,
        "client": agent.client.name,
        "compressor": (
            agent.context_compressor.model,
            agent.context_compressor.provider,
            agent.context_compressor.context_length,
        ),
        "overrides": dict(agent.request_overrides),
        "reasoning": dict(agent.reasoning_config or {}),
        "ui": dict(agent.ui_route),
        "session_model": db.row["model"],
        "session_billing": (
            db.row["billing_provider"],
            db.row["billing_base_url"],
            db.row["billing_mode"],
        ),
        "session_config_model": stored.get("model"),
        "session_config_provider": stored.get("provider"),
    }


def _issue_request(agent):
    return agent.client.complete(model=agent.model, provider=agent.provider)


def _make_agent(monkeypatch):
    from agent.agent_runtime_helpers import switch_model as live_switch_model

    old_client = _Client("old-client")
    db = _SessionDB()
    agent = SimpleNamespace(
        model="old-model",
        provider="old-provider",
        requested_provider="old-provider",
        base_url="https://old.example/v1",
        api_key="old-key",
        api_mode="chat_completions",
        client=old_client,
        _client_kwargs={"api_key": "old-key", "base_url": "https://old.example/v1"},
        _anthropic_client=None,
        _anthropic_api_key="",
        _anthropic_base_url=None,
        _is_anthropic_oauth=False,
        _config_context_length=32_000,
        _use_prompt_caching=False,
        _use_native_cache_layout=False,
        request_overrides={"service_tier": "default"},
        reasoning_config={"enabled": False},
        _primary_runtime={"model": "old-model", "provider": "old-provider"},
        _cached_system_prompt="old prompt",
        _fallback_chain=[],
        _fallback_model=None,
        _fallback_activated=False,
        _fallback_index=0,
        _transport_cache={},
        _credential_pool=None,
        _credential_pool_entry_id=None,
        _session_db=db,
        session_id="session-197",
        _session_init_model_config={"max_iterations": 7, "service_tier": "default"},
        context_compressor=_Compressor(
            model="old-model", provider="old-provider", context_length=32_000
        ),
        ui_route={"model": "old-model", "provider": "old-provider"},
        statuses=[],
    )

    def _switch(model, provider, api_key, base_url, api_mode):
        return live_switch_model(agent, model, provider, api_key, base_url, api_mode)

    agent.switch_model = _switch
    agent._read_reasoning_echo_from_config = staticmethod(lambda: False)
    agent._apply_client_headers_for_base_url = lambda *_a, **_k: None
    agent._create_openai_client = lambda *_a, **_k: _Client("new-client")
    agent._anthropic_prompt_cache_policy = lambda **_k: (False, False)
    agent._emit_status = agent.statuses.append
    agent._build_system_prompt = lambda _tools: "new prompt"

    monkeypatch.setattr(
        "hermes_cli.timeouts.get_provider_request_timeout", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        lambda *_a, **_k: None,
    )
    return agent, db, old_client


def _result():
    return ModelSwitchResult(
        success=True,
        new_model="new-model",
        target_provider="new-provider",
        api_key="new-key",
        base_url="https://new.example/v1",
        api_mode="chat_completions",
        provider_label="New Provider",
        context_length=100_000,
        reasoning_config={"enabled": True, "effort": "high"},
        is_after_compression=True,
    )


def _on_applied(agent):
    def _sync(_result, _old_model, _old_provider):
        agent.ui_route = {"model": agent.model, "provider": agent.provider}

    return _sync


@pytest.mark.parametrize("stage", STAGES)
def test_injected_failure_after_each_stage_restores_one_route(monkeypatch, stage):
    agent, db, _old_client = _make_agent(monkeypatch)
    result = _result()
    schedule_model_switch_after_compression(
        agent, result, on_applied=_on_applied(agent)
    )
    before = _route(agent, db)
    agent._deferred_model_switch_fault_after = stage

    assert apply_model_switch_after_compression(agent) == "failed"
    assert _route(agent, db) == before
    assert get_model_switch_after_compression(agent) is result

    used = _issue_request(agent)
    assert used == {
        "client": "old-client",
        "model": "old-model",
        "provider": "old-provider",
    }
    assert agent.client.name == "old-client"
    assert agent.client.calls == [used]
