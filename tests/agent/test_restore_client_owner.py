"""Restored production clients retain the live session, not runtime staging."""

import httpx
import pytest

from agent.context_compressor import ContextCompressor
from agent.credential_pool import CredentialPool, PooledCredential
from agent.served_model import result_model_fields
from run_agent import AIAgent


def _agent(monkeypatch, provider, *, pooled=False):
    transports = []

    def transport(*args, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"x-litellm-model-id": "fixture-deployment"})
        ))
        transports.append(client)
        return client

    monkeypatch.setattr(AIAgent, "_build_keepalive_http_client", staticmethod(transport))
    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider = "fallback", "openrouter"
    agent.base_url, agent.api_mode = "https://openrouter.ai/api/v1", "chat_completions"
    agent.api_key = "fixture-fallback"
    agent._client_kwargs = {"api_key": agent.api_key, "base_url": agent.base_url}
    agent._credential_pool, agent._credential_pool_entry_id = None, None
    agent._delegation_fixed_api_key = not pooled
    agent._fallback_activated, agent._fallback_index = True, 1
    agent._rate_limited_until, agent._restore_wait_logged = 0, False
    agent._use_prompt_caching = agent._use_native_cache_layout = False
    agent._provider_fallback_active = agent._interrupt_requested = False
    agent.quiet_mode, agent.log_prefix, agent._transport_cache = True, "fixture", {}
    agent.last_served_model = None
    agent.client = agent._create_openai_client(agent._client_kwargs, reason="fixture", shared=True)
    agent.context_compressor = ContextCompressor(
        "fallback", provider="openrouter", api_key=agent.api_key, base_url=agent.base_url,
        api_mode="chat_completions", config_context_length=64000, quiet_mode=True,
    )
    url = "moa://local" if provider == "moa" else "https://proxy.example/v1"
    model = "default" if provider == "moa" else "gpt-5.5"
    agent._primary_runtime = dict(
        model=model, provider=provider, requested_provider=provider, base_url=url,
        api_mode="chat_completions", api_key="fixture-primary",
        client_kwargs={} if provider == "moa" else {"api_key": "fixture-primary", "base_url": url},
        use_prompt_caching=False, use_native_cache_layout=False,
        compressor_model=model, compressor_base_url=url, compressor_api_key="fixture-primary",
        compressor_provider=provider, compressor_context_length=64000,
    )
    if pooled:
        agent._credential_pool = CredentialPool(provider, [PooledCredential(
            provider=provider, id="fixture-owned", label="fixture", auth_type="api_key",
            priority=0, source="manual", access_token="fixture-owned-key", base_url=url,
        )])
    return agent, transports


@pytest.mark.parametrize("pooled", [False, True], ids=["restore", "credential-rotation"])
def test_restored_response_hook_updates_live_result(monkeypatch, pooled):
    agent, transports = _agent(monkeypatch, "openai", pooled=pooled)
    try:
        assert agent._restore_primary_runtime()
        if pooled:
            assert agent.api_key == "fixture-owned-key"
            assert agent._credential_pool.current().request_count == 1
        # Public httpx API exercises the hook installed by the real OpenAI builder.
        transports[-1].get("https://fixture.invalid/headers")
        assert result_model_fields(agent) == {
            "requested_model": "gpt-5.5", "served_model": "fixture-deployment",
        }
    finally:
        for client in transports:
            client.close()


@pytest.mark.parametrize("control", ["callback", "interrupt"])
def test_restored_moa_reads_live_controls(monkeypatch, control):
    agent, transports = _agent(monkeypatch, "moa")
    old, new = [], []
    agent.tool_progress_callback = lambda *args, **kwargs: old.append(args)
    try:
        assert agent._restore_primary_runtime()
        completion = agent.client.chat.completions
        if control == "callback":
            agent.tool_progress_callback = lambda *args, **kwargs: new.append(args)
            completion._emit("moa.reference", label="fixture", text="reference")
            assert len(new) == 1 and not old
        else:
            agent._interrupt_requested = True
            assert completion._agent._interrupt_requested is True
            agent._interrupt_requested = False
            assert completion._agent._interrupt_requested is False
    finally:
        for client in transports:
            client.close()
