"""Codex idle-window survival through auxiliary fallbacks.

Split from ``test_auxiliary_client`` so that over-cap file does not grow.
1200.0 is the configured value under test, not a new default.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCodexIdleWindowSurvivesFallback:
    def test_fallback_and_credential_rebuild_keep_codex_idle_window(self):
        from agent import auxiliary_client as aux

        destination = aux._FallbackDestination(
            "openai-codex", "https://chatgpt.com/backend-api/codex/", "codex_responses", "gpt-6-luna")
        with (
            patch.object(aux, "_fallback_destination", return_value=destination),
            patch.object(aux, "_fallback_chain_entry", return_value={}),
            patch.object(aux, "_fallback_entry_timeout", return_value=None),
            patch.object(aux, "_get_auxiliary_task_config", return_value={}),
            patch.object(aux, "_get_task_no_progress_timeout", return_value=1200.0),
            patch.object(aux, "_replan_synchronous_cache_sections", side_effect=lambda messages, tools, **kw: (messages, tools)),
        ):
            _, kwargs, rebuild = aux._plan_fallback_candidate(
                object(), "gpt-6-luna", "fallback_chain[0](openai-codex)", task="compression",
                effective_timeout=3600, apply_fast_lane=False, messages=[{"role": "user", "content": "summary"}],
                tools=None, temperature=None, max_tokens=None, effective_extra_body={}, reasoning_config=None)
            assert kwargs["no_progress_timeout"] == 1200.0
            assert rebuild("openai-codex", object(), "gpt-6-luna")[1]["no_progress_timeout"] == 1200.0

            non_codex = destination._replace(provider="openai", api_mode="chat_completions")
            with patch.object(aux, "_fallback_destination", return_value=non_codex):
                _, kwargs, _ = aux._plan_fallback_candidate(
                    object(), "gpt-6-luna", "fallback_chain[0](openai)", task="compression",
                    effective_timeout=3600, apply_fast_lane=False, messages=[{"role": "user", "content": "summary"}],
                    tools=None, temperature=None, max_tokens=None, effective_extra_body={}, reasoning_config=None)
            assert "no_progress_timeout" not in kwargs

    def test_timeout_quarantine_is_transient_but_auth_is_long(self):
        from agent.auxiliary_health import fallback_candidate_quarantine_ttl

        assert fallback_candidate_quarantine_ttl("request timed out") == 60.0
        assert fallback_candidate_quarantine_ttl(None) is None


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("codex", [False, True])
def test_same_provider_recovery_retains_codex_guard_only(monkeypatch, async_mode, codex):
    """Exercise recovery callers through the real rebuilt adapter, without network I/O."""
    from agent import auxiliary_client as aux

    class ReachedGuard(Exception):
        pass

    observed = {}

    def stop_at_guard(guard):
        observed.update(idle=guard.no_progress_timeout, total=guard.total_timeout)
        raise ReachedGuard()

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])
    if codex:
        raw = SimpleNamespace(api_key="test", base_url="https://chatgpt.com/backend-api/codex/",
                              close=lambda: None, responses=SimpleNamespace(create=MagicMock()))
        client = aux.CodexAuxiliaryClient(raw, "gpt-6-luna")
        if async_mode:
            client = aux.AsyncCodexAuxiliaryClient(client)
        monkeypatch.setattr(aux._CodexStreamGuard, "start", stop_at_guard)
    else:
        def chat_create(**kwargs):
            assert "no_progress_timeout" not in kwargs
            assert kwargs["timeout"] == 3600
            observed["chat_called"] = True
            return response
        create = AsyncMock(side_effect=chat_create) if async_mode else chat_create
        client = SimpleNamespace(base_url="https://api.openai.com/v1",
                                 chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (client, "gpt-6-luna"))
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {"no_progress_timeout": 1200})
    monkeypatch.setattr(aux, "_relay_auxiliary_metadata", lambda **kw: None)
    kwargs = dict(task="compression", resolved_provider="openai-codex" if codex else "openai",
                  resolved_api_mode="codex_responses" if codex else "chat_completions",
                  resolved_model="gpt-6-luna", resolved_base_url=client.base_url, resolved_api_key="test",
                  main_runtime=None, final_model="gpt-6-luna", messages=[{"role": "user", "content": "summary"}],
                  temperature=None, max_tokens=None, tools=None, effective_timeout=3600,
                  effective_extra_body={}, reasoning_config=None)

    def retry():
        return (asyncio.run(aux._retry_same_provider_async(**kwargs)) if async_mode
                else aux._retry_same_provider_sync(**kwargs))

    if codex:
        with pytest.raises(ReachedGuard):
            retry()
        assert observed == {"idle": 1200.0, "total": 3600}
        raw.responses.create.assert_not_called()
    else:
        assert retry() is response
        assert observed == {"chat_called": True}
