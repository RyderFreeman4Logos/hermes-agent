"""Codex idle-window survival through auxiliary fallbacks.

Split from ``test_auxiliary_client`` so that over-cap file does not grow.
1200.0 is the configured value under test, not a new default.
"""
from unittest.mock import patch


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
