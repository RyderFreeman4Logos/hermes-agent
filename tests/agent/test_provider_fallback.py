"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import chat_completion_helpers
from agent.error_classifier import FailoverReason
from run_agent import AIAgent, _pool_may_recover_from_rate_limit


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None



    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"


    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (FailoverReason.auth, "authentication failed"),
        (FailoverReason.billing, "billing or quota exhausted"),
        (FailoverReason.rate_limit, "rate limit"),
        (FailoverReason.upstream_rate_limit, "upstream model rate limit"),
        (FailoverReason.overloaded, "provider overloaded"),
        (FailoverReason.server_error, "provider server error"),
        (FailoverReason.timeout, "request timeout"),
        (FailoverReason.model_not_found, "model not found"),
        (FailoverReason.unknown, "provider failure"),
    ],
)
def test_fallback_reason_text_is_operator_friendly(reason, expected):
    assert chat_completion_helpers._fallback_reason_text(reason) == expected


def test_fallback_reason_text_defaults_when_reason_is_missing():
    assert chat_completion_helpers._fallback_reason_text(None) == "provider failure"


class TestFallbackChainAdvancement:
    def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"
            assert agent._fallback_activated is True

    @patch("time.monotonic", return_value=1000.0)
    def test_records_user_visible_switch_with_reason(self, _clock):
        agent = _make_agent(
            fallback_model={"provider": "zai", "model": "glm-5.2"},
        )
        agent.model = "gpt-5.6-sol"
        agent.provider = "openai-codex"
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(base_url="https://api.z.ai/v1"), "glm-5.2"),
        ):
            assert agent._try_activate_fallback(FailoverReason.rate_limit) is True

        expected = (
            "⚠️ Model fallback: gpt-5.6-sol via openai-codex unavailable "
            "(rate limit); using glm-5.2 via zai. "
            "Primary retry eligible in ~60 s; recovery is not guaranteed."
        )
        assert agent._pending_fallback_notice == [expected]
        assert agent._retry_status_buffer[-1] == ("status", expected)

    @patch("time.monotonic", return_value=1000.0)
    def test_records_sequential_switches_in_order(self, _clock):
        agent = _make_agent(
            fallback_model=[
                {"provider": "zai", "model": "glm-5.2"},
                {"provider": "deepseek", "model": "deepseek-v4-flash"},
            ],
        )
        agent.model = "gpt-5.6-sol"
        agent.provider = "openai-codex"
        clients = [
            _mock_client(base_url="https://api.z.ai/v1"),
            _mock_client(base_url="https://api.deepseek.com/v1"),
        ]
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[(clients[0], "glm-5.2"), (clients[1], "deepseek-v4-flash")],
        ):
            assert agent._try_activate_fallback(FailoverReason.rate_limit) is True
            assert agent._try_activate_fallback(FailoverReason.overloaded) is True

        assert agent._pending_fallback_notice == [
            "⚠️ Model fallback: gpt-5.6-sol via openai-codex unavailable "
            "(rate limit); using glm-5.2 via zai. "
            "Primary retry eligible in ~60 s; recovery is not guaranteed.",
            "⚠️ Model fallback: glm-5.2 via zai unavailable "
            "(provider overloaded); using deepseek-v4-flash via deepseek.",
        ]
    def test_skips_unconfigured_provider_to_next(self):
        """If resolve_provider_client returns None, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                (None, None),                    # broken provider
                (_mock_client(), "gpt-4o"),       # fallback succeeds
            ]
            assert agent._try_activate_fallback(FailoverReason.rate_limit) is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2
            assert agent._rate_limit_backoff_count == 1

    def test_skips_provider_that_raises_to_next(self):
        """If resolve_provider_client raises, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                RuntimeError("auth failed"),
                (_mock_client(), "gpt-4o"),
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"

    def test_resolves_key_env_for_fallback_provider(self):
        fbs = [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.example/v1",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(
                        base_url="https://fallback.example/v1",
                        api_key="env-secret",
                    ),
                    "fallback-model",
                ),
            ) as mock_rpc,
        ):
            assert agent._try_activate_fallback() is True
            assert mock_rpc.call_args.kwargs["explicit_api_key"] == "env-secret"


    def test_nous_anthropic_fallback_uses_the_messages_wire(self, monkeypatch):
        """Portal Claude fallbacks must not stay on chat_completions when the native wire is selected.

        ``resolve_provider_client`` still returns an OpenAI client for Nous;
        activation has to re-derive api_mode from the model and rebuild the
        Anthropic client — otherwise the turn POSTs /chat/completions. The wire
        is opt-in since 2026-09-06 (``nous.anthropic_wire``, see ``nous_api_mode``).
        """
        from hermes_cli import providers as _providers
        monkeypatch.setattr(_providers, "_nous_anthropic_wire", lambda: "native")
        portal = "https://inference-api.nousresearch.com/v1"
        fbs = [
            {
                "provider": "nous",
                "model": "anthropic/claude-opus-4.8",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        rebuilt = {"count": 0}

        def _fake_build(api_key, base_url, timeout=None, **kwargs):
            rebuilt["count"] += 1
            rebuilt["api_key"] = api_key
            rebuilt["base_url"] = base_url
            return MagicMock(name="anthropic-client")

        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url=portal, api_key="portal-jwt"),
                    "anthropic/claude-opus-4.8",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=_fake_build,
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "anthropic_messages"
        assert agent.provider == "nous"
        assert agent.model == "anthropic/claude-opus-4.8"
        assert agent.client is None
        assert rebuilt["count"] == 1
        assert rebuilt["api_key"] == "portal-jwt"
        assert rebuilt["base_url"] == portal
        assert agent._anthropic_client is not None

    def test_nous_non_anthropic_fallback_stays_on_chat_completions(self):
        portal = "https://inference-api.nousresearch.com/v1"
        fbs = [{"provider": "nous", "model": "hermes-4-405b"}]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url=portal, api_key="portal-jwt"),
                    "hermes-4-405b",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=AssertionError("must not build Anthropic client"),
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "chat_completions"
        assert agent.client is not None


# ── Pool-rotation vs fallback gating (#11314) ────────────────────────────


def _pool(n_entries: int, has_available: bool = True):
    """Make a minimal credential-pool stand-in for rotation-room checks."""
    pool = MagicMock()
    pool.entries.return_value = [MagicMock() for _ in range(n_entries)]
    pool.has_available.return_value = has_available
    return pool


class TestPoolRotationRoom:
    def test_none_pool_returns_false(self):
        assert _pool_may_recover_from_rate_limit(None) is False







# ── Skip-self dedup (#22548) ───────────────────────────────────────────────


class TestFallbackChainDedup:
    """A fallback chain entry that resolves to the current provider/model
    (or the same custom-provider base_url) must be skipped, not retried.
    Otherwise a misconfigured chain or two custom_providers entries pointing
    at the same shim loop the same failure. See issue #22548."""

    def test_skips_entry_matching_current_provider_and_model(self):
        """Chain has [same-as-current, real-fallback]; activate must skip
        the first and use the second."""
        fbs = [
            # First entry == current state. Should be skipped.
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
            # Second entry: real fallback.
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        # Stub out resolve_provider_client so we can assert which entry was
        # actually used — return a MagicMock client tagged with the provider.
        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # The first entry was skipped — only the second reached resolve.
        assert called == [("zai", "glm-4.7")], (
            f"expected fallback to skip same-state entry, got call order: {called}"
        )


    def test_returns_false_when_only_self_matching_entries(self):
        """A chain with only self-matching entries exhausts to False."""
        fbs = [
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        mock_resolve.assert_not_called()

    def test_allows_xai_api_fallback_from_xai_oauth_same_host_model(self):
        """xai-oauth and xai share api.x.ai but use different credentials.

        A spending-limit 403 on OAuth must still be able to fall over to the
        API-key provider even when both entries use the same model slug and
        base URL.  Blind base_url+model dedup incorrectly skipped that path.
        """
        fbs = [
            {
                "provider": "xai",
                "model": "grok-4.5",
                "base_url": "https://api.x.ai/v1",
            },
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "xai-oauth"
        agent.model = "grok-4.5"
        agent.base_url = "https://api.x.ai/v1"

        called = []

        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(base_url="https://api.x.ai/v1"), model

        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ):
                ok = agent._try_activate_fallback()

        assert ok is True
        assert called == [("xai", "grok-4.5")]
        assert agent.provider == "xai"
        assert agent.model == "grok-4.5"


# ── extra_body re-resolution on fallback activation (#75091) ─────────────


class TestFallbackExtraBodyReResolution:
    """Fallback activation must re-resolve extra_body key-scoped.

    The old provider's custom_providers-contributed extra_body keys are
    stale on the new backend and must be dropped; caller-provided
    request_overrides keys must survive; the fallback provider's own
    extra_body must be merged in (salvage of #75139).
    """

    OLD_URL = "https://old-llm.example.com/v1"
    FB_URL = "https://fb-llm.example.com/v1"

    def _agent_with_custom_providers(self, caller_extra_body=None):
        agent = _make_agent(
            fallback_model={
                "provider": "custom:fbprov",
                "model": "fb-model",
                "base_url": self.FB_URL,
            },
        )
        agent.provider = "custom"
        agent.model = "old-model"
        agent.base_url = self.OLD_URL
        agent._custom_providers = [
            {
                "name": "oldprov",
                "base_url": self.OLD_URL,
                "extra_body": {"enable_thinking": True, "old_only": 1},
            },
            {
                "provider_key": "fbprov",
                "base_url": self.FB_URL,
                "extra_body": {"top_k": 20},
            },
        ]
        # Simulate the init-time merge: provider extra_body + caller keys
        # (caller wins on conflict — agent_init._merge_custom_provider_extra_body).
        merged = {"enable_thinking": True, "old_only": 1}
        merged.update(caller_extra_body or {})
        agent.request_overrides = {"extra_body": merged}
        return agent

    def _activate(self, agent):
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(base_url=self.FB_URL), "fb-model"),
        ), patch(
            "agent.model_metadata.get_model_context_length",
            return_value=128_000,
        ):
            assert agent._try_activate_fallback() is True

    def test_stale_provider_keys_removed_and_new_provider_merged(self):
        agent = self._agent_with_custom_providers()
        self._activate(agent)
        eb = agent.request_overrides.get("extra_body") or {}
        # Old provider's contributed keys are gone.
        assert "enable_thinking" not in eb
        assert "old_only" not in eb
        # Fallback provider's own extra_body is applied.
        assert eb.get("top_k") == 20

    def test_caller_override_keys_survive_fallback(self):
        agent = self._agent_with_custom_providers(
            caller_extra_body={"reasoning": {"effort": "high"}, "enable_thinking": False},
        )
        self._activate(agent)
        eb = agent.request_overrides.get("extra_body") or {}
        # Pure caller key survives untouched.
        assert eb.get("reasoning") == {"effort": "high"}
        # Caller redefined a key the old provider also set (caller won at
        # init: False != True) — the caller's value must survive key-scoped
        # removal.
        assert eb.get("enable_thinking") is False
        # But the key the old provider alone contributed is dropped.
        assert "old_only" not in eb
        assert eb.get("top_k") == 20

    def test_non_extra_body_overrides_untouched(self):
        agent = self._agent_with_custom_providers()
        agent.request_overrides["temperature"] = 0.2
        self._activate(agent)
        assert agent.request_overrides.get("temperature") == 0.2



    def test_first_fallback_freezes_pre_rescope_primary_overrides(self):
        """Live contract: failed/restored fallback keeps original nested overrides."""
        agent = self._agent_with_custom_providers()
        original = agent.request_overrides
        original["extra_body"] = dict(original["extra_body"])
        original["extra_body"]["nested"] = {"value": "primary"}
        agent._primary_runtime = {
            "provider": agent.provider,
            "request_overrides": original,
        }
        self._activate(agent)
        frozen = agent._primary_runtime["request_overrides"]
        assert frozen["extra_body"]["nested"]["value"] == "primary"
        assert "old_only" in frozen["extra_body"]
        assert frozen is not original
        frozen["extra_body"]["nested"]["value"] = "poison"
        assert original["extra_body"]["nested"]["value"] == "primary"
        live = agent.request_overrides.get("extra_body") or {}
        assert "old_only" not in live

    def test_late_b_failure_restores_a_before_real_c_rescope(self):
        """A rejected B cannot make C retain A-only provider overrides.

        B's extra-body derivation is intentionally real.  The only injected fault is
        the later notice sink, after identity/client/override publication, so the next
        chain entry exercises the public fallback loop rather than a rescope stub.
        """
        b_url = "https://b-llm.example.com/v1"
        c_url = "https://c-llm.example.com/v1"
        agent = _make_agent(fallback_model=[
            {"provider": "custom:b", "model": "b-model", "base_url": b_url},
            {"provider": "custom:c", "model": "c-model", "base_url": c_url},
        ])
        agent.provider = "custom"
        agent.model = "a-model"
        agent.base_url = self.OLD_URL
        agent.requested_provider = "custom"
        agent._custom_providers = [
            {"name": "aprov", "base_url": self.OLD_URL, "extra_body": {"a_only": 1}},
            {"provider_key": "b", "base_url": b_url, "extra_body": {"b_only": 2}},
            {"provider_key": "c", "base_url": c_url, "extra_body": {"c_only": 3}},
        ]
        agent.request_overrides = {"extra_body": {"a_only": 1, "caller": "kept"}}
        original_notice = chat_completion_helpers._buffer_fallback_notice

        def fail_only_after_b_rescope(live_agent, notice):
            if live_agent.model == "b-model":
                raise RuntimeError("late B notice sink failure")
            return original_notice(live_agent, notice)

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[
                (_mock_client(base_url=b_url), "b-model"),
                (_mock_client(base_url=c_url), "c-model"),
            ],
        ), patch(
            "agent.model_metadata.get_model_context_length", return_value=128_000
        ), patch(
            "agent.chat_completion_helpers._buffer_fallback_notice",
            side_effect=fail_only_after_b_rescope,
        ):
            assert agent._try_activate_fallback() is True

        assert (agent.model, agent.provider, agent.base_url) == (
            "c-model", "custom:c", c_url
        )
        extra = agent.request_overrides["extra_body"]
        assert extra == {"caller": "kept", "c_only": 3}
        assert "a_only" not in extra
        assert "b_only" not in extra
        # Build the next request through the active transport path without a
        # network call.  The captured kwargs are the actual C-route request
        # projection, not only the mutable runtime dictionary above.
        next_kwargs = chat_completion_helpers.build_api_kwargs(
            agent, [{"role": "user", "content": "capture C route"}], tools_for_api=[]
        )
        assert next_kwargs["model"] == "c-model"
        assert agent.client.base_url == c_url
        assert next_kwargs["extra_body"] == {"caller": "kept", "c_only": 3}

    def test_repeated_late_native_failures_leave_only_successful_d_runtime(self):
        """Two rejected native candidates cannot leak into the later successful route."""
        urls = {
            name: f"https://{name.lower()}.example.com/anthropic"
            for name in ("A", "B", "C", "D")
        }
        agent = _make_agent(fallback_model=[
            {
                "provider": f"custom:{name.lower()}",
                "model": f"{name.lower()}-model",
                "base_url": urls[name],
                "api_mode": "anthropic_messages",
            }
            for name in ("B", "C", "D")
        ])
        agent.provider = agent.requested_provider = "custom:a"
        agent.model = "a-model"
        agent.base_url = urls["A"]
        agent.api_mode = "anthropic_messages"
        agent.api_key = agent._anthropic_api_key = "a-key"
        agent._anthropic_base_url = urls["A"]
        agent._anthropic_client = MagicMock(name="A-native-client")
        agent._is_anthropic_oauth = False
        agent.client = None
        agent._client_kwargs = {}
        agent._credential_pool = MagicMock(name="A-pool", provider="custom:a")
        agent._credential_pool_entry_id = "a-entry"
        agent.runtime_capabilities = {"route": "a"}
        agent._cached_system_prompt = "Model: a-model\nProvider: custom:a"
        agent._pending_fallback_notice = ["A notice"]
        agent.request_overrides = {"extra_body": {"caller": "kept", "a_only": 1}}
        agent._custom_providers = [
            {
                "provider_key": name.lower(),
                "base_url": urls[name],
                "extra_body": {f"{name.lower()}_only": index},
            }
            for index, name in enumerate(("A", "B", "C", "D"), 1)
        ]
        compressor = MagicMock()
        compressor.model = "a-model"
        compressor.provider = "custom:a"
        compressor.base_url = urls["A"]
        compressor.api_key = "a-key"
        compressor.api_mode = "anthropic_messages"
        compressor.context_length = 111
        compressor.update_model.side_effect = lambda **kw: [setattr(compressor, k, v) for k, v in kw.items()]
        agent.context_compressor = compressor

        clients = {
            name: _mock_client(base_url=urls[name], api_key=f"{name.lower()}-key")
            for name in ("B", "C", "D")
        }

        def install_candidate(_agent, _client, provider, *_args):
            name = provider.rsplit(":", 1)[-1].upper()
            agent.api_key = agent._anthropic_api_key = f"{name.lower()}-key"
            agent._anthropic_base_url = urls[name]
            agent._anthropic_client = MagicMock(name=f"{name}-native-client")
            pool = MagicMock(name=f"{name}-pool", provider=provider)
            pool.entry_id_for_api_key.return_value = f"{name.lower()}-entry"
            agent._credential_pool = pool
            agent._credential_pool_entry_id = f"{name.lower()}-entry"

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[(clients[name], f"{name.lower()}-model") for name in ("B", "C", "D")],
        ), patch(
            "agent.client_lifecycle._swap_fallback_clients", side_effect=install_candidate
        ), patch(
            "agent.model_metadata.get_model_context_length", return_value=222
        ), patch(
            "agent.native_compaction.resolve_native_compaction_capabilities",
            side_effect=[RuntimeError("late B"), RuntimeError("late C"), {"route": "d"}],
        ):
            assert agent._try_activate_fallback() is True

        request_client = MagicMock(name="D-request-client")
        with patch.object(agent, "_try_refresh_anthropic_client_credentials", return_value=False), patch.object(
            agent, "_checkout_request_slot", return_value=(None, None)
        ), patch.object(
            agent, "_build_anthropic_client_for_key", return_value=request_client
        ) as build_client, patch.object(agent, "_store_request_slot"):
            assert agent._create_request_anthropic_client(reason="post-repeated-fallback") is request_client

        assert build_client.call_args.args[0][:3] == ("direct", "d-key", urls["D"])
        assert (agent.model, agent.provider, agent.base_url, agent.api_mode) == (
            "d-model", "custom:d", urls["D"], "anthropic_messages"
        )
        assert (agent._anthropic_api_key, agent._anthropic_base_url) == ("d-key", urls["D"])
        assert agent._anthropic_client._mock_name == "D-native-client"
        assert (agent._credential_pool.provider, agent._credential_pool_entry_id) == ("custom:d", "d-entry")
        assert (compressor.model, compressor.provider, compressor.base_url, compressor.context_length) == (
            "d-model", "custom:d", urls["D"], 222
        )
        assert agent.request_overrides == {"extra_body": {"caller": "kept", "d_only": 4}}
        assert agent.runtime_capabilities == {"route": "d"}
        assert agent._cached_system_prompt == "Model: d-model\nProvider: custom:d"
        assert len(agent._pending_fallback_notice) == 2
        assert "d-model via custom:d" in agent._pending_fallback_notice[-1]
        assert "b-model" not in agent._pending_fallback_notice[-1]
        assert "c-model" not in agent._pending_fallback_notice[-1]

    def test_exhausted_late_native_failure_restores_complete_primary_runtime(self):
        """A rejected native fallback cannot leave its transport or compressor on B."""
        a_url = "https://a.example.com/anthropic"
        b_url = "https://b.example.com/anthropic"
        agent = _make_agent(fallback_model=[{
            "provider": "custom:b", "model": "b-model", "base_url": b_url,
            "api_mode": "anthropic_messages",
        }])
        agent.provider = agent.requested_provider = "custom:a"
        agent.model = "a-model"
        agent.base_url = a_url
        agent.api_mode = "anthropic_messages"
        agent.api_key = agent._anthropic_api_key = "a-key"
        agent._anthropic_base_url = a_url
        agent._anthropic_client = MagicMock(name="A-native-client")
        agent._is_anthropic_oauth = False
        agent.client = None
        agent._client_kwargs = {}
        agent._credential_pool = MagicMock(name="A-pool", provider="custom:a")
        agent._credential_pool_entry_id = "a-entry"
        agent._use_prompt_caching = True
        agent._use_native_cache_layout = True
        agent.reasoning_config = {"effort": "high"}
        agent.runtime_capabilities = {"route": "a"}
        agent._provider_fallback_active = False
        agent._provider_fallback_route = None
        agent._cached_system_prompt = "Model: a-model\nProvider: custom:a"
        agent._pending_fallback_notice = ["A notice"]
        agent._consecutive_stale_streams = 4
        agent.request_overrides = {"extra_body": {"route": "a"}}
        compressor = MagicMock()
        compressor.model = "a-model"
        compressor.provider = "custom:a"
        compressor.base_url = a_url
        compressor.api_key = "a-key"
        compressor.api_mode = "anthropic_messages"
        compressor.context_length = 111
        agent.context_compressor = compressor

        def mutate_compressor(**kwargs):
            for key, value in kwargs.items():
                setattr(compressor, key, value)

        compressor.update_model.side_effect = mutate_compressor
        fb_client = _mock_client(base_url=b_url, api_key="b-key")

        with patch(
            "agent.auxiliary_client.resolve_provider_client", return_value=(fb_client, "b-model")
        ), patch(
            "agent.client_lifecycle._swap_fallback_clients"
        ) as swap, patch(
            "agent.model_metadata.get_model_context_length", return_value=222
        ), patch(
            "agent.native_compaction.resolve_native_compaction_capabilities",
            side_effect=RuntimeError("late capability failure"),
        ):
            def install_b(*_args):
                agent.api_key = agent._anthropic_api_key = "b-key"
                agent._anthropic_base_url = b_url
                agent._anthropic_client = MagicMock(name="B-native-client")
                agent._credential_pool = MagicMock(name="B-pool", provider="custom:b")
                agent._credential_pool_entry_id = "b-entry"
            swap.side_effect = install_b
            assert agent._try_activate_fallback() is False

        # Exercise the request-local physical transport key without opening a
        # socket.  The next native request must be constructed from A's restored
        # credential and endpoint rather than B's rejected client state.
        request_client = MagicMock(name="A-request-client")
        with patch.object(agent, "_try_refresh_anthropic_client_credentials", return_value=False), patch.object(
            agent, "_checkout_request_slot", return_value=(None, None)
        ), patch.object(
            agent, "_build_anthropic_client_for_key", return_value=request_client
        ) as build_client, patch.object(agent, "_store_request_slot"):
            assert agent._create_request_anthropic_client(reason="post-rejected-fallback") is request_client
        built_key = build_client.call_args.args[0]
        assert built_key[:3] == ("direct", "a-key", a_url)

        assert (agent.model, agent.provider, agent.base_url, agent.api_mode) == (
            "a-model", "custom:a", a_url, "anthropic_messages"
        )
        assert (agent._anthropic_api_key, agent._anthropic_base_url) == ("a-key", a_url)
        assert agent._anthropic_client._mock_name == "A-native-client"
        assert (agent._credential_pool.provider, agent._credential_pool_entry_id) == ("custom:a", "a-entry")
        assert (compressor.model, compressor.provider, compressor.base_url, compressor.context_length) == (
            "a-model", "custom:a", a_url, 111
        )
        assert agent.request_overrides == {"extra_body": {"route": "a"}}
        assert agent.runtime_capabilities == {"route": "a"}
        assert agent._cached_system_prompt == "Model: a-model\nProvider: custom:a"
        assert agent._pending_fallback_notice == ["A notice"]
        assert agent._consecutive_stale_streams == 4

# ── MoA preset as a fallback entry (#112525, #112623) ─────────────────────


def _write_moa_home(tmp_path, monkeypatch):
    """Real config.yaml with a MoA preset under a temp HERMES_HOME (genuine preset resolution)."""
    import yaml

    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "moa": {"default_preset": "default", "presets": {"default": {
            "enabled": True,
            "reference_models": [{"provider": "xai", "model": "grok-4-fast"}],
            "aggregator": {"provider": "xai", "model": "grok-4.6"},
        }}},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _assert_bound_to_moa_preset(agent, preset="default"):
    from agent.conversation_loop import _moa_client_consumes_prepared_request

    assert (agent.provider, agent.requested_provider, agent.model) == ("moa", "moa", preset)
    assert (agent.base_url, agent.api_mode) == ("moa://local", "chat_completions")
    assert agent._client_kwargs == {}
    assert _moa_client_consumes_prepared_request(agent.client)


class TestMoaPresetFallback:
    def test_runtime_fallback_to_moa_preset_binds_the_facade(self, tmp_path, monkeypatch):
        """#112525 / #112623: a ``{provider: moa, model: <preset>}`` fallback entry activates the
        preset (facade, ``moa://local``), never the aggregator's HTTP client wearing the virtual
        identity (preset name on the aggregator wire → 404; ``provider == "moa"`` guards misfire)."""
        _write_moa_home(tmp_path, monkeypatch)
        agent = _make_agent(fallback_model={"provider": "moa", "model": "default"})
        aggregator_client = _mock_client(base_url="https://api.x.ai/v1/", api_key="xai-key")
        with patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(aggregator_client, "grok-4.6")):
            assert agent._try_activate_fallback() is True
        _assert_bound_to_moa_preset(agent)
        assert agent.client is not aggregator_client
        assert agent._provider_fallback_route == ("default", "moa")

    def test_init_time_fallback_to_moa_preset_binds_the_facade(self, tmp_path, monkeypatch):
        """Primary without credentials at init walks the chain: a MoA entry lands on the preset
        with virtual pins, not on the aggregator slug with the aggregator's kwargs."""
        _write_moa_home(tmp_path, monkeypatch)
        aggregator_client = _mock_client(base_url="https://api.x.ai/v1/", api_key="xai-key")

        def _route(provider, model=None, **_kw):
            return (aggregator_client, "grok-4.6") if provider == "moa" else (None, None)

        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.auxiliary_client.resolve_provider_client", side_effect=_route),
        ):
            agent = AIAgent(model="anthropic/claude-sonnet-4.5", provider="openrouter",
                            quiet_mode=True, skip_context_files=True, skip_memory=True,
                            fallback_model={"provider": "moa", "model": "default"})
        _assert_bound_to_moa_preset(agent)
        assert agent._fallback_activated is True
