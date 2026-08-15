"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.credential_pool import STATUS_EXHAUSTED
from run_agent import AIAgent, _pool_may_recover_from_rate_limit
from tests.run_agent.test_run_agent import _mock_response


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
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
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2

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


    def test_nous_anthropic_fallback_uses_the_messages_wire(self):
        """Portal Claude fallbacks must not stay on chat_completions.

        ``resolve_provider_client`` still returns an OpenAI client for Nous;
        activation has to re-derive api_mode from the model and rebuild the
        Anthropic client — otherwise the turn POSTs /chat/completions.
        """
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


class _UsageLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("The usage limit has been reached")
        self.body = {
            "error": {
                "type": "usage_limit_reached",
                "message": "The usage limit has been reached",
            }
        }
        self.response = SimpleNamespace(headers={})


class _QuotaPool:
    def __init__(self, api_key: str, *, distinct_account_available: bool):
        self.primary = SimpleNamespace(
            id="primary", label="primary", runtime_api_key=api_key, last_status=None
        )
        self.secondary = SimpleNamespace(
            id="secondary",
            label="secondary",
            runtime_api_key="secondary-key",
            last_status=None,
        )
        self._distinct_account_available = distinct_account_available
        self.rotate_calls = 0

    def current(self):
        return self.primary

    def entries(self):
        return [self.primary, self.secondary]

    def has_available(self):
        return any(row.last_status != STATUS_EXHAUSTED for row in self.entries())

    def mark_exhausted_and_rotate(self, **_kwargs):
        self.rotate_calls += 1
        self.primary.last_status = STATUS_EXHAUSTED
        if self._distinct_account_available:
            return self.secondary
        self.secondary.last_status = STATUS_EXHAUSTED
        return None


def _run_usage_limit_turn(agent, api_side_effect):
    agent._api_max_retries = 3
    agent._interruptible_api_call = MagicMock(side_effect=api_side_effect)
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.agent_runtime_helpers.time.sleep") as helper_sleep,
        patch("agent.conversation_loop.time.sleep") as loop_sleep,
    ):
        result = agent.run_conversation("hello")
    return result, helper_sleep, loop_sleep


class TestUsageLimitResolutionChain:
    def test_distinct_account_rotation_precedes_fallback(self):
        agent = _make_agent(
            fallback_model=[{"provider": "fallback", "model": "fallback-model"}]
        )
        pool = _QuotaPool(agent.api_key, distinct_account_available=True)
        agent._credential_pool = pool
        responses = [_UsageLimitError(), _mock_response(content="rotated account")]

        def api_call(_kwargs):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        with (
            patch.object(agent, "_swap_credential") as swap,
            patch.object(agent, "_try_activate_fallback") as fallback,
        ):
            result, helper_sleep, loop_sleep = _run_usage_limit_turn(agent, api_call)

        assert result["final_response"] == "rotated account"
        assert agent._interruptible_api_call.call_count == 2
        swap.assert_called_once_with(pool.secondary)
        fallback.assert_not_called()
        helper_sleep.assert_not_called()

    def test_same_account_quota_wall_activates_fallback_before_backoff(self):
        agent = _make_agent(
            fallback_model=[{"provider": "fallback", "model": "fallback-model"}]
        )
        pool = _QuotaPool(agent.api_key, distinct_account_available=False)
        agent._credential_pool = pool
        responses = [_UsageLimitError(), _mock_response(content="fallback answer")]

        def api_call(_kwargs):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def activate(reason=None):
            agent._fallback_index = len(agent._fallback_chain)
            agent._fallback_activated = True
            agent.provider = "fallback"
            agent.model = "fallback-model"
            return True

        with patch.object(
            agent, "_try_activate_fallback", side_effect=activate
        ) as fallback:
            result, helper_sleep, loop_sleep = _run_usage_limit_turn(agent, api_call)

        assert result["final_response"] == "fallback answer"
        assert agent._interruptible_api_call.call_count == 2
        fallback.assert_called_once()
        helper_sleep.assert_not_called()

    def test_same_account_quota_wall_without_fallback_terminates_immediately(self):
        agent = _make_agent(fallback_model=None)
        pool = _QuotaPool(agent.api_key, distinct_account_available=False)
        agent._credential_pool = pool

        result, helper_sleep, loop_sleep = _run_usage_limit_turn(
            agent, lambda _kwargs: (_ for _ in ()).throw(_UsageLimitError())
        )

        assert result["completed"] is False
        assert result["failed"] is True
        assert agent._interruptible_api_call.call_count == 1
        assert pool.rotate_calls == 1
        helper_sleep.assert_not_called()







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
