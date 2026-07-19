"""Focused contract tests for opt-in quota-only auxiliary fallback."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import auxiliary_quota_policy as policy
from agent import error_classifier
from agent import secret_scope


class ProviderError(RuntimeError):
    def __init__(self, message: str = "provider error", **attrs):
        super().__init__(message)
        for name, value in attrs.items():
            setattr(self, name, value)


class SyncCompletions:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class AsyncCompletions(SyncCompletions):
    async def create(self, **kwargs):
        return super().create(**kwargs)


class Client:
    def __init__(self, completions, base_url="https://primary.example/v1"):
        self.chat = SimpleNamespace(completions=completions)
        self.base_url = base_url


def response(text="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def quota_error(message="quota exhausted"):
    return ProviderError(
        message,
        status_code=429,
        body={"error": {"code": "quota_exhausted"}},
    )


def quota_config(chain_size=1):
    return {
        "provider": "custom",
        "model": "primary-model",
        "base_url": "https://primary.example/v1",
        "api_key": "primary-key",
        "timeout": 91,
        "extra_body": {"snapshot": "a"},
        "fallback_on": ["quota_exhausted"],
        "fallback_chain": [
            {
                "provider": "custom",
                "model": f"backup-{index}",
                "base_url": f"https://backup-{index}.example/v1",
                "api_key": f"backup-key-{index}",
                "timeout": 10 + index,
            }
            for index in range(chain_size)
        ],
    }


@pytest.fixture(autouse=True)
def clean_auxiliary_state():
    context = getattr(aux, "_AUXILIARY_TASK_SNAPSHOT_CONTEXT", None)
    if context is not None:
        context.set(None)
    with aux._client_cache_lock:
        aux._client_cache.clear()
    aux._aux_unhealthy_until.clear()
    aux._aux_unhealthy_logged_at.clear()
    yield
    if context is not None:
        context.set(None)
    with aux._client_cache_lock:
        aux._client_cache.clear()
    aux._aux_unhealthy_until.clear()
    aux._aux_unhealthy_logged_at.clear()


def install_sync_chain(monkeypatch, primary, backups, config=None):
    config = config or quota_config(len(backups))
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
    candidates = iter(backups)

    def get_cached(*_args, **kwargs):
        if kwargs.get("base_url") == "https://primary.example/v1":
            return primary, "primary-model"
        return next(candidates), "backup-model"

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)


def call_public(async_mode, **kwargs):
    if async_mode:
        return asyncio.run(aux.async_call_llm(**kwargs))
    return aux.call_llm(**kwargs)


class TestPolicyParsing:
    @pytest.mark.parametrize(
        "value",
        [None, [], {}, "quota_exhausted", ["unknown"], [""], [3]],
    )
    def test_present_malformed_or_unknown_policy_fails_closed(self, monkeypatch, value):
        config = quota_config()
        config["fallback_on"] = value
        original = quota_error()
        primary_calls = SyncCompletions(original)
        install_sync_chain(monkeypatch, Client(primary_calls), [], config)

        with pytest.raises(ProviderError) as caught:
            aux.call_llm(
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )
        assert caught.value is original
        assert len(primary_calls.calls) == 1

    def test_absent_policy_preserves_legacy_mode(self, monkeypatch):
        config = quota_config()
        del config["fallback_on"]
        primary_calls = SyncCompletions(
            ProviderError("payment required", status_code=402)
        )
        fallback_calls = SyncCompletions(response("legacy fallback"))
        install_sync_chain(monkeypatch, Client(primary_calls), [], config)
        monkeypatch.setattr(
            aux,
            "_try_configured_fallback_chain",
            lambda *_args, **_kwargs: (
                Client(fallback_calls),
                "backup-model",
                "legacy",
            ),
        )

        result = aux.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
        )
        assert result.choices[0].message.content == "legacy fallback"
        assert len(fallback_calls.calls) == 1


class TestNarrowClassifier:
    @pytest.mark.parametrize(
        "error",
        [
            quota_error(),
            ProviderError(body={"error": {"type": "insufficient_quota"}}),
            ProviderError(error={"reason": "usage_limit_reached"}),
            ProviderError(
                status_code=403,
                body={"error": {"code": "personal-team-blocked:spending-limit"}},
            ),
            type("DeviceCodeExhaustedError", (ProviderError,), {})(
                "weekly credits exhausted"
            ),
            type("GoUsageLimitError", (ProviderError,), {})("weekly usage reached"),
        ],
    )
    def test_accepts_only_structured_exhaustion_authority(self, error):
        assert error_classifier.is_explicit_usage_quota_exhaustion(error)

    @pytest.mark.parametrize(
        "error",
        [
            ProviderError("weekly quota exhausted"),
            ProviderError(status_code=429),
            ProviderError(
                status_code=429,
                body={
                    "error": {"type": "rate_limit_error", "message": "quota exhausted"}
                },
            ),
            TimeoutError("quota_exhausted"),
            ConnectionError("quota exhausted"),
            ProviderError(
                status_code=503,
                body={"error": {"code": "quota_exhausted"}},
            ),
            ProviderError(
                status_code=403,
                body={"error": {"code": "quota_exhausted"}},
            ),
        ],
    )
    def test_rejects_text_transient_transport_and_server_signals(self, error):
        assert not error_classifier.is_explicit_usage_quota_exhaustion(error)

    def test_ignores_metadata_and_bounds_exception_graph(self):
        payload = {"metadata": {"debug": "quota_exhausted"}}
        error = ProviderError(body={"error": payload})
        assert not error_classifier.is_explicit_usage_quota_exhaustion(error)

        first = ProviderError(body={"error": {"code": "quota_exhausted"}})
        second = RuntimeError("cycle")
        first.__cause__ = second
        second.__context__ = first
        assert not error_classifier.is_explicit_usage_quota_exhaustion(first)

    @pytest.mark.parametrize(
        "status",
        [HTTPStatus(code) for code in (401, 403, 404, 408, 500, 503)],
    )
    def test_integral_status_subclasses_preserve_hard_vetoes(self, status):
        error = ProviderError(
            status_code=status,
            body={"error": {"code": "quota_exhausted"}},
        )
        assert not error_classifier.is_explicit_usage_quota_exhaustion(error)

    def test_xai_integral_403_remains_the_narrow_exception(self):
        error = ProviderError(
            status_code=HTTPStatus.FORBIDDEN,
            body={"error": {"code": "personal-team-blocked:spending-limit"}},
        )
        assert error_classifier.is_explicit_usage_quota_exhaustion(error)

    def test_bool_status_is_not_treated_as_an_integral_http_status(self):
        error = ProviderError(
            status_code=True,
            body={"error": {"code": "quota_exhausted"}},
        )
        assert error_classifier.is_explicit_usage_quota_exhaustion(error)

    def test_nested_hard_status_vetoes_outer_quota_authority(self):
        outer = ProviderError(body={"error": {"code": "quota_exhausted"}})
        outer.__context__ = ProviderError(status_code=503)
        assert not error_classifier.is_explicit_usage_quota_exhaustion(outer)

    def test_authority_markers_require_exact_structured_values(self):
        assert not error_classifier.is_explicit_usage_quota_exhaustion(
            ProviderError(body={"error": {"code": "prefix_quota_exhausted_suffix"}})
        )

    def test_stop_boundary_applies_to_response_edges(self):
        original = quota_error("primary quota")
        candidate = ProviderError("candidate transport failure", response=original)
        assert not error_classifier.is_explicit_usage_quota_exhaustion(
            candidate,
            stop_exceptions=(original,),
        )

    def test_classifier_never_calls_response_json(self):
        class Response:
            body = {"error": {"message": "ordinary throttling"}}

            def __init__(self):
                self.called = False

            def json(self):
                self.called = True
                return {"error": {"code": "quota_exhausted"}}

        response_obj = Response()
        error = ProviderError(response=response_obj)
        assert not error_classifier.is_explicit_usage_quota_exhaustion(error)
        assert response_obj.called is False

    @pytest.mark.parametrize(
        "body",
        [
            {"code": "quota_exhausted"},
            {"error": {"details": {"reason": "usage_limit_reached"}}},
            {"error": {"error": {"type": "insufficient_quota"}}},
        ],
    )
    def test_accepts_bounded_designated_nested_envelopes(self, body):
        assert error_classifier.is_explicit_usage_quota_exhaustion(
            ProviderError(body=body)
        )

    @pytest.mark.parametrize(
        "error",
        [
            ProviderError(body={"error": {"code": "resource_exhausted"}}),
            ProviderError(body={"error": {"code": "rate_limit_exceeded"}}),
            ProviderError(
                body={
                    "error": {
                        "code": "quota_exhausted",
                        "message": "RPM quota metric",
                    }
                }
            ),
            ProviderError(
                body={
                    "error": {"code": "quota_exhausted", "message": "tokens per minute"}
                }
            ),
            type("AuthenticationError", (ProviderError,), {})(
                body={"error": {"code": "quota_exhausted"}}
            ),
        ],
    )
    def test_authoritative_nonquota_conflicts_fail_closed(self, error):
        assert not error_classifier.is_explicit_usage_quota_exhaustion(error)

    @pytest.mark.parametrize(
        "error",
        [
            ProviderError(body={"error": {"code": "x" * 4097}}),
            ProviderError(
                body={
                    "error": {
                        "message": "x" * 4097,
                        "code": "quota_exhausted",
                    }
                }
            ),
            ProviderError(
                body={
                    "error": {
                        "error": {
                            "error": {"error": {"error": {"code": "quota_exhausted"}}}
                        }
                    }
                }
            ),
        ],
    )
    def test_scalar_message_and_depth_bounds_fail_closed(self, error):
        assert not error_classifier.is_explicit_usage_quota_exhaustion(error)


class TestClosedRouting:
    def test_sync_quota_walks_only_configured_chain(self, monkeypatch):
        primary_calls = SyncCompletions(quota_error())
        first_calls = SyncCompletions(quota_error("backup quota"))
        second_calls = SyncCompletions(response("second backup"))
        install_sync_chain(
            monkeypatch,
            Client(primary_calls),
            [Client(first_calls), Client(second_calls)],
        )
        monkeypatch.setattr(
            aux,
            "_try_main_fallback_chain",
            lambda *_a, **_k: pytest.fail("ambient main chain leaked"),
        )
        monkeypatch.setattr(
            aux,
            "_try_main_agent_model_fallback",
            lambda *_a, **_k: pytest.fail("ambient main model leaked"),
        )
        monkeypatch.setattr(
            aux,
            "_try_payment_fallback",
            lambda *_a, **_k: pytest.fail("discovery chain leaked"),
        )

        result = aux.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
        )

        assert result.choices[0].message.content == "second backup"
        assert [
            len(item.calls) for item in (primary_calls, first_calls, second_calls)
        ] == [1, 1, 1]
        assert first_calls.calls[0]["timeout"] == 10
        assert second_calls.calls[0]["timeout"] == 11
        assert second_calls.calls[0]["extra_body"] == {"snapshot": "a"}

    def test_nonquota_never_enters_chain(self, monkeypatch):
        original = ProviderError("ordinary rate limit", status_code=429)
        primary_calls = SyncCompletions(original)
        backup_calls = SyncCompletions(response("must not run"))
        install_sync_chain(monkeypatch, Client(primary_calls), [Client(backup_calls)])

        with pytest.raises(ProviderError) as caught:
            aux.call_llm(
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )

        assert caught.value is original
        assert len(primary_calls.calls) == 1
        assert backup_calls.calls == []

    def test_nonquota_candidate_reraises_original_with_candidate_as_cause(
        self, monkeypatch
    ):
        original = quota_error("primary quota")
        candidate_error = ProviderError("candidate transport failure")
        primary_calls = SyncCompletions(original)
        backup_calls = SyncCompletions(candidate_error)
        install_sync_chain(monkeypatch, Client(primary_calls), [Client(backup_calls)])

        with pytest.raises(ProviderError) as caught:
            aux.call_llm(
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )

        assert caught.value is original
        assert caught.value.__cause__ is candidate_error
        assert backup_calls.calls and len(backup_calls.calls) == 1

    @pytest.mark.parametrize("chain", [None, [], {}, ["bad"]])
    def test_missing_empty_or_malformed_chain_fails_closed(self, monkeypatch, chain):
        config = quota_config()
        if chain is None:
            del config["fallback_chain"]
        else:
            config["fallback_chain"] = chain
        original = quota_error()
        primary_calls = SyncCompletions(original)
        install_sync_chain(monkeypatch, Client(primary_calls), [], config)

        with pytest.raises(ProviderError) as caught:
            aux.call_llm(
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )

        assert caught.value is original
        assert len(primary_calls.calls) == 1

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_ambient_auto_primary_is_rejected_before_any_request(
        self, monkeypatch, async_mode
    ):
        config = quota_config()
        config.update(provider="auto")
        config.pop("base_url")
        config.pop("api_key")
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
        monkeypatch.setattr(
            aux,
            "_get_cached_client",
            lambda *_args, **_kwargs: pytest.fail("ambient primary was resolved"),
        )
        with pytest.raises(RuntimeError, match="no concrete primary"):
            call_public(
                async_mode,
                task="web_extract",
                messages=[{"role": "user", "content": "extract"}],
            )

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_malformed_entry_invalidates_the_entire_closed_chain(
        self, monkeypatch, async_mode
    ):
        config = quota_config(2)
        config["fallback_chain"][0] = {"provider": "auto"}
        original = quota_error()
        primary_calls = (
            AsyncCompletions(original) if async_mode else SyncCompletions(original)
        )
        later_calls = (
            AsyncCompletions(response("must not run"))
            if async_mode
            else SyncCompletions(response("must not run"))
        )
        install_sync_chain(
            monkeypatch,
            Client(primary_calls),
            [Client(later_calls)],
            config,
        )

        with pytest.raises(ProviderError) as caught:
            call_public(
                async_mode,
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )
        assert caught.value is original
        assert later_calls.calls == []

    def test_ambient_builtin_candidate_is_rejected_before_resolution(self, monkeypatch):
        config = quota_config()
        config["fallback_chain"] = [
            {
                "provider": "nous",
                "model": "ambient-model",
                "base_url": "https://frozen.example/v1",
                "api_key": "frozen-key",
            }
        ]
        original = quota_error()
        primary_calls = SyncCompletions(original)
        install_sync_chain(monkeypatch, Client(primary_calls), [], config)

        with pytest.raises(ProviderError) as caught:
            aux.call_llm(
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )
        assert caught.value is original

    @pytest.mark.asyncio
    async def test_async_matches_sync_chain_and_exception_precedence(self, monkeypatch):
        original = quota_error("primary quota")
        candidate_error = ProviderError("candidate transport failure")
        primary_calls = AsyncCompletions(original)
        backup_calls = AsyncCompletions(candidate_error)
        install_sync_chain(monkeypatch, Client(primary_calls), [Client(backup_calls)])

        with pytest.raises(ProviderError) as caught:
            await aux.async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )

        assert caught.value is original
        assert caught.value.__cause__ is candidate_error
        assert len(primary_calls.calls) == 1
        assert len(backup_calls.calls) == 1

    def test_identical_entries_advance_once_per_frozen_index(self, monkeypatch):
        config = quota_config(2)
        config["fallback_chain"][1] = dict(config["fallback_chain"][0])
        primary_calls = SyncCompletions(quota_error("primary"))
        first_calls = SyncCompletions(quota_error("first"))
        second_calls = SyncCompletions(response("second"))
        install_sync_chain(
            monkeypatch,
            Client(primary_calls),
            [Client(first_calls), Client(second_calls)],
            config,
        )

        result = aux.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
        )

        assert result.choices[0].message.content == "second"
        assert [len(first_calls.calls), len(second_calls.calls)] == [1, 1]

    def test_adversarial_response_getter_cannot_authorize_chain(self, monkeypatch):
        embedded_quota = quota_error("getter quota")

        class MalformedResponse:
            @property
            def choices(self):
                raise embedded_quota

        primary_calls = SyncCompletions(MalformedResponse())
        backup_calls = SyncCompletions(response("must not run"))
        install_sync_chain(monkeypatch, Client(primary_calls), [Client(backup_calls)])

        with pytest.raises(aux.AuxiliaryResponseValidationError):
            aux.call_llm(
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )
        assert backup_calls.calls == []


class TestFrozenSnapshot:
    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize(
        "override",
        [
            {"base_url": "https://attacker.example/v1"},
            {"provider": "openrouter"},
        ],
    )
    def test_route_identity_override_never_splices_task_credentials(
        self, monkeypatch, async_mode, override
    ):
        config = quota_config(0)
        config["api_key"] = "SECRET"
        observed = []
        calls = (
            AsyncCompletions(response()) if async_mode else SyncCompletions(response())
        )
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

        def get_cached(*args, **kwargs):
            observed.append((args, dict(kwargs)))
            return Client(calls), "primary-model"

        monkeypatch.setattr(aux, "_get_cached_client", get_cached)
        call_public(
            async_mode,
            task="web_extract",
            messages=[{"role": "user", "content": "extract"}],
            **override,
        )
        assert observed[0][1].get("api_key") != "SECRET"
        assert "SECRET" not in str(observed)

    def test_key_env_uses_profile_secret_scope_and_never_borrows_ambient(
        self, monkeypatch
    ):
        monkeypatch.setenv("BACKUP_API_KEY", "ambient-other-profile")
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({"BACKUP_API_KEY": "scoped-key"})
        try:
            snapshot = policy.capture_closed_plan(
                "compression",
                {
                    **quota_config(0),
                    "fallback_chain": [
                        {
                            "provider": "custom",
                            "model": "backup",
                            "base_url": "https://backup.example/v1",
                            "key_env": "BACKUP_API_KEY",
                        }
                    ],
                },
            )
            assert snapshot.candidates[0].api_key == "scoped-key"
        finally:
            secret_scope.reset_secret_scope(token)

        with pytest.raises(secret_scope.UnscopedSecretError):
            policy.capture_closed_plan(
                "compression",
                {
                    **quota_config(0),
                    "fallback_chain": [
                        {
                            "provider": "custom",
                            "model": "backup",
                            "base_url": "https://backup.example/v1",
                            "key_env": "BACKUP_API_KEY",
                        }
                    ],
                },
            )

        token = secret_scope.set_secret_scope({})
        try:
            snapshot = policy.capture_closed_plan(
                "compression",
                {
                    **quota_config(0),
                    "fallback_chain": [
                        {
                            "provider": "custom",
                            "model": "backup",
                            "base_url": "https://backup.example/v1",
                            "api_key_env": "BACKUP_API_KEY",
                        }
                    ],
                },
            )
            assert snapshot.candidates[0].api_key == "no-key-required"
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

    def test_route_body_timeout_chain_and_named_custom_are_one_generation(
        self, monkeypatch
    ):
        named = {
            "name": "backup",
            "base_url": "https://named-a.example/v1",
            "api_key": "named-key-a",
            "api_mode": "codex_responses",
        }
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda _provider: dict(named),
        )
        config = quota_config()
        config["fallback_chain"] = [
            {"provider": "custom:backup", "model": "frozen-model", "timeout": 17}
        ]

        snapshot = policy.capture_closed_plan("compression", config)
        config["timeout"] = 1
        config["extra_body"]["snapshot"] = "mutated"
        named.update(
            base_url="https://named-b.example/v1",
            api_key="named-key-b",
            api_mode="chat_completions",
        )
        route = snapshot.candidates[0]

        assert isinstance(snapshot.config, Mapping)
        assert snapshot.config["timeout"] == 91
        assert snapshot.config["extra_body"] == {"snapshot": "a"}
        assert route is not None
        assert (route.model, route.base_url, route.api_key, route.api_mode) == (
            "frozen-model",
            "https://named-a.example/v1",
            "named-key-a",
            "codex_responses",
        )
        with pytest.raises(TypeError):
            snapshot.config["timeout"] = 3

    def test_named_route_is_not_reloaded_after_primary_failure(self, monkeypatch):
        named = {
            "base_url": "https://named-a.example/v1",
            "api_key": "named-key-a",
            "api_mode": "chat_completions",
        }
        reads = []

        def load_named(_provider):
            reads.append(dict(named))
            return dict(named)

        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider", load_named
        )
        config = quota_config()
        config["fallback_chain"] = [
            {"provider": "custom:backup", "model": "backup-model"}
        ]
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
        primary_calls = SyncCompletions(quota_error())
        backup_calls = SyncCompletions(response("frozen"))
        observed = []

        def get_cached(*_args, **kwargs):
            observed.append(dict(kwargs))
            if kwargs.get("base_url") == "https://primary.example/v1":
                named.update(
                    base_url="https://named-b.example/v1",
                    api_key="named-key-b",
                )
                return Client(primary_calls), "primary-model"
            return Client(backup_calls), "backup-model"

        monkeypatch.setattr(aux, "_get_cached_client", get_cached)

        result = aux.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
        )

        assert result.choices[0].message.content == "frozen"
        assert len(reads) == 1
        assert observed[1]["base_url"] == "https://named-a.example/v1"
        assert observed[1]["api_key"] == "named-key-a"

    def test_main_entry_uses_frozen_runtime_route(self):
        runtime = {
            "provider": "custom",
            "model": "main-a",
            "base_url": "https://main-a.example/v1",
            "api_key": "main-key-a",
            "api_mode": "chat_completions",
        }
        config = quota_config()
        config["fallback_chain"] = [{"provider": "main", "timeout": 22}]
        snapshot = policy.capture_closed_plan(
            "compression", config, main_runtime=runtime
        )
        runtime.update(
            model="main-b",
            base_url="https://main-b.example/v1",
            api_key="main-key-b",
        )

        route = snapshot.candidates[0]
        assert (route.model, route.base_url, route.api_key, route.timeout) == (
            "main-a",
            "https://main-a.example/v1",
            "main-key-a",
            22,
        )

    def test_main_sentinel_without_frozen_credentials_is_not_a_candidate(self):
        snapshot = policy.capture_closed_plan(
            "compression",
            {
                "fallback_on": ["quota_exhausted"],
                "fallback_chain": [{"provider": "main"}],
            },
            main_runtime={
                "provider": "openai-codex",
                "model": "gpt-main",
                "base_url": "https://chatgpt.com/backend-api/codex",
            },
        )
        assert snapshot.candidates == ()

    @pytest.mark.parametrize("provider", ["custom", "custom:auto", "auto"])
    def test_model_less_or_ambient_custom_routes_are_unavailable(
        self, monkeypatch, provider
    ):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
        snapshot = policy.capture_closed_plan(
            "compression",
            {
                "fallback_on": ["quota_exhausted"],
                "fallback_chain": [{"provider": provider}],
            },
        )
        assert snapshot.candidates == ()

    def test_candidate_rejects_deferred_callable_credentials(self):
        snapshot = policy.capture_closed_plan(
            "compression",
            {
                "fallback_on": ["quota_exhausted"],
                "fallback_chain": [
                    {
                        "provider": "custom",
                        "model": "backup",
                        "base_url": "https://backup.example/v1",
                        "api_key": lambda: "late-secret",
                    }
                ],
            },
        )
        assert snapshot.candidates == ()

    def test_real_config_loader_and_public_call_freeze_named_key_env_route(
        self, tmp_path, monkeypatch
    ):
        import yaml

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("ARCHIVE_API_KEY", "archive-key-a")
        config = {
            "model": {"provider": "custom", "default": "main-model"},
            "custom_providers": [
                {
                    "name": "archive",
                    "base_url": "https://archive.example/v1",
                    "key_env": "ARCHIVE_API_KEY",
                    "api_mode": "codex_responses",
                }
            ],
            "auxiliary": {
                "compression": {
                    "provider": "custom",
                    "model": "primary-model",
                    "base_url": "https://primary.example/v1",
                    "api_key": "primary-key",
                    "fallback_on": ["quota_exhausted"],
                    "fallback_chain": [
                        {"provider": "custom:archive", "model": "archive-model"}
                    ],
                }
            },
        }
        (hermes_home / "config.yaml").write_text(yaml.safe_dump(config))
        primary_calls = SyncCompletions(quota_error())
        fallback_calls = SyncCompletions(response("frozen"))
        observed = []

        def get_cached(*_args, **kwargs):
            observed.append(dict(kwargs))
            if kwargs.get("base_url") == "https://primary.example/v1":
                config["custom_providers"][0].update(
                    base_url="https://mutated.example/v1",
                    api_mode="chat_completions",
                )
                (hermes_home / "config.yaml").write_text(yaml.safe_dump(config))
                monkeypatch.setenv("ARCHIVE_API_KEY", "archive-key-b")
                return Client(primary_calls), "primary-model"
            return Client(fallback_calls), "archive-model"

        monkeypatch.setattr(aux, "_get_cached_client", get_cached)
        result = aux.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
        )
        assert result.choices[0].message.content == "frozen"
        assert observed[1]["base_url"] == "https://archive.example/v1"
        assert observed[1]["api_key"] == "archive-key-a"
        assert observed[1]["api_mode"] == "codex_responses"


@pytest.mark.parametrize("async_mode", [False, True])
def test_model_less_closed_chain_never_discovers_glm_or_grok(monkeypatch, async_mode):
    monkeypatch.setenv("GLM_API_KEY", "ambient-glm")
    monkeypatch.setenv("XAI_API_KEY", "ambient-grok")
    config = quota_config()
    config["fallback_chain"] = [
        {"provider": "custom"},
        {"provider": "auto"},
        {"provider": "custom:auto"},
    ]
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
    original = quota_error()
    calls = AsyncCompletions(original) if async_mode else SyncCompletions(original)

    def get_cached(*_args, **kwargs):
        if kwargs.get("base_url") != "https://primary.example/v1":
            pytest.fail("model-less route attempted client creation")
        return Client(calls), "primary-model"

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)

    if async_mode:

        async def run():
            with pytest.raises(ProviderError) as caught:
                await aux.async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "compress"}],
                )
            assert caught.value is original

        import asyncio

        asyncio.run(run())
    else:
        with pytest.raises(ProviderError) as caught:
            aux.call_llm(
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )
        assert caught.value is original
    assert len(calls.calls) == 1


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("retry_count, expected_calls", [(0, 1), (2, 2)])
def test_closed_policy_transient_retry_decision(
    monkeypatch, async_mode, retry_count, expected_calls
):
    transport_error = ConnectionError("connection reset")
    calls = (
        AsyncCompletions(transport_error)
        if async_mode
        else SyncCompletions(transport_error)
    )
    install_sync_chain(monkeypatch, Client(calls), [])
    monkeypatch.setattr(aux, "_transient_retry_count", lambda: retry_count)
    monkeypatch.setattr(aux.time, "sleep", lambda _seconds: None)

    if async_mode:

        async def run():
            with pytest.raises(ConnectionError) as caught:
                await aux.async_call_llm(
                    task="web_extract",
                    messages=[{"role": "user", "content": "extract"}],
                )
            assert caught.value is transport_error

        import asyncio

        asyncio.run(run())
    else:
        with pytest.raises(ConnectionError) as caught:
            aux.call_llm(
                task="web_extract",
                messages=[{"role": "user", "content": "extract"}],
            )
        assert caught.value is transport_error

    assert len(calls.calls) == expected_calls


def test_legacy_async_keeps_one_immediate_retry(monkeypatch):
    config = quota_config(0)
    del config["fallback_on"]
    transport_error = ConnectionError("connection reset")
    calls = AsyncCompletions(transport_error)
    install_sync_chain(monkeypatch, Client(calls), [], config)
    monkeypatch.setattr(aux, "_transient_retry_count", lambda: 6)

    with pytest.raises(ConnectionError) as caught:
        asyncio.run(
            aux.async_call_llm(
                task="web_extract",
                messages=[{"role": "user", "content": "extract"}],
            )
        )
    assert caught.value is transport_error
    assert len(calls.calls) == 2


def test_closed_async_retry_uses_sync_equivalent_backoff(monkeypatch):
    transport_error = ConnectionError("connection reset")
    calls = AsyncCompletions(transport_error)
    install_sync_chain(monkeypatch, Client(calls), [])
    monkeypatch.setattr(aux, "_transient_retry_count", lambda: 2)
    sleeps = []

    async def record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(aux.asyncio, "sleep", record_sleep)
    with pytest.raises(ConnectionError):
        asyncio.run(
            aux.async_call_llm(
                task="web_extract",
                messages=[{"role": "user", "content": "extract"}],
            )
        )
    assert sleeps == [aux._TRANSIENT_RETRY_BACKOFF_BASE]


def test_async_freezes_explicit_api_mode(monkeypatch):
    calls = AsyncCompletions(response())
    config = quota_config(0)
    observed = []
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

    def get_cached(*_args, **kwargs):
        observed.append(dict(kwargs))
        return Client(calls), "primary-model"

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)
    asyncio.run(
        aux.async_call_llm(
            task="web_extract",
            api_mode="codex_responses",
            messages=[{"role": "user", "content": "extract"}],
        )
    )
    assert observed[0]["api_mode"] == "codex_responses"


@pytest.mark.parametrize("async_mode", [False, True])
def test_closed_vision_consumes_frozen_route_without_legacy_reresolution(
    monkeypatch, async_mode
):
    calls = AsyncCompletions(response()) if async_mode else SyncCompletions(response())
    config = quota_config(0)
    config["api_mode"] = "chat_completions"
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
    monkeypatch.setattr(
        aux,
        "resolve_vision_provider_client",
        lambda *_args, **_kwargs: pytest.fail("closed vision route was re-resolved"),
    )
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda *_args, **_kwargs: (Client(calls), "primary-model"),
    )
    call_public(
        async_mode,
        task="vision",
        messages=[{"role": "user", "content": "inspect"}],
    )
    assert len(calls.calls) == 1


@pytest.mark.parametrize("async_mode", [False, True])
def test_compression_closed_chain_skips_known_small_context_candidate(
    monkeypatch, async_mode
):
    config = quota_config(2)
    config["fallback_chain"][0]["model"] = "known-small"
    config["fallback_chain"][1]["model"] = "unknown-custom"
    original = quota_error()
    primary_calls = (
        AsyncCompletions(original) if async_mode else SyncCompletions(original)
    )
    small_calls = (
        AsyncCompletions(response("small"))
        if async_mode
        else SyncCompletions(response("small"))
    )
    unknown_calls = (
        AsyncCompletions(response("unknown"))
        if async_mode
        else SyncCompletions(response("unknown"))
    )
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

    def get_cached(*_args, **kwargs):
        return {
            "https://primary.example/v1": (Client(primary_calls), "primary-model"),
            "https://backup-0.example/v1": (Client(small_calls), "known-small"),
            "https://backup-1.example/v1": (Client(unknown_calls), "unknown-custom"),
        }[kwargs.get("base_url")]

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)
    monkeypatch.setattr(
        aux,
        "_candidate_context_window",
        lambda _provider, model, **_kwargs: 32_000 if model == "known-small" else None,
    )
    result = call_public(
        async_mode,
        task="compression",
        messages=[{"role": "user", "content": "compress"}],
    )
    assert result.choices[0].message.content == "unknown"
    assert small_calls.calls == []
    assert len(unknown_calls.calls) == 1


@pytest.mark.parametrize("async_mode", [False, True])
def test_compression_all_known_small_routes_raise_the_original(monkeypatch, async_mode):
    config = quota_config(1)
    config["fallback_chain"][0]["model"] = "known-small"
    original = quota_error()
    primary_calls = (
        AsyncCompletions(original) if async_mode else SyncCompletions(original)
    )
    small_calls = (
        AsyncCompletions(response("must not run"))
        if async_mode
        else SyncCompletions(response("must not run"))
    )
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

    def get_cached(*_args, **kwargs):
        if kwargs.get("base_url") == "https://primary.example/v1":
            return Client(primary_calls), "primary-model"
        return Client(small_calls), "known-small"

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)
    monkeypatch.setattr(
        aux, "_candidate_context_window", lambda *_args, **_kwargs: 32_000
    )
    with pytest.raises(ProviderError) as caught:
        call_public(
            async_mode,
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
        )
    assert caught.value is original
    assert small_calls.calls == []


@pytest.mark.parametrize("async_mode", [False, True])
def test_closed_vision_skips_known_text_only_candidate(monkeypatch, async_mode):
    config = quota_config(2)
    config["fallback_chain"][0]["model"] = "text-only"
    config["fallback_chain"][1]["model"] = "vision-model"
    original = quota_error()
    primary_calls = (
        AsyncCompletions(original) if async_mode else SyncCompletions(original)
    )
    text_calls = (
        AsyncCompletions(response("must not run"))
        if async_mode
        else SyncCompletions(response("must not run"))
    )
    vision_calls = (
        AsyncCompletions(response("vision"))
        if async_mode
        else SyncCompletions(response("vision"))
    )
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
    monkeypatch.setattr(
        aux,
        "_main_model_supports_vision",
        lambda _provider, model: model != "text-only",
    )

    def get_cached(*_args, **kwargs):
        return {
            "https://primary.example/v1": (Client(primary_calls), "primary-model"),
            "https://backup-0.example/v1": (Client(text_calls), "text-only"),
            "https://backup-1.example/v1": (Client(vision_calls), "vision-model"),
        }[kwargs.get("base_url")]

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)
    result = call_public(
        async_mode,
        task="vision",
        messages=[{"role": "user", "content": "inspect"}],
    )
    assert result.choices[0].message.content == "vision"
    assert text_calls.calls == []
    assert len(vision_calls.calls) == 1


@pytest.mark.parametrize("async_mode", [False, True])
def test_closed_openai_codex_primary_preserves_auth_refresh(monkeypatch, async_mode):
    config = {
        **quota_config(0),
        "provider": "main",
    }
    auth_error = type("AuthenticationError", (ProviderError,), {})("expired token")
    first_calls = (
        AsyncCompletions(auth_error) if async_mode else SyncCompletions(auth_error)
    )
    refreshed_calls = (
        AsyncCompletions(response("refreshed"))
        if async_mode
        else SyncCompletions(response("refreshed"))
    )
    resolutions = []
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

    def get_cached(*_args, **_kwargs):
        resolutions.append(True)
        calls = first_calls if len(resolutions) == 1 else refreshed_calls
        return Client(calls), "gpt-refresh"

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)
    monkeypatch.setattr(
        aux,
        "_refresh_provider_credentials",
        lambda provider: provider == "openai-codex",
    )
    result = call_public(
        async_mode,
        task="web_extract",
        main_runtime={"provider": "openai-codex", "model": "gpt-refresh"},
        messages=[{"role": "user", "content": "extract"}],
    )
    assert result.choices[0].message.content == "refreshed"
    assert len(resolutions) == 2


@pytest.mark.parametrize("async_mode", [False, True])
def test_compression_timeout_skips_same_provider_retry(monkeypatch, async_mode):
    timeout_error = TimeoutError("timed out")
    calls = (
        AsyncCompletions(timeout_error)
        if async_mode
        else SyncCompletions(timeout_error)
    )
    install_sync_chain(monkeypatch, Client(calls), [])
    monkeypatch.setattr(aux, "_transient_retry_count", lambda: 3)

    if async_mode:

        async def run():
            with pytest.raises(TimeoutError) as caught:
                await aux.async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "compress"}],
                )
            assert caught.value is timeout_error

        import asyncio

        asyncio.run(run())
    else:
        with pytest.raises(TimeoutError) as caught:
            aux.call_llm(
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )
        assert caught.value is timeout_error

    assert len(calls.calls) == 1
