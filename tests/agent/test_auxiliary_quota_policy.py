"""Focused contract tests for opt-in quota-only auxiliary fallback."""

from __future__ import annotations

import asyncio
import json
import sys
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

    def test_bool_status_is_invalid_and_fails_closed(self):
        error = ProviderError(
            status_code=True,
            body={"error": {"code": "quota_exhausted"}},
        )
        assert not error_classifier.is_explicit_usage_quota_exhaustion(error)

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

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize(
        ("message", "surface"),
        [
            ("Rate limit exceeded", "same_envelope"),
            ("Too many requests", "nested_details"),
            ("ordinary throttling", "causal_sibling"),
        ],
    )
    def test_conflicting_throttling_message_vetoes_structured_quota_fallback(
        self, monkeypatch, async_mode, message, surface
    ):
        error_body: dict[str, object] = {"code": "quota_exhausted"}
        body = {"error": error_body}
        if surface == "same_envelope":
            error_body["message"] = message
        elif surface == "nested_details":
            error_body["details"] = {"message": message}
        original = ProviderError("provider error", status_code=429, body=body)
        if surface == "causal_sibling":
            original.__cause__ = ProviderError(message)
        primary_calls = (
            AsyncCompletions(original) if async_mode else SyncCompletions(original)
        )
        backup_calls = (
            AsyncCompletions(response("must not run"))
            if async_mode
            else SyncCompletions(response("must not run"))
        )
        install_sync_chain(
            monkeypatch,
            Client(primary_calls),
            [Client(backup_calls)],
        )

        with pytest.raises(ProviderError) as caught:
            call_public(
                async_mode,
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )

        assert caught.value is original
        assert len(primary_calls.calls) == 1
        assert backup_calls.calls == []

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("surface", ["direct", "nested", "causal_sibling"])
    def test_bare_generic_quota_exceeded_never_authorizes_fallback(
        self, monkeypatch, async_mode, surface
    ):
        marker = {"code": "quota_exceeded"}
        body = {"error": marker if surface == "direct" else {"details": marker}}
        original = ProviderError("provider error", status_code=429, body=body)
        if surface == "causal_sibling":
            setattr(original, "body", {})
            original.__cause__ = ProviderError(body={"error": marker})
        primary_calls = (
            AsyncCompletions(original) if async_mode else SyncCompletions(original)
        )
        backup_calls = (
            AsyncCompletions(response("must not run"))
            if async_mode
            else SyncCompletions(response("must not run"))
        )
        install_sync_chain(
            monkeypatch,
            Client(primary_calls),
            [Client(backup_calls)],
        )

        with pytest.raises(ProviderError) as caught:
            call_public(
                async_mode,
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
    def test_closed_plan_normalizes_all_three_route_sources(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda _provider: {
                "base_url": "https://named.example/v1",
                "api_key": "named-key",
                "model": "named-default",
            },
        )
        config = quota_config(0)
        config["model"] = "auto"
        config["fallback_chain"] = [
            {"provider": "custom:backup", "model": "auto"},
            {"provider": "main"},
        ]

        snapshot = policy.capture_closed_plan(
            "compression",
            config,
            main_runtime={
                "provider": "openai",
                "model": "gpt-main",
                "api_key": "main-key",
            },
        )

        assert snapshot is not None
        assert snapshot.primary is not None
        assert snapshot.primary.model == "gpt-main"
        assert snapshot.candidates[0].model == "named-default"
        assert (
            snapshot.candidates[1].provider,
            snapshot.candidates[1].base_url,
        ) == ("custom", "https://api.openai.com/v1")

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_public_primary_normalizes_auto_model_and_openai_alias(
        self, monkeypatch, async_mode
    ):
        config = quota_config(0)
        config.update(provider="openai", model="auto", api_key="openai-key")
        config.pop("base_url")
        calls = (
            AsyncCompletions(response("ok"))
            if async_mode
            else SyncCompletions(response("ok"))
        )
        resolutions = []
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

        def get_cached(provider, model, **kwargs):
            resolutions.append((provider, model, dict(kwargs)))
            return Client(calls, str(kwargs.get("base_url") or "")), "gpt-runtime-default"

        monkeypatch.setattr(aux, "_get_cached_client", get_cached)
        call_public(
            async_mode,
            task="web_extract",
            messages=[{"role": "user", "content": "extract"}],
        )

        provider, model, client_kwargs = resolutions[0]
        assert (provider, model) == ("custom", "gpt-4o-mini")
        assert client_kwargs["base_url"] == "https://api.openai.com/v1"
        assert client_kwargs["api_key"] == "openai-key"
        assert calls.calls[0]["model"] == "gpt-runtime-default"

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_public_candidate_auto_model_uses_frozen_named_default(
        self, monkeypatch, async_mode
    ):
        config = quota_config(0)
        config["fallback_chain"] = [
            {"provider": "custom:backup", "model": "auto"}
        ]
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda _provider: {
                "base_url": "https://named.example/v1",
                "api_key": "named-key",
                "model": "named-default",
            },
        )
        original = quota_error()
        primary_calls = (
            AsyncCompletions(original) if async_mode else SyncCompletions(original)
        )
        backup_calls = (
            AsyncCompletions(response("fallback"))
            if async_mode
            else SyncCompletions(response("fallback"))
        )
        resolutions = []
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

        def get_cached(provider, model, **kwargs):
            resolutions.append((provider, model, dict(kwargs)))
            if kwargs.get("base_url") == "https://primary.example/v1":
                return Client(primary_calls), "primary-model"
            return Client(backup_calls, str(kwargs.get("base_url") or "")), model

        monkeypatch.setattr(aux, "_get_cached_client", get_cached)
        result = call_public(
            async_mode,
            task="web_extract",
            messages=[{"role": "user", "content": "extract"}],
        )

        assert result.choices[0].message.content == "fallback"
        assert resolutions[1][1] == "named-default"
        assert backup_calls.calls[0]["model"] == "named-default"

    @pytest.mark.parametrize("sink", ["task_config", "main_runtime"])
    def test_closed_plan_rejects_cycles_with_policy_error(self, sink):
        config = quota_config(0)
        runtime = {}
        cyclic = {}
        cyclic["self"] = cyclic
        if sink == "task_config":
            config["extra_body"] = cyclic
        else:
            runtime["nested"] = cyclic

        with pytest.raises(ValueError, match="cycle"):
            policy.capture_closed_plan(
                "compression",
                config,
                main_runtime=runtime,
            )

    def test_closed_plan_rejects_excessive_snapshot_depth(self):
        nested = []
        for _ in range(34):
            nested = [nested]
        config = quota_config(0)
        config["extra_body"] = nested

        with pytest.raises(ValueError, match="depth limit"):
            policy.capture_closed_plan("compression", config)

    def test_closed_plan_rejects_excessive_snapshot_nodes(self):
        config = quota_config(0)
        config["extra_body"] = {"items": [None] * 10_000}

        with pytest.raises(ValueError, match="node limit"):
            policy.capture_closed_plan("compression", config)

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize(
        "invalid_payload",
        [
            {"value": b"not-json"},
            {"value": float("nan")},
            {"nested": {1: "non-string-key"}},
        ],
    )
    def test_public_payload_rejects_non_json_values_before_client_resolution(
        self, monkeypatch, async_mode, invalid_payload
    ):
        config = quota_config(0)
        config["extra_body"] = invalid_payload
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
        monkeypatch.setattr(
            aux,
            "_get_cached_client",
            lambda *_args, **_kwargs: pytest.fail("invalid payload reached client"),
        )

        with pytest.raises(ValueError):
            call_public(
                async_mode,
                task="web_extract",
                messages=[{"role": "user", "content": "extract"}],
            )

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_nested_extra_body_is_json_payload_and_detached_from_config(
        self, monkeypatch, async_mode
    ):
        configured_extra_body = {
            "reasoning": {
                "enabled": True,
                "effort": "high",
                "metadata": {
                    "steps": [1, {"label": "preserved"}],
                    "alternatives": ["first"],
                },
            }
        }
        config = quota_config(0)
        config["extra_body"] = configured_extra_body
        calls = (
            AsyncCompletions(response("ok"))
            if async_mode
            else SyncCompletions(response("ok"))
        )
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
        monkeypatch.setattr(
            aux,
            "_get_cached_client",
            lambda *_args, **_kwargs: (Client(calls), "primary-model"),
        )

        call_public(
            async_mode,
            task="web_extract",
            messages=[{"role": "user", "content": "extract"}],
        )

        payload = calls.calls[0]["extra_body"]
        assert payload == configured_extra_body
        json.dumps(payload)
        payload["reasoning"]["metadata"]["steps"][1]["label"] = "provider-mutated"
        payload["reasoning"]["metadata"]["alternatives"].append("provider-added")
        assert configured_extra_body["reasoning"]["metadata"] == {
            "steps": [1, {"label": "preserved"}],
            "alternatives": ["first"],
        }

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
        with pytest.raises(RuntimeError, match="no concrete primary"):
            call_public(
                async_mode,
                task="web_extract",
                messages=[{"role": "user", "content": "extract"}],
                **override,
            )
        assert observed == []
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
            assert snapshot is not None
            assert snapshot.candidates == ()
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

    def test_candidate_freezes_callable_credential_identity_without_repr_leak(self):
        def credential():
            return "late-secret"

        snapshot = policy.capture_closed_plan(
            "compression",
            {
                "fallback_on": ["quota_exhausted"],
                "fallback_chain": [
                    {
                        "provider": "custom",
                        "model": "backup",
                        "base_url": "https://backup.example/v1",
                        "api_key": credential,
                    }
                ],
            },
        )
        assert snapshot is not None
        assert snapshot.candidates[0].api_key is credential
        assert "late-secret" not in repr(snapshot)

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
    config["fallback_chain"][0]["provider"] = "custom:small"
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
    context_probes = []

    def candidate_context_window(provider, model, **kwargs):
        context_probes.append((provider, kwargs.get("api_key")))
        return 32_000 if model == "known-small" else None

    monkeypatch.setattr(aux, "_candidate_context_window", candidate_context_window)
    result = call_public(
        async_mode,
        task="compression",
        messages=[{"role": "user", "content": "compress"}],
    )
    assert result.choices[0].message.content == "unknown"
    assert context_probes == [
        ("custom", "backup-key-0"),
        ("custom", "backup-key-1"),
    ]
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
@pytest.mark.parametrize(
    ("candidate_supports_vision", "ambient_supports_vision"),
    [(True, False), (False, True)],
)
def test_named_custom_vision_candidate_uses_frozen_capability(
    monkeypatch,
    async_mode,
    candidate_supports_vision,
    ambient_supports_vision,
):
    config = quota_config(0)
    config["supports_vision"] = True
    config["fallback_chain"] = [
        {
            "provider": "custom:backup",
            "model": "backup-vision-model",
            "supports_vision": candidate_supports_vision,
        }
    ]
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_named_custom_provider",
        lambda _provider: {
            "base_url": "https://named-backup.example/v1",
            "api_key": "named-backup-key",
        },
    )
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {
                "provider": "custom:active",
                "default": "active-model",
                "supports_vision": ambient_supports_vision,
            }
        },
    )
    original = quota_error("primary quota")
    primary_calls = (
        AsyncCompletions(original) if async_mode else SyncCompletions(original)
    )
    backup_calls = (
        AsyncCompletions(response("vision backup"))
        if async_mode
        else SyncCompletions(response("vision backup"))
    )

    def get_cached(*_args, **kwargs):
        if kwargs.get("base_url") == "https://primary.example/v1":
            return Client(primary_calls), "primary-model"
        return Client(backup_calls), "backup-vision-model"

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)

    if candidate_supports_vision:
        result = call_public(
            async_mode,
            task="vision",
            messages=[{"role": "user", "content": "inspect"}],
        )
        assert result.choices[0].message.content == "vision backup"
    else:
        with pytest.raises(ProviderError) as caught:
            call_public(
                async_mode,
                task="vision",
                messages=[{"role": "user", "content": "inspect"}],
            )
        assert caught.value is original

    assert len(primary_calls.calls) == 1
    assert len(backup_calls.calls) == int(candidate_supports_vision)


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
        lambda _provider, model, **_kwargs: model != "text-only",
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
def test_closed_openai_codex_primary_without_binding_fails_capture(monkeypatch, async_mode):
    config = {
        **quota_config(0),
        "provider": "main",
    }
    resolutions = []
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

    def get_cached(*_args, **_kwargs):
        resolutions.append(True)
        raise AssertionError("closed capture must reject before client resolution")

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)
    monkeypatch.setattr(aux, "_refresh_provider_credentials", lambda _provider: False)
    with pytest.raises(RuntimeError, match="no concrete primary"):
        call_public(
            async_mode,
            task="web_extract",
            main_runtime={"provider": "openai-codex", "model": "gpt-refresh"},
            messages=[{"role": "user", "content": "extract"}],
        )
    assert resolutions == []


@pytest.mark.parametrize("async_mode", [False, True])
def test_closed_main_anthropic_does_not_ambient_refresh(
    monkeypatch, async_mode
):
    config = {**quota_config(0), "provider": "main"}
    auth_error = type("AuthenticationError", (ProviderError,), {})("expired token")
    expired_calls = (
        AsyncCompletions(auth_error) if async_mode else SyncCompletions(auth_error)
    )
    resolutions = []
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

    def get_cached(*_args, **kwargs):
        resolutions.append(kwargs.get("api_key"))
        return Client(expired_calls, "https://api.anthropic.com"), "claude-refresh"

    def refresh(_provider):
        raise AssertionError("closed route must not ambient refresh")

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)
    monkeypatch.setattr(aux, "_refresh_provider_credentials", refresh)
    with pytest.raises(type(auth_error)) as caught:
        call_public(
            async_mode,
            task="web_extract",
            main_runtime={
                "provider": "anthropic",
                "model": "claude-refresh",
                "base_url": "https://api.anthropic.com",
                "api_key": "expired-anthropic-token",
            },
            messages=[{"role": "user", "content": "extract"}],
        )
    assert caught.value is auth_error
    assert resolutions == ["expired-anthropic-token"]


@pytest.mark.parametrize("async_mode", [False, True])
def test_closed_explicit_anthropic_route_never_refreshes(
    monkeypatch, async_mode
):
    config = {
        **quota_config(0),
        "provider": "anthropic",
        "model": "claude-frozen",
        "base_url": "https://api.anthropic.com",
        "api_key": "frozen-anthropic-token",
    }
    auth_error = type("AuthenticationError", (ProviderError,), {})("refreshable")
    first_calls = (
        AsyncCompletions(auth_error) if async_mode else SyncCompletions(auth_error)
    )

    resolutions = []
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

    def get_cached(*_args, **kwargs):
        resolutions.append(kwargs.get("api_key"))
        return Client(first_calls, "https://api.anthropic.com"), "claude-frozen"

    monkeypatch.setattr(aux, "_get_cached_client", get_cached)
    monkeypatch.setattr(
        aux,
        "_refresh_provider_credentials",
        lambda _provider: (_ for _ in ()).throw(
            AssertionError("closed route must not ambient refresh")
        ),
    )
    with pytest.raises(type(auth_error)) as caught:
        call_public(
            async_mode,
            task="web_extract",
            messages=[{"role": "user", "content": "extract"}],
        )

    assert caught.value is auth_error
    assert resolutions == ["frozen-anthropic-token"]


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


class TestQ3ClassARegression:
    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize(
        "authority",
        [
            "quota exhausted",
            "QUOTA_EXHAUSTED",
            "quota-exhausted",
            "quotaexhausted",
            "insufficient quota",
            "usage-limit-reached",
        ],
    )
    def test_public_accepts_only_bounded_generic_authority_variants(
        self, monkeypatch, async_mode, authority
    ):
        original = ProviderError(
            "provider error",
            status_code=429,
            body={"error": {"code": authority}},
        )
        primary_calls = (
            AsyncCompletions(original) if async_mode else SyncCompletions(original)
        )
        backup_calls = (
            AsyncCompletions(response("fallback"))
            if async_mode
            else SyncCompletions(response("fallback"))
        )
        install_sync_chain(monkeypatch, Client(primary_calls), [Client(backup_calls)])
        result = call_public(
            async_mode,
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
        )
        assert result.choices[0].message.content == "fallback"

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize(
        "conflict",
        [
            "authentication error",
            "AUTHENTICATION_ERROR",
            "authentication-error",
            "authenticationerror",
            "model not found",
            "service_unavailable",
            "network-failure",
            "validationerror",
            "context length exceeded",
            "payload-too-large",
            "rate_limit_exceeded",
            "toomanyrequests",
        ],
    )
    def test_public_negative_classes_veto_quota_across_separator_variants(
        self, monkeypatch, async_mode, conflict
    ):
        original = ProviderError(
            "provider error",
            status_code=429,
            body={
                "error": {
                    "code": "quota_exhausted",
                    "details": {"message": conflict},
                }
            },
        )
        primary_calls = AsyncCompletions(original) if async_mode else SyncCompletions(original)
        backup_calls = (
            AsyncCompletions(response("must not run"))
            if async_mode
            else SyncCompletions(response("must not run"))
        )
        install_sync_chain(monkeypatch, Client(primary_calls), [Client(backup_calls)])
        with pytest.raises(ProviderError) as caught:
            call_public(
                async_mode,
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )
        assert caught.value is original
        assert backup_calls.calls == []

    @pytest.mark.parametrize(
        "authority",
        [
            "quota.exhausted",
            "quota/exhausted",
            "quota:exhausted",
            "quota__exhausted",
            "_quota_exhausted",
            "quota_exhausted_",
            "quota\u200bexhausted",
            "quota_exhausted🙂",
        ],
    )
    def test_generic_authority_rejects_forbidden_punctuation(self, authority):
        assert not error_classifier.is_explicit_usage_quota_exhaustion(
            ProviderError(body={"error": {"code": authority}})
        )

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_status_must_share_the_authority_component(self, monkeypatch, async_mode):
        original = ProviderError(body={"error": {"code": "quota_exhausted"}})
        original.__cause__ = ProviderError("sibling", status_code=429)
        primary_calls = AsyncCompletions(original) if async_mode else SyncCompletions(original)
        backup_calls = (
            AsyncCompletions(response("must not run"))
            if async_mode
            else SyncCompletions(response("must not run"))
        )
        install_sync_chain(monkeypatch, Client(primary_calls), [Client(backup_calls)])
        with pytest.raises(ProviderError) as caught:
            call_public(
                async_mode,
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )
        assert caught.value is original
        assert backup_calls.calls == []

    def test_xai_403_cannot_be_borrowed_from_a_causal_sibling(self):
        original = ProviderError(
            body={"error": {"code": "personal-team-blocked:spending-limit"}}
        )
        original.__context__ = ProviderError(status_code=403)
        assert not error_classifier.is_explicit_usage_quota_exhaustion(original)

    def test_exception_details_has_mapping_parity(self):
        assert error_classifier.is_explicit_usage_quota_exhaustion(
            ProviderError(details={"error": {"reason": "usage_limit_reached"}})
        )
        assert not error_classifier.is_explicit_usage_quota_exhaustion(
            ProviderError(
                details={
                    "error": {
                        "reason": "usage_limit_reached",
                        "message": "authentication failure",
                    }
                }
            )
        )

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_public_nonquota_candidate_diagnostic_graph_is_acyclic(
        self, monkeypatch, async_mode
    ):
        original = quota_error("primary quota")
        candidate = ProviderError("candidate transport failure")
        primary_calls = AsyncCompletions(original) if async_mode else SyncCompletions(original)
        candidate_calls = AsyncCompletions(candidate) if async_mode else SyncCompletions(candidate)
        install_sync_chain(monkeypatch, Client(primary_calls), [Client(candidate_calls)])
        with pytest.raises(ProviderError) as caught:
            call_public(
                async_mode,
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )
        assert caught.value is original
        assert original.__cause__ is candidate
        seen = set()
        current = original
        for _ in range(8):
            if current is None:
                break
            assert id(current) not in seen
            seen.add(id(current))
            current = current.__cause__ or current.__context__


class TestQ3ClassBCredentialBinding:
    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("provider", ["openrouter", "anthropic", "xai"])
    def test_public_builtin_primary_without_captured_key_fails_before_resolution(
        self, monkeypatch, async_mode, provider
    ):
        config = {
            "provider": provider,
            "model": "explicit-model",
            "fallback_on": ["quota_exhausted"],
            "fallback_chain": [
                {
                    "provider": "custom",
                    "model": "backup",
                    "base_url": "https://backup.example/v1",
                    "api_key": "backup-key",
                }
            ],
        }
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
        monkeypatch.setattr(
            aux,
            "_get_cached_client",
            lambda *_args, **_kwargs: pytest.fail("ambient resolver was reached"),
        )
        for name in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(RuntimeError, match="no concrete primary"):
            call_public(
                async_mode,
                task="web_extract",
                messages=[{"role": "user", "content": "extract"}],
            )

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_missing_key_env_candidate_is_rejected_without_ambient_resolution(
        self, monkeypatch, async_mode
    ):
        config = quota_config(0)
        config["fallback_chain"] = [
            {
                "provider": "custom",
                "model": "backup",
                "base_url": "https://backup.example/v1",
                "key_env": "ABSENT_BACKUP_KEY",
            }
        ]
        monkeypatch.delenv("ABSENT_BACKUP_KEY", raising=False)
        original = quota_error()
        primary_calls = AsyncCompletions(original) if async_mode else SyncCompletions(original)
        resolutions = []
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

        def get_cached(*_args, **kwargs):
            resolutions.append(dict(kwargs))
            if kwargs.get("base_url") != "https://primary.example/v1":
                pytest.fail("candidate with missing key_env reached resolution")
            return Client(primary_calls), "primary-model"

        monkeypatch.setattr(aux, "_get_cached_client", get_cached)
        with pytest.raises(ProviderError) as caught:
            call_public(
                async_mode,
                task="compression",
                messages=[{"role": "user", "content": "compress"}],
            )
        assert caught.value is original
        assert len(resolutions) == 1

    def test_remote_custom_requires_auth_but_explicit_loopback_is_no_auth(self):
        remote = policy.capture_closed_plan(
            "compression",
            {
                "fallback_on": ["quota_exhausted"],
                "fallback_chain": [
                    {
                        "provider": "custom",
                        "model": "remote",
                        "base_url": "https://remote.example/v1",
                    }
                ],
            },
        )
        local = policy.capture_closed_plan(
            "compression",
            {
                "fallback_on": ["quota_exhausted"],
                "fallback_chain": [
                    {
                        "provider": "custom",
                        "model": "local",
                        "base_url": "http://127.0.0.1:8000/v1",
                    }
                ],
            },
        )
        assert remote.candidates == ()
        assert local.candidates[0].credential.kind == "no_auth"
        assert "no-key-required" not in repr(local.candidates[0])

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_closed_resolution_is_strict_and_bound_to_captured_credential(
        self, monkeypatch, async_mode
    ):
        config = quota_config(0)
        calls = AsyncCompletions(response("ok")) if async_mode else SyncCompletions(response("ok"))
        observed = []
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

        def get_cached(*_args, **kwargs):
            observed.append(dict(kwargs))
            return Client(calls), "primary-model"

        monkeypatch.setattr(aux, "_get_cached_client", get_cached)
        call_public(
            async_mode,
            task="web_extract",
            messages=[{"role": "user", "content": "extract"}],
        )
        assert observed[0]["strict_runtime"] is True
        assert observed[0]["api_key"] == "primary-key"
        assert observed[0]["credential_identity"]
        assert "primary-key" not in observed[0]["credential_identity"]

    def test_callable_binding_is_preserved_and_secret_safe(self):
        class TokenProvider:
            def __call__(self):
                return "CALLABLE-RAW-TOKEN"

            def __repr__(self):
                return "TokenProvider(CALLABLE-RAW-TOKEN)"

        provider = TokenProvider()
        snapshot = policy.capture_closed_plan(
            "compression",
            {
                "provider": "custom",
                "model": "primary",
                "base_url": "https://callable.example/v1",
                "api_key": provider,
                "fallback_on": ["quota_exhausted"],
                "fallback_chain": [
                    {
                        "provider": "custom",
                        "model": "backup",
                        "base_url": "https://backup.example/v1",
                        "api_key": "backup-key",
                    }
                ],
            },
        )
        assert snapshot.primary.api_key is provider
        assert snapshot.primary.credential.kind == "callable"
        assert "CALLABLE-RAW-TOKEN" not in repr(snapshot)
        assert "CALLABLE-RAW-TOKEN" not in snapshot.primary.credential.identity


class TestQ3ClassCControlsAndCapabilities:
    @pytest.mark.parametrize("async_mode", [False, True])
    def test_strict_openrouter_resolution_uses_only_frozen_headers(
        self, monkeypatch, async_mode
    ):
        constructed = []

        class StrictClient:
            def __init__(self, **kwargs):
                constructed.append(dict(kwargs))
                self.api_key = kwargs["api_key"]
                self.base_url = kwargs["base_url"]

        monkeypatch.setattr(
            aux,
            "_create_openai_client",
            lambda **kwargs: StrictClient(**kwargs),
        )
        monkeypatch.setattr(sys.modules["openai"], "AsyncOpenAI", StrictClient)
        monkeypatch.setattr(
            aux,
            "build_or_headers",
            lambda *_args, **_kwargs: pytest.fail("strict route rebuilt ambient headers"),
        )
        monkeypatch.setattr(
            aux,
            "_apply_user_default_headers",
            lambda *_args, **_kwargs: pytest.fail("strict route loaded ambient headers"),
        )

        client, model = aux.resolve_provider_client(
            "openrouter",
            "vendor/frozen-model",
            async_mode=async_mode,
            explicit_base_url="https://openrouter.ai/api/v1",
            explicit_api_key="FROZEN-OPENROUTER-KEY",
            strict_runtime=True,
            default_headers={
                "HTTP-Referer": "https://frozen.example",
                "Authorization": "Bearer FROZEN-HEADER-TOKEN",
            },
        )

        assert isinstance(client, StrictClient)
        assert model == "vendor/frozen-model"
        assert constructed[-1]["default_headers"] == {
            "HTTP-Referer": "https://frozen.example",
            "Authorization": "Bearer FROZEN-HEADER-TOKEN",
        }

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_named_candidate_freezes_route_local_controls(self, monkeypatch, async_mode):
        named = {
            "base_url": "https://named.example/v1",
            "api_key": "named-key",
            "model": "named-model",
            "extra_headers": {"X-Frozen": "header-a"},
            "extra_body": {"provider_default": "a", "precedence": "provider"},
            "max_output_tokens": 321,
            "supports_vision": True,
            "context_length": 131072,
        }
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            lambda _provider: dict(named),
        )
        config = quota_config(0)
        config["fallback_chain"] = [
            {
                "provider": "custom:backup",
                "model": "named-model",
                "timeout": 17,
                "extra_body": {"precedence": "route", "route": "b"},
                "extra_headers": {"X-Route": "header-b"},
                "max_output_tokens": 222,
            }
        ]
        original = quota_error()
        primary_calls = AsyncCompletions(original) if async_mode else SyncCompletions(original)
        backup_calls = (
            AsyncCompletions(response("frozen"))
            if async_mode
            else SyncCompletions(response("frozen"))
        )
        resolutions = []
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

        def get_cached(*_args, **kwargs):
            resolutions.append(dict(kwargs))
            if kwargs.get("base_url") == "https://primary.example/v1":
                named.update(
                    extra_headers={"X-Mutated": "later"},
                    max_output_tokens=999,
                    supports_vision=False,
                    context_length=1,
                )
                return Client(primary_calls), "primary-model"
            return Client(backup_calls), "named-model"

        monkeypatch.setattr(aux, "_get_cached_client", get_cached)
        result = call_public(
            async_mode,
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
        )
        assert result.choices[0].message.content == "frozen"
        candidate_resolution = resolutions[1]
        assert candidate_resolution["default_headers"] == {
            "X-Frozen": "header-a",
            "X-Route": "header-b",
        }
        assert candidate_resolution["strict_runtime"] is True
        assert backup_calls.calls[0]["timeout"] == 17
        assert backup_calls.calls[0]["max_tokens"] == 222
        assert backup_calls.calls[0]["extra_body"] == {
            "provider_default": "a",
            "precedence": "route",
            "route": "b",
            "snapshot": "a",
        }

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_primary_scoped_controls_do_not_leak_to_candidate(self, monkeypatch, async_mode):
        config = quota_config(1)
        config.update(
            extra_headers={"X-Primary": "only"},
            max_output_tokens=111,
            request_overrides={"extra_body": {"primary": "only"}},
        )
        original = quota_error()
        primary_calls = AsyncCompletions(original) if async_mode else SyncCompletions(original)
        backup_calls = (
            AsyncCompletions(response("backup"))
            if async_mode
            else SyncCompletions(response("backup"))
        )
        resolutions = []
        monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)

        def get_cached(*_args, **kwargs):
            resolutions.append(dict(kwargs))
            return (
                (Client(primary_calls), "primary-model")
                if len(resolutions) == 1
                else (Client(backup_calls), "backup-model")
            )

        monkeypatch.setattr(aux, "_get_cached_client", get_cached)
        result = call_public(
            async_mode,
            task="web_extract",
            messages=[{"role": "user", "content": "extract"}],
        )
        assert result.choices[0].message.content == "backup"
        assert resolutions[0]["default_headers"] == {"X-Primary": "only"}
        assert resolutions[1].get("default_headers") in (None, {})
        assert "max_tokens" not in backup_calls.calls[0]
        assert backup_calls.calls[0].get("extra_body") == {"snapshot": "a"}

    def test_frozen_route_repr_hides_header_and_credential_values(self):
        snapshot = policy.capture_closed_plan(
            "compression",
            {
                **quota_config(0),
                "extra_headers": {"Authorization": "Bearer RAW-HEADER-SECRET"},
            },
        )
        rendered = repr(snapshot)
        assert "primary-key" not in rendered
        assert "RAW-HEADER-SECRET" not in rendered
