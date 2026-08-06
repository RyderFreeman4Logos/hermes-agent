"""Shared parser contract for deferred /model switches."""

from hermes_cli.model_switch import (
    MODEL_SWITCH_ERR_AFTER_COMPRESSION_REQUIRES_TARGET,
    MODEL_SWITCH_ERR_AFTER_COMPRESSION_WITH_GLOBAL,
    MODEL_SWITCH_ERR_AFTER_COMPRESSION_WITH_ONCE,
    MODEL_SWITCH_ERROR_TEXT,
    parse_model_flags_detailed,
    parse_model_switch_args,
    switch_model,
)


def test_after_compression_parses_target_and_provider():
    request = parse_model_switch_args(
        "claude-sonnet-4 --after-compression --provider anthropic"
    )

    assert request.target == "claude-sonnet-4"
    assert request.explicit_provider == "anthropic"
    assert request.is_after_compression is True
    assert request.scope == "after_compression"
    assert request.errors == ()
    assert parse_model_flags_detailed(request.raw).is_after_compression is True


def test_after_compression_rejects_immediate_scopes():
    request = parse_model_switch_args(
        "claude-sonnet-4 --after-compression --once --global"
    )

    assert MODEL_SWITCH_ERR_AFTER_COMPRESSION_WITH_ONCE in request.errors
    assert MODEL_SWITCH_ERR_AFTER_COMPRESSION_WITH_GLOBAL in request.errors
    assert (
        MODEL_SWITCH_ERROR_TEXT[MODEL_SWITCH_ERR_AFTER_COMPRESSION_WITH_ONCE]
        == "/model --after-compression cannot be combined with --once"
    )
    assert (
        MODEL_SWITCH_ERROR_TEXT[MODEL_SWITCH_ERR_AFTER_COMPRESSION_WITH_GLOBAL]
        == "/model --after-compression cannot be combined with --global"
    )


def test_after_compression_requires_model_target_or_provider():
    request = parse_model_switch_args("--after-compression")
    provider_only = parse_model_switch_args(
        "--after-compression --provider anthropic"
    )

    assert MODEL_SWITCH_ERR_AFTER_COMPRESSION_REQUIRES_TARGET in request.errors
    assert provider_only.errors == ()
    assert provider_only.target == ""
    assert provider_only.explicit_provider == "anthropic"


def test_after_compression_allows_explicit_session_scope():
    request = parse_model_switch_args("claude-sonnet-4 --after-compression --session")

    assert request.is_after_compression is True
    assert request.is_session is True
    assert request.scope == "after_compression"
    assert request.errors == ()


def _pm_config(*, with_models=True):
    provider = {
        "base_url": "https://pm.invalid/v1",
        "api_key": "test-token",
        "api_mode": "codex_responses",
        "discover_models": False,
    }
    if with_models:
        provider.update(
            {
                "default_model": "current-model",
                "models": {
                    "current-model": {"context_length": 262_144},
                    "gpt-5.6-sol": {"context_length": 1_048_576},
                },
            }
        )
    return {"providers": {"pm": provider}}


def _install_network_tripwires(monkeypatch):
    calls = []

    def fail(name):
        def trip(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(name)

        return trip

    monkeypatch.setattr("agent.models_dev.fetch_models_dev", fail("models.dev"))
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", fail("metadata"))
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", fail("capabilities"))
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", fail("model info"))
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fail("provider /models"))
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", fail("provider catalog"))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._auto_detect_local_model",
        fail("local /models"),
    )
    monkeypatch.setattr("socket.getaddrinfo", fail("DNS"))
    return calls


def _install_copilot_catalog_tripwire(monkeypatch):
    requests = []

    def trip(request, **_kwargs):
        requests.append((request.full_url, request.get_header("Authorization")))
        raise AssertionError("Copilot catalog network request")

    monkeypatch.setattr("hermes_cli.models._urlopen_model_catalog_request", trip)
    return requests


def _install_copilot_runtime(monkeypatch):
    config = {
        "model": {"provider": "copilot"},
        "providers": {
            "copilot": {
                "base_url": "https://api.githubcopilot.com",
                "api_key": "copilot-secret",
            }
        },
    }
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)
    return config


def test_deferred_copilot_switch_never_fetches_catalog_or_forwards_token(monkeypatch):
    requests = _install_copilot_catalog_tripwire(monkeypatch)
    config = _install_copilot_runtime(monkeypatch)

    result = switch_model(
        raw_input="gpt-5.4",
        explicit_provider="copilot",
        current_provider="openai-codex",
        current_model="gpt-5.3-codex",
        user_providers=config["providers"],
        validate_live=False,
    )

    assert result.success is True
    assert result.api_mode == "codex_responses"
    assert requests == []


def test_deferred_unknown_copilot_model_fails_before_scheduling(monkeypatch):
    requests = _install_copilot_catalog_tripwire(monkeypatch)
    config = _install_copilot_runtime(monkeypatch)

    result = switch_model(
        raw_input="copilot-future-unknown",
        explicit_provider="copilot",
        current_provider="openai-codex",
        current_model="gpt-5.3-codex",
        user_providers=config["providers"],
        validate_live=False,
    )

    assert result.success is False
    assert "local" in result.error_message.lower()
    assert requests == []


def test_live_copilot_switch_still_uses_injected_catalog(monkeypatch):
    calls = []
    config = _install_copilot_runtime(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.models.fetch_github_model_catalog",
        lambda api_key=None, **kwargs: calls.append((api_key, kwargs)) or [
            {"id": "gpt-5.4", "capabilities": {"type": "chat"}}
        ],
    )
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *_args: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *_args: None)

    result = switch_model(
        raw_input="gpt-5.4",
        explicit_provider="copilot",
        current_provider="openai-codex",
        current_model="gpt-5.3-codex",
        user_providers=config["providers"],
        validate_live=True,
    )

    assert result.success is True
    assert result.api_mode == "codex_responses"
    assert calls
    assert all(api_key == "copilot-secret" for api_key, _kwargs in calls)


def test_deferred_exact_configured_model_is_strictly_local(monkeypatch):
    config = _pm_config()
    calls = _install_network_tripwires(monkeypatch)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)

    result = switch_model(
        raw_input="gpt-5.6-sol",
        explicit_provider="pm",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=config["providers"],
        validate_live=False,
    )

    assert result.success is True
    assert (result.target_provider, result.new_model) == ("pm", "gpt-5.6-sol")
    assert calls == []


def test_deferred_provider_only_uses_configured_default_and_context(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli.model_switch import (
        get_model_switch_after_compression,
        schedule_model_switch_after_compression,
    )

    config = _pm_config()
    calls = _install_network_tripwires(monkeypatch)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)

    result = switch_model(
        raw_input="",
        explicit_provider="pm",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=config["providers"],
        validate_live=False,
    )
    agent = SimpleNamespace(
        model="gpt-5.4",
        provider="openai-codex",
        api_mode="chat_completions",
    )
    schedule_model_switch_after_compression(agent, result)

    pending = get_model_switch_after_compression(agent)
    assert pending is not None
    assert result.success is True
    assert (pending.target_provider, pending.new_model) == ("pm", "current-model")
    assert (pending.base_url, pending.api_mode) == (
        "https://pm.invalid/v1",
        "codex_responses",
    )
    assert pending.context_length == 262_144
    assert (agent.model, agent.provider) == ("gpt-5.4", "openai-codex")
    assert calls == []


def test_deferred_true_space_rejects_before_resolution(monkeypatch):
    calls = []

    def trip(*_args, **_kwargs):
        calls.append("metadata")
        raise AssertionError("metadata")

    monkeypatch.setattr("hermes_cli.model_switch.resolve_provider_full", trip)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", trip)

    result = switch_model(
        raw_input="gpt-5.6 sol",
        explicit_provider="pm",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        validate_live=False,
    )

    assert result.success is False
    assert "spaces" in result.error_message
    assert calls == []


def test_deferred_unknown_custom_model_fails_closed_without_network(monkeypatch):
    config = _pm_config(with_models=False)
    calls = _install_network_tripwires(monkeypatch)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)

    result = switch_model(
        raw_input="gpt-5.6-sol",
        explicit_provider="pm",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=config["providers"],
        validate_live=False,
    )

    assert result.success is False
    assert "local" in result.error_message.lower()
    assert calls == []


def test_deferred_custom_current_direct_alias_propagates_local_only_policy(monkeypatch):
    from hermes_cli import model_switch as ms

    config = _pm_config()
    calls = _install_network_tripwires(monkeypatch)

    def provider_info(_provider, *, allow_network=True):
        calls.append("provider registry")
        raise AssertionError("provider registry")

    monkeypatch.setattr("agent.models_dev.get_provider_info", provider_info)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)
    monkeypatch.setattr(
        ms,
        "DIRECT_ALIASES",
        {
            "next": ms.DirectAlias(
                model="gpt-5.6-sol",
                provider="pm",
                base_url="",
            )
        },
    )
    monkeypatch.setattr(
        "agent.credential_pool.write_credential_pool",
        lambda *_a, **_k: calls.append("credential pool write"),
    )
    monkeypatch.setattr(
        "hermes_cli.auth._save_auth_store",
        lambda *_a, **_k: calls.append("auth store write"),
    )

    result = switch_model(
        raw_input="next",
        current_provider="custom:current",
        current_model="old-model",
        user_providers=config["providers"],
        validate_live=False,
    )

    assert result.success is True
    assert (result.target_provider, result.new_model) == ("pm", "gpt-5.6-sol")
    assert result.warning_message == ""
    assert calls == []


def test_deferred_local_auto_correction_queues_warning(monkeypatch):
    from types import SimpleNamespace

    from hermes_cli.model_switch import (
        get_model_switch_after_compression,
        schedule_model_switch_after_compression,
    )

    config = {
        "providers": {
            "openai-codex": {
                "api_key": "test-token",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_mode": "codex_responses",
                "models": {"gpt-5.3-codex": {"context_length": 400_000}},
            }
        }
    }
    calls = _install_network_tripwires(monkeypatch)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "openai-codex",
            "api_key": "test-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
        },
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)

    result = switch_model(
        raw_input="gpt5.3-codex",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        current_base_url="https://chatgpt.com/backend-api/codex",
        current_api_key="test-token",
        user_providers=config["providers"],
        validate_live=False,
    )
    assert result.success, result.error_message
    agent = SimpleNamespace(
        model="gpt-5.4",
        provider="openai-codex",
        api_mode="codex_responses",
    )
    schedule_model_switch_after_compression(agent, result)

    pending = get_model_switch_after_compression(agent)
    assert pending is not None
    assert pending.new_model == "gpt-5.3-codex"
    assert result.warning_message == (
        "Auto-corrected `gpt5.3-codex` → `gpt-5.3-codex`"
    )
    assert calls == []


def test_deferred_provider_alias_auth_enumeration_is_read_only(monkeypatch):
    calls = []

    def provider_info(_provider, *, allow_network=True):
        calls.append("provider registry")
        raise AssertionError("provider registry")

    class EmptyPool:
        def has_credentials(self):
            return False

    def load_pool(_provider, *, read_only=False):
        if not read_only:
            calls.append("writable credential pool")
        return EmptyPool()

    monkeypatch.setattr("agent.models_dev.get_provider_info", provider_info)
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda **_kwargs: calls.append("models.dev") or {},
    )
    monkeypatch.setattr("agent.credential_pool.load_pool", load_pool)
    monkeypatch.setattr(
        "agent.credential_pool.write_credential_pool",
        lambda *_a, **_k: calls.append("credential pool write"),
    )
    monkeypatch.setattr(
        "hermes_cli.auth._save_auth_store",
        lambda *_a, **_k: calls.append("auth store write"),
    )

    result = switch_model(
        raw_input="gpt-5.6-sol",
        explicit_provider="openai",
        current_provider="custom:current",
        current_model="old-model",
        validate_live=False,
    )

    assert result.success is False
    assert "no credentials configured" in result.error_message
    assert calls == []


def test_deferred_missing_local_api_mode_fails_closed(monkeypatch):
    config = _pm_config()
    config["providers"]["pm"].pop("api_mode")
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "pm",
            "api_key": "test-token",
            "base_url": "https://pm.invalid/v1",
            "api_mode": "",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.determine_api_mode",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("deferred scheduling must not guess an API mode")
        ),
    )

    result = switch_model(
        raw_input="gpt-5.6-sol",
        explicit_provider="pm",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        user_providers=config["providers"],
        validate_live=False,
    )

    assert result.success is False
    assert "api mode" in result.error_message.lower()
