"""Focused parser and resolver checks for deferred model switches."""

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
    assert MODEL_SWITCH_ERR_AFTER_COMPRESSION_REQUIRES_TARGET in request.errors

    provider_only = parse_model_switch_args(
        "--after-compression --provider anthropic"
    )
    assert provider_only.errors == ()


def test_after_compression_allows_reasoning_only_request():
    request = parse_model_switch_args("--after-compression --reasoning low")

    assert request.target == ""
    assert request.reasoning == "low"
    assert request.errors == ()


def test_after_compression_keeps_explicit_reasoning_for_model_target():
    request = parse_model_switch_args(
        "grok-4.6 --after-compression --reasoning low"
    )

    assert request.target == "grok-4.6"
    assert request.reasoning == "low"
    assert request.errors == ()


def test_invalid_reasoning_is_rejected_before_deferred_scheduling():
    request = parse_model_switch_args(
        "grok-4.6 --after-compression --reasoning definitely-not-valid"
    )

    assert request.errors


def test_provider_only_after_compression_uses_configured_default(monkeypatch):
    config = {
        "providers": {
            "pm": {
                "base_url": "https://pm.invalid/v1",
                "api_key": "[REDACTED]",
                "api_mode": "codex_responses",
                "default_model": "current-model",
                "models": {"current-model": {"context_length": 262_144}},
            }
        }
    }
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *_a: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities", lambda *_a: None
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_a, **_k: {"accepted": True, "persist": True, "recognized": True},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._auto_detect_local_model",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("provider-only deferred switch must use configured default")
        ),
    )

    result = switch_model(
        raw_input="",
        explicit_provider="pm",
        current_provider="openai-codex",
        current_model="old-model",
        user_providers=config["providers"],
        validate_live=False,
    )

    assert result.success, result.error_message
    assert (result.target_provider, result.new_model) == ("pm", "current-model")


def test_after_compression_allows_explicit_session_scope():
    request = parse_model_switch_args(
        "claude-sonnet-4 --after-compression --session"
    )
    assert request.is_after_compression is True
    assert request.is_session is True
    assert request.scope == "after_compression"
    assert request.errors == ()


def test_deferred_resolution_does_not_probe_provider(monkeypatch):
    def fail_live_validation(*_args, **_kwargs):
        raise AssertionError("live provider validation")

    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        fail_live_validation,
    )

    result = switch_model(
        raw_input="openai/gpt-4o-mini",
        current_provider="openrouter",
        current_model="old-model",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="secret",
        validate_live=False,
    )

    assert result.success
    assert result.new_model == "openai/gpt-4o-mini"
