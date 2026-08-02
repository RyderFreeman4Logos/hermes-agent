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


def test_after_compression_requires_model_target():
    request = parse_model_switch_args("--after-compression")
    provider_only = parse_model_switch_args(
        "--after-compression --provider anthropic"
    )

    assert MODEL_SWITCH_ERR_AFTER_COMPRESSION_REQUIRES_TARGET in request.errors
    assert (
        MODEL_SWITCH_ERR_AFTER_COMPRESSION_REQUIRES_TARGET
        in provider_only.errors
    )


def test_after_compression_allows_explicit_session_scope():
    request = parse_model_switch_args("claude-sonnet-4 --after-compression --session")

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
