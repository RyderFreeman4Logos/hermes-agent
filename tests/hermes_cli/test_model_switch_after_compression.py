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


def test_deferred_resolution_runs_local_validation_without_provider_discovery(
    monkeypatch,
):
    def fail_network_discovery(*_args, **_kwargs):
        raise AssertionError("network model discovery")

    def cache_only_models_dev(*_args, **kwargs):
        assert kwargs.get("allow_network") is False
        return {}

    monkeypatch.setattr(
        "hermes_cli.models.provider_model_ids",
        fail_network_discovery,
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        fail_network_discovery,
    )
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        cache_only_models_dev,
    )

    invalid = switch_model(
        raw_input="qwen3.5-4b",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        current_base_url="https://chatgpt.com/backend-api/codex",
        current_api_key="test-token",
        validate_live=False,
    )
    valid = switch_model(
        raw_input="gpt-5.4",
        current_provider="openai-codex",
        current_model="gpt-5.3-codex",
        current_base_url="https://chatgpt.com/backend-api/codex",
        current_api_key="test-token",
        validate_live=False,
    )

    assert invalid.success is False
    assert "doesn't look like" in invalid.error_message
    assert valid.success is True
    assert valid.new_model == "gpt-5.4"
