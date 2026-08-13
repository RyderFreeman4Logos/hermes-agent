"""Runtime heartbeat provider identity regressions."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("provider", "canonical"),
    [("pm", "custom:pm"), ("localrouter", "custom:localrouter")],
)
def test_runtime_heartbeat_canonicalizes_named_custom_aliases(
    monkeypatch, provider, canonical
):
    from tools.runtime_heartbeat import canonical_runtime_provider_identity

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.canonical_custom_identity",
        lambda **_kwargs: canonical,
    )

    assert canonical_runtime_provider_identity(
        SimpleNamespace(
            provider=provider,
            requested_provider=provider,
            base_url="https://proxy.invalid/v1",
            model="test-model",
        )
    ) == canonical


def test_child_admission_reuses_pool_for_provider_alias(monkeypatch):
    from tools import delegate_tool

    pool = object()
    parent = SimpleNamespace(
        provider="custom",
        requested_provider="custom:pm",
        base_url="https://pm.invalid/v1",
        model="test-model",
        _credential_pool=pool,
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.canonical_custom_identity",
        lambda **_kwargs: "custom:pm",
    )
    monkeypatch.setattr(
        "agent.credential_pool.get_custom_provider_pool_key",
        lambda *_args: "custom:pm",
    )

    assert delegate_tool._resolve_child_credential_pool(
        "pm", parent, parent.base_url, "pm"
    ) is pool


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("pm", "custom:pm"), ("localrouter", "custom:localrouter")],
)
def test_child_credential_resolution_keeps_named_custom_identity(
    monkeypatch, alias, canonical
):
    from tools import delegate_tool

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "custom",
            "base_url": "https://proxy.example/v1",
            "api_key": "test-key",
            "model": "test-model",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.canonical_custom_identity",
        lambda **_kwargs: canonical,
    )

    assert delegate_tool._resolve_delegation_credentials(
        {"provider": alias}, SimpleNamespace()
    )["provider"] == canonical
