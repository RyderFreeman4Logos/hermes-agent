"""Regression coverage for compression-gated delegate model-profile schemas."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import tools.delegate_tool as delegate_tool


def _delegate_definition() -> dict:
    overrides = delegate_tool._build_dynamic_schema_overrides()
    return {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "parameters": overrides["parameters"],
        },
    }


def _profile_enum(definition: dict) -> list[str] | None:
    return (
        definition["function"]["parameters"]["properties"]
        ["model_profile"].get("enum")
    )


@pytest.fixture
def model_pool_snapshot(monkeypatch):
    # The process-global snapshot is intentional in production: it freezes the
    # schema until a completed compaction. Reset it only to isolate this test.
    monkeypatch.setattr(delegate_tool, "_MODEL_POOL_SCHEMA_NAMES", None)
    yield


def test_model_pool_enum_changes_only_after_compression_refresh(
    monkeypatch, model_pool_snapshot
):
    config = {"model_pool": {"fast": {}}}
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: config)

    initial = _delegate_definition()
    assert _profile_enum(initial) == ["fast"]

    # A normal config edit must not churn the schema (and thus the prompt cache).
    config["model_pool"] = {"smart": {}}
    assert _profile_enum(_delegate_definition()) == ["fast"]

    agent = SimpleNamespace(tools=[initial])
    assert delegate_tool.refresh_model_pool_schema_after_compression(agent) is True
    assert _profile_enum(agent.tools[0]) == ["smart"]
    # Future tool-definition rebuilds now use the freshly committed snapshot.
    assert _profile_enum(_delegate_definition()) == ["smart"]


def test_unchanged_model_pool_is_a_noop_after_compression(
    monkeypatch, model_pool_snapshot
):
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"model_pool": {"fast": {}}})
    initial = _delegate_definition()
    tools = [initial]
    agent = SimpleNamespace(tools=tools)

    assert delegate_tool.refresh_model_pool_schema_after_compression(agent) is False
    assert agent.tools is tools
