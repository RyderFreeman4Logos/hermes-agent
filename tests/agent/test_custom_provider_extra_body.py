from types import SimpleNamespace

import pytest

from agent.agent_init import (
    _RequestOverrideProjectionError,
    _compose_request_overrides,
    _project_request_overrides,
    _request_override_projections,
)


def _agent(*, caller=None):
    return SimpleNamespace(
        provider="custom:zai-coding-plan",
        model="glm-5.2",
        base_url="https://api.z.ai/api/coding/paas/v4",
        service_tier=None,
        _caller_request_overrides=caller or {},
        request_overrides={"safe": True},
    )


def _providers(extra_body):
    return [
        {
            "provider_key": "other-provider",
            "base_url": "https://api.z.ai/api/coding/paas/v4",
            "model": "glm-5.2",
            "extra_body": {"wrong": True},
        },
        {
            "provider_key": "zai-coding-plan",
            "base_url": "https://api.z.ai/api/coding/paas/v4/",
            "model": "glm-5.2",
            "extra_body": extra_body,
        },
    ]


def test_custom_provider_projection_preserves_precedence_and_detaches():
    configured = {
        "enable_thinking": True,
        "reasoning_effort": "high",
        "nested": {"items": ["configured"]},
    }
    caller = {
        "extra_body": {"reasoning_effort": "low", "caller_only": True}
    }
    derived = {"extra_body": {"derived_only": True}}
    agent = _agent(caller=caller)

    projected = _project_request_overrides(
        agent,
        provider=agent.provider,
        model=agent.model,
        base_url=agent.base_url,
        service_tier=None,
        derived_overrides=derived,
        custom_providers=_providers(configured),
    )

    assert projected["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
        "nested": {"items": ["configured"]},
        "derived_only": True,
        "caller_only": True,
    }
    projected["extra_body"]["nested"]["items"].append("active")
    assert configured["nested"]["items"] == ["configured"]
    assert caller == {
        "extra_body": {"reasoning_effort": "low", "caller_only": True}
    }
    assert agent.request_overrides == {"safe": True}


def test_compose_publishes_once_after_cycle_safe_projection():
    cycle = {}
    cycle["self"] = cycle
    agent = _agent(caller={"cycle": cycle})

    result = _compose_request_overrides(
        agent,
        {"speed": "fast"},
        custom_providers=_providers({"enable_thinking": False}),
    )

    assert result is None
    assert agent.request_overrides["cycle"] is not cycle
    assert agent.request_overrides["cycle"]["self"] is agent.request_overrides["cycle"]
    assert agent.request_overrides["extra_body"] == {"enable_thinking": False}
    assert agent.request_overrides["speed"] == "fast"


def test_hostile_custom_body_rejects_without_hook_or_publication():
    calls = []

    class Hostile:
        def __deepcopy__(self, _memo):
            calls.append("copy")
            raise RuntimeError("private payload")

    agent = _agent()
    safe = agent.request_overrides

    with pytest.raises(
        _RequestOverrideProjectionError,
        match="^request override projection rejected$",
    ) as caught:
        _compose_request_overrides(
            agent,
            {},
            custom_providers=_providers({"hostile": Hostile()}),
        )

    assert caught.value.__cause__ is None
    assert calls == []
    assert agent.request_overrides is safe


def test_request_override_provenance_reads_only_owned_instance_state():
    class DynamicParent:
        def __init__(self):
            self.lookups = 0

        def __getattr__(self, _name):
            self.lookups += 1
            return {"forged": True}

    parent = DynamicParent()
    assert _request_override_projections(parent) == ({}, {})
    assert parent.lookups == 0

    parent.__dict__["_caller_request_overrides"] = object()
    with pytest.raises(_RequestOverrideProjectionError):
        _request_override_projections(parent)
