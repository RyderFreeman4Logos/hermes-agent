"""ACP test factories keep session identity without swallowing factory errors.

A factory that accepts the memory-mode keyword is called with session identity.
A factory that rejects that keyword is called without it, then the resulting
agent carries session_id, cwd, and model. A TypeError raised inside the factory
propagates. A zero-argument call is not a substitute for a rejected keyword.
"""
import pytest
from types import SimpleNamespace

from acp_adapter.session import SessionManager


def _call(factory, **kwargs):
    manager = SessionManager(agent_factory=factory)
    return manager._make_agent(session_id="sid", cwd="/workspace/kept", model="fixture", **kwargs)


def test_keyword_factory_receives_identity_and_mode():
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        if "memory_provider_mode_override" not in kwargs:
            raise TypeError("unexpected")
        return SimpleNamespace(model="from-factory")

    agent = _call(factory, memory_provider_mode_override="authoritative")

    assert seen == {
        "session_id": "sid",
        "cwd": "/workspace/kept",
        "model": "fixture",
        "memory_provider_mode_override": "authoritative",
    }
    assert agent.model == "from-factory"
    assert not hasattr(agent, "session_id")


def test_rejecting_keyword_passes_identity_without_the_keyword():
    seen = []

    def factory(session_id, cwd, model=None):
        seen.append((session_id, cwd, model))
        return SimpleNamespace(model="from-factory")

    agent = _call(factory, memory_provider_mode_override="authoritative")

    assert seen == [("sid", "/workspace/kept", "fixture")]
    assert agent.model == "from-factory"
    assert not hasattr(agent, "session_id")


def test_mode_only_factory_keeps_identity_on_the_agent():
    seen = []

    def factory(memory_provider_mode_override):
        seen.append(memory_provider_mode_override)
        return SimpleNamespace()

    agent = _call(factory, memory_provider_mode_override="hybrid")

    assert seen == ["hybrid"]
    assert agent.session_id == "sid"
    assert agent.cwd == "/workspace/kept"
    assert agent.model == "fixture"


def test_zero_arg_factory_keeps_identity_without_a_keyword_retry():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return SimpleNamespace(model="kept")

    agent = _call(factory, memory_provider_mode_override="authoritative")

    assert calls["n"] == 1
    assert agent.session_id == "sid"
    assert agent.cwd == "/workspace/kept"
    assert agent.model == "kept"


def test_factory_without_mode_is_not_called_again():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return SimpleNamespace(model="plain")

    agent = _call(factory)

    assert calls["n"] == 1
    assert agent.session_id == "sid"
    assert agent.cwd == "/workspace/kept"
    assert agent.model == "plain"


def test_internal_type_error_propagates():
    def factory(**_kwargs):
        raise TypeError("factory internal")

    with pytest.raises(TypeError, match="factory internal"):
        _call(factory, memory_provider_mode_override="authoritative")
