import sys
from types import ModuleType

import pytest

from acp_adapter.session import SessionManager
from agent.agent_init import (
    _RequestOverrideProjectionError,
    _project_request_overrides,
)


class NoopDb:
    pass


def _module(name, **attrs):
    module = ModuleType(name)
    vars(module).update(attrs)
    return module


def _install_runtime(monkeypatch, runtime, attempts, completed):
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda *_args: None)

    class ProjectingAgent:
        def __init__(self, **kwargs):
            attempts.append(kwargs)
            self.kwargs = kwargs
            self.provider = kwargs.get("provider", "")
            self.model = kwargs.get("model", "")
            self.base_url = kwargs.get("base_url", "")
            self.service_tier = None
            self._caller_request_overrides = kwargs.get("request_overrides") or {}
            self.request_overrides = _project_request_overrides(
                self,
                provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                service_tier=None,
                derived_overrides=kwargs.get("fast_mode_overrides"),
                custom_providers=[],
            )
            completed.append(self)

    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=ProjectingAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module(
            "hermes_cli.config",
            load_config=lambda: {"model": {"default": "model", "provider": "p"}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: runtime,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        _module(
            "hermes_cli.mcp_startup",
            ensure_mcp_discovery_before_agent_build=lambda **_kwargs: None,
        ),
    )


def _runtime(derived, pool):
    return {
        "provider": "custom",
        "requested_provider": "custom:p",
        "api_mode": "chat_completions",
        "base_url": "https://p.example/v1",
        "api_key": "key",
        "command": "provider-command",
        "args": ["--stdio"],
        "credential_pool": pool,
        "request_overrides": derived,
        "max_output_tokens": 4096,
    }


def test_acp_projects_resolved_runtime_through_shared_projector(monkeypatch):
    derived = {"extra_body": {"thinking": {"type": "enabled"}}}
    pool = object()
    attempts = []
    completed = []
    _install_runtime(monkeypatch, _runtime(derived, pool), attempts, completed)

    agent = SessionManager(db=NoopDb())._make_agent(session_id="acp-session", cwd=".")

    assert len(attempts) == 1
    assert completed == [agent]
    assert agent.kwargs["provider"] == "custom"
    assert agent.kwargs["requested_provider"] == "custom:p"
    assert agent.kwargs["api_mode"] == "chat_completions"
    assert agent.kwargs["base_url"] == "https://p.example/v1"
    assert agent.kwargs["api_key"] == "key"
    assert agent.kwargs["command"] == "provider-command"
    assert agent.kwargs["args"] == ["--stdio"]
    assert agent.kwargs["credential_pool"] is pool
    assert agent.kwargs["max_tokens"] == 4096
    assert agent.kwargs["request_overrides"] == {}
    assert agent.kwargs["fast_mode_overrides"] is derived
    assert agent.request_overrides == derived
    assert agent.request_overrides is not derived
    assert agent.request_overrides["extra_body"] is not derived["extra_body"]


def test_acp_hostile_runtime_rejects_without_copy_log_or_fallback(
    monkeypatch, caplog
):
    marker = "ACP_RUNTIME_PROJECTION_SECRET_MARKER"
    hooks = []
    attempts = []
    completed = []

    class Hostile:
        def __deepcopy__(self, memo):
            hooks.append(memo)
            raise RuntimeError(marker)

    derived = {"extra_body": {"hostile": Hostile()}}
    pool = object()
    _install_runtime(monkeypatch, _runtime(derived, pool), attempts, completed)

    with caplog.at_level("DEBUG", logger="acp_adapter.session"):
        with pytest.raises(_RequestOverrideProjectionError) as exc:
            SessionManager(db=NoopDb())._make_agent(session_id="acp-session", cwd=".")

    assert type(exc.value) is _RequestOverrideProjectionError
    assert str(exc.value) == "request override projection rejected"
    assert exc.value.__cause__ is None
    assert hooks == []
    assert len(attempts) == 1
    assert attempts[0]["provider"] == "custom"
    assert attempts[0]["requested_provider"] == "custom:p"
    assert attempts[0]["fast_mode_overrides"] is derived
    assert attempts[0]["request_overrides"] == {}
    assert completed == []
    for record in caplog.records:
        assert marker not in str(record.msg)
        assert marker not in repr(record.args)
        assert marker not in record.getMessage()
        assert record.exc_info is None
    assert marker not in caplog.text


def test_acp_resolution_failure_logs_static_message_without_exception(
    monkeypatch, caplog
):
    marker = "ACP_RESOLVER_SECRET_MARKER"
    attempts = []
    completed = []
    _install_runtime(monkeypatch, {}, attempts, completed)

    def fail_resolution(**_kwargs):
        raise RuntimeError(marker)

    sys.modules["hermes_cli.runtime_provider"].resolve_runtime_provider = fail_resolution

    with caplog.at_level("DEBUG", logger="acp_adapter.session"):
        agent = SessionManager(db=NoopDb())._make_agent(
            session_id="acp-session", cwd="."
        )

    assert completed == [agent]
    assert len(attempts) == 1
    assert "provider" not in attempts[0]
    records = [
        record
        for record in caplog.records
        if record.name == "acp_adapter.session"
    ]
    assert len(records) == 1
    assert records[0].msg == "ACP session using default provider resolution"
    assert records[0].args == ()
    assert records[0].exc_info is None
    assert marker not in records[0].getMessage()
    assert marker not in caplog.text
