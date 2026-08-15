from __future__ import annotations

import copy
import threading
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


GLM_URL = "https://api.z.ai/api/coding/paas/v4"
OPENAI_URL = "https://api.openai.com/v1"
CALLER = {
    "extra_headers": {"X-Caller": "keep"},
    "extra_body": {"caller_flag": {"nested": True}},
}
FAST = {"service_tier": "priority", "speed": "fast"}
GLM_DERIVED = {"thinking": {"type": "enabled"}}


def _custom_providers():
    return [
        {
            "provider_key": "glm",
            "base_url": GLM_URL,
            "model": "glm-5.2",
            "extra_body": GLM_DERIVED,
        }
    ]


@pytest.fixture
def route_env():
    config = {"custom_providers": _custom_providers()}
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_custom_providers(),
        ),
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        yield


def _agent(*, fallback_model=None):
    return AIAgent(
        api_key="openai-key",
        base_url=OPENAI_URL,
        provider="openai",
        model="gpt-5.4-mini",
        service_tier="priority",
        request_overrides=CALLER,
        fast_mode_overrides=FAST,
        fallback_model=fallback_model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _switch_to_glm(agent):
    agent.switch_model(
        new_model="glm-5.2",
        new_provider="custom:glm",
        api_key="glm-key",
        base_url=GLM_URL,
        api_mode="chat_completions",
    )


def test_constructor_separates_caller_and_derived_overrides(route_env):
    explicit = copy.deepcopy(CALLER)
    agent = _agent()

    assert agent._caller_request_overrides == explicit
    assert agent.request_overrides == {**FAST, **explicit}
    assert agent._primary_runtime["request_overrides"] == agent.request_overrides

    CALLER["extra_body"]["caller_flag"]["nested"] = False
    try:
        assert agent._caller_request_overrides == explicit
    finally:
        CALLER["extra_body"]["caller_flag"]["nested"] = True


def test_switch_rebuilds_destination_overrides_and_preserves_caller(route_env):
    agent = _agent()

    _switch_to_glm(agent)

    assert agent._caller_request_overrides == CALLER
    assert agent.request_overrides == {
        "extra_headers": {"X-Caller": "keep"},
        "extra_body": {
            "thinking": {"type": "enabled"},
            "caller_flag": {"nested": True},
        },
    }
    assert "service_tier" not in agent.request_overrides
    assert "speed" not in agent.request_overrides
    assert agent._primary_runtime["request_overrides"] == agent.request_overrides


def test_failed_switch_restores_nested_override_state(route_env):
    agent = _agent()
    _switch_to_glm(agent)
    original = copy.deepcopy(agent.request_overrides)
    original_primary = copy.deepcopy(agent._primary_runtime)
    agent._create_openai_client = MagicMock(side_effect=RuntimeError("client failed"))

    with pytest.raises(RuntimeError, match="client failed"):
        agent.switch_model(
            new_model="gpt-5.4-mini",
            new_provider="openai",
            api_key="openai-key",
            base_url=OPENAI_URL,
            api_mode="chat_completions",
        )

    assert agent.request_overrides == original
    assert agent._primary_runtime == original_primary
    agent.request_overrides["extra_body"]["thinking"]["type"] = "mutated"
    assert original["extra_body"]["thinking"]["type"] == "enabled"


def test_fallback_and_restore_rebuild_route_overrides(route_env):
    fallback_client = MagicMock(base_url=GLM_URL, api_key="glm-key")
    fallback_client._custom_headers = {}
    fallback_client.default_headers = {}
    agent = _agent(
        fallback_model=[
            {
                "provider": "custom:glm",
                "model": "glm-5.2",
                "base_url": GLM_URL,
                "api_key": "glm-key",
                "api_mode": "chat_completions",
            }
        ]
    )

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fallback_client, "glm-5.2"),
    ):
        assert agent._try_activate_fallback() is True
        assert agent.request_overrides["extra_body"] == {
            "thinking": {"type": "enabled"},
            "caller_flag": {"nested": True},
        }
        assert agent._restore_primary_runtime() is True

    assert agent.request_overrides == {"service_tier": "priority", **CALLER}


class _RecordingAgent:
    captured = None

    def __init__(self, **kwargs):
        type(self).captured = kwargs
        self.session_id = "child"
        self._session_init_model_config = {}


def test_delegate_sibling_preserves_provenance_without_aliasing(route_env, monkeypatch):
    import run_agent
    from tools.delegate_tool import _build_child_agent

    parent = _agent()
    parent.enabled_toolsets = []
    parent.disabled_toolsets = []
    parent._delegate_depth = 1
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.prefill_messages = None
    parent._session_db = None
    parent.session_id = "parent"
    monkeypatch.setattr(run_agent, "AIAgent", _RecordingAgent)
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {})

    _build_child_agent(
        task_index=0,
        goal="check overrides",
        context=None,
        toolsets=None,
        model=None,
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
    )

    kwargs = _RecordingAgent.captured
    assert kwargs["request_overrides"] == CALLER
    assert kwargs["fast_mode_overrides"] == {**FAST, **CALLER}
    assert kwargs["request_overrides"] is not parent._caller_request_overrides
    assert kwargs["fast_mode_overrides"] is not parent.request_overrides


def test_tui_fast_toggle_cannot_overwrite_explicit_named_scalars(
    route_env, monkeypatch
):
    import tui_gateway.server as server

    explicit = {**CALLER, "service_tier": "flex", "speed": "fast"}
    agent = AIAgent(
        api_key="openai-key",
        base_url=OPENAI_URL,
        provider="openai",
        model="gpt-5.4-mini",
        service_tier="priority",
        request_overrides=explicit,
        fast_mode_overrides=FAST,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    sid = "request-override-provenance"
    monkeypatch.setitem(server._sessions, sid, {"agent": agent, "session_key": sid})
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)

    response = server._methods["config.set"](
        "rid", {"key": "fast", "session_id": sid, "value": "normal"}
    )

    assert response["result"]["value"] == "normal"
    assert agent._caller_request_overrides == explicit
    assert agent.request_overrides == explicit
