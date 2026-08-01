"""Regression coverage for request overrides on live model switches."""

import copy
import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from agent.transports.chat_completions import ChatCompletionsTransport
from run_agent import AIAgent


GLM_URL = "https://api.z.ai/api/coding/paas/v4"
OPENAI_URL = "https://api.openai.com/v1"
CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
PM_URL = "https://pm.example/v1"
CALLER_OVERRIDES = {
    "extra_headers": {"X-Caller": "keep"},
    "extra_body": {"caller_flag": True},
}
GLM_THINKING = {"thinking": {"type": "enabled"}}
FAST_MODE_OVERRIDES = {"service_tier": "priority", "speed": "fast"}


def _agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.model = "glm-5.2"
    agent.provider = "custom:glm"
    agent.requested_provider = agent.provider
    agent.base_url = GLM_URL
    agent.api_key = "glm-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="glm-client")
    agent._client_kwargs = {"api_key": agent.api_key, "base_url": agent.base_url}
    agent._caller_request_overrides = copy.deepcopy(CALLER_OVERRIDES)
    agent.request_overrides = copy.deepcopy(CALLER_OVERRIDES)
    agent.request_overrides["extra_body"].update(GLM_THINKING)
    agent.context_compressor = None
    agent.service_tier = None
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    agent._transport_cache = {}
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None
    agent._consecutive_stale_streams = 0
    agent._create_openai_client = MagicMock(return_value=MagicMock())
    agent._apply_client_headers_for_base_url = MagicMock()
    agent._ensure_lmstudio_runtime_loaded = MagicMock(return_value=None)
    agent._lmstudio_load_was_unverified = MagicMock(return_value=False)
    agent._effective_lmstudio_context_length = MagicMock(return_value=None)
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    return agent


def _configs():
    return [
        {
            "provider_key": "glm",
            "base_url": GLM_URL,
            "model": "glm-5.2",
            "extra_body": GLM_THINKING,
        },
        {
            "provider_key": "pm",
            "base_url": PM_URL,
            "model": "gpt-5.4-mini",
            "extra_body": {"pm_target": True},
        },
    ]


@pytest.fixture
def runtime_override_env():
    config = {"custom_providers": _configs()}
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_configs(),
        ),
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        yield


def _openai_fast_agent(explicit_overrides):
    return AIAgent(
        api_key="openai-key",
        base_url=OPENAI_URL,
        provider="openai",
        model="gpt-5.4-mini",
        service_tier="priority",
        request_overrides=explicit_overrides,
        fast_mode_overrides=FAST_MODE_OVERRIDES,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _openai_session_agent(*, initial_fast, fallback_model=None):
    return AIAgent(
        api_key="openai-key",
        base_url=OPENAI_URL,
        provider="openai",
        model="gpt-5.4-mini",
        request_overrides=CALLER_OVERRIDES,
        fast_mode_overrides={"service_tier": "priority"} if initial_fast else None,
        service_tier="priority" if initial_fast else None,
        fallback_model=fallback_model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _set_tui_fast(monkeypatch, agent, *, enabled):
    import tui_gateway.server as server

    session_id = "request-overrides-live-fast"
    monkeypatch.setitem(
        server._sessions,
        session_id,
        {"agent": agent, "session_key": session_id},
    )
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    response = server._methods["config.set"](
        "rid-fast",
        {
            "key": "fast",
            "session_id": session_id,
            "value": "fast" if enabled else "normal",
        },
    )
    assert response["result"]["value"] == ("fast" if enabled else "normal")


def _switch_to_glm(agent):
    agent.switch_model(
        new_model="glm-5.2",
        new_provider="custom:glm",
        api_key="glm-key",
        base_url=GLM_URL,
        api_mode="chat_completions",
    )


def _switch(agent, *, provider, base_url):
    agent.switch_model(
        new_model="gpt-5.4-mini",
        new_provider=provider,
        api_key="gpt-key",
        base_url=base_url,
        api_mode="codex_responses",
    )


class _ImmediateThread:
    def __init__(self, *, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


class _RecordingAgent:
    captured = None

    def __init__(self, **kwargs):
        type(self).captured = kwargs
        self._session_messages = []

    def run_conversation(self, **_kwargs):
        return {"final_response": "done"}

    def shutdown_memory_provider(self):
        pass

    def close(self):
        pass


def _capture_sibling_kwargs(monkeypatch, parent, sibling):
    import run_agent

    _RecordingAgent.captured = None
    monkeypatch.setattr(run_agent, "AIAgent", _RecordingAgent)

    if sibling.startswith("tui_"):
        from tui_gateway import server

        sid = f"request-overrides-{sibling}"
        session = {
            "agent": parent,
            "session_key": "parent-session",
            "history": [],
            "history_lock": threading.Lock(),
            "cwd": ".",
        }
        monkeypatch.setitem(server._sessions, sid, session)
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"max_turns": 25}})
        monkeypatch.setattr(server, "_get_db", lambda: None)
        monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
        method = "prompt.background" if sibling == "tui_background" else "preview.restart"
        params = {"session_id": sid, "text": "work"}
        if sibling == "tui_preview":
            params = {"session_id": sid, "url": "http://127.0.0.1:3000"}
        response = server._methods[method]("rid", params)
        assert "result" in response
    elif sibling == "delegate":
        from tools.delegate_tool import _build_child_agent

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
    else:
        monkeypatch.setattr(run_agent.threading, "Thread", _ImmediateThread)
        parent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
        )

    assert _RecordingAgent.captured is not None
    return _RecordingAgent.captured


def _assert_sibling_switch_contract(parent, kwargs):
    explicit = copy.deepcopy(parent._caller_request_overrides)
    parent_composed = copy.deepcopy(parent.request_overrides)
    child = AIAgent(**kwargs)

    assert child._caller_request_overrides == explicit
    assert child.request_overrides == parent_composed

    child._caller_request_overrides["extra_body"]["caller_flag"]["nested"] = False
    child.request_overrides["extra_headers"]["X-Caller"] = "changed"
    assert parent._caller_request_overrides == explicit
    assert parent.request_overrides == parent_composed
    child._caller_request_overrides = copy.deepcopy(explicit)
    child.request_overrides = copy.deepcopy(parent_composed)

    _switch_to_glm(child)

    expected = {
        **explicit,
        "extra_body": {**GLM_THINKING, **explicit["extra_body"]},
    }
    assert child.request_overrides == expected
    wire_kwargs = ChatCompletionsTransport().build_kwargs(
        model=child.model,
        messages=[{"role": "user", "content": "test"}],
        request_overrides=child.request_overrides,
    )
    assert wire_kwargs["extra_headers"] == explicit["extra_headers"]
    assert wire_kwargs["extra_body"] == expected["extra_body"]
    assert wire_kwargs["timeout"] == explicit["timeout"]
    assert "service_tier" not in wire_kwargs
    assert "speed" not in wire_kwargs


@pytest.mark.parametrize(
    "sibling",
    ["tui_background", "tui_preview", "delegate", "background_review"],
)
def test_sibling_agents_keep_caller_and_fast_override_provenance_separate(
    runtime_override_env,
    monkeypatch,
    sibling,
):
    explicit = {
        "extra_headers": {"X-Caller": "keep"},
        "extra_body": {"caller_flag": {"nested": True}},
        "timeout": 17,
    }
    parent = _openai_fast_agent(explicit)
    parent.enabled_toolsets = ["file"]

    kwargs = _capture_sibling_kwargs(monkeypatch, parent, sibling)

    _assert_sibling_switch_contract(parent, kwargs)


@pytest.mark.parametrize(
    "sibling",
    ["tui_background", "tui_preview", "delegate", "background_review"],
)
def test_sibling_agents_preserve_explicit_fast_named_scalars(
    runtime_override_env,
    monkeypatch,
    sibling,
):
    explicit = {
        **CALLER_OVERRIDES,
        "service_tier": "flex",
        "speed": "fast",
    }
    parent = _openai_fast_agent(explicit)
    parent.enabled_toolsets = ["file"]

    child = AIAgent(**_capture_sibling_kwargs(monkeypatch, parent, sibling))
    _switch_to_glm(child)

    assert child._caller_request_overrides == explicit
    assert child.request_overrides == {
        **explicit,
        "extra_body": {**GLM_THINKING, **explicit["extra_body"]},
    }


def test_delegate_differing_model_rebuilds_fast_overrides(runtime_override_env):
    from tools.delegate_tool import _build_child_agent

    parent = _openai_fast_agent(CALLER_OVERRIDES)
    child = _build_child_agent(
        task_index=0,
        goal="different-model child",
        context=None,
        toolsets=None,
        model="gpt-5.3-codex",
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
    )

    wire = ChatCompletionsTransport().build_kwargs(
        model=child.model,
        messages=[{"role": "user", "content": "hi"}],
        request_overrides=child.request_overrides,
    )
    assert child._caller_request_overrides == CALLER_OVERRIDES
    assert child.request_overrides == CALLER_OVERRIDES
    assert "service_tier" not in wire
    assert "speed" not in wire


def test_delegate_provider_overrides_drop_on_child_fallback(runtime_override_env):
    from tools.delegate_tool import _build_child_agent

    parent = _openai_fast_agent(CALLER_OVERRIDES)
    fallback_client = MagicMock(base_url=OPENAI_URL, api_key="openai-key")
    fallback_client._custom_headers = {}
    fallback_client.default_headers = {}

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fallback_client, "gpt-5.4-mini"),
    ):
        child = _build_child_agent(
            task_index=0,
            goal="provider child with fallback",
            context=None,
            toolsets=None,
            model="glm-5.2",
            max_iterations=1,
            task_count=1,
            parent_agent=parent,
            override_provider="custom:glm",
            override_base_url=GLM_URL,
            override_api_key="glm-key",
            override_api_mode="chat_completions",
            override_request_overrides={"extra_body": copy.deepcopy(GLM_THINKING)},
            override_fallback_chain=[
                {
                    "provider": "openai",
                    "model": "gpt-5.4-mini",
                    "base_url": OPENAI_URL,
                    "api_key": "openai-key",
                    "api_mode": "chat_completions",
                }
            ],
        )
        before_fallback = copy.deepcopy(child.request_overrides)
        assert child._try_activate_fallback() is True

    wire = ChatCompletionsTransport().build_kwargs(
        model=child.model,
        messages=[{"role": "user", "content": "hi"}],
        request_overrides=child.request_overrides,
    )
    assert before_fallback == {
        "extra_headers": {"X-Caller": "keep"},
        "extra_body": {"thinking": {"type": "enabled"}, "caller_flag": True},
    }
    assert child._caller_request_overrides == CALLER_OVERRIDES
    assert child.request_overrides == CALLER_OVERRIDES
    assert wire["extra_body"] == {"caller_flag": True}
    assert "thinking" not in wire["extra_body"]


def test_runtime_fast_overrides_do_not_cross_switches_or_reach_glm_wire(
    runtime_override_env,
):
    explicit = {
        "extra_headers": {"X-Caller": "keep"},
        "extra_body": {"caller_flag": {"nested": True}},
        "timeout": 17,
    }
    agent = _openai_fast_agent(explicit)

    assert agent._caller_request_overrides == explicit
    assert agent.request_overrides == {**FAST_MODE_OVERRIDES, **explicit}

    _switch_to_glm(agent)

    expected_glm = {
        **explicit,
        "extra_body": {**GLM_THINKING, **explicit["extra_body"]},
    }
    assert agent.request_overrides == expected_glm
    wire_kwargs = ChatCompletionsTransport().build_kwargs(
        model=agent.model,
        messages=[{"role": "user", "content": "test"}],
        request_overrides=agent.request_overrides,
    )
    assert wire_kwargs["extra_headers"] == explicit["extra_headers"]
    assert wire_kwargs["extra_body"] == expected_glm["extra_body"]
    assert wire_kwargs["timeout"] == 17
    assert "service_tier" not in wire_kwargs
    assert "speed" not in wire_kwargs

    agent.switch_model(
        new_model="gpt-5.4-mini",
        new_provider="openai",
        api_key="openai-key",
        base_url=OPENAI_URL,
        api_mode="chat_completions",
    )

    assert agent.request_overrides == {"service_tier": "priority", **explicit}
    assert "thinking" not in agent.request_overrides["extra_body"]
    assert "speed" not in agent.request_overrides


def test_explicit_fast_named_scalars_remain_caller_owned(runtime_override_env):
    explicit = {
        **CALLER_OVERRIDES,
        "service_tier": "flex",
        "speed": "fast",
    }
    agent = _openai_fast_agent(explicit)

    assert agent._caller_request_overrides == explicit
    assert agent.request_overrides == explicit

    _switch_to_glm(agent)

    assert agent.request_overrides == {
        **explicit,
        "extra_body": {**GLM_THINKING, **explicit["extra_body"]},
    }


def test_failed_reverse_switch_restores_runtime_and_caller_provenance(
    runtime_override_env,
):
    explicit = copy.deepcopy(CALLER_OVERRIDES)
    agent = _openai_fast_agent(explicit)
    _switch_to_glm(agent)
    old_runtime = {
        name: copy.deepcopy(getattr(agent, name))
        for name in (
            "model",
            "provider",
            "base_url",
            "api_mode",
            "api_key",
            "request_overrides",
            "_primary_runtime",
        )
    }
    old_caller = copy.deepcopy(agent._caller_request_overrides)
    agent._create_openai_client = MagicMock(side_effect=RuntimeError("client failed"))

    with pytest.raises(RuntimeError, match="client failed"):
        agent.switch_model(
            new_model="gpt-5.4-mini",
            new_provider="openai",
            api_key="openai-key",
            base_url=OPENAI_URL,
            api_mode="chat_completions",
        )

    for name, value in old_runtime.items():
        assert getattr(agent, name) == value
    assert agent._caller_request_overrides == old_caller == explicit
    assert "service_tier" not in agent.request_overrides
    assert "speed" not in agent.request_overrides


@pytest.mark.parametrize(
    ("provider", "base_url", "target_extra"),
    [
        ("openai-codex", CODEX_URL, {"caller_flag": True}),
        ("custom:pm", PM_URL, {"caller_flag": True, "pm_target": True}),
    ],
)
def test_real_switch_drops_glm_thinking_and_reverse_restores_target_extras(
    provider,
    base_url,
    target_extra,
):
    agent = _agent()
    config = {
        "agent": {
            "reasoning_effort": "medium",
            "reasoning_overrides": {"gpt-5.4-mini": "low"},
        }
    }

    with (
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_configs(),
        ),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        _switch(agent, provider=provider, base_url=base_url)

        assert agent.request_overrides == {
            "extra_headers": {"X-Caller": "keep"},
            "extra_body": target_extra,
        }
        assert "thinking" not in agent.request_overrides["extra_body"]
        assert agent.reasoning_config == {"enabled": True, "effort": "low"}
        assert agent._primary_runtime["request_overrides"] == agent.request_overrides

        agent.switch_model(
            new_model="glm-5.2",
            new_provider="custom:glm",
            api_key="glm-key",
            base_url=GLM_URL,
            api_mode="chat_completions",
        )

    assert agent.request_overrides == {
        "extra_headers": {"X-Caller": "keep"},
        "extra_body": {"caller_flag": True, **GLM_THINKING},
    }


def test_failed_switch_restores_nested_request_overrides():
    agent = _agent()
    original = copy.deepcopy(agent.request_overrides)
    agent._create_openai_client.side_effect = RuntimeError("client failed")

    with (
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        pytest.raises(RuntimeError, match="client failed"),
    ):
        _switch(agent, provider="openai-codex", base_url=CODEX_URL)

    assert agent.request_overrides == original
    agent.request_overrides["extra_body"]["thinking"]["type"] = "mutated"
    assert original["extra_body"]["thinking"]["type"] == "enabled"


@pytest.mark.parametrize("initial_fast", [False, True])
def test_live_tui_fast_toggle_survives_transient_recovery(
    runtime_override_env,
    monkeypatch,
    initial_fast,
):
    agent = _openai_session_agent(initial_fast=initial_fast)
    target_fast = not initial_fast
    _set_tui_fast(monkeypatch, agent, enabled=target_fast)
    error = type("ReadTimeout", (Exception,), {})("transient")

    with patch("agent.agent_runtime_helpers.time.sleep"):
        assert agent._try_recover_primary_transport(
            error,
            retry_count=3,
            max_retries=3,
        ) is True

    expected = copy.deepcopy(CALLER_OVERRIDES)
    if target_fast:
        expected["service_tier"] = "priority"
    assert agent.service_tier == ("priority" if target_fast else None)
    assert agent.request_overrides == expected


@pytest.mark.parametrize("initial_fast", [False, True])
def test_live_tui_fast_toggle_survives_fallback_restore(
    runtime_override_env,
    monkeypatch,
    initial_fast,
):
    fallback_client = MagicMock(base_url=GLM_URL, api_key="glm-key")
    fallback_client._custom_headers = {}
    fallback_client.default_headers = {}
    agent = _openai_session_agent(
        initial_fast=initial_fast,
        fallback_model=[
            {
                "provider": "custom:glm",
                "model": "glm-5.2",
                "base_url": GLM_URL,
                "api_key": "glm-key",
            }
        ],
    )
    target_fast = not initial_fast
    _set_tui_fast(monkeypatch, agent, enabled=target_fast)

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fallback_client, "glm-5.2"),
    ):
        assert agent._try_activate_fallback() is True
        assert agent._restore_primary_runtime() is True

    expected = copy.deepcopy(CALLER_OVERRIDES)
    if target_fast:
        expected["service_tier"] = "priority"
    assert agent.service_tier == ("priority" if target_fast else None)
    assert agent.request_overrides == expected


def test_real_fallback_activation_rebuilds_and_restores_request_overrides():
    config = {"custom_providers": _configs()}
    fallbacks = [
        {
            "provider": "openai-codex",
            "model": "gpt-5.4-mini",
            "base_url": CODEX_URL,
            "api_key": "codex-key",
        },
        {
            "provider": "custom:pm",
            "model": "gpt-5.4-mini",
            "base_url": PM_URL,
            "api_key": "pm-key",
        },
    ]
    clients = []
    for base_url, api_key in ((CODEX_URL, "codex-key"), (PM_URL, "pm-key")):
        client = MagicMock(base_url=base_url, api_key=api_key)
        client._custom_headers = {}
        client.default_headers = {}
        clients.append(client)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=_configs(),
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=[(clients[0], "gpt-5.4-mini"), (clients[1], "gpt-5.4-mini")],
        ),
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent = AIAgent(
            api_key="glm-key",
            base_url=GLM_URL,
            provider="custom:glm",
            model="glm-5.2",
            request_overrides=CALLER_OVERRIDES,
            fallback_model=fallbacks,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        primary = copy.deepcopy(agent._primary_runtime["request_overrides"])
        assert primary["extra_body"] == {"thinking": {"type": "enabled"}, "caller_flag": True}

        assert agent._try_activate_fallback() is True
        assert agent.request_overrides == CALLER_OVERRIDES

        assert agent._try_activate_fallback() is True
        assert agent.request_overrides == {
            "extra_headers": {"X-Caller": "keep"},
            "extra_body": {"pm_target": True, "caller_flag": True},
        }
        agent.request_overrides["extra_body"]["caller_flag"] = False
        assert agent._caller_request_overrides == CALLER_OVERRIDES
        assert agent._primary_runtime["request_overrides"] == primary

        assert agent._restore_primary_runtime() is True

    assert agent.request_overrides == primary


def test_constructor_snapshots_explicit_caller_overrides():
    explicit = {"extra_body": {"caller_flag": {"nested": True}}}
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url=GLM_URL,
            provider="custom:glm",
            model="glm-5.2",
            request_overrides=explicit,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    explicit["extra_body"]["caller_flag"]["nested"] = False
    assert agent._caller_request_overrides["extra_body"]["caller_flag"] == {
        "nested": True
    }
    assert agent._primary_runtime["request_overrides"]["extra_body"][
        "caller_flag"
    ] == {"nested": True}
