"""Regression tests for standard-profile native child runtime fallback policy."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import ClassifiedError, FailoverReason
from run_agent import AIAgent
from tools.delegate_tool import _build_child_agent, _run_single_child

PRIMARY = {
    "provider": "primary-provider",
    "model": "primary-model",
    "base_url": "https://primary.invalid/v1",
}
NOUS = {
    "provider": "nous",
    "model": "nous-model",
    "base_url": "https://inference-api.nousresearch.com/v1",
}
FALLBACK_CHAIN = [
    {
        "provider": "fallback-provider",
        "model": "fallback-model",
        "base_url": "https://fallback.invalid/v1",
    },
    {
        "provider": "fallback-provider-2",
        "model": "fallback-model-2",
        "base_url": "https://fallback-2.invalid/v1",
    },
]


class _HTTPError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": message}}


def _response(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason="stop",
            )
        ],
        model=PRIMARY["model"],
        usage=None,
    )


def _make_standard_child(*, max_retries: int = 2, route=PRIMARY):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key",
            base_url=route["base_url"],
            provider=route["provider"],
            model=route["model"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=FALLBACK_CHAIN,
        )
    agent.client = MagicMock()
    agent._api_max_retries = max_retries
    agent._delegate_model_profile = "standard"
    agent._delegate_has_successful_llm_request = False
    return agent


def _common_patches(agent):
    return (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch("agent.conversation_loop.time.sleep"),
        patch("agent.conversation_loop.jittered_backoff", return_value=0),
    )


@pytest.mark.parametrize(
    "route",
    [
        pytest.param(PRIMARY, id="primary-provider"),
        pytest.param(NOUS, id="nous-standard-child"),
    ],
)
@pytest.mark.parametrize(
    "message",
    [
        "Weekly usage limit reached",
        "usage limit has been reached",
        "insufficient credits",
    ],
)
def test_standard_child_terminal_quota_429_advances_without_pool_retry_or_cooldown(
    route, message
):
    agent = _make_standard_child(max_retries=3, route=route)
    calls = []

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) == 1:
            raise _HTTPError(429, message)
        return _response("fallback result")

    pool_recovery = MagicMock(return_value=(True, True))
    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-key"
    fallback_client.base_url = FALLBACK_CHAIN[0]["base_url"]
    fallback_client._custom_headers = None
    fallback_client.default_headers = None

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(agent, "_interruptible_api_call", side_effect=api_call)
        )
        stack.enter_context(
            patch.object(agent, "_recover_with_credential_pool", pool_recovery)
        )
        stack.enter_context(
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(fallback_client, FALLBACK_CHAIN[0]["model"]),
            )
        )
        stack.enter_context(
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda model, _provider: model,
            )
        )
        stack.enter_context(
            patch("agent.model_metadata.get_model_context_length", return_value=200000)
        )
        if route["provider"] == NOUS["provider"]:
            stack.enter_context(
                patch(
                    "agent.conversation_loop.classify_api_error",
                    return_value=ClassifiedError(
                        reason=FailoverReason.billing,
                        status_code=429,
                        retryable=False,
                        should_fallback=True,
                    ),
                )
            )
        nous_refresh = stack.enter_context(
            patch(
                "agent.conversation_loop._try_refresh_nous_paid_entitlement_credentials",
                return_value=True,
            )
        )
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert calls == [
        (route["provider"], route["model"]),
        (FALLBACK_CHAIN[0]["provider"], FALLBACK_CHAIN[0]["model"]),
    ]
    nous_refresh.assert_not_called()
    pool_recovery.assert_not_called()
    assert getattr(agent, "_rate_limited_until", 0) == 0


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_HTTPError(401, "unauthorized"), id="401"),
        pytest.param(_HTTPError(402, "payment required"), id="402"),
        pytest.param(_HTTPError(500, "server error"), id="5xx"),
        pytest.param(TimeoutError("request timed out"), id="timeout"),
        pytest.param(ConnectionError("network connection failed"), id="network"),
    ],
)
def test_standard_child_other_errors_retry_same_route_then_fail(error):
    agent = _make_standard_child(max_retries=2)
    calls = []
    fallback = MagicMock(return_value=False)

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        raise error

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(agent, "_interruptible_api_call", side_effect=api_call)
        )
        stack.enter_context(patch.object(agent, "_try_activate_fallback", fallback))
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is False
    assert result["failed"] is True
    assert calls == [
        (PRIMARY["provider"], PRIMARY["model"]),
        (PRIMARY["provider"], PRIMARY["model"]),
    ]
    fallback.assert_not_called()


def test_standard_child_never_switches_after_first_successful_request():
    agent = _make_standard_child(max_retries=2)
    calls = []
    fallback = MagicMock(return_value=False)

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) == 1:
            return _response("first turn")
        raise _HTTPError(429, "quota exhausted")

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(agent, "_interruptible_api_call", side_effect=api_call)
        )
        stack.enter_context(patch.object(agent, "_try_activate_fallback", fallback))
        for context in _common_patches(agent):
            stack.enter_context(context)
        first = agent.run_conversation("first")
        second = agent.run_conversation("second")

    assert first["completed"] is True
    assert agent._delegate_has_successful_llm_request is True
    assert second["completed"] is False
    assert second["failed"] is True
    assert calls == [
        (PRIMARY["provider"], PRIMARY["model"]),
        (PRIMARY["provider"], PRIMARY["model"]),
        (PRIMARY["provider"], PRIMARY["model"]),
    ]
    fallback.assert_not_called()


def test_delegate_progress_and_result_use_only_successful_fallback_identity():
    events = []
    parent = SimpleNamespace(
        base_url=PRIMARY["base_url"],
        api_key="primary-key",
        provider=PRIMARY["provider"],
        api_mode="chat_completions",
        model=PRIMARY["model"],
        platform="cli",
        enabled_toolsets=[],
        disabled_toolsets=[],
        _fallback_chain=FALLBACK_CHAIN,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=None,
        _print_fn=None,
        _session_db=None,
        tool_progress_callback=lambda *args, **kwargs: events.append(kwargs),
    )
    child = MagicMock()
    child.provider = PRIMARY["provider"]
    child.model = PRIMARY["model"]
    child._credential_pool = None
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    attempts = []

    def quota_429_then_fallback(**_kwargs):
        attempts.append((child.provider, child.model))
        failed = _HTTPError(429, "usage limit has been reached")
        assert failed.status_code == 429
        child.provider = FALLBACK_CHAIN[0]["provider"]
        child.model = FALLBACK_CHAIN[0]["model"]
        child._delegate_has_successful_llm_request = True
        attempts.append((child.provider, child.model))
        child.tool_progress_callback("tool.started", tool_name="terminal")
        return {"final_response": "fallback result", "completed": True, "api_calls": 2}

    child.run_conversation.side_effect = quota_429_then_fallback
    with patch("run_agent.AIAgent", return_value=child) as mock_agent:
        built = _build_child_agent(
            task_index=0,
            goal="Use the fallback",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=2,
            parent_agent=parent,
            task_count=1,
            model_profile="standard",
        )
        child.tool_progress_callback = mock_agent.call_args.kwargs["tool_progress_callback"]
        result = _run_single_child(0, "Use the fallback", built, parent)

    assert attempts == [
        (PRIMARY["provider"], PRIMARY["model"]),
        (FALLBACK_CHAIN[0]["provider"], FALLBACK_CHAIN[0]["model"]),
    ]
    assert result["model"] == FALLBACK_CHAIN[0]["model"]
    assert result["provider"] == FALLBACK_CHAIN[0]["provider"]
    assert events
    assert all(event.get("model") != PRIMARY["model"] for event in events)
    assert all(event.get("provider") != PRIMARY["provider"] for event in events)
    assert events[-1]["model"] == FALLBACK_CHAIN[0]["model"]
    assert events[-1]["provider"] == FALLBACK_CHAIN[0]["provider"]
