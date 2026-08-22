"""Regression tests for standard-profile native child runtime fallback policy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

import pytest

from run_agent import AIAgent


PRIMARY = {
    "provider": "primary-provider",
    "model": "primary-model",
    "base_url": "https://primary.invalid/v1",
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


def _make_standard_child(*, max_retries: int = 2):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key",
            base_url=PRIMARY["base_url"],
            provider=PRIMARY["provider"],
            model=PRIMARY["model"],
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
    "message",
    [
        "Weekly usage limit reached",
        "usage limit has been reached",
    ],
)
def test_standard_child_terminal_quota_429_advances_without_pool_retry_or_cooldown(message):
    agent = _make_standard_child(max_retries=3)
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
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert calls == [
        (PRIMARY["provider"], PRIMARY["model"]),
        (FALLBACK_CHAIN[0]["provider"], FALLBACK_CHAIN[0]["model"]),
    ]
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
