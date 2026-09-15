"""Regression tests for standard-profile native child runtime fallback policy."""

from __future__ import annotations

from contextlib import ExitStack
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import ClassifiedError, FailoverReason
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID
from run_agent import AIAgent
from tools.delegate_tool import _build_child_agent, _run_single_child, delegate_task
from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry
from tools.delegate_tool_progress import _ChildProgressRelay

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
                message=SimpleNamespace(
                    content=text, tool_calls=None, reasoning_content=None,
                    reasoning=None, reasoning_details=None, model_extra={},
                ),
                finish_reason="stop",
            )
        ],
        model=PRIMARY["model"],
        usage=None,
    )


def _make_child(*, max_retries: int = 2, route=PRIMARY, profile=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
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
    if profile is not None:
        agent._delegate_model_profile = profile
    agent._delegate_successful_llm_route = None
    return agent


def _make_standard_child(*, max_retries: int = 2, route=PRIMARY):
    return _make_child(max_retries=max_retries, route=route, profile="standard")


def _common_patches(agent):
    return (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch("agent.turn_recovery.time.sleep"),
        patch("agent.retry_utils.jittered_backoff", return_value=0),
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
    notices = []

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
        stack.enter_context(patch.object(agent, "_buffer_status", side_effect=notices.append))
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
                    "agent.turn_api_error.classify_api_error",
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
                "agent.turn_recovery._try_refresh_nous_paid_entitlement_credentials",
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
    assert all("Primary retry eligible" not in notice for notice in notices)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_HTTPError(402, "billing exhausted"), id="billing-402"),
        pytest.param(
            _HTTPError(503, "auth_unavailable: no auth available"),
            id="auth-unavailable-503",
        ),
        pytest.param(ValueError("unexpected provider runtime failure"), id="unexpected-runtime"),
        pytest.param(ConnectionError("network connection failed"), id="network"),
    ],
)
def test_standard_child_any_execution_error_advances_without_retrying_same_hop(error):
    """A pre-success child never burns its retry budget on a failed provider hop."""
    agent = _make_standard_child(max_retries=3)
    calls = []
    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-key"
    fallback_client.base_url = FALLBACK_CHAIN[0]["base_url"]
    fallback_client._custom_headers = None
    fallback_client.default_headers = None

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) == 1:
            raise error
        return _response("fallback result")

    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=api_call))
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
        stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=200000))
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert calls == [
        (PRIMARY["provider"], PRIMARY["model"]),
        (FALLBACK_CHAIN[0]["provider"], FALLBACK_CHAIN[0]["model"]),
    ]


def test_main_agent_first_transport_error_retries_same_hop():
    """Main-agent eager fallback stays rate-limit or transport-after-2-retries."""
    agent = _make_child(max_retries=3)
    calls = []
    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-key"
    fallback_client.base_url = FALLBACK_CHAIN[0]["base_url"]
    fallback_client._custom_headers = None
    fallback_client.default_headers = None

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) == 1:
            raise ConnectionError("network connection failed")
        return _response("same hop result")

    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=api_call))
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
        stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=200000))
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert calls == [
        (PRIMARY["provider"], PRIMARY["model"]),
        (PRIMARY["provider"], PRIMARY["model"]),
    ]


def test_standard_child_advances_across_multiple_failed_hops_then_succeeds():
    agent = _make_standard_child(max_retries=3)
    calls = []
    fallback_clients = []
    for entry in FALLBACK_CHAIN:
        client = MagicMock()
        client.api_key = "fallback-key"
        client.base_url = entry["base_url"]
        client._custom_headers = None
        client.default_headers = None
        fallback_clients.append(client)

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) < 3:
            raise _HTTPError(503, "auth_unavailable: no auth available")
        return _response("second fallback result")

    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=api_call))
        stack.enter_context(
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                side_effect=[
                    (fallback_clients[0], FALLBACK_CHAIN[0]["model"]),
                    (fallback_clients[1], FALLBACK_CHAIN[1]["model"]),
                ],
            )
        )
        stack.enter_context(
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda model, _provider: model,
            )
        )
        stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=200000))
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert calls == [
        (PRIMARY["provider"], PRIMARY["model"]),
        (FALLBACK_CHAIN[0]["provider"], FALLBACK_CHAIN[0]["model"]),
        (FALLBACK_CHAIN[1]["provider"], FALLBACK_CHAIN[1]["model"]),
    ]


def test_standard_child_cancellation_does_not_activate_fallback():
    agent = _make_standard_child(max_retries=3)
    fallback = MagicMock(return_value=True)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(agent, "_interruptible_api_call", side_effect=KeyboardInterrupt)
        )
        stack.enter_context(patch.object(agent, "_try_activate_fallback", fallback))
        for context in _common_patches(agent):
            stack.enter_context(context)
        with pytest.raises(KeyboardInterrupt):
            agent.run_conversation("hello")

    fallback.assert_not_called()


def test_standard_child_falls_back_after_first_successful_request():
    agent = _make_standard_child(max_retries=1)
    calls = []
    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-key"
    fallback_client.base_url = FALLBACK_CHAIN[0]["base_url"]
    fallback_client._custom_headers = None
    fallback_client.default_headers = None

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) == 1:
            return _response("first turn")
        if len(calls) == 2:
            raise _HTTPError(429, "quota exhausted")
        return _response("fallback turn")

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(agent, "_interruptible_api_call", side_effect=api_call)
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
        stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=200000))
        for context in _common_patches(agent):
            stack.enter_context(context)
        first = agent.run_conversation("first")
        second = agent.run_conversation("second")

    assert first["completed"] is True
    assert agent._delegate_successful_llm_route == (
        FALLBACK_CHAIN[0]["model"],
        FALLBACK_CHAIN[0]["provider"],
    )
    assert second["completed"] is True
    assert calls == [
        (PRIMARY["provider"], PRIMARY["model"]),
        (PRIMARY["provider"], PRIMARY["model"]),
        (FALLBACK_CHAIN[0]["provider"], FALLBACK_CHAIN[0]["model"]),
    ]


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
        child._delegate_successful_llm_route = (
            FALLBACK_CHAIN[0]["model"],
            FALLBACK_CHAIN[0]["provider"],
        )
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


def test_delegate_task_ordinary_child_records_primary_success_identity():
    """The public no-profile child records identity only after a real response."""
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
        request_overrides={},
        _fallback_chain=[],
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        _session_db=None,
        session_id=None,
        tool_progress_callback=lambda *args, **kwargs: events.append(kwargs),
    )

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch.object(AIAgent, "_interruptible_api_call", return_value=_response("ordinary result")),
        patch.object(AIAgent, "_interruptible_streaming_api_call", return_value=_response("ordinary result")),
        patch.object(AIAgent, "_persist_session"),
        patch.object(AIAgent, "_save_trajectory"),
        patch.object(AIAgent, "_cleanup_task_resources"),
    ):
        result = json.loads(
            delegate_task(tasks=[{"goal": "return a normal response"}], parent_agent=parent, background=False)
        )

    entry = result["results"][0]
    assert entry["status"] == "completed"
    assert (entry["model"], entry["provider"]) == (PRIMARY["model"], PRIMARY["provider"])
    complete = [event for event in events if event.get("status") == "completed"]
    assert complete
    assert (complete[-1]["model"], complete[-1]["provider"]) == (PRIMARY["model"], PRIMARY["provider"])


def test_delegate_task_zero_iteration_summary_records_successful_route(monkeypatch):
    """A real summary-only child retains the route that produced its answer."""
    import tools.delegate_tool as delegate_mod

    events = []
    summary_client = MagicMock()
    summary_client.chat.completions.create.return_value = _response("summary result")
    monkeypatch.setattr(delegate_mod, "_load_config", lambda: {"max_iterations": 0})

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch.object(AIAgent, "_ensure_primary_openai_client", return_value=summary_client),
        patch.object(AIAgent, "_persist_session"),
        patch.object(AIAgent, "_save_trajectory"),
        patch.object(AIAgent, "_cleanup_task_resources"),
    ):
        result = json.loads(
            delegate_task(
                tasks=[{"goal": "return the iteration-limit summary"}],
                parent_agent=_delegate_parent(events),
                background=False,
            )
        )

    entry = result["results"][0]
    assert entry["status"] == "completed"
    assert entry["truncated"] is True
    assert entry["summary"].startswith("summary result")
    assert (entry["model"], entry["provider"]) == (PRIMARY["model"], PRIMARY["provider"])
    complete = [event for event in events if event.get("status") == "completed"]
    assert complete
    assert (complete[-1]["model"], complete[-1]["provider"]) == (PRIMARY["model"], PRIMARY["provider"])


def test_delegate_task_background_zero_iteration_summary_persists_successful_route(monkeypatch, tmp_path):
    """The detached public path persists the summary-producing child route."""
    import time

    from tools import async_delegation as async_delegation
    from tools.process_registry import process_registry
    import tools.delegate_tool as delegate_mod

    summary_client = MagicMock()
    summary_client.chat.completions.create.return_value = _response("background summary")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delegate_mod, "_load_config", lambda: {"max_iterations": 0})

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch.object(AIAgent, "_ensure_primary_openai_client", return_value=summary_client),
        patch.object(AIAgent, "_persist_session"),
        patch.object(AIAgent, "_save_trajectory"),
        patch.object(AIAgent, "_cleanup_task_resources"),
    ):
        handle = json.loads(
            delegate_task(
                tasks=[{"goal": "persist the iteration-limit summary"}],
                parent_agent=_delegate_parent([]),
                background=True,
            )
        )
        deadline = time.monotonic() + 5
        event = None
        while time.monotonic() < deadline:
            if not process_registry.completion_queue.empty():
                candidate = process_registry.completion_queue.get_nowait()
                if candidate.get("delegation_id") == handle["delegation_id"]:
                    event = candidate
                    break
            time.sleep(0.02)

    assert event is not None
    assert (event["results"][0]["model"], event["results"][0]["provider"]) == (
        PRIMARY["model"], PRIMARY["provider"],
    )
    with async_delegation._connect() as conn:
        event_json, result_json = conn.execute(
            "SELECT event_json, result_json FROM async_delegations WHERE delegation_id=?",
            (handle["delegation_id"],),
        ).fetchone()
    persisted_event, persisted_result = json.loads(event_json), json.loads(result_json)
    assert (persisted_event["results"][0]["model"], persisted_event["results"][0]["provider"]) == (
        PRIMARY["model"], PRIMARY["provider"],
    )
    assert (persisted_result["results"][0]["model"], persisted_result["results"][0]["provider"]) == (
        PRIMARY["model"], PRIMARY["provider"],
    )


def test_delegate_task_ordinary_child_records_fallback_success_identity():
    """The public no-profile child reports only its accepted fallback response."""
    events = []
    attempts = []
    parent = SimpleNamespace(
        base_url=PRIMARY["base_url"],
        api_key="primary-key",
        provider=PRIMARY["provider"],
        api_mode="chat_completions",
        model=PRIMARY["model"],
        platform="cli",
        enabled_toolsets=[],
        disabled_toolsets=[],
        request_overrides={},
        _fallback_chain=FALLBACK_CHAIN,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        _session_db=None,
        session_id=None,
        tool_progress_callback=lambda *args, **kwargs: events.append(kwargs),
    )
    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-key"
    fallback_client.base_url = FALLBACK_CHAIN[0]["base_url"]
    fallback_client._custom_headers = None
    fallback_client.default_headers = None

    def stream_call(agent, _kwargs, **_ignored):
        attempts.append((agent.provider, agent.model))
        if len(attempts) == 1:
            raise _HTTPError(429, "usage limit has been reached")
        return _response("fallback result")

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch.object(AIAgent, "_interruptible_streaming_api_call", stream_call),
        patch.object(AIAgent, "_persist_session"),
        patch.object(AIAgent, "_save_trajectory"),
        patch.object(AIAgent, "_cleanup_task_resources"),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client, FALLBACK_CHAIN[0]["model"])),
        patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda model, _provider: model),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
        patch("agent.turn_recovery.time.sleep"),
        patch("agent.retry_utils.jittered_backoff", return_value=0),
    ):
        result = json.loads(
            delegate_task(
                tasks=[{"goal": "return a fallback response"}],
                parent_agent=parent,
                credentials_cfg={},
                background=False,
            )
        )

    entry = result["results"][0]
    assert entry["status"] == "completed"
    assert attempts == [
        (PRIMARY["provider"], PRIMARY["model"]),
        (FALLBACK_CHAIN[0]["provider"], FALLBACK_CHAIN[0]["model"]),
    ]
    assert (entry["model"], entry["provider"]) == (FALLBACK_CHAIN[0]["model"], FALLBACK_CHAIN[0]["provider"])
    complete = [event for event in events if event.get("status") == "completed"]
    assert complete
    assert (complete[-1]["model"], complete[-1]["provider"]) == (FALLBACK_CHAIN[0]["model"], FALLBACK_CHAIN[0]["provider"])


def test_standard_child_without_fallback_keeps_nous_entitlement_recovery():
    """Fallback availability alone may defer same-route Nous repair."""
    agent = _make_standard_child(max_retries=2, route=NOUS)
    agent._fallback_chain = []
    calls = []

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) == 1:
            raise _HTTPError(429, "usage limit has been reached")
        return _response("recovered on the same route")

    refreshed = MagicMock(return_value=True)
    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=api_call))
        stack.enter_context(
            patch(
                "agent.turn_api_error.classify_api_error",
                return_value=ClassifiedError(
                    reason=FailoverReason.billing,
                    status_code=429,
                    retryable=False,
                    should_fallback=True,
                ),
            )
        )
        stack.enter_context(
            patch(
                "agent.turn_recovery._try_refresh_nous_paid_entitlement_credentials",
                refreshed,
            )
        )
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert calls == [(NOUS["provider"], NOUS["model"])] * 2
    refreshed.assert_called_once_with(agent)



@pytest.mark.parametrize("unusable", [
    pytest.param("same-backend", id="same-backend"),
    pytest.param("already-unavailable", id="already-unavailable"),
    pytest.param("known-unentitled", id="known-unentitled"),
    pytest.param("locally-unusable", id="locally-unusable"),
    pytest.param("resolver-none", id="resolver-none"),
])
def test_standard_child_unusable_fallback_suffix_reenters_nous_recovery(unusable):
    """A nonempty standard-child suffix may still contain no activatable route."""
    agent = _make_standard_child(max_retries=1, route=NOUS)
    fallback = dict(FALLBACK_CHAIN[0])
    if unusable == "same-backend":
        fallback = dict(NOUS)
    agent._fallback_chain = [fallback]
    if unusable == "already-unavailable":
        agent._unavailable_fallback_keys = {
            (fallback["provider"], fallback["model"], fallback["base_url"])
        }
    calls = []

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) == 1:
            raise _HTTPError(429, "usage limit has been reached")
        return _response("recovered on the same route")

    refreshed = MagicMock(return_value=True)
    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=api_call))
        stack.enter_context(
            patch(
                "agent.turn_api_error.classify_api_error",
                return_value=ClassifiedError(
                    reason=FailoverReason.billing,
                    status_code=429,
                    retryable=False,
                    should_fallback=True,
                ),
            )
        )
        stack.enter_context(
            patch(
                "agent.turn_recovery._try_refresh_nous_paid_entitlement_credentials",
                refreshed,
            )
        )
        if unusable == "same-backend":
            stack.enter_context(
                patch("agent.chat_completion_helpers._fallback_entry_unavailable_without_network", return_value=None)
            )
        elif unusable == "known-unentitled":
            stack.enter_context(
                patch("agent.fallback_cooldown._is_entitlement_rejected", return_value=True)
            )
        elif unusable == "locally-unusable":
            stack.enter_context(
                patch(
                    "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                    return_value="not locally configured",
                )
            )
        elif unusable == "resolver-none":
            stack.enter_context(
                patch(
                    "agent.auxiliary_client.resolve_provider_client",
                    return_value=(None, fallback["model"]),
                )
            )
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert calls == [(NOUS["provider"], NOUS["model"])] * 2
    assert agent._fallback_index == len(agent._fallback_chain)
    refreshed.assert_called_once_with(agent)


def test_standard_child_post_mutation_fallback_failure_does_not_recover_original_route():
    """A failed switch may not send the original error through changed runtime state."""
    agent = _make_standard_child(max_retries=1, route=NOUS)
    agent._fallback_chain = [dict(FALLBACK_CHAIN[0])]
    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-key"
    fallback_client.base_url = FALLBACK_CHAIN[0]["base_url"]
    fallback_client._custom_headers = None
    fallback_client.default_headers = None
    refreshed = MagicMock(return_value=True)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                agent, "_interruptible_api_call",
                side_effect=_HTTPError(429, "usage limit has been reached"),
            )
        )
        stack.enter_context(
            patch(
                "agent.turn_api_error.classify_api_error",
                return_value=ClassifiedError(
                    reason=FailoverReason.billing,
                    status_code=429,
                    retryable=False,
                    should_fallback=True,
                ),
            )
        )
        stack.enter_context(
            patch(
                "agent.turn_recovery._try_refresh_nous_paid_entitlement_credentials",
                refreshed,
            )
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
            patch(
                "agent.agent_runtime_helpers.sync_credential_pool_entry_id",
                side_effect=RuntimeError("late switch setup failed"),
            )
        )
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is False
    assert agent.provider == FALLBACK_CHAIN[0]["provider"]
    refreshed.assert_not_called()

def test_standard_child_direct_output_cap_retries_same_route_before_fallback():
    """A direct 400 max-output error is a request-shape repair, not failover."""
    agent = _make_standard_child(max_retries=2)
    agent.max_tokens = 98_304
    agent.context_compressor.context_length = 200_000
    calls = []

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) == 1:
            raise _HTTPError(
                400,
                "max_tokens (98304) exceeds model's maximum output tokens (65536)",
            )
        return _response("clamped result")

    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=api_call))
        stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=200_000))
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert calls == [(PRIMARY["provider"], PRIMARY["model"])] * 2
    assert agent._fallback_index == 0


def test_standard_child_forbidden_fallback_is_terminal_after_one_request():
    """A content-policy rejection cannot retry or consume a configured fallback."""
    agent = _make_standard_child(max_retries=3)
    calls = []

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        raise _HTTPError(403, "content policy blocked")

    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=api_call))
        stack.enter_context(
            patch(
                "agent.turn_api_error.classify_api_error",
                return_value=ClassifiedError(
                    reason=FailoverReason.content_policy_blocked,
                    status_code=403,
                    retryable=False,
                    should_fallback=True,
                ),
            )
        )
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("disallowed")

    assert result["completed"] is False
    assert result["failed"] is True
    assert calls == [(PRIMARY["provider"], PRIMARY["model"])]
    assert agent._fallback_index == 0


@pytest.mark.parametrize(
    ("policy_response", "model_profile", "fallback_expected"),
    [
        pytest.param("http-200", "standard", False, id="standard-http-200"),
        pytest.param("stream", "standard", False, id="standard-stream"),
        pytest.param("http-200", None, True, id="ordinary-http-200-control"),
        pytest.param("stream", None, True, id="ordinary-stream-control"),
    ],
)
def test_delegate_task_content_policy_respects_profile_fallback_boundary(
    monkeypatch, policy_response, model_profile, fallback_expected
):
    """The public child path cannot move a standard policy refusal to another provider."""
    import tools.delegate_tool as delegate_mod

    attempts = []
    parent = _delegate_parent([])
    parent._fallback_chain = FALLBACK_CHAIN
    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-key"
    fallback_client.base_url = FALLBACK_CHAIN[0]["base_url"]
    fallback_client._custom_headers = None
    fallback_client.default_headers = None
    monkeypatch.setattr(delegate_mod, "_load_config", lambda: {"max_iterations": 2})

    def policy_result():
        if policy_response == "http-200":
            return SimpleNamespace(
                id="policy-refusal",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=None,
                            reasoning_content=None,
                            reasoning=None,
                            reasoning_details=None,
                            refusal="request blocked by policy",
                            model_extra={},
                        ),
                        finish_reason="content_filter",
                    )
                ],
                model=PRIMARY["model"],
                usage=None,
            )
        return SimpleNamespace(
            id=PARTIAL_STREAM_STUB_ID,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="partial policy-filtered stream",
                        tool_calls=None,
                        reasoning_content=None,
                        reasoning=None,
                        reasoning_details=None,
                        model_extra={},
                    ),
                    finish_reason=FINISH_REASON_LENGTH,
                )
            ],
            model=PRIMARY["model"],
            usage=None,
            _content_filter_terminated=True,
        )

    def stream_call(agent, _kwargs, **_ignored):
        attempts.append((agent.provider, agent.model))
        if len(attempts) == 1:
            return policy_result()
        return _response("response after policy handling")

    task = {"goal": "handle the synthetic policy response"}
    if model_profile is not None:
        task["model_profile"] = model_profile
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch.object(AIAgent, "_interruptible_streaming_api_call", stream_call),
        patch.object(AIAgent, "_persist_session"),
        patch.object(AIAgent, "_save_trajectory"),
        patch.object(AIAgent, "_cleanup_task_resources"),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, FALLBACK_CHAIN[0]["model"]),
        ),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda model, _provider: model,
        ),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
        patch("agent.turn_recovery.time.sleep"),
        patch("agent.retry_utils.jittered_backoff", return_value=0),
    ):
        result = json.loads(
            delegate_task(
                tasks=[task],
                parent_agent=parent,
                credentials_cfg={},
                background=False,
            )
        )

    fallback_route = (FALLBACK_CHAIN[0]["provider"], FALLBACK_CHAIN[0]["model"])
    assert (fallback_route in attempts) is fallback_expected
    if fallback_expected:
        assert result["results"][0]["status"] == "completed"
        assert attempts == [
            (PRIMARY["provider"], PRIMARY["model"]),
            fallback_route,
        ]
    else:
        assert attempts
        assert all(route == (PRIMARY["provider"], PRIMARY["model"]) for route in attempts)


def test_nonstandard_child_records_its_successful_primary_route():
    agent = _make_child(profile="premium")
    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", return_value=_response("ok")))
        for context in _common_patches(agent):
            stack.enter_context(context)
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert agent._delegate_successful_llm_route == (PRIMARY["model"], PRIMARY["provider"])


def test_delegate_result_keeps_primary_success_identity_after_failed_fallback():
    """A later failed fallback must not relabel an already-successful child."""
    parent = SimpleNamespace(
        base_url=PRIMARY["base_url"], api_key="primary-key", provider=PRIMARY["provider"],
        api_mode="chat_completions", model=PRIMARY["model"], platform="cli",
        enabled_toolsets=[], disabled_toolsets=[], _fallback_chain=FALLBACK_CHAIN,
        _delegate_depth=0, _active_children=[], _active_children_lock=None, _print_fn=None,
        _session_db=None, tool_progress_callback=None,
    )
    child = MagicMock()
    child.provider = PRIMARY["provider"]
    child.model = PRIMARY["model"]
    child._credential_pool = None
    child.session_prompt_tokens = child.session_completion_tokens = 0

    def primary_then_failed_fallback(*_args, **_kwargs):
        child._delegate_successful_llm_route = (PRIMARY["model"], PRIMARY["provider"])
        child.provider, child.model = FALLBACK_CHAIN[0]["provider"], FALLBACK_CHAIN[0]["model"]
        return {"completed": False, "failed": True, "error": "fallback failed", "api_calls": 2}

    child.run_conversation.side_effect = primary_then_failed_fallback
    with patch("run_agent.AIAgent", return_value=child):
        built = _build_child_agent(
            task_index=0, goal="keep successful identity", context=None, toolsets=None,
            model=None, max_iterations=2, parent_agent=parent, task_count=1,
            model_profile="premium",
        )
        result = _run_single_child(0, "keep successful identity", built, parent)

    assert result["model"] == PRIMARY["model"]
    assert result["provider"] == PRIMARY["provider"]


def test_app_server_success_projects_unknown_identity_without_erasing_known_route():
    """An app-server completion has no selected route, even after known success."""
    child = SimpleNamespace(
        api_mode="codex_app_server",
        _delegate_successful_llm_route=(PRIMARY["model"], PRIMARY["provider"]),
        session_estimated_cost_usd=0, session_cost_status=None,
        session_prompt_tokens=0, session_completion_tokens=0, _delegate_role="leaf",
    )
    result = {"completed": True, "final_response": "app-server result", "codex_turn_id": None}
    entry = _build_result_entry(child, result, 0, 0.1, _SchemaOutcome(None, None, [], 0))

    assert (entry["model"], entry["provider"]) == (None, None)
    assert child._delegate_successful_llm_route == (PRIMARY["model"], PRIMARY["provider"])
    relay = _ChildProgressRelay(
        0, "task", None, None, 1, "subagent", None, 0, "cached-model", None, {"child": child}
    )
    assert relay._identity_kwargs()["model"] is None
    assert relay._identity_kwargs()["provider"] is None


def test_app_server_failure_keeps_last_known_success_identity():
    """Unknown app-server success does not erase K needed by a later failure."""
    child = SimpleNamespace(
        api_mode="codex_app_server",
        _delegate_successful_llm_route=(PRIMARY["model"], PRIMARY["provider"]),
        session_estimated_cost_usd=0, session_cost_status=None,
        session_prompt_tokens=0, session_completion_tokens=0, _delegate_role="leaf",
    )
    entry = _build_result_entry(
        child, {"failed": True, "error": "later app-server failure", "codex_turn_id": None},
        0, 0.1, _SchemaOutcome(None, None, [], 0),
    )
    assert (entry["model"], entry["provider"]) == (PRIMARY["model"], PRIMARY["provider"])


def _delegate_parent(events):
    return SimpleNamespace(
        base_url=PRIMARY["base_url"], api_key="primary-key", provider=PRIMARY["provider"],
        api_mode="chat_completions", model=PRIMARY["model"], platform="cli",
        enabled_toolsets=[], disabled_toolsets=[], request_overrides={}, _fallback_chain=[],
        _delegate_depth=0, _active_children=[], _active_children_lock=threading.Lock(),
        _print_fn=None, _session_db=None, session_id=None,
        tool_progress_callback=lambda *args, **kwargs: events.append(kwargs),
    )


def _codex_delegate_credentials():
    return {
        "provider": "openai", "model": "stub-model", "base_url": "https://stub.invalid/v1",
        "api_key": "stub-key", "api_mode": "codex_app_server", "command": None, "args": None,
    }


@pytest.mark.parametrize(
    ("turn", "expected_status"),
    [
        (TurnResult(final_text="unknown app-server success", projected_messages=[], turn_id="u", thread_id="t"), "completed"),
        (TurnResult(final_text="", projected_messages=[], error="unknown app-server failure", turn_id="f", thread_id="t"), "failed"),
    ],
    ids=("fresh-unknown-success", "fresh-unknown-then-failure"),
)
def test_delegate_task_app_server_unknown_identity_uses_real_child_path(monkeypatch, turn, expected_status):
    """The public delegation path cannot infer an app-server selected route."""
    import tools.delegate_tool as delegate_mod

    events = []
    monkeypatch.setattr(delegate_mod, "_resolve_delegation_credentials", lambda *_args, **_kwargs: _codex_delegate_credentials())
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda _self: "t")
    monkeypatch.setattr(CodexAppServerSession, "run_turn", lambda _self, **_kwargs: turn)

    result = json.loads(delegate_task(
        tasks=[{"goal": "return the fake app-server turn"}], parent_agent=_delegate_parent(events),
        credentials_cfg={"provider": "openai", "api_mode": "codex_app_server"}, background=False,
    ))

    entry = result["results"][0]
    assert entry["status"] == expected_status
    assert (entry["model"], entry["provider"]) == (None, None)
    complete = [event for event in events if event.get("status") in {"completed", "failed"}]
    assert complete
    assert (complete[-1]["model"], complete[-1]["provider"]) == (None, None)


def test_delegate_task_background_preserves_unknown_app_server_identity_in_durable_batch(monkeypatch, tmp_path):
    """The detached public path must retain a child's explicit null route."""
    import time

    from tools import async_delegation as async_delegation
    from tools.process_registry import process_registry
    import tools.delegate_tool as delegate_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delegate_mod, "_resolve_delegation_credentials", lambda *_args, **_kwargs: _codex_delegate_credentials())
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda _self: "t")
    monkeypatch.setattr(
        CodexAppServerSession, "run_turn",
        lambda _self, **_kwargs: TurnResult(final_text="unknown", projected_messages=[], turn_id="u", thread_id="t"),
    )

    handle = json.loads(delegate_task(
        tasks=[{"goal": "persist a fake app-server turn"}], parent_agent=_delegate_parent([]),
        credentials_cfg={"provider": "openai", "api_mode": "codex_app_server"}, background=True,
    ))
    deadline = time.monotonic() + 5
    event = None
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            candidate = process_registry.completion_queue.get_nowait()
            if candidate.get("delegation_id") == handle["delegation_id"]:
                event = candidate
                break
        time.sleep(0.02)
    assert event is not None
    assert (event["results"][0]["model"], event["results"][0]["provider"]) == (None, None)
    with async_delegation._connect() as conn:
        event_json, result_json = conn.execute(
            "SELECT event_json, result_json FROM async_delegations WHERE delegation_id=?", (handle["delegation_id"],)
        ).fetchone()
    persisted_event, persisted_result = json.loads(event_json), json.loads(result_json)
    assert (persisted_event["results"][0]["model"], persisted_event["results"][0]["provider"]) == (None, None)
    assert (persisted_result["results"][0]["model"], persisted_result["results"][0]["provider"]) == (None, None)


def test_real_child_known_unknown_known_sequence_replaces_only_with_later_normal_success(monkeypatch):
    """A→U→C retains A across U and replaces it only at normal C admission."""
    child = _make_child()
    later = {"provider": "later-provider", "model": "later-model", "base_url": "https://later.invalid/v1"}
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda _self: "t")
    monkeypatch.setattr(
        CodexAppServerSession, "run_turn",
        lambda _self, **_kwargs: TurnResult(final_text="unknown", projected_messages=[], turn_id="u", thread_id="t"),
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(child, "_interruptible_api_call", return_value=_response("known A")))
        for context in _common_patches(child):
            stack.enter_context(context)
        assert child.run_conversation("known A")["completed"] is True
        assert child._delegate_successful_llm_route == (PRIMARY["model"], PRIMARY["provider"])

        child.api_mode = "codex_app_server"
        unknown = child.run_conversation("unknown U")
        unknown_entry = _build_result_entry(child, unknown, 0, 0.1, _SchemaOutcome(None, None, [], 0))
        assert (unknown_entry["model"], unknown_entry["provider"]) == (None, None)
        assert child._delegate_successful_llm_route == (PRIMARY["model"], PRIMARY["provider"])

        child.api_mode, child.provider, child.model, child.base_url = (
            "chat_completions", later["provider"], later["model"], later["base_url"]
        )
        assert child.run_conversation("known C")["completed"] is True
    assert child._delegate_successful_llm_route == (later["model"], later["provider"])
