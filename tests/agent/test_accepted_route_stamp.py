"""Only an accepted, non-refused response stamps a successful route.

A shape-valid content_filter refusal is not acceptance. Empty choices never
reach the stamp. A valid stop does.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_response_check import check_api_response


def _agent(model, provider):
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent.model = model
    agent.provider = provider
    agent.quiet_mode = True
    agent.verbose_logging = False
    agent.log_prefix = ""
    agent._delegate_successful_llm_route = None
    agent._turn_received_provider_response = False
    transport = MagicMock()
    transport.normalize_response.return_value = SimpleNamespace(finish_reason="stop")
    agent._get_transport.return_value = transport
    agent._should_treat_stop_as_truncated.return_value = False
    return agent


def _check(agent, response):
    return check_api_response(
        agent,
        response=response,
        _retry=SimpleNamespace(has_retried_429=False),
        thinking_spinner=None,
        messages=[],
        api_messages=[],
        api_kwargs={},
        active_system_prompt=None,
        conversation_history=[],
        finish_reason=None,
        retry_count=0,
        max_retries=1,
        compression_attempts=0,
        max_compression_attempts=1,
        length_continue_retries=0,
        truncated_response_parts=[],
        truncated_tool_call_retries=0,
        current_turn_user_idx=0,
        api_call_count=1,
        api_request_id="req",
        api_start_time=0,
        effective_task_id="task",
        turn_id="turn",
        _preflight_compression_blocked=False,
        _last_preflight_pressure=None,
    )


def test_content_filter_does_not_stamp_successful_route(monkeypatch):
    agent = _agent("rejected-model", "rejected-provider")
    agent._get_transport.return_value.normalize_response.return_value = SimpleNamespace(
        finish_reason="content_filter", content="no", provider_data={},
    )
    monkeypatch.setattr(
        "agent.turn_recovery.validate_response_shape",
        lambda *_a, **_k: (False, []),
    )
    monkeypatch.setattr(
        "agent.turn_truncation.handle_content_policy_refusal",
        lambda *_a, **_k: SimpleNamespace(action="return", result={"failed": True}, active_system_prompt=None),
    )

    _check(agent, SimpleNamespace(choices=[object()], model="rejected-model"))

    assert agent._delegate_successful_llm_route is None


def test_empty_choices_does_not_stamp(monkeypatch):
    agent = _agent("rejected-model", "rejected-provider")
    monkeypatch.setattr(
        "agent.turn_recovery.validate_response_shape",
        lambda *_a, **_k: (True, ["response.choices is empty"]),
    )
    monkeypatch.setattr(
        "agent.turn_response_check.retry_invalid_response",
        lambda *_a, **_k: SimpleNamespace(
            action="return", thinking_spinner=None, active_system_prompt=None,
            retry_count=1, compression_attempts=0, result={"failed": True},
        ),
    )

    _check(agent, SimpleNamespace(choices=[]))

    assert agent._delegate_successful_llm_route is None


def test_valid_stop_stamps_current_route(monkeypatch):
    agent = _agent("accepted-model", "anthropic")
    monkeypatch.setattr(
        "agent.turn_recovery.validate_response_shape",
        lambda *_a, **_k: (False, []),
    )
    monkeypatch.setattr(
        "agent.turn_response_check.record_response_usage",
        lambda *_a, **_k: SimpleNamespace(compression_attempts=0, rearmed=False),
    )
    monkeypatch.setattr(
        "agent.relay_llm.complete_logical_call",
        lambda *_a, **_k: None,
    )

    verdict = _check(agent, SimpleNamespace(choices=[object()], model="accepted-model"))

    assert verdict.action == "break"
    assert agent._delegate_successful_llm_route == ("accepted-model", "anthropic")
