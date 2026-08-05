from types import SimpleNamespace
from unittest.mock import MagicMock

import run_agent
from agent import relay_llm
from tools.delegate_tool import _run_single_child


def _codex_response(*output):
    return SimpleNamespace(
        output=list(output),
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        status="completed",
        model="gpt-5-codex",
    )


def test_delegated_child_terminal_slot_blocks_relay_tool_reinjection(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **_kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run a command.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})

    child = run_agent.AIAgent(
        model="gpt-5-codex",
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="test-token",
        quiet_mode=True,
        max_iterations=2,
        skip_context_files=True,
        skip_memory=True,
    )
    child._delegate_depth = 1
    child._cleanup_task_resources = lambda _task_id: None
    child._persist_session = lambda _messages, _history=None: None
    child._save_trajectory = lambda _messages, _user_message, _completed: None
    child._handle_max_iterations = MagicMock(
        side_effect=AssertionError("out-of-budget finalization was called")
    )
    child._api_max_retries = 2

    requests = []
    responses = [
        _codex_response(
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="terminal",
                arguments="{}",
            )
        ),
        _codex_response(
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="DONE")],
            )
        ),
    ]

    def provider_call(api_kwargs):
        if not api_kwargs.get("tools"):
            relay_body = {
                **api_kwargs,
                "tools": [{"type": "function", "name": "terminal"}],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            }
            api_kwargs = relay_llm._provider_request(
                api_kwargs,
                SimpleNamespace(content=relay_body),
                relay_request_body=api_kwargs,
                codec_baseline_body=api_kwargs,
                metadata={"api_mode": "custom"},
            )
        requests.append(api_kwargs)
        assert bool(api_kwargs.get("tools")) is (len(requests) == 1)
        if len(requests) > 1:
            assert not {
                "tools", "tool_choice", "parallel_tool_calls", "toolConfig"
            } & api_kwargs.keys()
        if len(requests) == 2:
            raise RuntimeError("transient provider failure")
        return responses.pop(0)

    monkeypatch.setattr(child, "_interruptible_api_call", provider_call)
    monkeypatch.setattr(
        "agent.conversation_loop.jittered_backoff", lambda *_args, **_kwargs: 0
    )

    def execute_tool_calls(assistant_message, messages, *_args):
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"ok":true}',
                }
            )

    monkeypatch.setattr(child, "_execute_tool_calls", execute_tool_calls)

    result = _run_single_child(
        task_index=0,
        goal="Finish the task",
        child=child,
        parent_agent=SimpleNamespace(_current_task_id=None),
    )

    assert len(requests) == 3
    assert result["api_calls"] == 2
    assert result["completed"] is True
    assert result["turn_exit_reason"].startswith("text_response(")
    assert result["status"] == "completed"
    assert result["summary"] == "DONE"
    child._handle_max_iterations.assert_not_called()


def test_delegated_child_unfinished_scratchpad_exhausts_terminal_slot(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **_kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})

    child = run_agent.AIAgent(
        model="gpt-5-codex",
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="test-token",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    child._delegate_depth = 1
    child._cleanup_task_resources = lambda _task_id: None
    child._persist_session = lambda _messages, _history=None: None
    child._save_trajectory = lambda _messages, _user_message, _completed: None
    child._handle_max_iterations = MagicMock(
        side_effect=AssertionError("out-of-budget finalization was called")
    )
    monkeypatch.setattr(
        child,
        "_interruptible_api_call",
        lambda _kwargs: _codex_response(
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        text="Draft\n<REASONING_SCRATCHPAD>still working",
                    )
                ],
            )
        ),
    )

    result = _run_single_child(
        task_index=0,
        goal="Finish the task",
        child=child,
        parent_agent=SimpleNamespace(_current_task_id=None),
    )

    assert result["status"] == "failed"
    assert result["turn_exit_reason"] == "terminal_slot_exhausted(1/1)"
    assert result["api_calls"] == 1
    assert child._incomplete_scratchpad_retries == 0
    assert child._handle_max_iterations.call_count == 0
