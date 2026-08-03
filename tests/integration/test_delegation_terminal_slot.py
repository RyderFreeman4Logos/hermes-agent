from types import SimpleNamespace
from unittest.mock import MagicMock

import run_agent
from tools.delegate_tool import _run_single_child


def _codex_response(*output):
    return SimpleNamespace(
        output=list(output),
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        status="completed",
        model="gpt-5-codex",
    )


def test_delegated_codex_child_reserves_last_owned_iteration_for_terminal_text(
    monkeypatch,
):
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

    published = []

    def progress(event_type, *_args, **kwargs):
        if event_type == "subagent.complete":
            published.append(kwargs)

    child = run_agent.AIAgent(
        model="gpt-5-codex",
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="test-token",
        quiet_mode=True,
        max_iterations=2,
        tool_progress_callback=progress,
        skip_context_files=True,
        skip_memory=True,
    )
    child._cleanup_task_resources = lambda _task_id: None
    child._persist_session = lambda _messages, _history=None: None
    child._save_trajectory = lambda _messages, _user_message, _completed: None
    child._handle_max_iterations = MagicMock(
        side_effect=AssertionError("budget-outside finalization was called")
    )

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
        requests.append(api_kwargs)
        if len(requests) == 1:
            assert api_kwargs.get("tools")
        else:
            assert "tools" not in api_kwargs
        return responses.pop(0)

    monkeypatch.setattr(child, "_interruptible_api_call", provider_call)

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

    assert len(requests) == 2
    assert result["api_calls"] == 2
    assert result["completed"] is True
    assert result["turn_exit_reason"].startswith("text_response(")
    assert result["status"] == "completed"
    assert result["summary"] == "DONE"
    assert len(published) == 1
    assert published[0]["completed"] is True
    assert published[0]["turn_exit_reason"].startswith("text_response(")
    child._handle_max_iterations.assert_not_called()
