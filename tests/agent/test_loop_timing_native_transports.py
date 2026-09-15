"""Wire contracts for durable loop-timing rows on native destination transports."""

import copy

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.transports.anthropic import AnthropicTransport
from agent.transports.bedrock import BedrockTransport
from agent.transports.codex import ResponsesApiTransport
from run_agent import AIAgent


class _LocalEventStream:
    """Small local Responses event stream used at the SDK send boundary."""

    def __init__(self, events):
        self._events = list(events)

    def __iter__(self):
        return iter(self._events)

    def close(self):
        pass


def _completed_chat_response(text="done"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None), finish_reason="stop",
        )],
        model="test/model",
        usage=None,
    )


def _completed_anthropic_response(text="done"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn", usage=None,
    )


def _completed_bedrock_response(text="done"):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }


def _completed_responses_stream(text="done"):
    item = SimpleNamespace(
        type="message", id="msg_local", status="completed",
        content=[SimpleNamespace(type="output_text", text=text)],
    )
    return _LocalEventStream([
        SimpleNamespace(type="response.output_item.done", item=item),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status="completed", id="resp_local", usage=None),
        ),
    ])


def _make_local_agent(monkeypatch, *, api_mode, provider, model):
    """Build an AIAgent while replacing only its eventual local SDK client."""
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock)
    with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()):
        agent = AIAgent(
            model=model, provider=provider, api_mode=api_mode,
            api_key="test-key-1234567890", base_url="https://example.invalid/v1",
            quiet_mode=True, max_iterations=1, skip_context_files=True, skip_memory=True,
        )
    agent._disable_streaming = True
    agent._cached_system_prompt = "Stable instructions"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent._fallback_chain = []
    monkeypatch.setattr(agent, "_flush_messages_to_session_db", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(agent, "_persist_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent, "_save_trajectory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *_args, **_kwargs: None)
    return agent


def _wire_text(row):
    content = row.get("content", "")
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _timing_rows(payload):
    """Return the user-projected timing rows from each supported native wire shape."""
    # The Responses SDK-transform bypass intentionally moves large input lists
    # under extra_body immediately before its physical create() call.
    body = payload.get("extra_body", payload)
    rows = body.get("messages", body.get("input", []))
    return [row for row in rows if "[Agent loop timing]" in _wire_text(row)]


def _wire_rows(payload):
    body = payload.get("extra_body", payload)
    return body.get("messages", body.get("input", []))


def _history():
    return [
        {"role": "system", "content": "Stable instructions"},
        {"role": "user", "content": "Ask"},
        {
            "role": "system",
            "content": "[Agent loop timing] Current loop start: now",
            "display_kind": "hidden",
            "display_metadata": {"loop_timing_turn_id": "turn-current"},
        },
    ]


@pytest.mark.parametrize(
    ("transport", "model", "destination"),
    [
        (AnthropicTransport(), "claude-test", "anthropic"),
        (BedrockTransport(), "bedrock-test", "bedrock"),
        (ResponsesApiTransport(), "gpt-test", "responses"),
    ],
)
def test_native_destination_projects_hidden_timing_as_user(transport, model, destination):
    """The same transport builder serves native fallback after route activation."""
    payload = transport.build_kwargs(model=model, messages=_history())

    if destination == "anthropic":
        assert payload["system"] == "Stable instructions"
        wire = payload["messages"]
    elif destination == "bedrock":
        assert [block["text"] for block in payload["system"]] == ["Stable instructions"]
        wire = payload["messages"]
    else:
        wire = payload["input"]

    timing = [item for item in wire if "[Agent loop timing]" in str(item.get("content", ""))]
    assert len(timing) == 1
    assert timing[0]["role"] == "user"


def test_two_real_agent_turns_project_each_native_destination_without_reordering():
    """Two real turns retain T1/T2 before each native request projection."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    done = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop")],
        model="test/model", usage=None,
    )
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [done, done]
    agent._cached_system_prompt = "Stable instructions"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent._fallback_chain = []
    with (
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("first native turn")
        second = agent.run_conversation("second native turn", conversation_history=first["messages"])

    timing = [row for row in second["messages"] if row.get("display_kind") == "hidden"]
    assert len(timing) == 2
    assert len({row["display_metadata"]["loop_timing_turn_id"] for row in timing}) == 2
    for transport, model, destination in (
        (AnthropicTransport(), "claude-test", "anthropic"),
        (BedrockTransport(), "bedrock-test", "bedrock"),
        (ResponsesApiTransport(), "gpt-test", "responses"),
    ):
        payload = transport.build_kwargs(model=model, messages=second["messages"])
        wire = payload["input"] if destination == "responses" else payload["messages"]
        positions = []
        for source_row in timing:
            source_text = source_row["content"]
            matching = [row for row in wire if source_text in _wire_text(row)]
            assert len(matching) == 1
            assert matching[0]["role"] == "user"
            positions.append("\n".join(_wire_text(row) for row in wire).index(source_text))
        assert positions == sorted(positions)


@pytest.mark.parametrize("api_mode,provider,model", [
    ("anthropic_messages", "anthropic", "claude-test"),
    ("bedrock_converse", "bedrock", "anthropic.claude-test-v1:0"),
    ("codex_responses", "openai-codex", "gpt-5-codex"),
])
def test_two_agent_turns_reach_each_native_sdk_send_with_ordered_timing(
    monkeypatch, api_mode, provider, model,
):
    """W1: two public turns reach each native SDK send, not just a builder."""
    agent = _make_local_agent(
        monkeypatch, api_mode=api_mode, provider=provider, model=model,
    )
    captured = []
    if api_mode == "anthropic_messages":
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = lambda **kwargs: (
            captured.append(copy.deepcopy(kwargs)) or _completed_anthropic_response()
        )
        monkeypatch.setattr(agent, "_build_anthropic_client_for_key", lambda _key: fake_client)
    elif api_mode == "bedrock_converse":
        fake_client = MagicMock()
        fake_client.converse.side_effect = lambda **kwargs: (
            captured.append(copy.deepcopy(kwargs)) or _completed_bedrock_response()
        )
        monkeypatch.setattr(
            "agent.bedrock_adapter._get_bedrock_runtime_client", lambda _region: fake_client,
        )
    else:
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **kwargs: (
                    captured.append(copy.deepcopy(kwargs)) or _completed_responses_stream()
                )
            )
        )
        monkeypatch.setattr(agent, "_create_request_openai_client", lambda **_kwargs: fake_client)

    first = agent.run_conversation("first native turn")
    second = agent.run_conversation("second native turn", conversation_history=first["messages"])

    assert second["completed"] is True
    assert len(captured) == 2
    expected = [row["content"] for row in second["messages"] if row.get("display_kind") == "hidden"]
    assert len(expected) == 2
    for index, payload in enumerate(captured):
        positions = []
        for timing_text in expected[:index + 1]:
            matching_rows = [row for row in _wire_rows(payload) if timing_text in _wire_text(row)]
            # Anthropic/Bedrock may legally coalesce an adjacent user prompt and
            # timing row, but the actual native payload must retain the marker
            # exactly once as user content in chronological order.
            assert len(matching_rows) == 1
            assert matching_rows[0]["role"] == "user"
            positions.append("\n".join(_wire_text(row) for row in _wire_rows(payload)).index(timing_text))
        assert positions == sorted(positions)
