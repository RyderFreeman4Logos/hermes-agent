"""Wire contracts for durable loop-timing rows on native destination transports."""

import copy

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.transports.anthropic import AnthropicTransport
from agent.transports.bedrock import BedrockTransport
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.codex import ResponsesApiTransport
from providers.base import ProviderProfile
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


@pytest.mark.parametrize(
    "profile",
    [None, ProviderProfile(name="identity-profile")],
    ids=["legacy-no-profile", "identity-profile"],
)
def test_chat_profile_and_legacy_first_send_coalesce_timing_without_mutation(monkeypatch, profile):
    """W2: both Chat builder modes keep S,U,T as one ordered user request."""
    history = _history()
    original = copy.deepcopy(history)
    transport = ChatCompletionsTransport()
    kwargs = transport.build_kwargs(
        model="chat-test", messages=history, provider_profile=profile,
        provider_name="identity-profile" if profile else "unknown-profile",
    )
    assert history == original
    assert [row["role"] for row in kwargs["messages"]] == ["system", "user"]
    assert kwargs["messages"][0]["content"] == "Stable instructions"
    assert kwargs["messages"][1]["content"].startswith("Ask")
    assert "[Agent loop timing] Current loop start: now" in kwargs["messages"][1]["content"]

    monkeypatch.setattr(
        "providers.get_provider_profile",
        lambda _name: profile,
    )
    agent = _make_local_agent(
        monkeypatch, api_mode="chat_completions",
        provider="identity-profile" if profile else "unknown-profile", model="chat-test",
    )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _completed_chat_response()

    result = agent.run_conversation("chat first send")

    assert result["completed"] is True
    sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert [row["role"] for row in sent] == ["system", "user"]
    assert sent[0]["content"] == "Stable instructions"
    assert sent[1]["content"].startswith("chat first send")
    assert sent[1]["content"].count("[Agent loop timing]") == 1


def test_chat_send_keeps_two_hidden_timing_rows_through_mixed_content(monkeypatch):
    """W6: the real Chat call demotes only timing rows through mixed blocks.

    Empty string/list/``None`` neighbours are intentional public history
    shapes.  Their presence must neither stringify into the payload nor make
    either durable timing event disappear or reorder.
    """
    agent = _make_local_agent(
        monkeypatch, api_mode="chat_completions", provider="unknown-profile", model="chat-test",
    )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _completed_chat_response()
    timing_one = "[Agent loop timing] first durable timing"
    timing_two = "[Agent loop timing] second durable timing"
    history = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Stable instructions", "cache_control": {"type": "ephemeral"}}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "marked user", "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "user", "content": ""},
        {"role": "assistant", "content": None},
        {
            "role": "system", "content": timing_one, "display_kind": "hidden",
            "display_metadata": {"loop_timing_turn_id": "timing-one"},
        },
        {"role": "user", "content": []},
        {
            "role": "system",
            "content": [{"type": "text", "text": timing_two}],
            "display_kind": "hidden",
            "display_metadata": {"loop_timing_turn_id": "timing-two"},
        },
    ]
    original = copy.deepcopy(history)

    result = agent.run_conversation("current user", conversation_history=history)

    assert result["completed"] is True
    assert history == original
    wire = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert all(row["role"] != "system" or "[Agent loop timing]" not in _wire_text(row) for row in wire)
    wire_text = "\n".join(_wire_text(row) for row in wire)
    durable = [timing_one, timing_two]
    current = [
        row for row in result["messages"]
        if row.get("display_kind") == "hidden" and "[Agent loop timing]" in _wire_text(row)
    ]
    assert len(current) == 3
    durable.append(_wire_text(current[-1]))
    positions = []
    for timing in durable:
        matched = [row for row in wire if timing in _wire_text(row)]
        assert len(matched) == 1
        assert matched[0]["role"] == "user"
        positions.append(wire_text.index(timing))
    assert positions == sorted(positions)
    assert "None" not in wire_text
    assert all(
        block.get("text")
        for row in wire
        if isinstance(row.get("content"), list)
        for block in row["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    )


class _ClassifiedPolicyError(Exception):
    """Local non-retryable error that follows the configured fallback branch."""


@pytest.mark.parametrize("fallback_mode", [
    "anthropic_messages", "bedrock_converse", "codex_responses",
])
def test_chat_policy_fallback_keeps_timing_at_each_native_send(monkeypatch, fallback_mode):
    """W4: an actual Chat retry restart preserves its event at every native edge."""
    agent = _make_local_agent(
        monkeypatch, api_mode="chat_completions", provider="unknown-profile", model="chat-test",
    )
    agent._fallback_chain = [{
        "provider": "custom", "model": "fallback-test", "api_key": "fallback-key",
        "base_url": "https://fallback.invalid/v1", "api_mode": fallback_mode,
    }]
    agent._fallback_index = 0
    agent.client = MagicMock()
    primary_client = agent.client
    primary_client.chat.completions.create.side_effect = _ClassifiedPolicyError(
        "This content was flagged for possible cybersecurity risk."
    )
    captured = []
    resolved_client = MagicMock()
    resolved_client.base_url = "https://fallback.invalid/v1"
    resolved_client.api_key = "fallback-key"

    if fallback_mode == "anthropic_messages":
        native_client = MagicMock()
        native_client.messages.create.side_effect = lambda **kwargs: (
            captured.append(copy.deepcopy(kwargs)) or _completed_anthropic_response("fallback done")
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.build_anthropic_client", lambda *_args, **_kwargs: native_client,
        )
    elif fallback_mode == "bedrock_converse":
        native_client = MagicMock()
        native_client.converse.side_effect = lambda **kwargs: (
            captured.append(copy.deepcopy(kwargs)) or _completed_bedrock_response("fallback done")
        )
        monkeypatch.setattr(
            "agent.bedrock_adapter._get_bedrock_runtime_client", lambda _region: native_client,
        )
    else:
        resolved_client.responses.create.side_effect = lambda **kwargs: (
            captured.append(copy.deepcopy(kwargs)) or _completed_responses_stream("fallback done")
        )

    monkeypatch.setattr(
        "agent.chat_completion_helpers._fallback_entry_unavailable_without_network", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client", lambda *_args, **_kwargs: (resolved_client, "fallback-test"),
    )
    monkeypatch.setattr(
        "hermes_cli.model_normalize.normalize_model_for_provider", lambda model, _provider: model,
    )

    result = agent.run_conversation("retry on native fallback")

    assert result["completed"] is True
    assert result["final_response"] == "fallback done"
    # The public result counts completed calls only; both physical sends are
    # instead established directly at their SDK boundaries below.
    assert result["api_calls"] == 1
    assert primary_client.chat.completions.create.call_count == 1
    assert agent._fallback_activated is True
    assert len(captured) == 1
    timing = [row for row in result["messages"] if row.get("display_kind") == "hidden"]
    assert len(timing) == 1
    timing_text = timing[0]["content"]
    primary_wire = primary_client.chat.completions.create.call_args.kwargs["messages"]
    primary_matching = [row for row in primary_wire if timing_text in _wire_text(row)]
    assert len(primary_matching) == 1
    assert primary_matching[0]["role"] == "user"
    matching = [row for row in _wire_rows(captured[0]) if timing_text in _wire_text(row)]
    assert len(matching) == 1
    assert matching[0]["role"] == "user"
