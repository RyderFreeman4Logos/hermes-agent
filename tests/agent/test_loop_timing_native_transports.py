"""Wire contracts for durable loop-timing rows on native destination transports."""

import pytest

from agent.transports.anthropic import AnthropicTransport
from agent.transports.bedrock import BedrockTransport
from agent.transports.codex import ResponsesApiTransport


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
