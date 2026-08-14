"""Privacy contract for opt-in streamed stage-latency diagnostics (#82)."""

from __future__ import annotations

import json


def _config(enabled: bool) -> dict[str, object]:
    return {"observability": {"stream_stage_latency": {"enabled": enabled}}}


def test_stream_stage_latency_is_default_off(monkeypatch, tmp_path):
    from agent import stream_stage_diagnostics as diagnostics
    from hermes_cli import config
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(False))

    assert DEFAULT_CONFIG["observability"]["stream_stage_latency"]["enabled"] is False
    assert diagnostics.start_attempt() is None
    assert not (tmp_path / "observability").exists()


def test_enabled_stream_stage_latency_persists_only_stage_timings(
    monkeypatch, tmp_path
):
    from agent import stream_stage_diagnostics as diagnostics
    from hermes_cli import config

    sentinel = "ISSUE82-PRIVATE-STREAM-CONTENT"
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))

    attempt = diagnostics.start_attempt()
    assert attempt is not None
    attempt.observe_chunk()
    with attempt.stage("callback"):
        _callback_result = sentinel
    attempt.observe_chunk()
    attempt.finish()

    records_path = tmp_path / "observability" / "stream_stage_latency.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["schema"] == "hermes.stream_stage_latency.v1"
    stages = {stage["name"]: stage for stage in record["stages"]}
    assert {"first_byte", "later_chunk", "callback", "aggregate"} <= set(stages)
    assert all(
        set(stage) <= {"name", "duration_ms", "events"}
        and stage["duration_ms"] >= 0
        and stage["events"] >= 1
        for stage in record["stages"]
    )
    assert stages["first_byte"]["events"] == 1
    assert stages["later_chunk"]["events"] == 1
    assert stages["callback"]["events"] == 1
    assert stages["aggregate"]["events"] == 2
    assert sentinel not in json.dumps(record, sort_keys=True)


def test_enabled_stream_stage_latency_captures_a_real_stream(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace
    from unittest.mock import patch

    from agent import stream_stage_diagnostics as diagnostics
    from hermes_cli import config
    from run_agent import AIAgent

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))

    def chunk(content=None, finish_reason=None):
        delta = SimpleNamespace(
            content=content,
            reasoning_content=None,
            reasoning=None,
            tool_calls=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
            model="test/model",
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: iter(
                    [
                        chunk("ISSUE82-STREAM-TEXT-ONE"),
                        chunk("ISSUE82-STREAM-TEXT-TWO"),
                        chunk(finish_reason="stop"),
                    ]
                )
            )
        )
    )
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent.stream_delta_callback = lambda _text: None

    with (
        patch.object(agent, "_create_request_openai_client", return_value=client),
        patch.object(agent, "_close_request_openai_client"),
    ):
        response = agent._interruptible_streaming_api_call({})

    assert response.choices[0].message.content == (
        "ISSUE82-STREAM-TEXT-ONEISSUE82-STREAM-TEXT-TWO"
    )
    records_path = tmp_path / "observability" / "stream_stage_latency.jsonl"
    record = json.loads(records_path.read_text().splitlines()[0])
    stages = {stage["name"]: stage for stage in record["stages"]}
    assert {"first_byte", "later_chunk", "callback", "aggregate"} <= set(stages)
    assert "ISSUE82-STREAM-TEXT" not in json.dumps(record, sort_keys=True)


def test_enabled_stream_stage_latency_captures_codex_responses_stream(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from agent import stream_stage_diagnostics as diagnostics
    from hermes_cli import config
    from run_agent import AIAgent

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))

    sentinel_one = "ISSUE82-CODEX-STREAM-ONE"
    sentinel_two = "ISSUE82-CODEX-STREAM-TWO"
    output_item = SimpleNamespace(
        type="message",
        phase="final_answer",
        status="completed",
        content=[SimpleNamespace(type="output_text", text=sentinel_one + sentinel_two)],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="message", phase="final_answer"),
        ),
        SimpleNamespace(type="response.output_text.delta", delta=sentinel_one),
        SimpleNamespace(type="response.output_text.delta", delta=sentinel_two),
        SimpleNamespace(type="response.output_item.done", item=output_item),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="resp_test",
                status="completed",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            ),
        ),
    ]

    class _Stream:
        def __iter__(self):
            return iter(events)

        def close(self):
            pass

    agent = AIAgent(
        api_key="test-key",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        model="gpt-5-codex",
        api_mode="codex_responses",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    delivered = []
    agent.stream_delta_callback = delivered.append
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_kwargs: _Stream())
    )

    response = agent._run_codex_stream(
        {"model": "gpt-5-codex", "input": []},
        client=client,
    )

    assert response.output_text == sentinel_one + sentinel_two
    assert delivered == [sentinel_one, sentinel_two]
    records_path = tmp_path / "observability" / "stream_stage_latency.jsonl"
    record = json.loads(records_path.read_text().splitlines()[0])
    stages = {stage["name"]: stage for stage in record["stages"]}
    assert {"first_byte", "later_chunk", "callback", "aggregate"} <= set(stages)
    assert stages["first_byte"]["events"] == 1
    assert stages["later_chunk"]["events"] >= 1
    assert stages["callback"]["events"] >= 2
    assert "ISSUE82-CODEX-STREAM" not in json.dumps(record, sort_keys=True)


def test_enabled_stream_stage_latency_captures_bedrock_converse_stream(
    monkeypatch, tmp_path
):
    import pytest
    from unittest.mock import MagicMock, patch

    pytest.importorskip("botocore")

    from agent import stream_stage_diagnostics as diagnostics
    from hermes_cli import config
    from run_agent import AIAgent

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))

    sentinel_one = "ISSUE82-BEDROCK-STREAM-ONE"
    sentinel_two = "ISSUE82-BEDROCK-STREAM-TWO"
    events = [
        {"contentBlockDelta": {"delta": {"text": sentinel_one}}},
        {"contentBlockDelta": {"delta": {"text": sentinel_two}}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2}}},
    ]
    client = MagicMock()
    client.converse_stream.return_value = {"stream": iter(events)}
    agent = AIAgent(
        api_key="test-key",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        provider="bedrock",
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "bedrock_converse"
    delivered = []
    agent.stream_delta_callback = delivered.append

    with patch(
        "agent.bedrock_adapter._get_bedrock_runtime_client",
        return_value=client,
    ):
        response = agent._interruptible_streaming_api_call(
            {
                "modelId": agent.model,
                "messages": [],
                "__bedrock_region__": "us-east-1",
            }
        )

    assert response.choices[0].message.content == sentinel_one + sentinel_two
    assert delivered == [sentinel_one, sentinel_two]
    records_path = tmp_path / "observability" / "stream_stage_latency.jsonl"
    record = json.loads(records_path.read_text().splitlines()[0])
    stages = {stage["name"]: stage for stage in record["stages"]}
    assert {"first_byte", "later_chunk", "callback", "aggregate"} <= set(stages)
    assert stages["first_byte"]["events"] == 1
    assert stages["later_chunk"]["events"] >= 1
    assert stages["callback"]["events"] >= 2
    assert "ISSUE82-BEDROCK-STREAM" not in json.dumps(record, sort_keys=True)


def test_disabled_stream_stage_latency_calls_real_callback_directly(
    monkeypatch, tmp_path
):
    import inspect
    from types import SimpleNamespace
    from unittest.mock import patch

    from agent import stream_stage_diagnostics as diagnostics
    from hermes_cli import config
    from run_agent import AIAgent

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(False))

    delta_one = "ISSUE82-DISABLED-STREAM-ONE"
    delta_two = "ISSUE82-DISABLED-STREAM-TWO"

    def chunk(content=None, finish_reason=None):
        delta = SimpleNamespace(
            content=content,
            reasoning_content=None,
            reasoning=None,
            tool_calls=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
            model="test/model",
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: iter(
                    [chunk(delta_one), chunk(delta_two), chunk(finish_reason="stop")]
                )
            )
        )
    )
    callback_calls = []

    def direct_callback(text):
        caller = inspect.currentframe().f_back
        callback_calls.append((text, caller.f_code.co_name))

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._fire_stream_delta = direct_callback

    with (
        patch.object(agent, "_create_request_openai_client", return_value=client),
        patch.object(agent, "_close_request_openai_client"),
    ):
        response = agent._interruptible_streaming_api_call({})

    assert response.choices[0].message.content == delta_one + delta_two
    assert callback_calls == [
        (delta_one, "_call_chat_completions"),
        (delta_two, "_call_chat_completions"),
    ]
    assert not (tmp_path / "observability").exists()
