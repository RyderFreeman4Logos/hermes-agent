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
