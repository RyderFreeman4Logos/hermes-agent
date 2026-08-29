"""Regression coverage for long internal-notice compaction (#80449)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _ACTIVE_TASK_MAX_CHARS,
    _estimate_msg_budget_tokens,
)


@pytest.fixture()
def compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=0,
            protect_last_n=3,
            quiet_mode=True,
        )
        _ = instance.context_length
    return instance


def _tool_group(index: int) -> list[dict]:
    call_id = f"call_{index}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "inspect_shard",
                        "arguments": "x" * 440,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"result-{index}:" + "r" * 110,
        },
    ]


def _assert_tool_pairs_are_complete(messages: list[dict]) -> None:
    call_ids = {
        call["id"]
        for message in messages
        for call in message.get("tool_calls") or []
    }
    result_ids = {
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    }
    assert call_ids == result_ids


def test_long_internal_completion_does_not_hoard_the_whole_transcript(
    compressor: ContextCompressor,
) -> None:
    completion = (
        "[ASYNC DELEGATION BATCH COMPLETE — deleg_9dc8cde8]\n"
        + ("A background fan-out finished. Details: " + "x" * 80 + "\n") * 40
    )
    assert len(completion) > _ACTIVE_TASK_MAX_CHARS

    notice_budget = 1_200
    compressor.tail_token_budget = notice_budget
    messages = [
        {
            "role": "user",
            "content": "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted.",
        },
        {"role": "assistant", "content": "acked prior handoff"},
        {"role": "user", "content": completion},
    ]
    for index in range(12):
        messages.extend(_tool_group(index))
    messages.append({"role": "assistant", "content": "visible reply after tools"})

    completion_idx = 2
    assert _estimate_msg_budget_tokens(messages[completion_idx]) <= int(
        notice_budget * 1.5
    )
    head_end = compressor._protect_head_size(messages)
    cut = compressor._find_tail_cut_by_tokens(
        messages,
        head_end,
        token_budget=notice_budget,
    )
    tail_tokens = sum(_estimate_msg_budget_tokens(msg) for msg in messages[cut:])
    assert cut > completion_idx
    assert tail_tokens < sum(
        _estimate_msg_budget_tokens(msg) for msg in messages[completion_idx:]
    )

    with patch.object(compressor, "_generate_summary", return_value=None):
        compressed = compressor.compress(messages, current_tokens=90_000)

    assert len(compressed) < len(messages)
    _assert_tool_pairs_are_complete(compressed)
