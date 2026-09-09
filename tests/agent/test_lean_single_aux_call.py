"""Lean compaction keeps one primary summary request and bounded chunk harvest.

Contract (#165): the primary summary is one ``call_llm`` request; oversized
lean coverage is supplemented by bounded per-chunk digest requests, with
ordered, cancellable harvest and the LLM-free anchor/recovery sections.
"""

from unittest.mock import patch, MagicMock

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _LEAN_ANCHOR_HEADING,
    _LEAN_DIGEST_MAX_CHUNKS,
    _LEAN_RECOVERY_HEADING,
    _LEAN_SESSION_LOG_HEADING,
)


def _mk_compressor(**overrides):
    kwargs = dict(
        model="test/model",
        threshold_percent=0.85,
        protect_first_n=2,
        protect_last_n=2,
        quiet_mode=True,
        tail_mode="lean",
    )
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        c = ContextCompressor(**kwargs)
        _ = c.context_length
    c._session_id = "sess-lean-1call"
    return c


def _llm_response(text):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = text
    return resp


def _big_region(n_rounds=60, tool_chars=6_000):
    """Synthetic compacted region: user + assistant/tool rounds, ~360K chars."""
    turns = [
        {"role": "user", "content": "Please fix PR #12345 in agent/foo.py"},
    ]
    for i in range(n_rounds):
        turns.append({
            "role": "assistant",
            "content": f"Working on step {i}: editing agent/foo.py line {i}",
        })
        turns.append({
            "role": "tool",
            "tool_call_id": f"tc{i}",
            "tool_name": "terminal",
            "content": f"round {i} output: " + ("x" * tool_chars),
        })
    return turns


SUMMARY_BODY = (
    "## Historical Task Snapshot\nUser asked: 'Please fix PR #12345 in agent/foo.py'\n\n"
    "## Goal\nFix the bug.\n\n"
    "## Completed Actions\n1. EDIT agent/foo.py — done [tool: patch]\n\n"
    f"{_LEAN_SESSION_LOG_HEADING}\n- Edited agent/foo.py; PR #12345; ran pytest.\n"
)


class TestLeanSingleAuxiliaryCall:
    def test_one_primary_call_with_bounded_chunk_harvest(self):
        c = _mk_compressor()
        turns = _big_region()
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as main_call, patch(
            "agent.auxiliary_client.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as aux_call:
            summary = c._generate_summary(turns)
        assert summary is not None
        # The primary summary remains exactly one request; #165 intentionally
        # adds bounded per-chunk digest requests for oversized lean coverage.
        assert main_call.call_count == 1
        assert 1 <= aux_call.call_count <= _LEAN_DIGEST_MAX_CHUNKS

    def test_session_log_heading_lands_in_summary(self):
        c = _mk_compressor()
        turns = _big_region(n_rounds=5)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ):
            summary = c._generate_summary(turns)
        assert _LEAN_SESSION_LOG_HEADING in summary

    def test_prompt_requests_session_log_section(self):
        c = _mk_compressor()
        turns = _big_region(n_rounds=5)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as mock_call:
            c._generate_summary(turns)
        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert _LEAN_SESSION_LOG_HEADING in prompt
        # The digest HARD RULES carried over into the single request.
        assert "PRESERVE EXACTLY" in prompt

    def test_anchor_index_and_recovery_footer_present(self):
        c = _mk_compressor()
        turns = _big_region(n_rounds=5)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ):
            summary = c._generate_summary(turns)
        assert _LEAN_ANCHOR_HEADING in summary
        assert _LEAN_RECOVERY_HEADING in summary
        assert "session_search" in summary

    def test_oversized_region_sampled_not_split_into_more_requests(self):
        c = _mk_compressor()
        turns = _big_region(n_rounds=120, tool_chars=6_000)  # ~720K chars raw
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response(SUMMARY_BODY),
        ) as mock_call:
            c._generate_summary(turns)
        assert mock_call.call_count == 1
        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        # Bounded input with explicit elision markers, not a second request.
        assert len(prompt) <= c._SUMMARY_INPUT_MAX_CHARS + 20_000
        assert "chars elided" in prompt


class TestSampledSummaryInput:
    def test_small_input_passes_through(self):
        content = "abc" * 100
        assert ContextCompressor._sample_summary_input(content) == content

    def test_sampling_is_bounded_ordered_and_marked(self):
        # Distinct decade markers let us verify oldest-to-newest order and
        # uniform coverage across the whole region.
        content = "".join(
            f"<seg{i:02d}>" + ("x" * 50_000) for i in range(10)
        )
        out = ContextCompressor._sample_summary_input(content)
        assert len(out) <= ContextCompressor._SUMMARY_INPUT_MAX_CHARS
        assert "chars elided" in out
        seen = [i for i in range(10) if f"<seg{i:02d}>" in out]
        # Coverage reaches past the head AND includes the newest end.
        assert seen == sorted(seen)
        assert any(i >= 5 for i in seen)
        assert ("<seg09>" in out) or (content[-500:] in out)

    def test_legacy_mode_keeps_head_tail_bound(self):
        c = _mk_compressor(tail_mode="legacy")
        turns = _big_region(n_rounds=120, tool_chars=6_000)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_llm_response("## Historical Task Snapshot\nNone."),
        ) as mock_call:
            c._generate_summary(turns)
        assert mock_call.call_count == 1
        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert "summary input truncated" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
