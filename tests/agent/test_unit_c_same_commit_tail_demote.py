"""Unit C: coalesce post-summary protected-tail demotion with compression."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.model_metadata import estimate_messages_tokens_rough


def _tail_fixture(tool_chars: int = 48_000) -> list[dict]:
    """Middle turns plus a bulky-but-not-pressure-sized retained tail.

    Sized so last-4 tokens sit between ``tail_token_budget`` and
    ``1.5 * tail_token_budget`` on the official pin estimator: pressure
    demote (1.5x) must not already shrink it; Unit C (1.0x) must.
    """
    return [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Investigate this task."},
        {"role": "assistant", "content": "I will inspect the repository."},
        {"role": "user", "content": "Record the important findings."},
        {"role": "assistant", "content": "The first pass is complete."},
        {"role": "user", "content": "Continue with the final report."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_tail",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"report.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_tail",
            "content": "REPORT_START\n" + ("unique report line " * 6_000)[:tool_chars],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_recent",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"summary.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_recent",
            "content": "recent report is intact",
        },
        {"role": "user", "content": "Now summarize the report for me."},
    ]


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=128_000,
    ):
        compressor = ContextCompressor(
            model="openai-codex/gpt-test",
            threshold_percent=0.50,
            summary_target_ratio=0.20,
            protect_first_n=3,
            protect_last_n=20,
            quiet_mode=True,
            config_context_length=128_000,
        )
    compressor._generate_summary = lambda *args, **kwargs: "compact middle summary"
    return compressor


def test_post_summary_tail_demote_preserves_active_task_text():
    compressor = _compressor()
    messages = _tail_fixture()
    before = estimate_messages_tokens_rough(messages)

    tail_tool = next(message for message in messages if message.get("role") == "tool")
    assert compressor.tail_token_budget < estimate_messages_tokens_rough(messages[-4:])
    assert estimate_messages_tokens_rough(messages[-4:]) < int(
        compressor.tail_token_budget * 1.5
    )
    assert tail_tool["content"].startswith("REPORT_START")

    compressed = compressor.compress(list(messages), current_tokens=before)

    compressed_tool = next(
        message for message in compressed if message.get("tool_call_id") == "call_tail"
    )
    assert compressed_tool["content"] != tail_tool["content"]
    assert compressed_tool["content"].startswith("[read_file]")
    assert compressed[-1]["content"] == messages[-1]["content"]


def test_post_summary_demote_is_coalesced_into_one_durable_commit():
    from hermes_state import SessionDB
    from agent.conversation_compression import compress_context
    from run_agent import AIAgent

    messages = _tail_fixture()
    with TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "unit-c.db")
        session_id = "20260814_040200_unitc"
        db.create_session(session_id, "cli", model="test/model")
        for message in messages:
            db.append_message(
                session_id=session_id,
                role=message["role"],
                content=message.get("content"),
                tool_calls=message.get("tool_calls"),
                tool_call_id=message.get("tool_call_id"),
            )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), patch(
            "agent.context_compressor.get_model_context_length",
            return_value=128_000,
        ):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=db,
                session_id=session_id,
                skip_context_files=True,
                skip_memory=True,
            )
            agent.compression_in_place = True
            agent._compression_feasibility_checked = True
            agent.context_compressor._generate_summary = (
                lambda *args, **kwargs: "compact middle summary"
            )
            assert agent.context_compressor.proactive_prune_tokens == 0

            with patch.object(
                db, "archive_and_compact", wraps=db.archive_and_compact
            ) as archive_and_compact:
                compress_context(
                    agent,
                    list(messages),
                    approx_tokens=estimate_messages_tokens_rough(messages),
                    system_message="sys",
                )

        assert archive_and_compact.call_count == 1
        committed_messages = archive_and_compact.call_args.args[1]
        committed_tool = next(
            message
            for message in committed_messages
            if message.get("tool_call_id") == "call_tail"
        )
        assert committed_tool["content"].startswith("[read_file]")
        assert committed_messages[-1]["content"] == messages[-1]["content"]

        # The retained tool row was rewritten by Unit C.  Its original is
        # archived history, not an unchanged rewind duplicate: normal search
        # and display history must still recover the raw bytes.
        hits = db.search_messages("REPORT_START")
        assert len(hits) == 1
        display_history = db.get_messages_as_conversation(
            session_id, include_compacted=True
        )
        assert any(
            message.get("role") == "tool"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("REPORT_START")
            for message in display_history
        )
        raw_original = db._conn.execute(
            "SELECT active, compacted FROM messages "
            "WHERE session_id = ? AND role = 'tool' AND content LIKE 'REPORT_START%'",
            (session_id,),
        ).fetchone()
        assert tuple(raw_original) == (0, 1)

        unchanged_hits = db.search_messages("recent report is intact")
        assert len(unchanged_hits) == 1
        unchanged_original = db._conn.execute(
            "SELECT active, compacted FROM messages "
            "WHERE session_id = ? AND role = 'tool' AND content = ? ORDER BY id LIMIT 1",
            (session_id, "recent report is intact"),
        ).fetchone()
        assert tuple(unchanged_original) == (0, 0)


def test_archive_classifies_interspersed_multimodal_tail_by_payload():
    from hermes_state import SessionDB

    original = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "changed", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "changed",
            "content": [
                {"type": "text", "text": "multimodal archive token"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "same", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "same", "content": "UNCHANGED_TAIL_TOKEN"},
        {"role": "user", "content": "continue"},
    ]
    retained = [dict(message) for message in original]
    retained[1] = {
        **retained[1],
        "content": [{"type": "text", "text": "[screenshot removed]"}],
    }

    with TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "interspersed.db")
        session_id = "20260914_186_interspersed"
        db.create_session(session_id, "cli", model="test/model")
        for message in original:
            db.append_message(
                session_id=session_id,
                role=message["role"],
                content=message.get("content"),
                tool_calls=message.get("tool_calls"),
                tool_call_id=message.get("tool_call_id"),
            )

        db.archive_and_compact(session_id, retained, tail_count=len(retained))

        assert len(db.search_messages("UNCHANGED_TAIL_TOKEN")) == 1
        display_history = db.get_messages_as_conversation(
            session_id, include_compacted=True
        )
        assert any(
            message.get("tool_call_id") == "changed"
            and message.get("content") == original[1]["content"]
            for message in display_history
        )
        rows = db._conn.execute(
            "SELECT tool_call_id, active, compacted FROM messages "
            "WHERE session_id = ? AND role = 'tool' ORDER BY id",
            (session_id,),
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("changed", 0, 1),
            ("same", 0, 0),
            ("changed", 1, 0),
            ("same", 1, 0),
        ]
