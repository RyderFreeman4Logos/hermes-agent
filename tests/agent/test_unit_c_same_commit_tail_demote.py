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


def test_public_compaction_archives_rewritten_multimodal_tail_and_keeps_newest_three():
    from hermes_state import SessionDB
    from agent.conversation_compression import compress_context
    from run_agent import AIAgent

    original = _tail_fixture()
    changed = next(
        message for message in original if message.get("tool_call_id") == "call_tail"
    )
    changed["content"] = [
        {
            "type": "text",
            "text": "MULTIMODAL_ARCHIVE_TOKEN\n" + ("large screenshot detail " * 2_500),
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AA=="},
        },
    ]

    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "interspersed.db"
        db = SessionDB(db_path=db_path)
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
            compressed, _ = compress_context(
                agent,
                list(original),
                approx_tokens=estimate_messages_tokens_rough(original),
                system_message="sys",
            )

        newest_three = [
            {
                key: message.get(key)
                for key in ("role", "content", "tool_calls", "tool_call_id")
                if message.get(key) is not None
            }
            for message in original[-3:]
        ]
        assert [
            {
                key: message.get(key)
                for key in ("role", "content", "tool_calls", "tool_call_id")
                if message.get(key) is not None
            }
            for message in compressed[-3:]
        ] == newest_three
        db.close()

        reloaded = SessionDB(db_path=db_path)
        active = reloaded.get_messages_as_conversation(session_id)
        assert [
            {
                key: message.get(key)
                for key in ("role", "content", "tool_calls", "tool_call_id")
                if message.get(key) is not None
            }
            for message in active[-3:]
        ] == newest_three
        display_history = reloaded.get_messages_as_conversation(
            session_id, include_compacted=True
        )
        assert any(
            message.get("tool_call_id") == "call_tail"
            and message.get("content") == changed["content"]
            for message in display_history
        )
        rows = reloaded._conn.execute(
            "SELECT tool_call_id, active, compacted FROM messages "
            "WHERE session_id = ? AND role = 'tool' ORDER BY id",
            (session_id,),
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("call_tail", 0, 1),
            ("call_recent", 0, 0),
            ("call_tail", 1, 0),
            ("call_recent", 1, 0),
        ]
        reloaded.close()
