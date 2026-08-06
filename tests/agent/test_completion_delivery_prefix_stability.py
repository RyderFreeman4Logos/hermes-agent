"""Completion-delivery turns must not rewrite provider-visible history.

These tests use the real conversation loop, serializer, and SessionDB with a
fake chat-completions client.  No provider request leaves the process.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.replay_cleanup import sanitize_replay_history
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.run_agent.test_run_agent import _mock_response, _mock_tool_call
from tools.process_registry import completion_delivery_prompt, format_process_notification


def _tool_defs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _make_agent(tmp_path: Path, db: SessionDB, session_id: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", tmp_path / ".hermes"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
            session_db=db,
            session_id=session_id,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "BYTE-STABLE SYSTEM"
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.tool_delay = 0
    return agent


def _completion_texts() -> tuple[str, str]:
    event = {
        "type": "completion",
        "session_id": "proc_prefix",
        "command": "pytest -q",
        "exit_code": 0,
        "output": "2 passed",
    }
    canonical = format_process_notification(event)
    assert canonical is not None
    wire = completion_delivery_prompt(event, canonical)
    assert wire is not None and wire != canonical
    return canonical, wire


def _stage_completion(agent: AIAgent, wire: str) -> None:
    agent._pending_cli_user_message = {
        "role": "user",
        "content": wire,
        "_completion_delivery_synthetic": True,
    }


def _capture_client(agent: AIAgent, responses: list) -> list[list[dict]]:
    requests: list[list[dict]] = []

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        return responses.pop(0)

    agent.client.chat.completions.create.side_effect = create
    return requests


def _public_shape(messages: list[dict]) -> list[dict]:
    fields = (
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "name",
        "api_content",
        "display_kind",
    )
    return [{key: row[key] for key in fields if key in row} for row in messages]


def _exact_completion_event_shape(row: dict) -> dict:
    """Return every durable event field, excluding only storage-local fields."""
    return {
        key: value
        for key, value in row.items()
        if key not in {"timestamp", "_db_persisted"}
    }


def _assert_provider_protocol_is_closed(messages: list[dict]) -> None:
    """Pin the strict-provider role/tool invariants used on cold resume."""
    prior_role = None
    outstanding_tool_calls: set[str] = set()
    for row in messages:
        role = row.get("role")
        if role == "system":
            continue
        if role in {"user", "assistant"} and prior_role in {"user", "assistant"}:
            assert role != prior_role
        prior_role = role
        if role == "assistant":
            outstanding_tool_calls.update(
                call.get("id")
                for call in row.get("tool_calls", [])
                if isinstance(call, dict) and call.get("id")
            )
        elif role == "tool":
            assert row.get("tool_call_id") in outstanding_tool_calls
            outstanding_tool_calls.discard(row.get("tool_call_id"))
    assert not outstanding_tool_calls


@pytest.mark.parametrize("with_tool", [False, True])
def test_meaningful_completion_is_atomic_and_next_request_replays_exact_prefix(
    tmp_path, monkeypatch, with_tool
):
    """Text and tool-bearing completions survive atomically and byte-stably."""
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook, **_kw: (
            [{"context": "PLUGIN-CTX"}] if hook == "pre_llm_call" else []
        ),
    )
    db_path = tmp_path / "state.db"
    session_id = f"completion-prefix-{with_tool}"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "working in background"},
        ],
    )
    canonical, wire = _completion_texts()

    try:
        first_history = db.get_messages_as_conversation(session_id)
        first = _make_agent(tmp_path, db, session_id)
        _stage_completion(first, wire)
        if with_tool:
            responses = [
                _mock_response(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        _mock_tool_call(
                            name="web_search", arguments="{}", call_id="call-prefix"
                        )
                    ],
                ),
                _mock_response(content="Build passed", finish_reason="stop"),
            ]
        else:
            responses = [_mock_response(content="Build passed", finish_reason="stop")]
        first_requests = _capture_client(first, responses)

        with patch("run_agent.handle_function_call", return_value="checked"):
            first_result = first.run_conversation(
                wire, conversation_history=first_history, task_id="completion"
            )

        completion_rows = [
            row
            for row in first_result["messages"]
            if row.get("role") == "user" and row.get("content") == canonical
        ]
        assert len(completion_rows) == 1
        assert completion_rows[0]["display_kind"] == "hidden"
        first_wire_completion = next(
            row["content"]
            for row in first_requests[0]
            if row.get("role") == "user" and row.get("content", "").startswith(wire)
        )
        assert completion_rows[0]["api_content"] == first_wire_completion
        assert "_completion_delivery_synthetic" not in completion_rows[0]
        assert first_result["messages"][-1]["content"] == "Build passed"
        expected_event = {
            "role": "user",
            "content": canonical,
            "api_content": first_wire_completion,
            "display_kind": "hidden",
            "display_metadata": {"completion_delivery_status": "complete"},
        }
        assert _exact_completion_event_shape(completion_rows[0]) == expected_event

        resumed = db.get_messages_as_conversation(session_id)
        resumed_event = next(
            row
            for row in resumed
            if row.get("role") == "user" and row.get("content") == canonical
        )
        assert _exact_completion_event_shape(resumed_event) == expected_event
        assert _public_shape(resumed) == _public_shape(first_result["messages"])

        second = _make_agent(tmp_path, db, session_id)
        second_requests = _capture_client(
            second, [_mock_response(content="Acknowledged", finish_reason="stop")]
        )
        second.run_conversation(
            "review again", conversation_history=resumed, task_id="next-turn"
        )

        previous_wire_history = first_requests[-1]
        next_wire_history = second_requests[0]
        assert next_wire_history[: len(previous_wire_history)] == previous_wire_history
        assert next_wire_history[0] == {
            "role": "system",
            "content": "BYTE-STABLE SYSTEM",
        }
        assert all("display_kind" not in row for row in next_wire_history)
    finally:
        db.close()


def test_meaningful_completion_can_manual_compress_and_cold_resume(
    tmp_path, monkeypatch
):
    """A committed completion remains legal through manual in-place compression."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db_path = tmp_path / "state.db"
    session_id = "completion-manual-compress"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "working in background"},
        ],
    )
    canonical, wire = _completion_texts()

    try:
        agent = _make_agent(tmp_path, db, session_id)
        _stage_completion(agent, wire)
        _capture_client(
            agent, [_mock_response(content="Build passed", finish_reason="stop")]
        )
        result = agent.run_conversation(
            wire,
            conversation_history=db.get_messages_as_conversation(session_id),
            task_id="completion-before-manual-compress",
        )
        assert any(row.get("content") == canonical for row in result["messages"])

        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] completed build"},
            {"role": "assistant", "content": "Build passed"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        compressor._last_compression_made_progress = True
        compressor._last_summary_fallback_used = False
        agent.context_compressor = compressor
        agent.compression_in_place = True

        compacted, _ = agent._compress_context(
            result["messages"],
            "BYTE-STABLE SYSTEM",
            approx_tokens=100_000,
        )
        assert agent.session_id == session_id
        assert compressor._last_compress_aborted is False
        assert _public_shape(db.get_messages_as_conversation(session_id)) == (
            _public_shape(compacted)
        )
    finally:
        db.close()

    resumed_db = SessionDB(db_path=db_path)
    try:
        resumed = resumed_db.get_messages_as_conversation(session_id)
        _assert_provider_protocol_is_closed(resumed)
        assert [row["content"] for row in resumed] == [
            "[CONTEXT COMPACTION] completed build",
            "Build passed",
        ]

        cold = _make_agent(tmp_path, resumed_db, session_id)
        requests = _capture_client(
            cold, [_mock_response(content="Reviewed", finish_reason="stop")]
        )
        cold.run_conversation(
            "review again", conversation_history=resumed, task_id="cold-resume"
        )
        assert requests[0][1:3] == [
            {"role": "user", "content": "[CONTEXT COMPACTION] completed build"},
            {"role": "assistant", "content": "Build passed"},
        ]
    finally:
        resumed_db.close()


def test_empty_completion_is_one_call_silent_noop_and_leaves_db_unchanged(
    tmp_path, monkeypatch
):
    """The instructed literal-empty answer is a no-op, not a retry storm."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db_path = tmp_path / "state.db"
    session_id = "completion-noop"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "working in background"},
        ],
    )
    _canonical, wire = _completion_texts()

    try:
        before = db.get_messages_as_conversation(session_id)
        first = _make_agent(tmp_path, db, session_id)
        _stage_completion(first, wire)
        first_requests = _capture_client(
            first, [_mock_response(content="", finish_reason="stop")]
        )
        result = first.run_conversation(
            wire, conversation_history=before, task_id="completion-noop"
        )

        assert len(first_requests) == 1
        assert result["final_response"] == ""
        assert result["turn_exit_reason"] == "completion_delivery_noop"
        assert _public_shape(result["messages"]) == _public_shape(before)
        resumed = db.get_messages_as_conversation(session_id)
        assert _public_shape(resumed) == _public_shape(before)

        second = _make_agent(tmp_path, db, session_id)
        second_requests = _capture_client(
            second, [_mock_response(content="ok", finish_reason="stop")]
        )
        second.run_conversation("review again", conversation_history=resumed)

        stable_prefix = first_requests[0][:-1]
        assert second_requests[0][: len(stable_prefix)] == stable_prefix
        assert all(row.get("content") != wire for row in second_requests[0])
    finally:
        db.close()


def test_tool_then_literal_empty_commits_hidden_closed_suffix(tmp_path, monkeypatch):
    """A terminal empty answer is not a no-op after meaningful tool work."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "completion-tool-empty"
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "working in background"},
        ],
    )
    canonical, wire = _completion_texts()

    try:
        before = db.get_messages_as_conversation(session_id)
        first = _make_agent(tmp_path, db, session_id)
        _stage_completion(first, wire)
        first_requests = _capture_client(
            first,
            [
                _mock_response(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        _mock_tool_call(
                            name="web_search",
                            arguments="{}",
                            call_id="call-empty",
                        )
                    ],
                ),
                _mock_response(content="", finish_reason="stop"),
            ],
        )

        with patch("run_agent.handle_function_call", return_value="checked"):
            result = first.run_conversation(
                wire,
                conversation_history=before,
                task_id="completion-tool-empty",
            )

        assert len(first_requests) == 2
        assert result["completed"] is True
        assert result["turn_exit_reason"] == "completion_delivery_effect_complete"
        assert result["final_response"] == ""
        assert _public_shape([result["messages"][-1]]) == [{
            "role": "assistant",
            "content": "Operation completed.",
            "display_kind": "hidden",
        }]
        assert any(
            row.get("role") == "user"
            and row.get("content") == canonical
            and row.get("api_content") == wire
            for row in result["messages"]
        )

        resumed = db.get_messages_as_conversation(session_id)
        assert _public_shape(resumed) == _public_shape(result["messages"])
        second = _make_agent(tmp_path, db, session_id)
        second_requests = _capture_client(
            second, [_mock_response(content="ok", finish_reason="stop")]
        )
        second.run_conversation("review again", conversation_history=resumed)
        assert second_requests[0][: len(first_requests[-1])] == first_requests[-1]
    finally:
        db.close()


def test_completion_suffix_is_deferred_until_finalizer_commits_it(tmp_path):
    """A crash-time flush cannot persist assistant/tool rows without the event."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "completion-deferred"
    db.create_session(session_id, source="tui", model="test-model")
    canonical, wire = _completion_texts()
    agent = _make_agent(tmp_path, db, session_id)
    messages = [
        {
            "role": "user",
            "content": wire,
            "_completion_delivery_synthetic": True,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "web_search",
            "content": "checked",
        },
    ]
    try:
        agent._session_json_enabled = True
        agent._flush_messages_to_session_db(messages, conversation_history=[])
        agent._save_session_log(messages)

        assert db.get_messages_as_conversation(session_id) == []
        snapshot = agent.logs_dir / f"session_{session_id}.json"
        if snapshot.exists():
            snapshot_text = snapshot.read_text(encoding="utf-8")
            assert wire not in snapshot_text
            assert "checked" not in snapshot_text
        assert messages[0]["content"] == wire
        assert canonical not in [row.get("content") for row in messages]
    finally:
        db.close()


@pytest.mark.parametrize(
    ("tool_name", "expected_disposition"),
    [("read_file", None), ("write_file", "unknown")],
)
def test_crash_after_completion_tool_intent_recovers_before_next_user(
    tmp_path, monkeypatch, tool_name, expected_disposition
):
    """Cold resume treats the durable completion event and intent as one group."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db_path = tmp_path / "state.db"
    session_id = f"completion-intent-crash-{tool_name}"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "working in background"},
        ],
    )
    canonical, wire = _completion_texts()
    before = db.get_messages_as_conversation(session_id)
    crashing_agent = _make_agent(tmp_path, db, session_id)
    crashing_agent.valid_tool_names.add(tool_name)
    _stage_completion(crashing_agent, wire)
    _capture_client(
        crashing_agent,
        [
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        name=tool_name,
                        arguments="{}",
                        call_id="call-crash",
                    )
                ],
            )
        ],
    )

    class SimulatedProcessDeath(BaseException):
        pass

    with (
        patch.object(
            crashing_agent,
            "_execute_tool_calls",
            side_effect=SimulatedProcessDeath,
        ),
        pytest.raises(SimulatedProcessDeath),
    ):
        crashing_agent.run_conversation(wire, conversation_history=before)
    crashed_rows = db.get_messages_as_conversation(session_id)
    assert crashed_rows[-2]["content"] == canonical
    assert crashed_rows[-2]["api_content"] == wire
    assert crashed_rows[-2]["display_metadata"] == {
        "completion_delivery_status": "effect_started"
    }
    assert crashed_rows[-1]["tool_calls"][0]["function"]["name"] == tool_name
    db.close()

    resumed_db = SessionDB(db_path=db_path)
    try:
        recovery_status = resumed_db.recover_dangling_completion_tool_intent(
            session_id
        )
        assert recovery_status == (
            "read_only_archived"
            if expected_disposition is None
            else "side_effect_closed"
        )
        cold_rows = resumed_db.get_messages_as_conversation(session_id)
        recovered = sanitize_replay_history(cold_rows)

        if expected_disposition is None:
            assert _public_shape(recovered) == _public_shape(before)
            assert all(row.get("content") not in {canonical, wire} for row in recovered)
        else:
            event_row = recovered[-4]
            tool_result = recovered[-2]
            closure = recovered[-1]
            assert event_row["content"] == canonical
            assert event_row["api_content"] == wire
            assert event_row["display_metadata"] == {
                "completion_delivery_status": "interrupted"
            }
            assert tool_result["tool_call_id"] == "call-crash"
            assert tool_result["effect_disposition"] == expected_disposition
            assert "unknown" in tool_result["content"].lower()
            assert _public_shape([closure]) == [{
                "role": "assistant",
                "content": "Operation interrupted.",
                "display_kind": "hidden",
            }]
            _assert_provider_protocol_is_closed(recovered)

        next_agent = _make_agent(tmp_path, resumed_db, session_id)
        requests = _capture_client(
            next_agent, [_mock_response(content="ok", finish_reason="stop")]
        )
        with patch("run_agent.handle_function_call") as execute_tool:
            next_agent.run_conversation(
                "next real user", conversation_history=recovered
            )
        execute_tool.assert_not_called()

        request = requests[0]
        assert request[-1] == {"role": "user", "content": "next real user"}
        _assert_provider_protocol_is_closed(request)
        request_contents = [row.get("content") for row in request]
        if expected_disposition is None:
            assert canonical not in request_contents
            assert wire not in request_contents
            assert not any(row.get("tool_calls") for row in request)
        else:
            assert wire in request_contents
            assert any(
                row.get("role") == "tool"
                and row.get("tool_call_id") == "call-crash"
                and "unknown" in row.get("content", "").lower()
                for row in request
            )
            assert {
                "role": "assistant",
                "content": "Operation interrupted.",
            } in request
    finally:
        resumed_db.close()

    second_cold_db = SessionDB(db_path=db_path)
    try:
        assert second_cold_db.recover_dangling_completion_tool_intent(
            session_id
        ) == "none"
        second_cold_rows = second_cold_db.get_messages_as_conversation(session_id)
        _assert_provider_protocol_is_closed(second_cold_rows)
        assert any(
            row.get("role") == "user" and row.get("content") == "next real user"
            for row in second_cold_rows
        )
        all_rows = second_cold_db.get_messages(
            session_id, include_inactive=True
        )
        event_rows = [
            row for row in all_rows
            if row.get("role") == "user" and row.get("content") == canonical
        ]
        assert len(event_rows) == 1
        assert event_rows[0]["display_metadata"] == {
            "completion_delivery_status": "interrupted"
        }
        assert all(
            (row.get("display_metadata") or {}).get(
                "completion_delivery_status"
            ) != "effect_started"
            for row in all_rows
        )
        if expected_disposition is None:
            archived_intents = [
                row for row in all_rows
                if row.get("tool_calls")
                and row["tool_calls"][0]["function"]["name"] == tool_name
            ]
            assert event_rows[0]["active"] == 0
            assert len(archived_intents) == 1
            assert archived_intents[0]["active"] == 0
            assert all(row.get("content") != canonical for row in second_cold_rows)
        else:
            assert event_rows[0]["active"] == 1
            assert any(
                row.get("role") == "tool"
                and row.get("tool_call_id") == "call-crash"
                and row.get("effect_disposition") == expected_disposition
                for row in all_rows
            )
            assert any(
                row.get("role") == "assistant"
                and row.get("content") == "Operation interrupted."
                and row.get("display_kind") == "hidden"
                for row in all_rows
            )
    finally:
        second_cold_db.close()


def test_provider_failure_leaves_no_orphan_completion_suffix(tmp_path, monkeypatch):
    """A failed completion request cannot persist its internal user row."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "completion-provider-failure"
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "working in background"},
        ],
    )
    before = db.get_messages_as_conversation(session_id)
    _canonical, wire = _completion_texts()
    agent = _make_agent(tmp_path, db, session_id)
    agent._api_max_retries = 1
    _stage_completion(agent, wire)
    failed_requests: list[list[dict]] = []

    def fail_request(**kwargs):
        failed_requests.append(copy.deepcopy(kwargs["messages"]))
        raise RuntimeError("provider failed")

    agent.client.chat.completions.create.side_effect = fail_request

    try:
        result = agent.run_conversation(
            wire, conversation_history=before, task_id="completion-failure"
        )

        assert result["completed"] is False
        assert "provider failed" in str(result.get("error", ""))
        assert len(failed_requests) == 1
        assert _public_shape(result["messages"]) == _public_shape(before)
        assert _public_shape(agent._session_messages) == _public_shape(before)
        resumed = db.get_messages_as_conversation(session_id)
        assert _public_shape(resumed) == _public_shape(before)
        assert all(row.get("content") != wire for row in resumed)

        next_agent = _make_agent(tmp_path, db, session_id)
        next_requests = _capture_client(
            next_agent, [_mock_response(content="ok", finish_reason="stop")]
        )
        next_agent.run_conversation("review again", conversation_history=resumed)
        settled_wire_prefix = failed_requests[0][:-1]
        assert next_requests[0][: len(settled_wire_prefix)] == settled_wire_prefix
    finally:
        db.close()


def test_durable_user_tail_gets_append_only_boundary_before_completion(tmp_path, monkeypatch):
    """Alternation repair must never merge the event into an older user row."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "completion-after-user-tail"
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "background task started"},
            {"role": "user", "content": "unanswered durable follow-up"},
        ],
    )
    canonical, wire = _completion_texts()

    try:
        before = db.get_messages_as_conversation(session_id)
        agent = _make_agent(tmp_path, db, session_id)
        _stage_completion(agent, wire)
        requests = _capture_client(
            agent, [_mock_response(content="Build passed", finish_reason="stop")]
        )
        result = agent.run_conversation(
            wire, conversation_history=before, task_id="completion-user-tail"
        )

        first_wire = requests[0]
        assert [row["role"] for row in first_wire] == [
            "system", "user", "assistant", "user", "assistant", "user"
        ]
        assert first_wire[-3]["content"] == "unanswered durable follow-up"
        assert first_wire[-2]["content"] == "Operation interrupted."
        assert first_wire[-1]["content"] == wire
        assert before[-1]["content"] == "unanswered durable follow-up"

        resumed = db.get_messages_as_conversation(session_id)
        assert [row.get("content") for row in resumed[-4:]] == [
            "unanswered durable follow-up",
            "Operation interrupted.",
            canonical,
            "Build passed",
        ]
        _assert_provider_protocol_is_closed(resumed)

        next_agent = _make_agent(tmp_path, db, session_id)
        next_requests = _capture_client(
            next_agent, [_mock_response(content="ok", finish_reason="stop")]
        )
        next_agent.run_conversation("review again", conversation_history=resumed)
        assert next_requests[0][: len(first_wire)] == first_wire
        assert result["completed"] is True
    finally:
        db.close()


def test_system_prompt_build_failure_clears_staged_completion(tmp_path, monkeypatch):
    """A prologue exception cannot leak a staged marker into the next turn."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "completion-prologue-failure"
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "background task started"},
        ],
    )
    _canonical, wire = _completion_texts()

    try:
        before = db.get_messages_as_conversation(session_id)
        agent = _make_agent(tmp_path, db, session_id)
        agent._cached_system_prompt = None
        _stage_completion(agent, wire)
        with patch.object(
            agent, "_build_system_prompt", side_effect=RuntimeError("prompt build failed")
        ):
            with pytest.raises(RuntimeError, match="prompt build failed"):
                agent.run_conversation(wire, conversation_history=before)

        assert agent._pending_cli_user_message is None
        assert all(
            not row.get("_completion_delivery_synthetic") for row in before
        )
        assert _public_shape(db.get_messages_as_conversation(session_id)) == _public_shape(before)

        caller_owned_marker = {
            "role": "user",
            "content": wire,
            "_completion_delivery_synthetic": True,
        }
        caller_history = [*before, caller_owned_marker]
        with patch.object(
            agent, "_build_system_prompt", side_effect=RuntimeError("prompt build failed")
        ):
            with pytest.raises(RuntimeError, match="prompt build failed"):
                agent.run_conversation("ordinary turn", conversation_history=caller_history)
        assert caller_history == before
    finally:
        db.close()


@pytest.mark.parametrize("interrupt_after_tool", [False, True])
def test_tool_side_effect_then_failure_or_interrupt_is_durable(
    tmp_path, monkeypatch, interrupt_after_tool
):
    """Once a tool ran, restart must audit it and must not execute it again."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = f"completion-side-effect-{interrupt_after_tool}"
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "background task started"},
        ],
    )
    canonical, wire = _completion_texts()
    agent = _make_agent(tmp_path, db, session_id)
    agent._api_max_retries = 1
    _stage_completion(agent, wire)
    requests: list[list[dict]] = []
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    name="web_search", arguments="{}", call_id="call-effect"
                )
            ],
        )
    ]
    if not interrupt_after_tool:
        responses.append(RuntimeError("provider failed after side effect"))

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    agent.client.chat.completions.create.side_effect = create
    tool_calls = 0

    def execute_tool(*_args, **_kwargs):
        nonlocal tool_calls
        tool_calls += 1
        if interrupt_after_tool:
            agent.interrupt()
        return "SIDE EFFECT COMPLETE"

    try:
        before = db.get_messages_as_conversation(session_id)
        with patch("run_agent.handle_function_call", side_effect=execute_tool):
            result = agent.run_conversation(
                wire, conversation_history=before, task_id="completion-side-effect"
            )

        assert tool_calls == 1
        assert result["completed"] is False
        assert result.get("interrupted", False) is interrupt_after_tool
        if not interrupt_after_tool:
            assert "provider failed after side effect" in str(result.get("error", ""))

        resumed = db.get_messages_as_conversation(session_id)
        resumed_event = next(
            row
            for row in resumed
            if row.get("role") == "user" and row.get("content") == canonical
        )
        expected_event = {
            "role": "user",
            "content": canonical,
            "api_content": wire,
            "display_kind": "hidden",
            "display_metadata": {
                "completion_delivery_status": (
                    "interrupted" if interrupt_after_tool else "failed"
                )
            },
        }
        assert _exact_completion_event_shape(resumed_event) == expected_event
        live_event = next(
            row
            for row in result["messages"]
            if row.get("role") == "user" and row.get("content") == canonical
        )
        assert _exact_completion_event_shape(live_event) == expected_event
        assert any(row.get("content") == "SIDE EFFECT COMPLETE" for row in resumed)
        assert resumed[-1].get("role") == "assistant"
        assert resumed[-1].get("content") == "Operation interrupted."
        assert resumed[-1].get("display_kind") == "hidden"
        _assert_provider_protocol_is_closed(resumed)

        next_agent = _make_agent(tmp_path, db, session_id)
        next_requests = _capture_client(
            next_agent, [_mock_response(content="ok", finish_reason="stop")]
        )
        with patch("run_agent.handle_function_call", side_effect=AssertionError("replayed tool")):
            next_agent.run_conversation("review again", conversation_history=resumed)
        assert next_requests[0][: len(requests[-1])] == requests[-1]
        assert any(
            row.get("role") == "assistant"
            and row.get("content") == "Operation interrupted."
            for row in next_requests[0][len(requests[-1]) :]
        )
    finally:
        db.close()


def test_completion_commit_retries_before_publishing_live_state(tmp_path, monkeypatch):
    """A transient DB append failure cannot split DB/live/JSON authority."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "completion-transient-commit"
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "background task started"},
            {"role": "user", "content": "unanswered durable follow-up"},
        ],
    )
    canonical, wire = _completion_texts()
    before = db.get_messages_as_conversation(session_id)
    agent = _make_agent(tmp_path, db, session_id)
    agent._session_json_enabled = True
    _stage_completion(agent, wire)
    _capture_client(agent, [_mock_response(content="Build passed", finish_reason="stop")])
    original_append = db.append_messages_batch
    completion_attempts = 0

    def flaky_append(*args, **kwargs):
        nonlocal completion_attempts
        rows = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
        if any(row.get("content") == canonical for row in rows):
            completion_attempts += 1
            if completion_attempts == 1:
                raise RuntimeError("transient sqlite failure")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(db, "append_messages_batch", flaky_append)

    try:
        result = agent.run_conversation(wire, conversation_history=before)
        resumed = db.get_messages_as_conversation(session_id)

        assert completion_attempts == 2
        assert result["completed"] is True
        assert not result.get("cleanup_errors")
        assert _public_shape(result["messages"]) == _public_shape(resumed)
        assert sum(row.get("content") == canonical for row in resumed) == 1
        assert sum(row.get("content") == "Build passed" for row in resumed) == 1
        assert agent._completion_delivery_commit_failed is False

        snapshot = agent.logs_dir / f"session_{session_id}.json"
        assert snapshot.exists()
        snapshot_rows = json.loads(snapshot.read_text(encoding="utf-8"))["messages"]
        snapshot_event = next(
            row for row in snapshot_rows if row.get("content") == canonical
        )
        assert snapshot_event["api_content"] == wire
    finally:
        db.close()


def test_failed_tool_intent_commit_never_executes_or_replays_tool(tmp_path, monkeypatch):
    """A completion tool call must be canonical before its side effect runs."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "completion-tool-intent-failure"
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(session_id, [
        {"role": "user", "content": "original request"},
        {"role": "assistant", "content": "background task started"},
    ])
    before = db.get_messages_as_conversation(session_id)
    _canonical, wire = _completion_texts()
    agent = _make_agent(tmp_path, db, session_id)
    _stage_completion(agent, wire)
    _capture_client(agent, [_mock_response(
        content="I will check",
        finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(
            name="web_search", arguments="{}", call_id="call-uncommitted"
        )],
    )])
    original_append = db.append_messages_batch
    monkeypatch.setattr(
        db, "append_messages_batch", lambda *_a, **_kw: (_ for _ in ()).throw(
            RuntimeError("sqlite unavailable")
        )
    )

    try:
        with patch("run_agent.handle_function_call") as execute:
            result = agent.run_conversation(wire, conversation_history=before)
        execute.assert_not_called()
        assert result["failed"] is True
        assert result["completion_delivery_status"] == "none"
        assert agent._completion_delivery_commit_failed is False
        assert agent._pending_completion_delivery_suffix is None
        assert _public_shape(result["messages"]) == _public_shape(before)

        monkeypatch.setattr(db, "append_messages_batch", original_append)
        requests = _capture_client(
            agent, [_mock_response(content="ok", finish_reason="stop")]
        )
        agent.run_conversation("review again", conversation_history=before)
        assert all(row.get("content") != wire for row in requests[0])
        assert all(not row.get("tool_calls") for row in requests[0])
    finally:
        db.close()


def test_persistent_completion_commit_failure_keeps_restart_on_old_prefix(
    tmp_path, monkeypatch
):
    """A loud pending result is safer than publishing DB-divergent history."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "completion-persistent-commit-failure"
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "background task started"},
        ],
    )
    _canonical, wire = _completion_texts()
    before = db.get_messages_as_conversation(session_id)
    agent = _make_agent(tmp_path, db, session_id)
    agent._session_json_enabled = True
    _stage_completion(agent, wire)
    _capture_client(agent, [_mock_response(content="Build passed", finish_reason="stop")])
    original_append = db.append_messages_batch
    attempts = 0

    def reject_append(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("persistent sqlite failure")

    monkeypatch.setattr(db, "append_messages_batch", reject_append)

    try:
        result = agent.run_conversation(wire, conversation_history=before)

        assert attempts == 2
        assert result["completed"] is False
        assert result["failed"] is True
        assert "completion_delivery_commit" in " ".join(result["cleanup_errors"])
        assert agent._completion_delivery_commit_failed is True
        assert agent._pending_completion_delivery_suffix
        assert _public_shape(result["messages"]) == _public_shape(before)
        assert any(
            row.get("_completion_delivery_synthetic")
            for row in agent._session_messages
        )

        resumed = db.get_messages_as_conversation(session_id)
        assert _public_shape(resumed) == _public_shape(before)
        snapshot = agent.logs_dir / f"session_{session_id}.json"
        if snapshot.exists():
            snapshot_rows = json.loads(snapshot.read_text(encoding="utf-8"))["messages"]
            assert _public_shape(snapshot_rows) == _public_shape(before)

        # Once storage recovers, the cached agent retries the exact retained
        # suffix before accepting the next user turn.
        monkeypatch.setattr(db, "append_messages_batch", original_append)
        same_agent_requests = _capture_client(
            agent, [_mock_response(content="ok", finish_reason="stop")]
        )
        agent.run_conversation("review again", conversation_history=result["messages"])
        assert any(row.get("content") == wire for row in same_agent_requests[0])
        recovered = db.get_messages_as_conversation(session_id)
        assert any(row.get("content") == "Build passed" for row in recovered)
        assert agent._completion_delivery_commit_failed is False

        # A fresh process now sees the same recovered authoritative prefix.
        next_agent = _make_agent(tmp_path, db, session_id)
        requests = _capture_client(
            next_agent, [_mock_response(content="ok", finish_reason="stop")]
        )
        next_agent.run_conversation("review once more", conversation_history=recovered)
        assert any(row.get("content") == wire for row in requests[0])
        assert any(row.get("content") == "Build passed" for row in requests[0])
    finally:
        db.close()


def test_micro_compaction_durable_rewrite_stops_before_pending_suffix(tmp_path):
    """Micro-compaction rewrites only the settled prefix, not live tool work."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    session_id = "completion-micro"
    db.create_session(session_id, source="tui", model="test-model")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old response"},
        ],
    )
    before = db.get_messages_as_conversation(session_id)
    _canonical, wire = _completion_texts()
    pending = [
        *before,
        {
            "role": "user",
            "content": wire,
            "_completion_delivery_synthetic": True,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-micro",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-micro",
            "name": "web_search",
            "content": "pending tool result",
        },
    ]
    agent = _make_agent(tmp_path, db, session_id)
    try:
        committed = agent.context_compressor._sync_micro_compact_to_db(
            pending,
            expected_active_fingerprint=db.get_compaction_fingerprint(session_id),
        )

        assert committed is True
        assert pending[-1]["content"] == "pending tool result"
        assert not pending[-1].get("_db_persisted")
    finally:
        db.close()

    resumed_db = SessionDB(db_path=db_path)
    try:
        assert _public_shape(
            resumed_db.get_messages_as_conversation(session_id)
        ) == _public_shape(before)
    finally:
        resumed_db.close()
