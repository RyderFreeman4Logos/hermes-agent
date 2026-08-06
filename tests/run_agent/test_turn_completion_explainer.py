"""Tests for the end-of-turn completion explainer (#34452).

When a turn ends abnormally after tools (empty content after retries, a
partial/truncated stream, exhausted retries, or an iteration/budget limit)
the user should get a single user-visible explanation of why the reply
stopped instead of a blank or fragmentary response box.  Normal short
replies (e.g. ``Done.``) must stay quiet.

These tests exercise:
  1. ``_format_turn_completion_explanation`` — the pure reason→message map.
  2. ``_turn_completion_explainer_enabled`` — the env/config seam.
  3. An end-to-end ``run_conversation`` turn that exhausts empty-response
     retries and verifies the explanation reaches ``final_response``.

All assertions work under the mocked OpenAI SDK used elsewhere in this
suite (we patch ``run_agent.OpenAI`` and drive ``agent.client``), so they
pass identically in CI and locally.
"""

import os
import queue
import threading
import uuid
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_engine import ContextEngine
from agent.copilot_acp_client import CopilotACPClient
from run_agent import AIAgent


# --------------------------------------------------------------------------
# Fixtures (mirrors tests/run_agent/test_tool_call_guardrail_runtime.py)
# --------------------------------------------------------------------------
def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _mock_stream_chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(
        role="assistant" if content else None,
        content=content,
        tool_calls=None,
        reasoning_content=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    # No fallback chain so empty responses exhaust deterministically.
    agent._fallback_chain = []
    return agent


@pytest.fixture
def heartbeat_event(monkeypatch):
    event = {
        "type": "heartbeat",
        "target_id": "proc-heartbeat",
        "target_ids": ["proc-heartbeat"],
        "generations": [7],
        "generation": 7,
        "session_key": "owner-session",
        "provider": "openrouter",
        "cache_context": "test-cache-context",
        "status": "ALIVE",
        "evidence": "output grew",
    }
    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        lambda candidate, agent=None, **_kwargs: candidate is event,
    )
    return event


# --------------------------------------------------------------------------
# 1. Pure formatter
# --------------------------------------------------------------------------
def test_explanation_quiet_for_normal_text_response():
    """A healthy text_response exit must NOT produce any explanation."""
    out = AIAgent._format_turn_completion_explanation(
        "text_response(finish_reason=stop)"
    )
    assert out == ""


def test_explanation_quiet_for_empty_reason():
    assert AIAgent._format_turn_completion_explanation("") == ""
    assert AIAgent._format_turn_completion_explanation("unknown") == ""
    # guardrail_halt surfaces its own message; explainer stays out of the way.
    assert AIAgent._format_turn_completion_explanation("guardrail_halt") == ""






def test_explanation_for_max_iterations_reached_prefix_match():
    """``max_iterations_reached(...)`` carries a parenthetical suffix."""
    out = AIAgent._format_turn_completion_explanation(
        "max_iterations_reached(10/10)"
    )
    assert "iteration" in out.lower()






# --------------------------------------------------------------------------
# 1b. Cause-aware session-persistence wording
# --------------------------------------------------------------------------
def test_explanation_persistence_locked_cause_says_busy_not_disk():
    """Write-lock contention must NOT be misdiagnosed as a disk problem."""
    out = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "locked"
    )
    lower = out.lower()
    assert "busy" in lower
    assert "saved" in lower
    assert "send it again" in lower
    assert "disk" not in lower
    assert "permission" not in lower


def test_explanation_persistence_disk_cause_keeps_disk_wording():
    out = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "disk"
    )
    lower = out.lower()
    assert "disk" in lower
    assert "free some space" in lower or "disk space" in lower


def test_explanation_persistence_unknown_cause_is_neutral():
    """None/'unknown' cause must not claim disk-full — point at diagnostics."""
    for cause in (None, "unknown"):
        out = AIAgent._format_turn_completion_explanation(
            "session_persistence_failed", cause
        )
        lower = out.lower()
        assert out.strip() != ""
        assert "disk space" not in lower
        assert "full disk" not in lower
        assert "hermes doctor" in lower
        assert "again" in lower


def test_explanation_persistence_one_arg_backward_compat():
    """Existing one-arg callers must keep working (optional second param)."""
    out = AIAgent._format_turn_completion_explanation("session_persistence_failed")
    assert out.strip() != ""
    assert "session storage" in out.lower()


def test_explanation_cause_ignored_for_other_reasons():
    """The cause parameter must not perturb non-persistence reasons."""
    assert (
        AIAgent._format_turn_completion_explanation(
            "text_response(finish_reason=stop)", "locked"
        )
        == ""
    )
    out = AIAgent._format_turn_completion_explanation(
        "max_iterations_reached(10/10)", "locked"
    )
    assert "iteration" in out.lower()


# --------------------------------------------------------------------------
# 1c. classify_persistence_error — the pure cause classifier
# --------------------------------------------------------------------------
def test_classify_persistence_error_categories():
    import sqlite3

    from hermes_state import classify_persistence_error

    assert classify_persistence_error(
        sqlite3.OperationalError("database is locked")
    ) == "locked"
    assert classify_persistence_error("SQLITE_BUSY: busy") == "locked"
    assert classify_persistence_error(
        sqlite3.OperationalError("database or disk is full")
    ) == "disk"
    assert classify_persistence_error("attempt to write a readonly database") == "disk"
    assert classify_persistence_error("read-only file system") == "disk"
    assert classify_persistence_error("no space left on device") == "disk"
    assert classify_persistence_error("disk I/O error") == "disk"
    assert classify_persistence_error("something else entirely") == "unknown"
    assert classify_persistence_error(None) == "unknown"
    assert classify_persistence_error("") == "unknown"


def test_classify_persistence_error_reuses_disk_full_markers():
    """The disk bucket delegates to hermes_state.is_disk_full_error, so
    every marker that helper recognizes (ENOSPC, 'not enough space', ...)
    must classify as 'disk' — the two classifiers can never drift apart."""
    import errno

    from hermes_state import classify_persistence_error

    assert classify_persistence_error("ENOSPC writing state.db") == "disk"
    assert classify_persistence_error(
        "There is not enough space on the disk"
    ) == "disk"
    assert classify_persistence_error(
        OSError(errno.ENOSPC, "No space left on device")
    ) == "disk"


def test_classify_persistence_error_compression_busy_is_locked():
    """A live compression lease refusing the write is contention, not
    storage damage — but its message contains neither 'locked' nor 'busy',
    so it must classify by exception type (and by phrase for RPC-wrapped
    strings). This is the exact failure mode of issue #81227."""
    from hermes_state import (
        CompressionSessionBusyError,
        SessionCompressionInProgressError,
    )
    from hermes_state import classify_persistence_error

    assert classify_persistence_error(
        SessionCompressionInProgressError(
            "Session 'abc' is being compressed by another writer"
        )
    ) == "locked"
    assert classify_persistence_error(
        CompressionSessionBusyError("Compression lease lost before publication: abc")
    ) == "locked"
    # RPC-wrapped string forms (exception type lost in transit).
    assert classify_persistence_error(
        "Session 'abc' is being compressed by another writer"
    ) == "locked"
    assert classify_persistence_error(
        "Compression lease lost before publication: abc"
    ) == "locked"


def test_persistence_error_causes_tuple_matches_classifier():
    """PERSISTENCE_ERROR_CAUSES must cover every value the classifier can
    return (consumers like cron suppression iterate it)."""
    from hermes_state import PERSISTENCE_ERROR_CAUSES, classify_persistence_error

    probes = (
        "database is locked",
        "database or disk is full",
        "something else entirely",
        None,
    )
    for probe in probes:
        assert classify_persistence_error(probe) in PERSISTENCE_ERROR_CAUSES


# --------------------------------------------------------------------------
# 2. Enable/disable seam
# --------------------------------------------------------------------------
def test_explainer_enabled_by_default():
    agent = _make_agent()
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HERMES_TURN_COMPLETION_EXPLAINER", None)
        with patch("hermes_cli.config.load_config", return_value={}):
            assert agent._turn_completion_explainer_enabled() is True


def test_explainer_disabled_via_env():
    agent = _make_agent()
    with patch.dict(
        os.environ, {"HERMES_TURN_COMPLETION_EXPLAINER": "0"}, clear=False
    ):
        assert agent._turn_completion_explainer_enabled() is False




# --------------------------------------------------------------------------
# 3. End-to-end: empty-response exhaustion surfaces the explanation
# --------------------------------------------------------------------------
def test_run_conversation_empty_exhausted_surfaces_explanation():
    """Four empty responses in a row should exhaust retries and the final
    response should be the actionable explanation, not a bare '(empty)'."""
    agent = _make_agent(max_iterations=10)
    # 4 empty responses: retries 1..3 then the terminal on the 4th.
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop") for _ in range(8)
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do something")

    assert result["turn_exit_reason"] == "empty_response_exhausted"
    # The user must NOT be left with a bare sentinel; the explanation wins.
    assert result["final_response"] != "(empty)"
    assert result["final_response"].strip() != ""
    assert "No reply:" in result["final_response"]


def test_heartbeat_silent_noop_leaves_no_durable_or_live_history(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop")
    ]
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]
    agent._session_messages = history

    with (
        patch.object(agent, "_persist_session") as persist,
        patch.object(agent, "_save_trajectory") as trajectory,
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_sync_external_memory_for_turn") as external_memory,
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] target remains ALIVE",
            conversation_history=history,
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    assert result["final_response"] == ""
    assert result["messages"] == history
    assert agent._session_messages == history
    persist.assert_not_called()
    trajectory.assert_not_called()
    external_memory.assert_not_called()


@pytest.mark.parametrize("status", ["ALIVE", "STUCK"])
def test_live_heartbeat_uses_one_provider_response_with_tools_disabled_on_wire(
    heartbeat_event, status,
):
    heartbeat_event["status"] = status
    agent = _make_agent(max_iterations=10)
    agent.tools = [
        {
            "type": "function",
            "function": {"name": "web_search", "parameters": {}},
        }
    ]
    tool_call = SimpleNamespace(
        id="heartbeat-tool",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{}"),
    )
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response(content="Target is still working.", finish_reason="stop"),
    ]
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]
    agent._session_messages = history
    persisted = []

    with (
        patch.object(agent, "_execute_tool_calls") as execute,
        patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=lambda messages, _history=None: persisted.append(list(messages)),
        ),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            conversation_history=history,
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    assert result["final_response"] == ""
    assert result["messages"] == history
    assert agent._session_messages == history
    assert agent.client.chat.completions.create.call_count == 1
    execute.assert_not_called()
    assert persisted == []
    request = agent.client.chat.completions.create.call_args.kwargs
    assert request["messages"][0]["role"] == "system"
    assert "You are helpful." in request["messages"][0]["content"]
    assert request["tools"] == agent.tools
    assert request["tool_choice"] == "none"
    assert request["stream"] is False


def test_heartbeat_matches_ordinary_effective_cache_prefix(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.ephemeral_system_prompt = "EPHEMERAL-SYSTEM"
    agent.tools = [
        {
            "type": "function",
            "function": {"name": "web_search", "parameters": {}},
        }
    ]
    agent.client.chat.completions.create.return_value = _mock_response("ok")

    heartbeat = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )
    heartbeat_request = agent.client.chat.completions.create.call_args.kwargs
    agent.client.chat.completions.create.reset_mock()

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("ordinary request")
    ordinary_request = agent.client.chat.completions.create.call_args.kwargs

    assert heartbeat["silent_noop"] is True
    assert heartbeat_request["messages"][0] == ordinary_request["messages"][0]
    assert heartbeat_request["tools"] == ordinary_request["tools"]


@pytest.mark.parametrize(
    ("threshold", "request_tokens"),
    [(100, 101), (400_000, 272_000)],
)
def test_heartbeat_skips_provider_at_compression_or_hard_limit(
    threshold, request_tokens, heartbeat_event
):
    agent = _make_agent(max_iterations=10)
    agent.context_compressor.threshold_tokens = threshold
    agent.client.chat.completions.create.return_value = _mock_response("unexpected")

    with (
        patch(
            "agent.conversation_loop.estimate_request_context_tokens",
            return_value=request_tokens,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    assert result["messages"] == []
    agent.client.chat.completions.create.assert_not_called()


def test_successful_provider_dispatches_reset_exact_heartbeat_group(heartbeat_event):
    from tools.approval import reset_current_session_key, set_current_session_key
    from tools.runtime_heartbeat import (
        canonical_runtime_cache_context_identity,
        canonical_runtime_provider_identity,
    )

    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop"),
        _mock_response(content="real reply", finish_reason="stop"),
    ]
    token = set_current_session_key("owner-session")
    try:
        with (
            patch(
                "tools.runtime_heartbeat.runtime_heartbeat.reset_for_caller"
            ) as reset_deadline,
            patch("agent.conversation_loop.time.monotonic", return_value=123.0),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation(
                "[HEARTBEAT] inspect target",
                turn_origin="heartbeat_warm",
                heartbeat_event=heartbeat_event,
            )
            reset_deadline.assert_called_once_with(
                "owner-session",
                provider="openrouter",
                cache_context="test-cache-context",
                activity_at=123.0,
            )

            reset_deadline.reset_mock()
            agent.run_conversation("real user turn")
            reset_deadline.assert_called_once_with(
                "owner-session",
                provider=canonical_runtime_provider_identity(agent),
                cache_context=canonical_runtime_cache_context_identity(agent),
                activity_at=123.0,
            )
    finally:
        reset_current_session_key(token)


def test_failed_provider_calls_do_not_extend_heartbeat_lease(heartbeat_event):
    from tools.approval import reset_current_session_key, set_current_session_key

    agent = _make_agent(max_iterations=10)
    agent._api_max_retries = 1
    agent.client.chat.completions.create.side_effect = [
        RuntimeError("warm failed before response"),
        RuntimeError("ordinary failed before response"),
    ]
    token = set_current_session_key("owner-session")
    try:
        with (
            patch(
                "tools.runtime_heartbeat.runtime_heartbeat.reset_for_caller"
            ) as reset_deadline,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation(
                "[HEARTBEAT] inspect target",
                turn_origin="heartbeat_warm",
                heartbeat_event=heartbeat_event,
            )
            agent.run_conversation("real user turn")
    finally:
        reset_current_session_key(token)

    reset_deadline.assert_not_called()


@pytest.mark.parametrize(
    ("streaming", "short_circuit"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=[
        "nonstream-physical",
        "nonstream-short-circuit",
        "stream-physical",
        "stream-short-circuit",
    ],
)
def test_provider_lease_requires_physical_relay_dispatch(
    tmp_path, monkeypatch, streaming, short_circuit
):
    relay = pytest.importorskip("nemo_relay")
    from agent import relay_llm, relay_runtime
    from tools.approval import reset_current_session_key, set_current_session_key

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    agent = _make_agent(max_iterations=10)
    host = relay_runtime.get_runtime()
    assert host is not None
    host.retain_managed_execution("test.provider_lease")
    physical_call = MagicMock()
    intercept_name = f"provider-lease-{'stream' if streaming else 'execute'}"
    wire_client = None

    if streaming:
        chunks = [
            _mock_stream_chunk(content="relay reply"),
            _mock_stream_chunk(finish_reason="stop"),
        ]
        physical_call.return_value = iter(chunks)
        wire_client = MagicMock()
        wire_client.chat.completions.create = physical_call
        agent.client = SimpleNamespace()

        def stream_intercept(request, next_call):
            async def generate():
                if short_circuit:
                    for chunk in chunks:
                        yield relay_llm._jsonable(chunk)
                    return
                upstream = await next_call(request)
                async for chunk in upstream:
                    yield chunk

            return generate()

        relay.intercepts.register_llm_stream_execution(
            intercept_name, 1, stream_intercept
        )
    else:
        assert agent.client is not None
        physical_call = agent.client.chat.completions.create
        physical_call.return_value = _mock_response(content="physical reply")

        def execute_intercept(_name, request, next_call):
            if short_circuit:
                return relay_llm._jsonable(_mock_response(content="relay reply"))
            return next_call(request)

        relay.intercepts.register_llm_execution(intercept_name, 1, execute_intercept)

    token = set_current_session_key("owner-session")
    try:
        with (
            patch(
                "tools.runtime_heartbeat.runtime_heartbeat.reset_for_caller"
            ) as reset_deadline,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(
                agent, "_create_request_openai_client", return_value=wire_client
            )
            if streaming
            else nullcontext(),
            patch.object(agent, "_close_request_openai_client")
            if streaming
            else nullcontext(),
        ):
            result = agent.run_conversation(
                "real user turn",
                stream_callback=(lambda _delta: None) if streaming else None,
            )
        assert result["completed"] is True
        assert physical_call.call_count == (0 if short_circuit else 1)
        assert reset_deadline.call_count == (0 if short_circuit else 1)
    finally:
        reset_current_session_key(token)
        if streaming:
            relay.intercepts.deregister_llm_stream_execution(intercept_name)
        else:
            relay.intercepts.deregister_llm_execution(intercept_name)
        host.release_managed_execution("test.provider_lease")
        relay_runtime._reset_for_tests()


def test_heartbeat_early_error_leaves_no_unmatched_synthetic_user_row(
    heartbeat_event,
):
    agent = _make_agent(max_iterations=10)
    agent.provider = "nous"
    history = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]
    agent._session_messages = history

    with (
        patch("agent.nous_rate_guard.nous_rate_limit_remaining", return_value=60),
        patch("agent.nous_rate_guard.format_remaining", return_value="1m"),
        patch.object(agent, "_try_activate_fallback", return_value=True) as fallback,
        patch.object(agent, "_save_session_log") as save_log,
        patch.object(agent, "_flush_messages_to_session_db") as flush,
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            conversation_history=history,
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["messages"] == history
    assert agent._session_messages == history
    assert agent.client.chat.completions.create.call_count == 1
    fallback.assert_not_called()
    save_log.assert_not_called()
    flush.assert_not_called()


def test_heartbeat_bypasses_ordinary_lifecycle_hooks(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.return_value = _mock_response("still alive")

    with (
        patch.dict(os.environ, {"HERMES_DUMP_REQUESTS": "1"}),
        patch("hermes_cli.lifecycle.invoke_hook") as invoke_hook,
        patch.object(agent, "_dump_api_request_debug") as dump_request,
        patch.object(agent, "_persist_session") as persist,
        patch.object(agent, "_save_trajectory") as trajectory,
        patch.object(agent, "_sync_external_memory_for_turn") as external_memory,
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    invoke_hook.assert_not_called()
    dump_request.assert_not_called()
    persist.assert_not_called()
    trajectory.assert_not_called()
    external_memory.assert_not_called()


def test_heartbeat_does_not_consume_user_maintenance_triggers(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent._user_turn_count = 9
    agent._turns_since_memory = 4
    agent._memory_nudge_interval = 5
    agent._memory_store = MagicMock()
    agent._iters_since_skill = 5
    agent._skill_nudge_interval = 5
    agent.valid_tool_names = {"memory", "skill_manage"}
    agent._memory_manager = MagicMock()
    agent._memory_manager.prefetch_all.return_value = ""
    agent._pending_steer = "queued user steer"
    agent._pending_redirect = "queued user redirect"
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop"),
        _mock_response(content="real reply", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_spawn_background_review") as review,
    ):
        agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )
        assert agent._user_turn_count == 9
        assert agent._turns_since_memory == 4
        assert agent._iters_since_skill == 5
        assert agent._pending_steer == "queued user steer"
        assert agent._pending_redirect == "queued user redirect"
        agent._memory_manager.on_turn_start.assert_not_called()
        agent._memory_manager.prefetch_all.assert_not_called()
        review.assert_not_called()
        cached_system_prompt = agent._cached_system_prompt

        agent._pending_steer = None
        agent._pending_redirect = None
        result = agent.run_conversation("real user turn")

    assert result["final_response"] == "real reply"
    assert agent._user_turn_count == 10
    assert agent._turns_since_memory == 0
    assert agent._iters_since_skill == 0
    assert agent._cached_system_prompt == cached_system_prompt
    agent._memory_manager.on_turn_start.assert_called_once()
    agent._memory_manager.prefetch_all.assert_called_once()
    review.assert_called_once()


@pytest.mark.parametrize(
    ("cached_tokens", "expected_pct", "should_be_red"),
    [(1_880, 94, True), (1_900, 95, False)],
)
def test_first_api_call_reports_cache_hit_to_tui_callback(
    cached_tokens, expected_pct, should_be_red
):
    agent = _make_agent(max_iterations=10)
    agent.quiet_mode = False
    response = _mock_response(content="Done.", finish_reason="stop")
    response.usage = SimpleNamespace(
        prompt_tokens=2_000,
        completion_tokens=10,
        total_tokens=2_010,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    agent.client.chat.completions.create.side_effect = [response]
    cache_events = []
    agent._tui_cache_callback = lambda state, pct, read, prompt: cache_events.append(
        (state, pct, read, prompt)
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_vprint") as vprint,
    ):
        result = agent.run_conversation("do something")

    assert result["final_response"] == "Done."
    assert cache_events == [("hit", expected_pct, cached_tokens, 2_000)]
    assert "cache_attribution" not in agent._first_turn_usage
    cache_lines = [
        call.args[0] for call in vprint.call_args_list if "💾 Cache:" in call.args[0]
    ]
    assert len(cache_lines) == 1
    red_pct = f"\033[31m{expected_pct}%\033[0m"
    assert (red_pct in cache_lines[0]) is should_be_red
    assert "post-compression cold prefix" not in cache_lines[0]


def test_post_compression_cache_attribution_survives_retry_then_clears():
    class _RateLimitError(Exception):
        status_code = 429

        def __str__(self):
            return "Error code: 429 - Rate limit exceeded."

    class _NoOpContextEngine(ContextEngine):
        @property
        def name(self):
            return "no-op"

        def update_from_response(self, usage):
            pass

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(
            self,
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ):
            return messages

    def response(cached_tokens: int | None, *, tool_id: str = "", content: str = ""):
        tool_calls = []
        if tool_id:
            tool_calls = [
                SimpleNamespace(
                    id=tool_id,
                    type="function",
                    function=SimpleNamespace(name="noop", arguments="{}"),
                )
            ]
        result = _mock_response(
            content=content,
            finish_reason="tool_calls" if tool_id else "stop",
            tool_calls=tool_calls,
        )
        if cached_tokens is not None:
            result.usage = SimpleNamespace(
                prompt_tokens=2_000,
                completion_tokens=10,
                total_tokens=2_010,
                prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            )
        return result

    agent = _make_agent(max_iterations=10)
    agent.quiet_mode = False
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "noop",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    agent.valid_tool_names = {"noop"}
    engine = _NoOpContextEngine()
    setattr(agent, "context_compressor", engine)
    agent.client.chat.completions.create.side_effect = [
        response(2_000, tool_id="before"),
        _RateLimitError(),
        response(None, tool_id="no-usage"),
        response(1_880, tool_id="after"),
        response(1_900, content="Done."),
    ]
    cache_usage = []
    agent._tui_cache_callback = lambda *_args: cache_usage.append(
        dict(agent._first_turn_usage or {})
    )
    executions = 0

    def execute_tools(assistant_message, messages, _task_id, _api_call_count):
        nonlocal executions
        call_id = assistant_message.tool_calls[0].id
        messages.append({"role": "tool", "tool_call_id": call_id, "content": "ok"})
        if executions == 0:
            setattr(agent, "_awaiting_cache_usage_after_compression", True)
        executions += 1

    with (
        patch("run_agent.time.sleep", return_value=None),
        patch.object(agent, "_execute_tool_calls", side_effect=execute_tools),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_vprint") as vprint,
    ):
        result = agent.run_conversation("work")

    assert result["final_response"] == "Done."
    assert executions == 3
    assert len(cache_usage) == 3
    assert "cache_attribution" not in cache_usage[0]
    assert cache_usage[1]["cache_attribution"] == "post_compression"
    assert cache_usage[1]["cache_read_tokens"] == 1_880
    assert "cache_attribution" not in cache_usage[2]
    assert cache_usage[2]["cache_read_tokens"] == 1_900
    assert agent._first_turn_usage == cache_usage[2]
    assert getattr(agent, "_awaiting_cache_usage_after_compression") is False
    assert not hasattr(engine, "awaiting_real_usage_after_compression")
    assert agent.client.chat.completions.create.call_count == 5
    cache_lines = [
        call.args[0] for call in vprint.call_args_list if "💾 Cache:" in call.args[0]
    ]
    assert len(cache_lines) == 3
    assert ["post-compression warmup (expected)" in line for line in cache_lines] == [
        False,
        True,
        False,
    ]
    assert "\033[31m94%\033[0m" not in cache_lines[1]


def test_run_conversation_partial_stream_recovery_surfaces_explanation():
    """A long recovered partial stream still needs the visible footer.

    Without this, the gateway marks the turn as previewed and suppresses
    the final send, leaving messaging users with a fragment and no reason.
    """
    agent = _make_agent(max_iterations=10)
    empty_stub = _mock_response(content=None, finish_reason="stop")
    recovered = (
        "I inspected the running gateway and found that the current turn "
        "stopped after the provider stream timed out."
    )

    def _fake_api_call(_api_kwargs):
        agent._current_streamed_assistant_text = recovered
        return empty_stub

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do something")

    assert result["turn_exit_reason"] == "partial_stream_recovery"
    assert result["final_response"].startswith(recovered)
    assert "No reply:" in result["final_response"]
    assert result["response_previewed"] is False

def test_heartbeat_malformed_response_never_retries_or_falls_back(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[], usage=None),
        _mock_response(content="unexpected retry", finish_reason="stop"),
    ]

    with (
        patch(
            "tools.runtime_heartbeat.runtime_heartbeat.reset_for_caller"
        ) as reset_deadline,
        patch.object(agent, "_try_activate_fallback", return_value=False) as fallback,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    assert agent.client.chat.completions.create.call_count == 1
    reset_deadline.assert_not_called()
    fallback.assert_not_called()


def test_heartbeat_skips_moa_fanout(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.provider = "moa"
    agent.client.chat.completions.prepare.side_effect = AssertionError(
        "heartbeat must not fan out through MoA"
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    agent.client.chat.completions.prepare.assert_not_called()
    agent.client.chat.completions.create.assert_not_called()


def test_anthropic_heartbeat_skips_without_any_transport_or_fallback(
    heartbeat_event,
):
    agent = _make_agent(max_iterations=10)
    agent.provider = "anthropic"
    agent.api_mode = "anthropic_messages"

    with (
        patch.object(agent, "_create_request_openai_client") as openai_client,
        patch.object(agent, "_create_request_anthropic_client") as anthropic_client,
        patch.object(agent, "_anthropic_messages_create") as anthropic_dispatch,
        patch.object(agent, "_try_activate_fallback") as fallback,
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    openai_client.assert_not_called()
    anthropic_client.assert_not_called()
    anthropic_dispatch.assert_not_called()
    agent.client.chat.completions.create.assert_not_called()
    fallback.assert_not_called()


def test_gemini_chat_completions_heartbeat_skips_transport(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.provider = "gemini"
    agent.requested_provider = "gemini"
    agent.api_mode = "chat_completions"
    agent.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_not_called()


def test_copilot_acp_heartbeat_skips_without_transport_dispatch(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.provider = "copilot-acp"
    agent.requested_provider = "copilot-acp"
    agent.api_mode = "chat_completions"

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_not_called()


def test_heartbeat_skips_provider_switched_during_final_target_inspection(
    monkeypatch,
):
    from tools.runtime_heartbeat import (
        RuntimeHeartbeat,
        canonical_runtime_cache_context_identity,
        canonical_runtime_provider_identity,
    )

    timers = []

    class Timer:
        def __init__(self, _interval, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    inspections = 0
    final_inspection_started = threading.Event()
    release_final_inspection = threading.Event()

    def inspect():
        nonlocal inspections
        inspections += 1
        if inspections == 4:
            final_inspection_started.set()
            assert release_final_inspection.wait(timeout=2)
        return {"alive": True, "progress": True}

    events = queue.Queue()
    manager = RuntimeHeartbeat(event_queue=events, timer_factory=Timer)
    agent = _make_agent(max_iterations=10)
    old_client = agent.client
    old_client.chat.completions.create.return_value = _mock_response("old")
    manager.arm(
        "target",
        caller_id="owner-session",
        kind="delegation",
        interval=1700,
        inspect=inspect,
        provider=canonical_runtime_provider_identity(agent),
        cache_context=canonical_runtime_cache_context_identity(agent),
    )
    timers[0].callback()
    event = events.get_nowait()
    monkeypatch.setattr("tools.runtime_heartbeat.runtime_heartbeat", manager)

    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            agent.run_conversation(
                "[HEARTBEAT] inspect target",
                turn_origin="heartbeat_warm",
                    heartbeat_event=event,
            )
        )
    )
    worker.start()
    assert final_inspection_started.wait(timeout=2)

    new_client = MagicMock()
    new_client.chat.completions.create.return_value = _mock_response("new")
    agent.provider = "openai"
    agent.requested_provider = "openai"
    agent.base_url = "https://api.openai.com/v1"
    agent.client = new_client
    release_final_inspection.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert result[0]["silent_noop"] is True
    old_client.chat.completions.create.assert_not_called()
    new_client.chat.completions.create.assert_not_called()


def test_heartbeat_revalidates_generation_at_provider_boundary(
    heartbeat_event, monkeypatch
):
    agent = _make_agent(max_iterations=10)
    checks = iter((True, False))
    calls = []

    def current(candidate, agent=None, **kwargs):
        calls.append(kwargs)
        return next(checks)

    monkeypatch.setattr(
        "tools.runtime_heartbeat.runtime_heartbeat.is_event_current",
        current,
    )

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    assert calls == [{}, {"consume": True}]
    agent.client.chat.completions.create.assert_not_called()


def test_heartbeat_never_enters_request_middleware(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    agent.client.chat.completions.create.return_value = _mock_response("still alive")

    with (
        patch("hermes_cli.middleware.run_llm_execution_middleware") as middleware,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "[HEARTBEAT] inspect target",
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert result["silent_noop"] is True
    middleware.assert_not_called()
    agent.client.chat.completions.create.assert_called_once()


def test_unknown_heartbeat_is_structured_visible_without_model_call(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    heartbeat_event["status"] = "UNKNOWN"
    heartbeat_event["evidence"] = "no output or CPU progress"

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is False
    assert "UNKNOWN" in result["final_response"]
    assert "no output or CPU progress" in result["final_response"]
    agent.client.chat.completions.create.assert_not_called()


def test_heartbeat_completion_preserves_unowned_marker_and_history(heartbeat_event):
    agent = _make_agent(max_iterations=10)
    heartbeat_history = [{"role": "assistant", "content": "before heartbeat"}]
    ordinary_history = [{"role": "user", "content": "ordinary turn"}]

    def complete_heartbeat(**_kwargs):
        agent._inflight_turn_id = "ordinary-turn"
        agent._session_messages = ordinary_history
        return _mock_response(content="", finish_reason="stop")

    agent.client.chat.completions.create.side_effect = complete_heartbeat

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation(
            "[HEARTBEAT] inspect target",
            conversation_history=heartbeat_history,
            turn_origin="heartbeat_warm",
            heartbeat_event=heartbeat_event,
        )

    assert agent._inflight_turn_id == "ordinary-turn"
    assert agent._session_messages is ordinary_history


def test_heartbeat_skips_custom_alias_resolved_to_gemini_native(heartbeat_event):
    from agent.gemini_native_adapter import GeminiNativeClient

    agent = _make_agent()
    agent.provider = "custom"
    agent.requested_provider = "custom:gemini"
    agent.base_url = "https://generativelanguage.googleapis.com/v1beta"
    gemini_client = GeminiNativeClient(
        api_key="test-key", http_client=MagicMock()
    )
    gemini_client._create_chat_completion = MagicMock()
    agent.client = gemini_client

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    gemini_client._create_chat_completion.assert_not_called()


def test_heartbeat_skips_custom_alias_resolved_to_copilot_acp(heartbeat_event):
    agent = _make_agent()
    agent.provider = "custom"
    agent.base_url = "acp://copilot"
    acp_client = CopilotACPClient(base_url=agent.base_url)
    acp_client._run_prompt = MagicMock(return_value=("", ""))
    agent.client = acp_client

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    acp_client._run_prompt.assert_not_called()


def test_heartbeat_allows_supported_custom_openai_transport(heartbeat_event):
    agent = _make_agent()
    agent.provider = "custom"
    agent.base_url = "https://custom.invalid/v1"

    result = agent.run_conversation(
        "[HEARTBEAT] inspect target",
        turn_origin="heartbeat_warm",
        heartbeat_event=heartbeat_event,
    )

    assert result["silent_noop"] is True
    agent.client.chat.completions.create.assert_called_once()
