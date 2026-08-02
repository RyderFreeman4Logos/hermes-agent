"""Runtime tests for tool-call loop guardrails."""

import json
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _json_payload(content: str) -> dict:
    """Parse a JSON result with or without the untrusted-source wrapper."""
    return json.loads(content[content.index("{") : content.rindex("}") + 1])


def _make_agent(*tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
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
    return agent


def _seed_exact_failures(agent: AIAgent, tool_name: str, args: dict, count: int = 2) -> None:
    for _ in range(count):
        agent._tool_guardrails.after_call(
            tool_name,
            args,
            json.dumps({"error": "boom"}),
            failed=True,
        )


def _hard_stop_config(**overrides) -> dict:
    cfg = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }
    cfg["tool_loop_guardrails"].update(overrides)
    return cfg


def test_default_sequential_path_warns_repeated_exact_failure_without_blocking_execution():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-soft")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    assert len(starts) == 1
    assert any(event[0][0] == "tool.completed" for event in progress)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-soft"
    assert "repeated_exact_failure_warning" in messages[0]["content"]
    assert "repeated_exact_failure_block" not in messages[0]["content"]
    assert agent._tool_guardrail_halt_decision is None


def test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution():
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == []
    assert progress == []
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-block"
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_sequential_after_call_appends_guidance_to_tool_result_without_extra_messages():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args, count=1)
    tc = _mock_tool_call("web_search", json.dumps(args), "c-warn")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert [m["role"] for m in messages] == ["tool"]
    assert messages[0]["tool_call_id"] == "c-warn"
    assert "Tool loop warning" in messages[0]["content"]
    assert "repeated_exact_failure_warning" in messages[0]["content"]


def test_same_tool_failure_warning_tells_model_to_recover_with_tools():
    agent = _make_agent("terminal")
    guardrails = getattr(agent, "_tool_guardrails")
    guardrails.after_call(
        "terminal",
        {"command": "bad-1"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    guardrails.after_call(
        "terminal",
        {"command": "bad-2"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    tc = _mock_tool_call("terminal", json.dumps({"command": "bad-3"}), "c-recover")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"exit_code": 1})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    content = messages[0]["content"]
    assert "same_tool_failure_warning" in content
    assert "Do not switch to text-only replies" in content
    assert "keep using tools" in content
    assert "pwd && ls -la" in content
    assert "absolute path" in content
    assert "different tool" in content


def test_config_enabled_hard_stop_concurrent_path_does_not_submit_blocked_calls_and_preserves_result_order():
    agent = _make_agent("web_search", config=_hard_stop_config())
    blocked_args = {"query": "blocked"}
    allowed_args = {"query": "allowed"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    starts = []
    progress_events = []
    agent.tool_start_callback = lambda tool_call_id, name, args: starts.append((tool_call_id, name, args))
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress_events.append((event, name, args, kw))
    calls = [
        _mock_tool_call("web_search", json.dumps(blocked_args), "c-block"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allow"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allow")]
    assert [m["tool_call_id"] for m in messages] == ["c-block", "c-allow"]
    assert "repeated_exact_failure_block" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
    assert starts == [("c-allow", "web_search", allowed_args)]
    started_events = [event for event in progress_events if event[0] == "tool.started"]
    completed_events = [event for event in progress_events if event[0] == "tool.completed"]
    assert started_events == [("tool.started", "web_search", allowed_args, {})]
    assert len(completed_events) == 1
    assert completed_events[0][1] == "web_search"


def test_relay_rewrite_precedes_sequential_policy_approval_checkpoint_and_dispatch():
    agent = _make_agent("write_file")
    original_args = {"path": "/original/path", "content": "old"}
    final_args = {"path": "/approved/path", "content": "new"}
    tc = _mock_tool_call("write_file", json.dumps(original_args), "c-rewrite")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    observed = {
        "plugin": [],
        "guardrail": [],
        "approval": [],
        "checkpoint": [],
        "start": [],
        "dispatch": [],
    }

    original_before_call = agent._tool_guardrails.before_call

    def observe_guardrail(name, args):
        observed["guardrail"].append((name, dict(args)))
        return original_before_call(name, args)

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(final_args)), dict(final_args)

    def observe_plugin(name, args, **kwargs):
        del kwargs
        observed["plugin"].append((name, dict(args)))
        return None

    def observe_approval(name, args):
        observed["approval"].append((name, dict(args)))
        return None

    def dispatch(name, args, task_id, **kwargs):
        del task_id, kwargs
        observed["dispatch"].append((name, dict(args)))
        return json.dumps({"ok": True})

    agent._checkpoint_mgr = SimpleNamespace(
        enabled=True,
        get_working_dir_for_path=lambda path: path,
        ensure_checkpoint=lambda path, reason: observed["checkpoint"].append(
            (path, reason)
        ),
    )
    agent.tool_start_callback = lambda _call_id, name, args: observed["start"].append(
        (name, dict(args))
    )

    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch(
            "hermes_cli.plugins.resolve_pre_tool_block",
            side_effect=observe_plugin,
        ),
        patch.object(agent._tool_guardrails, "before_call", side_effect=observe_guardrail),
        patch(
            "acp_adapter.edit_approval.maybe_require_edit_approval",
            side_effect=observe_approval,
        ),
        patch("model_tools.registry.dispatch", side_effect=dispatch),
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    expected = [("write_file", final_args)]
    assert observed["plugin"] == expected
    assert observed["guardrail"] == expected
    assert observed["approval"] == expected
    assert observed["start"] == expected
    assert observed["dispatch"] == expected
    assert observed["checkpoint"] == [
        ("/approved/path", "before write_file")
    ]


def test_relay_rewrite_is_guarded_before_dispatch_in_concurrent_path():
    agent = _make_agent("web_search", config=_hard_stop_config())
    original_args = {"query": "original"}
    blocked_args = {"query": "blocked"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    tc = _mock_tool_call("web_search", json.dumps(original_args), "c-rewrite-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    starts = []

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(blocked_args)), dict(blocked_args)

    agent.tool_start_callback = lambda *args: starts.append(args)
    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    dispatch.assert_not_called()
    assert starts == []
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch("hermes_cli.plugins.resolve_pre_tool_block", return_value="plugin policy"),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert "plugin policy" in messages[0]["content"]
    assert agent._tool_guardrails.before_call("web_search", args).action == "allow"


def test_default_run_conversation_warns_without_guardrail_halt():
    agent = _make_agent("web_search", max_iterations=10)
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 4)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
    assert result["final_response"] == "done"
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("repeated_exact_failure_warning" in content for content in tool_contents)




def test_guardrail_halt_emits_final_response_through_stream_delta_callback():
    """Regression for #30770: when the guardrail halts the loop, the
    synthesized halt message must be pushed through ``stream_delta_callback``
    so SSE/TUI clients see why the agent stopped instead of a silent stream
    close.  Without this the chat-completions SSE writer drains an empty
    queue and emits a finish chunk with zero content (indistinguishable
    from a crash for Open WebUI and similar clients).
    """
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)
    # The mocked client returns SimpleNamespace responses which aren't
    # iterable as streaming chunks; force the non-streaming code path so
    # the guardrail-halt branch is reached without engaging the real
    # streaming machinery.
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    halt_text = result["final_response"]
    assert "stopped retrying" in halt_text

    # The halt message must have been pushed through the callback at least
    # once.  Empty-queue SSE writers were the bug — clients saw no content
    # delta before the finish chunk.
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert halt_text in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )


def _run_tool_sequence(
    agent, steps, final_response="done", prompt="exercise the tool sequence"
):
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(name, json.dumps(args), f"step-{i}")
            ],
        )
        for i, (name, args, _result) in enumerate(steps)
    ] + [_mock_response(content=final_response, finish_reason="stop", tool_calls=None)]

    with (
        patch(
            "run_agent.handle_function_call",
            side_effect=[result for _name, _args, result in steps],
        ) as dispatch,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(prompt)

    return result, dispatch


def test_identical_tool_and_result_loop_halts_after_five_without_sixth_work():
    agent = _make_agent("custom_tool", "skill_manage", max_iterations=200)
    agent._skill_nudge_interval = 1
    starts = []
    completions = []
    progress = []
    agent.tool_start_callback = lambda *args: starts.append(args)
    agent.tool_complete_callback = lambda *args: completions.append(args)
    agent.tool_progress_callback = lambda *args, **kwargs: progress.append((args, kwargs))

    args_a = {"secret": "args-must-not-leak", "nested": {"b": 2, "a": 1}}
    args_b = {"nested": {"a": 1, "b": 2}, "secret": "args-must-not-leak"}
    result_a = json.dumps({"secret": "result-must-not-leak", "ok": True})
    result_b = json.dumps({"ok": True, "secret": "result-must-not-leak"})
    steps = [
        ("custom_tool", args_a if i % 2 else args_b, result_a if i % 2 else result_b)
        for i in range(1, 7)
    ]
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(name, json.dumps(args), f"same-{i}")
            ],
        )
        for i, (name, args, _result) in enumerate(steps, 1)
    ]

    with (
        patch(
            "run_agent.handle_function_call",
            side_effect=[result for _name, _args, result in steps],
        ) as dispatch,
        patch.object(agent, "_persist_session") as persist,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_spawn_background_review") as background_review,
    ):
        result = agent.run_conversation("repeat the same work forever")

    assert agent.max_iterations == 200
    assert result["turn_exit_reason"] == "no_progress_loop"
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["api_calls"] == 5
    assert result["guardrail"]["code"] == "no_progress_loop"
    assert result["guardrail"]["count"] == 5
    audit = json.dumps(result["guardrail"])
    assert "args-must-not-leak" not in audit
    assert "result-must-not-leak" not in audit

    assert agent.client.chat.completions.create.call_count == 5
    assert dispatch.call_count == 5
    assert len(starts) == 5
    assert len(completions) == 5
    assert sum(event[0][0] == "tool.completed" for event in progress) == 5
    assert persist.call_count == 2
    background_review.assert_not_called()

    roles = [message["role"] for message in result["messages"]]
    assert roles == ["user"] + [role for _ in range(5) for role in ("assistant", "tool")] + ["assistant"]
    user_messages = [message for message in result["messages"] if message["role"] == "user"]
    assert len(user_messages) == 1
    assert all(message.get("display_kind") != "auto_continue" for message in result["messages"])


@pytest.mark.parametrize(
    ("tool_name", "tool_args"),
    [
        pytest.param("web_search", {"query": "same"}, id="parallel-safe"),
        pytest.param("terminal", {"command": "pwd"}, id="sequential"),
    ],
)
def test_identical_tool_batch_fences_sixth_dispatch_and_preserves_every_result_id(
    tool_name, tool_args
):
    agent = _make_agent(tool_name, max_iterations=200)
    # Distinct raw JSON survives provider-call deduplication while parsing to
    # the same canonical tool signature used by the no-progress guardrail.
    calls = [
        _mock_tool_call(tool_name, json.dumps(tool_args) + (" " * i), f"batch-{i}")
        for i in range(1, 7)
    ]
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=calls)
    ]
    starts = []
    completions = []
    progress = []
    agent.tool_start_callback = lambda *args: starts.append(args)
    agent.tool_complete_callback = lambda *args: completions.append(args)
    agent.tool_progress_callback = lambda *args, **kwargs: progress.append((args, kwargs))

    with (
        patch(
            "run_agent.handle_function_call",
            return_value=json.dumps({"ok": "same-result"}),
        ) as dispatch,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("run one repeated batch")

    assert result["turn_exit_reason"] == "no_progress_loop"
    assert result["api_calls"] == 1
    assert result["guardrail"]["code"] == "no_progress_loop"
    assert result["guardrail"]["count"] == 5
    assert agent.client.chat.completions.create.call_count == 1
    expected_physical_ids = [f"batch-{i}" for i in range(1, 6)]
    assert dispatch.call_count == 5
    assert [call.kwargs["tool_call_id"] for call in dispatch.call_args_list] == (
        expected_physical_ids
    )
    assert [event[0] for event in starts] == expected_physical_ids
    assert [event[0] for event in completions] == expected_physical_ids
    assert sum(event[0][0] == "tool.completed" for event in progress) == 5

    tool_messages = [m for m in result["messages"] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == [
        f"batch-{i}" for i in range(1, 7)
    ]
    skipped = _json_payload(tool_messages[-1]["content"])
    assert skipped["skipped"] is True
    assert skipped["guardrail"]["code"] == "no_progress_loop"
    assert skipped["guardrail"]["count"] == 5
    assert [m["role"] for m in result["messages"]] == [
        "user",
        "assistant",
        *(["tool"] * 6),
        "assistant",
    ]


def test_streak_four_then_matching_batch_call_skips_every_later_physical_call():
    agent = _make_agent("web_search", "terminal", max_iterations=200)
    repeated_args = {"query": "same"}
    repeated_result = json.dumps({"ok": "same-result"})
    for expected_count in range(1, 5):
        decision = agent._tool_guardrails.after_call(
            "web_search", repeated_args, repeated_result, failed=False
        )
        assert decision.count == expected_count

    calls = [
        _mock_tool_call("web_search", json.dumps(repeated_args), "mixed-1"),
        _mock_tool_call("terminal", json.dumps({"command": "pwd"}), "mixed-2"),
        _mock_tool_call("web_search", json.dumps({"query": "different"}), "mixed-3"),
    ]
    messages = []
    with patch(
        "run_agent.handle_function_call", return_value=repeated_result
    ) as dispatch:
        agent._execute_tool_calls(
            SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
        )

    assert dispatch.call_count == 1
    assert [m["tool_call_id"] for m in messages] == [
        "mixed-1",
        "mixed-2",
        "mixed-3",
    ]
    for skipped_message in messages[1:]:
        skipped = _json_payload(skipped_message["content"])
        assert skipped["skipped"] is True
        assert skipped["guardrail"]["count"] == 5


def test_identical_args_with_changing_results_dispatches_all_six_without_halt():
    agent = _make_agent("web_search", max_iterations=200)
    calls = [
        _mock_tool_call("web_search", json.dumps({"query": "same"}), f"changing-{i}")
        for i in range(1, 7)
    ]
    messages = []
    physical = []

    def changing_result(name, args, task_id, **kwargs):
        physical.append(kwargs["tool_call_id"])
        return json.dumps({"result": len(physical)})

    with patch("run_agent.handle_function_call", side_effect=changing_result):
        agent._execute_tool_calls(
            SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
        )

    assert physical == [f"changing-{i}" for i in range(1, 7)]
    assert [m["tool_call_id"] for m in messages] == physical
    assert agent._tool_guardrail_halt_decision is None


def test_six_distinct_parallel_safe_calls_still_overlap():
    agent = _make_agent("web_search", max_iterations=200)
    calls = [
        _mock_tool_call(
            "web_search", json.dumps({"query": f"distinct-{i}"}), f"distinct-{i}"
        )
        for i in range(1, 7)
    ]
    messages = []
    rendezvous = threading.Barrier(6)
    physical = []
    lock = threading.Lock()

    def overlapping_result(name, args, task_id, **kwargs):
        with lock:
            physical.append(kwargs["tool_call_id"])
        rendezvous.wait(timeout=3)
        return json.dumps({"result": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=overlapping_result):
        agent._execute_tool_calls(
            SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
        )

    assert not rendezvous.broken
    assert set(physical) == {f"distinct-{i}" for i in range(1, 7)}
    assert [m["tool_call_id"] for m in messages] == [
        f"distinct-{i}" for i in range(1, 7)
    ]
    assert agent._tool_guardrail_halt_decision is None


@pytest.mark.parametrize("reset_kind", ["args", "result", "tool"])
def test_no_progress_loop_only_counts_consecutive_equivalent_observations(reset_kind):
    agent = _make_agent(
        "web_search", "read_file", max_iterations=200, config=_hard_stop_config()
    )
    base = ("web_search", {"value": 1}, json.dumps({"value": 1}))
    reset = {
        "args": ("web_search", {"value": 2}, base[2]),
        "result": ("web_search", base[1], json.dumps({"value": 2})),
        "tool": ("read_file", base[1], base[2]),
    }[reset_kind]

    result, dispatch = _run_tool_sequence(agent, [base] * 4 + [reset] + [base] * 4)

    assert dispatch.call_count == 9
    assert result["final_response"] == "done"
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result


def test_no_progress_loop_state_resets_for_a_real_new_user_turn():
    agent = _make_agent("web_search", max_iterations=200, config=_hard_stop_config())
    step = ("web_search", {"value": 1}, json.dumps({"value": 1}))

    first, first_dispatch = _run_tool_sequence(
        agent, [step] * 4, final_response="first done", prompt="first user turn"
    )
    second, second_dispatch = _run_tool_sequence(
        agent, [step] * 4, final_response="second done", prompt="second user turn"
    )

    assert first_dispatch.call_count == 4
    assert second_dispatch.call_count == 4
    assert first["final_response"] == "first done"
    assert second["final_response"] == "second done"
    assert first["session_id"] == second["session_id"]
    assert first["turn_exit_reason"].startswith("text_response")
    assert second["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in first
    assert "guardrail" not in second
