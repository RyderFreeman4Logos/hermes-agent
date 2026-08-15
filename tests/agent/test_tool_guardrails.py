"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    canonical_tool_args,
    classify_tool_failure,
)


def test_tool_call_signature_hashes_canonical_nested_unicode_args_without_exposing_raw_args():
    args_a = {
        "z": [{"β": "☤", "a": 1}],
        "a": {"y": 2, "x": "secret-token-value"},
    }
    args_b = {
        "a": {"x": "secret-token-value", "y": 2},
        "z": [{"a": 1, "β": "☤"}],
    }

    assert canonical_tool_args(args_a) == canonical_tool_args(args_b)
    sig_a = ToolCallSignature.from_call("web_search", args_a)
    sig_b = ToolCallSignature.from_call("web_search", args_b)

    assert sig_a == sig_b
    assert len(sig_a.args_hash) == 64
    metadata = sig_a.to_metadata()
    assert metadata == {"tool_name": "web_search", "args_hash": sig_a.args_hash}
    assert "secret-token-value" not in json.dumps(metadata)
    assert "☤" not in json.dumps(metadata)




def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
            },
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8


def test_default_repeated_identical_failed_call_warns_then_halts():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("web_search", args).action == "allow"
        decisions.append(
            controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
        )

    assert decisions[0].action == "allow"
    assert [d.action for d in decisions[1:]] == ["warn", "warn", "warn", "halt"]
    assert {d.code for d in decisions[1:4]} == {"repeated_exact_failure_warning"}
    assert decisions[4].code == "no_progress_loop"
    controller.break_observation_streak()
    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "guardrail_turn_halted"
    assert controller.halt_decision == decisions[4]


def test_default_repeated_equivalent_success_result_halts_after_five():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}
    results = ['{"value":1,"ok":true}', '{"ok":true,"value":1}']

    decisions = [
        controller.after_call("web_search", args, results[index % 2], failed=False)
        for index in range(5)
    ]

    assert all(decision.code != "no_progress_loop" for decision in decisions[:4])
    assert decisions[4].action == "halt"
    assert decisions[4].code == "no_progress_loop"
    assert decisions[4].count == 5
    assert "equivalent result 5 consecutive times" in decisions[4].message


def test_hard_stop_enabled_blocks_repeated_exact_failure_before_next_execution():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"query": "same"}

    assert controller.before_call("web_search", args).action == "allow"
    first = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert first.action == "allow"

    assert controller.before_call("web_search", args).action == "allow"
    second = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert second.action == "warn"
    assert second.code == "repeated_exact_failure_warning"

    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.count == 2














def test_stateful_or_unknown_tools_fail_open_for_repeated_identical_success_output():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )

    for tool_name, args in (
        ("browser_scroll", {"direction": "down"}),
        ("custom_tool", {"x": 1}),
    ):
        decisions = [
            controller.after_call(tool_name, args, "ok", failed=False)
            for _ in range(6)
        ]
        assert all(decision.code != "no_progress_loop" for decision in decisions)


def test_no_progress_halt_requires_consecutive_equivalent_call_and_result():
    controller = ToolCallGuardrailController()
    repeated = ("web_search", {"query": "same"}, '{"value":1}')

    for changed in (
        ("web_search", {"query": "different"}, repeated[2], False),
        ("read_file", repeated[1], repeated[2], False),
        ("web_search", repeated[1], '{"value":2}', False),
        ("web_search", repeated[1], repeated[2], True),
    ):
        decisions = [
            controller.after_call(*repeated, failed=False) for _ in range(4)
        ]
        decisions.append(
            controller.after_call(*changed[:3], failed=changed[3])
        )
        decisions.extend(
            controller.after_call(*repeated, failed=False) for _ in range(4)
        )
        assert all(decision.code != "no_progress_loop" for decision in decisions)
        controller.reset_for_turn()


def test_no_progress_halt_fails_open_for_none_and_structured_results():
    controller = ToolCallGuardrailController()

    for result in (None, {"value": 1}):
        decisions = [
            controller.after_call("web_search", {"x": 1}, result)
            for _ in range(6)
        ]
        assert all(decision.code != "no_progress_loop" for decision in decisions)
        controller.reset_for_turn()






# ── Per-turn runaway-loop caps (Claude Code v2.1.212, Week 29) ──────────────

from agent.tool_guardrails import LoopCapConfig  # noqa: E402






def test_loop_cap_zero_disables_and_junk_falls_back():
    # 0 is a legitimate "unlimited" value; negatives / junk fall back to default.
    assert LoopCapConfig.from_mapping({"max_web_searches": 0}).max_web_searches == 0
    assert LoopCapConfig.from_mapping({"max_web_searches": -5}).max_web_searches == 50
    assert LoopCapConfig.from_mapping({"max_subagents": "nope"}).max_subagents == 50


def test_web_search_cap_blocks_after_limit_regardless_of_hard_stop():
    # Loop caps fire even with hard_stop_enabled=False (the per-turn loop
    # detector's flag). Each distinct query avoids the loop detector so we know
    # the block came from the loop cap, not exact-failure repetition.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(max_web_searches=3),
        )
    )
    for i in range(3):
        assert controller.before_call("web_search", {"query": f"q{i}"}).action == "allow"
    decision = controller.before_call("web_search", {"query": "q4"})
    assert decision.action == "block"
    assert decision.code == "loop_web_search_cap"
    assert decision.should_halt is True









