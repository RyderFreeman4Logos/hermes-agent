"""Completion backlogs preserve results without multiplying autonomous turns."""
from evals.completion_backlog_probe import probe


def test_ready_completions_share_one_turn_across_interactive_routes(tmp_path):
    for surface in ("cli", "poller", "post-turn"):
        for scenario in ("backlog", "single", "mixed"):
            result = probe(surface, scenario, tmp_path / surface / scenario)
            expected_turns = 3 if (surface, scenario) == ("poller", "mixed") else (5 if scenario == "mixed" else 1)
            assert result["wire_turns"] == expected_turns, result
            if (surface, scenario) == ("poller", "mixed"):
                assert result["payload_order"] == [2, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2], result
                assert result["payload_delivery_counts"] == [1] * 14, result
                assert result["turn_payload_indices"] == [[4], [9], [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]], result
            else:
                assert result["payload_order"] == sorted(result["payload_order"]), result
            if scenario == "mixed":
                assert result["delegation_delivered_once"], result
            assert result["all_payloads_preserved"], result
            if scenario == "single":
                assert result["single_exact"], result


def test_consumed_or_foreign_completions_never_start_a_turn(tmp_path):
    for surface in ("cli", "poller", "post-turn"):
        for scenario in ("consumed", "foreign"):
            result = probe(surface, scenario, tmp_path / surface / scenario)
            assert result["wire_turns"] == 0, result
            if scenario == "foreign":
                assert result["queue_remaining"] == result["children"], result
