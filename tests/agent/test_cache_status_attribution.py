from types import SimpleNamespace

import agent.conversation_loop as conversation_loop


def test_ingest_usage_reports_first_call_and_post_compression_replacement():
    seen = []
    agent = SimpleNamespace(
        _awaiting_cache_usage_after_compression=False,
        _first_turn_usage=None,
        _tui_cache_callback=lambda *args: seen.append(args),
    )

    first = {
        "cache_read_tokens": 1_900,
        "cache_write_tokens": 0,
        "cache_telemetry_present": True,
        "prompt_tokens": 2_000,
    }
    assert conversation_loop._ingest_successful_provider_usage(agent, first, first_call=True) is False

    agent._awaiting_cache_usage_after_compression = True
    post_compression = {**first, "cache_read_tokens": 1_880}
    assert (
        conversation_loop._ingest_successful_provider_usage(
            agent, post_compression, first_call=False
        )
        is True
    )

    stable = {**first, "cache_read_tokens": 1_900}
    assert conversation_loop._ingest_successful_provider_usage(agent, stable, first_call=False) is False

    assert [event[:2] for event in seen] == [("hit", 95), ("hit", 94), ("hit", 95)]
    assert "cache_attribution" not in agent._first_turn_usage
    assert agent._awaiting_cache_usage_after_compression is False
