"""Post-compression cache attribution is session-scoped and display-safe."""

from __future__ import annotations

from types import SimpleNamespace

from agent import cache_attribution


class _SessionDB:
    def __init__(self) -> None:
        self.get_calls = 0
        self.values: dict[tuple[str, str], object] = {}

    def get_session_model_config_value(self, session_id, key, default=None):
        self.get_calls += 1
        return self.values.get((session_id, key), default)

    def patch_session_model_config(self, session_id, patch) -> None:
        for key, value in patch.items():
            if value is None:
                self.values.pop((session_id, key), None)
            else:
                self.values[(session_id, key)] = value


def test_post_compression_marker_survives_restart_and_sample_is_one_shot():
    db = _SessionDB()
    before_restart = SimpleNamespace(
        session_id="conversation-a",
        _session_db=db,
    )
    cache_attribution.set_post_compression_cache_pending(before_restart, True)
    assert db.get_session_model_config_value(
        "conversation-a", cache_attribution.POST_COMPRESSION_CACHE_PENDING_KEY
    ) is True

    resumed = SimpleNamespace(session_id="conversation-a", _session_db=db)
    info = cache_attribution.record_first_turn_cache_info(
        resumed,
        {
            "cache_read_tokens": 1,
            "cache_write_tokens": 0,
            "prompt_tokens": 10_000,
        },
        telemetry_present=True,
    )

    assert info == {
        "attribution": "post_compression",
        "cached_tokens": 1,
        "prompt_tokens": 10_000,
        "state": "hit",
        "text": "Cache: 1/10,000 tokens (<1% hit, 0 written)",
        "write_tokens": 0,
    }
    assert db.get_session_model_config_value(
        "conversation-a", cache_attribution.POST_COMPRESSION_CACHE_PENDING_KEY
    ) is None
    assert cache_attribution.consume_turn_cache_info(resumed) == info
    assert cache_attribution.consume_turn_cache_info(resumed) is None

    other = SimpleNamespace(session_id="conversation-b", _session_db=db)
    assert cache_attribution.consume_turn_cache_info(other) is None


def test_cache_states_and_logs_preserve_real_scalars_without_rounding_hits_to_zero():
    cases = [
        (
            {"cache_read_tokens": 128, "cache_write_tokens": 0, "prompt_tokens": 165_611},
            True,
            "hit",
            "Cache: 128/165,611 tokens (<1% hit, 0 written)",
            "cache_state=hit cache_read=128 cache_write=0 cache_prompt=165611",
        ),
        (
            {"cache_read_tokens": 0, "cache_write_tokens": 0, "prompt_tokens": 2_000},
            True,
            "miss",
            "Cache: 0/2,000 tokens (0% hit, 0 written)",
            "cache_state=miss cache_read=0 cache_write=0 cache_prompt=2000",
        ),
        (
            {"cache_read_tokens": 0, "cache_write_tokens": 2_000, "prompt_tokens": 2_000},
            True,
            "cold_write",
            "Cache: 0/2,000 tokens (0% hit, 2,000 written)",
            "cache_state=cold_write cache_read=0 cache_write=2000 cache_prompt=2000",
        ),
        (
            {"cache_read_tokens": 0, "cache_write_tokens": 0, "prompt_tokens": 2_000},
            False,
            "no_field",
            "Cache: unavailable",
            "cache_state=no_field cache_prompt=2000",
        ),
    ]

    for usage, telemetry_present, state, text, suffix in cases:
        info = cache_attribution.cache_info_from_usage(
            usage, telemetry_present=telemetry_present
        )
        assert info["state"] == state
        assert info["text"] == text
        assert cache_attribution.cache_log_suffix(info) == suffix


def test_first_provider_sample_is_not_replaced_by_later_tool_loop_calls():
    agent = SimpleNamespace(session_id="conversation-a", _session_db=_SessionDB())
    first = cache_attribution.record_first_turn_cache_info(
        agent,
        {"cache_read_tokens": 0, "cache_write_tokens": 0, "prompt_tokens": 2_000},
        telemetry_present=True,
    )
    later = cache_attribution.record_first_turn_cache_info(
        agent,
        {"cache_read_tokens": 1_900, "cache_write_tokens": 0, "prompt_tokens": 2_000},
        telemetry_present=True,
    )

    assert first["state"] == "miss"
    assert later is None
    assert cache_attribution.consume_turn_cache_info(agent) == first


def test_post_compression_sample_replaces_an_earlier_sample_in_the_same_turn():
    agent = SimpleNamespace(session_id="conversation-a", _session_db=_SessionDB())
    first = cache_attribution.record_first_turn_cache_info(
        agent,
        {"cache_read_tokens": 1_900, "cache_write_tokens": 0, "prompt_tokens": 2_000},
        telemetry_present=True,
    )
    cache_attribution.set_post_compression_cache_pending(agent, True)

    post_compression = cache_attribution.record_first_turn_cache_info(
        agent,
        {"cache_read_tokens": 0, "cache_write_tokens": 2_000, "prompt_tokens": 2_000},
        telemetry_present=True,
    )

    assert first["state"] == "hit"
    assert post_compression["attribution"] == "post_compression"
    assert post_compression["state"] == "cold_write"
    assert cache_attribution.consume_turn_cache_info(agent) == post_compression


def test_omitted_pydantic_cache_field_is_not_treated_as_provider_telemetry():
    usage = SimpleNamespace(
        cached_tokens=0,
        model_fields_set={"prompt_tokens"},
        prompt_tokens=2_000,
    )

    assert cache_attribution.cache_telemetry_present(usage) is False


def test_absent_durable_marker_is_lazy_loaded_only_once_per_agent_instance():
    db = _SessionDB()
    agent = SimpleNamespace(session_id="conversation-a", _session_db=db)

    assert cache_attribution.clear_post_compression_cache_pending(agent) is False
    assert cache_attribution.clear_post_compression_cache_pending(agent) is False
    assert db.get_calls == 1
