"""Public-constructor regressions for fallback admission and override provenance (#41)."""

import copy
import json
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from agent import chat_completion_helpers as completion
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.fixture
def runtime():
    providers = [
        dict(name=name, provider_key=name, base_url=f"https://{name}.example.invalid/v1",
             api_key=f"{name}-test-key", context_length=128000,
             extra_body={f"{name}_only": {"items": [name]}})
        for name in "abc"
    ]
    config = {
        "model": {"default": "gpt-4o", "provider": "custom:a",
                  "base_url": providers[0]["base_url"], "context_length": 128000},
        "custom_providers": providers,
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "compression": {"enabled": True},
    }
    (get_hermes_home() / "config.yaml").write_text(json.dumps(config))

    def make(overrides=None):
        return AIAgent(
            model="gpt-4o", provider="custom:a", api_mode="chat_completions",
            base_url=providers[0]["base_url"], api_key="a-test-key",
            request_overrides=overrides,
            fallback_model=[dict(provider=f"custom:{name}", model="gpt-4o",
                                 base_url=f"https://{name}.example.invalid/v1",
                                 api_key=f"{name}-test-key", api_mode="chat_completions")
                            for name in "bc"],
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            skip_background_review=True, enabled_toolsets=[],
        )

    with patch("model_tools.get_tool_definitions", return_value=[]), patch(
        "model_tools.check_toolset_requirements", return_value={}
    ):
        yield make


def _request_body(agent):
    return completion.build_api_kwargs(
        agent, [{"role": "user", "content": "boundary"}], tools_for_api=[]
    )["extra_body"]


@pytest.mark.parametrize("boundary", ["notice", "capabilities"])
def test_rejected_fallback_preserves_compressor_until_admission(runtime, tmp_path, boundary):
    agent = runtime()
    compressor = agent.context_compressor
    db = SessionDB(db_path=tmp_path / "compression.db")
    try:
        db.create_session("admission", source="cli")
        compressor.bind_session_state(db, "admission")
        compressor._record_ineffective_compression_verdict(2)
        compressor._fallback_compression_streak = 3
        compressor._persist_fallback_compression_streak()
        compressor._consecutive_overload_aborts = 4
        compressor._persist_consecutive_overload_aborts()
        compressor._record_compression_failure_cooldown(300, "existing runtime fault")
        db.patch_session_model_config("admission", {"_proactive_prune_rearm_tokens": 90000})
        compressor._proactive_prune_rearm_tokens = 90000
        compressor._last_reclaim_block_warn = ("existing", 90000)
        compressor.last_real_prompt_tokens = 81000
        compressor.last_compression_rough_tokens = 79000
        compressor.awaiting_real_usage_after_compression = True
        compressor._provider_omits_usage = True
        compressor._consecutive_timeout_failures = 2
        compressor._consecutive_truncation_failures = 3
        compressor._aux_context_ceiling = 70000
        compressor._apply_threshold_tokens_cap()
        # Capture the actual owner, not a hand-maintained rollback field allowlist.
        before = dict(vars(compressor))
        durable = db.get_session("admission")
        primary = copy.deepcopy(agent._primary_runtime)
        target = ("agent.chat_completion_helpers._buffer_fallback_notice" if boundary == "notice"
                  else "agent.native_compaction.resolve_native_compaction_capabilities")
        with patch(target, side_effect=RuntimeError("late candidate rejection")):
            assert not agent._try_activate_fallback()
        assert {"memory": vars(compressor), "durable": db.get_session("admission")} == {
            "memory": before, "durable": durable,
        }
        assert agent.provider == "custom:a"
        assert agent._primary_runtime == primary
        assert not agent._fallback_activated

        # Admission still performs all normal runtime-change resets, including durable ones.
        assert not agent._restore_primary_runtime()
        assert agent._try_activate_fallback()
        assert compressor.provider == agent.provider == "custom:b"
        assert compressor._aux_context_ceiling is None
        assert compressor.last_real_prompt_tokens == compressor.last_compression_rough_tokens == 0
        assert not compressor.awaiting_real_usage_after_compression
        assert not compressor._provider_omits_usage
        assert compressor._consecutive_timeout_failures == compressor._consecutive_truncation_failures == 0
        assert compressor._consecutive_overload_aborts == compressor._ineffective_compression_count == 0
        assert compressor._proactive_prune_rearm_tokens == 0
        assert compressor._last_reclaim_block_warn is None
        assert db.get_compression_ineffective_count("admission") == 0
        assert db.get_compression_fallback_streak("admission") == 0
        assert db.get_compression_overload_streak("admission") == 0
        assert db.get_compression_failure_cooldown("admission") is None
        assert db.get_session_model_config_value("admission", "_proactive_prune_rearm_tokens", 0) == 0
        assert not agent._restore_primary_runtime()  # exhaustion cooldown still gates recovery
        with patch("agent.agent_runtime_helpers.time.monotonic",
                   return_value=agent._rate_limited_until + 1):
            assert agent._restore_primary_runtime()
        assert compressor.provider == agent.provider == "custom:a"
    finally:
        db.close()


@pytest.mark.parametrize("entry", ["fallback", "switch"])
@pytest.mark.parametrize("boundary", ["threshold", "tail", "entry", "durable_prefix", "complete"])
def test_compressor_publication_failure_respects_commit(runtime, tmp_path, entry, boundary):
    agent = runtime()
    agent._fallback_chain = agent._fallback_chain[:1]
    agent._fallback_chain[0]["model"] = "gpt-4o-mini"
    cc = agent.context_compressor
    db = SessionDB(db_path=tmp_path / "publication.db")
    try:
        db.create_session("publication", source="cli")
        cc.bind_session_state(db, "publication")
        cc._record_ineffective_compression_verdict(2)
        cc._fallback_compression_streak = 3
        cc._persist_fallback_compression_streak()
        cc._consecutive_overload_aborts = 4
        cc._persist_consecutive_overload_aborts()
        cc._record_compression_failure_cooldown(300, "primary fault")
        cc.last_real_prompt_tokens = 81000
        cc.last_compression_rough_tokens = 79000
        cc.awaiting_real_usage_after_compression = True
        cc._aux_context_ceiling = 70000
        cc._apply_threshold_tokens_cap()
        if boundary == "threshold":
            cc.model_thresholds = {"gpt-4o-mini": float("nan")}
        elif boundary == "tail":
            cc.tail_mode = "balanced"
            cc.summary_target_ratio = float("nan")
        before, durable = dict(vars(cc)), db.get_session("publication")
        primary = copy.deepcopy(agent._primary_runtime)
        update = cc.update_model

        def completed_then_fail(**kwargs):
            update(**kwargs)
            raise RuntimeError("after completed publication")

        fault = {
            "entry": lambda: patch.object(cc, "update_model", side_effect=RuntimeError("entry")),
            "durable_prefix": lambda: patch.object(cc, "_clear_compression_failure_cooldown",
                                                   side_effect=RuntimeError("durable prefix")),
            "complete": lambda: patch.object(cc, "update_model", side_effect=completed_then_fail),
        }.get(boundary, nullcontext)
        committed = boundary in {"durable_prefix", "complete"}
        with fault():
            if entry == "fallback":
                assert agent._try_activate_fallback() is committed
            else:
                rejection = nullcontext() if committed else pytest.raises((ValueError, RuntimeError))
                with rejection:
                    agent.switch_model("gpt-4o-mini", "custom:b", api_key="b-test-key",
                                       base_url="https://b.example.invalid/v1",
                                       api_mode="chat_completions", capabilities={})
        if not committed:
            assert vars(cc) == before
            assert db.get_session("publication") == durable
            assert agent.provider == "custom:a"
            assert agent._primary_runtime == primary
            assert not agent._fallback_activated
            assert not agent._restore_primary_runtime()
            # Repair the invalid setting; the same public entry must remain usable.
            cc.model_thresholds = {}
            cc.summary_target_ratio = 0.2
            if entry == "fallback":
                assert agent._try_activate_fallback()
            else:
                agent.switch_model("gpt-4o-mini", "custom:b", api_key="b-test-key",
                                   base_url="https://b.example.invalid/v1",
                                   api_mode="chat_completions", capabilities={})
        assert cc.provider == agent.provider == "custom:b"
        assert cc.model == agent.model == "gpt-4o-mini"
        assert cc._aux_context_ceiling is None
        assert cc.last_real_prompt_tokens == cc.last_compression_rough_tokens == 0
        assert not cc.awaiting_real_usage_after_compression
        assert db.get_compression_ineffective_count("publication") == 0
        assert db.get_compression_fallback_streak("publication") == 0
        if boundary == "durable_prefix":
            # This write prefix really committed. Do not pretend a memory rollback undid it.
            expected = dict(durable)
            after = db.get_session("publication")
            for key in ("compression_ineffective_count", "compression_fallback_streak"):
                expected[key] = 0
            assert after == expected
            assert cc._consecutive_overload_aborts == 4
        else:
            assert db.get_compression_failure_cooldown("publication") is None
            assert db.get_compression_overload_streak("publication") == 0
        if entry == "fallback":
            assert agent._fallback_activated and agent._provider_fallback_active
            assert agent._primary_runtime == primary
            with patch("agent.agent_runtime_helpers.time.monotonic",
                       return_value=getattr(agent, "_rate_limited_until", 0) + 1):
                assert agent._restore_primary_runtime()
            assert cc.provider == agent.provider == "custom:a"
            assert not agent._fallback_activated
        else:
            assert agent._primary_runtime["provider"] == "custom:b"
            assert not agent._restore_primary_runtime()
    finally:
        db.close()


@pytest.mark.parametrize("overrides", [None, {}, {"extra_body": {"caller": {"items": ["kept"]}}}])
def test_constructor_and_switch_own_override_graphs(runtime, overrides):
    caller_before = copy.deepcopy(overrides)
    agent = runtime(overrides)
    expected = copy.deepcopy(agent.request_overrides)
    agent.request_overrides["extra_body"]["a_only"]["items"].append("poison")
    if overrides:
        agent.request_overrides["extra_body"]["caller"]["items"].append("poison")
    assert overrides == caller_before
    assert agent._primary_runtime["request_overrides"] == expected
    assert runtime(overrides).request_overrides == expected

    agent.switch_model("gpt-4o", "custom:b", api_key="b-test-key",
                       base_url="https://b.example.invalid/v1", api_mode="chat_completions",
                       capabilities={})
    switched = copy.deepcopy(agent.request_overrides)
    assert _request_body(agent) == {"b_only": {"items": ["b"]}}
    agent.request_overrides["extra_body"]["b_only"]["items"].append("mutated")
    assert agent._primary_runtime["request_overrides"] == switched
    agent.request_overrides = copy.deepcopy(switched)
    assert agent._try_activate_fallback()
    assert agent.provider == "custom:c"
    assert agent._restore_primary_runtime()
    assert agent.provider == "custom:b"
    assert agent.request_overrides == switched
    assert runtime().request_overrides == {"extra_body": {"a_only": {"items": ["a"]}}}
