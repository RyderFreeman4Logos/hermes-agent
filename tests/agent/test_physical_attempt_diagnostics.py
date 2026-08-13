import json

import pytest


def _config(enabled: bool) -> dict:
    return {"observability": {"physical_attempt_digests": {"enabled": enabled}}}


def test_paired_digests_are_default_off(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(False))

    assert diagnostics.start_attempt(
        {"messages": []},
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        role="main",
        retry=0,
        continuation=0,
    ) is None
    assert not (tmp_path / "observability").exists()


def test_paired_digests_report_first_difference_without_request_content(
    monkeypatch, tmp_path
):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    sentinel = "ISSUE108-PRIVATE-SENTINEL"
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    first = diagnostics.start_attempt(
        {
            "messages": [{"role": "system", "content": sentinel}],
            "instructions": sentinel,
            "input": [{"role": "user", "content": sentinel}],
            "tools": [{"type": "function", "function": {"name": sentinel}}],
            "prompt_cache_key": sentinel,
            "extra_headers": {"authorization": f"Bearer {sentinel}"},
        },
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        role="main",
        retry=0,
        continuation=0,
        loop=1,
        correlation="test",
    )
    diagnostics.start_attempt(
        {
            "messages": [{"role": "system", "content": f"{sentinel}-old-static"}],
            "tools": [{"type": "function", "function": {"name": f"{sentinel}-old-tool"}}],
            "prompt_cache_key": sentinel,
        },
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        role="main",
        retry=1,
        continuation=0,
        loop=1,
        correlation="test",
    )
    diagnostics.start_attempt(
        {
            "messages": [{"role": "system", "content": sentinel}],
            "instructions": sentinel,
            "input": [{"role": "user", "content": sentinel}],
            "tools": [{"type": "function", "function": {"name": f"{sentinel}-last-loop"}}],
            "prompt_cache_key": sentinel,
        },
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        role="main",
        retry=2,
        continuation=0,
        loop=1,
        correlation="test",
    )
    diagnostics.start_attempt(
        {
            "messages": [{"role": "system", "content": sentinel}],
            "instructions": sentinel,
            "input": [{"role": "user", "content": sentinel}],
            "tools": [{"type": "function", "function": {"name": f"{sentinel}-next"}}],
            "prompt_cache_key": sentinel,
        },
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        role="main",
        retry=0,
        continuation=0,
        loop=2,
        correlation="test",
    )
    diagnostics.finish_attempt(
        first,
        usage=None,
        outcome="completed",
        api_mode="chat_completions",
        provider="provider",
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "observability" / "physical_attempt_digests.jsonl").read_text().splitlines()
    ]
    pair = next(record for record in records if record["phase"] == "pair")
    assert pair["first_differing_class"] == "tools"
    assert sentinel.encode() not in json.dumps(records, sort_keys=True).encode()


def test_relay_digest_sees_final_rewritten_kwargs(monkeypatch, tmp_path):
    pytest.importorskip("nemo_relay")
    from agent import physical_attempt_diagnostics as diagnostics
    from agent import relay_llm, relay_runtime
    from hermes_cli import config

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    recorded = []
    original_start = diagnostics.start_attempt

    def record_start(request, **kwargs):
        recorded.append(request)
        return original_start(request, **kwargs)

    monkeypatch.setattr(diagnostics, "start_attempt", record_start)
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="issue108", platform="cli"
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="turn", task_id="task")
    lease.host.retain_managed_execution("issue108")
    relay = lease.host.relay

    def rewrite(_name, request, annotated):
        return relay.LLMRequestInterceptOutcome(
            relay.LLMRequest(request.headers, {**request.content, "messages": [{"role": "system", "content": "rewritten"}]}),
            annotated,
        )

    relay.intercepts.register_llm_request("issue108", 1, False, rewrite)
    try:
        response = relay_llm.execute(
            {"model": "model", "messages": [{"role": "system", "content": "original"}]},
            lambda request: {"usage": {}, "received": request},
            session_id="issue108",
            name="provider",
            model_name="model",
            metadata={"api_mode": "custom", "api_request_id": "turn:api:1", "call_role": "primary"},
        )
    finally:
        relay.intercepts.deregister_llm_request("issue108")
        lease.host.release_managed_execution("issue108")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()

    assert response["received"]["messages"][0]["content"] == "rewritten"
    assert recorded[0]["messages"][0]["content"] == "rewritten"
    assert all(not key.startswith("_hermes_") for key in response["received"])
