import json
import stat


def _sentinel():
    return "PROMPT-BODY-COOKIE-AUTH-RAW-ID"


def test_physical_attempt_digests_are_default_off(tmp_path):
    from agent.attempt_digests import PhysicalAttemptDigestSink

    sink = PhysicalAttemptDigestSink(tmp_path, enabled=False)
    assert sink.start(
        route="codex_responses",
        provider="openai",
        model="gpt-test",
        role="primary",
        retry=0,
        continuation="unknown",
        cache_scope=_sentinel(),
        cache_key=_sentinel(),
        tools=[{"name": _sentinel()}],
        instructions=_sentinel(),
        wire_prefix={"input": _sentinel()},
    ) is None
    assert not (tmp_path / "observability").exists()


def test_physical_attempt_digests_are_hmac_only_and_pair_terminal_usage(tmp_path):
    from agent.attempt_digests import PhysicalAttemptDigestSink

    sink = PhysicalAttemptDigestSink(tmp_path, enabled=True)
    attempt = sink.start(
        route="codex_responses",
        provider="openai",
        model="gpt-test",
        role="reviewer",
        retry=1,
        continuation="unknown",
        cache_scope=_sentinel(),
        cache_key=_sentinel(),
        tools=[{"name": _sentinel()}],
        instructions=_sentinel(),
        wire_prefix={"input": _sentinel()},
    )
    sink.finish(attempt, {"prompt_tokens": 9, "cache_read_tokens": 0})

    key = tmp_path / "observability" / "physical_attempt_digests.key"
    records = (tmp_path / "observability" / "physical_attempt_digests.jsonl").read_text().splitlines()
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert len(records) == 2
    payload = "\n".join(records)
    assert _sentinel() not in payload
    start, terminal = (json.loads(line) for line in records)
    assert start["event"] == "start"
    assert start["role"] == "reviewer"
    assert start["retry"] == 1
    assert start["digests"]["wire_prefix"]["value"].startswith("hmac-sha256:")
    assert start["digests"]["static_instructions"]["value"].startswith("hmac-sha256:")
    assert start["digests"]["wire_prefix"]["bytes"] > 0
    assert terminal["event"] == "terminal"
    assert terminal["attempt"] == start["attempt"]
    assert terminal["usage"] == {
        "input_tokens": 9,
        "cache_read_tokens": 0,
        "cache_write_tokens": "unknown",
    }


def test_physical_attempt_digest_config_is_explicitly_default_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["observability"]["physical_attempt_digests"]["enabled"] is False


def test_codex_transport_carries_final_wire_identity_for_the_dispatch_seam():
    from agent.transports.codex import ResponsesApiTransport

    sentinel = _sentinel()
    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-test",
        messages=[{"role": "user", "content": sentinel}],
        tools=[{"type": "function", "function": {"name": "tool", "parameters": {}}}],
        session_id="scope-1",
    )

    identity = kwargs["_hermes_physical_attempt_identity"]
    assert identity["cache_scope"] == "scope-1"
    assert identity["cache_key"] == kwargs["prompt_cache_key"]
    assert identity["instructions"] == kwargs["instructions"]
    assert identity["tools"] == kwargs["tools"]
    assert identity["wire_prefix"] == kwargs["input"]
