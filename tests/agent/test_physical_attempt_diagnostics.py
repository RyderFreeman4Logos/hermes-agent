import json
import stat
from types import SimpleNamespace


def test_physical_attempt_diagnostics_are_default_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert (
        DEFAULT_CONFIG["observability"]["physical_attempt_digests"]["enabled"] is False
    )


def test_physical_attempt_sink_is_private_content_free_and_preserves_unknown(
    monkeypatch, tmp_path
):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    sentinel = "PLAINTEXT-ISSUE68-SENTINEL"
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {})

    request = {
        "model": "gpt-test",
        "instructions": sentinel,
        "input": [{"role": "user", "content": sentinel}],
        "tools": [{"type": "function", "name": sentinel}],
        "prompt_cache_key": sentinel,
    }
    assert diagnostics.prepare_cache_scope(sentinel) is None
    assert (
        diagnostics.start_responses_attempt(
            request,
            scope=None,
            route="codex_responses",
            provider="openai",
            model="gpt-test",
            role="main",
            retry=0,
            continuation=0,
        )
        is None
    )
    assert not (tmp_path / "observability").exists()

    monkeypatch.setattr(
        config,
        "read_raw_config_readonly",
        lambda: {"observability": {"physical_attempt_digests": {"enabled": True}}},
    )
    scope = diagnostics.prepare_cache_scope(sentinel)
    first = diagnostics.start_responses_attempt(
        request,
        scope=scope,
        route="codex_responses",
        provider="openai",
        model="gpt-test",
        role="main",
        retry=0,
        continuation=0,
    )
    diagnostics.finish_responses_attempt(
        first,
        usage=SimpleNamespace(input_tokens=11, output_tokens=2),
        outcome="completed",
    )

    continued = {
        **request,
        "input": [*request["input"], {"role": "user", "content": sentinel}],
    }
    second = diagnostics.start_responses_attempt(
        continued,
        scope=scope,
        route="codex_responses",
        provider="openai",
        model="gpt-test",
        role="reviewer",
        retry=1,
        continuation=2,
    )
    diagnostics.finish_responses_attempt(
        second,
        usage=SimpleNamespace(
            input_tokens=13,
            output_tokens=3,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        outcome="completed",
    )

    root = tmp_path / "observability"
    records = [
        json.loads(line)
        for line in (root / "physical_attempt_digests.jsonl").read_text().splitlines()
    ]
    starts = [record for record in records if record["phase"] == "start"]
    terminals = [record for record in records if record["phase"] == "terminal"]

    assert set(starts[0]) == {
        "schema",
        "phase",
        "attempt_digest",
        "monotonic_ns",
        "route",
        "provider",
        "model",
        "role",
        "retry",
        "continuation",
        "scope_digest",
        "scope_bytes",
        "key_digest",
        "key_bytes",
        "prefix_digest",
        "prefix_bytes",
        "tool_digest",
        "tool_bytes",
        "equivalent_digest",
        "equivalent_bytes",
    }
    assert set(terminals[0]) == {
        "schema",
        "phase",
        "attempt_digest",
        "monotonic_ns",
        "outcome",
        "cache_state",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    }
    assert starts[0]["attempt_digest"] == terminals[0]["attempt_digest"]
    assert starts[1]["attempt_digest"] == terminals[1]["attempt_digest"]
    assert starts[0]["equivalent_digest"] == starts[1]["equivalent_digest"]
    assert starts[0]["prefix_digest"] != starts[1]["prefix_digest"]
    assert terminals[0]["cache_state"] == "unknown"
    assert terminals[0]["cache_read_tokens"] is None
    assert terminals[1]["cache_state"] == "miss"
    assert terminals[1]["cache_read_tokens"] == 0
    assert sentinel.encode() not in b"".join(
        path.read_bytes() for path in root.iterdir()
    )
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "physical_attempt_digests.key").stat().st_mode) == 0o600
    assert (
        stat.S_IMODE((root / "physical_attempt_digests.jsonl").stat().st_mode) == 0o600
    )


def test_non_responses_identity_keeps_static_equivalence_across_append_only_tail(
    monkeypatch, tmp_path
):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    sentinel = "PLAINTEXT-ISSUE68-NONRESPONSES"
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        config,
        "read_raw_config_readonly",
        lambda: {"observability": {"physical_attempt_digests": {"enabled": True}}},
    )
    request = {
        "model": "chat-test",
        "messages": [
            {"role": "system", "content": sentinel},
            {"role": "user", "content": sentinel},
        ],
        "tools": [{"type": "function", "function": {"name": sentinel}}],
        "prompt_cache_key": sentinel,
    }

    first = diagnostics.start_attempt(
        request,
        api_mode="chat_completions",
        route="chat_completions",
        provider="openai",
        model="chat-test",
        role="main",
        retry=0,
        continuation=0,
    )
    diagnostics.finish_attempt(
        first,
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=2),
        outcome="completed",
        api_mode="chat_completions",
        provider="openai",
    )
    continued = {
        **request,
        "messages": [
            *request["messages"],
            {"role": "assistant", "content": sentinel},
            {"role": "user", "content": sentinel},
        ],
    }
    second = diagnostics.start_attempt(
        continued,
        api_mode="chat_completions",
        route="chat_completions",
        provider="openai",
        model="chat-test",
        role="reviewer",
        retry=1,
        continuation=1,
    )
    diagnostics.finish_attempt(
        second,
        usage=SimpleNamespace(
            prompt_tokens=17,
            completion_tokens=3,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        outcome="completed",
        api_mode="chat_completions",
        provider="openai",
    )

    records = [
        json.loads(line)
        for line in (
            tmp_path / "observability" / "physical_attempt_digests.jsonl"
        ).read_text().splitlines()
    ]
    starts = [record for record in records if record["phase"] == "start"]
    terminals = [record for record in records if record["phase"] == "terminal"]

    assert starts[0]["equivalent_digest"] == starts[1]["equivalent_digest"]
    assert starts[0]["prefix_digest"] != starts[1]["prefix_digest"]
    assert starts[1]["role"] == "reviewer"
    assert starts[1]["retry"] == 1
    assert starts[1]["continuation"] == 1
    assert terminals[0]["cache_state"] == "unknown"
    assert terminals[1]["cache_state"] == "miss"
    assert sentinel.encode() not in b"".join(
        path.read_bytes() for path in (tmp_path / "observability").iterdir()
    )
