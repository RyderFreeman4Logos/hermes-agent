"""Privacy contract for opt-in paired physical-attempt diagnostics (#108)."""

from __future__ import annotations

import json


def _config(enabled: bool) -> dict[str, object]:
    return {"observability": {"physical_attempt_digests": {"enabled": enabled}}}


def test_paired_digests_are_default_off(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(False))

    assert DEFAULT_CONFIG["observability"]["physical_attempt_digests"]["enabled"] is False
    assert diagnostics.start_attempt(
        {"messages": []},
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        retry=0,
        loop=1,
        correlation="test",
    ) is None
    assert not (tmp_path / "observability").exists()


def test_pair_emits_only_hmac_digests_for_final_then_next_first_attempt(
    monkeypatch, tmp_path
):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    sentinel = "ISSUE108-PRIVATE-SENTINEL"
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    diagnostics._LAST_ATTEMPT.clear()

    scope = diagnostics.prepare_cache_scope({"private_scope": sentinel})
    diagnostics.start_attempt(
        {
            "messages": [{"role": "system", "content": f"{sentinel}-old"}],
            "tools": [{"type": "function", "function": {"name": f"{sentinel}-old"}}],
            "prompt_cache_key": f"{sentinel}-old",
            "extra_headers": {"authorization": f"Bearer {sentinel}"},
            "cookies": {"session": sentinel},
        },
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        retry=0,
        loop=1,
        correlation="test",
        scope=scope,
    )
    diagnostics.start_attempt(
        {
            "messages": [{"role": "system", "content": f"{sentinel}-final"}],
            "tools": [{"type": "function", "function": {"name": f"{sentinel}-final"}}],
            "prompt_cache_key": f"{sentinel}-final",
        },
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        retry=1,
        loop=1,
        correlation="test",
        scope=scope,
    )
    diagnostics.start_attempt(
        {
            "messages": [{"role": "system", "content": f"{sentinel}-next"}],
            "tools": [{"type": "function", "function": {"name": f"{sentinel}-next"}}],
            "prompt_cache_key": f"{sentinel}-next",
            "extra_headers": {"referer": f"https://private.example/{sentinel}"},
        },
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        retry=0,
        loop=2,
        correlation="test",
        scope=scope,
    )

    records = [
        json.loads(line)
        for line in (
            tmp_path / "observability" / "physical_attempt_digests.jsonl"
        ).read_text().splitlines()
    ]
    pair = next(record for record in records if record["phase"] == "pair")
    assert pair["previous_loop"] == 1
    assert pair["current_loop"] == 2
    assert pair["previous_attempt_retry"] == 1
    assert set(pair["digests"]) == {"cache_scope", "prefix", "tools"}
    assert all(len(value) == 64 for value in pair["digests"].values())
    assert sentinel not in json.dumps(records, sort_keys=True)
    assert all(
        forbidden not in json.dumps(records, sort_keys=True)
        for forbidden in ("messages", "extra_headers", "authorization", "cookies", "private.example")
    )
