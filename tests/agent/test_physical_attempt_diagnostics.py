"""Exact payload/body pairing for opt-in physical-attempt diagnostics (#174)."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace


def _config(enabled: bool) -> dict[str, object]:
    return {"observability": {"physical_attempt_digests": {"enabled": enabled}}}


def _records(tmp_path):
    return [
        json.loads(line)
        for line in (
            tmp_path / "observability" / "physical_attempt_digests.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]


def _body(record: dict) -> dict:
    return json.loads(base64.b64decode(record["body_bytes"]["data"]))


def test_paired_records_are_default_off(monkeypatch, tmp_path):
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


def test_pair_keeps_payload_and_body_for_final_then_next_first_attempt(
    monkeypatch, tmp_path
):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    sentinel = "ISSUE108-PRIVATE-SENTINEL"
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    timestamps = iter((101, 102, 103, 104))
    monkeypatch.setattr(
        diagnostics,
        "time",
        SimpleNamespace(time_ns=lambda: next(timestamps)),
        raising=False,
    )
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
    records = _records(tmp_path)
    pair = next(record for record in records if record["phase"] == "pair")
    assert [record["timestamp_ns"] for record in records] == [101, 102, 103, 104]
    assert pair["previous_loop"] == 1
    assert pair["current_loop"] == 2
    assert pair["previous_attempt_retry"] == 1
    assert pair["first_differing_segment"] == "prefix"
    current = pair["current"]
    assert current["request"]["messages"][0]["content"] == f"{sentinel}-next"
    assert _body(current) == current["request"]
    serialized = json.dumps(records, sort_keys=True)
    assert f"Bearer {sentinel}" not in serialized
    assert all(
        record["request"]["extra_headers"]["authorization"] == "[REDACTED]"
        for record in records
        if record["phase"] == "attempt" and "extra_headers" in record["request"]
    )


def test_pair_does_not_cross_provider_or_model_routes(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    diagnostics._LAST_ATTEMPT.clear()
    for correlation, later_provider, later_model in (
        ("provider-switch", "fallback", "grok-4.6"),
        ("model-switch", "custom", "other-model"),
    ):
        for loop, provider, model in (
            (1, "custom", "grok-4.6"),
            (2, later_provider, later_model),
        ):
            diagnostics.start_attempt(
                {"messages": [{"role": "system", "content": "fixed"}]},
                api_mode="chat_completions",
                route="chat_completions",
                provider=provider,
                model=model,
                retry=0,
                loop=loop,
                correlation=correlation,
            )
    assert [record["phase"] for record in _records(tmp_path)] == ["attempt"] * 4


def test_pair_classifies_first_changed_tools_segment(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    diagnostics._LAST_ATTEMPT.clear()
    common = {
        "messages": [{"role": "system", "content": "fixed"}],
        "prompt_cache_key": "fixed",
    }
    diagnostics.start_attempt(
        {**common, "tools": [{"name": "old"}]},
        api_mode="chat_completions", route="chat_completions", provider="provider",
        model="model", retry=0, loop=1, correlation="tools",
    )
    diagnostics.start_attempt(
        {**common, "tools": [{"name": "new"}]},
        api_mode="chat_completions", route="chat_completions", provider="provider",
        model="model", retry=0, loop=2, correlation="tools",
    )
    pair = next(record for record in _records(tmp_path) if record["phase"] == "pair")
    assert pair["first_differing_segment"] == "tools"
    assert pair["current"]["request"]["tools"] == [{"name": "new"}]


def test_pair_distinguishes_later_history_from_complete_equality(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    diagnostics._LAST_ATTEMPT.clear()
    common = {
        "messages": [
            {"role": "system", "content": "fixed"},
            {"role": "user", "content": "old"},
        ],
        "prompt_cache_key": "fixed",
        "tools": [{"name": "fixed"}],
    }
    changed = {
        **common,
        "messages": [
            {"role": "system", "content": "fixed"},
            {"role": "user", "content": "nëw"},
        ],
    }
    for loop, request in ((1, common), (2, changed), (3, changed)):
        diagnostics.start_attempt(
            request,
            api_mode="chat_completions",
            route="chat_completions",
            provider="provider",
            model="model",
            retry=0,
            loop=loop,
            correlation="later-history",
        )
    pairs = [record for record in _records(tmp_path) if record["phase"] == "pair"]
    assert [pair["first_differing_segment"] for pair in pairs] == [
        "later_history",
        "none",
    ]
    assert pairs[0]["current"]["request"]["messages"][1]["content"] == "nëw"


def test_enabled_diagnostics_work_without_posix_uid_apis(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    monkeypatch.delattr(diagnostics.os, "geteuid")
    diagnostics._LAST_ATTEMPT.clear()
    attempts = [
        diagnostics.start_attempt(
            {"messages": [{"role": "system", "content": "fixed"}]},
            api_mode="chat_completions",
            route="chat_completions",
            provider="provider",
            model="model",
            retry=0,
            loop=loop,
            correlation="non-posix",
        )
        for loop in (1, 2)
    ]
    assert all(attempt is not None for attempt in attempts)
    assert [record["phase"] for record in _records(tmp_path)] == [
        "attempt",
        "attempt",
        "pair",
    ]


def test_retention_stays_bounded_for_unique_correlations_and_records(
    monkeypatch, tmp_path
):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    monkeypatch.setattr(diagnostics, "_MAX_TRACKED_CORRELATIONS", 2, raising=False)
    monkeypatch.setattr(diagnostics, "_MAX_RECORDS_BYTES", 1024, raising=False)
    diagnostics._LAST_ATTEMPT.clear()
    for index in range(8):
        diagnostics.start_attempt(
            {"messages": [{"role": "system", "content": "fixed"}]},
            api_mode="chat_completions",
            route="chat_completions",
            provider="provider",
            model="model",
            retry=0,
            loop=1,
            correlation=f"correlation-{index}",
        )
    records_path = tmp_path / "observability" / "physical_attempt_digests.jsonl"
    assert len(diagnostics._LAST_ATTEMPT) == 2
    assert records_path.stat().st_size <= 1024


def test_persisted_labels_redact_url_encoded_and_credential_shaped_values(
    monkeypatch, tmp_path
):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    sentinel = "ISSUE170-PRIVATE-LABEL"
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    diagnostics._LAST_ATTEMPT.clear()
    labels = (
        (f"https://provider.example/v1/{sentinel}", f"provider-{sentinel}", f"model-{sentinel}"),
        (f"route%3A%2F%2F{sentinel}", f"Bearer-{sentinel}", f"sk-proj-{sentinel}"),
    )
    for loop, (route, provider, model) in enumerate(labels, start=1):
        diagnostics.start_attempt(
            {"messages": []},
            api_mode="chat_completions",
            route=route,
            provider=provider,
            model=model,
            retry=0,
            loop=loop,
            correlation="label-privacy",
        )
    records = _records(tmp_path)
    serialized = json.dumps(records, sort_keys=True)
    assert sentinel not in serialized
    assert sentinel not in repr(diagnostics._LAST_ATTEMPT)
