"""Privacy contract for opt-in paired physical-attempt diagnostics (#108)."""

from __future__ import annotations

import base64
import json
import multiprocessing
import threading
from multiprocessing.synchronize import Barrier, Event
from pathlib import Path
from types import SimpleNamespace


def _concurrent_capped_append(
    root: str, start: Event, size_barrier: Barrier,
) -> None:
    from agent import physical_attempt_diagnostics as diagnostics

    diagnostics.get_hermes_home = lambda: Path(root)
    diagnostics._MAX_RECORDS_BYTES = 1024
    original_fstat = diagnostics.os.fstat
    calls = 0

    def synchronize_pre_write_size(fd: int):
        nonlocal calls
        calls += 1
        info = original_fstat(fd)
        if calls == 2:
            try:
                size_barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return info

    diagnostics.os.fstat = synchronize_pre_write_size
    start.wait(5)
    diagnostics._append({"value": "x" * 900})


def _config(enabled: bool) -> dict[str, object]:
    return {"observability": {"physical_attempt_digests": {"enabled": enabled}}}


def test_opt_in_attempt_records_include_redacted_payload_and_body(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    diagnostics._LAST_ATTEMPT.clear()
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "keep-this"}],
        "headers": {"Authorization": "Bearer ACTUAL-SECRET"},
        "endpoint": "https://user:password@api.example.test/v1",
    }
    diagnostics.start_attempt(
        request,
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        retry=0,
        loop=1,
        correlation="payload",
    )
    diagnostics.start_attempt(
        {**request, "messages": [{"role": "user", "content": "keep-this-2"}]},
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        retry=0,
        loop=2,
        correlation="payload",
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "observability" / "physical_attempt_digests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    pair = next(record for record in records if record["phase"] == "pair")
    saved = pair["current"]
    assert pair["schema"] == "hermes.physical_attempt.v2"
    assert saved["request"]["headers"]["Authorization"] == "[REDACTED]"
    body = base64.b64decode(saved["body_bytes"]["data"])
    assert json.loads(body) == saved["request"]
    assert b"ACTUAL-SECRET" not in body
    assert b"password@" not in body


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


def test_disabling_diagnostics_revokes_retained_snapshot(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    settings = [True]
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(settings[0]))
    diagnostics._LAST_ATTEMPT.clear()
    diagnostics.start_attempt(
        {"model": "model", "messages": []},
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        retry=0,
        loop=1,
        correlation="revoke",
    )
    records_path = tmp_path / "observability" / "physical_attempt_digests.jsonl"
    records_before_disable = records_path.read_text(encoding="utf-8")
    assert diagnostics._LAST_ATTEMPT

    settings[0] = False
    assert diagnostics.start_attempt(
        {"model": "model", "messages": []},
        api_mode="chat_completions",
        route="chat_completions",
        provider="provider",
        model="model",
        retry=0,
        loop=2,
        correlation="revoke",
    ) is None

    assert diagnostics._LAST_ATTEMPT == {}
    assert records_path.read_text(encoding="utf-8") == records_before_disable


def test_pair_emits_digests_and_redacted_payloads_for_final_then_next_first_attempt(
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
    assert [record["timestamp_ns"] for record in records] == [101, 102, 103, 104]
    assert pair["route"] == "chat_completions"
    assert len(pair["provider"]) == len(pair["model"]) == 64
    assert all(
        character in "0123456789abcdef"
        for character in pair["provider"] + pair["model"]
    )
    assert pair["previous_loop"] == 1
    assert pair["current_loop"] == 2
    assert pair["previous_attempt_retry"] == 1
    assert set(pair["digests"]) == {"cache_scope", "later_history", "prefix", "tools"}
    assert all(len(value) == 64 for value in pair["digests"].values())
    assert pair["first_differing_segment"] == "prefix"
    assert pair["equal"] == {
        "cache_scope": False,
        "later_history": True,
        "prefix": False,
        "tools": False,
    }
    assert pair["byte_lengths"] == {
        "prefix": len(json.dumps(
            [{"role": "system", "content": f"{sentinel}-next"}],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")),
        "tools": len(json.dumps(
            [{"type": "function", "function": {"name": f"{sentinel}-next"}}],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")),
        "cache_scope": len(json.dumps(
            {"scope": scope["digest"], "key": f"{sentinel}-next"},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")),
        "later_history": len(json.dumps(
            [], ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")),
    }
    assert all(
        set(record["byte_lengths"]) == {"cache_scope", "later_history", "prefix", "tools"}
        for record in records
    )
    serialized = json.dumps(records, sort_keys=True)
    assert f"Bearer {sentinel}" not in serialized
    assert '"cookies": "[REDACTED]"' in serialized
    assert '"authorization": "[REDACTED]"' in serialized
    assert sentinel + "-old" in serialized
    assert sentinel + "-next" in serialized
    assert '"messages"' in serialized


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

    records = [
        json.loads(line)
        for line in (
            tmp_path / "observability" / "physical_attempt_digests.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["phase"] for record in records] == ["attempt"] * 4


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

    records = [
        json.loads(line)
        for line in (
            tmp_path / "observability" / "physical_attempt_digests.jsonl"
        ).read_text().splitlines()
    ]
    pair = next(record for record in records if record["phase"] == "pair")
    assert pair["first_differing_segment"] == "tools"
    assert pair["equal"] == {
        "cache_scope": True,
        "later_history": True,
        "prefix": True,
        "tools": False,
    }


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
    changed_history = {
        **common,
        "messages": [
            {"role": "system", "content": "fixed"},
            {"role": "user", "content": "nëw"},
        ],
    }
    for loop, request in ((1, common), (2, changed_history), (3, changed_history)):
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

    records = [
        json.loads(line)
        for line in (
            tmp_path / "observability" / "physical_attempt_digests.jsonl"
        ).read_text().splitlines()
    ]
    pairs = [record for record in records if record["phase"] == "pair"]
    assert [pair["first_differing_segment"] for pair in pairs] == [
        "later_history",
        "none",
    ]
    assert pairs[0]["equal"] == {
        "cache_scope": True,
        "later_history": False,
        "prefix": True,
        "tools": True,
    }
    assert pairs[1]["equal"] == {
        "cache_scope": True,
        "later_history": True,
        "prefix": True,
        "tools": True,
    }
    assert pairs[0]["byte_lengths"]["later_history"] == len(json.dumps(
        changed_history["messages"][1:], ensure_ascii=False, separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8"))
    assert all(len(pair["digests"]["later_history"]) == 64 for pair in pairs)


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

    records = [
        json.loads(line)
        for line in (
            tmp_path / "observability" / "physical_attempt_digests.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["phase"] for record in records] == ["attempt", "attempt", "pair"]


def test_retention_stays_bounded_across_processes(tmp_path):
    (tmp_path / "observability").mkdir(mode=0o700)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    size_barrier = context.Barrier(8, timeout=2)
    processes = [
        context.Process(
            target=_concurrent_capped_append,
            args=(str(tmp_path), start, size_barrier),
        )
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0

    records_path = tmp_path / "observability" / "physical_attempt_digests.jsonl"
    assert records_path.stat().st_size <= 1024


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
    assert all(key[1] == "chat_completions" for key in diagnostics._LAST_ATTEMPT)
    assert all(
        len(key[2]) == len(key[3]) == 64
        for key in diagnostics._LAST_ATTEMPT
    )
    assert all(character in "0123456789abcdef" for key in diagnostics._LAST_ATTEMPT for character in "".join(key[2:]))
    assert records_path.stat().st_size <= 1024
    assert all(json.loads(line)["phase"] == "attempt" for line in records_path.read_text().splitlines())


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

    records = [
        json.loads(line)
        for line in (
            tmp_path / "observability" / "physical_attempt_digests.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    serialized = json.dumps(records, sort_keys=True)
    assert sentinel not in serialized
    assert sentinel not in repr(diagnostics._LAST_ATTEMPT)
    for record, raw_labels in zip(records, labels):
        assert all(record[field] != raw for field, raw in zip(("route", "provider", "model"), raw_labels))
