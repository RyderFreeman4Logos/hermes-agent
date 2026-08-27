import json
from pathlib import Path
from typing import Any


def _usage(*, cache_read: int, prompt: int, telemetry: str = "reported"):
    from agent.usage_pricing import CanonicalUsage

    return CanonicalUsage(
        input_tokens=max(0, prompt - cache_read),
        cache_read_tokens=cache_read,
        cache_telemetry=telemetry,  # type: ignore[arg-type]
    )


def _remember_pair(dump, prefix_a: str, prefix_b: str) -> None:
    request = {
        "messages": [{"role": "system", "content": prefix_a}],
        "input": f"INPUT-{prefix_a}",
        "tools": [{"name": f"TOOL-{prefix_a}"}],
        "prompt_cache_key": f"PCK-{prefix_a}",
        "model": "model-safe",
    }
    dump.remember_sent_request(
        request, route="same-route", provider="provider-safe", model="model-safe"
    )
    dump.maybe_dump_on_usage(
        _usage(cache_read=9_500, prompt=10_000),
        route="same-route",
        provider="provider-safe",
        model="model-safe",
        api_mode="chat_completions",
    )
    dump.remember_sent_request(
        {
            **request,
            "messages": [{"role": "system", "content": prefix_b}],
            "input": f"INPUT-{prefix_b}",
            "tools": [{"name": f"TOOL-{prefix_b}"}],
            "prompt_cache_key": f"PCK-{prefix_b}",
        },
        route="same-route",
        provider="provider-safe",
        model="model-safe",
    )


def _finish(
    dump,
    usage,
    *,
    route: str = "same-route",
    provider: str = "provider-safe",
    model: str = "model-safe",
    api_mode: str = "chat_completions",
) -> None:
    dump.maybe_dump_on_usage(
        usage,
        route=route,
        provider=provider,
        model=model,
        api_mode=api_mode,
    )


def _pair_dirs(root: Path) -> list[Path]:
    directory = root / "observability" / "cache_lowhit"
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_dir())


def _dump_files(root: Path) -> list[Path]:
    return [pair / "pair.json" for pair in _pair_dirs(root)]


def _keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key))
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found


def _assert_pair(payload: dict[str, Any], *secrets: str) -> None:
    assert payload["schema"] == "hermes.cache_lowhit_pair.v1"
    requests = payload["requests"]
    assert len(requests) == 2
    assert payload["comparison"]["first_differing_segment"] in {
        "system_or_static_prefix",
        "tools",
        "cache_key_or_scope",
        "later_history",
        "none",
    }
    keys = _keys(payload)
    for forbidden in {"messages", "input", "prompt_cache_key", "headers"}:
        assert forbidden not in keys
    text = json.dumps(payload)
    for secret in secrets:
        assert secret not in text
    for request in requests:
        assert request["structure"]["tool_names"]
        assert request["structure"]["parameter_names"] or "parameters" not in text
        assert request["usage"]["prompt_tokens"] >= 0
        assert request["log_lines"]
        assert "prompt_tokens=" in request["log_lines"][0]
        assert "request_count=" in request["log_lines"][0]
        assert "cache_telemetry=" in request["log_lines"][0]
        for component in request["components"].values():
            assert len(component["digest"]) == 64
            int(component["digest"], 16)


def test_low_hit_writes_sanitized_same_route_pair_directory(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    secret = "fixture-secret-9f7d"
    opaque = "opaque-sentinel-31c4"
    for index, prefix in enumerate(("previous", "current")):
        dump.remember_sent_request(
            {
                "model": "model-safe",
                "messages": [{"role": "system", "content": f"{prefix}-{opaque}"}],
                "input": f"input-{opaque}",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": f"weather-{opaque}",
                            "parameters": {"properties": {f"location-{opaque}": {}}},
                        },
                    }
                ],
                "prompt_cache_key": secret,
                "headers": {"Authorization": secret},
                "extra_body": {"cache_control": {"type": "ephemeral"}},
                "private_url": f"postgres://user:{secret}@host/db",
            },
            api_mode="chat_completions",
            route="same-route",
            provider="provider-safe",
        )
        if index == 0:
            _finish(dump, _usage(cache_read=9_500, prompt=10_000))

    _finish(dump, _usage(cache_read=512, prompt=60_246))

    root = tmp_path / "observability" / "cache_lowhit"
    pairs = _pair_dirs(tmp_path)
    assert len(pairs) == 1
    assert root.stat().st_mode & 0o777 == 0o700
    assert pairs[0].stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in pairs[0].iterdir())
    files = list(pairs[0].iterdir())
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert secret not in text
    assert opaque not in text
    payload = json.loads((pairs[0] / "pair.json").read_text(encoding="utf-8"))
    _assert_pair(payload, secret, opaque)
    assert payload["requests"][0]["structure"]["cache_control"]
    assert "request_dump" not in text
    assert len(dump._BUFFERS[("same-route", "provider-safe", "model-safe", "chat_completions")]) == 1


def test_near_zero_zero_read_dumps_last_two_sanitized_packets(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")

    _finish(dump, _usage(cache_read=0, prompt=10_000))

    files = _dump_files(tmp_path)
    assert len(files) == 1
    _assert_pair(json.loads(files[0].read_text(encoding="utf-8")), "UNREDACTED-PREFIX-A")


def test_near_zero_94_percent_hit_dumps_pair(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "PREFIX-A", "PREFIX-B")

    _finish(dump, _usage(cache_read=9_400, prompt=10_000))

    assert len(_dump_files(tmp_path)) == 1


def test_95_percent_hit_does_not_dump(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "PREFIX-A", "PREFIX-B")

    dump.maybe_dump_on_usage(
        _usage(cache_read=9_500, prompt=10_000),
        route="same-route",
        provider="provider-safe",
        model="model-safe",
        api_mode="chat_completions",
    )

    assert _dump_files(tmp_path) == []


def test_unavailable_telemetry_does_not_dump(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "PREFIX-A", "PREFIX-B")

    _finish(dump, _usage(cache_read=0, prompt=10_000, telemetry="unavailable"))

    assert _dump_files(tmp_path) == []


def test_no_previous_same_route_packet_does_not_dump(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    dump.remember_sent_request({"model": "model-safe"}, route="only-route")

    _finish(dump, _usage(cache_read=0, prompt=10_000))

    assert _dump_files(tmp_path) == []


def test_different_routes_are_not_paired(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    dump.remember_sent_request({"model": "model-safe"}, route="route-a")
    dump.maybe_dump_on_usage(
        _usage(cache_read=9_500, prompt=10_000),
        route="same-route",
        provider="provider-safe",
        model="model-safe",
        api_mode="chat_completions",
    )
    dump.remember_sent_request({"model": "model-safe"}, route="route-b")

    _finish(dump, _usage(cache_read=0, prompt=10_000))

    assert _dump_files(tmp_path) == []


def test_model_identity_matching_is_case_insensitive(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    request = {"model": "MODEL-SAFE", "messages": [{"role": "system", "content": "body"}]}
    dump.remember_sent_request(
        request, route="same-route", provider="provider-safe", model="MODEL-SAFE"
    )
    dump.maybe_dump_on_usage(
        _usage(cache_read=9_500, prompt=10_000),
        route="same-route",
        provider="provider-safe",
        model="model-safe",
        api_mode="chat_completions",
    )
    dump.remember_sent_request(
        request, route="same-route", provider="provider-safe", model="MODEL-SAFE"
    )

    _finish(dump, _usage(cache_read=0, prompt=10_000))

    assert len(_dump_files(tmp_path)) == 1


def test_persistence_errors_are_swallowed(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(dump, "_exclusive_write", lambda path, value: (_ for _ in ()).throw(OSError("blocked")))
    dump.reset_for_tests()
    _remember_pair(dump, "PREFIX-A", "PREFIX-B")

    _finish(dump, _usage(cache_read=0, prompt=10_000))


def test_symlink_output_directory_fails_closed(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    observability = tmp_path / "observability"
    observability.mkdir()
    (observability / "cache_lowhit").symlink_to(outside, target_is_directory=True)
    dump.reset_for_tests()
    _remember_pair(dump, "PREFIX-A", "PREFIX-B")

    _finish(dump, _usage(cache_read=0, prompt=10_000))

    assert list(outside.iterdir()) == []


def test_ancestor_symlink_fails_closed(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "observability").symlink_to(outside, target_is_directory=True)
    dump.reset_for_tests()
    _remember_pair(dump, "PREFIX-A", "PREFIX-B")

    _finish(dump, _usage(cache_read=0, prompt=10_000))

    assert list(outside.iterdir()) == []


def test_existing_artifact_is_never_overwritten(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "PREFIX-A", "PREFIX-B")
    root = tmp_path / "observability" / "cache_lowhit"
    root.mkdir(mode=0o700, parents=True)
    existing = root / "existing"
    existing.mkdir(mode=0o700)
    pair_file = existing / "pair.json"
    pair_file.write_text("keep-me", encoding="utf-8")
    monkeypatch.setattr(dump, "_new_pair_dir", lambda _: existing)

    _finish(dump, _usage(cache_read=0, prompt=10_000))

    assert pair_file.read_text(encoding="utf-8") == "keep-me"


def test_retention_keeps_only_max_pair_directories(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "OLD-0", "NEW-0")
    _finish(dump, _usage(cache_read=0, prompt=1_000))
    first = _pair_dirs(tmp_path)
    assert len(first) == 1
    first_name = first[0].name

    for index in range(1, dump.MAX_DUMPS + 2):
        dump.reset_for_tests()
        _remember_pair(dump, f"OLD-{index}", f"NEW-{index}")
        _finish(dump, _usage(cache_read=0, prompt=1_000))

    pairs = _pair_dirs(tmp_path)
    assert len(pairs) == dump.MAX_DUMPS
    assert first_name not in {path.name for path in pairs}
    joined = "\n".join(path.read_text(encoding="utf-8") for pair in pairs for path in pair.iterdir())
    assert "OLD-0" not in joined
    assert "NEW-0" not in joined
    for path in _dump_files(tmp_path):
        _assert_pair(json.loads(path.read_text(encoding="utf-8")))


def test_retention_does_not_delete_unexpected_files(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "OLD-0", "NEW-0")
    _finish(dump, _usage(cache_read=0, prompt=1_000))
    first = _pair_dirs(tmp_path)[0]
    unexpected = first / "keep.txt"
    unexpected.write_text("keep-me", encoding="utf-8")

    for index in range(1, dump.MAX_DUMPS + 1):
        dump.reset_for_tests()
        _remember_pair(dump, f"OLD-{index}", f"NEW-{index}")
        _finish(dump, _usage(cache_read=0, prompt=1_000))

    assert unexpected.read_text(encoding="utf-8") == "keep-me"


def test_interleaved_out_of_order_completions_pair_by_owning_model(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump
    from agent import usage_pricing

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    common = {"route": "same-route", "provider": "provider-safe", "api_mode": "chat_completions"}
    def send(model: str, prefix: str) -> None:
        dump.remember_sent_request(
            {
                "model": model,
                "messages": [{"role": "system", "content": prefix}],
                "tools": [{"name": f"tool-{prefix}"}],
            },
            model=model,
            **common,
        )

    send("model-a", "A1")
    send("model-b", "B1")

    def complete(model: str, prompt_tokens: int) -> None:
        usage_pricing.normalize_usage(
            {
                "prompt_tokens": prompt_tokens,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens": 1,
            },
            model=model,
            **common,
        )

    complete("model-a", 10_000)
    send("model-a", "A2")
    complete("model-b", 20_000)
    send("model-b", "B2")
    complete("model-a", 10_000)
    complete("model-b", 20_000)

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in _dump_files(tmp_path)]
    assert len(payloads) == 2
    expected_models = {
        dump._label("model-a", dump._SANITIZE_KEY),
        dump._label("model-b", dump._SANITIZE_KEY),
    }
    assert {payload["route"]["model"] for payload in payloads} == expected_models
    for payload in payloads:
        assert {request["route"]["model"] for request in payload["requests"]} == {
            payload["route"]["model"]
        }


def test_normalize_usage_passes_cache_dump_route_metadata(monkeypatch):
    from agent import cache_lowhit_request_dump as dump
    from agent import usage_pricing

    captured: dict[str, Any] = {}

    def capture(usage, **metadata):
        captured["usage"] = usage
        captured.update(metadata)

    monkeypatch.setattr(dump, "maybe_dump_on_usage", capture)
    usage_pricing.normalize_usage(
        {
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens": 1,
        },
        provider="provider-safe",
        api_mode="chat_completions",
        model="model-safe",
        route="same-route",
    )

    assert captured["provider"] == "provider-safe"
    assert captured["route"] == "same-route"
    assert captured["api_mode"] == "chat_completions"
    assert captured["model"] == "model-safe"
    assert captured["usage"].prompt_tokens == 100


def test_auxiliary_interleaving_does_not_finalize_other_api_mode(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump
    from agent import usage_pricing

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()

    def send(api_mode: str, prefix: str) -> None:
        dump.remember_sent_request(
            {
                "model": "same-model",
                "messages": [{"role": "system", "content": prefix}],
            },
            route="same-route",
            provider="same-provider",
            model="same-model",
            api_mode=api_mode,
        )

    for api_mode in ("chat_completions", "anthropic_messages"):
        send(api_mode, f"{api_mode}-previous")
        usage_pricing.normalize_usage(
            {
                "prompt_tokens": 10_000,
                "prompt_tokens_details": {"cached_tokens": 9_500},
                "completion_tokens": 1,
            },
            provider="same-provider",
            model="same-model",
            api_mode=api_mode,
        )
        send(api_mode, f"{api_mode}-current")

    usage_pricing.normalize_usage(
        {
            "prompt_tokens": 10_000,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens": 1,
        },
        provider="same-provider",
        model="same-model",
    )

    assert _dump_files(tmp_path) == []
    assert not dump._BUFFERS[("same-route", "same-provider", "same-model", "chat_completions")][-1]["_terminal"]
    assert not dump._BUFFERS[("same-route", "same-provider", "same-model", "anthropic_messages")][-1]["_terminal"]


def test_out_of_order_same_identity_finalizes_exact_correlation(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    kwargs = {"route": "same-route", "provider": "provider", "model": "model", "api_mode": "chat_completions"}
    dump.remember_sent_request({"messages": [{"role": "system", "content": "a"}]}, correlation="logical-a", attempt_id="attempt-a", **kwargs)
    dump.remember_sent_request({"messages": [{"role": "system", "content": "b"}]}, correlation="logical-b", attempt_id="attempt-b", **kwargs)
    usage = _usage(cache_read=0, prompt=1_000)

    dump.maybe_dump_on_usage(usage, correlation="logical-a", attempt_id="attempt-a", **kwargs)

    history = dump._BUFFERS[("same-route", "provider", "model", "chat_completions")]
    by_owner = {packet["_correlation"]: packet for packet in history}
    assert by_owner["logical-a"]["_terminal"] is True
    assert by_owner["logical-b"]["_terminal"] is False


def test_failed_pending_owner_cannot_become_predecessor(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    kwargs = {"route": "same-route", "provider": "provider", "model": "model", "api_mode": "chat_completions", "correlation": "logical",}
    dump.remember_sent_request({"messages": [{"role": "system", "content": "failed"}]}, **kwargs)
    dump.remember_sent_request({"messages": [{"role": "system", "content": "current"}]}, **kwargs)
    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=1_000), **kwargs)

    assert _dump_files(tmp_path) == []


def test_partial_pair_publication_removes_staging(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    payload = {"route": {}, "requests": [], "comparison": {}, "log_lines": []}
    calls = 0
    original = dump._exclusive_write

    def fail_second(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publication failure")
        return original(path, value)

    monkeypatch.setattr(dump, "_exclusive_write", fail_second)
    try:
        dump._persist_pair((payload, payload))
    except OSError:
        pass
    root = tmp_path / "observability" / "cache_lowhit"
    assert not any(path.is_dir() for path in root.iterdir()) if root.exists() else True


def test_retention_ignores_unsealed_matching_directory(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    root = tmp_path / "observability" / "cache_lowhit"
    root.mkdir(parents=True)
    unowned = root / "pair-unsealed"
    unowned.mkdir()
    (unowned / "pair.json").write_text("{}", encoding="utf-8")
    (unowned / "log_lines.jsonl").write_text("", encoding="utf-8")
    for index in range(dump.MAX_DUMPS):
        owned = root / f"pair-owned-{index}"
        owned.mkdir()
        (owned / dump._PAIR_MARKER).write_text(dump._PAIR_SCHEMA, encoding="utf-8")
        (owned / "pair.json").write_text(json.dumps({"schema": dump._PAIR_SCHEMA}), encoding="utf-8")
        (owned / "log_lines.jsonl").write_text("", encoding="utf-8")
    dump._retain(root)
    assert unowned.exists()


def test_physical_append_uses_process_lock(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    calls = []
    real_flock = diagnostics.fcntl.flock
    monkeypatch.setattr(diagnostics.fcntl, "flock", lambda fd, op: (calls.append(op), real_flock(fd, op))[1])
    diagnostics._append({"schema": "test"})
    assert diagnostics.fcntl.LOCK_EX in calls and diagnostics.fcntl.LOCK_UN in calls


def test_raw_values_are_sanitized_before_physical_boundaries(monkeypatch, tmp_path):
    from agent import physical_attempt_diagnostics as diagnostics
    from hermes_cli import config

    sentinel = "REPAIR-SENTINEL"
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: _config(True))
    seen = []
    monkeypatch.setattr(diagnostics, "_serialized", lambda value: (seen.append(value), b"{}")[1])
    diagnostics._LAST_ATTEMPT.clear()
    diagnostics.start_attempt({"messages": [{"role": "system", "content": sentinel}], "prompt_cache_key": sentinel}, api_mode="chat_completions", route="chat_completions", provider="provider", model="model", retry=0, loop=1, correlation="repair")
    assert sentinel not in repr(seen)
