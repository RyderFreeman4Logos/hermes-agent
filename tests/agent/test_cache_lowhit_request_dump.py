"""Send-time last-2 fingerprint dump on economically near-zero cache hits."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.usage_pricing import normalize_usage


_FORBIDDEN_KEYS = (
    "prefix",
    "messages",
    "input",
    "tools",
    "prompt_cache_key",
    "later_history",
)


def _usage(*, cache_read: int, prompt: int, telemetry: str = "reported") -> Any:
    return SimpleNamespace(
        cache_read_tokens=cache_read,
        prompt_tokens=prompt,
        cache_telemetry=telemetry,  # type: ignore[arg-type]
    )


def _remember_pair(dump, prefix_a: str, prefix_b: str) -> None:
    dump.remember_sent_request(
        {
            "messages": [{"role": "system", "content": prefix_a}],
            "input": f"INPUT-{prefix_a}",
            "tools": [{"name": f"TOOL-{prefix_a}"}],
            "prompt_cache_key": f"PCK-{prefix_a}",
            "model": "grok-4.6",
        }
    )
    dump.remember_sent_request(
        {
            "messages": [{"role": "system", "content": prefix_b}],
            "input": f"INPUT-{prefix_b}",
            "tools": [{"name": f"TOOL-{prefix_b}"}],
            "prompt_cache_key": f"PCK-{prefix_b}",
            "model": "grok-4.6",
        }
    )


def _dump_files(root: Path) -> list[Path]:
    directory = root / "observability" / "cache_lowhit"
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix == ".json")


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


def _assert_fingerprint_only(payload: dict[str, Any], *secrets: str) -> None:
    assert "cache_read_tokens" in payload
    assert "prompt_tokens" in payload
    requests = payload["requests"]
    assert len(requests) == 2
    keys = _keys(payload)
    for forbidden in _FORBIDDEN_KEYS:
        assert forbidden not in keys
    text = json.dumps(payload)
    for secret in secrets:
        assert secret not in text
    for request in requests:
        fingerprint = request.get("fingerprint")
        sizes = request.get("sizes")
        assert isinstance(fingerprint, str) and len(fingerprint) == 64
        int(fingerprint, 16)
        assert sizes
        assert request.get("model") == "grok-4.6"


def test_near_zero_zero_read_dumps_last_two_fingerprints(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")

    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=10_000))

    files = _dump_files(tmp_path)
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    _assert_fingerprint_only(
        payload,
        "UNREDACTED-PREFIX-A",
        "UNREDACTED-PREFIX-B",
        "INPUT-UNREDACTED-PREFIX-A",
        "TOOL-UNREDACTED-PREFIX-A",
        "PCK-UNREDACTED-PREFIX-A",
    )


def test_near_zero_sub_percent_read_dumps_last_two_fingerprints(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")

    dump.maybe_dump_on_usage(_usage(cache_read=512, prompt=60_246))

    payload = json.loads(_dump_files(tmp_path)[0].read_text(encoding="utf-8"))
    _assert_fingerprint_only(payload, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")


def test_high_hit_does_not_dump(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")

    dump.maybe_dump_on_usage(_usage(cache_read=9_000, prompt=10_000))

    assert _dump_files(tmp_path) == []


def test_unavailable_telemetry_does_not_dump(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")

    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=10_000, telemetry="unavailable"))

    assert _dump_files(tmp_path) == []


def test_retention_overwrites_oldest(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "OLD-0", "NEW-0")
    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=1_000))
    first = _dump_files(tmp_path)
    assert len(first) == 1
    first_name = first[0].name

    for index in range(1, dump.MAX_DUMPS + 2):
        dump.reset_for_tests()
        _remember_pair(dump, f"OLD-{index}", f"NEW-{index}")
        dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=1_000))

    files = _dump_files(tmp_path)
    assert len(files) == dump.MAX_DUMPS
    assert first_name not in {path.name for path in files}
    joined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "OLD-0" not in joined
    assert "NEW-0" not in joined
    for path in files:
        _assert_fingerprint_only(json.loads(path.read_text(encoding="utf-8")))


def test_dump_module_is_loaded_from_this_checkout() -> None:
    from agent import cache_lowhit_request_dump as dump

    assert Path(dump.__file__).is_relative_to(Path(__file__).parents[2])


def test_normalize_usage_canonical_telemetry_drives_dump(monkeypatch, tmp_path) -> None:
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")

    usage = normalize_usage(
        SimpleNamespace(
            prompt_tokens=10_000,
            completion_tokens=100,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        provider="openai",
        api_mode="chat_completions",
    )

    assert usage.cache_telemetry == "reported"
    files = _dump_files(tmp_path)
    assert len(files) == 1
    _assert_fingerprint_only(json.loads(files[0].read_text(encoding="utf-8")))


def test_normalize_usage_without_cache_fields_marks_telemetry_unavailable() -> None:
    usage = normalize_usage(
        SimpleNamespace(prompt_tokens=10_000, completion_tokens=100),
        provider="openai",
        api_mode="chat_completions",
    )

    assert usage.cache_telemetry == "unavailable"


def test_relay_send_paths_remember_exact_request_and_api_mode(monkeypatch) -> None:
    from agent import cache_lowhit_request_dump as dump
    from agent import relay_llm

    remembered: list[tuple[dict[str, Any], str]] = []
    monkeypatch.setattr(
        dump,
        "remember_sent_request",
        lambda request, *, api_mode: remembered.append((request, api_mode)),
    )
    for api_mode in ("chat_completions", "codex_responses", "anthropic_messages"):
        request = {
            "model": "gpt-test",
            "messages": [{"role": "system", "content": "exact-send"}],
            "api_mode_marker": api_mode,
        }
        metadata = {"api_mode": api_mode}
        observed: list[dict[str, Any]] = []

        relay_llm.execute(
            request,
            lambda sent: observed.append(sent) or {"ok": True},
            session_id="no-live-session",
            name="openai",
            model_name="gpt-test",
            metadata=metadata,
        )

        async def async_provider(sent: dict[str, Any]) -> dict[str, Any]:
            observed.append(sent)
            return {"ok": True}

        asyncio.run(
            relay_llm.execute_async(
                request,
                async_provider,
                session_id="no-live-session",
                name="openai",
                model_name="gpt-test",
                metadata=metadata,
            )
        )

        stream = relay_llm.stream_current(
            request,
            lambda sent: observed.append(sent) or iter(({"ok": True},)),
            name="openai",
            model_name="gpt-test",
            finalizer=dict,
            metadata=metadata,
        )
        assert list(stream) == [{"ok": True}]
        assert observed == [request, request, request]

    assert [(request["api_mode_marker"], mode) for request, mode in remembered] == [
        (api_mode, api_mode)
        for api_mode in ("chat_completions", "codex_responses", "anthropic_messages")
        for _ in range(3)
    ]
