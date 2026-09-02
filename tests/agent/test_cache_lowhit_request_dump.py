"""Exact payload/body low-hit dump behavior ported from #205."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.usage_pricing import CanonicalUsage, normalize_usage


def _usage(*, cache_read: int, prompt: int, telemetry: str = "reported") -> Any:
    return SimpleNamespace(
        cache_read_tokens=cache_read,
        prompt_tokens=prompt,
        cache_telemetry=telemetry,  # type: ignore[arg-type]
    )


def _enable(monkeypatch) -> None:
    from agent import cache_request_capture as capture

    monkeypatch.setattr(capture, "_settings", lambda: {"enabled": True})


def _request(marker: str) -> dict[str, Any]:
    return {
        "model": "grok-4.6",
        "messages": [{"role": "system", "content": marker}],
        "input": f"input-{marker}",
        "tools": [{"name": f"tool-{marker}"}],
        "prompt_cache_key": f"pck-{marker}",
        "headers": {"Authorization": f"Bearer {marker}"},
    }


def _remember_pair(dump, prefix_a: str, prefix_b: str) -> None:
    dump.remember_sent_request(_request(prefix_a), api_mode="chat_completions")
    dump.remember_sent_request(_request(prefix_b), api_mode="chat_completions")


def _files(root: Path) -> list[Path]:
    directory = root / "observability" / "cache_lowhit"
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def _assert_exact_pair(payload: dict[str, Any], *markers: str) -> None:
    requests = payload["requests"]
    assert len(requests) == 2
    text = json.dumps(payload)
    for marker in markers:
        assert f"Bearer {marker}" not in text
    for saved, marker in zip(requests, markers, strict=True):
        assert saved["request"]["headers"]["Authorization"] == "[REDACTED]"
        assert saved["request"]["messages"][0]["content"] == marker
        assert saved["request"]["input"] == f"input-{marker}"
        assert saved["request"]["tools"][0]["name"] == f"tool-{marker}"
        assert saved["request"]["prompt_cache_key"] == f"pck-{marker}"
        assert saved["api_mode"] == "chat_completions"
        body = base64.b64decode(saved["body_bytes"]["data"])
        assert json.loads(body) == saved["request"]
        assert f"Bearer {marker}".encode() not in body


def test_near_zero_dump_contains_redacted_payload_and_exact_body_bytes(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    _enable(monkeypatch)
    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    dump.remember_sent_request(_request("hello"), api_mode="chat_completions")
    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=10_000))

    payload = json.loads(_files(tmp_path)[0].read_text(encoding="utf-8"))
    saved = payload["requests"][0]
    assert saved["request"]["headers"]["Authorization"] == "[REDACTED]"
    body = base64.b64decode(saved["body_bytes"]["data"])
    assert json.loads(body) == saved["request"]
    assert b"Bearer hello" not in body


def test_near_zero_zero_read_dumps_last_two_payloads(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    _enable(monkeypatch)
    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "PREFIX-A", "PREFIX-B")
    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=10_000))
    payload = json.loads(_files(tmp_path)[0].read_text(encoding="utf-8"))
    _assert_exact_pair(payload, "PREFIX-A", "PREFIX-B")


def test_near_zero_sub_percent_read_dumps_last_two_payloads(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    _enable(monkeypatch)
    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "PREFIX-A", "PREFIX-B")
    dump.maybe_dump_on_usage(_usage(cache_read=512, prompt=60_246))
    payload = json.loads(_files(tmp_path)[0].read_text(encoding="utf-8"))
    _assert_exact_pair(payload, "PREFIX-A", "PREFIX-B")


def test_default_off_does_not_retain_or_dump(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump
    from agent import cache_request_capture as capture

    monkeypatch.setattr(capture, "_settings", lambda: {"enabled": False})
    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    dump.remember_sent_request(_request("default-off"))

    assert list(dump._LAST) == []
    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=10_000))
    assert _files(tmp_path) == []


def test_high_hit_and_unavailable_telemetry_do_not_dump(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    _enable(monkeypatch)
    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    dump.remember_sent_request(_request("secret"))
    for usage in (
        _usage(cache_read=9_000, prompt=10_000),
        _usage(cache_read=0, prompt=10_000, telemetry="unavailable"),
    ):
        dump.maybe_dump_on_usage(usage)
    assert _files(tmp_path) == []


def test_retention_overwrites_oldest(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    _enable(monkeypatch)
    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "OLD-0", "NEW-0")
    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=1_000))
    first_name = _files(tmp_path)[0].name
    for index in range(1, dump.MAX_DUMPS + 2):
        dump.reset_for_tests()
        _remember_pair(dump, f"OLD-{index}", f"NEW-{index}")
        dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=1_000))
    files = _files(tmp_path)
    assert len(files) == dump.MAX_DUMPS
    assert first_name not in {path.name for path in files}
    joined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "OLD-0" not in joined
    assert "NEW-0" not in joined


def test_dump_module_is_loaded_from_this_checkout() -> None:
    from agent import cache_lowhit_request_dump as dump

    assert Path(dump.__file__).is_relative_to(Path(__file__).parents[2])


def test_normalize_usage_canonical_telemetry_drives_dump(monkeypatch, tmp_path) -> None:
    from agent import cache_lowhit_request_dump as dump

    _enable(monkeypatch)
    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "PREFIX-A", "PREFIX-B")
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
    payload = json.loads(_files(tmp_path)[0].read_text(encoding="utf-8"))
    _assert_exact_pair(payload, "PREFIX-A", "PREFIX-B")


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
        lambda request, *, api_mode, **_kwargs: remembered.append((request, api_mode)),
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


def test_current_wrappers_do_not_double_remember(monkeypatch) -> None:
    from agent import cache_lowhit_request_dump as dump
    from agent import relay_llm, relay_runtime

    remembered: list[tuple[dict[str, Any], str]] = []
    monkeypatch.setattr(
        dump,
        "remember_sent_request",
        lambda request, *, api_mode, **_kwargs: remembered.append((request, api_mode)),
    )
    monkeypatch.setattr(
        relay_runtime,
        "active_turn",
        lambda _session_id=None: SimpleNamespace(
            lease=SimpleNamespace(
                host=object(), session=None, session_id="no-live-session"
            ),
            handle=None,
            relay_enabled=True,
            closed=False,
        ),
    )
    relay_llm.execute_current(
        {"model": "gpt-test", "messages": [], "api_mode_marker": "sync"},
        lambda sent: {"ok": sent["api_mode_marker"]},
        name="openai",
        model_name="gpt-test",
        metadata={"api_mode": "chat_completions"},
    )

    async def async_provider(sent: dict[str, Any]) -> dict[str, Any]:
        return {"ok": sent["api_mode_marker"]}

    asyncio.run(
        relay_llm.execute_current_async(
            {"model": "gpt-test", "messages": [], "api_mode_marker": "async"},
            async_provider,
            name="openai",
            model_name="gpt-test",
            metadata={"api_mode": "codex_responses"},
        )
    )
    stream = relay_llm.stream_current(
        {"model": "gpt-test", "messages": [], "api_mode_marker": "stream"},
        lambda _sent: iter(({"ok": True},)),
        name="openai",
        model_name="gpt-test",
        finalizer=dict,
        metadata={"api_mode": "anthropic_messages"},
    )
    assert list(stream) == [{"ok": True}]
    assert [(request["api_mode_marker"], mode) for request, mode in remembered] == [
        ("sync", "chat_completions"),
        ("async", "codex_responses"),
        ("stream", "anthropic_messages"),
    ]
