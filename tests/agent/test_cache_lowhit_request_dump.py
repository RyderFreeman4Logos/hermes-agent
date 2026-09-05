"""Send-time last-2 exact request/body dump on economically near-zero cache hits."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.usage_pricing import normalize_usage


@pytest.fixture(autouse=True)
def _enable_cache_request_capture(monkeypatch):
    from agent import cache_request_capture as capture

    monkeypatch.setattr(capture, "_settings", lambda: {"enabled": True})


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


def _assert_exact_pairs(payload: dict[str, Any]) -> None:
    assert payload["schema"] == "hermes.cache_lowhit.v2"
    assert "cache_read_tokens" in payload
    assert "prompt_tokens" in payload
    requests = payload["requests"]
    assert len(requests) == 2
    for request in requests:
        fingerprint = request.get("fingerprint")
        sizes = request.get("sizes")
        assert isinstance(fingerprint, str) and len(fingerprint) == 64
        int(fingerprint, 16)
        assert sizes
        assert request.get("model") == "grok-4.6"
        assert request["api_mode"] == "chat_completions"
        saved = request["request"]
        body = base64.b64decode(request["body_bytes"]["data"])
        assert json.loads(body) == saved


def test_default_off_does_not_retain_or_dump(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump
    from agent import cache_request_capture as capture

    monkeypatch.setattr(capture, "_settings", lambda: {"enabled": False})
    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    dump.remember_sent_request({"model": "gpt-test", "messages": []})

    assert list(dump._LAST) == []
    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=10_000))
    assert _dump_files(tmp_path) == []


def test_disabling_capture_revokes_retained_snapshot(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump
    from agent import cache_request_capture as capture

    settings = {"enabled": True}
    monkeypatch.setattr(capture, "_settings", lambda: settings)
    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    dump.remember_sent_request({"model": "gpt-test", "messages": []})
    assert list(dump._LAST)

    settings["enabled"] = False
    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=10_000))

    assert list(dump._LAST) == []
    assert _dump_files(tmp_path) == []


def test_opt_in_dump_retains_redacted_payload_and_exact_body(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump
    from agent import cache_request_capture as capture

    monkeypatch.setattr(capture, "_settings", lambda: {"enabled": True})
    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    request = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "keep-this"}],
        "headers": {"Authorization": "Bearer ACTUAL-SECRET"},
        "endpoint": "https://user:password@api.example.test/v1",
    }
    dump.remember_sent_request(request, api_mode="chat_completions")
    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=10_000))

    payload = json.loads(_dump_files(tmp_path)[0].read_text(encoding="utf-8"))
    assert payload["schema"] == "hermes.cache_lowhit.v2"
    saved = payload["requests"][0]
    assert saved["request"]["headers"]["Authorization"] == "[REDACTED]"
    body = base64.b64decode(saved["body_bytes"]["data"])
    assert json.loads(body) == saved["request"]
    assert b"ACTUAL-SECRET" not in body
    assert b"password@" not in body


def test_near_zero_zero_read_dumps_last_two_exact_pairs(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")

    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=10_000))

    files = _dump_files(tmp_path)
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    _assert_exact_pairs(payload)


def test_near_zero_sub_percent_read_dumps_last_two_exact_pairs(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")

    dump.maybe_dump_on_usage(_usage(cache_read=512, prompt=60_246))

    payload = json.loads(_dump_files(tmp_path)[0].read_text(encoding="utf-8"))
    _assert_exact_pairs(payload)


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
        _assert_exact_pairs(json.loads(path.read_text(encoding="utf-8")))


def test_dump_module_is_loaded_from_this_checkout() -> None:
    from agent import cache_lowhit_request_dump as dump

    assert Path(dump.__file__).is_relative_to(Path(__file__).parents[2])


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


def test_current_wrappers_do_not_double_remember(monkeypatch) -> None:
    from agent import cache_lowhit_request_dump as dump
    from agent import relay_llm, relay_runtime

    remembered: list[tuple[dict[str, Any], str]] = []
    monkeypatch.setattr(
        dump,
        "remember_sent_request",
        lambda request, *, api_mode: remembered.append((request, api_mode)),
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

    request = {"model": "gpt-test", "messages": [], "api_mode_marker": "sync"}
    relay_llm.execute_current(
        request,
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
