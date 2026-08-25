"""Send-time last-2 dump on economically near-zero cache hits (#190)."""

from __future__ import annotations

import json
from pathlib import Path

from agent.usage_pricing import CanonicalUsage


def _usage(*, cache_read: int, prompt: int, telemetry: str = "reported") -> CanonicalUsage:
    return CanonicalUsage(
        input_tokens=max(0, prompt - cache_read),
        cache_read_tokens=cache_read,
        cache_telemetry=telemetry,  # type: ignore[arg-type]
    )


def _remember_pair(dump, prefix_a: str, prefix_b: str) -> None:
    dump.remember_sent_request(
        {"messages": [{"role": "system", "content": prefix_a}], "model": "grok-4.6"}
    )
    dump.remember_sent_request(
        {"messages": [{"role": "system", "content": prefix_b}], "model": "grok-4.6"}
    )


def _dump_files(root: Path) -> list[Path]:
    directory = root / "observability" / "cache_lowhit"
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix == ".json")


def test_near_zero_zero_read_dumps_last_two_unredacted_prefixes(monkeypatch, tmp_path):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")

    dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=10_000))

    files = _dump_files(tmp_path)
    assert len(files) == 1
    payload = files[0].read_text(encoding="utf-8")
    assert "UNREDACTED-PREFIX-A" in payload
    assert "UNREDACTED-PREFIX-B" in payload
    assert json.loads(payload)["requests"][0]["prefix"] == [
        {"role": "system", "content": "UNREDACTED-PREFIX-A"}
    ]
    assert json.loads(payload)["requests"][1]["prefix"] == [
        {"role": "system", "content": "UNREDACTED-PREFIX-B"}
    ]


def test_near_zero_sub_percent_read_dumps_last_two_unredacted_prefixes(
    monkeypatch, tmp_path
):
    from agent import cache_lowhit_request_dump as dump

    monkeypatch.setattr(dump, "get_hermes_home", lambda: tmp_path)
    dump.reset_for_tests()
    _remember_pair(dump, "UNREDACTED-PREFIX-A", "UNREDACTED-PREFIX-B")

    dump.maybe_dump_on_usage(_usage(cache_read=512, prompt=60_246))

    payload = _dump_files(tmp_path)[0].read_text(encoding="utf-8")
    assert "UNREDACTED-PREFIX-A" in payload
    assert "UNREDACTED-PREFIX-B" in payload


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

    for index in range(dump.MAX_DUMPS + 2):
        dump.reset_for_tests()
        _remember_pair(dump, f"OLD-{index}", f"NEW-{index}")
        dump.maybe_dump_on_usage(_usage(cache_read=0, prompt=1_000))

    files = _dump_files(tmp_path)
    assert len(files) == dump.MAX_DUMPS
    joined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "OLD-0" not in joined
    assert f"NEW-{dump.MAX_DUMPS + 1}" in joined
