"""Tests for config-driven allocator trimming."""

from unittest.mock import Mock

import pytest

import hermes_cli.mem_trim as mem_trim


@pytest.fixture(autouse=True)
def _reset_trim_state(monkeypatch):
    monkeypatch.setattr(mem_trim, "_last_trim_monotonic", 0.0)
    monkeypatch.setattr(mem_trim, "_probe_done", True)
    monkeypatch.setattr(mem_trim, "_malloc_trim", None)
    monkeypatch.setattr(mem_trim, "_trim_call_count", 0)


def test_unsupported_allocator_is_noop_without_gc(monkeypatch):
    collect = Mock()
    monkeypatch.setattr(mem_trim.gc, "collect", collect)

    assert mem_trim.trim_memory(force=True, reason="test") is False
    collect.assert_not_called()


def test_disabled_config_is_noop_in_isolated_hermes_home(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "context:\n  memory_trim:\n    enabled: false\n",
        encoding="utf-8",
    )
    trim = Mock(return_value=1)
    collect = Mock()
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(mem_trim.gc, "collect", collect)
    token = set_hermes_home_override(hermes_home)
    try:
        assert mem_trim.trim_memory(force=True) is False
    finally:
        reset_hermes_home_override(token)

    collect.assert_not_called()
    trim.assert_not_called()


def test_default_config_declares_memory_trim_controls():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["context"]["memory_trim"] == {
        "enabled": True,
        "cooldown_seconds": 60.0,
        "log_every_n": 1,
        "info_log_min_delta_mb": 0.0,
    }


def test_gc_runs_before_malloc_trim(monkeypatch):
    calls = []
    monkeypatch.setattr(mem_trim.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(
        mem_trim, "_malloc_trim", lambda pad: calls.append(("trim", pad)) or 1
    )
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)

    assert mem_trim.trim_memory(reason="turn", cooldown_seconds=60) is True
    assert calls == ["gc", ("trim", 0)]


def test_cooldown_suppresses_collection_but_force_bypasses_it(monkeypatch):
    collect = Mock()
    trim = Mock(return_value=1)
    monkeypatch.setattr(mem_trim.gc, "collect", collect)
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(mem_trim, "_last_trim_monotonic", 95.0)
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)

    assert mem_trim.trim_memory(cooldown_seconds=60) is False
    collect.assert_not_called()
    trim.assert_not_called()
    assert mem_trim.trim_memory(force=True, cooldown_seconds=60) is True


def test_config_cooldown_controls_rate_limit(monkeypatch):
    trim = Mock(return_value=1)
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(mem_trim, "_last_trim_monotonic", 1.0)
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "context": {
                "memory_trim": {"enabled": True, "cooldown_seconds": 120.0}
            }
        },
    )

    assert mem_trim.trim_memory() is False
    trim.assert_not_called()


def test_force_logs_even_when_sampling_and_delta_would_skip(monkeypatch, caplog):
    monkeypatch.setattr(mem_trim.gc, "collect", lambda: None)
    monkeypatch.setattr(mem_trim, "_malloc_trim", lambda _pad: 1)
    monkeypatch.setattr(mem_trim, "_config_settings", lambda: (True, 0.0, 99, 1.0))
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        mem_trim,
        "collect_memory_snapshot",
        lambda: {"rss_kib": 4096, "rss_anon_kib": 3072, "thread_count": 3},
    )

    with caplog.at_level("INFO", logger="hermes_cli.mem_trim"):
        assert mem_trim.trim_memory(reason="periodic") is True
        assert mem_trim.trim_memory(force=True, reason="close") is True

    messages = [record.getMessage() for record in caplog.records]
    assert not any("reason=periodic" in message for message in messages)
    assert any("reason=close" in message for message in messages)


def test_memory_snapshot_parses_linux_telemetry(monkeypatch):
    monkeypatch.setattr(mem_trim.sys, "platform", "linux")
    monkeypatch.setattr(
        mem_trim,
        "_read_proc_status",
        lambda: "Name:\tpython\nVmRSS:\t1234 kB\nRssAnon:\t567 kB\n",
    )
    monkeypatch.setattr(mem_trim.threading, "active_count", lambda: 9)

    assert mem_trim.collect_memory_snapshot(history_bytes=42) == {
        "rss_kib": 1234,
        "rss_anon_kib": 567,
        "thread_count": 9,
        "history_bytes": 42,
    }


def test_failed_trim_is_fail_open_and_rate_limited(monkeypatch):
    trim = Mock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(mem_trim.time, "monotonic", lambda: 100.0)

    assert mem_trim.trim_memory(reason="test", cooldown_seconds=60) is False
    assert mem_trim.trim_memory(cooldown_seconds=60) is False
    assert trim.call_count == 1
