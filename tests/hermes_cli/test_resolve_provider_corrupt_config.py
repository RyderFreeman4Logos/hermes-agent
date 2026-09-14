"""Regression tests for issue #81952 (silent OpenRouter paid-default class fix).

When ``~/.hermes/config.yaml`` EXISTS but fails to parse, ``load_config()``
falls back to ``DEFAULT_CONFIG`` — so ``resolve_provider('auto')``'s tier-2
config check finds no ``model.provider`` and the tier-3 env sniff
(OPENROUTER_API_KEY / OPENAI_API_KEY) or tier-4 pool probe silently adopts the
PAID openrouter provider, even though the user's real (broken) config may name
a completely different provider (e.g. ``openai-codex``). Real money, zero
consent.

The fix: ``hermes_cli.config`` records active parse failures
(``get_active_config_parse_failure``), and ``resolve_provider`` refuses
env/pool auto-adoption with ``AuthError(code='corrupt_config')`` while the
active config is corrupt. Explicit provider requests and valid-config
env-sniff flows are untouched, and fixing the file in place clears the block.
"""

import logging
import os
import threading
import uuid

import pytest


@pytest.fixture(autouse=True)
def _clean_inference_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "NOUS_API_KEY",
        "HERMES_INFERENCE_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


CORRUPT_YAML = "model:\n  provider: openai-codex\n  default: gpt-5.5\n broken: [unterminated\n"
VALID_YAML = "gateway:\n  enabled: false\n"


def _setup_home(tmp_path, monkeypatch, config_text):
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg = home / "config.yaml"
    cfg.write_text(config_text)
    return home, cfg


def _load_config_fresh():
    """Call load_config so a corrupt file goes through the warn/record funnel."""
    from hermes_cli.config import load_config

    return load_config()


class TestParseFailureProbe:
    def test_probe_reports_active_corrupt_config(self, tmp_path, monkeypatch):
        _home, _cfg = _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        _load_config_fresh()

        from hermes_cli.config import get_active_config_parse_failure

        err = get_active_config_parse_failure()
        assert err, "expected an active parse failure to be reported"

    def test_probe_clears_when_file_fixed_in_place(self, tmp_path, monkeypatch):
        _home, cfg = _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        _load_config_fresh()

        from hermes_cli.config import get_active_config_parse_failure

        assert get_active_config_parse_failure()
        cfg.write_text(VALID_YAML)  # user fixes the YAML — different size/mtime
        assert get_active_config_parse_failure() is None

    def test_probe_none_for_valid_config(self, tmp_path, monkeypatch):
        _setup_home(tmp_path, monkeypatch, VALID_YAML)
        _load_config_fresh()

        from hermes_cli.config import get_active_config_parse_failure

        assert get_active_config_parse_failure() is None


class TestResolveProviderCorruptConfig:
    def test_corrupt_config_blocks_env_sniff_adoption(self, tmp_path, monkeypatch):
        """Corrupt config + OPENROUTER_API_KEY must NOT resolve to openrouter."""
        _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
        _load_config_fresh()

        from hermes_cli.auth import AuthError, resolve_provider

        with pytest.raises(AuthError) as excinfo:
            resolve_provider("auto")
        assert excinfo.value.code == "corrupt_config"
        assert "config.yaml" in str(excinfo.value)

    def test_corrupt_config_blocks_pool_probe_adoption(self, tmp_path, monkeypatch):
        """Corrupt config + pool-only credential must NOT resolve to openrouter."""
        _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        _load_config_fresh()

        from agent.credential_pool import (
            AUTH_TYPE_API_KEY,
            SOURCE_MANUAL,
            PooledCredential,
            load_pool,
        )

        pool = load_pool("openrouter")
        pool.add_entry(
            PooledCredential(
                provider="openrouter",
                id=uuid.uuid4().hex[:6],
                label="api-key-1",
                auth_type=AUTH_TYPE_API_KEY,
                priority=0,
                source=SOURCE_MANUAL,
                access_token="sk-or-FAKEKEY123",
                base_url="https://openrouter.ai/api/v1",
            )
        )

        from hermes_cli.auth import AuthError, resolve_provider

        with pytest.raises(AuthError) as excinfo:
            resolve_provider("auto")
        assert excinfo.value.code == "corrupt_config"

    def test_valid_config_env_sniff_keep_path_unbroken(self, tmp_path, monkeypatch):
        """KEEP: valid config that names no provider + env key still resolves."""
        _setup_home(tmp_path, monkeypatch, VALID_YAML)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
        _load_config_fresh()

        from hermes_cli.auth import resolve_provider

        assert resolve_provider("auto") == "openrouter"

    def test_fixed_config_clears_block(self, tmp_path, monkeypatch):
        """Rewriting the corrupt file valid clears the refusal immediately."""
        _home, cfg = _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
        _load_config_fresh()

        from hermes_cli.auth import AuthError, resolve_provider

        with pytest.raises(AuthError):
            resolve_provider("auto")

        cfg.write_text(VALID_YAML)
        assert resolve_provider("auto") == "openrouter"

    def test_explicit_provider_request_untouched(self, tmp_path, monkeypatch):
        """Explicit user intent (requested != auto) resolves even with corrupt config."""
        _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
        _load_config_fresh()

        from hermes_cli.auth import resolve_provider

        assert resolve_provider("openrouter") == "openrouter"


def test_same_metadata_failure_recovery_and_repeat_warning_require_raw_validation(tmp_path, monkeypatch, caplog):
    """W12: only validated bytes retire a failure, and a later identical failure warns again."""
    from hermes_cli import config as config_mod
    from hermes_cli.auth import AuthError, resolve_provider
    _home, cfg = _setup_home(tmp_path, monkeypatch, "gateway:\n  enabled: false\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
    original = cfg.read_text()
    broken = "gateway:\n  enabled: [false"
    assert len(original) == len(broken)
    config_mod.read_raw_config_readonly()
    caplog.set_level(logging.WARNING, logger="hermes_cli.config")
    cfg.write_text(broken)
    config_mod.load_config()
    failed_stat = cfg.stat()
    with pytest.raises(AuthError, match="corrupt"):
        resolve_provider("auto")

    cfg.write_text(original)
    os.utime(cfg, ns=(failed_stat.st_atime_ns, failed_stat.st_mtime_ns))
    with pytest.raises(AuthError, match="corrupt"):
        resolve_provider("auto")
    config_mod.read_raw_config_readonly()
    assert resolve_provider("auto") == "openrouter"

    cfg.write_text(broken)
    os.utime(cfg, ns=(failed_stat.st_atime_ns, failed_stat.st_mtime_ns))
    assert config_mod.read_raw_config_readonly() == {}
    with pytest.raises(AuthError, match="corrupt"):
        resolve_provider("auto")
    assert sum("Failed to parse" in record.message for record in caplog.records) == 2


def test_later_corrupt_reader_survives_an_earlier_stale_valid_reader(tmp_path, monkeypatch):
    """W12: a later corrupt observation cannot be retired by stale valid bytes."""
    from hermes_cli import config as config_mod
    from hermes_cli.auth import AuthError, resolve_provider
    from pathlib import Path

    _home, cfg = _setup_home(tmp_path, monkeypatch, VALID_YAML)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
    assert config_mod.read_raw_config_readonly() == {"gateway": {"enabled": False}}
    valid = cfg.read_text(encoding="utf-8")
    broken = "gateway:\n  enabled: [false"
    assert len(valid) == len(broken)

    def replace_config(text):
        replacement = cfg.with_suffix(".next")
        replacement.write_text(text, encoding="utf-8")
        replacement.replace(cfg)

    first_digest_read = threading.Event()
    release_first_digest = threading.Event()
    first_parser_open = threading.Event()
    release_first_parser = threading.Event()
    original_open = Path.open
    first_opens = 0
    first_result = {}

    class ForwardingBarrierReader:
        def __init__(self, source, phase):
            self._source = source
            self._phase = phase
            self._first_read = True

        def read(self, size=-1):
            if self._phase == "parser" and self._first_read:
                first_parser_open.set()
                assert release_first_parser.wait(5)
            value = self._source.read(size)
            if self._phase == "digest" and self._first_read and value:
                first_digest_read.set()
                assert release_first_digest.wait(5)
            self._first_read = False
            return value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._source.close()

        def __getattr__(self, name):
            return getattr(self._source, name)

    def barrier_open(self, *args, **kwargs):
        nonlocal first_opens
        source = original_open(self, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == cfg and mode == "rb" and threading.current_thread().name == "first-reader":
            first_opens += 1
            return ForwardingBarrierReader(source, "digest" if first_opens == 1 else "parser")
        return source

    monkeypatch.setattr(Path, "open", barrier_open)

    first = threading.Thread(
        target=lambda: first_result.setdefault("value", config_mod.read_raw_config_readonly()),
        name="first-reader",
    )
    first.start()
    assert first_digest_read.wait(5)

    replace_config(broken)
    later = config_mod.read_raw_config_readonly()
    later_stat = cfg.stat()
    assert later == {}
    assert config_mod.get_active_config_parse_failure()

    replace_config(valid)
    release_first_digest.set()
    assert first_parser_open.wait(5)
    replace_config(broken)
    os.utime(cfg, ns=(later_stat.st_atime_ns, later_stat.st_mtime_ns))
    release_first_parser.set()
    first.join(timeout=5)
    assert not first.is_alive()
    assert first_result["value"] == {"gateway": {"enabled": False}}

    assert config_mod.get_active_config_parse_failure()
    with pytest.raises(AuthError) as excinfo:
        resolve_provider("auto")
    assert excinfo.value.code == "corrupt_config"
