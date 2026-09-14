"""Tests for read_raw_config_readonly() — the no-deepcopy raw config read.

The readonly variant exists for per-turn policy checks (e.g. the
shared-metrics gate) that were paying a full config deepcopy on every call.
Contract under test:

1. identity invariant — repeat calls return the SAME cached object,
   including across the very first (cache-miss) call;
2. freshness — an edited config.yaml (mtime/size change) is picked up;
3. parity — content equals read_raw_config()'s result;
4. missing/broken config degrades to {} exactly like read_raw_config().
"""

import os
import time
from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def isolated_hermes_home():
    """Per-test HERMES_HOME dir (already redirected by the autouse conftest
    fixture) as a Path, with the raw-config cache cleared around the test."""
    from pathlib import Path

    import hermes_cli.config as config_mod

    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    config_mod._RAW_CONFIG_CACHE.clear()
    yield home
    config_mod._RAW_CONFIG_CACHE.clear()


def _write_config(home, data):
    cfg = home / "config.yaml"
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    return cfg




def test_freshness_after_config_edit(isolated_hermes_home):
    from hermes_cli.config import read_raw_config_readonly

    cfg = _write_config(isolated_hermes_home, {"display": {"ephemeral_system_ttl": 1}})
    first = read_raw_config_readonly()
    assert first["display"]["ephemeral_system_ttl"] == 1

    _write_config(isolated_hermes_home, {"display": {"ephemeral_system_ttl": 7}})
    # Force a distinct mtime_ns even on coarse-timestamp filesystems.
    st = cfg.stat()
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    second = read_raw_config_readonly()
    assert second["display"]["ephemeral_system_ttl"] == 7


def test_missing_config_returns_empty(isolated_hermes_home):
    from hermes_cli.config import read_raw_config_readonly

    cfg = isolated_hermes_home / "config.yaml"
    if cfg.exists():
        cfg.unlink()
    assert read_raw_config_readonly() == {}


def test_warm_nonmissing_read_failure_keeps_explicit_selection(
    isolated_hermes_home, monkeypatch
):
    """A known explicit provider must not become an implicit selection on read failure."""
    from hermes_cli import config as config_mod
    from tools.tool_backend_helpers import read_selection

    cfg = _write_config(isolated_hermes_home, {"image_gen": {"provider": "nous"}})
    assert read_selection("image_gen") == "nous"

    def deny_read(self, *args, **kwargs):
        if self == cfg:
            raise PermissionError("configured file became unreadable")
        return original_open(self, *args, **kwargs)

    original_open = Path.open
    original_read_bytes = Path.read_bytes

    def deny_read_bytes(self):
        if self == cfg:
            raise PermissionError("configured file became unreadable")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", deny_read_bytes)
    monkeypatch.setattr(Path, "open", deny_read)

    assert config_mod.read_raw_config_readonly()["image_gen"]["provider"] == "nous"
    assert read_selection("image_gen") == "nous"


def test_warm_read_hashes_bounded_chunks_outside_config_lock(
    isolated_hermes_home, monkeypatch
):
    """Warm freshness hashing must not retain whole comment-heavy files under the lock."""
    from hermes_cli import config as config_mod

    cfg = isolated_hermes_home / "config.yaml"
    cfg.write_text("display: {}\n" + "# comment\n" * 20000, encoding="utf-8")
    first = config_mod.read_raw_config_readonly()
    reads = []
    original_open = Path.open

    class TrackedReader:
        def __init__(self, fileobj):
            self._fileobj = fileobj

        def read(self, size=-1):
            reads.append((size, config_mod._CONFIG_LOCK._is_owned()))
            return self._fileobj.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._fileobj.close()

        def __getattr__(self, name):
            return getattr(self._fileobj, name)

    def track_open(self, *args, **kwargs):
        fileobj = original_open(self, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        return TrackedReader(fileobj) if self == cfg and mode == "rb" else fileobj

    monkeypatch.setattr(Path, "open", track_open)
    assert config_mod.read_raw_config_readonly() is first
    assert reads
    assert all(0 < size <= 64 * 1024 and not locked
               for size, locked in reads)
