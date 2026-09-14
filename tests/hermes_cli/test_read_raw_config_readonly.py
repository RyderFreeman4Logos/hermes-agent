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


def test_validated_raw_recovery_retires_same_metadata_failure(isolated_hermes_home):
    """A warm digest hit cannot keep an old corrupt-config refusal alive."""
    from hermes_cli import config as config_mod

    cfg = isolated_hermes_home / "config.yaml"
    valid = "model:\n  provider: openrouter\n"
    broken = "model:\n  provider: [openrouter"
    assert len(valid) == len(broken)
    cfg.write_text(valid, encoding="utf-8")
    original_stat = cfg.stat()
    assert config_mod.read_raw_config_readonly()["model"]["provider"] == "openrouter"

    cfg.write_text(broken, encoding="utf-8")
    os.utime(cfg, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    config_mod.load_config()
    assert config_mod.get_active_config_parse_failure()

    cfg.write_text(valid, encoding="utf-8")
    os.utime(cfg, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert config_mod.read_raw_config_readonly()["model"]["provider"] == "openrouter"
    assert config_mod.get_active_config_parse_failure() is None


def test_second_read_oserror_keeps_warm_raw_value(isolated_hermes_home, monkeypatch):
    """The parser's same-read pass has the same last-good policy as hashing."""
    from hermes_cli import config as config_mod

    cfg = _write_config(isolated_hermes_home, {"image_gen": {"provider": "nous"}})
    assert config_mod.read_raw_config_readonly()["image_gen"]["provider"] == "nous"
    cfg.write_text("image_gen:\n  provider: krea\n", encoding="utf-8")
    original_open = Path.open
    target_opens = 0

    def fail_parser_open(self, *args, **kwargs):
        nonlocal target_opens
        if self == cfg and (args[0] if args else kwargs.get("mode")) == "rb":
            target_opens += 1
            if target_opens == 2:
                raise PermissionError("second raw read denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_parser_open)
    assert config_mod.read_raw_config_readonly()["image_gen"]["provider"] == "nous"
    assert target_opens == 2


def test_second_read_failure_keeps_nous_selection_out_of_direct_fal_sink(isolated_hermes_home, monkeypatch):
    """A second raw read failure cannot turn a stored managed pick into FAL_KEY direct submit."""
    import importlib
    from unittest.mock import MagicMock
    from tools import image_generation_tool
    from hermes_cli import config as config_mod

    tool = importlib.reload(image_generation_tool)
    cfg = _write_config(isolated_hermes_home, {"image_gen": {"provider": "nous"}})
    assert config_mod.read_raw_config_readonly()["image_gen"]["provider"] == "nous"
    cfg.write_text("image_gen:\n  provider: krea\n", encoding="utf-8")
    original_open = Path.open
    opens = 0
    def fail_second(self, *args, **kwargs):
        nonlocal opens
        if self == cfg and (args[0] if args else kwargs.get("mode")) == "rb":
            opens += 1
            if opens == 2:
                raise PermissionError("second parser read denied")
        return original_open(self, *args, **kwargs)
    direct = MagicMock()
    monkeypatch.setattr(Path, "open", fail_second)
    monkeypatch.setattr(tool, "fal_client", direct)
    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setattr(tool, "resolve_managed_tool_gateway", lambda _name: None)
    with pytest.raises(ValueError, match="Nous"):
        tool._submit_fal_request("fal-ai/test", {"prompt": "x"})
    assert opens == 2
    direct.submit.assert_not_called()

@pytest.mark.parametrize("reader_name", ["read_raw_config", "read_raw_config_readonly"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("fault_phase", [1, 2], ids=["digest", "parser"])
@pytest.mark.parametrize(
    ("fault_name", "fault_type"),
    [("missing", FileNotFoundError), ("denied", PermissionError)],
)
def test_raw_reader_phase_fault_outcome_matrix(
    isolated_hermes_home,
    monkeypatch,
    reader_name,
    warm,
    fault_phase,
    fault_name,
    fault_type,
):
    """W5: both public readers keep their distinct missing and last-good contracts.

    ``_digest_file`` and the YAML parser intentionally open the same real file
    separately.  This wrapper forwards every other operation and faults the
    first ``read`` of exactly one counted ``rb`` handle, so the assertion is
    about the public raw APIs rather than a mocked reader implementation.
    """
    from hermes_cli import config as config_mod

    cfg = _write_config(isolated_hermes_home, {"image_gen": {"provider": "nous"}})
    reader = getattr(config_mod, reader_name)
    last_good = {"image_gen": {"provider": "nous"}}
    if warm:
        assert reader() == last_good
        _write_config(isolated_hermes_home, {"image_gen": {"provider": "krea"}})

    original_open = Path.open
    target_opens = 0
    fault_reads = 0

    class PhaseFaultReader:
        def __init__(self, source):
            self._source = source

        def read(self, size=-1):
            nonlocal fault_reads
            fault_reads += 1
            if fault_reads == 1:
                raise fault_type(f"{fault_name} phase-{fault_phase} read")
            return self._source.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._source.close()

        def __getattr__(self, name):
            return getattr(self._source, name)

    def phase_fault_open(self, *args, **kwargs):
        nonlocal target_opens
        source = original_open(self, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == cfg and mode == "rb":
            target_opens += 1
            if target_opens == fault_phase:
                return PhaseFaultReader(source)
        return source

    monkeypatch.setattr(Path, "open", phase_fault_open)
    result = reader()

    assert target_opens == fault_phase
    assert fault_reads == 1
    if fault_type is FileNotFoundError or not warm:
        assert result == {}
    else:
        assert result == last_good


@pytest.mark.parametrize("reader_name", ["read_raw_config", "read_raw_config_readonly"])
def test_malformed_raw_yaml_does_not_admit_a_digest_or_replace_last_good(
    isolated_hermes_home, reader_name
):
    """W5: a malformed revision is empty, while the prior valid mapping survives."""
    from hermes_cli import config as config_mod

    cfg = isolated_hermes_home / "config.yaml"
    valid = "image_gen:\n  provider: nous\n"
    malformed = "image_gen: [nous\n"
    cfg.write_text(valid, encoding="utf-8")
    reader = getattr(config_mod, reader_name)
    first = reader()
    assert first == {"image_gen": {"provider": "nous"}}

    cfg.write_text(malformed, encoding="utf-8")
    assert reader() == {}

    cfg.write_text(valid, encoding="utf-8")
    restored = reader()
    assert restored == first
    if reader_name == "read_raw_config_readonly":
        assert restored is first


@pytest.mark.parametrize("reader_name", ["read_raw_config", "read_raw_config_readonly"])
def test_nonmapping_raw_yaml_cannot_retire_an_outstanding_parse_refusal(
    isolated_hermes_home, monkeypatch, reader_name
):
    """W5: normalizing a valid non-mapping root must preserve the real refusal."""
    from hermes_cli import config as config_mod
    from hermes_cli.auth import AuthError, resolve_provider

    cfg = isolated_hermes_home / "config.yaml"
    broken = "model: [openrouter"
    nonmapping = "[]".ljust(len(broken))
    cfg.write_text(broken, encoding="utf-8")
    broken_stat = cfg.stat()
    config_mod.load_config()
    assert config_mod.get_active_config_parse_failure()

    cfg.write_text(nonmapping, encoding="utf-8")
    os.utime(cfg, ns=(broken_stat.st_atime_ns, broken_stat.st_mtime_ns))
    assert getattr(config_mod, reader_name)() == {}
    assert config_mod.get_active_config_parse_failure()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
    with pytest.raises(AuthError, match="corrupt"):
        resolve_provider("auto")
