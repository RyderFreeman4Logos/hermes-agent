"""config.yaml backups: one dir, deduped, bounded — never a pile of siblings in HERMES_HOME."""
import errno
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import hermes_cli.config_backups as config_backups
from hermes_cli.config_backups import backup_config, list_config_backups


def test_repeat_backups_dedupe_and_rotate(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    stamps = iter(f"2026010100000{i}" for i in range(10))
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))

    first = backup_config(cfg, "pre-setup", keep=2)
    assert first is not None and first.parent == tmp_path / "backups" / "config"
    original_mtime_ns = cfg.stat().st_mtime_ns
    assert first.stat().st_mtime_ns == original_mtime_ns
    # Same bytes again → no new file (the hermes-setup-three-times case).
    assert backup_config(cfg, "pre-setup", keep=2) is None
    for i in range(3):
        cfg.write_text(f"model: {i}\n")
        os.utime(cfg, ns=(original_mtime_ns, original_mtime_ns))
        assert cfg.stat().st_mtime_ns == original_mtime_ns
        backup_config(cfg, "pre-setup", keep=2)
    kept = list_config_backups(cfg, "pre-setup")
    assert len(kept) == 2 and kept[0].read_text() == "model: 2\n"
    # Nothing left beside config.yaml in the home root.
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["config.yaml"]


def test_legacy_siblings_move_but_user_named_copies_stay(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    for name in ("config.yaml.bak.1778718391", "config.yaml.corrupt.20260729-093706.bak",
                 "config.yaml.bak-pre-migrate-xai-20260515-120000"):
        (tmp_path / name).write_text("old")
    (tmp_path / "config.yaml.bak-my-note").write_text("mine")

    backup_config(cfg, "corrupt")

    root = tmp_path / "backups" / "config"
    assert (root / "config.yaml.bak.1778718391").exists()
    assert (root / "config.yaml.corrupt.20260729-093706.bak").exists()
    assert (tmp_path / "config.yaml.bak-my-note").read_text() == "mine"
    assert not list(tmp_path.glob("config.yaml.bak.*")) and not list(tmp_path.glob("config.yaml.corrupt.*"))


def _seed_backup(config_path: Path, reason: str, contents: bytes) -> Path:
    root = config_path.parent / "backups" / "config"
    root.mkdir(parents=True, exist_ok=True)
    prior = root / f"{config_path.name}.{reason}.20260101-000000"
    prior.write_bytes(contents)
    return prior


@contextmanager
def _bounded_fifo_call(seconds: float = 2.0):
    def _timeout(_signum, _frame):
        raise TimeoutError("backup comparison blocked on a FIFO")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize(
    "case",
    ["unreadable-unequal", "backup-directory", "backup-fifo", "source-fifo", "missing-backup", "missing-source"],
)
def test_backup_rechecks_metadata_before_comparing(tmp_path: Path, monkeypatch, case: str):
    config_path = tmp_path / "config.yaml"
    current = b"current configuration\n"
    config_path.write_bytes(current)
    prior = _seed_backup(config_path, "race", b"old\n")
    before = set(prior.parent.iterdir())
    reached = False
    fifo = tmp_path / "comparison.fifo"
    old_mode = None

    if case == "unreadable-unequal":
        old_mode = prior.stat().st_mode
        prior.chmod(0)
        assert prior.stat().st_size != config_path.stat().st_size
        with pytest.raises(PermissionError):
            prior.read_bytes()
    else:
        real_list = config_backups.list_config_backups

        def mutate_after_enumeration(path: Path, reason: str | None = None):
            nonlocal reached
            selected = real_list(path, reason)
            assert selected and selected[0] == prior
            reached = True
            if case == "backup-directory":
                prior.unlink()
                prior.mkdir()
            elif case == "backup-fifo":
                target = tmp_path / "regular-target"
                target.write_bytes(b"old\n")
                prior.unlink()
                prior.symlink_to(target)
                os.mkfifo(fifo)
                prior.unlink()
                prior.symlink_to(fifo)
            elif case == "source-fifo":
                config_path.unlink()
                os.mkfifo(config_path)
            elif case == "missing-backup":
                prior.unlink()
            elif case == "missing-source":
                config_path.unlink()
            return selected

        monkeypatch.setattr(config_backups, "list_config_backups", mutate_after_enumeration)

    try:
        started = time.monotonic()
        with _bounded_fifo_call():
            result = backup_config(config_path, "race")
        elapsed = time.monotonic() - started
    finally:
        if old_mode is not None:
            prior.chmod(old_mode)

    if case != "unreadable-unequal":
        assert reached
    after = set(prior.parent.iterdir())
    if case in {"unreadable-unequal", "backup-directory", "backup-fifo"}:
        assert result is not None
        assert result.read_bytes() == current
        assert after - before == {result}
    else:
        assert result is None
        assert after - before == set()
        if case == "source-fifo":
            assert elapsed < 1.0


class _FaultAfterFirstChunk:
    def __init__(self, raw):
        self.raw = raw
        self.first = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.raw.__exit__(*args)

    def read(self, size: int = -1):
        if size < 0 or not self.first:
            raise OSError(errno.EIO, "injected trailing read failure")
        self.first = False
        return self.raw.read(size)


@pytest.mark.parametrize(
    "case",
    [
        "equal-8191", "equal-8192", "equal-8193", "equal-16384", "equal-16385",
        "mismatch-8191", "mismatch-8192", "mismatch-16384",
        "source-grows", "backup-grows", "unreadable-equal", "early-mismatch-eio", "bounded-large",
    ],
)
def test_backup_compares_bytes_with_bounded_reads(tmp_path: Path, monkeypatch, case: str):
    config_path = tmp_path / "config.yaml"
    if case == "bounded-large":
        size = 64 * 1024 * 1024 + 1
        with config_path.open("wb") as stream:
            stream.truncate(size)
        prior = _seed_backup(config_path, "bytes", b"")
        with prior.open("r+b") as stream:
            stream.truncate(size)
        script = """
import os, resource, sys
from pathlib import Path
from hermes_cli.config_backups import backup_config
expected = Path(sys.argv[3]).resolve()
import hermes_cli.config_backups as module
assert Path(module.__file__).resolve() == expected
pages = int(Path('/proc/self/statm').read_text().split()[0])
vms = pages * os.sysconf('SC_PAGE_SIZE')
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
limit = vms + 24 * 1024 * 1024
if hard != resource.RLIM_INFINITY:
    limit = min(limit, hard)
resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
assert resource.getrlimit(resource.RLIMIT_AS)[0] == limit
result = backup_config(Path(sys.argv[1]), sys.argv[2])
print('NONE' if result is None else result)
"""
        proc = subprocess.run(
            [sys.executable, "-c", script, str(config_path), "bytes", str(Path(config_backups.__file__))],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "NONE"
        assert list(prior.parent.iterdir()) == [prior]
        return

    if case.startswith("equal-"):
        length = int(case.split("-")[1])
        payload = bytes((i * 37) % 256 for i in range(length))
        current = previous = payload
    elif case.startswith("mismatch-"):
        offset = int(case.split("-")[1])
        previous = bytearray(bytes((i * 37) % 256 for i in range(16385)))
        current_bytes = bytearray(previous)
        current_bytes[offset] ^= 0xFF
        current, previous = bytes(current_bytes), bytes(previous)
    else:
        current = previous = bytes((i * 37) % 256 for i in range(16385))
        if case == "early-mismatch-eio":
            previous = bytes([current[0] ^ 0xFF]) + current[1:]

    config_path.write_bytes(current)
    prior = _seed_backup(config_path, "bytes", previous)
    before = set(prior.parent.iterdir())
    old_mode = None
    real_open = Path.open

    if case in {"source-grows", "backup-grows"}:
        changed = False

        def append_before_open(self, mode="r", *args, **kwargs):
            nonlocal changed
            if not changed and self == config_path and mode == "rb":
                changed = True
                target = config_path if case == "source-grows" else prior
                with open(target, "ab") as stream:
                    stream.write(b"!")
            return real_open(self, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", append_before_open)
    elif case == "unreadable-equal":
        old_mode = prior.stat().st_mode
        prior.chmod(0)
        with pytest.raises(PermissionError):
            prior.read_bytes()
    elif case == "early-mismatch-eio":
        def fail_after_first_chunk(self, mode="r", *args, **kwargs):
            raw = real_open(self, mode, *args, **kwargs)
            if self == prior and mode == "rb":
                return _FaultAfterFirstChunk(raw)
            return raw

        monkeypatch.setattr(Path, "open", fail_after_first_chunk)

    try:
        result = backup_config(config_path, "bytes")
    finally:
        if old_mode is not None:
            prior.chmod(old_mode)

    if case.startswith("equal-") or case == "unreadable-equal":
        assert result is None
        assert set(prior.parent.iterdir()) == before
    else:
        assert result is not None
        assert result.read_bytes() == config_path.read_bytes()
        assert set(prior.parent.iterdir()) - before == {result}
