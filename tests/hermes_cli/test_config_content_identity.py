"""Config readers must reload current bytes when their stat signature collides."""
import os
import pytest
from hermes_cli import config
from utils import file_signature

@pytest.mark.parametrize('reader', ['load_config', 'load_config_readonly', 'read_raw_config', 'read_raw_config_readonly'])
def test_content_change_reloads_when_stat_signature_collides(tmp_path, monkeypatch, reader):
    path = tmp_path / 'config.yaml'
    monkeypatch.setattr(config, 'get_config_path', lambda: path)
    load = getattr(config, reader)
    # Backup I/O is independent of cache admission and can cross the filesystem clock tick.
    monkeypatch.setattr('hermes_cli.config_backups.backup_config', lambda *a, **kw: None)
    old = 'mcp_servers:\n  bb: {command: b}\n'
    new = 'mcp_servers:\n  aa: {command: a}\n'
    path.write_text(old)
    before = path.stat()
    # Unit seam: force config's stat signature to collide; this is not a physical ctime witness.
    signature = file_signature(before)
    monkeypatch.setattr(config, 'file_signature', lambda _stat: signature)
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    assert load()['mcp_servers'] == {'bb': {'command': 'b'}}

    path.write_text(new)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    # The real rewrite preserves inode, size and mtime. ctime remains OS-managed.
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
    )
    assert path.read_text() == new
    assert load()['mcp_servers'] == {'aa': {'command': 'a'}}
