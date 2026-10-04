"""Config readers must see in-place edits whose complete stat tuple is unchanged."""
import os
import pytest
from hermes_cli import config
from utils import file_signature

@pytest.mark.parametrize('reader', ['load_config', 'load_config_readonly', 'read_raw_config', 'read_raw_config_readonly'])
def test_same_stat_bytes_invalidate_config_cache(tmp_path, monkeypatch, reader):
    path = tmp_path / 'config.yaml'
    monkeypatch.setattr(config, 'get_config_path', lambda: path)
    load = getattr(config, reader)
    # Backup I/O is independent of cache admission and can cross the filesystem clock tick.
    monkeypatch.setattr('hermes_cli.config_backups.backup_config', lambda *a, **kw: None)
    old = 'mcp_servers:\n  bb: {command: b}\n'
    new = 'mcp_servers:\n  aa: {command: a}\n'
    for _ in range(256):
        path.write_text(old)
        config._LOAD_CONFIG_CACHE.clear()
        config._RAW_CONFIG_CACHE.clear()
        assert load()['mcp_servers'] == {'bb': {'command': 'b'}}
        before = path.stat()
        path.write_text(new)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        if file_signature(path.stat()) == file_signature(before):
            break
    assert file_signature(path.stat()) == file_signature(before), 'no unchanged-stat witness'
    assert load()['mcp_servers'] == {'aa': {'command': 'a'}}
