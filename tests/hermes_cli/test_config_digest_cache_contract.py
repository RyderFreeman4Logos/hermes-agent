"""Content identity must remain coherent across digest failures and cache consumers."""
import pytest
from hermes_cli import config, config_effective
from utils import file_signature


@pytest.mark.parametrize('reader', ['read_raw_config', 'read_raw_config_readonly', 'load_config', 'load_config_readonly', 'load_user_config_effective'])
def test_digest_error_never_reuses_unverified_bytes(tmp_path, monkeypatch, reader):
    path = tmp_path / 'config.yaml'
    monkeypatch.setattr(config, 'get_config_path', lambda: path)
    monkeypatch.setattr('hermes_cli.config_backups.backup_config', lambda *a, **kw: None)
    load = getattr(config_effective if reader == 'load_user_config_effective' else config, reader)
    path.write_text('agent: {max_turns: 7}\n')
    assert load()['agent']['max_turns'] == 7
    stat = path.stat()
    # Freeze metadata, not bytes: a failed digest cannot grant a warm cache hit.
    monkeypatch.setattr(config, 'file_signature', lambda st: file_signature(stat))
    path.write_text('agent: {max_turns: 8}\n')
    def unavailable(path):
        raise OSError('synthetic transient EIO')
    with monkeypatch.context() as m:
        m.setattr('pm.filesystem.file_digest', unavailable)
        result = load()
        assert isinstance(result, dict)
        assert result['agent']['max_turns'] == 8
        path.write_text('agent: {max_turns: 9}\n')
        assert load()['agent']['max_turns'] == 9
    assert load()['agent']['max_turns'] == 9


@pytest.mark.parametrize('managed', [False, True])
def test_effective_cache_shares_raw_identity_and_invalidates_env(tmp_path, monkeypatch, managed):
    from hermes_cli import managed_scope
    path = tmp_path / 'config.yaml'
    monkeypatch.setattr(config, 'get_config_path', lambda: path)
    monkeypatch.setattr('hermes_cli.config_backups.backup_config', lambda *a, **kw: None)
    overlay = tmp_path / 'managed'; overlay.mkdir()
    monkeypatch.setattr(managed_scope, 'get_managed_dir', lambda: overlay if managed else None)
    if managed:
        (overlay / 'config.yaml').write_text('agent: {max_turns: 8}\n')
    monkeypatch.setenv('CACHE_CONTRACT_VALUE', 'first')
    path.write_text('agent: {max_turns: 7}\ncontract: ${CACHE_CONTRACT_VALUE}\n')
    raw = config.read_raw_config_readonly()
    first = config_effective.load_user_config_effective()
    assert config.read_raw_config_readonly() is raw
    cached = config_effective._EFFECTIVE_CACHE[str(path)]
    assert config_effective.load_user_config_effective() == first
    assert config_effective._EFFECTIVE_CACHE[str(path)] is cached
    monkeypatch.setenv('CACHE_CONTRACT_VALUE', 'second')
    second = config_effective.load_user_config_effective()
    assert second['contract'] == 'second'
    assert second['agent']['max_turns'] == (8 if managed else 7)
    assert config.read_raw_config_readonly() is raw
