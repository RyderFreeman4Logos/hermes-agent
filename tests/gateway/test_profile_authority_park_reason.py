"""A reserved profile parked for an unusable store names the open failure, not an ownership mismatch.

The runner's scoped handle cache answers None when a profile's ``state.db`` fails to open, and
``_build_profile_authority`` reported every None as "session authority database does not belong to
the reserved profile": a legacy store crashing its migration on every open was parked under a
reason that sent the operator after the wrong problem (#141467). The reason is the exception the
failed open recorded; building it never opens (so never creates or migrates) the store.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import run_runtime
from gateway.session_db_recovery import RecoverableHandleCache


class _Registry:
    def add(self, *a, **kw):
        raise AssertionError('an unusable store must not reach registration')


def _runner(cache):
    return SimpleNamespace(_session_db=None, session_authorities=_Registry(), _session_db_handle_cache=cache,
                           session_runtime_descriptor={'instance_id': 'owner'})


@pytest.mark.asyncio
async def test_an_unopenable_store_is_parked_under_its_open_failure(tmp_path, monkeypatch):
    from hermes_state_registry import acquire
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    (home / 'state.db').mkdir()  # sqlite cannot open a directory: a store that fails on every open
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    cache = RecoverableHandleCache()
    assert cache.get(home / 'state.db', lambda: acquire(home / 'state.db')) is None  # the runner's failed open

    with pytest.raises(RuntimeError) as exc:
        await run_runtime._build_profile_authority(_runner(cache), 'worker', home, register=False)
    reason = str(exc.value)
    assert 'does not belong' not in reason
    assert str(Path(home) / 'state.db') in reason and 'did not open' in reason, reason


@pytest.mark.asyncio
async def test_the_park_reason_never_creates_the_store(tmp_path, monkeypatch):
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))

    with pytest.raises(RuntimeError) as exc:
        await run_runtime._build_profile_authority(_runner(RecoverableHandleCache()), 'worker', home, register=False)
    assert 'is unavailable' in str(exc.value)
    assert not (home / 'state.db').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_path", ["other.db", "../foreign/state.db"])
async def test_authority_refuses_a_different_database(tmp_path, monkeypatch, wrong_path):
    home = (tmp_path / 'profile').resolve()
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    runner = _runner(RecoverableHandleCache())
    runner._session_db = SimpleNamespace(db_path=home / wrong_path)
    with pytest.raises(RuntimeError, match='does not belong'):
        await run_runtime._build_profile_authority(runner, 'default', home, register=True)


@pytest.mark.parametrize("relocated", [False, True])
def test_profile_store_consumers_accept_its_file_and_refuse_other_stores(tmp_path, monkeypatch, relocated):
    from gateway.session_authorities import authority_for_home
    from gateway import session_ingress_media
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    target = tmp_path / 'relocated.db' if relocated else home / 'state.db'
    target.touch()
    if relocated:
        (home / 'state.db').symlink_to(target)
    db = SimpleNamespace(db_path=target)
    authority = SimpleNamespace(db=db)
    runner = SimpleNamespace(session_authority=authority)
    assert authority_for_home(runner, home) is authority
    monkeypatch.setattr(session_ingress_media, '_media_root', lambda: home / 'media')
    monkeypatch.setattr(session_ingress_media, 'release_unheld_media', lambda db, refs: 7)
    assert session_ingress_media.collect_unheld_api_images(db) == 7
    for wrong in (home / 'other.db', tmp_path / 'foreign' / 'state.db'):
        db.db_path = wrong
        assert authority_for_home(runner, home) is None
        assert session_ingress_media.collect_unheld_api_images(db) == 0
