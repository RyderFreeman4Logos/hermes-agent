"""Corrupt auth store must stay fail-closed under read-only enumeration."""

from pathlib import Path

from hermes_cli import auth as auth_mod


def test_corrupt_auth_store_read_only_does_not_write_backup(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    auth_path = hermes_home / "auth.json"
    auth_path.write_text("{malformed-auth", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    store = auth_mod._load_auth_store(read_only=True)
    assert store == {"version": auth_mod.AUTH_STORE_VERSION, "providers": {}}
    assert auth_mod.read_credential_pool(read_only=True) == {}
    assert auth_mod.get_provider_auth_state("anthropic", read_only=True) is None
    assert not list(hermes_home.glob("auth.json.corrupt*"))
    assert auth_path.read_text(encoding="utf-8") == "{malformed-auth"


def test_corrupt_auth_store_normal_path_still_backs_up(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    auth_path = hermes_home / "auth.json"
    auth_path.write_text("{malformed-auth", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    store = auth_mod._load_auth_store(read_only=False)
    assert store == {"version": auth_mod.AUTH_STORE_VERSION, "providers": {}}
    backups = sorted(hermes_home.glob("auth.json.corrupt*"))
    assert backups
    assert backups[0].read_text(encoding="utf-8") == "{malformed-auth"
