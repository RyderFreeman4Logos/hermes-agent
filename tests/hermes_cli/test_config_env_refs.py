import textwrap

from hermes_cli.config import load_config, read_raw_config, save_config


def _write_config(tmp_path, body: str):
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def _read_config(tmp_path) -> str:
    return (tmp_path / "config.yaml").read_text(encoding="utf-8")




def test_save_config_preserves_unresolved_env_refs(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: unresolved
            api_key: ${MISSING_SECRET}
            model: claude-opus-4-6
        model:
          default: claude-opus-4-6
        """,
    )

    config = load_config()
    config["display"]["compact"] = True
    save_config(config)

    assert "api_key: ${MISSING_SECRET}" in _read_config(tmp_path)


def test_save_config_allows_intentional_secret_value_change(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TU_ZI_API_KEY", "sk-old-secret")
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: tuzi
            api_key: ${TU_ZI_API_KEY}
            model: claude-opus-4-6
        model:
          default: claude-opus-4-6
        """,
    )

    config = load_config()
    config["custom_providers"][0]["api_key"] = "sk-new-secret"
    save_config(config)

    saved = _read_config(tmp_path)
    assert "api_key: sk-new-secret" in saved
    assert "${TU_ZI_API_KEY}" not in saved


def test_partial_save_preserves_raw_template_and_unrelated_values(monkeypatch, tmp_path):
    """W7: the public partial-save path preserves untouched raw YAML values."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("W7_PROVIDER_KEY", "resolved-only-in-memory")
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: retained
            api_key: ${W7_PROVIDER_KEY}
            model: retained-model
        unrelated:
          nested: keep-me
        agent:
          max_turns: 7
        """,
    )

    assert load_config()["custom_providers"][0]["api_key"] == "resolved-only-in-memory"
    save_config({"agent": {"max_turns": 8}}, merge_existing=True)

    raw = read_raw_config()
    assert raw["agent"]["max_turns"] == 8
    assert raw["unrelated"]["nested"] == "keep-me"
    assert raw["custom_providers"][0]["api_key"] == "${W7_PROVIDER_KEY}"


def test_persist_migration_preserves_raw_template_and_is_visible_to_raw_read(
    monkeypatch, tmp_path
):
    """W7: the migration chokepoint retains raw templates and unrelated data."""
    from hermes_cli.config import _persist_migration

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("W7_MIGRATION_KEY", "resolved-only-in-memory")
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: retained
            api_key: ${W7_MIGRATION_KEY}
            model: retained-model
        unrelated:
          nested: keep-me
        agent:
          max_turns: 7
        """,
    )

    migrated = load_config()
    assert migrated["custom_providers"][0]["api_key"] == "resolved-only-in-memory"
    migrated["agent"]["max_turns"] = 8
    _persist_migration(migrated)

    raw = read_raw_config()
    assert raw["agent"]["max_turns"] == 8
    assert raw["unrelated"]["nested"] == "keep-me"
    assert raw["custom_providers"][0]["api_key"] == "${W7_MIGRATION_KEY}"



