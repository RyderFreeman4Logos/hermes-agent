"""Regression tests for the terminal config → env fallback bridge.

``terminal_tool._get_env_config()`` reads all settings from TERMINAL_* env
vars, which the CLI / gateway / TUI-PTY launchers bridge from config.yaml at
startup. Processes that skip every launcher bridge (``hermes serve`` and the
Desktop app's in-process agents, the desktop cron ticker, ACP) used to fall
back silently to the local backend even when config.yaml selected
``terminal.backend: docker`` — commands the user intended to sandbox ran on
the host (#63141 / #54449 / #61115 / #65696).

``_ensure_terminal_env_bridged()`` closes that hole at the chokepoint and
hot-reloads when config.yaml's (mtime_ns, size) changes. When config.yaml
has a ``terminal:`` section, keys present there are authoritative (matches
``apply_terminal_config_to_env`` with ``override=None``).
"""

import os
import time

import pytest

import tools.terminal_tool as terminal_tool
from hermes_constants import get_hermes_home


@pytest.fixture(autouse=True)
def _reset_bridge_state(monkeypatch):
    """Each test starts with an empty signature cache and no TERMINAL_ENV."""
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_sig", None)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.delenv("TERMINAL_DOCKER_IMAGE", raising=False)
    monkeypatch.delenv("TERMINAL_AUTO_BACKGROUND_TIMEOUT_THRESHOLD", raising=False)
    # The config layer caches by (path, mtime, size); leave it alone — each
    # test writes its own config.yaml which changes the signature.
    yield


def _write_config(text: str) -> None:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.yaml"
    path.write_text(text)
    # Ensure size/mtime differ across rewrites on fast filesystems.
    path.touch()


def test_unset_terminal_env_backfills_backend_from_config():
    """The core #63141 fix: config's docker backend reaches _get_env_config
    even when no launcher bridged TERMINAL_ENV into this process."""
    _write_config(
        "terminal:\n"
        "  backend: docker\n"
        "  docker_image: custom/image:1\n"
    )

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert config["docker_image"] == "custom/image:1"
    assert os.environ.get("TERMINAL_ENV") == "docker"


def test_explicit_terminal_env_wins_over_config_without_backend_key(monkeypatch):
    """Env-only TERMINAL_ENV survives when config has no terminal.backend key.

    apply_terminal_config_to_env only writes keys present under terminal:; a
    config section without ``backend`` does not clobber an explicit env choice.
    """
    _write_config(
        "terminal:\n"
        "  docker_image: from-config:1\n"
    )
    monkeypatch.setenv("TERMINAL_ENV", "local")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "local"
    assert config["docker_image"] == "from-config:1"


def test_config_terminal_backend_is_authoritative_over_env(monkeypatch):
    """When config.yaml has terminal.backend, config wins on bridge.

    Hot-reload uses apply_terminal_config_to_env(override=None); a present
    terminal: section is authoritative for keys listed there.
    """
    _write_config("terminal:\n  backend: docker\n")
    monkeypatch.setenv("TERMINAL_ENV", "local")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert os.environ.get("TERMINAL_ENV") == "docker"


def test_preset_terminal_vars_survive_when_no_terminal_section(monkeypatch):
    """Without a terminal: section, already-set TERMINAL_* env values stay.

    apply_terminal_config_to_env(override=None) only backfills missing env when
    the file has no terminal section (defaults are non-authoritative).
    """
    _write_config("{}\n")  # no terminal: key
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "env/image:2")

    config = terminal_tool._get_env_config()

    assert config["docker_image"] == "env/image:2"


def test_terminal_section_overrides_env_with_merged_defaults(monkeypatch):
    """A present terminal: section is authoritative for mapped keys.

    load_config merges DEFAULT_CONFIG, so keys not listed in the user file
    still take the default value and override a pre-set env var when
    override=None and the file has terminal:.
    """
    _write_config("terminal:\n  backend: docker\n")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "env/image:2")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    # Default docker_image from DEFAULT_CONFIG wins over the pre-set env.
    assert config["docker_image"] != "env/image:2"
    assert "python-nodejs" in config["docker_image"]


def test_bridge_failure_falls_back_to_local(monkeypatch):
    """A broken config layer must not take the terminal tool down."""

    def _boom(*_a, **_k):
        raise RuntimeError("config exploded")

    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "apply_terminal_config_to_env", _boom)

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "local"
    # Failure must not permanently disable reload attempts.
    assert terminal_tool._terminal_config_bridge_sig is None


def test_bridge_same_signature_only_once(monkeypatch):
    """Same mtime/size only bridges once; second call skips re-apply."""
    calls = []

    import hermes_cli.config as config_mod

    real = config_mod.apply_terminal_config_to_env

    def _counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(config_mod, "apply_terminal_config_to_env", _counting)
    _write_config("{}\n")

    terminal_tool._get_env_config()
    terminal_tool._get_env_config()

    assert len(calls) == 1


def test_bridge_hot_reloads_auto_background_threshold_on_mtime_change():
    """Editing config.yaml threshold updates env + _get_env_config without restart."""
    _write_config(
        "terminal:\n"
        "  auto_background_timeout_threshold: 200\n"
    )
    config = terminal_tool._get_env_config()
    assert config["auto_background_timeout_threshold"] == 200
    assert os.environ.get("TERMINAL_AUTO_BACKGROUND_TIMEOUT_THRESHOLD") == "200"

    # Ensure mtime/size changes (tiny sleep + different content size).
    time.sleep(0.01)
    _write_config(
        "terminal:\n"
        "  auto_background_timeout_threshold: 19\n"
        "  # pad for size change\n"
    )

    config2 = terminal_tool._get_env_config()
    assert config2["auto_background_timeout_threshold"] == 19
    assert os.environ.get("TERMINAL_AUTO_BACKGROUND_TIMEOUT_THRESHOLD") == "19"
