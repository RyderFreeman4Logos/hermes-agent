"""Tests for the ``pinPeerName`` / ``pinUserPeer`` config flag.

Under a gateway (Telegram, Discord, Slack, ...) Hermes passes the
platform-native user ID as ``runtime_user_peer_name`` into
``HonchoSessionManager``.  By default that ID wins over any configured
``peer_name`` so multi-user bots scope memory per user.

For single-user deployments connecting over multiple platforms,
``pinUserPeer: true`` pins the user peer to ``peer_name`` so memory stays
unified across platforms.

Tests cover config parsing (``client.py::from_global_config``) and resolver
order (``session.py::get_or_create``), stubbing Honcho API calls so the
chosen ``user_peer_id`` can be asserted without touching the network.
"""

import hashlib
import os
import json
from pathlib import Path
from unittest.mock import MagicMock


from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestPinPeerNameConfigParsing:
    def test_default_is_false(self):
        """Default preserves existing behaviour — multi-user bots unaffected."""
        config = HonchoClientConfig()
        assert config.pin_peer_name is False

    def test_root_level_true(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "Igor",
            "pinPeerName": True,
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is True
        assert config.peer_name == "Igor"

    def test_host_block_true(self, tmp_path, monkeypatch):
        """Host-level flag works the same as root-level."""
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "Igor",
            "hosts": {
                "hermes": {"pinPeerName": True},
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is True


    def test_explicit_false_parses(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "Igor",
            "pinPeerName": False,
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is False


class TestRuntimePeerMappingConfigParsing:
    def test_defaults_are_empty(self):
        config = HonchoClientConfig()
        assert config.user_peer_aliases == {}
        assert config.runtime_peer_prefix == ""


    def test_malformed_alias_config_is_ignored(self, tmp_path):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "userPeerAliases": ["not", "a", "map"],
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)

        assert config.user_peer_aliases == {}


# ---------------------------------------------------------------------------
# Peer resolution (the actual bug fix)
# ---------------------------------------------------------------------------


def _patch_manager_for_resolution_test(mgr: HonchoSessionManager) -> None:
    """Stub out the Honcho client so ``get_or_create`` doesn't try to talk
    to the network — we only care about the user_peer_id chosen before
    those calls happen.
    """
    fake_peer = MagicMock()
    mgr._get_or_create_peer = MagicMock(return_value=fake_peer)
    mgr._get_or_create_honcho_session = MagicMock(
        return_value=(MagicMock(), [])
    )


class TestPeerResolutionOrder:
    """Matrix of (runtime_id, pin_peer_name, peer_name) → expected user_peer_id."""

    def _config(
        self,
        *,
        peer_name: str | None,
        pin_peer_name: bool,
        user_peer_aliases: dict[str, str] | None = None,
        runtime_peer_prefix: str = "",
        session_peer_prefix: bool = False,
    ) -> HonchoClientConfig:
        # The test doesn't need auth / Honcho — disable the provider so
        # the manager doesn't try to open a real client.
        return HonchoClientConfig(
            api_key="test-key",
            peer_name=peer_name,
            pin_peer_name=pin_peer_name,
            user_peer_aliases=user_peer_aliases or {},
            runtime_peer_prefix=runtime_peer_prefix,
            session_peer_prefix=session_peer_prefix,
            enabled=False,
            write_frequency="turn",  # avoid spawning the async writer thread
        )

    def test_runtime_wins_when_pin_is_false(self):
        """Regression guard: default behaviour must stay unchanged.
        Multi-user bots rely on the platform-native ID winning."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name="Igor", pin_peer_name=False),
            runtime_user_peer_name="7654321",  # e.g. Telegram UID
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "7654321", (
            "pin_peer_name=False is the multi-user default — the gateway's "
            "platform-native user ID must win so each user gets their own "
            "peer scope.  If this regresses, every Telegram/Discord/Slack "
            "bot immediately merges memory across users."
        )

    def test_alias_wins_for_known_runtime_id(self):
        """Known platform IDs can preserve an existing stable Honcho peer."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=False,
                user_peer_aliases={"7654321": "Igor"},
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "Igor"

    def test_unknown_runtime_id_uses_prefix(self):
        """Unknown gateway users stay isolated but become platform-scoped."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "telegram_7654321"

    def test_prefixed_runtime_id_hashes_when_sanitization_is_lossy(self):
        """Generated prefixed IDs avoid merges caused by lossy sanitization."""
        raw_peer_id = "telegram_user:42"
        expected_hash = hashlib.sha256(raw_peer_id.encode("utf-8")).hexdigest()[:8]
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="user:42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:user:42")
        assert session.user_peer_id == f"telegram_user-42-{expected_hash}"

    def test_prefixed_runtime_id_hashes_when_it_collides_with_peer_name(self):
        """Unknown generated peers should not silently merge into peerName."""
        raw_peer_id = "telegram_7654321"
        expected_hash = hashlib.sha256(raw_peer_id.encode("utf-8")).hexdigest()[:8]
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="telegram_7654321",
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == f"telegram_7654321-{expected_hash}"


    def test_alias_value_is_sanitized_after_selection(self):
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={"7654321": "Alice Smith!"},
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "Alice-Smith-"

    def test_alias_keys_match_raw_runtime_id_before_sanitization(self):
        """Alias selection is exact on platform IDs before Honcho ID cleanup."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={
                    "user:42": "raw-match",
                    "user-42": "sanitized-match",
                },
            ),
            runtime_user_peer_name="user:42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:user:42")
        assert session.user_peer_id == "raw-match"

    def test_session_peer_prefix_is_orthogonal_to_runtime_peer_prefix(self):
        """sessionPeerPrefix scopes session IDs; runtimePeerPrefix scopes user peers."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
                session_peer_prefix=True,
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "telegram_7654321"
        assert session.honcho_session_id == "telegram-7654321"

    def test_config_wins_when_pin_is_true(self):
        """With pin enabled, configured peer_name beats runtime ID."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=True,
                user_peer_aliases={"7654321": "Alias"},
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="7654321",  # Telegram pushes this in
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "Igor", (
            "With pinPeerName=true the user's configured peer_name must "
            "beat the platform-native runtime ID so memory stays unified "
            "across Telegram/Discord/Slack for the same person."
        )

    def test_pin_noop_when_peer_name_missing(self):
        """Safety: pinPeerName alone (no peer_name) must not silently drop
        the runtime identity.  Without a configured peer_name there's
        nothing to pin to — fall through to runtime mapping."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=True,
                user_peer_aliases={"7654321": "Igor"},
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:7654321")
        assert session.user_peer_id == "Igor"


    def test_alt_runtime_id_can_match_alias_without_changing_raw_fallback(self):
        """Stable alternate IDs can map known users while primary ID fallback stays unchanged."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={"union-user": "Igor"},
                runtime_peer_prefix="feishu_",
            ),
            runtime_user_peer_name="open-id",
            runtime_user_peer_name_alt="union-user",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:chat")
        assert session.user_peer_id == "Igor"


    def test_everything_missing_falls_back_to_session_key(self):
        """Deepest fallback: no runtime identity, no peer_name, no pin.
        Must still produce a deterministic peer_id from the session key."""
        # Config with no peer_name and default pin_peer_name=False
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name=None, pin_peer_name=False),
            runtime_user_peer_name=None,
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:123")
        assert session.user_peer_id == "user-telegram-123"


class TestCrossPlatformMemoryUnification:
    """The same physical user talking to Hermes via Telegram AND Discord
    lands on ONE peer when ``pinPeerName`` is opted in.
    """

    def _config_pinned(self) -> HonchoClientConfig:
        return HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=True,
            enabled=False,
            write_frequency="turn",
        )

    def test_telegram_and_discord_collapse_to_one_peer_when_pinned(self):
        """Single-user deployment: Telegram UID and Discord snowflake
        both resolve to the same configured peer_name."""
        # Telegram turn
        mgr_telegram = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config_pinned(),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr_telegram)
        telegram_session = mgr_telegram.get_or_create("telegram:7654321")

        # Discord turn (separate manager instance — simulates a fresh
        # platform-adapter invocation)
        mgr_discord = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config_pinned(),
            runtime_user_peer_name="1348750102029926454",
        )
        _patch_manager_for_resolution_test(mgr_discord)
        discord_session = mgr_discord.get_or_create("discord:1348750102029926454")

        assert telegram_session.user_peer_id == "Igor"
        assert discord_session.user_peer_id == "Igor"
        assert telegram_session.user_peer_id == discord_session.user_peer_id, (
            "cross-platform memory unification is the whole point of "
            "pinPeerName — both platforms must land on the same Honcho peer"
        )

    def test_multiuser_default_keeps_platforms_separate(self):
        """Negative control: with pinPeerName=false (the default), two
        different platform IDs must produce two different peers so
        multi-user bots don't merge users."""
        cfg = HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=False,
            enabled=False,
            write_frequency="turn",
        )
        mgr_a = HonchoSessionManager(
            honcho=MagicMock(), config=cfg, runtime_user_peer_name="user_a",
        )
        mgr_b = HonchoSessionManager(
            honcho=MagicMock(), config=cfg, runtime_user_peer_name="user_b",
        )
        _patch_manager_for_resolution_test(mgr_a)
        _patch_manager_for_resolution_test(mgr_b)

        sess_a = mgr_a.get_or_create("telegram:a")
        sess_b = mgr_b.get_or_create("telegram:b")

        assert sess_a.user_peer_id == "user_a"
        assert sess_b.user_peer_id == "user_b"
        assert sess_a.user_peer_id != sess_b.user_peer_id, (
            "multi-user default MUST keep users separate — a regression "
            "here would silently merge unrelated users' memory"
        )


class TestPinUserPeerAlias:
    """``pinUserPeer`` and ``pinPeerName`` both resolve to the same internal
    ``pin_peer_name`` field.  Precedence when both appear: host pinUserPeer →
    host pinPeerName → root pinUserPeer → root pinPeerName → default.
    """


    def test_pinPeerName_still_works_unchanged(self, tmp_path):
        from plugins.memory.honcho.client import HonchoClientConfig
        import json
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "***",
            "peerName": "eri",
            "hosts": {"hermes": {"pinPeerName": True}},
        }))
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is True


class TestPinTransition:
    """Behavior when honcho.json flips ``pinPeerName`` true → false.

    Covers two contracts:
      1. A freshly-built manager picks up the flipped config and resolves
         the same runtime ID to a new peer (no resolver staleness).
      2. The gateway's agent-cache signature reflects honcho identity-mapping
         changes, so a config edit busts the cached AIAgent on the next turn.
    """

    def _pinned(self) -> HonchoClientConfig:
        return HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=True,
            enabled=False,
            write_frequency="turn",
        )

    def _unpinned(self) -> HonchoClientConfig:
        return HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=False,
            enabled=False,
            write_frequency="turn",
        )

    def test_fresh_manager_after_flip_resolves_to_runtime(self):
        pinned_mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._pinned(),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(pinned_mgr)
        before = pinned_mgr.get_or_create("telegram:7654321")
        assert before.user_peer_id == "Igor"

        unpinned_mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._unpinned(),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(unpinned_mgr)
        after = unpinned_mgr.get_or_create("telegram:7654321")
        assert after.user_peer_id == "7654321", (
            "After flipping pinPeerName off, the same runtime ID must resolve "
            "to its own peer — otherwise multi-user mode silently merges users."
        )

    def test_cached_session_survives_config_flip_in_same_manager(self):
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._pinned(),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr)
        first = mgr.get_or_create("telegram:7654321")
        assert first.user_peer_id == "Igor"

        mgr._config = self._unpinned()
        second = mgr.get_or_create("telegram:7654321")
        assert second.user_peer_id == "Igor", (
            "The per-key session cache is keyed by session-key, not by "
            "resolved peer.  In-process flips don't invalidate it — the "
            "gateway cache must bust the whole manager instead."
        )

    def test_cache_busting_signature_reflects_pin_peer_name(self, tmp_path, monkeypatch):
        """Gateway agent cache must bust when honcho.json's pinPeerName flips."""
        from gateway.run import GatewayRunner

        cfg_path = tmp_path / "honcho.json"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        cfg_path.write_text(json.dumps({"apiKey": "k", "peerName": "Igor", "pinPeerName": True}))
        sig_pinned = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})

        cfg_path.write_text(json.dumps({"apiKey": "k", "peerName": "Igor", "pinPeerName": False}))
        sig_unpinned = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})

        assert sig_pinned["honcho.pin_peer_name"] != sig_unpinned["honcho.pin_peer_name"]

    def test_cache_busting_identity_matches_the_digest_snapshot(self, tmp_path, monkeypatch):
        """A rewrite after hashing must not parse a different Honcho identity."""
        import gateway.run_agent_cache as agent_cache
        from gateway.run import GatewayRunner

        cfg_path = tmp_path / "honcho.json"
        first = {"apiKey": "k", "peerName": "Alice", "pinPeerName": True}
        second = {"apiKey": "k", "peerName": "Bob", "pinPeerName": False}
        cfg_path.write_text(json.dumps(first))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(GatewayRunner, "_HONCHO_CACHE_BUSTING_MEMO", {})

        original_sha256 = agent_cache.hashlib.sha256
        swapped = False

        def swap_after_digest(*args, **kwargs):
            digest = original_sha256(*args, **kwargs)

            class SwitchingDigest:
                def digest(self):
                    nonlocal swapped
                    if not swapped:
                        cfg_path.write_text(json.dumps(second))
                        swapped = True
                    return digest.digest()

                def __getattr__(self, name):
                    return getattr(digest, name)

            return SwitchingDigest()

        monkeypatch.setattr(agent_cache.hashlib, "sha256", swap_after_digest)
        signature = GatewayRunner._extract_cache_busting_config(
            {"memory": {"provider": "honcho"}}
        )

        assert signature["honcho.peer_name"] == "Alice"
        assert signature["honcho.pin_peer_name"] is True



class TestProfilePeerUniqueness:
    """Each Hermes profile can pin to its own unique peerName.

    Profile cloning copies host blocks, but operators routinely diverge them
    afterwards (e.g. `hermes -p partner` pinned to a different person's peer).
    The resolver must honor host-level ``peerName`` so two profiles in the
    same workspace stay scoped to different Honcho peers.
    """

    def _pinned_to(self, name: str) -> HonchoClientConfig:
        return HonchoClientConfig(
            api_key="k",
            peer_name=name,
            pin_peer_name=True,
            enabled=False,
            write_frequency="turn",
        )

    def test_two_profiles_pinned_to_different_peer_names_resolve_distinctly(self):
        mgr_a = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._pinned_to("alice"),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr_a)
        sess_a = mgr_a.get_or_create("telegram:7654321")

        mgr_b = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._pinned_to("bob"),
            runtime_user_peer_name="7654321",
        )
        _patch_manager_for_resolution_test(mgr_b)
        sess_b = mgr_b.get_or_create("telegram:7654321")

        assert sess_a.user_peer_id == "alice"
        assert sess_b.user_peer_id == "bob"
        assert sess_a.user_peer_id != sess_b.user_peer_id, (
            "Profiles pinned to distinct peer names must not collapse to "
            "the same Honcho peer — otherwise profile isolation is fictional."
        )


def test_cache_busting_oversized_honcho_snapshot_tracks_content(tmp_path, monkeypatch):
    """W11: oversized Honcho observer is bounded, fully hashed, and never parses retained bytes."""
    from gateway import run_agent_cache as agent_cache
    from gateway.run import GatewayRunner
    from plugins.memory.honcho.client import HonchoClientConfig

    path = tmp_path / "honcho.json"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(GatewayRunner, "_HONCHO_CACHE_BUSTING_MEMO", {})

    def write(peer: str, extra_padding: int = 0) -> None:
        path.write_text(json.dumps({"apiKey": "k", "peerName": peer, "pinPeerName": True,
                                    "padding": " " * (1024 * 1024 + 32 + extra_padding)}))

    write("Alice")
    assert HonchoClientConfig.from_global_config(config_path=path).peer_name == "Alice"
    reads = []
    original_open = Path.open

    class TrackedReader:
        def __init__(self, fileobj):
            self._fileobj = fileobj

        def read(self, size=-1):
            reads.append(size)
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
        return TrackedReader(fileobj) if self == path and mode == "rb" else fileobj

    def forbid_full_parser(*_args, **_kwargs):
        raise AssertionError("oversized observer invoked full JSON parser")

    monkeypatch.setattr(Path, "open", track_open)
    monkeypatch.setattr(agent_cache.json, "loads", forbid_full_parser)
    first = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})
    again = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})
    write("Blice")
    second = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})
    write("Blice", extra_padding=1)
    whitespace_changed = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})
    assert reads
    assert all(0 < size <= 64 * 1024 for size in reads)
    assert first["honcho.overflow_content"] is not None
    assert first == again
    assert first["honcho.overflow_content"] != second["honcho.overflow_content"]
    assert second["honcho.overflow_content"] != whitespace_changed["honcho.overflow_content"]
    assert second["honcho.peer_name"] is None


def test_cache_busting_honcho_projection_boundaries_preserve_live_identity(tmp_path, monkeypatch):
    """W1: projection size never changes live Honcho identity or cache freshness."""
    from gateway.run import GatewayRunner

    path = tmp_path / "honcho.json"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(GatewayRunner, "_HONCHO_CACHE_BUSTING_MEMO", {})

    def document(peer_name, pin_peer_name, size):
        body = {"apiKey": "k", "peerName": peer_name, "pinPeerName": pin_peer_name, "padding": ""}
        padding = size - len(json.dumps(body, separators=(",", ":")).encode())
        assert padding >= 0
        body["padding"] = " " * padding
        encoded = json.dumps(body, separators=(",", ":")).encode()
        assert len(encoded) == size
        return encoded

    def observe(peer_name, pin_peer_name, size):
        path.write_bytes(document(peer_name, pin_peer_name, size))
        live = HonchoClientConfig.from_global_config(config_path=path)
        keys = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})
        signature = GatewayRunner._agent_config_signature("test-model", {}, [], "", cache_keys=keys)
        manager = HonchoSessionManager(
            honcho=MagicMock(), config=live, runtime_user_peer_name="runtime-42"
        )
        _patch_manager_for_resolution_test(manager)
        session = manager.get_or_create("telegram:42")
        return live, keys, signature, session

    below = 1024 * 1024 - 1
    edge = 1024 * 1024
    above = 1024 * 1024 + 1
    small_live, small_keys, small_signature, small_session = observe("Alice", True, below)
    edge_live, edge_keys, edge_signature, edge_session = observe("Alice", True, edge)
    large_live, large_keys, large_signature, large_session = observe("Alice", True, above)
    stable_large = GatewayRunner._agent_config_signature(
        "test-model", {}, [], "",
        cache_keys=GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}}),
    )
    peer_live, peer_keys, peer_signature, peer_session = observe("Blice", True, above)
    pin_live, pin_keys, pin_signature, pin_session = observe("Blice", False, above)
    returned_live, returned_keys, returned_signature, returned_session = observe("Alice", True, below)

    for live in (small_live, edge_live, large_live, peer_live, pin_live, returned_live):
        assert live.api_key == "k"
    assert [small_live.peer_name, edge_live.peer_name, large_live.peer_name] == ["Alice"] * 3
    assert small_keys["honcho.overflow_content"] is None
    assert edge_keys["honcho.overflow_content"] is None
    assert large_keys["honcho.overflow_content"] is not None
    assert large_signature == stable_large
    assert large_signature != peer_signature
    assert peer_signature != pin_signature
    assert small_signature == returned_signature
    assert edge_signature != large_signature
    assert small_session.user_peer_id == edge_session.user_peer_id == large_session.user_peer_id == "Alice"
    assert peer_session.user_peer_id == "Blice"
    assert pin_session.user_peer_id == "runtime-42"
    assert returned_session.user_peer_id == "Alice"
    assert peer_keys["honcho.overflow_content"] != pin_keys["honcho.overflow_content"]
    assert returned_keys["honcho.overflow_content"] is None


def test_cache_busting_null_snapshot_never_reopens_to_new_identity(tmp_path, monkeypatch):
    """A null read is invalid snapshot data, not permission to reopen a newer file."""
    from gateway.run import GatewayRunner

    path = tmp_path / "honcho.json"
    path.write_text("null")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(GatewayRunner, "_HONCHO_CACHE_BUSTING_MEMO", {})
    original_open = Path.open
    swapped = False

    class SwitchingReader:
        def __init__(self, stream): self._stream = stream
        def read(self, *args, **kwargs):
            nonlocal swapped
            value = self._stream.read(*args, **kwargs)
            if not swapped:
                path.write_text(json.dumps({"apiKey": "k", "peerName": "Bob"}))
                swapped = True
            return value
        def __enter__(self): return self
        def __exit__(self, *args): return self._stream.__exit__(*args)
        def __getattr__(self, name): return getattr(self._stream, name)

    def switching_open(self, *args, **kwargs):
        stream = original_open(self, *args, **kwargs)
        return SwitchingReader(stream) if self == path and (args[0] if args else kwargs.get("mode")) == "rb" else stream

    monkeypatch.setattr(Path, "open", switching_open)
    result = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})
    assert swapped
    assert result["honcho.peer_name"] is None
    assert result["honcho.overflow_content"] is None
    assert HonchoClientConfig.from_global_config(config_path=path).peer_name == "Bob"
    path.write_text("null")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    warm = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})
    assert warm["honcho.peer_name"] is None
    assert warm["honcho.overflow_content"] is None


def test_cache_busting_same_metadata_honcho_rewrite_changes_identity_and_signature(
    tmp_path, monkeypatch
):
    """W6: a small same-size/same-mtime rewrite still rebuilds Honcho identity."""
    from gateway.run import GatewayRunner

    path = tmp_path / "honcho.json"
    first = {"apiKey": "k", "peerName": "Alice", "pinPeerName": True}
    second = {"apiKey": "k", "peerName": "Blice", "pinPeerName": True}
    first_bytes = json.dumps(first).encode()
    second_bytes = json.dumps(second).encode()
    assert len(first_bytes) == len(second_bytes)
    path.write_bytes(first_bytes)
    original_stat = path.stat()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(GatewayRunner, "_HONCHO_CACHE_BUSTING_MEMO", {})

    first_values = GatewayRunner._extract_cache_busting_config(
        {"memory": {"provider": "honcho"}}
    )
    first_signature = GatewayRunner._agent_config_signature(
        "test-model", {}, [], "", cache_keys=first_values
    )

    path.write_bytes(second_bytes)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second_values = GatewayRunner._extract_cache_busting_config(
        {"memory": {"provider": "honcho"}}
    )
    second_signature = GatewayRunner._agent_config_signature(
        "test-model", {}, [], "", cache_keys=second_values
    )

    assert first_values["honcho.peer_name"] == "Alice"
    assert second_values["honcho.peer_name"] == "Blice"
    assert first_signature != second_signature


def test_cache_busting_read_barrier_uses_the_already_read_small_snapshot(
    tmp_path, monkeypatch
):
    """W6: swapping after the real stream yields bytes cannot alter that extraction."""
    from gateway.run import GatewayRunner

    path = tmp_path / "honcho.json"
    first = {"apiKey": "k", "peerName": "Alice", "pinPeerName": True}
    second = {"apiKey": "k", "peerName": "Bob", "pinPeerName": False}
    path.write_text(json.dumps(first), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(GatewayRunner, "_HONCHO_CACHE_BUSTING_MEMO", {})
    original_open = Path.open
    swapped = False

    class SwapAfterRead:
        def __init__(self, source):
            self._source = source

        def read(self, *args, **kwargs):
            nonlocal swapped
            result = self._source.read(*args, **kwargs)
            if not swapped:
                path.write_text(json.dumps(second), encoding="utf-8")
                swapped = True
            return result

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._source.close()

        def __getattr__(self, name):
            return getattr(self._source, name)

    def swap_after_target_read(self, *args, **kwargs):
        source = original_open(self, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        return SwapAfterRead(source) if self == path and mode == "rb" else source

    monkeypatch.setattr(Path, "open", swap_after_target_read)
    values = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})

    assert swapped
    assert values["honcho.peer_name"] == "Alice"
    assert values["honcho.pin_peer_name"] is True


def test_profile_host_precedence_and_shared_oversized_snapshot_scope(tmp_path, monkeypatch):
    """W9: profile resolution keeps provenance and projects shared overflow per scope."""
    from gateway.run import GatewayRunner
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / "hermes-root"
    root.mkdir()
    default_home = root
    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    alpha_home = root / "profiles" / "alpha"
    beta_home = root / "profiles" / "beta"
    alpha_home.mkdir(parents=True)
    beta_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_HONCHO_HOST", raising=False)

    def write_config(path, default_host, hosts, padding=""):
        path.write_text(json.dumps({
            "defaultHost": default_host,
            "hosts": hosts,
            "padding": padding,
        }), encoding="utf-8")

    write_config(default_home / "honcho.json", "default-local", {
        "default-local": {"apiKey": "default-key", "workspace": "default-ws"},
    })
    write_config(custom_home / "honcho.json", "custom-local", {
        "custom-local": {"apiKey": "custom-key", "workspace": "custom-ws"},
    })

    def resolve_in(home, host=None):
        token = set_hermes_home_override(home)
        try:
            return HonchoClientConfig.from_global_config(host=host)
        finally:
            reset_hermes_home_override(token)

    default_cfg = resolve_in(default_home)
    custom_cfg = resolve_in(custom_home)
    assert default_cfg.host == "default-local"
    assert custom_cfg.host == "custom-local"
    assert default_cfg.config_path == default_home / "honcho.json"
    assert default_cfg.hermes_home == default_home
    assert custom_cfg.config_path == custom_home / "honcho.json"
    assert custom_cfg.hermes_home == custom_home

    shared_path = root / "honcho.json"
    write_config(shared_path, "must-not-select-for-named", {
        "hermes_alpha": {"apiKey": "alpha-key", "workspace": "alpha-ws"},
        "hermes_beta": {"apiKey": "beta-key", "workspace": "beta-ws"},
        "env-host": {"apiKey": "env-key", "workspace": "env-ws"},
        "call-host": {"apiKey": "call-key", "workspace": "call-ws"},
    })

    alpha_cfg = resolve_in(alpha_home)
    beta_cfg = resolve_in(beta_home)
    assert alpha_cfg.host == "hermes_alpha"
    assert beta_cfg.host == "hermes_beta"
    assert alpha_cfg.config_path == shared_path
    assert beta_cfg.config_path == shared_path
    assert alpha_cfg.hermes_home == alpha_home
    assert beta_cfg.hermes_home == beta_home

    monkeypatch.setenv("HERMES_HONCHO_HOST", "env-host")
    assert resolve_in(alpha_home).host == "env-host"
    assert resolve_in(alpha_home, host="call-host").host == "call-host"
    monkeypatch.delenv("HERMES_HONCHO_HOST", raising=False)

    write_config(shared_path, "must-not-select-for-named", {
        "hermes_alpha": {"apiKey": "alpha-key", "workspace": "alpha-ws"},
        "hermes_beta": {"apiKey": "beta-key", "workspace": "beta-ws"},
    }, padding=" " * (1024 * 1024 + 32))
    monkeypatch.setattr(GatewayRunner, "_HONCHO_CACHE_BUSTING_MEMO", {})

    def extract_in(home):
        token = set_hermes_home_override(home)
        try:
            return GatewayRunner._extract_honcho_cache_busting_config()
        finally:
            reset_hermes_home_override(token)

    alpha_first = extract_in(alpha_home)
    beta = extract_in(beta_home)
    alpha_again = extract_in(alpha_home)
    assert len(GatewayRunner._HONCHO_CACHE_BUSTING_MEMO) == 1
    assert alpha_first["honcho.overflow_content"] is not None
    assert alpha_first["honcho.overflow_content"] != beta["honcho.overflow_content"]
    assert alpha_first["honcho.overflow_content"] == alpha_again["honcho.overflow_content"]
