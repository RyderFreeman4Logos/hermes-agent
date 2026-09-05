"""Persist config-resolvable route aliases, never private URLs.

Synthetic hosts only (.test / .invalid). Config is monkeypatched.
"""

from __future__ import annotations

import json

from gateway.session import sanitize_model_override
from hermes_cli.model_switch import (
    ModelSwitchResult,
    apply_model_switch_after_compression,
    schedule_model_switch_after_compression,
)
from hermes_state import SessionDB
from utils import sanitize_persisted_base_url, sanitize_persisted_model_config

RESOLVABLE_URL = "https://lab.example.test/v1"
UNRESOLVABLE_URL = "https://unresolvable.invalid/v1"
CATALOG_URL = "https://openrouter.ai/api/v1"


def _lab_config():
    return {"providers": {"lab": {"api": RESOLVABLE_URL}}}


def _patch_config(monkeypatch, config=None):
    import hermes_cli.runtime_provider as rp

    payload = _lab_config() if config is None else config
    monkeypatch.setattr(rp, "load_config", lambda: payload)


def test_catalog_url_may_persist(monkeypatch):
    _patch_config(monkeypatch, config={})
    assert sanitize_persisted_base_url(CATALOG_URL) == CATALOG_URL


def test_unresolvable_url_is_omitted_not_rewritten(monkeypatch):
    _patch_config(monkeypatch, config={})
    assert sanitize_persisted_base_url(UNRESOLVABLE_URL) is None
    cleaned = sanitize_persisted_model_config(
        {"provider": "custom", "base_url": UNRESOLVABLE_URL}
    )
    dumped = json.dumps(cleaned)
    assert "base_url" not in cleaned
    assert "unresolvable.invalid" not in dumped
    assert cleaned.get("provider") != "custom:unresolvable"


def test_resolvable_custom_url_persists_alias_not_url(monkeypatch):
    _patch_config(monkeypatch)
    cleaned = sanitize_persisted_model_config(
        {"provider": "custom", "base_url": RESOLVABLE_URL, "model": "lab-model"}
    )
    dumped = json.dumps(cleaned)
    assert "base_url" not in cleaned
    assert "lab.example.test" not in dumped
    assert cleaned["provider"] == "custom:lab"
    assert cleaned["model"] == "lab-model"


def test_public_looking_unknown_url_is_not_kept(monkeypatch):
    _patch_config(monkeypatch, config={})
    override = {
        "model": "gpt-5o",
        "provider": "openai",
        "base_url": "https://api.example/v1?api-version=2024-01-01&region=us",
    }
    cleaned = sanitize_model_override(override)
    assert cleaned == {"model": "gpt-5o", "provider": "openai"}
    assert "api.example" not in json.dumps(cleaned)


def test_session_and_billing_omit_unresolvable_url(tmp_path, monkeypatch):
    _patch_config(monkeypatch, config={})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="mix-u", source="cli", model="m0")
    db.patch_session_model_config(
        "mix-u",
        {"provider": "custom", "base_url": UNRESOLVABLE_URL},
    )
    db.update_session_billing_route(
        "mix-u",
        provider="custom",
        base_url=UNRESOLVABLE_URL,
    )
    row = db.get_session("mix-u")
    dumped = json.dumps(dict(row))
    assert "unresolvable.invalid" not in dumped
    assert row["billing_base_url"] in (None, "")
    config = json.loads(row["model_config"])
    assert "base_url" not in config
    db.close()


def test_session_and_billing_persist_alias_roundtrip(tmp_path, monkeypatch):
    _patch_config(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="mix-r", source="cli", model="m0")
    db.patch_session_model_config(
        "mix-r",
        {"provider": "custom", "base_url": RESOLVABLE_URL},
    )
    db.update_session_billing_route(
        "mix-r",
        provider="custom",
        base_url=RESOLVABLE_URL,
    )
    row = db.get_session("mix-r")
    dumped = json.dumps(dict(row))
    assert "lab.example.test" not in dumped
    assert row["billing_base_url"] in (None, "")
    config = json.loads(row["model_config"])
    assert config.get("provider") == "custom:lab"
    assert "base_url" not in config
    db.close()


def test_cli_persist_restore_uses_config_url_not_disk(tmp_path, monkeypatch):
    import cli as cli_mod
    import hermes_cli.runtime_provider as rp

    _patch_config(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        rp,
        "resolve_runtime_provider",
        lambda requested=None, **_k: {
            "provider": requested,
            "base_url": RESOLVABLE_URL,
            "api_key": "resolved-key",
        },
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="rt-alias", source="cli", model="ambient-model")

    class _Result:
        new_model = "lab-model"
        target_provider = "custom:lab"
        base_url = RESOLVABLE_URL
        api_mode = ""

    writer = object.__new__(cli_mod.HermesCLI)
    writer._session_db = db
    writer.session_id = "rt-alias"
    writer._persist_model_switch_to_session(_Result())

    meta = db.get_session("rt-alias")
    assert "lab.example.test" not in json.dumps(dict(meta))

    restored = object.__new__(cli_mod.HermesCLI)
    restored.model = "ambient-model"
    restored.provider = "openrouter"
    restored.requested_provider = "openrouter"
    restored.base_url = "https://openrouter.ai/api/v1"
    restored.api_key = "ambient-key"
    restored.api_mode = ""
    restored.agent = None
    restored._console_print = lambda s: None
    restored._explicit_model_override = False
    restored._restore_session_model(meta)
    assert restored.provider == "custom:lab"
    assert restored.base_url == RESOLVABLE_URL
    db.close()


def test_deferred_apply_keeps_runtime_url_omits_unresolvable(monkeypatch):
    _patch_config(monkeypatch, config={})
    agent = type("Agent", (), {})()
    agent.model = "old-model"
    agent.provider = "old-provider"
    agent.base_url = "https://old.example.test/v1"
    agent.api_key = "old-key"
    agent.api_mode = "chat_completions"
    agent.session_id = "session-1"
    agent._session_init_model_config = {"max_iterations": 7}
    agent.statuses = []
    agent.calls = []

    class _DB:
        def __init__(self):
            self.row = {
                "model": "old-model",
                "model_config": json.dumps({"max_iterations": 7}),
                "system_prompt": "old prompt",
                "billing_provider": "old-provider",
                "billing_base_url": "https://old.example.test/v1",
                "billing_mode": "chat_completions",
            }

        def get_session(self, _session_id):
            return dict(self.row)

        def update_session_meta(self, _session_id, model_config, model=None):
            self.row["model_config"] = model_config
            if model is not None:
                self.row["model"] = model

        def update_system_prompt(self, _session_id, prompt):
            self.row["system_prompt"] = prompt

        def update_session_billing_route(
            self, _session_id, *, provider, base_url, billing_mode=None
        ):
            self.row.update(
                billing_provider=provider,
                billing_base_url=base_url,
                billing_mode=billing_mode,
            )

    agent._session_db = _DB()
    agent._emit_status = agent.statuses.append

    def _switch(model, provider, api_key, base_url, api_mode):
        agent.calls.append(("switch", model, provider))
        agent.model = model
        agent.provider = provider
        agent.api_key = api_key
        agent.base_url = base_url
        agent.api_mode = api_mode

    agent.switch_model = _switch
    result = ModelSwitchResult(
        success=True,
        new_model="new-model",
        target_provider="custom",
        api_key="new-key",
        base_url=UNRESOLVABLE_URL,
        api_mode="responses",
        provider_label="Custom",
    )
    schedule_model_switch_after_compression(agent, result)
    assert apply_model_switch_after_compression(agent) == "applied"
    durable = json.dumps(agent._session_db.row)
    assert "unresolvable.invalid" not in durable
    stored = json.loads(agent._session_db.row["model_config"])
    assert "base_url" not in stored
    assert agent.base_url == UNRESOLVABLE_URL
    assert agent._session_db.row["billing_base_url"] in (None, "")
