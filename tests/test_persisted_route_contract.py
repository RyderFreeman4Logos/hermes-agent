"""Real producer/resume matrix; config and credential-pool I/O only are replaced."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
import gateway.run as gateway
import hermes_cli.config as cfg
import hermes_cli.model_switch as ms
import hermes_cli.runtime_provider as rp
import utils
from gateway.session import sanitize_model_override
from hermes_state import SessionDB

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = "https://" + "lab.example.test/v1"
OTHER = "https://" + "other.example.test/v1"
MODEL = "saved-model"
PRODUCERS = ("cli", "gateway", "pending", "applied", "db")
IDENTITIES = ("alias", "bare", "branded", "unsafe", "credential", "collision",
              "unknown", "deleted", "config-failure")
CONSUMERS = ("cli-same", "cli-different", "gateway", "deferred")


def test_unresolved_marker_cannot_be_a_configured_runtime(config_boundary):
    config_boundary["config"] = {"providers": {"unresolved": {"api": OTHER}}}
    with pytest.raises(ValueError, match="restore.*route"):
        rp.resolve_runtime_provider(requested="unresolved")


def test_billing_writer_validates_alias(config_boundary, tmp_path):
    db = SessionDB(db_path=tmp_path / "billing.db")
    try:
        db.create_session(session_id="billing", source="cli", model=MODEL)
        db.update_session_billing_route("billing", provider="custom:" + PRIVATE, base_url=PRIVATE)
        assert PRIVATE not in json.dumps(db.get_session("billing"))
        assert db.get_session("billing")["billing_provider"] == "unresolved"
    finally:
        db.close()


@pytest.mark.parametrize("kind", ("slug", "same-url", "legacy", "disabled", "unknown-alias", "encoded", "secret-prefix", "secret-prefix-bare"))
def test_invalid_alias_owners_are_rejected(kind, config_boundary):
    entries = {"lab": {"api": PRIVATE}}
    state = config_boundary
    provider = "custom:lab"
    if kind == "slug":
        entries = {"Lab A": {"api": OTHER}, "lab-a": {"api": PRIVATE}}
    elif kind == "same-url":
        entries["other"] = {"api": PRIVATE}
    elif kind == "legacy":
        state["config"]["custom_providers"] = [{"name": "lab", "base_url": OTHER}]
    elif kind == "disabled":
        entries["lab"]["enabled"] = False
    elif kind == "unknown-alias":
        entries = {}
    else:
        name = "https%3A%2F%2Flab" if kind == "encoded" else "AKIA" + "A" * 16
        entries = {name: {"api": PRIVATE}}
        provider = "custom" if kind == "secret-prefix-bare" else "custom:" + name
    state["config"]["providers"] = entries
    route = utils.sanitize_persisted_model_config({"provider": provider, "base_url": PRIVATE})
    assert route == {"provider": "unresolved"}


def test_bare_cli_route_without_endpoint_is_rejected(config_boundary):
    obj = cli_agent("openrouter")
    written = {}
    obj.session_id = "route"
    obj._session_db = SimpleNamespace(
        update_session_model=lambda *a: None,
        patch_session_model_config=lambda _, payload: written.update(payload),
    )
    obj._persist_model_switch_to_session(ms.ModelSwitchResult(
        success=True, new_model=MODEL, target_provider="custom", base_url="",
    ))
    assert written["provider"] == "unresolved"


def test_catalog_route_real_resolution(config_boundary, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-route-key")
    public = "https://openrouter.ai/api/v1"
    route = utils.sanitize_persisted_model_config({"provider": "openrouter", "base_url": public})
    assert route == {"provider": "openrouter", "base_url": public}
    obj = cli_agent("custom:lab")
    obj._restore_session_model({"model": MODEL, "model_config": json.dumps(route)})
    assert obj.base_url == public and obj.api_key == "synthetic-route-key"



@pytest.fixture
def config_boundary(monkeypatch):
    for module in (utils, cli, gateway, rp, ms):
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
    state = {"config": {"providers": {"lab": {"api": PRIVATE, "api_key": "route-key"}}}}

    def load():
        if state.get("fail"):
            raise ValueError("synthetic configuration unavailable")
        return state["config"]

    monkeypatch.setattr(rp, "load_config", load)
    monkeypatch.setattr(cfg, "load_config", load)
    monkeypatch.setattr(cfg, "load_config_readonly", load)
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)
    monkeypatch.setattr(gateway, "_credential_pool_for_provider", lambda *a: None)
    return state


def agent():
    obj = SimpleNamespace(model="ambient-model", provider="openrouter", base_url=OTHER,
                          api_key="ambient-key", api_mode="chat_completions",
                          _session_init_model_config={})

    def switch(model, provider, api_key, base_url, api_mode):
        obj.model, obj.provider, obj.api_key, obj.base_url, obj.api_mode = (
            model, provider, api_key, base_url, api_mode)

    obj.switch_model = switch
    return obj


def cli_agent(provider):
    obj = object.__new__(cli.HermesCLI)
    obj.model = "ambient-model"
    obj.provider = obj.requested_provider = provider
    obj.base_url, obj.api_key, obj.api_mode = OTHER, "ambient-key", "chat_completions"
    obj.agent = None
    obj._console_print = lambda *a, **k: None
    obj._explicit_model_override = False
    return obj


def produce(producer, result, db):
    route = {"model": MODEL, "provider": result.target_provider, "base_url": result.base_url}
    if producer == "gateway":
        return sanitize_model_override(route)
    if producer in ("pending", "applied"):
        obj = agent()
        obj._session_db, obj.session_id = db, "route"
        ms.schedule_model_switch_after_compression(obj, result)
        if producer == "pending":
            config = json.loads(db.get_session("route")["model_config"])
            return config["pending_model_switch_after_compression"]
        assert ms.apply_model_switch_after_compression(obj) == "applied"
        assert obj.base_url == result.base_url  # memory route is not redacted
    elif producer == "cli":
        obj = cli_agent("openrouter")
        obj._session_db, obj.session_id = db, "route"
        obj._persist_model_switch_to_session(result)
    else:
        db.patch_session_model_config("route", route)
    return SessionDB.session_gateway_runtime(db.get_session("route"))


@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("identity", IDENTITIES)
@pytest.mark.parametrize("consumer", CONSUMERS)
def test_route_matrix(producer, identity, consumer, config_boundary, tmp_path):
    state = config_boundary
    provider, endpoint = "custom:lab", PRIVATE
    if identity in ("bare", "branded"):
        provider = "custom" if identity == "bare" else "openai"
    elif identity == "unsafe":
        state["config"] = {"providers": {PRIVATE: {"api": PRIVATE}}}
        provider = "custom:" + PRIVATE
    elif identity == "credential":
        state["config"] = {"providers": {"user:password@lab": {"api": PRIVATE}}}
        provider = "custom:user:password@lab"
    elif identity == "collision":
        state["config"] = {"providers": {"Lab A": {"api": OTHER}, "lab-a": {"api": PRIVATE}}}
        provider = "custom:lab-a"
    elif identity == "unknown":
        state["config"] = {}
        provider = "custom"
    result = ms.ModelSwitchResult(success=True, new_model=MODEL, target_provider=provider,
                                  base_url=endpoint, api_key="route-key", api_mode="chat_completions",
                                  reasoning_config={})
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="route", source="cli", model=MODEL)
        route = produce(producer, result, db)
        # Do not leak payloads into failure logs; structural assertions only.
        encoded = json.dumps(route)
        assert PRIVATE not in encoded
        assert "password" not in encoded
        if identity == "deleted":
            state["config"] = {}
        elif identity == "config-failure":
            state["fail"] = True
        valid = identity in ("alias", "bare", "branded")
        if consumer.startswith("cli"):
            obj = cli_agent("custom:lab" if consumer == "cli-same" else "openrouter")
            row = {"model": MODEL, "model_config": json.dumps(route)}
            if not valid:
                with pytest.raises(ValueError, match="restore.*route"):
                    obj._restore_session_model(row)
                assert not obj.api_key and not obj.base_url and obj.agent is None
            else:
                obj._restore_session_model(row)
                assert obj.base_url == PRIVATE and obj.api_key == "route-key"
                assert obj.provider == "custom:lab" and obj.model == MODEL
        elif consumer == "gateway":
            obj = object.__new__(gateway.GatewayRunner)
            obj._session_model_overrides = {}
            obj.session_store = SimpleNamespace(get_model_override=lambda _: {"model": MODEL, **route})
            if not valid:
                with pytest.raises(ValueError, match="restore.*route"):
                    obj._rehydrate_session_model_override("route")
            else:
                obj._rehydrate_session_model_override("route")
                model, runtime = obj._apply_session_model_override("route", "ambient", {
                    "provider": "openrouter", "base_url": OTHER, "api_key": "ambient-key",
                    "credential_pool": object()})
                assert runtime["credential_pool"] is None
                assert model == MODEL
                assert runtime["base_url"] == PRIVATE and runtime["api_key"] == "route-key"
                model, runtime = obj._resolve_session_agent_runtime(session_key="route", user_config={})
                assert model == MODEL
                assert runtime["base_url"] == PRIVATE and runtime["api_key"] == "route-key"
        else:
            obj = agent()
            obj._session_init_model_config = {"pending_model_switch_after_compression": {"model": MODEL, **route}}
            restored = ms.restore_model_switch_after_compression(obj)
            if not valid:
                assert restored is None
                assert ms.get_model_switch_after_compression(obj) is None
                assert obj.model == "ambient-model" and obj.base_url == OTHER
            else:
                assert restored is not None
                assert restored.base_url == PRIVATE and restored.api_key == "route-key"
                assert ms.apply_model_switch_after_compression(obj) == "applied"
                assert obj.model == MODEL and obj.base_url == PRIVATE and obj.api_key == "route-key"
    finally:
        db.close()
