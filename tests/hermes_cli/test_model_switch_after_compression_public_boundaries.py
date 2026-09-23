"""Public-surface lifecycle coverage for deferred model switches.

All provider resolution is synthetic.  The tests still cross the real Gateway,
JSON-RPC, and CLI command dispatchers and the real compression commit boundary.
"""

from __future__ import annotations

import json
import os
import threading
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_cli.model_switch import (
    ModelSwitchResult,
    get_model_switch_after_compression,
    schedule_model_switch_after_compression,
)
from hermes_state import SessionDB


OLD_MODEL = "old/model"
OLD_PROVIDER = "openrouter"
NEW_MODEL = "new/model"
NEW_PROVIDER = "custom:synthetic"
SECRET = "synthetic-secret"
BASE_URL = "http://127.0.0.1:9/v1"
LOW = {"enabled": True, "effort": "low"}


def _resolved(**kwargs) -> ModelSwitchResult:
    model = kwargs.get("raw_input") or OLD_MODEL
    provider = kwargs.get("explicit_provider") or OLD_PROVIDER
    return ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider=provider,
        api_key=SECRET,
        base_url=BASE_URL,
        api_mode="chat_completions",
        provider_label="Synthetic",
        reasoning_config={"enabled": True, "effort": "medium"},
    )


def _compression_agent(tmp_path, session_id: str, *, platform: str = "cli", db=None):
    db = db or SessionDB(db_path=tmp_path / "state.db")
    if db.get_session(session_id) is None:
        db.create_session(session_id, source=platform, model=OLD_MODEL)
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model=OLD_MODEL,
            provider=OLD_PROVIDER,
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            platform=platform,
            skip_context_files=True,
            skip_memory=True,
        )
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "assistant", "content": "summary acknowledged"},
        {"role": "user", "content": "live tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    calls = []

    def switch_model(new_model, new_provider, api_key, base_url, api_mode, **_kwargs):
        calls.append((new_model, new_provider, api_key, base_url, api_mode))
        agent.model = new_model
        agent.provider = new_provider
        agent.api_key = api_key
        agent.base_url = base_url
        agent.api_mode = api_mode
        pending_reasoning = getattr(agent, "_deferred_model_switch_reasoning_config", None)
        if pending_reasoning is not None:
            agent.reasoning_config = dict(pending_reasoning)

    agent.switch_model = switch_model
    return db, agent, calls


def _compress(agent, *, abort: bool = False):
    agent.context_compressor._last_compress_aborted = abort
    if abort:
        agent.context_compressor._last_summary_error = "synthetic abort"
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": f"message-{i}"} for i in range(8)
        ]
    else:
        agent.context_compressor._last_summary_error = None
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "summary acknowledged"},
            {"role": "user", "content": "live tail"},
        ]
    messages = [{"role": "user", "content": f"message-{i}"} for i in range(8)]
    return agent._compress_context(messages, "system", approx_tokens=120_000)


def _assert_secret_free_pending(db, session_id: str) -> None:
    stored = json.loads(db.get_session(session_id)["model_config"])
    pending = stored["pending_model_switch_after_compression"]
    assert pending == {
        "api_mode": "chat_completions",
        "model": NEW_MODEL,
        "provider": NEW_PROVIDER,
        "reasoning_config": LOW,
    }
    assert SECRET not in json.dumps(stored)
    assert BASE_URL not in json.dumps(stored)


def _cold_rebuilt_agent(tmp_path, monkeypatch, session_id: str, *, platform: str):
    db, agent, _calls = _compression_agent(tmp_path, session_id, platform=platform)
    pending = _resolved(raw_input=NEW_MODEL, explicit_provider=NEW_PROVIDER)
    pending.reasoning_config = dict(LOW)
    schedule_model_switch_after_compression(agent, pending)
    _assert_secret_free_pending(db, session_id)
    db.close()

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: _resolved(**kwargs))
    reopened = SessionDB(db_path=tmp_path / "state.db")
    _db2, rebuilt, calls = _compression_agent(
        tmp_path, session_id, platform=platform, db=reopened
    )
    assert get_model_switch_after_compression(rebuilt) is not None
    return reopened, rebuilt, calls


def test_gateway_public_command_crosses_real_compression_commit(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore

    db, agent, calls = _compression_agent(tmp_path, "gateway-session", platform="telegram")
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._sessions = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._pending_model_notes = {}
    runner._running_agents = {}
    runner._session_db = db
    sessions_dir = tmp_path / "gateway-routing"
    runner.session_store = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    runner._normalize_source_for_session_key = lambda source: source
    runner._model_selection_guard_reply = AsyncMock(return_value=(False, None))
    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="user", chat_id="chat", chat_type="dm"
    )
    session_key = runner._session_key_for_source(source)
    runner.session_store.get_or_create_session(source)
    runner._agent_cache[session_key] = (agent, "signature", 0, "gateway-session")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda **_kwargs: {
        "model": {"default": OLD_MODEL, "provider": OLD_PROVIDER}
    })
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: _resolved(**kwargs))

    reply = asyncio.run(runner._gateway_idle_command_handlers()["model"](
        MessageEvent(
            text=(f"/model {NEW_MODEL} --provider {NEW_PROVIDER} "
                  "--after-compression --reasoning low"),
            source=source,
        )
    ))

    assert "scheduled after the next successful compression" in reply
    assert (agent.model, agent.provider, calls) == (OLD_MODEL, OLD_PROVIDER, [])
    _assert_secret_free_pending(db, "gateway-session")

    _compress(agent)

    state = runner._session_state(session_key).conversation
    assert calls == [(NEW_MODEL, NEW_PROVIDER, SECRET, BASE_URL, "chat_completions")]
    assert state.after_compression_model_switch is None
    assert state.model_override["reasoning_config"] == LOW
    rebuilt_model, rebuilt_runtime = runner._apply_session_model_override(
        session_key, OLD_MODEL, {"provider": OLD_PROVIDER}
    )
    assert rebuilt_model == NEW_MODEL
    assert {key: rebuilt_runtime[key] for key in ("provider", "api_key", "base_url", "api_mode")} == {
        "provider": NEW_PROVIDER,
        "api_key": SECRET,
        "base_url": BASE_URL,
        "api_mode": "chat_completions",
    }
    reloaded_store = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    assert reloaded_store.get_model_override(session_key) == {
        "model": NEW_MODEL,
        "provider": NEW_PROVIDER,
        "base_url": BASE_URL,
    }


@pytest.mark.parametrize(
    ("value", "expected_model", "expected_provider"),
    [
        (f"{NEW_MODEL} --provider {NEW_PROVIDER} --after-compression --reasoning low",
         NEW_MODEL, NEW_PROVIDER),
        ("--after-compression --reasoning low", OLD_MODEL, OLD_PROVIDER),
    ],
)
def test_tui_jsonrpc_command_crosses_real_compression_commit(
    tmp_path, monkeypatch, value, expected_model, expected_provider
):
    from tui_gateway import server

    db, agent, calls = _compression_agent(tmp_path, "tui-session", platform="tui")
    session = {
        "agent": agent,
        "session_key": "tui-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
    }
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: _resolved(**kwargs))
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_append_model_switch_marker", lambda *_args, **_kwargs: None)
    with patch.dict(server._sessions, {"public-tui": session}, clear=True):
        response = server.dispatch({
            "jsonrpc": "2.0",
            "id": "switch",
            "method": "config.set",
            "params": {"session_id": "public-tui", "key": "model", "value": value,
                       "confirm_expensive_model": True},
        })
        assert response["result"]["scope"] == "after_compression"
        assert (agent.model, agent.provider, calls) == (OLD_MODEL, OLD_PROVIDER, [])

        _compress(agent)

        assert calls == [(expected_model, expected_provider, SECRET, BASE_URL, "chat_completions")]
        assert "after_compression_model_switch" not in session
        assert session["model_override"]["model"] == expected_model
        assert session["model_override"]["provider"] == expected_provider
        assert session["model_override"]["reasoning_config"] == LOW
        rebuild = server._deferred_build_agent_kwargs(
            {**session, "resume_session_id": "tui-session"}, db
        )
        assert rebuild["model_override"]["reasoning_config"] == LOW


def _public_cli(agent):
    from cli import HermesCLI

    cli = object.__new__(HermesCLI)
    for name, value in {
        "model": OLD_MODEL,
        "provider": OLD_PROVIDER,
        "requested_provider": OLD_PROVIDER,
        "api_key": "test-key",
        "_explicit_api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "_explicit_base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "_app": None,
        "_pending_resume_sessions": None,
        "_pending_one_turn_model_restore": None,
        "_pending_model_switch_note": None,
        "session_id": getattr(agent, "session_id", None),
        "agent": agent,
    }.items():
        setattr(cli, name, value)
    cli._confirm_expensive_model_switch = lambda _result: True
    return cli


@pytest.mark.parametrize(
    ("command", "expected_model", "expected_provider"),
    [
        (f"/model {NEW_MODEL} --provider {NEW_PROVIDER} --after-compression --reasoning low",
         NEW_MODEL, NEW_PROVIDER),
        ("/model --after-compression --reasoning low", OLD_MODEL, OLD_PROVIDER),
    ],
)
def test_cli_public_dispatch_crosses_real_compression_commit(
    tmp_path, monkeypatch, command, expected_model, expected_provider
):
    from cli import HermesCLI

    _db, agent, calls = _compression_agent(tmp_path, "cli-session")
    cli = _public_cli(agent)
    monkeypatch.setattr("cli._cprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            user_providers=None,
            custom_providers=None,
            with_overrides=lambda **_kwargs: SimpleNamespace(
                user_providers=None, custom_providers=None
            ),
        ),
    )
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: _resolved(**kwargs))

    assert HermesCLI.process_command(cli, command) is True
    assert (cli.model, cli.provider, agent.model, agent.provider, calls) == (
        OLD_MODEL, OLD_PROVIDER, OLD_MODEL, OLD_PROVIDER, []
    )

    _compress(agent)

    assert calls == [(expected_model, expected_provider, SECRET, BASE_URL, "chat_completions")]
    assert (cli.model, cli.provider, cli.reasoning_config) == (
        expected_model, expected_provider, LOW
    )


def test_public_schedule_survives_abort_and_apply_failure_then_applies_once(
    tmp_path, monkeypatch
):
    from cli import HermesCLI

    db, agent, calls = _compression_agent(tmp_path, "retry-session")
    cli = _public_cli(agent)
    monkeypatch.setattr("cli._cprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            user_providers=None, custom_providers=None,
            with_overrides=lambda **_kwargs: SimpleNamespace(
                user_providers=None, custom_providers=None),
        ),
    )
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: _resolved(**kwargs))
    command = (f"/model {NEW_MODEL} --provider {NEW_PROVIDER} "
               "--after-compression --reasoning low")

    HermesCLI.process_command(cli, command)
    _compress(agent, abort=True)
    assert calls == []
    assert get_model_switch_after_compression(agent) is not None
    _assert_secret_free_pending(db, "retry-session")

    original_switch = agent.switch_model
    fail_once = True

    def failing_switch(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("synthetic apply failure")
        return original_switch(*args, **kwargs)

    agent.switch_model = failing_switch
    _compress(agent)
    assert calls == []
    assert (agent.model, agent.provider) == (OLD_MODEL, OLD_PROVIDER)
    assert get_model_switch_after_compression(agent) is not None
    _assert_secret_free_pending(db, "retry-session")

    _compress(agent)
    assert calls == [(NEW_MODEL, NEW_PROVIDER, SECRET, BASE_URL, "chat_completions")]
    assert get_model_switch_after_compression(agent) is None
    assert json.loads(db.get_session("retry-session")["model_config"]).get(
        "pending_model_switch_after_compression"
    ) is None


def test_cold_sessiondb_recreation_restores_secret_free_pending_intent(
    tmp_path, monkeypatch
):
    from cli import HermesCLI
    from hermes_cli import config as config_module

    db, agent, _calls = _compression_agent(tmp_path, "cold-session")
    cli = _public_cli(agent)
    monkeypatch.setattr("cli._cprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            user_providers=None, custom_providers=None,
            with_overrides=lambda **_kwargs: SimpleNamespace(
                user_providers=None, custom_providers=None),
        ),
    )
    resolution_calls = []

    def resolve_from_current_config(**kwargs):
        resolution_calls.append(kwargs)
        return _resolved(**kwargs)

    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", resolve_from_current_config
    )
    HermesCLI.process_command(
        cli,
        f"/model {NEW_MODEL} --provider {NEW_PROVIDER} --after-compression --reasoning low",
    )
    _assert_secret_free_pending(db, "cold-session")
    db.close()

    resolution_calls.clear()
    current_config = {
        "providers": {
            "synthetic-current": {
                "base_url": BASE_URL,
                "api_key": SECRET,
                "models": {NEW_MODEL: {}},
            }
        }
    }
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: current_config)
    reopened = SessionDB(db_path=tmp_path / "state.db")
    _db2, rebuilt, calls = _compression_agent(
        tmp_path, "cold-session", platform="cli", db=reopened
    )
    pending = get_model_switch_after_compression(rebuilt)
    assert pending is not None
    assert (pending.new_model, pending.target_provider, pending.reasoning_config) == (
        NEW_MODEL, NEW_PROVIDER, LOW
    )
    assert resolution_calls == []
    assert (rebuilt.model, rebuilt.provider, calls) == (OLD_MODEL, OLD_PROVIDER, [])
    _compress(rebuilt)
    assert len(resolution_calls) == 1
    assert resolution_calls[0]["validate_live"] is True
    assert resolution_calls[0]["user_providers"] is current_config["providers"]
    assert calls == [(NEW_MODEL, NEW_PROVIDER, SECRET, BASE_URL, "chat_completions")]


def test_cold_restore_never_reaches_models_dev_network(tmp_path, monkeypatch):
    from agent import models_dev

    db, agent, _calls = _compression_agent(tmp_path, "offline-cold-session")
    pending = _resolved(raw_input="synthetic/offline", explicit_provider=OLD_PROVIDER)
    pending.reasoning_config = dict(LOW)
    schedule_model_switch_after_compression(agent, pending)
    db.close()

    monkeypatch.setattr(models_dev, "_models_dev_cache", {})
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", 0)
    monkeypatch.setattr(models_dev, "_models_dev_retry_after", 0)
    monkeypatch.setattr(models_dev, "_load_disk_cache", lambda: {})

    network_calls = []

    def reject_network(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("cold restore attempted models.dev network I/O")

    monkeypatch.setattr(models_dev.requests, "get", reject_network)
    reopened = SessionDB(db_path=tmp_path / "state.db")
    _db2, rebuilt, calls = _compression_agent(
        tmp_path, "offline-cold-session", platform="cli", db=reopened
    )

    restored = get_model_switch_after_compression(rebuilt)
    assert restored is not None
    assert (restored.new_model, restored.target_provider) == (
        "synthetic/offline", OLD_PROVIDER
    )
    assert network_calls == []
    assert calls == []


def test_gateway_cold_rebuild_adopts_pending_and_persists_applied_route(
    tmp_path, monkeypatch
):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore

    db, rebuilt, calls = _cold_rebuilt_agent(
        tmp_path, monkeypatch, "gateway-cold-session", platform="telegram"
    )
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._sessions = {}
    runner._pending_model_notes = {}
    sessions_dir = tmp_path / "gateway-cold-routing"
    runner.session_store = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="user", chat_id="cold-chat", chat_type="dm"
    )
    session_key = runner._session_key_for_source(source)
    runner.session_store.get_or_create_session(source)

    runner._attach_model_switch_after_compression(session_key, rebuilt)
    assert runner._session_state(
        session_key
    ).conversation.after_compression_model_switch is get_model_switch_after_compression(rebuilt)

    _compress(rebuilt)

    state = runner._session_state(session_key).conversation
    assert calls == [(NEW_MODEL, NEW_PROVIDER, SECRET, BASE_URL, "chat_completions")]
    assert state.after_compression_model_switch is None
    assert state.model_override["reasoning_config"] == LOW
    assert SessionStore(sessions_dir=sessions_dir, config=GatewayConfig()).get_model_override(
        session_key
    ) == {"model": NEW_MODEL, "provider": NEW_PROVIDER, "base_url": BASE_URL}
    db.close()


def test_tui_cold_rebuild_adopts_pending_and_updates_rebuild_surface(
    tmp_path, monkeypatch
):
    from tui_gateway import server

    db, rebuilt, calls = _cold_rebuilt_agent(
        tmp_path, monkeypatch, "tui-cold-session", platform="tui"
    )
    session = {
        "session_key": "tui-cold-session",
        "resume_session_id": "tui-cold-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
    }
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_append_model_switch_marker", lambda *_args, **_kwargs: None)

    server._attach_built_agent("public-tui-cold", session, rebuilt)
    assert session["after_compression_model_switch"] is get_model_switch_after_compression(
        rebuilt
    )

    _compress(rebuilt)

    assert calls == [(NEW_MODEL, NEW_PROVIDER, SECRET, BASE_URL, "chat_completions")]
    assert "after_compression_model_switch" not in session
    assert session["model_override"]["reasoning_config"] == LOW
    rebuilt_kwargs = server._deferred_build_agent_kwargs(session, db)
    assert rebuilt_kwargs["model_override"]["model"] == NEW_MODEL
    assert rebuilt_kwargs["model_override"]["provider"] == NEW_PROVIDER


@pytest.mark.parametrize("provider", ["openai-codex", "llamacpp"])
def test_cold_agent_restore_defers_every_network_capable_route_lookup(
    tmp_path, monkeypatch, provider
):
    """A public cold agent build restores intent without refreshing or probing."""
    session_id = f"offline-{provider}"
    db, agent, _calls = _compression_agent(tmp_path, session_id)
    pending = _resolved(raw_input=f"{provider}/model", explicit_provider=provider)
    pending.reasoning_config = dict(LOW)
    schedule_model_switch_after_compression(agent, pending)
    db.close()

    network_calls = []

    def reject_network(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError(f"cold restore attempted {provider} network I/O")

    if provider == "openai-codex":
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_codex_runtime_credentials",
            reject_network,
        )
    else:
        monkeypatch.setattr(
            "hermes_cli.local_runtime.endpoint.resolve_llamacpp_endpoint",
            reject_network,
        )

    reopened = SessionDB(db_path=tmp_path / "state.db")
    _db2, rebuilt, calls = _compression_agent(
        tmp_path, session_id, platform="cli", db=reopened
    )

    restored = get_model_switch_after_compression(rebuilt)
    assert restored is not None
    assert (restored.new_model, restored.target_provider) == (
        f"{provider}/model", provider
    )
    assert network_calls == []
    assert calls == []
    reopened.close()


class _RestoredAgent:
    model = OLD_MODEL
    provider = OLD_PROVIDER
    requested_provider = OLD_PROVIDER
    api_key = "old-key"
    base_url = "https://openrouter.ai/api/v1"
    api_mode = "chat_completions"

    def __init__(self, pending):
        self._session_init_model_config = {}
        self.calls = []
        schedule_model_switch_after_compression(self, pending)

    def switch_model(self, new_model, new_provider, api_key, base_url, api_mode):
        self.calls.append((new_model, new_provider, api_key, base_url, api_mode))
        self.model, self.provider = new_model, new_provider
        self.api_key, self.base_url, self.api_mode = api_key, base_url, api_mode


def _pending_result(model=NEW_MODEL, provider=NEW_PROVIDER):
    result = _resolved(raw_input=model, explicit_provider=provider)
    result.reasoning_config = dict(LOW)
    return result


def test_tui_public_eager_resume_attaches_restored_switch_before_publish(monkeypatch):
    from hermes_cli.model_switch import apply_model_switch_after_compression
    from tui_gateway import server

    class ResumeDB:
        def get_session(self, _session_id):
            return {"id": "resume-key", "message_count": 0, "cwd": None}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _session_id):
            return None

        def get_resume_conversations(self, _session_id):
            return [], []

        def get_ancestor_display_prefix(self, _session_id):
            return []

    agent = _RestoredAgent(_pending_result())
    db = ResumeDB()
    monkeypatch.setattr(server, "_profile_session_db", lambda _home: (db, False))
    monkeypatch.setattr(server, "_make_agent_in_context", lambda *_a, **_kw: agent)
    monkeypatch.setattr(server, "_profile_build_scope", lambda _home: nullcontext())
    monkeypatch.setattr(server, "_hydrate_session_cwd", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *_a, **_kw: False)
    monkeypatch.setattr(server, "_start_session_services", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_append_model_switch_marker", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_session_info", lambda a, _s=None: {"model": a.model})

    with patch.dict(server._sessions, {}, clear=True):
        response = server.handle_request({
            "id": "resume",
            "method": "session.resume",
            "params": {"session_id": "resume-key", "eager_build": True},
        })
        assert "error" not in response
        session = server._sessions[response["result"]["session_id"]]

        assert apply_model_switch_after_compression(agent) == "applied"
        assert "after_compression_model_switch" not in session
        assert session["model_override"]["model"] == NEW_MODEL


def test_tui_bot_capability_rebuild_attaches_restored_switch(monkeypatch):
    from hermes_cli.model_switch import apply_model_switch_after_compression
    from tui_gateway import server

    old_agent = SimpleNamespace(_session_title_hint="Bot Chat", _session_db=None)
    new_agent = _RestoredAgent(_pending_result())
    session = {
        "agent": old_agent,
        "session_key": "bot-key",
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": None,
        "source": "tui",
        "cwd": "/tmp",
        "bot_caps_seen": "before",
    }
    monkeypatch.setattr("tools.bot_mode_probe.capability_fingerprint", lambda _home: "after")
    monkeypatch.setattr(server, "_make_agent", lambda *_a, **_kw: new_agent)
    monkeypatch.setattr(server, "_config_model_target", lambda: (OLD_MODEL, OLD_PROVIDER))
    monkeypatch.setattr(server, "_set_session_context", lambda *_a, **_kw: ())
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_append_model_switch_marker", lambda *_a, **_kw: None)

    server._sync_bot_capabilities("bot-public", session)

    assert session["agent"] is new_agent
    assert apply_model_switch_after_compression(new_agent) == "applied"
    assert "after_compression_model_switch" not in session
    assert session["model_override"]["model"] == NEW_MODEL


def _clearing_switch(agent, calls):
    def commit(new_model, new_provider, api_key, base_url, api_mode, **_kwargs):
        from hermes_cli.model_switch import clear_model_switch_after_compression

        calls.append((new_model, new_provider, api_key, base_url, api_mode))
        agent.model, agent.provider = new_model, new_provider
        agent.api_key, agent.base_url, agent.api_mode = api_key, base_url, api_mode
        if not getattr(agent, "_applying_model_switch_after_compression", False):
            clear_model_switch_after_compression(agent)

    return commit


def test_tui_public_immediate_switch_cancels_host_and_durable_deferred_route(
    tmp_path, monkeypatch
):
    from tui_gateway import server

    db, agent, calls = _compression_agent(tmp_path, "tui-cancel", platform="tui")
    agent.switch_model = _clearing_switch(agent, calls)
    session = {
        "agent": agent,
        "session_key": "tui-cancel",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
    }

    def resolve(**kwargs):
        return _pending_result(kwargs["raw_input"], kwargs["explicit_provider"])

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", resolve)
    for name in (
        "_restart_slash_worker", "_persist_live_session_runtime",
        "_persist_live_session_system_prompt", "_append_model_switch_marker",
        "_emit_session_info", "_emit",
    ):
        monkeypatch.setattr(server, name, lambda *_a, **_kw: None)

    with patch.dict(server._sessions, {"public-tui": session}, clear=True):
        deferred = server.dispatch({
            "jsonrpc": "2.0", "id": "deferred", "method": "config.set",
            "params": {"session_id": "public-tui", "key": "model",
                       "value": f"{NEW_MODEL} --provider {NEW_PROVIDER} --after-compression",
                       "confirm_expensive_model": True},
        })
        assert deferred["result"]["scope"] == "after_compression"
        immediate = server.dispatch({
            "jsonrpc": "2.0", "id": "immediate", "method": "config.set",
            "params": {"session_id": "public-tui", "key": "model",
                       "value": "current/model --provider custom:current",
                       "confirm_expensive_model": True},
        })
        assert immediate.get("result", {}).get("value") == "current/model", immediate
        assert "after_compression_model_switch" not in session
        stored = json.loads(db.get_session("tui-cancel")["model_config"])
        assert "pending_model_switch_after_compression" not in stored

        _db2, rebuilt, _rebuilt_calls = _compression_agent(
            tmp_path, "tui-cancel", platform="tui", db=db
        )
        assert get_model_switch_after_compression(rebuilt) is None
    db.close()


@pytest.mark.parametrize("cached_at_immediate", [True, False])
def test_gateway_public_immediate_switch_cancels_host_and_durable_deferred_route(
    tmp_path, monkeypatch, cached_at_immediate
):
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._sessions = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._pending_model_notes = {}
    runner._running_agents = {}
    runner._session_model_overrides = {}
    sessions_dir = tmp_path / "gateway-cancel-routing"
    runner.session_store = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    runner._normalize_source_for_session_key = lambda source: source
    runner._model_selection_guard_reply = AsyncMock(return_value=(False, None))
    runner._release_evicted_agent_soft = lambda _agent: None
    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="user", chat_id="cancel-chat", chat_type="dm"
    )
    session_key = runner._session_key_for_source(source)
    entry = runner.session_store.get_or_create_session(source)
    db, agent, calls = _compression_agent(
        tmp_path, entry.session_id, platform="telegram"
    )
    class AsyncDB:
        async def update_session_model(self, *args, **kwargs):
            return db.update_session_model(*args, **kwargs)

        async def get_session(self, *args, **kwargs):
            return db.get_session(*args, **kwargs)

        async def update_session_meta(self, *args, **kwargs):
            return db.update_session_meta(*args, **kwargs)

    runner._session_db = AsyncDB()
    agent.switch_model = _clearing_switch(agent, calls)
    runner._agent_cache[session_key] = (agent, "signature", 0, entry.session_id)

    def resolve(**kwargs):
        return _pending_result(kwargs["raw_input"], kwargs["explicit_provider"])

    async def no_context(*_args, **_kwargs):
        return None

    monkeypatch.setattr("gateway.run._load_gateway_config", lambda **_kw: {
        "model": {"default": OLD_MODEL, "provider": OLD_PROVIDER}
    })
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", resolve)
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length_async", no_context
    )

    handler = runner._gateway_idle_command_handlers()["model"]
    deferred = asyncio.run(handler(MessageEvent(
        text=f"/model {NEW_MODEL} --provider {NEW_PROVIDER} --after-compression",
        source=source,
    )))
    assert "scheduled after the next successful compression" in deferred
    if not cached_at_immediate:
        runner._agent_cache.pop(session_key)
    immediate = asyncio.run(handler(MessageEvent(
        text="/model current/model --provider custom:current",
        source=source,
    )))
    assert "current/model" in immediate

    state = runner._session_state(session_key).conversation
    assert state.after_compression_model_switch is None
    stored = json.loads(db.get_session(entry.session_id)["model_config"])
    assert "pending_model_switch_after_compression" not in stored
    _db2, rebuilt, _rebuilt_calls = _compression_agent(
        tmp_path, entry.session_id, platform="telegram", db=db
    )
    assert get_model_switch_after_compression(rebuilt) is None
    db.close()


def test_tui_public_tool_rebuild_clears_deferred_route_at_conversation_boundary(
    tmp_path, monkeypatch
):
    from tui_gateway import server

    db, agent, _calls = _compression_agent(tmp_path, "tui-reset", platform="tui")
    schedule_model_switch_after_compression(agent, _pending_result())
    replacement = SimpleNamespace(_session_db=db, _owns_session_db=False)
    session = {
        "agent": agent,
        "session_key": "tui-reset",
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": None,
        "source": "tui",
        "cwd": "/tmp",
        "_queued_prompt_generation": 0,
    }
    config = SimpleNamespace(
        load_config=lambda: {}, save_config=lambda _cfg: None,
    )
    tools_config = SimpleNamespace(
        CONFIGURABLE_TOOLSETS=(("terminal", "Terminal", ""),),
        _get_mcp_servers=lambda: {},
        server_configs_with_sources=lambda _servers: ({}, {}),
        _get_plugin_toolset_keys=lambda: set(),
        _apply_toolset_change=lambda *_a, **_kw: None,
        _apply_mcp_change=lambda *_a, **_kw: set(),
        _get_platform_tools=lambda *_a, **_kw: set(),
    )
    monkeypatch.setattr(
        server, "_tools_mod",
        lambda name: config if name == "hermes_cli.config" else tools_config,
    )
    monkeypatch.setattr(server, "_make_agent", lambda *_a, **_kw: replacement)
    monkeypatch.setattr(server, "_config_model_target", lambda: (OLD_MODEL, OLD_PROVIDER))
    monkeypatch.setattr(server, "_set_session_context", lambda *_a, **_kw: ())
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: False)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "off")
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_kw: {})
    monkeypatch.setattr(server, "_emit", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_kw: None)

    with patch.dict(server._sessions, {"public-reset": session}, clear=True):
        response = server.handle_request({
            "id": "tools", "method": "tools.configure",
            "params": {"session_id": "public-reset", "action": "disable",
                       "names": ["terminal"]},
        })

    assert "error" not in response
    assert "after_compression_model_switch" not in session
    stored = json.loads(db.get_session("tui-reset")["model_config"])
    assert "pending_model_switch_after_compression" not in stored
    db.close()
