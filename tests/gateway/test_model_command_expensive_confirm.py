"""Gateway typed ``/model <name>`` must route through the expensive-model
confirmation gate.

The pickers (Telegram/Discord inline keyboards, TUI, dashboard) confirm
expensive models via their own UI affordances; the typed text command
previously bypassed the guard entirely — a user typing
``/model openai/gpt-5.5-pro`` switched silently while the picker warned.
These tests pin the typed path:

- warning fires → handler returns the slash-confirm prompt, switch NOT applied
- confirm ("once") → switch applies (session override set)
- cancel → switch not applied, current model unchanged
- no warning (cheap model) → switch applies immediately, no prompt
"""

from types import SimpleNamespace

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.model_switch import (
    apply_model_switch_after_compression,
    get_model_switch_after_compression,
)


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._after_compression_model_switches = {}
    runner._running_agents = {}
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _fake_switch_result():
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="openai/gpt-5.5-pro",
        target_provider="openrouter",
        provider_changed=False,
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        provider_label="OpenRouter",
    )


def _fake_warning():
    return SimpleNamespace(
        message=(
            "!!! EXPENSIVE MODEL WARNING !!!\n"
            "openai/gpt-5.5-pro has known pricing above Hermes' safety threshold.\n"
            "did you mean to select openai/gpt-5.5?"
        ),
    )


def _setup_isolated_home(tmp_path, monkeypatch, *, warn):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"model": {"default": "old-model", "provider": "openrouter"}, "providers": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        (lambda *a, **kw: _fake_warning()) if warn else (lambda *a, **kw: None),
    )
    return cfg_path


@pytest.mark.asyncio
async def test_typed_model_expensive_confirm_once_applies_switch(tmp_path, monkeypatch):
    """Resolving the confirm with "once" applies the switch."""
    _setup_isolated_home(tmp_path, monkeypatch, warn=True)
    runner = _make_runner()
    runner._evict_cached_agent = lambda session_key: None

    captured = {}

    async def _fake_request_slash_confirm(**kwargs):
        captured.update(kwargs)
        return None  # buttons rendered

    runner._request_slash_confirm = _fake_request_slash_confirm

    await runner._handle_model_command(_make_event("/model openai/gpt-5.5-pro"))
    assert runner._session_model_overrides == {}

    reply = await captured["handler"]("once")

    assert "gpt-5.5-pro" in reply
    overrides = list(runner._session_model_overrides.values())
    assert len(overrides) == 1
    assert overrides[0]["model"] == "openai/gpt-5.5-pro"


@pytest.mark.asyncio
async def test_failed_inplace_swap_aborts_commit(tmp_path, monkeypatch):
    """A failed in-place agent swap must be a no-op, not a dead session.

    Regression for #50163: the resolution pipeline succeeds (valid model name)
    but the cached agent's ``switch_model()`` raises mid-conversation (bad key /
    unreachable URL). The agent rolls itself back to the old working model; the
    gateway must NOT then commit the broken model as a session override or evict
    the working cached agent — otherwise the next message rebuilds a dead agent
    and the conversation is lost.
    """
    _setup_isolated_home(tmp_path, monkeypatch, warn=False)
    runner = _make_runner()

    # Working cached agent whose in-place swap fails (and rolls itself back).
    class _FailingAgent:
        def __init__(self):
            self.model = "old-model"
            self.provider = "openrouter"

        def switch_model(self, **kwargs):
            # Mirrors agent_runtime_helpers.switch_model: the real method
            # restores old state then re-raises. We keep model unchanged.
            raise RuntimeError("connection refused: bad base_url")

    import threading

    agent = _FailingAgent()
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    session_key = runner._session_key_for_source(_make_event("/model x").source)
    runner._agent_cache[session_key] = [agent, None]
    runner._session_db = None

    evicted = []
    runner._evict_cached_agent = lambda sk: evicted.append(sk)

    result = await runner._handle_model_command(_make_event("/model openai/gpt-5.5-pro"))

    # Error surfaced to the user, not a success confirmation.
    assert result is not None
    assert "failed" in result.lower()
    # The broken switch must NOT have been committed anywhere.
    assert runner._session_model_overrides == {}
    # The working cached agent must NOT have been evicted.
    assert evicted == []
    # The agent stayed on its old model (rolled back).
    assert agent.model == "old-model"


@pytest.mark.asyncio
async def test_gateway_deferred_switch_waits_for_compression_boundary(
    tmp_path, monkeypatch
):
    _setup_isolated_home(tmp_path, monkeypatch, warn=False)
    runner = _make_runner()

    class _Agent:
        def __init__(self):
            self.model = "old-model"
            self.provider = "openrouter"
            self.calls = []

        def switch_model(
            self,
            new_model,
            new_provider,
            api_key="",
            base_url="",
            api_mode="",
        ):
            self.calls.append((new_model, new_provider))
            self.model = new_model
            self.provider = new_provider

    import threading

    agent = _Agent()
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_db = None
    runner._evict_cached_agent = lambda _key: None
    class _Store:
        def __init__(self):
            self.writes = []

        def set_model_override(self, key, override):
            self.writes.append((key, dict(override)))

        def get_model_override(self, key):
            return dict(self.writes[-1][1]) if self.writes[-1][0] == key else None

    store = _Store()
    runner.session_store = store
    event = _make_event(
        "/model openai/gpt-5.5-pro --after-compression --provider openrouter"
    )
    session_key = runner._session_key_for_source(event.source)
    runner._agent_cache[session_key] = (agent, None)

    reply = await runner._handle_model_command(event)

    assert "successful compression" in reply
    assert agent.calls == []
    assert runner._session_model_overrides == {}
    assert runner._after_compression_model_switches[session_key].new_model == (
        "openai/gpt-5.5-pro"
    )
    assert get_model_switch_after_compression(agent) is not None

    assert apply_model_switch_after_compression(agent) == "applied"
    assert agent.calls == [("openai/gpt-5.5-pro", "openrouter")]
    assert session_key not in runner._after_compression_model_switches
    assert runner._session_model_overrides[session_key]["model"] == (
        "openai/gpt-5.5-pro"
    )
    assert store.writes == [
        (
            session_key,
            {
                "model": "openai/gpt-5.5-pro",
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
            },
        )
    ]
    restarted = _make_runner()
    restarted.session_store = store
    restarted._rehydrate_session_model_override(session_key)
    assert restarted._session_model_overrides[session_key]["model"] == (
        "openai/gpt-5.5-pro"
    )


@pytest.mark.asyncio
async def test_gateway_deferred_persistence_failure_rolls_back_and_keeps_pending(
    tmp_path, monkeypatch
):
    from hermes_state import SessionDB

    _setup_isolated_home(tmp_path, monkeypatch, warn=False)
    runner = _make_runner()

    class _Agent:
        def __init__(self, db):
            self.model = "old-model"
            self.provider = "openrouter"
            self.base_url = "https://old.example/v1"
            self.api_key = "old-key"
            self.api_mode = "chat_completions"
            self.session_id = "gateway-deferred-failure"
            self._session_db = db
            self._session_init_model_config = {"provider": "openrouter"}
            self._cached_system_prompt = "old prompt"

        def switch_model(
            self,
            new_model,
            new_provider,
            api_key="",
            base_url="",
            api_mode="",
        ):
            self.model = new_model
            self.provider = new_provider
            self.api_key = api_key
            self.base_url = base_url
            self.api_mode = api_mode
            self._cached_system_prompt = "new prompt"

    class _FailingStore:
        def set_model_override(self, _key, _override):
            raise OSError("injected override persistence failure")

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        "gateway-deferred-failure",
        "cli",
        model="old-model",
        model_config={"provider": "openrouter"},
        system_prompt="old prompt",
    )
    agent = _Agent(db)
    runner._agent_cache = {}
    runner._agent_cache_lock = __import__("threading").Lock()
    runner._session_db = None
    runner.session_store = _FailingStore()
    runner._evict_cached_agent = lambda _key: None
    event = _make_event(
        "/model openai/gpt-5.5-pro --after-compression --provider openrouter"
    )
    session_key = runner._session_key_for_source(event.source)
    runner._agent_cache[session_key] = (agent, None)

    try:
        reply = await runner._handle_model_command(event)
        assert "successful compression" in reply

        assert apply_model_switch_after_compression(agent) == "failed"
        row = db.get_session("gateway-deferred-failure")
        assert (agent.model, row["model"]) == ("old-model", "old-model")
        assert get_model_switch_after_compression(agent) is not None
        state = runner._session_state(session_key).conversation
        assert state.after_compression_model_switch is not None
        assert state.model_override is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_gateway_model_picker_exposes_pending_deferred_route(
    tmp_path, monkeypatch
):
    _setup_isolated_home(tmp_path, monkeypatch, warn=False)
    runner = _make_runner()
    captured = {}

    class _Picker:
        async def send_model_picker(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(success=True)

    picker = _Picker()
    runner._normalize_source_for_session_key = lambda source: source
    runner._adapter_for_source = lambda _source: picker
    runner._thread_metadata_for_source = lambda *_args: {}
    runner._reply_anchor_for_event = lambda _event: None
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers",
        lambda **_kwargs: [
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "is_current": True,
                "models": ["old-model"],
                "total_models": 1,
            }
        ],
    )
    event = _make_event("/model")
    session_key = runner._session_key_for_source(event.source)
    runner._after_compression_model_switches[session_key] = SimpleNamespace(
        new_model="next-model",
        target_provider="anthropic",
        provider_label="Anthropic",
    )

    assert await runner._handle_model_command(event) is None
    assert "next-model" in captured["pending_model_switch"]
    assert "Anthropic" in captured["pending_model_switch"]
