import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
    )
    runner.adapters = {}
    runner._pending_native_image_paths_by_session = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="273403055",
        chat_type="dm",
        user_id="42",
        user_name="Maxim",
    )


def _image_event(text: str = "look") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.PHOTO,
        source=_source(),
        media_urls=["/tmp/cashback.png"],
        media_types=["image/png"],
    )


def _auto_config() -> dict:
    return {
        "agent": {"image_input_mode": "auto"},
        "auxiliary": {"vision": {"provider": "auto", "model": "", "base_url": ""}},
        "model": {"provider": "xiaomi", "default": "mimo-v2.5-pro"},
    }


def test_pre_turn_named_custom_provider_identity_selects_vision_override(monkeypatch):
    """Gateway preprocessing must use the name retained by runtime resolution."""
    runner = _make_runner()
    cfg = {
        "agent": {"image_input_mode": "auto"},
        "model": {"provider": "default-proxy", "default": "shared-model"},
        "custom_providers": [
            {
                "name": "default-proxy",
                "models": {"shared-model": {"supports_vision": False}},
            },
            {
                "name": "vision-provider",
                "models": {"shared-model": {"supports_vision": True}},
            },
        ],
    }
    monkeypatch.setattr(
        runner,
        "_resolve_session_agent_runtime",
        lambda **_: (
            "shared-model",
            {
                "provider": "custom",
                "requested_provider": "vision-provider",
            },
        ),
    )

    assert runner._decide_image_input_mode(
        source=_source(),
        user_config=cfg,
    ) == "native"


@pytest.mark.asyncio
async def test_prepare_route_identity_check_keeps_event_loop_responsive(monkeypatch):
    """A slow route-identity check must not block gateway heartbeats."""
    import asyncio
    import threading
    from types import SimpleNamespace

    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="inspect @AGENTS.md",
        message_type=MessageType.TEXT,
        source=source,
    )
    started = threading.Event()
    released_by_event_loop = threading.Event()
    seen = {}
    main_thread = threading.current_thread()

    cfg = {
        "model": {
            "default": "test-model",
            "provider": "test-provider",
            "base_url": "https://example.invalid/v1",
            "context_length": 128000,
        }
    }
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: cfg)
    monkeypatch.setattr(
        runner,
        "_resolve_session_agent_runtime",
        lambda **_kwargs: (
            "test-model",
            {
                "provider": "test-provider",
                "base_url": "https://example.invalid/v1",
                "api_key": "",
            },
        ),
    )

    def blocking_route_identity_check(*_args):
        seen["thread"] = threading.current_thread()
        started.set()
        seen["event_loop_progressed"] = released_by_event_loop.wait(timeout=2)
        return False

    monkeypatch.setattr(
        "hermes_cli.route_identity.should_clear_context_pin",
        blocking_route_identity_check,
    )

    async def fake_context_length(*_args, **_kwargs):
        return 128000

    async def fake_preprocess(message, **_kwargs):
        return SimpleNamespace(
            blocked=False,
            expanded=False,
            message=message,
            warnings=[],
        )

    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length_async", fake_context_length
    )
    monkeypatch.setattr(
        "agent.context_references.preprocess_context_references_async",
        fake_preprocess,
    )

    async def heartbeat_ticker():
        while not started.is_set():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        released_by_event_loop.set()

    heartbeat = asyncio.create_task(heartbeat_ticker())
    result = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[]
    )
    await heartbeat

    assert result == "inspect @AGENTS.md"
    assert seen["event_loop_progressed"] is True
    assert seen["thread"] is not main_thread


@pytest.mark.asyncio
async def test_pre_turn_vision_uses_logical_scope_without_leaking(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from agent.auxiliary_client import _runtime_main_value, scoped_runtime_main
    from hermes_state import SessionDB

    runner = _make_runner()
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("root", source="gateway")
        db.end_session("root", "compression")
        db.create_session(
            "continuation",
            source="gateway",
            parent_session_id="root",
        )
        runner._session_db = SimpleNamespace(_db=db)
        runner.session_store = SimpleNamespace(
            peek_session_id=lambda session_key: (
                "continuation" if session_key == "routing-key" else None
            )
        )
        monkeypatch.setattr(
            runner,
            "_decide_image_input_mode",
            lambda **_kwargs: "text",
        )
        monkeypatch.setattr(
            runner,
            "_resolve_session_agent_runtime",
            lambda **_kwargs: (
                "gpt-5.4",
                {"provider": "openai-codex", "api_mode": "codex_responses"},
            ),
        )
        observed = []

        async def fake_enrich(message_text, _image_paths):
            observed.append(_runtime_main_value("cache_scope"))
            return f"vision: {message_text}"

        monkeypatch.setattr(runner, "_enrich_message_with_vision", fake_enrich)

        with scoped_runtime_main({"cache_scope": "caller"}):
            result = await runner._prepare_inbound_message_text(
                event=_image_event(),
                source=_source(),
                history=[],
                session_key="routing-key",
            )
            assert _runtime_main_value("cache_scope") == "caller"

        assert result == "vision: look"
        assert observed == ["root"]
        assert _runtime_main_value("cache_scope") == ""
    finally:
        db.close()
