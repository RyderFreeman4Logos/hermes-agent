"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_async_delivery=False) self-POSTs
  /v1/chat/completions with the RAW session id in X-Hermes-Session-Id, so the
  wake turn resumes the REAL session instead of a parallel invisible one
  keyed by build_session_key().
"""

import asyncio
from unittest.mock import patch

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.wake import deliver_wake, adapter_supports_push


class PushAdapter:
    """Default adapter shape — no supports_async_delivery attribute."""

    def __init__(self):
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self, host="0.0.0.0", port=0, key="test-key", model="hermes"):
        self._host = host
        self._port = port
        self._api_key = key
        self._model_name = model

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
    )


def test_adapter_supports_push_default_true():
    assert adapter_supports_push(PushAdapter()) is True
    assert adapter_supports_push(ApiServerLikeAdapter()) is False


async def _serve(handler):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_deliver_wake_non_push_self_posts_raw_session_id(monkeypatch):
    """The self-post carries the RAW session id header + bearer auth and a
    single user message with stream=false — the exact entry point real
    gateway turns use."""
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(host="0.0.0.0", port=port, key="sekrit")
            await deliver_wake(adapter, text="task done — wake", session_id="raw-sid-42")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-42"
    assert seen["auth"] == "Bearer sekrit"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [
        {"role": "user", "content": "task done — wake"}
    ]


def test_deliver_wake_retries_429_then_succeeds(monkeypatch):
    """HTTP 429 (max_concurrent_runs cap) is transient — retried with backoff."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2


async def _completion_self_post_commits_hidden_event_with_result(
    tmp_path, monkeypatch
):
    """The private self-post marker reaches the model but cannot be forged."""
    from aiohttp.test_utils import TestClient, TestServer

    from gateway.platforms.api_server import APIServerAdapter
    from gateway.wake import INTERNAL_COMPLETION_DELIVERY_HEADER
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.gateway.test_api_server import _create_app

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "completion-self-post"
    db.create_session(session_id, "api_server", model="test/model")
    db.append_message(session_id, "user", "start the build")
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "sk-secret"})
    )
    adapter._session_db = db
    agent = AIAgent(
        session_id=session_id,
        session_db=db,
        api_key="test-key",
        base_url="http://127.0.0.1:1/v1",
        provider="openai-compat",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    seen_prompts = []

    def fake_run_conversation(user_message, conversation_history, **_kwargs):
        pending = agent._pending_cli_user_message
        if pending is not None:
            assert pending["_completion_delivery_synthetic"] is True
            user_row = pending
        else:
            user_row = {"role": "user", "content": user_message}
        seen_prompts.append(user_message)
        assistant = {"role": "assistant", "content": f"handled: {user_message}"}
        messages = [*conversation_history, user_row, assistant]
        if pending is not None:
            from agent.turn_finalizer import finalize_completion_delivery_suffix

            finalize_completion_delivery_suffix(
                agent,
                messages,
                final_response=assistant["content"],
                failed=False,
                interrupted=False,
            )
        agent._persist_session(messages, conversation_history)
        agent._pending_cli_user_message = None
        return {
            "final_response": assistant["content"],
            "messages": messages,
            "api_calls": 1,
            "completed": True,
        }

    monkeypatch.setattr(agent, "run_conversation", fake_run_conversation)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as client:
        adapter._host = "127.0.0.1"
        adapter._port = client.server.port
        with patch.object(adapter, "_create_agent", return_value=agent):
            internal_prompt = "completion payload and model-only instruction"
            await deliver_wake(
                adapter,
                text=internal_prompt,
                session_id=session_id,
                completion_delivery=True,
            )
            external_prompt = "genuine external user message"
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer sk-secret",
                    "X-Hermes-Session-Id": session_id,
                    INTERNAL_COMPLETION_DELIVERY_HEADER: "forged",
                },
                json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": external_prompt}],
                },
            )
            assert response.status == 200

    resumed = db.get_messages_as_conversation(session_id)
    contents = [message["content"] for message in resumed]
    assert seen_prompts == [internal_prompt, external_prompt]
    internal_row = next(
        message for message in resumed if message["content"] == internal_prompt
    )
    assert internal_row["display_kind"] == "hidden"
    assert f"handled: {internal_prompt}" in contents
    assert external_prompt in contents
    assert f"handled: {external_prompt}" in contents
    db.close()


def test_completion_self_post_commits_hidden_event_with_result(
    tmp_path, monkeypatch
):
    asyncio.run(
        _completion_self_post_commits_hidden_event_with_result(tmp_path, monkeypatch)
    )
