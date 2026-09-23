"""Public /v1/runs SSE witnesses for terminal subagent identity."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.asyncio
async def test_public_run_sse_preserves_terminal_identity_and_only_its_explicit_nulls(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run_identity"
    queue = asyncio.Queue()
    adapter._run_streams[run_id] = queue

    owner_request = MagicMock()
    owner_request.headers = {}
    owner_request.path = f"/v1/runs/{run_id}/events"
    owner_request.method = "GET"
    adapter._run_owners[run_id] = adapter._run_idempotency_scope(owner_request)

    callback = adapter._make_run_event_callback(run_id, asyncio.get_running_loop())
    callback(
        "subagent.complete",
        status="failed",
        model="accepted-model",
        provider="accepted-provider",
        summary=None,
        duration_seconds=None,
    )
    callback(
        "subagent.complete",
        status="failed",
        model=None,
        provider=None,
        summary=None,
        duration_seconds=None,
    )
    callback(
        "subagent.start",
        model=None,
        provider=None,
        summary=None,
        duration_seconds=None,
    )
    await asyncio.sleep(0)
    queue.put_nowait(None)

    app = web.Application()
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    async with TestClient(TestServer(app)) as client:
        response = await client.get(f"/v1/runs/{run_id}/events")
        assert response.status == 200
        body = await response.text()

    events = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 3
    known, unknown, started = events
    assert (known["model"], known["provider"]) == (
        "accepted-model",
        "accepted-provider",
    )
    assert "model" in unknown and unknown["model"] is None
    assert "provider" in unknown and unknown["provider"] is None
    assert "summary" not in unknown
    assert "duration_seconds" not in unknown
    assert "model" not in started
    assert "provider" not in started
