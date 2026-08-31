"""Regression coverage for lean-digest configured fallback continuation."""

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.auxiliary_client import async_call_llm, call_llm
from agent.context_compressor import ContextCompressor


_CHUNK_CHARS = 88


def _turns(count=3):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [
        {"role": "user", "content": f"MARKER-{letters[i]} " + letters[i] * 70}
        for i in range(count)
    ]


def _body(messages):
    content = messages[0]["content"]
    marker = re.search(r"MARKER-([A-Z])", content)
    return f"DIGEST-{marker.group(1)}" if marker else "DIGEST-PAD"


def _segment_headers(out):
    return re.findall(r"### Segment (\d+)/(\d+)", out)


def _client(create, endpoint: str) -> SimpleNamespace:
    return SimpleNamespace(
        base_url=endpoint,
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


def _response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_stale_configured_fallback_continues_to_next_candidate(async_mode):
    create = AsyncMock if async_mode else MagicMock
    primary = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=create(side_effect=RuntimeError("connection refused"))
            )
        )
    )
    first, second = MagicMock(), MagicMock()
    labels = ["fallback_chain[0](first)", "fallback_chain[1](second)"]
    attempted = []

    def sync_fallback(_client, _model, label, **_kwargs):
        attempted.append(label)
        return None if label == labels[0] else _response("second")

    async def async_fallback(_client, _model, label, **_kwargs):
        attempted.append(label)
        return None if label == labels[0] else _response("second")

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch(
            "agent.auxiliary_client._to_async_client",
            side_effect=lambda client, model, **_kwargs: (client, model),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            side_effect=[(first, "first-model", labels[0]), (second, "second-model", labels[1])],
        ) as configured_chain,
        patch(
            "agent.auxiliary_client._call_fallback_candidate_async"
            if async_mode
            else "agent.auxiliary_client._call_fallback_candidate_sync",
            new=AsyncMock(side_effect=async_fallback) if async_mode else sync_fallback,
        ),
    ):
        request = (async_call_llm if async_mode else call_llm)(
            task="compression",
            messages=[{"role": "user", "content": "digest"}],
        )
        response = await request if async_mode else request

    assert response.choices[0].message.content == "second"
    assert attempted == labels
    assert configured_chain.call_count == 2
    assert configured_chain.call_args_list[1].kwargs["start_index"] == 1


def test_stale_selected_fallback_continues_without_reopening_primary():
    first = _client(
        MagicMock(side_effect=RuntimeError("connection refused")),
        "https://first.invalid/v1",
    )
    response = _response("second")
    labels = ["fallback_chain[0](first)", "fallback_chain[1](second)"]
    route_info = {"fallback_label": labels[0]}

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._selected_configured_fallback",
            return_value=(first, "first-model", labels[0]),
        ) as selected,
        patch("agent.auxiliary_client._get_cached_client") as cached,
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(MagicMock(), "second-model", labels[1]),
        ) as configured,
        patch(
            "agent.auxiliary_client._call_fallback_candidate_sync",
            return_value=response,
        ) as fallback,
    ):
        result = call_llm(
            task="compression",
            messages=[{"role": "user", "content": "digest"}],
            route_info=route_info,
        )

    assert result is response
    selected.assert_called_once_with("compression", labels[0])
    cached.assert_not_called()
    assert configured.call_args.args[:2] == ("compression", labels[0])
    assert fallback.call_args.args[2] == labels[1]


@pytest.mark.asyncio
async def test_async_stale_selected_fallback_continues_without_reopening_primary():
    first = _client(
        AsyncMock(side_effect=RuntimeError("connection refused")),
        "https://first.invalid/v1",
    )
    second = MagicMock()
    response = _response("second")
    labels = ["fallback_chain[0](first)", "fallback_chain[1](second)"]
    route_info = {"fallback_label": labels[0]}

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._selected_configured_fallback",
            return_value=(first, "first-model", labels[0]),
        ) as selected,
        patch("agent.auxiliary_client._get_cached_client") as cached,
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(second, "second-model", labels[1]),
        ) as configured,
        patch(
            "agent.auxiliary_client._to_async_client",
            return_value=(second, "second-model"),
        ),
        patch(
            "agent.auxiliary_client._call_fallback_candidate_async",
            new=AsyncMock(return_value=response),
        ) as fallback,
    ):
        result = await async_call_llm(
            task="compression",
            messages=[{"role": "user", "content": "digest"}],
            route_info=route_info,
        )

    assert result is response
    selected.assert_called_once_with("compression", labels[0], async_mode=True)
    cached.assert_not_called()
    assert configured.call_args.args[:2] == ("compression", labels[0])
    assert fallback.call_args.args[2] == labels[1]


@pytest.mark.asyncio
async def test_async_stale_selected_fallback_resolves_an_async_client():
    sync_client = _client(MagicMock(return_value=_response("sync")), "https://chatgpt.com/backend-api")
    async_create = AsyncMock(return_value=_response("async"))
    async_client = _client(async_create, "https://chatgpt.com/backend-api")
    entry = {
        "provider": "openai-codex",
        "model": "codex-model",
        "base_url": "https://chatgpt.com/backend-api",
    }
    resolved_async_modes = []

    def resolve_provider(provider, model=None, async_mode=False, **_kwargs):
        resolved_async_modes.append(async_mode)
        client = async_client if async_mode else sync_client
        return client, model

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "primary-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"fallback_chain": [entry]},
        ),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api",
                "api_mode": "codex_responses",
                "api_key": "focused-bound-key",
                "model": "codex-model",
            },
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=resolve_provider,
        ),
        patch("agent.auxiliary_client._provider_requires_stream", return_value=False),
    ):
        result = await async_call_llm(
            task="compression",
            messages=[{"role": "user", "content": "digest"}],
            route_info={"fallback_label": "fallback_chain[0](openai-codex)"},
        )

    assert result.choices[0].message.content == "async"
    assert resolved_async_modes == [True]
    async_create.assert_awaited_once()


def test_lean_harvest_cancel_keeps_slots_and_lean_recovery():
    """Queued Future.cancel() still yields slots; augment keeps recovery sections."""
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    turns = _turns() + [{"role": "user", "content": "MARKER-D " + ("D" * 70)}]

    two_started = threading.Event()
    submitted_three = threading.Event()
    hold = threading.Event()
    futures = []
    lock = threading.Lock()
    in_flight = 0
    orig_submit = ThreadPoolExecutor.submit

    def tracking_submit(self, fn, *args, **kwargs):
        future = orig_submit(self, fn, *args, **kwargs)
        futures.append(future)
        if len(futures) >= 3:
            submitted_three.set()
        return future

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        nonlocal in_flight
        assert task == "compression"
        content = messages[0]["content"]
        if "MARKER-A" in content:
            body = "DIGEST-A"
        else:
            with lock:
                in_flight += 1
                if in_flight >= 2:
                    two_started.set()
            assert hold.wait(timeout=2), "queued cancel never released workers"
            with lock:
                in_flight -= 1
            body = _body(messages)
        return _response(body)

    def cancel_queued():
        assert two_started.wait(timeout=2), "first two workers never started"
        assert submitted_three.wait(timeout=2), "third future never submitted"
        assert futures[2].cancel(), "later job was already running"
        hold.set()

    compressor._session_id = "sess-137-harvest"
    canceler = threading.Thread(target=cancel_queued, daemon=True)
    canceler.start()
    try:
        with (
            patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", _CHUNK_CHARS),
            patch("agent.auxiliary_client.call_llm", fake_call_llm),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": 2},
            ),
            patch.object(ThreadPoolExecutor, "submit", tracking_submit),
        ):
            output = compressor._augment_summary_lean("KEEP-SUMMARY", turns)
    finally:
        hold.set()
        canceler.join(timeout=2)

    assert output.startswith("KEEP-SUMMARY")
    assert _segment_headers(output) == [("1", "4"), ("2", "4"), ("3", "4"), ("4", "4")]
    assert "DIGEST-A" in output
    assert "DIGEST-B" in output
    assert "DIGEST-C" in output
    assert "[digest unavailable for segment 4/4" in output
    assert "recover via session_search" in output
    assert "## User Messages (verbatim, newest first)" in output
    assert "MARKER-D" in output
    assert "## Context Recovery" in output
    assert "session_id='sess-137-harvest'" in output
