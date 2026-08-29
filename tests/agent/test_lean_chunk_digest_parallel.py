"""Regression coverage for lean-digest configured fallback continuation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.auxiliary_client import async_call_llm, call_llm


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
    selected.assert_called_once_with("compression", labels[0])
    cached.assert_not_called()
    assert configured.call_args.args[:2] == ("compression", labels[0])
    assert fallback.call_args.args[2] == labels[1]
