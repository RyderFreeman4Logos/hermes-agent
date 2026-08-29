"""Regression coverage for lean-digest configured fallback continuation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.auxiliary_client import async_call_llm, call_llm


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
