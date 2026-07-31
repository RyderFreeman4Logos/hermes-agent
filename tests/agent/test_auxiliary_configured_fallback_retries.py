from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent.auxiliary_client as auxiliary_client


def _response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _quota_error():
    error = RuntimeError("quota")
    error.body = {"error": {"code": "quota_exhausted"}}
    return error


def _client(async_mode, side_effect):
    create = (
        AsyncMock(side_effect=side_effect)
        if async_mode
        else MagicMock(side_effect=side_effect)
    )
    return SimpleNamespace(
        base_url="https://example.test/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


async def _call(async_mode):
    kwargs = {
        "task": "session_search",
        "messages": [{"role": "user", "content": "hello"}],
    }
    if async_mode:
        return await auxiliary_client.async_call_llm(**kwargs)
    return auxiliary_client.call_llm(**kwargs)


def _install(
    monkeypatch, async_mode, primary_effects, fallback_effects, *, fallback_on=None
):
    entries = [
        {"provider": f"fallback-{index}", "model": f"model-{index}"}
        for index in range(len(fallback_effects))
    ]
    config = {"fallback_chain": entries}
    if fallback_on is not None:
        config["fallback_on"] = fallback_on

    primary = _client(async_mode, primary_effects)
    fallbacks = [_client(async_mode, effects) for effects in fallback_effects]
    by_provider = dict(zip((entry["provider"] for entry in entries), fallbacks))

    monkeypatch.setattr(auxiliary_client, "_TRANSIENT_RETRY_BACKOFF_BASE", 0)
    monkeypatch.setattr(
        auxiliary_client, "_get_auxiliary_task_config", lambda task: config
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        lambda *args: ("primary", "primary-model", None, None, None),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *args, **kwargs: (primary, "primary-model"),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_fallback_entry",
        lambda entry: (by_provider[entry["provider"]], entry["model"]),
    )
    if async_mode:
        monkeypatch.setattr(
            auxiliary_client,
            "_to_async_client",
            lambda client, model, **kwargs: (client, model),
        )
    return primary, fallbacks


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_primary_succeeds_after_retry(monkeypatch, async_mode):
    primary, fallbacks = _install(
        monkeypatch,
        async_mode,
        [RuntimeError("temporary"), _response("primary")],
        [[_response("unused")]],
    )

    result = await _call(async_mode)

    assert result.choices[0].message.content == "primary"
    assert primary.chat.completions.create.call_count == 2
    assert fallbacks[0].chat.completions.create.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_primary_exhausted_then_fallback_succeeds(monkeypatch, async_mode):
    primary_error = _quota_error()
    primary, fallbacks = _install(
        monkeypatch,
        async_mode,
        [primary_error] * 3,
        [[_response("fallback")]],
        fallback_on=["quota_exhausted"],
    )

    result = await _call(async_mode)

    assert result.choices[0].message.content == "fallback"
    assert primary.chat.completions.create.call_count == 3
    assert fallbacks[0].chat.completions.create.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_exhausted_fallback_advances_to_second(monkeypatch, async_mode):
    primary, fallbacks = _install(
        monkeypatch,
        async_mode,
        [RuntimeError("primary failed")] * 3,
        [[ValueError("first failed")] * 3, [_response("second")]],
    )

    result = await _call(async_mode)

    assert result.choices[0].message.content == "second"
    assert primary.chat.completions.create.call_count == 3
    assert [client.chat.completions.create.call_count for client in fallbacks] == [3, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_missing_fallback_on_allows_arbitrary_failure(monkeypatch, async_mode):
    _, fallbacks = _install(
        monkeypatch,
        async_mode,
        [LookupError("arbitrary")] * 3,
        [[_response("fallback")]],
    )

    result = await _call(async_mode)

    assert result.choices[0].message.content == "fallback"
    assert fallbacks[0].chat.completions.create.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_quota_only_does_not_advance_on_unrelated_failure(
    monkeypatch, async_mode
):
    original = RuntimeError("unrelated")
    primary, fallbacks = _install(
        monkeypatch,
        async_mode,
        [original] * 3,
        [[_response("unused")]],
        fallback_on=["quota_exhausted"],
    )

    with pytest.raises(RuntimeError) as raised:
        await _call(async_mode)

    assert raised.value is original
    assert primary.chat.completions.create.call_count == 3
    assert fallbacks[0].chat.completions.create.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_exhaustion_raises_original_primary_error(monkeypatch, async_mode):
    original = RuntimeError("original primary")
    primary, fallbacks = _install(
        monkeypatch,
        async_mode,
        [original, RuntimeError("later primary"), RuntimeError("last primary")],
        [[ValueError("fallback one")] * 3, [OSError("fallback two")] * 3],
    )

    with pytest.raises(RuntimeError) as raised:
        await _call(async_mode)

    assert raised.value is original
    assert primary.chat.completions.create.call_count == 3
    assert [client.chat.completions.create.call_count for client in fallbacks] == [3, 3]
