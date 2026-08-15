from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent.auxiliary_client as auxiliary_client


def _response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _client(async_mode, side_effect):
    create = (
        AsyncMock(side_effect=side_effect)
        if async_mode
        else MagicMock(side_effect=side_effect)
    )
    return SimpleNamespace(
        api_key="test-key",
        base_url="https://example.test/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


def _status_error(status, message):
    error = RuntimeError(message)
    error.status_code = status
    return error


class _RotatingPool:
    def __init__(self):
        self.rotate_calls = []

    def has_credentials(self):
        return True

    def try_refresh_current(self):
        return None

    def mark_exhausted_and_rotate(self, **kwargs):
        self.rotate_calls.append(kwargs)
        return SimpleNamespace(id="fresh")


def _install(monkeypatch, async_mode, primary_effects, fallback_effects, fallback_on=None):
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


async def _call(async_mode):
    kwargs = {
        "task": "session_search",
        "messages": [{"role": "user", "content": "hello"}],
    }
    if async_mode:
        return await auxiliary_client.async_call_llm(**kwargs)
    return auxiliary_client.call_llm(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_configured_fallback_advances_after_arbitrary_failures(
    monkeypatch, async_mode
):
    primary, fallbacks = _install(
        monkeypatch,
        async_mode,
        [LookupError("primary failed")] * 3,
        [[ValueError("first failed")] * 3, [_response("second")]],
    )

    result = await _call(async_mode)

    assert result.choices[0].message.content == "second"
    assert primary.chat.completions.create.call_count == 3
    assert [client.chat.completions.create.call_count for client in fallbacks] == [3, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_quota_only_fallback_preserves_unrelated_primary_error(
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
async def test_exhaustion_raises_original_primary_and_evicts_sync_async_siblings(
    monkeypatch, async_mode
):
    original = TimeoutError("original primary")
    primary, fallbacks = _install(
        monkeypatch,
        async_mode,
        [original, RuntimeError("later primary"), RuntimeError("last primary")],
        [[ValueError("fallback one")] * 3, [OSError("fallback two")] * 3],
    )
    underlying = object()
    primary._real_client = underlying
    sibling = SimpleNamespace(_real_client=underlying)
    cache = {
        ("primary", async_mode): (primary, "primary-model", None),
        ("primary", not async_mode): (sibling, "primary-model", None),
    }
    monkeypatch.setattr(auxiliary_client, "_client_cache", cache)

    with pytest.raises(TimeoutError) as raised:
        await _call(async_mode)

    assert raised.value is original
    assert cache == {}
    assert primary.chat.completions.create.call_count == 3
    assert [client.chat.completions.create.call_count for client in fallbacks] == [3, 3]


def test_cache_eviction_normalizes_wrapper_target(monkeypatch):
    underlying = object()
    wrapper = SimpleNamespace(_real_client=underlying)
    sibling = SimpleNamespace(_real_client=underlying)
    cache = {
        ("sync",): (wrapper, "model", None),
        ("async",): (sibling, "model", None),
    }
    monkeypatch.setattr(auxiliary_client, "_client_cache", cache)

    assert auxiliary_client._evict_cached_client_instance(wrapper) is True
    assert cache == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("hop", ["primary", "fallback"])
async def test_configured_hop_refreshes_auth_before_fallback(
    monkeypatch, async_mode, hop
):
    auth_error = _status_error(401, "expired token")
    primary_effects = [auth_error] * 3 if hop == "primary" else [LookupError("primary")] * 3
    fallback_effects = [[auth_error] * 3] if hop == "fallback" else [[_response("unused")]]
    primary, fallbacks = _install(
        monkeypatch, async_mode, primary_effects, fallback_effects
    )
    stale = primary if hop == "primary" else fallbacks[0]
    fresh = _client(async_mode, [_response("fresh")])
    if hop == "primary":
        monkeypatch.setattr(
            auxiliary_client,
            "_get_cached_client",
            MagicMock(side_effect=[(stale, "primary-model"), (fresh, "primary-model")]),
        )
    else:
        monkeypatch.setattr(
            auxiliary_client,
            "_resolve_fallback_entry",
            MagicMock(side_effect=[(stale, "model-0"), (fresh, "model-0")]),
        )
    refresh = MagicMock(return_value=True)
    monkeypatch.setattr(auxiliary_client, "_refresh_provider_credentials", refresh)

    result = await _call(async_mode)

    assert result.choices[0].message.content == "fresh"
    refresh.assert_called_once_with("primary" if hop == "primary" else "fallback-0")
    assert stale.chat.completions.create.call_count == 1
    assert fresh.chat.completions.create.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("hop", ["primary", "fallback"])
async def test_configured_hop_rotates_pool_before_fallback(
    monkeypatch, async_mode, hop
):
    rate_limit = _status_error(429, "rate limit")
    primary_effects = [rate_limit] * 3 if hop == "primary" else [LookupError("primary")] * 3
    fallback_effects = [[rate_limit] * 3] if hop == "fallback" else [[_response("unused")]]
    primary, fallbacks = _install(
        monkeypatch, async_mode, primary_effects, fallback_effects
    )
    stale = primary if hop == "primary" else fallbacks[0]
    fresh = _client(async_mode, [_response("fresh")])
    if hop == "primary":
        monkeypatch.setattr(
            auxiliary_client,
            "_get_cached_client",
            MagicMock(side_effect=[(stale, "primary-model"), (fresh, "primary-model")]),
        )
    else:
        monkeypatch.setattr(
            auxiliary_client,
            "_resolve_fallback_entry",
            MagicMock(side_effect=[(stale, "model-0"), (fresh, "model-0")]),
        )
    monkeypatch.setattr(
        auxiliary_client, "_refresh_provider_credentials", MagicMock(return_value=False)
    )
    pool = _RotatingPool()
    monkeypatch.setattr(auxiliary_client, "load_pool", lambda provider: pool)

    result = await _call(async_mode)

    assert result.choices[0].message.content == "fresh"
    assert stale.chat.completions.create.call_count == 2
    assert fresh.chat.completions.create.call_count == 1
    assert len(pool.rotate_calls) == 1
