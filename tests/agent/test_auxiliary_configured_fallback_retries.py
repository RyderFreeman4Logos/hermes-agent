from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent.auxiliary_client as auxiliary_client


def _response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _quota_error():
    error = RuntimeError("quota exceeded")
    error.status_code = 429
    error.body = {"error": {"code": "quota_exhausted"}}
    return error


def _status_error(status, message):
    error = RuntimeError(message)
    error.status_code = status
    return error


def _client(async_mode, side_effect):
    create = (
        AsyncMock(side_effect=side_effect)
        if async_mode
        else MagicMock(side_effect=side_effect)
    )
    return SimpleNamespace(
        api_key="stale-key",
        base_url="https://example.test/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


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


class _RealisticPool:
    def __init__(self, provider):
        base_url = {
            "deepseek": "https://api.deepseek.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }[provider]
        self.provider = provider
        self.rotated = False
        self.stale = SimpleNamespace(
            id="stale", provider=provider, runtime_api_key="stale-key",
            runtime_base_url=base_url,
        )
        self.fresh = SimpleNamespace(
            id="fresh", provider=provider, runtime_api_key="fresh-key",
            runtime_base_url=base_url,
        )

    def has_credentials(self):
        return True

    def current(self):
        return self.fresh if self.rotated else self.stale

    def peek(self):
        return self.current()

    def select(self):
        return self.current()

    def try_refresh_current(self):
        return None

    def mark_exhausted_and_rotate(self, **kwargs):
        self.rotated = True
        return self.fresh


def _install_real_transport(monkeypatch, async_mode, effects_for, *, sibling_keys=()):
    cache = {}
    built_keys = []
    requests = []
    canonicals = {}

    def make_client(api_key, base_url, effects, *, is_async):
        remaining = list(effects)
        canonical = canonicals.setdefault(api_key, object())

        def create(**kwargs):
            requests.append((api_key, str(base_url)))
            effect = remaining.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            return effect

        client = SimpleNamespace(
            api_key=api_key,
            base_url=str(base_url),
            _real_client=canonical,
            _async_effects=list(effects),
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(side_effect=create) if is_async else MagicMock(side_effect=create)
                )
            ),
        )
        return client

    def build(api_key, base_url, **kwargs):
        built_keys.append(api_key)
        client = make_client(
            api_key, base_url, effects_for(api_key, str(base_url)), is_async=False
        )
        if api_key in sibling_keys:
            cache[("sibling", api_key, async_mode)] = (
                SimpleNamespace(_real_client=client._real_client),
                "sibling-model",
                None,
            )
        return client

    def to_async(client, model, **kwargs):
        async_client = make_client(
            client.api_key, client.base_url, client._async_effects, is_async=True
        )
        async_client._real_client = client._real_client
        return async_client, model

    monkeypatch.setattr(auxiliary_client, "_client_cache", cache)
    monkeypatch.setattr(auxiliary_client, "_create_openai_client", build)
    if async_mode:
        monkeypatch.setattr(auxiliary_client, "_to_async_client", to_async)
    return cache, built_keys, requests, canonicals


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
@pytest.mark.parametrize(
    "primary_error",
    [_quota_error(), _status_error(429, "rate limit"), ConnectionError("offline")],
)
async def test_fallback_success_evicts_primary_and_sibling(
    monkeypatch, async_mode, primary_error
):
    primary, _ = _install(
        monkeypatch,
        async_mode,
        [primary_error] * 3,
        [[_response("fallback")]],
    )
    underlying = object()
    primary._real_client = underlying
    sibling = SimpleNamespace(_real_client=underlying)
    cache = {
        ("primary", async_mode): (primary, "primary-model", None),
        ("primary", not async_mode): (sibling, "primary-model", None),
    }
    monkeypatch.setattr(auxiliary_client, "_client_cache", cache)

    result = await _call(async_mode)

    assert result.choices[0].message.content == "fallback"
    assert cache == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_exhausted_fallback_advances_to_second(monkeypatch, async_mode):
    primary, fallbacks = _install(
        monkeypatch,
        async_mode,
        [RuntimeError("primary failed")] * 3,
        [[ValueError("first failed")] * 3, [_response("second")]],
    )
    underlying = object()
    fallbacks[0]._real_client = underlying
    sibling = SimpleNamespace(_real_client=underlying)
    cache = {
        ("fallback-0", async_mode): (fallbacks[0], "model-0", None),
        ("fallback-0", not async_mode): (sibling, "model-0", None),
    }
    monkeypatch.setattr(auxiliary_client, "_client_cache", cache)

    result = await _call(async_mode)

    assert result.choices[0].message.content == "second"
    assert cache == {}
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


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("hop", ["primary", "fallback"])
async def test_configured_hop_refreshes_auth_before_retry(
    monkeypatch, async_mode, hop
):
    auth_error = _status_error(401, "expired token")
    primary_effects = [auth_error] * 3 if hop == "primary" else [RuntimeError("primary")] * 3
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
            MagicMock(side_effect=[(stale, "fallback-model"), (fresh, "fallback-model")]),
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
@pytest.mark.parametrize(
    ("pool_error", "stale_attempts"),
    [(_status_error(429, "rate limit"), 2), (_quota_error(), 1)],
)
async def test_configured_hop_rotates_pool_after_exhaustion(
    monkeypatch, async_mode, hop, pool_error, stale_attempts
):
    primary_effects = [pool_error] * 3 if hop == "primary" else [RuntimeError("primary")] * 3
    fallback_effects = [[pool_error] * 3] if hop == "fallback" else [[_response("unused")]]
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
            MagicMock(side_effect=[(stale, "fallback-model"), (fresh, "fallback-model")]),
        )
    monkeypatch.setattr(
        auxiliary_client, "_refresh_provider_credentials", MagicMock(return_value=False)
    )
    pool = _RotatingPool()
    monkeypatch.setattr(auxiliary_client, "load_pool", lambda provider: pool)

    result = await _call(async_mode)

    assert result.choices[0].message.content == "fresh"
    assert stale.chat.completions.create.call_count == stale_attempts
    assert fresh.chat.completions.create.call_count == 1
    assert len(pool.rotate_calls) == 1
    assert pool.rotate_calls[0]["status_code"] == 429


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_configured_recovery_keeps_frozen_main_runtime(
    monkeypatch, async_mode
):
    import hermes_cli.auth as auth

    pool = _RealisticPool("openrouter")
    config = {
        "provider": "auto",
        "model": "right-model",
        "fallback_chain": [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.test/v1",
                "api_key": "fallback-key",
            }
        ],
    }

    def effects_for(api_key, base_url):
        if "wrong-provider.test" in base_url:
            return [_response("wrong-runtime-success")]
        if "fallback.test" in base_url:
            return [_response("fallback")]
        return [_quota_error(), _quota_error(), _quota_error()]

    _, _, requests, _ = _install_real_transport(
        monkeypatch, async_mode, effects_for
    )
    monkeypatch.setattr(auxiliary_client, "_TRANSIENT_RETRY_BACKOFF_BASE", 0)
    monkeypatch.setattr(
        auxiliary_client, "_get_auxiliary_task_config", lambda task: config
    )
    monkeypatch.setattr(
        auxiliary_client,
        "load_pool",
        lambda provider: pool if provider == "openrouter" else None,
    )
    monkeypatch.setattr(
        auth,
        "resolve_api_key_provider_credentials",
        lambda provider: {
            "api_key": "wrong-key",
            "base_url": "https://wrong-provider.test/v1",
        },
    )
    frozen = {
        "provider": "openrouter",
        "model": "right-model",
        "api_key": "right-key",
        "base_url": "https://openrouter.ai/api/v1",
    }
    ambient = {
        "provider": "deepseek",
        "model": "wrong-model",
        "api_key": "wrong-key",
        "base_url": "https://wrong-provider.test/v1",
    }

    kwargs = {
        "task": "session_search",
        "main_runtime": frozen,
        "messages": [{"role": "user", "content": "secret prompt"}],
    }
    with auxiliary_client.scoped_runtime_main(ambient):
        result = (
            await auxiliary_client.async_call_llm(**kwargs)
            if async_mode
            else auxiliary_client.call_llm(**kwargs)
        )

    assert result.choices[0].message.content == "fallback"
    assert all("wrong-provider.test" not in base_url for _, base_url in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_rotated_primary_evicts_current_client_and_siblings(
    monkeypatch, async_mode
):
    import hermes_cli.auth as auth

    pool = _RealisticPool("deepseek")
    config = {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "fallback_chain": [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.test/v1",
                "api_key": "fallback-key",
            }
        ],
    }

    def effects_for(api_key, base_url):
        if api_key == "fallback-key":
            return [_response("fallback")]
        if api_key == "fresh-key":
            return [_quota_error(), _quota_error()]
        return [_quota_error()]

    cache, _, requests, canonicals = _install_real_transport(
        monkeypatch,
        async_mode,
        effects_for,
        sibling_keys={"stale-key", "fresh-key"},
    )
    monkeypatch.setattr(auxiliary_client, "_TRANSIENT_RETRY_BACKOFF_BASE", 0)
    monkeypatch.setattr(
        auxiliary_client, "_get_auxiliary_task_config", lambda task: config
    )
    monkeypatch.setattr(
        auxiliary_client,
        "load_pool",
        lambda provider: pool if provider == "deepseek" else None,
    )
    monkeypatch.setattr(
        auth,
        "resolve_api_key_provider_credentials",
        lambda provider: {
            "api_key": "stale-key",
            "base_url": "https://api.deepseek.com/v1",
        },
    )

    result = await _call(async_mode)

    assert result.choices[0].message.content == "fallback"
    assert [api_key for api_key, _ in requests[:3]] == [
        "stale-key",
        "fresh-key",
        "fresh-key",
    ]
    assert all(
        getattr(entry[0], "_real_client", entry[0]) is not canonicals["fresh-key"]
        for entry in cache.values()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_fallback_rotation_rebuilds_through_pool_aware_cache(
    monkeypatch, async_mode
):
    import hermes_cli.auth as auth

    pool = _RealisticPool("deepseek")
    config = {
        "fallback_chain": [
            {"provider": "deepseek", "model": "deepseek-v4-flash"}
        ]
    }

    def effects_for(api_key, base_url):
        if api_key == "fresh-key":
            return [_response("fresh")]
        return [_status_error(429, "rate limit"), _status_error(429, "rate limit")]

    _, built_keys, requests, _ = _install_real_transport(
        monkeypatch, async_mode, effects_for
    )
    monkeypatch.setattr(
        auxiliary_client, "_get_auxiliary_task_config", lambda task: config
    )
    monkeypatch.setattr(
        auxiliary_client,
        "load_pool",
        lambda provider: pool if provider == "deepseek" else None,
    )
    monkeypatch.setattr(
        auth,
        "resolve_api_key_provider_credentials",
        lambda provider: {
            "api_key": "stale-key",
            "base_url": "https://api.deepseek.com/v1",
        },
    )
    kwargs = {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": None,
        "max_tokens": None,
        "tools": None,
        "effective_timeout": 1,
        "effective_extra_body": {},
        "reasoning_config": None,
    }

    result = (
        await auxiliary_client._try_configured_fallbacks_async(
            "session_search", **kwargs
        )
        if async_mode
        else auxiliary_client._try_configured_fallbacks_sync(
            "session_search", **kwargs
        )
    )

    assert result.choices[0].message.content == "fresh"
    assert built_keys == ["stale-key", "fresh-key"]
    assert [api_key for api_key, _ in requests] == [
        "stale-key",
        "stale-key",
        "fresh-key",
    ]
