"""Lean digest concurrency, fallback isolation, and retry contracts."""

import asyncio
import json
import pickle
import re
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agent.auxiliary_client import async_call_llm, call_llm
from agent.context_compressor import ContextCompressor
import pytest

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


def _response(body):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = body
    return response


def _run_real_digest(config, primary, fallback, *, count=3, resolve_calls=None, resolver=None):
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")

    def resolve_fallback(provider, model=None, **kwargs):
        if resolve_calls is not None:
            resolve_calls.append((provider, model, kwargs))
        return resolver(provider, model=model, **kwargs) if resolver else (fallback, model)

    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", _CHUNK_CHARS),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value=config,
        ),
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, "primary-model"),
        ),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve_fallback),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
    ):
        return compressor._build_chunk_digests(_turns(count))


def _fallback_config(limit=1):
    return {
        "max_concurrency": 2,
        "fallback_chain": [{
            "provider": "fallback-provider",
            "model": "fallback-model",
            "base_url": "https://fallback.invalid/anthropic",
            "api_mode": "anthropic_messages",
            "api_key": "SENTINEL_PROFILE_KEY",
            "max_concurrency": limit,
        }],
    }


def _client(create, *, endpoint, api_mode, credential_scope):
    return SimpleNamespace(
        api_key=credential_scope,
        base_url=endpoint,
        api_mode=api_mode,
        credential_scope=credential_scope,
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


def test_selected_fallback_route_reuses_exact_transport_identity_for_siblings(caplog):
    primary_calls = []
    fallback_calls = []
    resolve_calls = []

    def primary_create(**kwargs):
        primary_calls.append(kwargs)
        raise RuntimeError("connection refused")

    def fallback_create(**kwargs):
        fallback_calls.append(kwargs)
        return _response(_body(kwargs["messages"]))

    primary = _client(
        primary_create,
        endpoint="https://primary.invalid/v1",
        api_mode="chat_completions",
        credential_scope="PRIMARY_SCOPE",
    )
    fallback = _client(
        fallback_create,
        endpoint="https://fallback.invalid/anthropic",
        api_mode="anthropic_messages",
        credential_scope="SENTINEL_PROFILE_KEY",
    )

    output = _run_real_digest(
        _fallback_config(limit=1), primary, fallback, resolve_calls=resolve_calls,
    )

    assert len(primary_calls) == 1
    assert len(fallback_calls) == 3
    assert len(resolve_calls) == 3
    assert all(
        call[2] == {
            "explicit_base_url": "https://fallback.invalid/anthropic",
            "explicit_api_key": "SENTINEL_PROFILE_KEY",
            "api_mode": "anthropic_messages",
        }
        for call in resolve_calls
    )
    assert all(call["model"] == "fallback-model" for call in fallback_calls)
    assert fallback.base_url == "https://fallback.invalid/anthropic"
    assert fallback.api_mode == "anthropic_messages"
    assert fallback.credential_scope == "SENTINEL_PROFILE_KEY"
    assert "SENTINEL_PROFILE_KEY" not in "".join(record.getMessage() for record in caplog.records)
    assert "DIGEST-A" in output and "DIGEST-B" in output and "DIGEST-C" in output


@pytest.mark.asyncio
async def test_async_call_llm_uses_real_fallback_transport_and_exact_route():
    primary_calls = []
    fallback_calls = []

    async def primary_create(**kwargs):
        primary_calls.append(kwargs)
        raise RuntimeError("connection refused")

    async def fallback_create(**kwargs):
        fallback_calls.append(kwargs)
        return _response("ASYNC-DIGEST")

    primary = _client(
        primary_create,
        endpoint="https://primary.invalid/v1",
        api_mode="chat_completions",
        credential_scope="PRIMARY_SCOPE",
    )
    fallback = _client(
        fallback_create,
        endpoint="https://fallback.invalid/anthropic",
        api_mode="anthropic_messages",
        credential_scope="SENTINEL_PROFILE_KEY",
    )
    config = _fallback_config(limit=1)
    route_info = {}

    with (
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config),
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, "primary-model"),
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback, "fallback-model"),
        ),
        patch("agent.auxiliary_client._to_async_client", return_value=(fallback, "fallback-model")),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
    ):
        response = await async_call_llm(
            task="compression",
            messages=[{"role": "user", "content": "MARKER-A"}],
            max_tokens=32,
            route_info=route_info,
        )
        sibling = await async_call_llm(
            task="compression",
            messages=[{"role": "user", "content": "MARKER-B"}],
            max_tokens=32,
            route_info=route_info,
        )

    assert response.choices[0].message.content == "ASYNC-DIGEST"
    assert sibling.choices[0].message.content == "ASYNC-DIGEST"
    assert len(primary_calls) >= 1
    assert len(fallback_calls) == 2
    assert fallback_calls[0]["model"] == "fallback-model"
    assert fallback.base_url == "https://fallback.invalid/anthropic"
    assert fallback.api_mode == "anthropic_messages"
    assert fallback.credential_scope == "SENTINEL_PROFILE_KEY"


def _two_fallback_config():
    config = _fallback_config(limit=1)
    config["fallback_chain"][0].update(
        provider="first-provider", model="first-model",
        base_url="https://first.invalid/anthropic",
    )
    config["fallback_chain"].append({
        "provider": "second-provider", "model": "second-model",
        "base_url": "https://second.invalid/anthropic",
        "api_mode": "anthropic_messages", "max_concurrency": 1,
    })
    return config


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure_mode", ["raise", "empty", "rejected"])
async def test_selected_route_replans_after_success_then_failure(async_mode, failure_mode):
    first_calls, second_calls, primary_calls = [], [], []

    def primary_create(**kwargs):
        primary_calls.append(kwargs)
        raise RuntimeError("connection refused")

    def first_create(**kwargs):
        first_calls.append(kwargs)
        if len(first_calls) > 1:
            if failure_mode == "empty":
                return _response("")
            if failure_mode == "rejected":
                response = MagicMock()
                response.choices = []
                return response
            raise RuntimeError("first candidate down")
        return _response(_body(kwargs["messages"]))

    def second_create(**kwargs):
        second_calls.append(kwargs)
        return _response(_body(kwargs["messages"]))

    make = AsyncMock if async_mode else MagicMock
    first = _client(make(side_effect=first_create), endpoint="https://first.invalid/anthropic", api_mode="anthropic_messages", credential_scope="FIRST_SCOPE")
    primary = _client(make(side_effect=primary_create), endpoint="https://primary.invalid/v1", api_mode="chat_completions", credential_scope="PRIMARY_SCOPE")
    second = _client(make(side_effect=second_create), endpoint="https://second.invalid/anthropic", api_mode="anthropic_messages", credential_scope="SECOND_SCOPE")

    def resolve(provider, model=None, explicit_base_url=None, **kwargs):
        return (first if explicit_base_url == first.base_url else second), model

    route_info = {}
    with ExitStack() as stack:
        stack.enter_context(patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=_two_fallback_config()))
        stack.enter_context(patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("primary", "primary-model", None, None, None)))
        stack.enter_context(patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")))
        stack.enter_context(patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve))
        stack.enter_context(patch("agent.auxiliary_client._transient_retry_count", return_value=0))
        if async_mode:
            stack.enter_context(patch("agent.auxiliary_client._to_async_client", side_effect=lambda client, model, **_: (client, model)))
        responses = []
        primary_calls_after_first = 0
        for index, marker in enumerate(("MARKER-A", "MARKER-B", "MARKER-C")):
            request = (async_call_llm if async_mode else call_llm)(task="compression", messages=[{"role": "user", "content": marker}], route_info=route_info)
            responses.append(await request if async_mode else request)
            if index == 0:
                primary_calls_after_first = len(primary_calls)

    assert [response.choices[0].message.content for response in responses] == ["DIGEST-A", "DIGEST-B", "DIGEST-C"]
    assert len(first_calls) == len(second_calls) == 2
    assert len(primary_calls) == primary_calls_after_first
    assert "FIRST_SCOPE" not in repr(route_info) and "SECOND_SCOPE" not in repr(route_info)
    assert "client" not in repr(route_info)
    json.dumps(route_info)
    pickle.dumps(route_info)


@pytest.mark.parametrize("first_mode", ["raise", "empty"])
def test_failed_first_candidate_is_not_reused_by_retry_or_siblings(first_mode):
    first_calls = []
    second_calls = []

    def first_create(**kwargs):
        first_calls.append(kwargs)
        if first_mode == "empty":
            return _response("")
        raise RuntimeError("first candidate down")

    first = _client(
        first_create,
        endpoint="https://first.invalid/anthropic",
        api_mode="anthropic_messages",
        credential_scope="FIRST_SCOPE",
    )

    def primary_create(**kwargs):
        raise RuntimeError("connection refused")

    def second_create(**kwargs):
        second_calls.append(kwargs)
        return _response(_body(kwargs["messages"]))

    primary = _client(
        primary_create,
        endpoint="https://primary.invalid/v1",
        api_mode="chat_completions",
        credential_scope="PRIMARY_SCOPE",
    )
    second = _client(
        second_create,
        endpoint="https://second.invalid/anthropic",
        api_mode="anthropic_messages",
        credential_scope="SECOND_SCOPE",
    )
    config = _two_fallback_config()
    def resolve(provider, model=None, explicit_base_url=None, **kwargs):
        return (first if explicit_base_url == first.base_url else second), model
    output = _run_real_digest(config, primary, first, resolver=resolve)

    assert len(first_calls) == 1
    assert len(second_calls) == 3
    assert "DIGEST-A" in output and "DIGEST-B" in output and "DIGEST-C" in output
    assert "digest unavailable" not in output



@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_fallback_userinfo_is_rejected_without_metadata_or_log_leak(async_mode, caplog):
    sentinel = "USERINFO_SENTINEL"
    create = AsyncMock if async_mode else MagicMock
    primary = _client(
        create(side_effect=RuntimeError("connection refused")),
        endpoint="https://primary.invalid/v1",
        api_mode="chat_completions",
        credential_scope="PRIMARY_SCOPE",
    )
    config = _fallback_config(limit=1)
    config["fallback_chain"][0]["base_url"] = (
        f"https://{sentinel}:PASSWORD_SENTINEL@fallback.invalid/anthropic"
    )
    route_info = {}
    request = async_call_llm if async_mode else call_llm
    with ExitStack() as stack:
        stack.enter_context(patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config))
        stack.enter_context(patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ))
        stack.enter_context(patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")))
        resolver = stack.enter_context(patch("agent.auxiliary_client.resolve_provider_client"))
        stack.enter_context(patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")))
        stack.enter_context(patch("agent.auxiliary_client._transient_retry_count", return_value=0))
        with pytest.raises(RuntimeError, match="connection refused") as raised:
            result = request(
                task="compression",
                messages=[{"role": "user", "content": "userinfo"}],
                route_info=route_info,
            )
            if async_mode:
                await result
    visible = " ".join((str(raised.value), repr(route_info), caplog.text))
    assert sentinel not in visible
    assert "PASSWORD_SENTINEL" not in visible
    resolver.assert_not_called()
    json.dumps(route_info)
    pickle.dumps(route_info)


@pytest.mark.parametrize(
    "malformed",
    [
        "https://[::1",
        "https://[::1]:bad",
        "https://%5B::1",
        "https://example%ZZ.com/v1",
        "https://example.com:99999/v1",
        "https://",
        "https:///v1",
        "custom://",
        "https://example.com/%ZZ",
        "custom://example.com/%GG/path",
        "https://example.com/%",
        "https://example.com/%A",
        "https://example.com/v1?query=%ZZ",
        "https://example.com/v1#fragment=%GG",
        "example.com/v1",
        "",
        "   ",
        "https:// example.com/v1",
        " https://example.com/v1",
        "https://example.com/v1 ",
        "https://example.com/a b",
        r"https://example.com/a\b",
        r"https://\example.com/v1",
        "https://example.com/\x00v1",
        "https://example.com/\x01v1",
        "https://example.com/\nv1",
        "https://example.com/\rv1",
        "https://example.com/\tv1",
        "https://example.com/\u2003v1",
        "https://xn--.com/v1",
        f"https://{chr(0xD800)}.com/v1",
        *(f"https://example.com/a{chr(codepoint)}b" for codepoint in range(0x80, 0xA0)),
        *(f"https://a{chr(codepoint)}.example.com/v1" for codepoint in range(0x80, 0xA0)),
    ],
)
def test_malformed_fallback_url_rejected_before_resolver_without_leaks(malformed, caplog):
    from agent.auxiliary_client import _resolve_fallback_entry

    sentinel = "SENTINEL_PROFILE_KEY"
    route_info = {}
    entry = {
        "provider": "fallback-provider",
        "model": "fallback-model",
        "base_url": malformed,
        "api_key": sentinel,
    }
    with patch("agent.auxiliary_client.resolve_provider_client") as resolver:
        with pytest.raises(ValueError, match="^fallback base_url is malformed$") as raised:
            _resolve_fallback_entry(entry)
    visible = " ".join((str(raised.value), caplog.text, repr(route_info)))
    if malformed:
        assert malformed not in visible
    assert sentinel not in visible
    resolver.assert_not_called()
    json.dumps(route_info)
    pickle.dumps(route_info)


@pytest.mark.parametrize("include_base_url", [False, True])
def test_missing_or_none_fallback_url_stays_unset(include_base_url):
    from agent.auxiliary_client import _resolve_fallback_entry

    entry = {
        "provider": "fallback-provider",
        "model": "fallback-model",
        "base_url": None,
    }
    if not include_base_url:
        entry.pop("base_url")
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(None, None),
    ) as resolver:
        assert _resolve_fallback_entry(entry) == (None, None)
    assert resolver.call_args.kwargs["explicit_base_url"] is None


@pytest.mark.parametrize(
    "valid",
    [
        "https://[::1]:7700/v1",
        "custom://example%2Ecom/v1",
        "https://例え.テスト/v1",
        "https://xn--r8jz45g.xn--zckzah/v1",
        "https://example.com/%E2%9C%93",
        "https://example.com/%20",
        "https://example.com/v1?query=%2F#fragment=%20",
    ],
)
def test_valid_fallback_authority_forms_remain_accepted(valid):
    from agent.auxiliary_client import _validate_fallback_base_url

    assert _validate_fallback_base_url(valid) == valid


def test_configured_fallback_walk_excludes_stale_selected_index():
    from agent.auxiliary_client import _try_configured_fallback_chain

    config = _two_fallback_config()
    resolve_calls = []
    second = MagicMock(base_url="https://second.invalid/anthropic")

    def resolve(provider, model=None, explicit_base_url=None, **kwargs):
        resolve_calls.append(explicit_base_url)
        return second, model

    with (
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve),
        patch("agent.auxiliary_client._candidate_context_window", return_value=None),
    ):
        client, model, label = _try_configured_fallback_chain(
            "compression",
            "primary",
            failed_model="primary-model",
            skip_labels={"fallback_chain[0](old-provider)"},
        )

    assert client is second
    assert model == "second-model"
    assert label == "fallback_chain[1](second-provider)"
    assert resolve_calls == ["https://second.invalid/anthropic"]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_selected_route_mutation_replans_without_mixing_client_identity(async_mode):
    first_calls, mutated_calls, second_calls, primary_calls = [], [], [], []
    create = AsyncMock if async_mode else MagicMock

    def primary_create(**kwargs):
        primary_calls.append(kwargs)
        raise RuntimeError("connection refused")

    def first_create(**kwargs):
        first_calls.append(kwargs)
        return _response("FIRST")

    def mutated_create(**kwargs):
        mutated_calls.append(kwargs)
        return _response("MUTATED")

    def second_create(**kwargs):
        second_calls.append(kwargs)
        return _response("SECOND")

    primary = _client(
        create(side_effect=primary_create), endpoint="https://primary.invalid/v1",
        api_mode="chat_completions", credential_scope="PRIMARY_SCOPE",
    )
    first = _client(
        create(side_effect=first_create), endpoint="https://first.invalid/anthropic",
        api_mode="anthropic_messages", credential_scope="FIRST_SCOPE",
    )
    mutated = _client(
        create(side_effect=mutated_create), endpoint="https://mutated.invalid/v1",
        api_mode="chat_completions", credential_scope="MUTATED_SCOPE",
    )
    second = _client(
        create(side_effect=second_create), endpoint="https://second.invalid/anthropic",
        api_mode="anthropic_messages", credential_scope="SECOND_SCOPE",
    )
    config = _two_fallback_config()
    route_info = {}
    resolve_calls = []

    def resolve(provider, model=None, explicit_base_url=None, **kwargs):
        resolve_calls.append(explicit_base_url)
        return {
            first.base_url: (first, "first-model"),
            mutated.base_url: (mutated, "mutated-model"),
            second.base_url: (second, "second-model"),
        }[explicit_base_url]

    with ExitStack() as stack:
        stack.enter_context(patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config))
        stack.enter_context(patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ))
        stack.enter_context(patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")))
        stack.enter_context(patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve))
        stack.enter_context(patch("agent.auxiliary_client._to_async_client", side_effect=lambda client, model, **_: (client, model)))
        stack.enter_context(patch("agent.auxiliary_client._transient_retry_count", return_value=0))
        first_request = call_llm if not async_mode else async_call_llm
        result = first_request(task="compression", messages=[{"role": "user", "content": "first"}], route_info=route_info)
        if async_mode:
            await result
        primary_calls_before_mutation = len(primary_calls)
        config["fallback_chain"][0].update(
            provider="mutated-provider",
            model="mutated-model",
            base_url=mutated.base_url,
            api_mode="chat_completions",
        )
        result = first_request(task="compression", messages=[{"role": "user", "content": "second"}], route_info=route_info)
        response = await result if async_mode else result

    assert response.choices[0].message.content == "SECOND"
    assert len(primary_calls) == primary_calls_before_mutation
    assert all(call["messages"][0]["content"] != "second" for call in primary_calls)
    assert len(first_calls) == 1
    assert not mutated_calls
    assert len(second_calls) == 1
    assert resolve_calls == [first.base_url, mutated.base_url, second.base_url]
    assert route_info["provider"] == "second-provider"
    assert route_info["model"] == "second-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_selected_route_inline_key_mutation_during_resolution_replans_without_using_new_key(
    async_mode, caplog,
):
    key_a = "INLINE_KEY_A_SENTINEL"
    key_b = "INLINE_KEY_B_SENTINEL"
    primary_calls, key_a_calls, key_b_calls, second_calls, resolve_keys = [], [], [], [], []
    make = AsyncMock if async_mode else MagicMock

    def primary_create(**kwargs):
        primary_calls.append(kwargs)
        raise RuntimeError("connection refused")

    key_a_client = _client(
        make(side_effect=lambda **kwargs: key_a_calls.append(kwargs) or _response("KEY-A")),
        endpoint="https://same.invalid/anthropic", api_mode="anthropic_messages",
        credential_scope=key_a,
    )
    key_b_client = _client(
        make(side_effect=lambda **kwargs: key_b_calls.append(kwargs) or _response("KEY-B")),
        endpoint=key_a_client.base_url, api_mode="anthropic_messages",
        credential_scope=key_b,
    )
    second = _client(
        make(side_effect=lambda **kwargs: second_calls.append(kwargs) or _response("SECOND")),
        endpoint="https://second.invalid/anthropic", api_mode="anthropic_messages",
        credential_scope="SECOND_SCOPE",
    )
    primary = _client(
        make(side_effect=primary_create), endpoint="https://primary.invalid/v1",
        api_mode="chat_completions", credential_scope="PRIMARY_SCOPE",
    )
    config = _two_fallback_config()
    config["fallback_chain"][0].update(base_url=key_a_client.base_url, api_key=key_a)
    route_info = {}

    def resolve(provider, model=None, explicit_api_key=None, explicit_base_url=None, **kwargs):
        resolve_keys.append(explicit_api_key)
        if explicit_base_url == second.base_url:
            return second, model
        if explicit_api_key == key_a:
            config["fallback_chain"][0].pop("api_key")
        return (key_a_client if explicit_api_key == key_a else key_b_client), model

    with ExitStack() as stack:
        stack.enter_context(patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config))
        stack.enter_context(patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ))
        stack.enter_context(patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")))
        stack.enter_context(patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve))
        stack.enter_context(patch("agent.auxiliary_client._transient_retry_count", return_value=0))
        if async_mode:
            stack.enter_context(patch("agent.auxiliary_client._to_async_client", side_effect=lambda client, model, **_: (client, model)))
        request = async_call_llm if async_mode else call_llm
        first = request(task="compression", messages=[{"role": "user", "content": "first"}], route_info=route_info)
        first = await first if async_mode else first
        primary_calls_after_first = len(primary_calls)
        selected = route_info["_fallback_route"].destination
        assert selected.credential_discriminator
        selected_visible = " ".join((repr(route_info), json.dumps(route_info), caplog.text))
        pickle.dumps(route_info)
        sibling = request(task="compression", messages=[{"role": "user", "content": "sibling"}], route_info=route_info)
        sibling = await sibling if async_mode else sibling

    assert first.choices[0].message.content == "KEY-A"
    assert sibling.choices[0].message.content == "SECOND"
    assert len(primary_calls) == primary_calls_after_first
    assert len(key_a_calls) == 1
    assert not key_b_calls
    assert len(second_calls) == 1
    assert resolve_keys == [key_a, None, None]
    visible = " ".join((selected_visible, repr(route_info), json.dumps(route_info), caplog.text))
    assert key_a not in visible and key_b not in visible
    assert "client" not in repr(route_info)
    pickle.dumps(route_info)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_removed_selected_route_does_not_reprobe_primary(async_mode):
    primary_calls, first_calls = [], []
    make = AsyncMock if async_mode else MagicMock

    def primary_create(**kwargs):
        primary_calls.append(kwargs)
        raise RuntimeError("connection refused")

    def first_create(**kwargs):
        first_calls.append(kwargs)
        return _response("FIRST")

    primary = _client(
        make(side_effect=primary_create), endpoint="https://primary.invalid/v1",
        api_mode="chat_completions", credential_scope="PRIMARY_SCOPE",
    )
    first = _client(
        make(side_effect=first_create), endpoint="https://first.invalid/anthropic",
        api_mode="anthropic_messages", credential_scope="FIRST_SCOPE",
    )
    config = _fallback_config(limit=1)
    config["fallback_chain"][0].update(
        provider="first-provider", model="first-model",
        base_url=first.base_url,
    )
    route_info = {}

    with ExitStack() as stack:
        stack.enter_context(patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config))
        stack.enter_context(patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ))
        stack.enter_context(patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")))
        stack.enter_context(patch("agent.auxiliary_client.resolve_provider_client", return_value=(first, "first-model")))
        stack.enter_context(patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")))
        stack.enter_context(patch("agent.auxiliary_client._try_payment_fallback", return_value=(None, None, "")))
        stack.enter_context(patch("agent.auxiliary_client._transient_retry_count", return_value=0))
        if async_mode:
            stack.enter_context(patch("agent.auxiliary_client._to_async_client", return_value=(first, "first-model")))
        request = async_call_llm if async_mode else call_llm

        first_result = request(
            task="compression", messages=[{"role": "user", "content": "first"}],
            route_info=route_info,
        )
        if async_mode:
            await first_result
        primary_calls_after_first = len(primary_calls)
        config["fallback_chain"].clear()

        with pytest.raises(BaseException):
            second_result = request(
                task="compression", messages=[{"role": "user", "content": "removed"}],
                route_info=route_info,
            )
            if async_mode:
                await second_result

    assert len(first_calls) == 1
    assert len(primary_calls) == primary_calls_after_first
    assert route_info["_failed_fallback_labels"] == ["fallback_chain[0](first-provider)"]


@pytest.mark.asyncio
async def test_selected_async_explicit_cancellation_propagates_without_replan():
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    primary_calls, first_calls, second_calls = [], [], []

    async def primary_create(**kwargs):
        primary_calls.append(kwargs)
        raise RuntimeError("connection refused")

    async def first_create(**kwargs):
        first_calls.append(kwargs)
        if len(first_calls) == 2:
            raise AuxiliaryExplicitCancellation()
        return _response("FIRST")

    async def second_create(**kwargs):
        second_calls.append(kwargs)
        return _response("SECOND")

    primary = _client(
        AsyncMock(side_effect=primary_create), endpoint="https://primary.invalid/v1",
        api_mode="chat_completions", credential_scope="PRIMARY_SCOPE",
    )
    first = _client(
        AsyncMock(side_effect=first_create), endpoint="https://first.invalid/anthropic",
        api_mode="anthropic_messages", credential_scope="FIRST_SCOPE",
    )
    second = _client(
        AsyncMock(side_effect=second_create), endpoint="https://second.invalid/anthropic",
        api_mode="anthropic_messages", credential_scope="SECOND_SCOPE",
    )
    config = _two_fallback_config()
    route_info = {}

    def resolve(provider, model=None, explicit_base_url=None, **kwargs):
        return (first if explicit_base_url == first.base_url else second), model

    with ExitStack() as stack:
        stack.enter_context(patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config))
        stack.enter_context(patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ))
        stack.enter_context(patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")))
        stack.enter_context(patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve))
        stack.enter_context(patch("agent.auxiliary_client._to_async_client", side_effect=lambda client, model, **_: (client, model)))
        stack.enter_context(patch("agent.auxiliary_client._transient_retry_count", return_value=0))

        await async_call_llm(
            task="compression", messages=[{"role": "user", "content": "first"}],
            route_info=route_info,
        )
        primary_calls_after_first = len(primary_calls)
        with pytest.raises(AuxiliaryExplicitCancellation) as raised:
            await async_call_llm(
                task="compression", messages=[{"role": "user", "content": "cancel"}],
                route_info=route_info,
            )
        response = await async_call_llm(
            task="compression", messages=[{"role": "user", "content": "reusable"}],
            route_info=route_info,
        )

    assert raised.value.cause == "explicit_host_cancel"
    assert response.choices[0].message.content == "FIRST"
    assert len(primary_calls) == primary_calls_after_first
    assert len(first_calls) == 3
    assert not second_calls


def test_selected_sync_explicit_cancellation_propagates_and_releases_permit():
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    primary_calls, first_calls, second_calls = [], [], []

    def primary_create(**kwargs):
        primary_calls.append(kwargs)
        raise RuntimeError("connection refused")

    def first_create(**kwargs):
        first_calls.append(kwargs)
        if len(first_calls) == 2:
            raise AuxiliaryExplicitCancellation()
        return _response("FIRST")

    primary = _client(
        MagicMock(side_effect=primary_create), endpoint="https://primary.invalid/v1",
        api_mode="chat_completions", credential_scope="PRIMARY_SCOPE",
    )
    first = _client(
        MagicMock(side_effect=first_create), endpoint="https://first.invalid/anthropic",
        api_mode="anthropic_messages", credential_scope="FIRST_SCOPE",
    )
    second = _client(
        MagicMock(side_effect=lambda **kwargs: second_calls.append(kwargs) or _response("SECOND")),
        endpoint="https://second.invalid/anthropic", api_mode="anthropic_messages",
        credential_scope="SECOND_SCOPE",
    )
    config = _two_fallback_config()
    route_info = {}

    def resolve(provider, model=None, explicit_base_url=None, **kwargs):
        return (first if explicit_base_url == first.base_url else second), model

    with ExitStack() as stack:
        stack.enter_context(patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config))
        stack.enter_context(patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ))
        stack.enter_context(patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")))
        stack.enter_context(patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve))
        stack.enter_context(patch("agent.auxiliary_client._transient_retry_count", return_value=0))

        call_llm(task="compression", messages=[{"role": "user", "content": "first"}], route_info=route_info)
        primary_calls_after_first = len(primary_calls)
        with pytest.raises(AuxiliaryExplicitCancellation):
            call_llm(task="compression", messages=[{"role": "user", "content": "cancel"}], route_info=route_info)
        response = call_llm(
            task="compression", messages=[{"role": "user", "content": "reusable"}],
            route_info=route_info,
        )

    assert response.choices[0].message.content == "FIRST"
    assert len(primary_calls) == primary_calls_after_first
    assert len(first_calls) == 3
    assert not second_calls


def _run_retry_digest(fake_call_llm, config, count=3):
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", _CHUNK_CHARS),
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
    ):
        return compressor._build_chunk_digests(_turns(count))


@pytest.mark.parametrize("permanent", [False, True])
def test_digest_retry_isolated_by_one_bounded_retry(permanent):
    attempts = 0

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        nonlocal attempts
        assert task == "compression"
        if "MARKER-B" in messages[0]["content"]:
            attempts += 1
            if permanent or attempts == 1:
                raise RuntimeError("permanent digest failure" if permanent else "transient digest failure")
        return _response(_body(messages))

    output = _run_retry_digest(fake_call_llm, {"max_concurrency": 2})

    assert attempts == 2
    assert "DIGEST-A" in output
    assert "DIGEST-C" in output
    if permanent:
        assert "[digest unavailable for segment 2/3" in output
        assert "permanent digest failure" not in output
    else:
        assert "DIGEST-B" in output
        assert "digest unavailable" not in output


def test_map_level_digest_failure_preserves_lean_recovery_sections():
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    compressor._session_id = "sess-map-failure"
    with patch.object(
        compressor,
        "_build_chunk_digests",
        side_effect=RuntimeError("executor setup failed"),
    ):
        output = compressor._augment_summary_lean("KEEP-SUMMARY", _turns())

    assert output.startswith("KEEP-SUMMARY")
    assert "## User Messages (verbatim, newest first)" in output
    assert "## Context Recovery" in output


def test_digest_explicit_cancellation_stops_before_placeholder_or_later_segments(caplog):
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    calls = []

    def cancel_first(**kwargs):
        calls.append(kwargs)
        raise AuxiliaryExplicitCancellation()

    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", _CHUNK_CHARS),
        patch("agent.auxiliary_client.call_llm", side_effect=cancel_first),
    ):
        with pytest.raises(AuxiliaryExplicitCancellation):
            compressor._build_chunk_digests(_turns())

    assert len(calls) == 1
    assert "digest unavailable" not in caplog.text


def test_parallel_digest_hard_cancellation_fences_queued_segments():
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    calls = []
    calls_lock = threading.Lock()
    second_started = threading.Event()
    cancelled = threading.Event()
    fourth_started = threading.Event()
    release_second = threading.Event()
    outcome = {}

    def fake_call_llm(**kwargs):
        with calls_lock:
            call_number = len(calls) + 1
            calls.append(call_number)
        if call_number == 1:
            return _response("DIGEST-A")
        if call_number == 2:
            second_started.set()
            assert release_second.wait(5)
            return _response("DIGEST-B")
        if call_number == 3:
            assert second_started.wait(5)
            cancelled.set()
            raise AuxiliaryExplicitCancellation()
        fourth_started.set()
        return _response("RAN-AFTER-CANCEL")

    def run_digest():
        try:
            _run_retry_digest(fake_call_llm, {"max_concurrency": 2}, count=4)
        except BaseException as exc:
            outcome["exc"] = exc

    baseline_workers = {
        thread.ident for thread in threading.enumerate()
        if thread.name.startswith("ThreadPoolExecutor-")
    }
    runner = threading.Thread(target=run_digest, name="pr165-digest-cancel-test")
    runner.start()
    try:
        assert cancelled.wait(5)
        assert not fourth_started.wait(1)
    finally:
        release_second.set()
        runner.join(5)

    assert not runner.is_alive()
    assert isinstance(outcome.get("exc"), AuxiliaryExplicitCancellation)
    assert calls == [1, 2, 3]
    assert not {
        thread.ident for thread in threading.enumerate()
        if thread.name.startswith("ThreadPoolExecutor-")
    } - baseline_workers


def test_map_level_digest_explicit_cancellation_propagates():
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with patch.object(
        compressor,
        "_build_chunk_digests",
        side_effect=AuxiliaryExplicitCancellation(),
    ):
        with pytest.raises(AuxiliaryExplicitCancellation):
            compressor._augment_summary_lean("KEEP-SUMMARY", _turns())


def test_lean_chunk_digests_overlap_and_keep_segment_order():
    """Later chunks may finish first; concatenated output stays ordered."""
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        nonlocal active, max_active
        assert task == "compression"
        content = messages[0]["content"]
        is_first = "MARKER-A" in content
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05 if is_first else 0.01)
        finally:
            with lock:
                active -= 1
        return _response(
            "DIGEST-A"
            if is_first
            else "DIGEST-B"
            if "MARKER-B" in content
            else "DIGEST-C"
        )

    turns = [
        {"role": "user", "content": "MARKER-A " + ("aaaa " * 20)},
        {"role": "user", "content": "MARKER-B " + ("bbbb " * 20)},
        {"role": "user", "content": "MARKER-C " + ("cccc " * 20)},
    ]
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")

    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 40),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch("agent.auxiliary_client._get_task_max_concurrency", return_value=None),
    ):
        out = compressor._build_chunk_digests(turns)

    assert max_active >= 2
    first = out.index("### Segment 1/")
    second = out.index("### Segment 2/")
    assert first < second
    assert out.find("DIGEST-A") < out.find("DIGEST-B")
    assert "DIGEST-A" in out[first:second]
    assert "DIGEST-B" in out[second:]


def test_lean_chunk_digests_keep_session_contextvars_on_workers():
    """Worker dispatch preserves runtime and secret ContextVars."""
    from agent.auxiliary_client import (
        _RUNTIME_MAIN_CONTEXT,
        reset_runtime_main,
        set_runtime_main,
    )
    from agent.secret_scope import _SECRET_SCOPE, reset_secret_scope, set_secret_scope

    seen = []
    lock = threading.Lock()

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        with lock:
            seen.append({
                "is_worker": threading.current_thread() is not threading.main_thread(),
                "runtime": _RUNTIME_MAIN_CONTEXT.get(),
                "secret": _SECRET_SCOPE.get(),
            })
        return _response("DIGEST")

    turns = [
        {"role": "user", "content": "MARKER-A " + ("aaaa " * 20)},
        {"role": "user", "content": "MARKER-B " + ("bbbb " * 20)},
    ]
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    runtime_token = set_runtime_main(
        provider="custom",
        model="probe-model",
        base_url="http://probe.invalid",
        api_key="PROBE-KEY",
    )
    secret_token = set_secret_scope({"OPENAI_API_KEY": "PROFILE-SECRET"})
    try:
        with (
            patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 40),
            patch("agent.auxiliary_client.call_llm", fake_call_llm),
            patch("agent.auxiliary_client._get_task_max_concurrency", return_value=None),
        ):
            compressor._build_chunk_digests(turns)
    finally:
        reset_secret_scope(secret_token)
        reset_runtime_main(runtime_token)

    workers = [hit for hit in seen if hit["is_worker"]]
    assert workers
    for hit in workers:
        assert hit["runtime"] is not None
        assert hit["runtime"].get("provider") == "custom"
        assert hit["secret"] is not None
        assert hit["secret"].get("OPENAI_API_KEY") == "PROFILE-SECRET"


def _segment_headers(out):
    return re.findall(r"### Segment (\d+)/(\d+)", out)


def test_lean_chunk_digests_never_exceed_configured_max_concurrency():
    """Configured compression concurrency bounds in-flight workers."""
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        nonlocal active, max_active
        assert task == "compression"
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.12)
        finally:
            with lock:
                active -= 1
        return _response(_body(messages))

    output = _run_retry_digest(fake_call_llm, {"max_concurrency": 2})

    assert max_active <= 2
    assert max_active == 2
    assert _segment_headers(output) == [("1", "3"), ("2", "3"), ("3", "3")]
    assert output.find("DIGEST-A") < output.find("DIGEST-B") < output.find("DIGEST-C")


def test_lean_chunk_digests_serial_when_max_concurrency_is_1():
    """max_concurrency=1 never overlaps call_llm."""
    active = 0
    max_active = 0
    lock = threading.Lock()
    order = []

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        nonlocal active, max_active
        assert task == "compression"
        body = _body(messages)
        with lock:
            active += 1
            max_active = max(max_active, active)
            order.append(body)
        try:
            time.sleep(0.04)
        finally:
            with lock:
                active -= 1
        return _response(body)

    output = _run_retry_digest(fake_call_llm, {"max_concurrency": 1})

    assert max_active == 1
    assert order == ["DIGEST-A", "DIGEST-B", "DIGEST-C"]
    assert _segment_headers(output) == [("1", "3"), ("2", "3"), ("3", "3")]
    assert output.find("DIGEST-A") < output.find("DIGEST-B") < output.find("DIGEST-C")


def test_lean_chunk_digests_keep_slots_when_worker_raises_baseexception():
    """Unexpected worker BaseException preserves every digest slot."""
    class WorkerFatal(BaseException):
        pass

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        assert task == "compression"
        if "MARKER-B" in messages[0]["content"]:
            raise WorkerFatal("worker-fatal")
        return _response(_body(messages))

    output = _run_retry_digest(fake_call_llm, {"max_concurrency": 2})

    assert _segment_headers(output) == [("1", "3"), ("2", "3"), ("3", "3")]
    assert "DIGEST-A" in output
    assert "DIGEST-C" in output
    assert "DIGEST-B" not in output
    assert "[digest unavailable for segment 2/3" in output
    assert "recover via session_search" in output
