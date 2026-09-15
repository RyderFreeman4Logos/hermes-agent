"""Real resolver witness for complete auxiliary fallback destinations."""

from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.auxiliary_client import async_call_llm, call_llm
import agent.context_compressor as context_compressor_module
from agent.context_compressor import ContextCompressor


class _RouteFailure(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text),
            finish_reason="stop",
        )]
    )


def _summary_receipt_value(compressor):
    receipt = getattr(context_compressor_module, "_SUMMARY_ROUTE_RECEIPT", None)
    if receipt is not None:
        return receipt.get()
    return getattr(compressor, "_last_summary_route", None)


@contextmanager
def _summary_receipt_scope(compressor, route):
    receipt = getattr(context_compressor_module, "_SUMMARY_ROUTE_RECEIPT", None)
    if receipt is not None:
        token = receipt.set(dict(route))
        try:
            yield
        finally:
            receipt.reset(token)
    else:
        with patch.object(compressor, "_last_summary_route", dict(route), create=True):
            yield


def _client(*, api_key: str, outcome, async_mode: bool, base_url: str):
    if callable(outcome) or isinstance(outcome, list) or isinstance(outcome, BaseException):
        create = AsyncMock(side_effect=outcome) if async_mode else MagicMock(side_effect=outcome)
    else:
        create = AsyncMock(return_value=outcome) if async_mode else MagicMock(return_value=outcome)
    return SimpleNamespace(
        base_url=base_url,
        api_key=api_key,
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


def _recovery_scope(mode: str, *, async_mode: bool):
    """Synthetic physical clients for the three same-provider recovery rungs."""
    stack = ExitStack()
    if mode == "nous-heal":
        def nous_outcome(**kwargs):
            if kwargs.get("model") == "stale-model":
                raise _RouteFailure("model stale-model not found", 404)
            return _response("summary-ok")

        stale = _client(
            api_key="synthetic-nous-key",
            outcome=nous_outcome,
            async_mode=async_mode,
            base_url="https://inference-api.nousresearch.com/v1",
        )
        expected = {
            "provider": "nous", "model": "healed-model",
            "base_url": stale.base_url, "api_key": "synthetic-nous-key",
            "api_mode": "chat_completions",
        }
        stack.enter_context(patch("agent.auxiliary_client._get_cached_client", return_value=(stale, "stale-model")))
        stack.enter_context(patch("agent.auxiliary_client._refresh_nous_recommended_model", return_value="healed-model"))
    else:
        failure = (
            _RouteFailure("unauthorized", 401)
            if mode == "oauth-refresh"
            else _RouteFailure("payment required", 402)
        )
        retry_base = "https://chatgpt.com/backend-api/codex"
        stale = _client(
            api_key="synthetic-stale-key", outcome=failure,
            async_mode=async_mode, base_url=retry_base,
        )
        fresh = _client(
            api_key="synthetic-fresh-key", outcome=_response("summary-ok"),
            async_mode=async_mode, base_url=retry_base,
        )
        expected = {
            "provider": "openai-codex", "model": "summary-model",
            "base_url": fresh.base_url, "api_key": "synthetic-fresh-key",
            "api_mode": "chat_completions",
        }
        stack.enter_context(patch(
            "agent.auxiliary_client._get_cached_client",
            side_effect=[(stale, "summary-model"), (fresh, "summary-model")],
        ))
        if mode == "oauth-refresh":
            stack.enter_context(patch(
                "agent.auxiliary_client._auth_refresh_provider_for_route",
                return_value="openai-codex",
            ))
            stack.enter_context(patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True))
        else:
            stack.enter_context(patch(
                "agent.auxiliary_client._recoverable_pool_provider",
                return_value="openai-codex",
            ))
            stack.enter_context(patch("agent.auxiliary_client._recover_provider_pool", return_value=True))
    provider = "nous" if mode == "nous-heal" else "openai-codex"
    model = "stale-model" if mode == "nous-heal" else "summary-model"
    stack.enter_context(patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=(provider, model, stale.base_url, stale.api_key, "chat_completions"),
    ))
    stack.enter_context(patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={}))
    return stack, expected


def _compress_messages():
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn-{index} " + "x" * 240}
        for index in range(12)
    ]


@pytest.mark.parametrize("mode", ["nous-heal", "oauth-refresh", "pool-rotation"])
def test_public_compress_pins_each_successful_recovery_destination_for_digest(mode):
    compressor = ContextCompressor(
        "main-model", config_context_length=100_000, protect_first_n=1,
        protect_last_n=1, quiet_mode=True, tail_mode="lean",
    )
    digest_calls: list[dict] = []

    def fake_digest_call(**kwargs):
        digest_calls.append(kwargs)
        return _response("digest-ok")

    scope, expected = _recovery_scope(mode, async_mode=False)
    with scope, patch("agent.auxiliary_client.call_llm", fake_digest_call):
        result = compressor.compress(_compress_messages(), current_tokens=99_999, force=True)

    assert result != _compress_messages()
    assert digest_calls
    assert {key: digest_calls[0].get(key) for key in expected} == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nous-heal", "oauth-refresh", "pool-rotation"])
async def test_async_call_reports_each_successful_recovery_destination(mode):
    route_info: dict = {}
    scope, expected = _recovery_scope(mode, async_mode=True)
    with scope:
        response = await async_call_llm(
            task="compression",
            messages=[{"role": "user", "content": "synthetic summary"}],
            route_info=route_info,
        )

    assert response.choices[0].message.content == "summary-ok"
    assert {key: route_info.get(key) for key in expected} == expected


def test_overlapping_public_compress_attempts_keep_summary_routes_attempt_local():
    compressor = ContextCompressor(
        "main-model", config_context_length=100_000, protect_first_n=1,
        protect_last_n=1, quiet_mode=True, tail_mode="lean",
    )
    routes = {
        "primary": {
            "provider": "custom", "model": "primary-model",
            "base_url": "https://primary.invalid/v1", "api_key": "primary-key",
            "api_mode": "chat_completions", "timeout": 19.0,
        },
        "fallback": {
            "provider": "custom", "model": "fallback-model",
            "base_url": "https://fallback.invalid/v1", "api_key": "fallback-key",
            "api_mode": "chat_completions", "timeout": 23.0,
        },
    }
    primary_at_augment = threading.Event()
    fallback_at_augment = threading.Event()
    primary_digest_done = threading.Event()
    digest_routes: dict[str, dict] = {}
    errors: list[BaseException] = []
    real_augment = compressor._augment_summary_lean

    def fake_summary_call(*, route_info, **_kwargs):
        route_info.update(routes[threading.current_thread().name])
        return _response("summary-ok")

    def gated_augment(summary, turns, **kwargs):
        name = threading.current_thread().name
        if name == "primary":
            primary_at_augment.set()
            assert fallback_at_augment.wait(2)
        else:
            assert primary_at_augment.wait(2)
            fallback_at_augment.set()
            assert primary_digest_done.wait(2)
        return real_augment(summary, turns, **kwargs)

    def fake_digest_call(**kwargs):
        name = threading.current_thread().name
        digest_routes[name] = kwargs
        if name == "primary":
            primary_digest_done.set()
        return _response("digest-ok")

    def run():
        try:
            compressor.compress(_compress_messages(), current_tokens=99_999, force=True)
        except BaseException as exc:
            errors.append(exc)

    with (
        patch("agent.context_compressor.call_llm", fake_summary_call),
        patch("agent.auxiliary_client.call_llm", fake_digest_call),
        patch.object(compressor, "_augment_summary_lean", gated_augment),
    ):
        primary = threading.Thread(target=run, name="primary")
        fallback = threading.Thread(target=run, name="fallback")
        primary.start()
        assert primary_at_augment.wait(2)
        fallback.start()
        primary.join(3)
        fallback.join(3)

    assert not errors
    assert not primary.is_alive() and not fallback.is_alive()
    assert set(digest_routes) == set(routes)
    for name, expected in routes.items():
        assert {key: digest_routes[name].get(key) for key in expected} == expected


def test_call_llm_reports_complete_successful_custom_fallback_destination():
    class CapacityUnavailable(Exception):
        status_code = 402

    primary = MagicMock()
    primary.base_url = "https://primary.invalid/v1"
    primary.api_key = "synthetic-primary-key"
    primary.chat.completions.create.side_effect = CapacityUnavailable("payment required")

    fallback = MagicMock()
    fallback.base_url = "https://fallback.invalid/v1"
    fallback.api_key = "synthetic-fallback-key"
    fallback.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="fallback-ok"))]
    )
    fallback_entry = {
        "provider": "custom",
        "model": "fallback-model",
        "base_url": "https://fallback.invalid/v1",
        "api_key": "synthetic-fallback-key",
        "api_mode": "anthropic_messages",
        "timeout": 37,
    }
    route_info = {}

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=(
                "custom", "primary-model", "https://primary.invalid/v1",
                "synthetic-primary-key", "chat_completions",
            ),
        ),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"fallback_chain": [fallback_entry]},
        ),
        patch(
            "agent.auxiliary_client._resolve_fallback_entry",
            return_value=(fallback, "fallback-model"),
        ),
    ):
        response = call_llm(
            task="compression",
            messages=[{"role": "user", "content": "synthetic request"}],
            route_info=route_info,
        )

    assert response.choices[0].message.content == "fallback-ok"
    assert route_info == {
        "provider": "custom",
        "model": "fallback-model",
        "base_url": "https://fallback.invalid/v1",
        "api_key": "synthetic-fallback-key",
        "api_mode": "anthropic_messages",
        "timeout": 37.0,
    }


def test_generate_summary_carries_its_route_to_redacted_digest_egress():
    selected = {
        "provider": "custom",
        "model": "fallback-model",
        "base_url": "https://fallback.invalid/v1",
        "api_key": "synthetic-fallback-key",
        "api_mode": "anthropic_messages",
        "timeout": 37.0,
    }
    secret = "summary-digest-secret"
    primary_requests: list[str] = []
    digest_calls: list[dict] = []

    def fake_summary_call(*, messages, route_info, **_kwargs):
        primary_requests.append(messages[0]["content"])
        route_info.update(selected)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="successful summary"),
                finish_reason="stop",
            )]
        )

    def fake_digest_call(**kwargs):
        digest_calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="digest"))]
        )

    compressor = ContextCompressor("main-model", quiet_mode=True, tail_mode="lean")
    turns = [{
        "role": "user",
        "content": "benign-summary-id " + ("x" * 90) + f" https://user:{secret}@api.invalid/v1",
    }]
    with (
        patch("agent.context_compressor.call_llm", fake_summary_call),
        patch("agent.auxiliary_client.call_llm", fake_digest_call),
    ):
        summary = compressor._generate_summary(turns)

    assert summary is not None
    assert primary_requests and digest_calls
    assert all(secret not in request for request in primary_requests)
    assert all(secret not in call["messages"][0]["content"] for call in digest_calls)
    assert "benign-summary-id" in digest_calls[0]["messages"][0]["content"]
    assert {key: digest_calls[0].get(key) for key in selected} == selected
    assert _summary_receipt_value(compressor) is None


def test_first_digest_can_advance_a_summary_route_before_siblings_start():
    summary_route = {
        "provider": "custom", "model": "summary-route",
        "base_url": "https://summary.invalid/v1", "api_key": "summary-key",
        "api_mode": "chat_completions", "timeout": 19.0,
    }
    digest_fallback = {
        "provider": "custom", "model": "digest-fallback",
        "base_url": "https://digest.invalid/v1", "api_key": "digest-key",
        "api_mode": "anthropic_messages", "timeout": 31.0,
    }
    calls: list[dict] = []

    def fake_digest_call(*, route_info=None, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            route_info.update(digest_fallback)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="digest"))]
        )

    compressor = ContextCompressor("main-model", quiet_mode=True, tail_mode="lean")
    turns = [
        {"role": "user", "content": f"MARKER-{index} " + chr(65 + index) * 70}
        for index in range(3)
    ]
    with _summary_receipt_scope(compressor, summary_route):
        with (
            patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 88),
            patch("agent.auxiliary_client.call_llm", fake_digest_call),
            patch("agent.auxiliary_client._get_task_max_concurrency", return_value=2),
        ):
            compressor._build_chunk_digests(turns)

    assert {key: calls[0].get(key) for key in summary_route} == summary_route
    assert all(
        {key: call.get(key) for key in digest_fallback} == digest_fallback
        for call in calls[1:]
    )


def test_host_cancel_prevents_queued_digest_provider_dispatch():
    from agent import auxiliary_client as aux

    cancel = threading.Event()
    release = threading.Event()
    two_started = threading.Event()
    lock = threading.Lock()
    active = 0
    dispatched: list[str] = []

    def fake_digest_call(*, messages, **_kwargs):
        nonlocal active
        content = messages[0]["content"]
        marker = next(
            (name for name in ("MARKER-0", "MARKER-1", "MARKER-2", "MARKER-3") if name in content),
            "remainder",
        )
        dispatched.append(marker)
        if marker != "MARKER-0":
            with lock:
                active += 1
                if active == 2:
                    two_started.set()
            assert release.wait(2)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="digest"))]
        )

    def cancel_after_workers_start():
        assert two_started.wait(2)
        cancel.set()
        release.set()

    turns = [
        {"role": "user", "content": f"MARKER-{index} " + chr(65 + index) * 70}
        for index in range(4)
    ]
    compressor = ContextCompressor("main-model", quiet_mode=True, tail_mode="lean")
    canceller = threading.Thread(target=cancel_after_workers_start, daemon=True)
    canceller.start()
    try:
        with (
            aux.aux_interrupt_protection(cancel_event=cancel),
            patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 88),
            patch("agent.auxiliary_client.call_llm", fake_digest_call),
            patch("agent.auxiliary_client._get_task_max_concurrency", return_value=2),
            pytest.raises(aux.AuxiliaryExplicitCancellation),
        ):
            compressor._build_chunk_digests(turns)
    finally:
        release.set()
        canceller.join(2)

    assert "MARKER-3" not in dispatched
