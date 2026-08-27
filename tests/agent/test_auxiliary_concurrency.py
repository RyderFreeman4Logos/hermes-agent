"""Tests for per-task concurrency limiting on auxiliary LLM calls (#23324)."""

import asyncio
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from agent import auxiliary_client
from agent.auxiliary_client import (
    call_llm,
    async_call_llm,
    _acquire_sync_aux_semaphore,
    _acquire_async_aux_semaphore,
    _get_task_max_concurrency,
    _record_route_info,
    _reset_aux_semaphores,
)


@pytest.fixture(autouse=True)
def _clean_semaphore_cache():
    _reset_aux_semaphores()
    yield
    _reset_aux_semaphores()


def _fallback_test_config(limit, *, task_limit=None):
    config = {
        "fallback_chain": [{
            "provider": "fallback-provider",
            "model": "fallback-model",
            "base_url": "https://fallback.invalid/anthropic",
            "api_mode": "anthropic_messages",
            "api_key": "SENTINEL_PROFILE_KEY",
            "max_concurrency": limit,
        }],
    }
    if task_limit is not None:
        config["max_concurrency"] = task_limit
    return config


def _valid_response(text="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


async def _acquire_on_running_loop():
    semaphore = _acquire_async_aux_semaphore("compression")
    assert semaphore is not None
    return semaphore


def _fallback_probe_clients(async_mode, n_callers, target):
    state = {"primary_calls": 0, "active": 0, "max_active": 0}
    lock = asyncio.Lock() if async_mode else threading.Lock()
    release = asyncio.Event() if async_mode else threading.Event()
    primary_ready = asyncio.Event() if async_mode else threading.Event()
    candidate_ready = asyncio.Event() if async_mode else threading.Event()

    if async_mode:
        async def primary_create(**kwargs):
            async with lock:
                state["primary_calls"] += 1
                if state["primary_calls"] == n_callers:
                    primary_ready.set()
            raise RuntimeError("connection refused")

        async def fallback_create(**kwargs):
            async with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                if state["active"] == target:
                    candidate_ready.set()
            await release.wait()
            async with lock:
                state["active"] -= 1
            return _valid_response()
    else:
        def primary_create(**kwargs):
            with lock:
                state["primary_calls"] += 1
                if state["primary_calls"] == n_callers:
                    primary_ready.set()
            raise RuntimeError("connection refused")

        def fallback_create(**kwargs):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                if state["active"] == target:
                    candidate_ready.set()
            assert release.wait(2)
            with lock:
                state["active"] -= 1
            return _valid_response()

    def client(create, endpoint):
        return SimpleNamespace(
            base_url=endpoint, api_key="SENTINEL_PROFILE_KEY",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )

    return state, release, primary_ready, candidate_ready, client(
        primary_create, "https://primary.invalid/v1",
    ), client(fallback_create, "https://fallback.invalid/anthropic")


def _patch_fallback_probe(config, primary, fallback, *, async_mode):
    stack = ExitStack()
    for target, kwargs in (
        ("_get_auxiliary_task_config", {"return_value": config}),
        ("_resolve_task_provider_model", {"return_value": ("primary", "primary-model", None, None, None)}),
        ("_get_cached_client", {"return_value": (primary, "primary-model")}),
        ("resolve_provider_client", {"return_value": (fallback, "fallback-model")}),
        ("_is_transient_transport_error", {"return_value": False}),
        ("_candidate_context_window", {"return_value": None}),
    ):
        stack.enter_context(patch("agent.auxiliary_client." + target, **kwargs))
    if async_mode:
        stack.enter_context(patch(
            "agent.auxiliary_client._to_async_client",
            return_value=(fallback, "fallback-model"),
        ))
    return stack


def _run_sync_fallback_limit(limit, n_callers, *, task_limit=None):
    target = 1 if limit == 1 else n_callers
    task_limit = n_callers if task_limit is None else task_limit
    state, release, primary_ready, candidate_ready, primary, fallback = _fallback_probe_clients(False, n_callers, target)
    errors = []

    def worker():
        try:
            call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        except BaseException as exc:
            errors.append(exc)

    with _patch_fallback_probe(_fallback_test_config(limit, task_limit=task_limit), primary, fallback, async_mode=False):
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_callers)]
        for thread in threads:
            thread.start()
        try:
            assert getattr(primary_ready, "wait")(2) and getattr(candidate_ready, "wait")(2)
            observed = state["max_active"]
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=2)

    assert not errors
    assert not any(thread.is_alive() for thread in threads)
    return observed


@pytest.mark.parametrize(("limit", "n_callers", "expected"), [(1, 4, 1), (7, 7, 7)])
def test_real_sync_first_fallback_candidate_uses_configured_scope(limit, n_callers, expected):
    assert _run_sync_fallback_limit(limit, n_callers) == expected


def test_real_sync_equal_task_and_fallback_limits_do_not_deadlock():
    assert _run_sync_fallback_limit(1, 1, task_limit=1) == 1


async def _run_async_fallback_limit(limit, n_callers, *, task_limit=None):
    target = 1 if limit == 1 else n_callers
    task_limit = n_callers if task_limit is None else task_limit
    state, release, primary_ready, candidate_ready, primary, fallback = _fallback_probe_clients(True, n_callers, target)

    async def worker():
        await async_call_llm(
            task="compression", messages=[{"role": "user", "content": "hi"}],
        )

    with _patch_fallback_probe(_fallback_test_config(limit, task_limit=task_limit), primary, fallback, async_mode=True):
        tasks = [asyncio.create_task(worker()) for _ in range(n_callers)]
        try:
            await asyncio.wait_for(primary_ready.wait(), timeout=2)
            await asyncio.wait_for(candidate_ready.wait(), timeout=2)
            observed = state["max_active"]
        finally:
            release.set()
            _done, pending = await asyncio.wait(tasks, timeout=2)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    return observed


@pytest.mark.asyncio
@pytest.mark.parametrize(("limit", "n_callers", "expected"), [(1, 4, 1), (7, 7, 7)])
async def test_real_async_first_fallback_candidate_uses_configured_scope(limit, n_callers, expected):
    assert await _run_async_fallback_limit(limit, n_callers) == expected


@pytest.mark.asyncio
async def test_real_async_equal_task_and_fallback_limits_do_not_deadlock():
    assert await _run_async_fallback_limit(1, 1, task_limit=1) == 1


class TestGetTaskMaxConcurrency:
    def test_returns_none_for_missing_task(self):
        assert _get_task_max_concurrency(None) is None
        assert _get_task_max_concurrency("") is None

    def test_returns_none_when_unset(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={}
        ):
            assert _get_task_max_concurrency("title_generation") is None

    def test_does_not_reuse_vision_cpu_limit_for_llm_calls(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 1},
        ):
            assert _get_task_max_concurrency("vision") is None

    @pytest.mark.parametrize("raw", [3, "3"])
    def test_returns_int_when_configured(self, raw):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": raw},
        ):
            assert _get_task_max_concurrency("compression") == 3

    @pytest.mark.parametrize(
        "raw",
        [None, "not-a-number", 0, -2, True, False, 1.5, float("nan"), float("inf"), "1.5"],
    )
    def test_compression_defaults_to_two_for_invalid_task_limit(self, raw):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": raw},
        ):
            assert _get_task_max_concurrency("compression") == 2

    def test_compression_defaults_to_two_when_limit_is_absent(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={}
        ):
            assert _get_task_max_concurrency("compression") == 2


    def test_selected_fallback_entry_overrides_task_limit(self):
        config = {
            "max_concurrency": 5,
            "fallback_chain": [{"max_concurrency": 3}],
        }
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value=config,
        ):
            assert _get_task_max_concurrency(
                "compression", "fallback_chain[0](openai-codex)"
            ) == 3

    def test_invalid_fallback_entry_uses_task_limit(self):
        config = {
            "max_concurrency": 5,
            "fallback_chain": [{"max_concurrency": "invalid"}],
        }
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value=config,
        ):
            assert _get_task_max_concurrency(
                "compression", "fallback_chain[0](openai-codex)"
            ) == 5

    def test_route_info_keeps_selected_fallback_label(self):
        route_info = {}

        _record_route_info(
            route_info,
            "openai-codex",
            "codex-model",
            "fallback_chain[0](openai-codex)",
        )

        assert route_info == {
            "provider": "openai-codex",
            "model": "codex-model",
            "fallback_label": "fallback_chain[0](openai-codex)",
        }


class TestSemaphoreCache:
    def test_sync_returns_none_when_unset(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={}
        ):
            assert _acquire_sync_aux_semaphore("title_generation") is None

    def test_sync_reuses_semaphore_for_same_limit(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 2},
        ):
            sem1 = _acquire_sync_aux_semaphore("compression")
            sem2 = _acquire_sync_aux_semaphore("compression")
            assert sem1 is sem2

    def test_sync_rebuilds_when_limit_changes(self):
        cfg = {"max_concurrency": 2}
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value=cfg,
        ):
            sem1 = _acquire_sync_aux_semaphore("compression")
            cfg["max_concurrency"] = 5
            sem2 = _acquire_sync_aux_semaphore("compression")
            assert sem1 is not sem2

    def test_sync_unequal_fallback_limit_keeps_primary_permits_live(self):
        """A held fallback permit must not replace the active primary cap."""
        config = _fallback_test_config(1, task_limit=2)
        label = "fallback_chain[0](fallback-provider)"
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config):
            primary = _acquire_sync_aux_semaphore("compression")
            assert primary is not None
            assert primary.acquire(blocking=False)
            fallback = _acquire_sync_aux_semaphore("compression", label)
            assert fallback is not None
            assert fallback is not primary
            assert fallback.acquire(blocking=False)
            resumed_primary = _acquire_sync_aux_semaphore("compression")
            assert resumed_primary is primary
            assert resumed_primary.acquire(blocking=False)
            assert not resumed_primary.acquire(blocking=False)
            resumed_primary.release()
            fallback.release()
            primary.release()

    @pytest.mark.asyncio
    async def test_async_unequal_fallback_limit_keeps_primary_permits_live(self):
        """Async fallback permits must not replace the active loop-local cap."""
        config = _fallback_test_config(1, task_limit=2)
        label = "fallback_chain[0](fallback-provider)"
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=config):
            primary = _acquire_async_aux_semaphore("compression")
            assert primary is not None
            await primary.acquire()
            fallback = _acquire_async_aux_semaphore("compression", label)
            assert fallback is not None
            assert fallback is not primary
            await fallback.acquire()
            resumed_primary = _acquire_async_aux_semaphore("compression")
            assert resumed_primary is primary
            await resumed_primary.acquire()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(resumed_primary.acquire(), timeout=0.05)
            resumed_primary.release()
            fallback.release()
            primary.release()

    @pytest.mark.asyncio
    async def test_async_semaphore_cache_does_not_cross_event_loops(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 2},
        ):
            local = _acquire_async_aux_semaphore("compression")
            other = []

            def acquire_on_other_loop():
                other.append(asyncio.run(_acquire_on_running_loop()))

            thread = threading.Thread(target=acquire_on_other_loop)
            thread.start()
            thread.join(timeout=1)

        assert not thread.is_alive()
        assert other and other[0] is not local

    @pytest.mark.asyncio
    async def test_async_reuses_semaphore_within_same_loop(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 2},
        ):
            sem1 = _acquire_async_aux_semaphore("compression")
            sem2 = _acquire_async_aux_semaphore("compression")
            assert sem1 is sem2

    def test_async_returns_none_with_no_running_loop(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 2},
        ):
            # Called outside an asyncio loop — should bail rather than crash.
            assert _acquire_async_aux_semaphore("compression") is None


class TestSyncCallEnforcesLimit:
    def test_call_llm_caps_concurrent_inflight(self):
        limit = 2
        n_callers = 6

        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_create(**kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                if active > max_active:
                    max_active = active
            try:
                time.sleep(0.05)
            finally:
                with lock:
                    active -= 1
            return MagicMock()

        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create.side_effect = fake_create

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda resp, _task, **_kwargs: resp,
            ),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": limit},
            ),
        ):
            threads = [
                threading.Thread(
                    target=lambda: call_llm(
                        task="title_generation",
                        messages=[{"role": "user", "content": "hi"}],
                    )
                )
                for _ in range(n_callers)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        assert max_active <= limit, f"observed {max_active} > limit {limit}"
        assert client.chat.completions.create.call_count == n_callers

    def test_call_llm_unlimited_when_not_configured(self):
        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create.return_value = MagicMock()

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda resp, _task, **_kwargs: resp,
            ),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={},
            ),
        ):
            # With no max_concurrency in config, no semaphore is acquired.
            call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert client.chat.completions.create.call_count == 1

    def test_semaphore_released_on_exception(self):
        """Errors inside call_llm must release the semaphore so the next call proceeds."""
        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create.side_effect = RuntimeError("boom")

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda resp, _task, **_kwargs: resp,
            ),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": 1},
            ),
        ):
            for _ in range(3):
                with pytest.raises(RuntimeError, match="boom"):
                    call_llm(
                        task="title_generation",
                        messages=[{"role": "user", "content": "hi"}],
                    )

    def test_stream_holds_permit_until_consumed_and_preserves_options(self):
        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create.side_effect = [iter(["chunk"]), MagicMock()]
        second_call_started = threading.Event()

        def make_second_call():
            second_call_started.set()
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "second"}],
            )

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda response, _task, **_kwargs: response,
            ),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": 1},
            ),
        ):
            stream = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "first"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            thread = threading.Thread(target=make_second_call)
            thread.start()
            assert second_call_started.wait(timeout=1)
            time.sleep(0.05)
            assert client.chat.completions.create.call_count == 1
            assert list(stream) == ["chunk"]
            thread.join(timeout=1)

        assert not thread.is_alive()
        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[0].kwargs["stream"] is True
        assert client.chat.completions.create.call_args_list[0].kwargs["stream_options"] == {
            "include_usage": True
        }

    def test_api_mode_is_forwarded_to_client_resolution(self):
        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create.return_value = MagicMock()

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ) as get_client,
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda response, _task, **_kwargs: response,
            ),
        ):
            call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                api_mode="codex_responses",
            )

        assert get_client.call_args.kwargs["api_mode"] == "codex_responses"


class TestAsyncCallEnforcesLimit:
    @pytest.mark.asyncio
    async def test_async_call_llm_caps_concurrent_inflight(self):
        limit = 2
        n_callers = 6

        active = 0
        max_active = 0

        async def fake_create(**kwargs):
            nonlocal active, max_active
            active += 1
            if active > max_active:
                max_active = active
            try:
                await asyncio.sleep(0.05)
            finally:
                active -= 1
            return MagicMock()

        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create = AsyncMock(side_effect=fake_create)

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda resp, _task, **_kwargs: resp,
            ),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": limit},
            ),
        ):
            await asyncio.gather(*[
                async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                )
                for _ in range(n_callers)
            ])

        assert max_active <= limit, f"observed {max_active} > limit {limit}"
        assert client.chat.completions.create.await_count == n_callers

    @pytest.mark.asyncio
    async def test_async_semaphore_released_on_exception(self):
        client = MagicMock()
        client.base_url = "https://example.test/v1"
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "test-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "test-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda resp, _task, **_kwargs: resp,
            ),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": 1},
            ),
        ):
            for _ in range(3):
                with pytest.raises(RuntimeError, match="boom"):
                    await async_call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": "hi"}],
                    )


def test_sync_fallback_route_setup_failure_releases_candidate_permit():
    primary = SimpleNamespace(
        base_url="https://primary.invalid/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=MagicMock(side_effect=RuntimeError("connection refused")),
        )),
    )
    fallback = SimpleNamespace(
        base_url="https://fallback.invalid/anthropic",
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=MagicMock(return_value=_valid_response()),
        )),
    )
    with ExitStack() as stack:
        stack.enter_context(_patch_fallback_probe(_fallback_test_config(1), primary, fallback, async_mode=False))
        first = call_llm(task="compression", messages=[{"role": "user", "content": "first"}])
        response = call_llm(task="compression", messages=[{"role": "user", "content": "second"}])

    assert first.choices[0].message.content == "ok"
    assert response.choices[0].message.content == "ok"
    assert fallback.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_async_fallback_conversion_failure_releases_candidate_permit():
    async def primary_create(**kwargs):
        raise RuntimeError("connection refused")

    async def fallback_create(**kwargs):
        return _valid_response()

    primary = SimpleNamespace(
        base_url="https://primary.invalid/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=primary_create))),
    )
    fallback = SimpleNamespace(
        base_url="https://fallback.invalid/anthropic",
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=fallback_create))),
    )
    convert_calls = 0

    def convert(client, model, **kwargs):
        nonlocal convert_calls
        convert_calls += 1
        if convert_calls == 1:
            raise RuntimeError("async conversion failed")
        return client, model

    with ExitStack() as stack:
        stack.enter_context(_patch_fallback_probe(_fallback_test_config(1), primary, fallback, async_mode=True))
        stack.enter_context(patch("agent.auxiliary_client._to_async_client", side_effect=convert))
        with pytest.raises(RuntimeError, match="async conversion failed"):
            await async_call_llm(task="compression", messages=[{"role": "user", "content": "first"}])
        response = await asyncio.wait_for(
            async_call_llm(task="compression", messages=[{"role": "user", "content": "second"}]),
            timeout=1,
        )

    assert response.choices[0].message.content == "ok"
    assert convert_calls == 2
    assert fallback.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_selected_async_cancellation_releases_permit_without_replan():
    started = asyncio.Event()
    release = asyncio.Event()
    fallback_calls = 0

    async def primary_create(**kwargs):
        raise RuntimeError("connection refused")

    async def fallback_create(**kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        if fallback_calls == 2:
            started.set()
            await release.wait()
        return _valid_response()

    primary = SimpleNamespace(
        base_url="https://primary.invalid/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=primary_create))),
    )
    fallback = SimpleNamespace(
        base_url="https://fallback.invalid/anthropic",
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=fallback_create))),
    )
    route_info = auxiliary_client._SelectedRouteInfo()

    with _patch_fallback_probe(_fallback_test_config(1), primary, fallback, async_mode=True):
        await async_call_llm(
            task="compression", messages=[{"role": "user", "content": "first"}], route_info=route_info,
        )
        cancelled = asyncio.create_task(async_call_llm(
            task="compression", messages=[{"role": "user", "content": "cancel"}], route_info=route_info,
        ))
        await asyncio.wait_for(started.wait(), timeout=1)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        await asyncio.wait_for(async_call_llm(
            task="compression", messages=[{"role": "user", "content": "after-cancel"}], route_info=route_info,
        ), timeout=1)

    assert primary.chat.completions.create.await_count == 1
    assert fallback_calls == 3
    assert "_fallback_route" not in route_info
    assert route_info.fallback_route is not None
