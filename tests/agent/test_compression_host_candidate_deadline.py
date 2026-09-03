"""#128: host candidate deadline must walk fallback_chain, not abort.

A live first route can keep the connection open (empty/keepalive frames)
past the configured compression deadline while its own aux timeout is still
open. The host then fence-cancels the whole worker and returns
uncompressed context instead of advancing auxiliary.compression.fallback_chain.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.auxiliary_client as aux
from agent.auxiliary_client import (
    AsyncAnthropicAuxiliaryClient,
    AsyncCodexAuxiliaryClient,
    AnthropicAuxiliaryClient,
    CodexAuxiliaryClient,
    _ChatStreamAccumulator,
    _aux_async_create_callback,
    _aux_stream_total_ceiling,
    _aux_sync_create_callback,
    _call_fallback_candidate_sync,
    _retry_same_provider_async,
    _retry_same_provider_sync,
    async_call_llm,
    aux_progress_hook,
    call_llm,
    _create_with_progress,
)
from agent.conversation_compression import (
    CompressionCommitFence,
    resolve_context_compression_timeouts,
    run_compress_context_with_progress_timeout,
)


def _ok(text: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def _empty_chunk() -> SimpleNamespace:
    return SimpleNamespace(
        id="k",
        model="m",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, reasoning=None, tool_calls=None),
                finish_reason=None,
            )
        ],
        usage=None,
    )


def _text_chunk(text: str, finish: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id="t",
        model="m",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, reasoning=None, tool_calls=None),
                finish_reason=finish,
            )
        ],
        usage=None,
    )


class _SlowKeepaliveStream:
    """Yields empty frames longer than the host candidate deadline."""

    def __init__(self, seconds: float):
        self._seconds = seconds
        self.closed = False

    def __iter__(self):
        deadline = time.monotonic() + self._seconds
        while time.monotonic() < deadline:
            time.sleep(0.02)
            yield _empty_chunk()
        yield _text_chunk("too-late-first", finish="stop")

    def close(self):
        self.closed = True


class _SlowKeepaliveAsyncStream:
    def __init__(self, seconds: float):
        self._seconds = seconds
        self.closed = False
        self._started = 0.0
        self._done = False

    def __aiter__(self):
        self._started = time.monotonic()
        return self

    async def __anext__(self):
        import asyncio

        if time.monotonic() - self._started < self._seconds:
            await asyncio.sleep(0.02)
            return _empty_chunk()
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return _text_chunk("too-late-first", finish="stop")

    async def close(self):
        self.closed = True


def _patch_chain(primary, first_client, second_client):
    return (
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "primary-model")),
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("opencode-go", "primary-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            side_effect=[
                (first_client, "slow-model", "fallback_chain[0](slow)"),
                (second_client, "fast-model", "fallback_chain[1](fast)"),
            ],
        ),
        patch("agent.auxiliary_client._try_main_agent_model_fallback"),
        patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False),
    )


class TestHostCandidateDeadlineAdvancesFallback:
    def test_deadline_helper_prefers_shorter_host_candidate(self):
        resolve = getattr(aux, "resolve_aux_attempt_deadline", None)
        host_cm = getattr(aux, "aux_host_candidate_deadline", None)
        assert callable(resolve), "shared attempt-deadline helper is required"
        assert callable(host_cm), "host candidate deadline context is required"
        assert resolve(aux_timeout=30.0, host_candidate_deadline=0.2) == 0.2
        # No host override must preserve the configured local timeout.
        assert resolve(aux_timeout=30.0, host_candidate_deadline=None) == 30.0
        assert resolve(aux_timeout=1700.0, host_candidate_deadline=None) == 1700.0
        with host_cm(0.2):
            assert _aux_stream_total_ceiling(30.0) == 0.2

    def test_fallback_candidate_budget_expands_host_ceiling(self):
        idle, ceiling = resolve_context_compression_timeouts({
            "context_timeout_seconds": 120,
            "context_total_ceiling_seconds": 600,
            "auxiliary": {
                "compression": {
                    "fallback_chain": [
                        {"provider": "local", "model": "slow", "timeout": 1700},
                        {"provider": "local", "model": "slower", "timeout": 3300},
                    ],
                },
            },
        })
        assert idle == 120.0
        assert ceiling == 3300.0

    @pytest.mark.parametrize(
        "invalid", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0]
    )
    def test_deadline_helper_rejects_invalid_configured_values(self, invalid):
        assert aux.resolve_aux_attempt_deadline(invalid) == 30.0
        assert aux.resolve_aux_attempt_deadline(30.0, invalid) == 0.0

    @pytest.mark.parametrize(
        "invalid", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0]
    )
    def test_per_entry_timeout_rejects_invalid_values(self, invalid):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"fallback_chain": [{"timeout": invalid}]},
        ):
            assert aux._fallback_entry_timeout(
                "compression", "fallback_chain[0](stale)"
            ) is None
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"timeout": invalid},
        ):
            assert aux._get_task_timeout("compression") == 30.0

    def test_slow_infinite_timeout_never_reaches_event_wait_overflow(self):
        client = MagicMock()

        def _slow_create(**_kwargs):
            time.sleep(0.05)
            return _ok("bounded")

        client.chat.completions.create.side_effect = _slow_create
        with aux_progress_hook(lambda: None):
            result = _create_with_progress(
                client, {"model": "m", "messages": [], "timeout": float("inf")}
            )
        assert result.choices[0].message.content == "bounded"

    def test_repeated_bounded_timeouts_reap_daemon_workers(self):
        finished = threading.Event()
        client = MagicMock()

        def _blocked_create(**_kwargs):
            time.sleep(0.05)
            finished.set()
            return iter([_text_chunk("late", finish="stop")])

        client.chat.completions.create.side_effect = _blocked_create
        for _ in range(3):
            with aux_progress_hook(lambda: None), aux.aux_host_candidate_deadline(0.01):
                with pytest.raises(TimeoutError):
                    _create_with_progress(
                        client, {"model": "m", "messages": [], "timeout": 30}
                    )

        assert finished.wait(timeout=1)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and any(
            thread.name.startswith("hermes-aux-attempt-deadline")
            for thread in threading.enumerate()
        ):
            time.sleep(0.01)
        assert not any(
            thread.name.startswith("hermes-aux-attempt-deadline")
            for thread in threading.enumerate()
        )

    def test_refreshed_sync_candidate_keeps_host_deadline(self):
        first = MagicMock()
        first.base_url = "https://stale.example/v1"
        auth = Exception("expired token")
        auth.status_code = 401
        first.chat.completions.create.side_effect = auth

        refreshed = MagicMock()
        refreshed.base_url = first.base_url
        retry_destination = aux._FallbackDestination(
            "custom", first.base_url, None, "m"
        )

        def _blocked_create(**kwargs):
            assert kwargs["stream"] is True
            time.sleep(0.20)
            return iter([_text_chunk("too late", finish="stop")])

        refreshed.chat.completions.create.side_effect = _blocked_create
        with patch(
            "agent.auxiliary_client._refresh_provider_credentials",
            return_value=True,
        ), patch(
            "agent.auxiliary_client._resolve_fallback_entry",
            return_value=(refreshed, "m"),
        ), patch(
            "agent.auxiliary_client._fallback_destination_from_entry",
            return_value=retry_destination,
        ), aux_progress_hook(lambda: None), aux.aux_host_candidate_deadline(0.05):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                _call_fallback_candidate_sync(
                    first,
                    "m",
                    "fallback_providers[0](stale)",
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=None,
                    max_tokens=None,
                    tools=None,
                    effective_timeout=30.0,
                    effective_extra_body={},
                    reasoning_config=None,
                )
        assert time.monotonic() - started < 0.15

    @pytest.mark.asyncio
    async def test_refreshed_async_candidate_keeps_host_deadline(self):
        first = MagicMock()
        first.base_url = "https://stale.example/v1"
        auth = Exception("expired token")
        setattr(auth, "status_code", 401)
        first.chat.completions.create = AsyncMock(side_effect=auth)

        refreshed = MagicMock()
        refreshed.base_url = first.base_url
        retry_destination = aux._FallbackDestination(
            "custom", first.base_url, None, "m"
        )

        async def _blocked_create(**kwargs):
            assert kwargs["stream"] is True
            await asyncio.sleep(0.20)
            return _SlowKeepaliveAsyncStream(0.0)

        refreshed.chat.completions.create = AsyncMock(side_effect=_blocked_create)
        with patch(
            "agent.auxiliary_client._refresh_provider_credentials",
            return_value=True,
        ), patch(
            "agent.auxiliary_client._resolve_fallback_entry",
            return_value=(refreshed, "m"),
        ), patch(
            "agent.auxiliary_client._fallback_destination_from_entry",
            return_value=retry_destination,
        ), aux_progress_hook(lambda: None), aux.aux_host_candidate_deadline(0.05):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await aux._call_fallback_candidate_async(
                    first,
                    "m",
                    "fallback_providers[0](stale)",
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=None,
                    max_tokens=None,
                    tools=None,
                    effective_timeout=30.0,
                    effective_extra_body={},
                    reasoning_config=None,
                )
        assert time.monotonic() - started < 0.15

    def test_sync_first_expiry_uses_second_fallback(self):
        host_cm = getattr(aux, "aux_host_candidate_deadline", None)
        assert callable(host_cm)

        primary = MagicMock()
        primary.base_url = "https://opencode.ai/zen/v1"
        primary.chat.completions.create.side_effect = Exception("usage limit reached")

        first = MagicMock()
        first.base_url = "https://slow.example/v1"

        def _first_create(**kwargs):
            if kwargs.get("stream"):
                return _SlowKeepaliveStream(1.0)
            raise AssertionError("first fallback must use the streamed attempt path")

        first.chat.completions.create.side_effect = _first_create

        second = MagicMock()
        second.base_url = "https://fast.example/v1"
        second.chat.completions.create.return_value = _ok("from second configured fallback")

        p0, p1, p2, p3, p4 = _patch_chain(primary, first, second)
        with p0, p1, p2 as configured, p3 as main_fb, p4:
            with aux_progress_hook(lambda: None), host_cm(0.15):
                result = call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                    timeout=30.0,
                )

        assert result.choices[0].message.content == "from second configured fallback"
        assert configured.call_args_list[1].kwargs["start_index"] == 1
        main_fb.assert_not_called()
        assert second.chat.completions.create.called

    @pytest.mark.asyncio
    async def test_async_first_expiry_uses_second_fallback(self):
        host_cm = getattr(aux, "aux_host_candidate_deadline", None)
        assert callable(host_cm)

        primary = MagicMock()
        primary.base_url = "https://opencode.ai/zen/v1"
        primary_err = Exception("Weekly usage limit reached")
        setattr(primary_err, "status_code", 429)
        primary.chat.completions.create = AsyncMock(side_effect=primary_err)

        first = MagicMock()
        first.base_url = "https://slow.example/v1"

        async def _first_create(**kwargs):
            if kwargs.get("stream"):
                return _SlowKeepaliveAsyncStream(1.0)
            raise AssertionError("first fallback must use the streamed attempt path")

        first.chat.completions.create = AsyncMock(side_effect=_first_create)

        second = MagicMock()
        second.base_url = "https://fast.example/v1"
        second.chat.completions.create = AsyncMock(
            return_value=_ok("from second configured fallback")
        )

        p0, p1, p2, p3, p4 = _patch_chain(primary, first, second)
        with p0, p1, p2 as configured, p3 as main_fb, p4, patch(
            "agent.auxiliary_client._to_async_client",
            side_effect=lambda client, model, **_: (client, model),
        ):
            with aux_progress_hook(lambda: None), host_cm(0.15):
                result = await async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                    timeout=30.0,
                )

        assert result.choices[0].message.content == "from second configured fallback"
        assert configured.call_args_list[1].kwargs["start_index"] == 1
        main_fb.assert_not_called()
        assert second.chat.completions.create.called

    def test_watchdog_reasons_are_distinct(self):
        classify = getattr(aux, "classify_compression_watchdog", None)
        assert callable(classify)
        assert (
            classify(
                idle=120.0,
                waited=120.0,
                ceiling=1700.0,
                since_progress=120.0,
                had_meaningful_progress=False,
            )
            == "idle"
        )
        assert (
            classify(
                idle=120.0,
                waited=1700.0,
                ceiling=1700.0,
                since_progress=5.0,
                had_meaningful_progress=True,
            )
            == "total_ceiling"
        )
        assert (
            classify(
                idle=120.0,
                waited=30.0,
                ceiling=1700.0,
                since_progress=30.0,
                had_meaningful_progress=True,
            )
            == "candidate_fallback"
        )

    def test_explicit_401_advances_configured_chain(self):
        """#129: 401 on an explicit compression primary must walk fallback_chain."""
        primary = MagicMock()
        primary.base_url = "https://pay.example/v1"
        err = Exception("invalid Pay API token")
        setattr(err, "status_code", 401)
        primary.chat.completions.create.side_effect = err

        first = MagicMock()
        first.base_url = "https://fallback.example/v1"
        first.chat.completions.create.return_value = _ok("from first configured fallback")

        second = MagicMock()
        p0, p1, p2, p3, p4 = _patch_chain(primary, first, second)
        with p0, p1, p2 as configured, p3 as main_fb, p4, patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "pay-model", None, None, None),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert result.choices[0].message.content == "from first configured fallback"
        configured.assert_called()
        main_fb.assert_not_called()

    def test_stale_configured_sync_entry_advances_to_next_entry(self):
        primary = MagicMock()
        primary.base_url = "https://primary.example/v1"
        primary_err = Exception("usage limit reached")
        primary_err.status_code = 429
        primary.chat.completions.create.side_effect = primary_err

        stale = MagicMock()
        stale.base_url = "https://stale.example/v1"
        stale_err = Exception("expired fallback token")
        stale_err.status_code = 401
        stale.chat.completions.create.side_effect = stale_err

        healthy = MagicMock()
        healthy.base_url = "https://healthy.example/v1"
        healthy.chat.completions.create.return_value = _ok("from next configured entry")

        p0, p1, p2, p3, p4 = _patch_chain(primary, stale, healthy)
        with p0, p1, p2 as configured, p3 as main_fb, p4, patch(
            "agent.auxiliary_client._try_payment_fallback",
            return_value=(None, None, ""),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert result.choices[0].message.content == "from next configured entry"
        assert configured.call_args_list[1].kwargs["start_index"] == 1
        main_fb.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_configured_async_entry_advances_to_next_entry(self):
        primary = MagicMock()
        primary.base_url = "https://primary.example/v1"
        primary_err = Exception("usage limit reached")
        primary_err.status_code = 429
        primary.chat.completions.create = AsyncMock(side_effect=primary_err)

        stale = MagicMock()
        stale.base_url = "https://stale.example/v1"
        stale_err = Exception("expired fallback token")
        stale_err.status_code = 401
        stale.chat.completions.create = AsyncMock(side_effect=stale_err)

        healthy = MagicMock()
        healthy.base_url = "https://healthy.example/v1"
        healthy.chat.completions.create = AsyncMock(
            return_value=_ok("from next configured entry")
        )

        p0, p1, p2, p3, p4 = _patch_chain(primary, stale, healthy)
        with p0, p1, p2 as configured, p3 as main_fb, p4, patch(
            "agent.auxiliary_client._try_payment_fallback",
            return_value=(None, None, ""),
        ), patch(
            "agent.auxiliary_client._to_async_client",
            side_effect=lambda client, model, **_: (client, model),
        ):
            result = await async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert result.choices[0].message.content == "from next configured entry"
        assert configured.call_args_list[1].kwargs["start_index"] == 1
        main_fb.assert_not_called()

    def test_explicit_safety_refusal_advances_configured_chain(self):
        """#129: provider safety/policy refusal on explicit primary walks the chain."""
        primary = MagicMock()
        primary.base_url = "https://api.openai.com/v1"
        primary.chat.completions.create.side_effect = Exception(
            "prompt was flagged by our safety system"
        )

        first = MagicMock()
        first.base_url = "https://fallback.example/v1"
        first.chat.completions.create.return_value = _ok("from first configured fallback")

        second = MagicMock()
        p0, p1, p2, p3, p4 = _patch_chain(primary, first, second)
        with p0, p1, p2 as configured, p3 as main_fb, p4, patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("openai", "gpt-5.5", None, None, None),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert result.choices[0].message.content == "from first configured fallback"
        configured.assert_called()
        main_fb.assert_not_called()

    def test_host_wait_ttfb_longer_than_idle_advances_chain(self):
        """Host fence must not win when create() TTFB exceeds idle (#128 B1)."""
        original = [{"role": "user", "content": "keep-me"}]
        compressed = [{"role": "user", "content": "from second configured fallback"}]

        primary = MagicMock()
        primary.base_url = "https://opencode.ai/zen/v1"
        primary.chat.completions.create.side_effect = Exception("usage limit reached")

        first = MagicMock()
        first.base_url = "https://slow.example/v1"

        def _slow_ttfb(**kwargs):
            time.sleep(1.3)
            if kwargs.get("stream"):
                return iter([_text_chunk("too-late-first", finish="stop")])
            raise AssertionError("first fallback must use the streamed attempt path")

        first.chat.completions.create.side_effect = _slow_ttfb

        second = MagicMock()
        second.base_url = "https://fast.example/v1"
        second.chat.completions.create.return_value = _ok("from second configured fallback")

        def worker(fence: CompressionCommitFence):
            from agent.auxiliary_client import (
                aux_host_candidate_deadline,
                aux_progress_hook,
            )

            with aux_progress_hook(fence.touch_progress), aux_host_candidate_deadline(
                fence.remaining_candidate_deadline
            ):
                result = call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                    timeout=30.0,
                )
            if not fence.begin_commit():
                return (original, "aborted")
            try:
                return (
                    [{"role": "user", "content": result.choices[0].message.content}],
                    "ok-prompt",
                )
            finally:
                fence.finish_commit()

        p0, p1, p2, p3, p4 = _patch_chain(primary, first, second)
        with p0, p1, p2 as configured, p3 as main_fb, p4:
            result_msgs, result_prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="fallback-prompt",
                idle_timeout_seconds=1.0,
                total_ceiling_seconds=5.0,
            )

        assert result_msgs == compressed
        assert result_prompt == "ok-prompt"
        assert configured.call_args_list[1].kwargs["start_index"] == 1
        main_fb.assert_not_called()

    def test_codex_internal_stream_honors_host_candidate_deadline(self):
        """Codex adapters must expire on the host candidate deadline (#128 B2)."""
        primary = MagicMock()
        primary.base_url = "https://opencode.ai/zen/v1"
        primary.chat.completions.create.side_effect = Exception("usage limit reached")

        class _KeepaliveResponses:
            def create(self, **_kwargs):
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    time.sleep(0.02)
                    yield SimpleNamespace(type="response.output_text.delta", delta="…")
                message = SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="too-late-codex")],
                )
                yield SimpleNamespace(type="response.output_item.done", item=message)
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output=[message], usage=None),
                )

        class _CodexReal:
            def __init__(self):
                self.responses = _KeepaliveResponses()
                self.api_key = "test"
                self.base_url = "https://example.test/codex"

            def close(self):
                pass

        first = CodexAuxiliaryClient(_CodexReal(), "codex-slow")
        second = MagicMock()
        second.base_url = "https://fast.example/v1"
        second.chat.completions.create.return_value = _ok("from second configured fallback")

        p0, p1, p2, p3, p4 = _patch_chain(primary, first, second)
        with p0, p1, p2 as configured, p3 as main_fb, p4:
            with aux_progress_hook(lambda: None), aux.aux_host_candidate_deadline(0.15):
                result = call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                    timeout=30.0,
                )

        assert result.choices[0].message.content == "from second configured fallback"
        assert configured.call_args_list[1].kwargs["start_index"] == 1
        main_fb.assert_not_called()
        assert second.chat.completions.create.called

    def test_internal_codex_stream_uses_dynamic_idle_and_fixed_total(self):
        class _Responses:
            def __init__(self, count):
                self.count = count

            def create(self, **_kwargs):
                for _ in range(self.count):
                    time.sleep(0.03)
                    yield SimpleNamespace(
                        type="response.output_text.delta", delta="x"
                    )
                message = SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="done")],
                )
                yield SimpleNamespace(type="response.output_item.done", item=message)
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output=[message], usage=None),
                )

        class _Real:
            api_key = "test"
            base_url = "https://example.test/codex"

            def __init__(self, count):
                self.responses = _Responses(count)

            def close(self):
                pass

        def _call(count, async_client=False):
            fence = CompressionCommitFence()
            fence.configure_host_budget(
                idle_timeout_seconds=0.1, total_ceiling_seconds=0.6
            )
            sync_client = CodexAuxiliaryClient(_Real(count), "codex")
            client = AsyncCodexAuxiliaryClient(sync_client) if async_client else sync_client
            with (
                aux_progress_hook(fence.mark_meaningful_progress),
                aux.aux_host_candidate_deadline(
                    fence.next_wait,
                    total_deadline=fence.remaining_absolute_total,
                    idle_timeout=fence.configured_idle_timeout(),
                    fence=fence,
                ),
            ):
                return client, fence

        client, _fence = _call(8)
        with (
            aux_progress_hook(_fence.mark_meaningful_progress),
            aux.aux_host_candidate_deadline(
                _fence.next_wait,
                total_deadline=_fence.remaining_absolute_total,
                idle_timeout=0.1,
                fence=_fence,
            ),
        ):
            result = _create_with_progress(
                client, {"model": "m", "messages": [], "timeout": 30}
            )
        assert result.choices[0].message.content == "done"

    @pytest.mark.asyncio
    async def test_async_internal_codex_stream_inherits_fence_context(self):
        class _Responses:
            def create(self, **_kwargs):
                for _ in range(8):
                    time.sleep(0.03)
                    yield SimpleNamespace(
                        type="response.output_text.delta", delta="x"
                    )
                message = SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="done")],
                )
                yield SimpleNamespace(type="response.output_item.done", item=message)
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output=[message], usage=None),
                )

        class _Real:
            api_key = "test"
            base_url = "https://example.test/codex"
            responses = _Responses()

        sync_client = CodexAuxiliaryClient(_Real(), "codex")
        client = AsyncCodexAuxiliaryClient(sync_client)
        fence = CompressionCommitFence()
        fence.configure_host_budget(
            idle_timeout_seconds=0.1, total_ceiling_seconds=0.6
        )
        with (
            aux_progress_hook(fence.mark_meaningful_progress),
            aux.aux_host_candidate_deadline(
                fence.next_wait,
                total_deadline=fence.remaining_absolute_total,
                idle_timeout=0.1,
                fence=fence,
            ),
        ):
            result = await _aux_async_create_callback(client, "compression")(
                {"model": "m", "messages": [], "timeout": 30}
            )
        assert result.choices[0].message.content == "done"

    def test_internal_anthropic_events_use_the_same_progress_predicate(self):
        events = [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="x"),
            )
            for _ in range(8)
        ]

        class _Messages:
            pass

        class _Real:
            messages = _Messages()

        def _fake_create(_client, _kwargs, *, on_stream_event=None, **_unused):
            for event in events:
                time.sleep(0.03)
                on_stream_event(event)
            return SimpleNamespace(content=[], usage=None, stop_reason="end_turn")

        normalized = SimpleNamespace(
            content="done", tool_calls=[], reasoning=None, finish_reason="stop"
        )
        transport = MagicMock()
        transport.normalize_response.return_value = normalized
        client = AnthropicAuxiliaryClient(
            _Real(), "claude-test", "key", "https://example.test/anthropic"
        )
        fence = CompressionCommitFence()
        fence.configure_host_budget(
            idle_timeout_seconds=0.1, total_ceiling_seconds=0.6
        )
        with (
            patch("agent.anthropic_adapter.create_anthropic_message", _fake_create),
            patch("agent.transports.get_transport", return_value=transport),
            aux_progress_hook(fence.mark_meaningful_progress),
            aux.aux_host_candidate_deadline(
                fence.next_wait,
                total_deadline=fence.remaining_absolute_total,
                idle_timeout=0.1,
                fence=fence,
            ),
        ):
            result = _create_with_progress(
                client, {"model": "m", "messages": [], "timeout": 30}
            )
        assert result.choices[0].message.content == "done"

    def test_internal_silent_codex_stream_expires_current_idle(self):
        class _Responses:
            def create(self, **_kwargs):
                time.sleep(0.2)
                yield SimpleNamespace(type="response.created")

        class _Real:
            api_key = "test"
            base_url = "https://example.test/codex"
            responses = _Responses()

        client = CodexAuxiliaryClient(_Real(), "codex")
        fence = CompressionCommitFence()
        fence.configure_host_budget(
            idle_timeout_seconds=0.05, total_ceiling_seconds=0.6
        )
        with (
            aux_progress_hook(fence.mark_meaningful_progress),
            aux.aux_host_candidate_deadline(
                fence.next_wait,
                total_deadline=fence.remaining_absolute_total,
                idle_timeout=0.05,
                fence=fence,
            ),
        ):
            with pytest.raises(TimeoutError):
                _create_with_progress(
                    client, {"model": "m", "messages": [], "timeout": 30}
                )

    def test_internal_live_codex_stream_cannot_cross_fixed_total(self):
        class _Responses:
            def create(self, **_kwargs):
                while True:
                    time.sleep(0.03)
                    yield SimpleNamespace(
                        type="response.output_text.delta", delta="x"
                    )

        class _Real:
            api_key = "test"
            base_url = "https://example.test/codex"
            responses = _Responses()

        client = CodexAuxiliaryClient(_Real(), "codex")
        fence = CompressionCommitFence()
        fence.configure_host_budget(
            idle_timeout_seconds=0.1, total_ceiling_seconds=0.2
        )
        with (
            aux_progress_hook(fence.mark_meaningful_progress),
            aux.aux_host_candidate_deadline(
                fence.next_wait,
                total_deadline=fence.remaining_absolute_total,
                idle_timeout=0.1,
                fence=fence,
            ),
        ):
            with pytest.raises(TimeoutError):
                _create_with_progress(
                    client, {"model": "m", "messages": [], "timeout": 30}
                )

    @pytest.mark.parametrize("mode", ["silent", "total"])
    def test_internal_anthropic_stream_respects_idle_and_total(self, mode):
        if mode == "silent":
            events = [SimpleNamespace(type="message_start")]
            delay = 0.2
            idle, total = 0.05, 0.6
        else:
            events = [
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="x"),
                )
                for _ in range(20)
            ]
            delay = 0.03
            idle, total = 0.1, 0.2

        class _Messages:
            pass

        class _Real:
            messages = _Messages()

        def _fake_create(_client, _kwargs, *, on_stream_event=None, **_unused):
            for event in events:
                time.sleep(delay)
                on_stream_event(event)
            return SimpleNamespace(content=[], usage=None, stop_reason="end_turn")

        transport = MagicMock()
        transport.normalize_response.return_value = SimpleNamespace(
            content="done", tool_calls=[], reasoning=None, finish_reason="stop"
        )
        client = AnthropicAuxiliaryClient(
            _Real(), "claude-test", "key", "https://example.test/anthropic"
        )
        fence = CompressionCommitFence()
        fence.configure_host_budget(
            idle_timeout_seconds=idle, total_ceiling_seconds=total
        )
        with (
            patch("agent.anthropic_adapter.create_anthropic_message", _fake_create),
            patch("agent.transports.get_transport", return_value=transport),
            aux_progress_hook(fence.mark_meaningful_progress),
            aux.aux_host_candidate_deadline(
                fence.next_wait,
                total_deadline=fence.remaining_absolute_total,
                idle_timeout=idle,
                fence=fence,
            ),
        ):
            with pytest.raises(TimeoutError):
                _create_with_progress(
                    client, {"model": "m", "messages": [], "timeout": 30}
                )

    def test_primary_refresh_sync_keeps_host_deadline(self):
        refreshed = MagicMock()
        refreshed.base_url = "https://stale.example/v1"

        def _blocked_create(**kwargs):
            assert kwargs["stream"] is True
            time.sleep(0.20)
            return iter([_text_chunk("too late", finish="stop")])

        refreshed.chat.completions.create.side_effect = _blocked_create
        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(refreshed, "m"),
        ), patch(
            "agent.auxiliary_client._effective_provider_for_client",
            return_value="custom",
        ), aux_progress_hook(lambda: None), aux.aux_host_candidate_deadline(0.05):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                _retry_same_provider_sync(
                    task="compression",
                    resolved_provider="custom",
                    resolved_model="m",
                    resolved_base_url=refreshed.base_url,
                    resolved_api_key="key",
                    resolved_api_mode=None,
                    main_runtime=None,
                    final_model="m",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=None,
                    max_tokens=None,
                    tools=None,
                    effective_timeout=30.0,
                    effective_extra_body={},
                    reasoning_config=None,
                    force_stream=False,
                )
        assert time.monotonic() - started < 0.15

    @pytest.mark.asyncio
    async def test_primary_refresh_async_keeps_host_deadline(self):
        refreshed = MagicMock()
        refreshed.base_url = "https://stale.example/v1"

        async def _blocked_create(**kwargs):
            assert kwargs["stream"] is True
            await asyncio.sleep(0.20)
            return _SlowKeepaliveAsyncStream(0.0)

        refreshed.chat.completions.create = AsyncMock(side_effect=_blocked_create)
        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(refreshed, "m"),
        ), patch(
            "agent.auxiliary_client._effective_provider_for_client",
            return_value="custom",
        ), aux_progress_hook(lambda: None), aux.aux_host_candidate_deadline(0.05):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await _retry_same_provider_async(
                    task="compression",
                    resolved_provider="custom",
                    resolved_model="m",
                    resolved_base_url=refreshed.base_url,
                    resolved_api_key="key",
                    resolved_api_mode=None,
                    final_model="m",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=None,
                    max_tokens=None,
                    tools=None,
                    effective_timeout=30.0,
                    effective_extra_body={},
                    reasoning_config=None,
                    force_stream=False,
                )
        assert time.monotonic() - started < 0.15

    @pytest.mark.asyncio
    async def test_async_initial_plain_mode_is_bounded_by_deadline(self):
        calls = []
        client = MagicMock()

        async def _blocked_create(**kwargs):
            calls.append(kwargs)
            await asyncio.sleep(0.20)
            return _ok("too late")

        client.chat.completions.create = AsyncMock(side_effect=_blocked_create)
        create = _aux_async_create_callback(client, "compression", force_stream=False)
        with aux.aux_host_candidate_deadline(0.05):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await create({"model": "m", "messages": [], "timeout": 30.0})
        assert time.monotonic() - started < 0.15
        assert calls and "stream" not in calls[0]

    @pytest.mark.asyncio
    async def test_async_temperature_retry_reuses_current_stream_budget(self):
        calls = []
        seen_budgets = []
        client = MagicMock()
        client.base_url = "https://custom.example/v1"

        class _Stream:
            def __init__(self):
                self._items = [_text_chunk("ok", finish="stop")]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        async def _create(**kwargs):
            calls.append(kwargs)
            seen_budgets.append(
                (
                    aux._aux_budget_fence(),
                    fence._absolute_total_deadline,
                    fence.configured_idle_timeout(),
                )
            )
            if len(calls) == 1:
                raise RuntimeError("Unsupported parameter: 'temperature'")
            return _Stream()

        client.chat.completions.create = AsyncMock(side_effect=_create)
        fence = CompressionCommitFence()
        fence.configure_host_budget(
            idle_timeout_seconds=5.0, total_ceiling_seconds=30.0
        )
        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "m", None, None, "chat_completions"),
        ), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "m"),
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            side_effect=AssertionError("successful retry must not enter fallback"),
        ) as configured_fallback, patch(
            "agent.auxiliary_client._try_main_agent_model_fallback",
            side_effect=AssertionError("successful retry must not enter main fallback"),
        ) as main_fallback, patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            side_effect=AssertionError("successful retry must not enter main chain"),
        ) as main_chain, patch(
            "agent.auxiliary_client._try_payment_fallback",
            side_effect=AssertionError("successful retry must not enter builtin fallback"),
        ) as builtin_fallback, aux_progress_hook(lambda: None), aux.aux_host_candidate_deadline(
            fence.next_wait,
            total_deadline=fence.remaining_absolute_total,
            idle_timeout=fence.configured_idle_timeout(),
            fence=fence,
        ):
            result = await async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                temperature=0.3,
                timeout=30.0,
            )

        assert result.choices[0].message.content == "ok"
        assert len(calls) == 2
        assert calls[0]["temperature"] == 0.3
        assert "temperature" not in calls[1]
        assert all(call["stream"] is True for call in calls)
        assert all(call["timeout"] == pytest.approx(5.0) for call in calls)
        assert seen_budgets == [seen_budgets[0], seen_budgets[0]]
        assert seen_budgets[0][0] is fence
        configured_fallback.assert_not_called()
        main_fallback.assert_not_called()
        main_chain.assert_not_called()
        builtin_fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_temperature_retry_does_not_dispatch_after_total_expiry(self):
        calls = []
        client = MagicMock()
        client.base_url = "https://custom.example/v1"
        fence = CompressionCommitFence()
        fence.configure_host_budget(
            idle_timeout_seconds=5.0, total_ceiling_seconds=30.0
        )

        async def _create(**kwargs):
            calls.append(kwargs)
            fence._absolute_total_deadline = time.monotonic()
            raise RuntimeError("Unsupported parameter: 'temperature'")

        client.chat.completions.create = AsyncMock(side_effect=_create)
        no_fallback = (None, None, "")
        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "m", None, None, "chat_completions"),
        ), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "m"),
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=no_fallback,
        ) as configured_fallback, patch(
            "agent.auxiliary_client._try_main_agent_model_fallback",
            return_value=no_fallback,
        ) as main_fallback, patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            return_value=no_fallback,
        ) as main_chain, patch(
            "agent.auxiliary_client._try_payment_fallback",
            return_value=no_fallback,
        ) as builtin_fallback, aux_progress_hook(lambda: None), aux.aux_host_candidate_deadline(
            fence.next_wait,
            total_deadline=fence.remaining_absolute_total,
            idle_timeout=fence.configured_idle_timeout(),
            fence=fence,
        ):
            with pytest.raises(RuntimeError, match="Unsupported parameter: 'temperature'"):
                await async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                    temperature=0.3,
                    timeout=30.0,
                )

        assert len(calls) == 1
        assert calls[0]["temperature"] == 0.3
        configured_fallback.assert_called_once()
        main_fallback.assert_called_once()
        main_chain.assert_not_called()
        builtin_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_temperature_attempt_cancellation_propagates(self):
        client = MagicMock()
        client.base_url = "https://custom.example/v1"
        client.chat.completions.create = AsyncMock(
            side_effect=asyncio.CancelledError()
        )
        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "m", None, None, "chat_completions"),
        ), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "m"),
        ), patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            side_effect=AssertionError("cancellation must not enter fallback"),
        ) as configured_fallback, patch(
            "agent.auxiliary_client._try_main_agent_model_fallback",
            side_effect=AssertionError("cancellation must not enter main fallback"),
        ) as main_fallback, patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            side_effect=AssertionError("cancellation must not enter main chain"),
        ) as main_chain, patch(
            "agent.auxiliary_client._try_payment_fallback",
            side_effect=AssertionError("cancellation must not enter builtin fallback"),
        ) as builtin_fallback:
            with pytest.raises(asyncio.CancelledError):
                await async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                    temperature=0.3,
                )

        assert client.chat.completions.create.await_count == 1
        configured_fallback.assert_not_called()
        main_fallback.assert_not_called()
        main_chain.assert_not_called()
        builtin_fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_fallback_refresh_preserves_plain_mode_without_progress(self):
        first = MagicMock()
        first.base_url = "https://fallback.example/v1"
        auth = Exception("expired token")
        setattr(auth, "status_code", 401)
        first.chat.completions.create = AsyncMock(side_effect=auth)

        refreshed = MagicMock()
        refreshed.base_url = first.base_url
        retry_destination = aux._FallbackDestination(
            "custom", first.base_url, None, "m"
        )
        calls = []

        async def _plain_only(**kwargs):
            calls.append(kwargs)
            if kwargs.get("stream"):
                raise RuntimeError("provider rejects streamed requests")
            return _ok("plain refreshed")

        refreshed.chat.completions.create = AsyncMock(side_effect=_plain_only)
        with patch(
            "agent.auxiliary_client._refresh_provider_credentials",
            return_value=True,
        ), patch(
            "agent.auxiliary_client._resolve_fallback_entry",
            return_value=(refreshed, "m"),
        ), patch(
            "agent.auxiliary_client._fallback_destination_from_entry",
            return_value=retry_destination,
        ), patch(
            "agent.auxiliary_client._auth_refresh_provider_for_route",
            return_value="custom",
        ):
            result = await aux._call_fallback_candidate_async(
                first,
                "m",
                "fallback_providers[0](custom)",
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
                temperature=None,
                max_tokens=None,
                tools=None,
                effective_timeout=30.0,
                effective_extra_body={},
                reasoning_config=None,
            )
        assert result is not None
        assert result.choices[0].message.content == "plain refreshed"
        assert calls and all(not call.get("stream") for call in calls)

    def test_sync_primary_and_fallback_callback_modes_match(self):
        clients = []
        for _label in (
            "primary-initial",
            "primary-refresh",
            "fallback-initial",
            "fallback-refresh",
        ):
            client = MagicMock()
            client.chat.completions.create.side_effect = lambda **kwargs: (
                iter([_text_chunk("ok", finish="stop")])
                if kwargs.get("stream")
                else _ok("plain")
            )
            clients.append(client)
        with aux_progress_hook(lambda: None):
            for client in clients:
                result = _aux_sync_create_callback(
                    client, "compression", force_stream=False,
                )({"model": "m", "messages": [], "timeout": 30.0})
                assert result.choices[0].message.content == "ok"
                assert client.chat.completions.create.call_args.kwargs["stream"] is True
