"""#128: host candidate deadline must walk fallback_chain, not abort.

A live first route can keep the connection open (empty/keepalive frames)
past the host's 600s total ceiling while its own aux timeout is still
open. The host then fence-cancels the whole worker and returns
uncompressed context instead of advancing auxiliary.compression.fallback_chain.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.auxiliary_client as aux
from agent.auxiliary_client import (
    CodexAuxiliaryClient,
    _ChatStreamAccumulator,
    _aux_stream_total_ceiling,
    async_call_llm,
    aux_progress_hook,
    call_llm,
)
from agent.conversation_compression import (
    CompressionCommitFence,
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
        # No host override keeps the existing generous single-call ceiling.
        assert resolve(aux_timeout=30.0, host_candidate_deadline=None) == _aux_stream_total_ceiling(30.0)
        with host_cm(0.2):
            assert _aux_stream_total_ceiling(30.0) == 0.2

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
                ceiling=600.0,
                since_progress=120.0,
                had_meaningful_progress=False,
            )
            == "idle"
        )
        assert (
            classify(
                idle=120.0,
                waited=600.0,
                ceiling=600.0,
                since_progress=5.0,
                had_meaningful_progress=True,
            )
            == "total_ceiling"
        )
        assert (
            classify(
                idle=120.0,
                waited=30.0,
                ceiling=600.0,
                since_progress=30.0,
                had_meaningful_progress=True,
            )
            == "candidate_fallback"
        )

    def test_empty_frames_do_not_count_as_summary_progress(self):
        acc = _ChatStreamAccumulator(model="m", total_ceiling=5.0)
        ticks = {"n": 0}
        with aux_progress_hook(lambda: ticks.__setitem__("n", ticks["n"] + 1)):
            acc.feed(_empty_chunk())
        assert ticks["n"] == 0
        with aux_progress_hook(lambda: ticks.__setitem__("n", ticks["n"] + 1)):
            acc.feed(_text_chunk("hello"))
        assert ticks["n"] == 1

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
