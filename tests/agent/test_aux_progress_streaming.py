"""Tests for the auxiliary forward-progress streaming layer.

Slow summary models must not be punished like hung ones (#see PR): when a
forward-progress hook is installed (context compression), the primary
auxiliary call streams and ticks the hook per chunk, so outer watchdogs
(gateway session hygiene) can extend their deadline on liveness. Without a
hook, behavior is byte-for-byte the old non-streaming call.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.conversation_compression as compression
from agent.auxiliary_client import (
    _ChatStreamAccumulator,
    _acreate_with_stream,
    _aggregate_chat_stream,
    _aggregate_chat_stream_async,
    _aux_stream_total_ceiling,
    _create_with_progress,
    _notify_aux_progress,
    _provider_requires_stream,
    CodexAuxiliaryClient,
    aux_host_candidate_deadline,
    aux_progress_hook,
)
from agent.conversation_compression import CompressionCommitFence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(content=None, reasoning=None, finish_reason=None, usage=None,
           tool_calls=None, model="m1", chunk_id="c1"):
    delta = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(
        id=chunk_id, model=model, choices=[choice], usage=usage,
    )


class _FakeClient:
    """OpenAI-shaped client whose create() returns a canned value or stream."""

    def __init__(self, response=None, stream_chunks=None, stream_error=None):
        self.calls = []
        self._response = response
        self._stream_chunks = stream_chunks
        self._stream_error = stream_error
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            if self._stream_error is not None:
                raise self._stream_error
            return iter(self._stream_chunks or [])
        return self._response


class _TimedSyncStream:
    def __init__(self, count: int, interval: float):
        self._count = count
        self._interval = interval

    def __iter__(self):
        for index in range(self._count):
            time.sleep(self._interval)
            yield _chunk(content=str(index), finish_reason="stop")


class _TimedAsyncStream:
    def __init__(self, count: int, interval: float):
        self._remaining = count
        self._interval = interval

    def __aiter__(self):
        return self

    async def __anext__(self):
        import asyncio

        if self._remaining <= 0:
            raise StopAsyncIteration
        self._remaining -= 1
        await asyncio.sleep(self._interval)
        return _chunk(content="x", finish_reason="stop")


_COMPLETE = SimpleNamespace(
    id="r1", model="m1", object="chat.completion",
    choices=[SimpleNamespace(
        index=0,
        message=SimpleNamespace(role="assistant", content="non-streamed"),
        finish_reason="stop",
    )],
    usage=None,
)


# ---------------------------------------------------------------------------
# aux_progress_hook plumbing
# ---------------------------------------------------------------------------

class TestAuxProgressHook:
    def test_hook_installed_and_restored(self):
        ticks = []
        with aux_progress_hook(lambda: ticks.append(1)):
            _notify_aux_progress("progress")
        _notify_aux_progress()  # outside — must not tick
        assert ticks == [1]



    def test_hook_is_thread_local(self):
        ticks = []
        seen_in_thread = []

        def _other_thread():
            # No hook installed on this thread.
            _notify_aux_progress()
            seen_in_thread.append(len(ticks))

        with aux_progress_hook(lambda: ticks.append(1)):
            t = threading.Thread(target=_other_thread)
            t.start()
            t.join()
        assert seen_in_thread == [0]


# ---------------------------------------------------------------------------
# _create_with_progress
# ---------------------------------------------------------------------------

class TestCreateWithProgress:

    def test_sync_stream_establishment_uses_idle_and_total_minimum(self):
        client = _FakeClient()

        def _slow_create(**_kwargs):
            time.sleep(0.15)
            return iter([_chunk(content="late", finish_reason="stop")])

        client.chat.completions.create = _slow_create
        with (
            aux_progress_hook(lambda: None),
            aux_host_candidate_deadline(
                lambda: 0.05,
                total_deadline=lambda: 0.5,
                idle_timeout=0.05,
            ),
        ):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                _create_with_progress(
                    client, {"model": "m1", "messages": [], "timeout": 30}
                )
        assert time.monotonic() - started < 0.12

    def test_sync_plain_call_is_bounded_by_active_host_deadline(self):
        calls = []
        client = _FakeClient()

        def _slow_create(**_kwargs):
            calls.append(True)
            time.sleep(0.15)
            return _COMPLETE

        client.chat.completions.create = _slow_create
        with aux_host_candidate_deadline(0.05):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                _create_with_progress(
                    client, {"model": "m1", "messages": [], "timeout": 30}
                )
        assert time.monotonic() - started < 0.12
        assert calls == [True]

    def test_hook_upgrades_to_streaming_and_ticks_per_chunk(self):
        chunks = [
            _chunk(reasoning="thinking..."),
            _chunk(content="Hello "),
            _chunk(content="world", finish_reason="stop",
                   usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2,
                                         total_tokens=7)),
        ]
        client = _FakeClient(stream_chunks=chunks)
        ticks = []
        with aux_progress_hook(lambda: ticks.append(1)):
            result = _create_with_progress(
                client, {"model": "m1", "messages": [], "timeout": 30},
            )
        assert client.calls[0]["stream"] is True
        assert result.choices[0].message.content == "Hello world"
        assert result.choices[0].message.reasoning == "thinking..."
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.total_tokens == 7
        # 1 dispatch used to tick; dispatch is not summary progress (#128).
        assert len(ticks) >= len(chunks)

    def test_streaming_rejected_falls_back_to_plain_call(self):
        client = _FakeClient(
            response=_COMPLETE,
            stream_error=RuntimeError("stream is not supported by this model"),
        )
        with aux_progress_hook(lambda: None):
            result = _create_with_progress(
                client, {"model": "m1", "messages": []},
            )
        assert result is _COMPLETE
        # streamed attempt + non-streaming fallback
        assert len(client.calls) == 2
        assert client.calls[0].get("stream") is True
        assert "stream" not in client.calls[1]

    def test_streaming_rejection_fallback_keeps_absolute_total(self):
        calls = []
        client = _FakeClient()

        def _create(**kwargs):
            calls.append(kwargs)
            if kwargs.get("stream"):
                time.sleep(0.08)
                raise RuntimeError("stream is not supported by this model")
            time.sleep(0.15)
            return _COMPLETE

        client.chat.completions.create = _create
        with (
            aux_progress_hook(lambda: None),
            aux_host_candidate_deadline(
                lambda: 0.1,
                total_deadline=lambda: 0.1,
                idle_timeout=0.1,
            ),
        ):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                _create_with_progress(
                    client, {"model": "m1", "messages": [], "timeout": 30}
                )
        assert time.monotonic() - started < 0.18
        assert len(calls) == 2

    def test_streaming_rejection_after_total_does_not_dispatch_fallback(self):
        calls = []
        client = _FakeClient()

        def _create(**kwargs):
            calls.append(kwargs)
            time.sleep(0.12)
            if kwargs.get("stream"):
                raise RuntimeError("stream is not supported by this model")
            return _COMPLETE

        client.chat.completions.create = _create
        with (
            aux_progress_hook(lambda: None),
            aux_host_candidate_deadline(
                lambda: 0.1,
                total_deadline=lambda: 0.1,
                idle_timeout=0.1,
            ),
        ):
            with pytest.raises(TimeoutError):
                _create_with_progress(
                    client, {"model": "m1", "messages": [], "timeout": 30}
                )
        assert len(calls) == 1




# ---------------------------------------------------------------------------
# _aggregate_chat_stream
# ---------------------------------------------------------------------------

class TestAggregateChatStream:
    def test_tool_call_deltas_are_reassembled(self):
        tc0 = SimpleNamespace(
            index=0, id="call_1",
            function=SimpleNamespace(name="do_thing", arguments='{"a"'),
        )
        tc1 = SimpleNamespace(
            index=0, id=None,
            function=SimpleNamespace(name=None, arguments=': 1}'),
        )
        chunks = [
            _chunk(tool_calls=[tc0]),
            _chunk(tool_calls=[tc1], finish_reason="tool_calls"),
        ]
        result = _aggregate_chat_stream(iter(chunks))
        tool_calls = result.choices[0].message.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].function.name == "do_thing"
        assert tool_calls[0].function.arguments == '{"a": 1}'
        assert result.choices[0].finish_reason == "tool_calls"

    def test_whitespace_and_empty_tool_deltas_do_not_count_as_progress(self):
        empty_tool = SimpleNamespace(
            index=0,
            id="",
            function=SimpleNamespace(name="", arguments=""),
        )
        ticks = []
        with aux_progress_hook(lambda: ticks.append(True)):
            _aggregate_chat_stream(
                iter(
                    [
                        _chunk(content=" \n\t"),
                        _chunk(tool_calls=[empty_tool]),
                    ]
                )
            )
        assert ticks == []

        acc = _ChatStreamAccumulator(model="m", total_ceiling=5.0)
        with aux_progress_hook(lambda: ticks.append(True)):
            acc.feed(_chunk())
        assert ticks == []
        with aux_progress_hook(lambda: ticks.append(True)):
            acc.feed(_chunk(content="hello"))
        assert ticks == [True]


    def test_stream_close_is_called(self):
        closed = []

        class _Stream:
            def __iter__(self):
                return iter([_chunk(content="ok", finish_reason="stop")])

            def close(self):
                closed.append(True)

        result = _aggregate_chat_stream(_Stream())
        assert result.choices[0].message.content == "ok"
        assert closed == [True]

    def test_live_stream_resets_idle_without_resetting_total(self):
        started = time.monotonic()
        result = _aggregate_chat_stream(
            _TimedSyncStream(count=8, interval=0.03),
            idle_timeout=0.1,
            total_ceiling=0.6,
        )
        assert result.choices[0].message.content == "01234567"
        assert time.monotonic() - started > 0.1

    def test_silent_stream_expires_within_idle_window(self):
        started = time.monotonic()

        def _silent():
            time.sleep(0.2)
            yield _chunk(content="late")

        with pytest.raises(TimeoutError, match="idle"):
            _aggregate_chat_stream(
                _silent(), idle_timeout=0.05, total_ceiling=0.6
            )
        assert time.monotonic() - started < 0.15

    def test_progress_cannot_extend_absolute_total(self):
        ticks = []
        with aux_progress_hook(lambda: ticks.append(True)):
            with pytest.raises(TimeoutError, match="total ceiling"):
                _aggregate_chat_stream(
                    _TimedSyncStream(count=20, interval=0.02),
                    idle_timeout=0.1,
                    total_ceiling=0.18,
                )
        assert len(ticks) >= 5



# ---------------------------------------------------------------------------
# Ceiling arithmetic
# ---------------------------------------------------------------------------

class TestStreamCeiling:
    def test_uses_configured_timeout(self):
        assert _aux_stream_total_ceiling(30) == 30.0

    def test_host_idle_ttfb_and_absolute_total_are_separate(self):
        with aux_host_candidate_deadline(
            lambda: 0.1,
            total_deadline=lambda: 0.6,
            idle_timeout=0.1,
        ):
            assert _aux_stream_total_ceiling(30) == 0.6


    def test_none_timeout_uses_default(self):
        assert _aux_stream_total_ceiling(None) == 30.0


# ---------------------------------------------------------------------------
# CompressionCommitFence progress surface
# ---------------------------------------------------------------------------

class TestFenceProgress:
    def test_touch_progress_resets_idle_clock(self):
        fence = CompressionCommitFence()
        time.sleep(0.05)
        assert fence.seconds_since_progress() >= 0.04
        fence.touch_progress()
        assert fence.seconds_since_progress() < 0.05

    def test_fence_hook_wiring_matches_compressor_usage(self):
        # The shared seam marks the fence before notifying the host hook.
        fence = CompressionCommitFence()
        time.sleep(0.05)
        ticks = []
        with (
            aux_progress_hook(lambda: ticks.append(1)),
            aux_host_candidate_deadline(None, fence=fence),
        ):
            assert _notify_aux_progress("progress") is True
        assert fence.seconds_since_progress() < 0.05
        assert ticks == [1]

    def test_late_codex_progress_cannot_mutate_cancelled_fence(self):
        started = threading.Event()
        release = threading.Event()

        class _Responses:
            def create(self, **_kwargs):
                started.set()
                assert release.wait(timeout=1)
                yield SimpleNamespace(
                    type="response.output_text.delta", delta="late"
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

        real_client = SimpleNamespace(
            api_key="test",
            base_url="https://example.test/codex",
            responses=_Responses(),
            close=lambda: None,
        )
        client = CodexAuxiliaryClient(real_client, "codex")
        fence = CompressionCommitFence()
        fence.configure_host_budget(
            idle_timeout_seconds=1.0, total_ceiling_seconds=2.0
        )
        host_ticks = []
        outcome = {}

        def _worker():
            try:
                with (
                    aux_progress_hook(lambda: host_ticks.append(True)),
                    aux_host_candidate_deadline(
                        fence.next_wait,
                        total_deadline=fence.remaining_absolute_total,
                        idle_timeout=1.0,
                        fence=fence,
                    ),
                ):
                    outcome["result"] = _create_with_progress(
                        client, {"model": "m", "messages": [], "timeout": 30}
                    )
            except BaseException as exc:  # pragma: no cover - asserted below
                outcome["exc"] = exc

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        assert started.wait(timeout=1)
        assert fence.cancel_before_commit() is True
        last_progress = fence._last_progress
        release.set()
        worker.join(timeout=1)

        assert not worker.is_alive()
        assert "exc" not in outcome
        assert outcome["result"].choices[0].message.content == "done"
        assert fence._last_progress == last_progress
        assert host_ticks == []
        assert fence.begin_commit() is False

    def _run_external_transition_drain(self, transition, monkeypatch):
        fence = CompressionCommitFence()
        mark_returned = threading.Event()
        allow_hook_call = threading.Event()
        hook_started = threading.Event()
        allow_hook_return = threading.Event()
        original_begin = fence.begin_progress_publication

        def _begin(has_hook=False):
            token = original_begin(has_hook)
            mark_returned.set()
            assert allow_hook_call.wait(timeout=1)
            return token

        monkeypatch.setattr(fence, "begin_progress_publication", _begin)
        outcome = {}

        def _publish():
            def _hook():
                hook_started.set()
                assert allow_hook_return.wait(timeout=1)

            with aux_progress_hook(_hook):
                outcome["accepted"] = _notify_aux_progress("progress", fence=fence)

        publisher = threading.Thread(target=_publish, daemon=True)
        publisher.start()
        assert mark_returned.wait(timeout=1)
        transition_done = threading.Event()

        def _transition():
            if transition == "cancel":
                outcome["transition"] = fence.cancel_before_commit()
            elif transition == "revoke":
                fence.revoke_commit_admission()
                outcome["transition"] = True
            else:
                outcome["transition"] = fence.begin_commit()
                if outcome["transition"]:
                    fence.finish_commit()
            transition_done.set()

        waiter = threading.Thread(target=_transition, daemon=True)
        waiter.start()
        allow_hook_call.set()
        assert hook_started.wait(timeout=1)
        assert not transition_done.wait(timeout=0.05)
        allow_hook_return.set()
        publisher.join(timeout=1)
        waiter.join(timeout=1)

        assert not publisher.is_alive()
        assert not waiter.is_alive()
        assert outcome == {"accepted": True, "transition": True}
        assert fence.mark_meaningful_progress_if_active() is False

    @pytest.mark.parametrize("transition", ["cancel", "revoke", "begin"])
    def test_external_transition_waits_for_progress_hook(self, transition, monkeypatch):
        self._run_external_transition_drain(transition, monkeypatch)

    @pytest.mark.parametrize("publisher_count", [2, 3, 8])
    @pytest.mark.parametrize("transition", ["cancel", "revoke", "begin"])
    def test_reentrant_transition_drains_external_progress_hooks(
        self, transition, publisher_count
    ):
        fence = CompressionCommitFence()
        all_started = threading.Barrier(publisher_count + 1)
        release_external = threading.Event()
        transition_started = threading.Event()
        transition_done = threading.Event()
        external_finished = []
        snapshot_at_return = []
        outcome = {}

        def _publish(index):
            def _hook():
                all_started.wait(timeout=2)
                if index:
                    assert release_external.wait(timeout=2)
                    external_finished.append(index)
                    return
                transition_started.set()
                if transition == "cancel":
                    outcome["transition"] = fence.cancel_before_commit()
                elif transition == "revoke":
                    fence.revoke_commit_admission()
                    outcome["transition"] = True
                else:
                    outcome["transition"] = fence.begin_commit()
                    if outcome["transition"]:
                        fence.finish_commit()
                snapshot_at_return.extend(external_finished)
                transition_done.set()

            with aux_progress_hook(_hook):
                outcome[index] = _notify_aux_progress("progress", fence=fence)

        publishers = [
            threading.Thread(target=_publish, args=(index,), daemon=True)
            for index in range(publisher_count)
        ]
        for publisher in publishers:
            publisher.start()
        all_started.wait(timeout=2)
        assert transition_started.wait(timeout=2)
        assert not transition_done.wait(timeout=0.1)
        release_external.set()
        for publisher in publishers:
            publisher.join(timeout=2)

        assert all(not publisher.is_alive() for publisher in publishers)
        assert outcome == {
            **{index: True for index in range(publisher_count)},
            "transition": True,
        }
        assert sorted(snapshot_at_return) == list(range(1, publisher_count))
        assert fence._progress_publications == {}
        assert fence._progress_quiescent.is_set()
        assert _notify_aux_progress("late", fence=fence) is False

    @pytest.mark.parametrize(
        ("error_type", "prior_state", "prior_commit_started"),
        [
            (KeyboardInterrupt, "active", False),
            (SystemExit, "active", False),
            (RuntimeError, "active", False),
            (RuntimeError, "committed", True),
            (RuntimeError, "cancelled", False),
            (RuntimeError, "revoked", False),
        ],
    )
    def test_begin_commit_wait_failure_restores_fence(
        self, error_type, prior_state, prior_commit_started, monkeypatch
    ):
        fence = CompressionCommitFence()
        external_ready = threading.Event()
        release_external = threading.Event()

        def _publish_external():
            token = fence.begin_progress_publication(True)
            assert token
            external_ready.set()
            assert release_external.wait(timeout=1)
            fence.end_progress_publication(token)

        external = threading.Thread(target=_publish_external, daemon=True)
        external.start()
        assert external_ready.wait(timeout=1)
        own_token = fence.begin_progress_publication(True)
        assert own_token

        with fence._publication_condition:
            fence._commit_started = prior_commit_started
            if prior_state == "cancelled":
                fence._cancelled = True
            elif prior_state == "revoked":
                fence._admission_revoked = True

        error = error_type("publication wait failed")
        notifications = 0
        original_notify_all = fence._publication_condition.notify_all

        def _notify_all():
            nonlocal notifications
            notifications += 1
            original_notify_all()

        def _raise_from_wait(timeout=None):
            raise error

        monkeypatch.setattr(fence._publication_condition, "notify_all", _notify_all)
        monkeypatch.setattr(fence._publication_condition, "wait", _raise_from_wait)

        with pytest.raises(error_type) as caught:
            fence.begin_commit()
        assert caught.value is error
        assert fence._progress_transition_owner is None
        assert fence._commit_phase.is_set() is False
        assert fence._commit_started is prior_commit_started
        assert notifications >= 2
        if prior_state == "cancelled":
            assert fence._cancelled is True
        elif prior_state == "revoked":
            assert fence._admission_revoked is True

        assert fence._lock.acquire(blocking=False)
        fence._lock.release()
        release_external.set()
        external.join(timeout=1)
        assert not external.is_alive()
        fence.end_progress_publication(own_token)
        assert fence._progress_publications == {}
        assert fence._progress_quiescent.is_set()

        if prior_state == "active":
            token = fence.begin_progress_publication(True)
            assert token
            fence.end_progress_publication(token)
        else:
            assert fence.begin_progress_publication(True) is None
        if prior_state in ("active", "committed"):
            assert fence.begin_commit() is True
            fence.finish_commit()
        else:
            assert fence.begin_commit() is False
        assert fence.cancel_before_commit() is (
            prior_state in ("cancelled", "revoked")
        )
        fence.revoke_commit_admission()
        assert fence.begin_commit() is False
        assert fence.begin_progress_publication(True) is None

    @pytest.mark.parametrize("transition", ["cancel", "revoke", "begin"])
    def test_reentrant_transition_does_not_deadlock(self, transition):
        fence = CompressionCommitFence()
        outcome = {}
        finished = threading.Event()

        def _hook():
            if transition == "cancel":
                outcome[transition] = fence.cancel_before_commit()
            elif transition == "revoke":
                fence.revoke_commit_admission()
                outcome[transition] = True
            else:
                outcome[transition] = fence.begin_commit()
                if outcome[transition]:
                    fence.finish_commit()

        def _publish():
            with aux_progress_hook(_hook):
                outcome["accepted"] = _notify_aux_progress("progress", fence=fence)
            finished.set()

        publisher = threading.Thread(target=_publish, daemon=True)
        publisher.start()
        assert finished.wait(timeout=1)
        publisher.join(timeout=1)

        assert not publisher.is_alive()
        assert outcome["accepted"] is True
        assert outcome[transition] is True
        assert _notify_aux_progress("later", fence=fence) is False

    @pytest.mark.parametrize("transition", ["cancel", "revoke", "begin"])
    def test_two_reentrant_transitions_do_not_deadlock(self, transition):
        fence = CompressionCommitFence()
        hook_started = [threading.Event(), threading.Event()]
        finished = [threading.Event(), threading.Event()]
        outcome = {}
        both_hooks = threading.Barrier(2)

        def _publish(index):
            def _hook():
                hook_started[index].set()
                both_hooks.wait(timeout=1)
                if transition == "cancel":
                    outcome[(index, "transition")] = fence.cancel_before_commit()
                elif transition == "revoke":
                    fence.revoke_commit_admission()
                    outcome[(index, "transition")] = True
                else:
                    outcome[(index, "transition")] = fence.begin_commit()
                    if outcome[(index, "transition")]:
                        fence.finish_commit()

            with aux_progress_hook(_hook):
                outcome[index] = _notify_aux_progress("progress", fence=fence)
            finished[index].set()

        publishers = [
            threading.Thread(target=_publish, args=(index,), daemon=True)
            for index in range(2)
        ]
        for publisher in publishers:
            publisher.start()
        for started in hook_started:
            assert started.wait(timeout=1)
        for done in finished:
            assert done.wait(timeout=1)
        for publisher in publishers:
            publisher.join(timeout=1)

        assert all(not publisher.is_alive() for publisher in publishers)
        assert all(outcome[index] is True for index in range(2))
        assert all(outcome[(index, "transition")] is True for index in range(2))
        assert fence._progress_publications == {}
        assert fence._progress_quiescent.is_set()
        if transition == "cancel":
            assert fence._cancelled is True
            assert fence._admission_revoked is False
            assert fence._commit_started is False
        elif transition == "revoke":
            assert fence._admission_revoked is True
        else:
            assert fence._commit_started is True
            assert fence._commit_phase.is_set() is False
        assert _notify_aux_progress("late", fence=fence) is False

    @pytest.mark.parametrize("transition", ["cancel", "revoke", "begin"])
    def test_reentrant_transition_and_external_drain_do_not_deadlock(self, transition):
        fence = CompressionCommitFence()
        reentrant_done = threading.Event()
        allow_hook_return = threading.Event()
        external_started = threading.Event()
        external_done = threading.Event()
        outcome = {}

        def _hook():
            if transition == "cancel":
                outcome["reentrant"] = fence.cancel_before_commit()
            elif transition == "revoke":
                fence.revoke_commit_admission()
                outcome["reentrant"] = True
            else:
                outcome["reentrant"] = fence.begin_commit()
                if outcome["reentrant"]:
                    fence.finish_commit()
            reentrant_done.set()
            assert allow_hook_return.wait(timeout=1)

        def _publish():
            with aux_progress_hook(_hook):
                outcome["accepted"] = _notify_aux_progress("progress", fence=fence)

        publisher = threading.Thread(target=_publish, daemon=True)
        publisher.start()
        assert reentrant_done.wait(timeout=1)

        def _external():
            external_started.set()
            if transition == "cancel":
                outcome["external"] = fence.cancel_before_commit()
            elif transition == "revoke":
                fence.revoke_commit_admission()
                outcome["external"] = True
            else:
                outcome["external"] = fence.begin_commit()
                if outcome["external"]:
                    fence.finish_commit()
            external_done.set()

        external = threading.Thread(target=_external, daemon=True)
        external.start()
        assert external_started.wait(timeout=1)
        assert not external_done.wait(timeout=0.05)
        allow_hook_return.set()
        publisher.join(timeout=1)
        external.join(timeout=1)

        assert not publisher.is_alive()
        assert not external.is_alive()
        assert outcome == {"accepted": True, "reentrant": True, "external": True}
        assert fence._progress_publications == {}
        assert fence._progress_quiescent.is_set()
        assert _notify_aux_progress("late", fence=fence) is False

    def test_reentrant_revoke_waits_for_fence_lock_release(self, monkeypatch):
        fence = CompressionCommitFence()
        token_ready = threading.Event()
        allow_begin_return = threading.Event()
        original_begin = fence.begin_progress_publication

        def _begin(has_hook=False):
            token = original_begin(has_hook)
            token_ready.set()
            assert allow_begin_return.wait(timeout=1)
            return token

        monkeypatch.setattr(fence, "begin_progress_publication", _begin)
        lock_held = threading.Event()
        release_lock = threading.Event()

        def _hold_lock():
            fence._lock.acquire()
            lock_held.set()
            try:
                assert release_lock.wait(timeout=1)
            finally:
                fence._lock.release()

        hook_started = threading.Event()
        finished = threading.Event()

        def _hook():
            hook_started.set()
            fence.revoke_commit_admission()

        def _publish():
            with aux_progress_hook(_hook):
                assert _notify_aux_progress("progress", fence=fence) is True
            finished.set()

        publisher = threading.Thread(target=_publish, daemon=True)
        publisher.start()
        assert token_ready.wait(timeout=1)
        holder = threading.Thread(target=_hold_lock, daemon=True)
        holder.start()
        assert lock_held.wait(timeout=1)
        allow_begin_return.set()
        assert hook_started.wait(timeout=1)
        assert fence._admission_revoked is False
        assert not finished.wait(timeout=0.05)
        release_lock.set()
        holder.join(timeout=1)
        publisher.join(timeout=1)

        assert not holder.is_alive()
        assert not publisher.is_alive()
        assert finished.is_set()
        assert fence._admission_revoked is True
        assert fence._progress_publications == {}

    def test_hook_failure_releases_progress_publication(self):
        fence = CompressionCommitFence()

        def _raise():
            raise RuntimeError("hook failure")

        with aux_progress_hook(_raise):
            assert _notify_aux_progress("progress", fence=fence) is True
        assert fence._progress_publications == {}
        assert fence._progress_quiescent.is_set()
        assert fence.cancel_before_commit() is True

    @pytest.mark.parametrize("transition", ["cancel", "revoke", "begin"])
    def test_external_transition_waits_for_two_concurrent_progress_hooks(self, transition):
        fence = CompressionCommitFence()
        hook_started = [threading.Event(), threading.Event()]
        allow_hook_return = [threading.Event(), threading.Event()]
        hook_seen = []
        outcome = {}

        def _publish(index):
            def _hook():
                hook_started[index].set()
                assert allow_hook_return[index].wait(timeout=1)
                hook_seen.append(index)

            with aux_progress_hook(_hook):
                outcome[index] = _notify_aux_progress("progress", fence=fence)

        publishers = [
            threading.Thread(target=_publish, args=(index,), daemon=True)
            for index in range(2)
        ]
        for publisher in publishers:
            publisher.start()
        for started in hook_started:
            assert started.wait(timeout=1)

        transition_started = threading.Event()
        transition_done = threading.Event()

        def _transition():
            transition_started.set()
            if transition == "cancel":
                outcome["transition"] = fence.cancel_before_commit()
            elif transition == "revoke":
                fence.revoke_commit_admission()
                outcome["transition"] = True
            else:
                outcome["transition"] = fence.begin_commit()
                if outcome["transition"]:
                    fence.finish_commit()
            transition_done.set()

        waiter = threading.Thread(target=_transition, daemon=True)
        waiter.start()
        assert transition_started.wait(timeout=1)
        assert not transition_done.wait(timeout=0.05)
        allow_hook_return[0].set()
        publishers[0].join(timeout=1)
        assert not transition_done.wait(timeout=0.05)
        allow_hook_return[1].set()
        for publisher in publishers:
            publisher.join(timeout=1)
        waiter.join(timeout=1)

        assert all(not publisher.is_alive() for publisher in publishers)
        assert not waiter.is_alive()
        assert outcome == {0: True, 1: True, "transition": True}
        assert sorted(hook_seen) == [0, 1]
        assert fence._progress_publications == {}
        assert fence._progress_quiescent.is_set()
        assert _notify_aux_progress("late", fence=fence) is False
        if transition == "cancel":
            assert fence._cancelled is True
        elif transition == "revoke":
            assert fence._admission_revoked is True
        else:
            assert fence._commit_started is True
            assert fence._commit_phase.is_set() is False


# ---------------------------------------------------------------------------
# Stream-only providers (credit @kudi88, PR #60686)
# ---------------------------------------------------------------------------


class TestProviderRequiresStream:

    def test_normal_endpoints_are_not(self):
        assert _provider_requires_stream(
            "openrouter", "https://openrouter.ai/api/v1"
        ) is False
        assert _provider_requires_stream("auto", None) is False
        assert _provider_requires_stream("auto", "") is False

    def test_config_marker_matches_custom_endpoint(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"auxiliary": {"stream_only_base_urls": ["my-proxy.example.com"]}},
        ):
            assert _provider_requires_stream(
                "custom", "https://my-proxy.example.com/v1"
            ) is True
            assert _provider_requires_stream(
                "custom", "https://other.example.com/v1"
            ) is False



class TestForceStream:
    def test_force_stream_streams_without_a_hook(self):
        chunks = [_chunk(content="hi", finish_reason="stop")]
        client = _FakeClient(stream_chunks=chunks)
        # NO aux_progress_hook installed — force_stream alone must stream.
        result = _create_with_progress(
            client, {"model": "m1", "messages": []}, force_stream=True,
        )
        assert client.calls[0]["stream"] is True
        assert result.choices[0].message.content == "hi"

    def test_force_stream_does_not_retry_nonstreaming_on_failure(self):
        client = _FakeClient(
            response=_COMPLETE,
            stream_error=RuntimeError("HTTP 400 bad request"),
        )
        with pytest.raises(RuntimeError, match="bad request"):
            _create_with_progress(
                client, {"model": "m1", "messages": []}, force_stream=True,
            )
        # No silent non-streaming retry — the provider rejects those anyway.
        assert len(client.calls) == 1


class TestAsyncStreamAggregation:
    @pytest.mark.asyncio
    async def test_async_stream_is_consumed_with_async_for(self):
        # The sweeper review of PR #60686 flagged that awaiting create() and
        # then iterating synchronously raises — the async contract is
        # ``async for``. Verify the async aggregator consumes a real async
        # iterator and preserves tool-call deltas.
        tc0 = SimpleNamespace(
            index=0, id="call_9",
            function=SimpleNamespace(name="lookup", arguments='{"q":'),
        )
        tc1 = SimpleNamespace(
            index=0, id=None,
            function=SimpleNamespace(name=None, arguments='"x"}'),
        )
        raw_chunks = [
            _chunk(content="part1 "),
            _chunk(tool_calls=[tc0]),
            _chunk(tool_calls=[tc1], content="part2", finish_reason="tool_calls"),
        ]

        class _AsyncStream:
            def __init__(self, items):
                self._items = list(items)
                self.closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

            async def close(self):
                self.closed = True

        stream = _AsyncStream(raw_chunks)
        result = await _aggregate_chat_stream_async(stream)
        msg = result.choices[0].message
        assert msg.content == "part1 part2"
        assert msg.tool_calls[0].function.name == "lookup"
        assert msg.tool_calls[0].function.arguments == '{"q":"x"}'
        assert result.choices[0].finish_reason == "tool_calls"
        assert stream.closed is True

    @pytest.mark.asyncio
    async def test_acreate_with_stream_passes_stream_kwargs(self):
        calls = []

        class _AsyncStream:
            def __init__(self, items):
                self._items = list(items)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        class _AsyncClient:
            def __init__(self):
                completions = SimpleNamespace(create=self._create)
                self.chat = SimpleNamespace(completions=completions)

            async def _create(self, **kwargs):
                calls.append(kwargs)
                return _AsyncStream([_chunk(content="ok", finish_reason="stop")])

        result = await _acreate_with_stream(
            _AsyncClient(), {"model": "m1", "messages": [], "timeout": 30},
        )
        assert calls[0]["stream"] is True
        assert result.choices[0].message.content == "ok"

    @pytest.mark.asyncio
    async def test_live_async_stream_resets_idle_without_resetting_total(self):
        started = time.monotonic()
        result = await _aggregate_chat_stream_async(
            _TimedAsyncStream(count=8, interval=0.03),
            idle_timeout=0.1,
            total_ceiling=0.6,
        )
        assert result.choices[0].message.content == "xxxxxxxx"
        assert time.monotonic() - started > 0.1

    @pytest.mark.asyncio
    async def test_silent_async_stream_expires_within_idle_window(self):
        import asyncio

        class _SilentAsyncStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.sleep(0.2)
                return _chunk(content="late")

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="idle"):
            await _aggregate_chat_stream_async(
                _SilentAsyncStream(), idle_timeout=0.05, total_ceiling=0.6
            )
        assert time.monotonic() - started < 0.15

    @pytest.mark.asyncio
    async def test_async_whitespace_and_empty_tool_deltas_do_not_count(self):
        empty_tool = SimpleNamespace(
            index=0,
            id="",
            function=SimpleNamespace(name="", arguments=""),
        )

        class _AsyncStream:
            def __init__(self):
                self.items = [
                    _chunk(content=" \n\t"),
                    _chunk(tool_calls=[empty_tool]),
                ]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.items:
                    raise StopAsyncIteration
                return self.items.pop(0)

        ticks = []
        with aux_progress_hook(lambda: ticks.append(True)):
            await _aggregate_chat_stream_async(_AsyncStream())
        assert ticks == []

    @pytest.mark.asyncio
    async def test_async_progress_cannot_extend_absolute_total(self):
        ticks = []
        with aux_progress_hook(lambda: ticks.append(True)):
            with pytest.raises(TimeoutError, match="total ceiling"):
                await _aggregate_chat_stream_async(
                    _TimedAsyncStream(count=20, interval=0.02),
                    idle_timeout=0.1,
                    total_ceiling=0.18,
                )
        assert len(ticks) >= 5
