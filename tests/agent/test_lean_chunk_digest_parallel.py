"""Lean tail_mode chunk digests: parallel dispatch, original segment order (#133)."""

import re
import threading
import time
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


def test_lean_chunk_digests_overlap_and_keep_segment_order():
    """Later chunks may finish first; concatenated output stays ### Segment i/n."""
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
            if active > max_active:
                max_active = active
        try:
            # The first chunk is the route probe; sibling chunks overlap after
            # the probe so throughput remains parallel without primary re-hits.
            time.sleep(0.05 if is_first else 0.01)
        finally:
            with lock:
                active -= 1
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = (
            "DIGEST-A" if is_first
            else "DIGEST-B" if "MARKER-B" in content
            else "DIGEST-C"
        )
        return resp

    turns = [
        {"role": "user", "content": "MARKER-A " + ("aaaa " * 20)},
        {"role": "user", "content": "MARKER-B " + ("bbbb " * 20)},
        {"role": "user", "content": "MARKER-C " + ("cccc " * 20)},
    ]
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")

    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 40),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch(
            "agent.auxiliary_client._get_task_max_concurrency",
            return_value=None,
        ),
    ):
        out = compressor._build_chunk_digests(turns)

    assert max_active >= 2, f"chunks stayed serial (max_active={max_active})"
    a = out.index("### Segment 1/")
    b = out.index("### Segment 2/")
    assert a < b
    assert out.find("DIGEST-A") < out.find("DIGEST-B")
    assert "DIGEST-A" in out[a:b]
    assert "DIGEST-B" in out[b:]


def test_lean_chunk_digests_keep_session_contextvars_on_workers():
    """Pool workers must see the caller's runtime + secret ContextVars (#133)."""
    from agent.auxiliary_client import (
        _RUNTIME_MAIN_CONTEXT,
        reset_runtime_main,
        set_runtime_main,
    )
    from agent.secret_scope import _SECRET_SCOPE, reset_secret_scope, set_secret_scope

    seen: list[dict] = []
    lock = threading.Lock()

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        with lock:
            seen.append({
                "is_worker": threading.current_thread() is not threading.main_thread(),
                "runtime": _RUNTIME_MAIN_CONTEXT.get(),
                "secret": _SECRET_SCOPE.get(),
            })
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "DIGEST"
        return resp

    turns = [
        {"role": "user", "content": "MARKER-A " + ("aaaa " * 20)},
        {"role": "user", "content": "MARKER-B " + ("bbbb " * 20)},
    ]
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    rt = set_runtime_main(
        provider="custom", model="probe-model",
        base_url="http://probe.invalid", api_key="PROBE-KEY",
    )
    st = set_secret_scope({"OPENAI_API_KEY": "PROFILE-SECRET"})
    try:
        with (
            patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 40),
            patch("agent.auxiliary_client.call_llm", fake_call_llm),
            patch(
                "agent.auxiliary_client._get_task_max_concurrency",
                return_value=None,
            ),
        ):
            compressor._build_chunk_digests(turns)
    finally:
        reset_secret_scope(st)
        reset_runtime_main(rt)

    workers = [hit for hit in seen if hit["is_worker"]]
    assert workers, f"dispatch stayed on the caller thread: {seen}"
    for hit in workers:
        assert hit["runtime"] is not None
        assert hit["runtime"].get("provider") == "custom"
        assert hit["secret"] is not None
        assert hit["secret"].get("OPENAI_API_KEY") == "PROFILE-SECRET"


# Serialized as "[user] MARKER-X " + 70 chars = 86; two-char joins → 262.
# chunk_chars=88 yields exactly three jobs, one marker each.
_THREE_CHUNK_CHARS = 88


def _three_marker_turns():
    return [
        {"role": "user", "content": "MARKER-A " + ("A" * 70)},
        {"role": "user", "content": "MARKER-B " + ("B" * 70)},
        {"role": "user", "content": "MARKER-C " + ("C" * 70)},
    ]


def _digest_body(messages):
    content = messages[0]["content"]
    if "MARKER-A" in content:
        return "DIGEST-A"
    if "MARKER-B" in content:
        return "DIGEST-B"
    if "MARKER-C" in content:
        return "DIGEST-C"
    return "DIGEST-PAD"


def _segment_headers(out):
    return re.findall(r"### Segment (\d+)/(\d+)", out)


def test_lean_chunk_digests_never_exceed_configured_max_concurrency():
    """auxiliary.compression.max_concurrency bounds in-flight call_llm (#133)."""
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        nonlocal active, max_active
        assert task == "compression"
        with lock:
            active += 1
            if active > max_active:
                max_active = active
        try:
            time.sleep(0.12)
        finally:
            with lock:
                active -= 1
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = _digest_body(messages)
        return resp

    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", _THREE_CHUNK_CHARS),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 2},
        ),
    ):
        out = compressor._build_chunk_digests(_three_marker_turns())

    assert max_active <= 2, f"in-flight call_llm exceeded bound (max_active={max_active})"
    assert max_active == 2, f"bound path stayed serial (max_active={max_active})"
    headers = _segment_headers(out)
    assert headers == [("1", "3"), ("2", "3"), ("3", "3")]
    assert out.find("DIGEST-A") < out.find("DIGEST-B") < out.find("DIGEST-C")


def test_lean_chunk_digests_isolate_one_chunk_failure():
    """One raising chunk keeps the existing placeholder; others stay real."""
    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        assert task == "compression"
        if "MARKER-B" in messages[0]["content"]:
            raise RuntimeError("boom-middle")
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = _digest_body(messages)
        return resp

    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", _THREE_CHUNK_CHARS),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 2},
        ),
    ):
        out = compressor._build_chunk_digests(_three_marker_turns())

    headers = _segment_headers(out)
    assert headers == [("1", "3"), ("2", "3"), ("3", "3")]
    a = out.index("### Segment 1/")
    b = out.index("### Segment 2/")
    c = out.index("### Segment 3/")
    assert "DIGEST-A" in out[a:b]
    assert "DIGEST-B" not in out
    assert "[digest unavailable for segment 2/3" in out[b:c]
    assert "recover via session_search" in out[b:c]
    assert "DIGEST-C" in out[c:]
    assert "boom-middle" not in out


def test_lean_chunk_digests_serial_when_max_concurrency_is_1():
    """max_concurrency=1 never overlaps call_llm and still emits Segment 1/N..N/N."""
    active = 0
    max_active = 0
    lock = threading.Lock()
    order: list[str] = []

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        nonlocal active, max_active
        assert task == "compression"
        body = _digest_body(messages)
        with lock:
            active += 1
            if active > max_active:
                max_active = active
            order.append(body)
        try:
            time.sleep(0.04)
        finally:
            with lock:
                active -= 1
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = body
        return resp

    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", _THREE_CHUNK_CHARS),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 1},
        ),
    ):
        out = compressor._build_chunk_digests(_three_marker_turns())

    assert max_active == 1, f"serial=1 overlapped (max_active={max_active})"
    assert order == ["DIGEST-A", "DIGEST-B", "DIGEST-C"]
    assert _segment_headers(out) == [("1", "3"), ("2", "3"), ("3", "3")]
    assert out.find("DIGEST-A") < out.find("DIGEST-B") < out.find("DIGEST-C")


def test_lean_chunk_digests_reuse_selected_fallback_route():
    """Sibling chunks use the route selected after the primary model fails."""
    calls: list[dict] = []

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        assert task == "compression"
        calls.append(kwargs)
        route = kwargs.get("provider"), kwargs.get("model")
        route_info = kwargs.get("route_info")
        if route_info is not None:
            route_info.update(provider="openai-codex", model="gpt-5.6-luna")
        if route == ("openai-codex", "gpt-5.6-luna"):
            label = "FALLBACK"
        else:
            # This stands in for call_llm's real primary-400 → fallback result.
            label = "FALLBACK" if not calls[:-1] else "PRIMARY-REHIT"
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = label
        return resp

    turns = [
        {"role": "user", "content": "MARKER-A " + ("aaaa " * 20)},
        {"role": "user", "content": "MARKER-B " + ("bbbb " * 20)},
        {"role": "user", "content": "MARKER-C " + ("cccc " * 20)},
    ]
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")

    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 40),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch(
            "agent.auxiliary_client._get_task_max_concurrency",
            return_value=5,
        ),
    ):
        out = compressor._build_chunk_digests(turns)

    assert "provider" not in calls[0]
    assert calls[0]["route_info"] == {
        "provider": "openai-codex",
        "model": "gpt-5.6-luna",
    }
    assert len(calls) > 1
    assert all(
        call.get("provider") == "openai-codex"
        and call.get("model") == "gpt-5.6-luna"
        for call in calls[1:]
    )
    assert "PRIMARY-REHIT" not in out
    assert out.count("FALLBACK") == len(calls)


def test_lean_harvest_cancel_keeps_slots_and_lean_recovery():
    """Queued Future.cancel() still yields 1/N..N/N; augment keeps users/recovery."""
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    turns = _three_marker_turns() + [
        {"role": "user", "content": "MARKER-D " + ("D" * 70)},
    ]
    from concurrent.futures import ThreadPoolExecutor

    two_started = threading.Event()
    submitted_three = threading.Event()
    hold = threading.Event()
    futures: list = []
    lock = threading.Lock()
    in_flight = 0
    orig_submit = ThreadPoolExecutor.submit

    def tracking_submit(self, fn, *args, **kwargs):
        fut = orig_submit(self, fn, *args, **kwargs)
        futures.append(fut)
        if len(futures) >= 3:
            submitted_three.set()
        return fut

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        nonlocal in_flight
        assert task == "compression"
        content = messages[0]["content"]
        if "MARKER-A" in content:
            body = "DIGEST-A"
        else:
            with lock:
                in_flight += 1
                if in_flight >= 2:
                    two_started.set()
            assert hold.wait(timeout=2), "queued cancel never released workers"
            with lock:
                in_flight -= 1
            body = _digest_body(messages)
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = body
        return resp

    def cancel_queued():
        assert two_started.wait(timeout=2), "first two workers never started"
        assert submitted_three.wait(timeout=2), "third future never submitted"
        assert futures[2].cancel(), "later job was already running"
        hold.set()

    compressor._session_id = "sess-137-harvest"
    canceler = threading.Thread(target=cancel_queued, daemon=True)
    canceler.start()
    try:
        with (
            patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", _THREE_CHUNK_CHARS),
            patch("agent.auxiliary_client.call_llm", fake_call_llm),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": 2},
            ),
            patch.object(ThreadPoolExecutor, "submit", tracking_submit),
        ):
            out = compressor._augment_summary_lean("KEEP-SUMMARY", turns)
    finally:
        hold.set()
        canceler.join(timeout=2)

    assert out.startswith("KEEP-SUMMARY")
    headers = _segment_headers(out)
    assert headers == [("1", "4"), ("2", "4"), ("3", "4"), ("4", "4")]
    assert "DIGEST-A" in out
    assert "DIGEST-B" in out
    assert "DIGEST-C" in out
    assert "[digest unavailable for segment 4/4" in out
    assert "recover via session_search" in out
    assert "## User Messages (verbatim, newest first)" in out
    assert "MARKER-D" in out
    assert "## Context Recovery" in out
    assert "session_id='sess-137-harvest'" in out


def test_lean_chunk_digests_keep_slots_when_worker_raises_baseexception():
    """Worker BaseException / AuxiliaryExplicitCancellation still emit every slot."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    class WorkerFatal(BaseException):
        pass

    for boom in (AuxiliaryExplicitCancellation(), WorkerFatal("worker-fatal")):
        def fake_call_llm(*, messages, task, max_tokens, _boom=boom, **kwargs):
            assert task == "compression"
            if "MARKER-B" in messages[0]["content"]:
                raise _boom
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = _digest_body(messages)
            return resp

        compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
        with (
            patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", _THREE_CHUNK_CHARS),
            patch("agent.auxiliary_client.call_llm", fake_call_llm),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": 2},
            ),
        ):
            out = compressor._build_chunk_digests(_three_marker_turns())

        assert _segment_headers(out) == [("1", "3"), ("2", "3"), ("3", "3")]
        assert "DIGEST-A" in out
        assert "DIGEST-C" in out
        assert "DIGEST-B" not in out
        assert "[digest unavailable for segment 2/3" in out
        assert "recover via session_search" in out
