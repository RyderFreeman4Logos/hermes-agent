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
    started = threading.Event()
    release = threading.Event()

    def fake_call_llm(*, messages, task, max_tokens):
        nonlocal active, max_active
        assert task == "compression"
        content = messages[0]["content"]
        is_first = "MARKER-A" in content
        with lock:
            active += 1
            if active > max_active:
                max_active = active
            if active >= 2:
                started.set()
        try:
            if is_first:
                # First segment stays in-flight until the later one has started
                # (and then a beat longer) so finish order is 2 then 1.
                assert started.wait(timeout=2), "second chunk never overlapped"
                time.sleep(0.05)
            else:
                release.set()
                time.sleep(0.01)
        finally:
            with lock:
                active -= 1
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "DIGEST-A" if is_first else "DIGEST-B"
        return resp

    turns = [
        {"role": "user", "content": "MARKER-A " + ("aaaa " * 20)},
        {"role": "user", "content": "MARKER-B " + ("bbbb " * 20)},
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

    def fake_call_llm(*, messages, task, max_tokens):
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

    def fake_call_llm(*, messages, task, max_tokens):
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
    def fake_call_llm(*, messages, task, max_tokens):
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

    def fake_call_llm(*, messages, task, max_tokens):
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
