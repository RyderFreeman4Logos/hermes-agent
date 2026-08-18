"""Lean tail_mode chunk digests: parallel dispatch, original segment order (#133)."""

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
