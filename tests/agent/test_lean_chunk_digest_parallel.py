"""Regression coverage for lean-digest parallel harvesting."""

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


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


def _response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _segment_headers(out):
    return re.findall(r"### Segment (\d+)/(\d+)", out)


def test_lean_harvest_cancel_keeps_slots_and_lean_recovery():
    """Queued Future.cancel() still yields slots; augment keeps recovery sections."""
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    turns = _turns() + [{"role": "user", "content": "MARKER-D " + ("D" * 70)}]

    two_started = threading.Event()
    submitted_three = threading.Event()
    hold = threading.Event()
    futures = []
    lock = threading.Lock()
    in_flight = 0
    orig_submit = ThreadPoolExecutor.submit

    def tracking_submit(self, fn, *args, **kwargs):
        future = orig_submit(self, fn, *args, **kwargs)
        futures.append(future)
        if len(futures) >= 3:
            submitted_three.set()
        return future

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
            body = _body(messages)
        return _response(body)

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
            patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", _CHUNK_CHARS),
            patch("agent.auxiliary_client.call_llm", fake_call_llm),
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"max_concurrency": 2},
            ),
            patch.object(ThreadPoolExecutor, "submit", tracking_submit),
        ):
            output = compressor._augment_summary_lean("KEEP-SUMMARY", turns)
    finally:
        hold.set()
        canceler.join(timeout=2)

    assert output.startswith("KEEP-SUMMARY")
    assert _segment_headers(output) == [("1", "4"), ("2", "4"), ("3", "4"), ("4", "4")]
    assert "DIGEST-A" in output
    assert "DIGEST-B" in output
    assert "DIGEST-C" in output
    assert "[digest unavailable for segment 4/4" in output
    assert "recover via session_search" in output
    assert "## User Messages (verbatim, newest first)" in output
    assert "MARKER-D" in output
    assert "## Context Recovery" in output
    assert "session_id='sess-137-harvest'" in output
