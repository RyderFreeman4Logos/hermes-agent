"""Public boundary regressions for lean digest safety and lifecycle."""

from types import SimpleNamespace
import threading
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor, _SUMMARY_ROUTE_RECEIPT


def _response(text: str = "digest"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")]
    )


def _turns(count: int = 4):
    return [
        {"role": "user", "content": f"MARKER-{index} " + chr(65 + index) * 80}
        for index in range(count)
    ]


def test_digest_egress_strictly_redacts_selected_pristine_text():
    sent: list[str] = []
    secret = "cross-boundary-secret"
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    compressor._lean_pristine_tools = {
        "tool-1": "benign-tool-id " + ("x" * 30) + f" postgres://user:{secret}@db.invalid/data " + ("y" * 90)
    }
    turns = [
        {"role": "user", "content": "benign-user-id " + ("a" * 70) + f" https://user:{secret}@api.invalid/v1"},
        {"role": "tool", "tool_call_id": "tool-1", "content": "[trimmed]"},
    ]

    def fake_call_llm(*, messages, **_kwargs):
        sent.append(messages[0]["content"])
        return _response()

    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 73),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch("agent.auxiliary_client._get_task_max_concurrency", return_value=1),
    ):
        compressor._build_chunk_digests(turns)

    assert sent
    assert all(secret not in request for request in sent)
    digest_input = "".join(
        request.rsplit("TRANSCRIPT SEGMENT:\n", 1)[1].removesuffix("\n")
        for request in sent
    )
    assert "benign-user-id" in digest_input
    assert "benign-tool-id" in digest_input


def test_digest_siblings_reuse_complete_selected_destination():
    calls: list[dict] = []
    selected = {
        "provider": "custom",
        "model": "fallback-model",
        "base_url": "https://fallback.invalid/v1",
        "api_key": "synthetic-fallback-key",
        "api_mode": "anthropic_messages",
        "timeout": 37.0,
    }

    def fake_call_llm(*, route_info=None, **kwargs):
        calls.append(kwargs)
        if route_info is not None:
            route_info.update(selected)
        return _response()

    compressor = ContextCompressor("main-model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 88),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch("agent.auxiliary_client._get_task_max_concurrency", return_value=2),
    ):
        compressor._build_chunk_digests(_turns(3))

    assert len(calls) >= 2
    assert all({key: call.get(key) for key in selected} == selected for call in calls[1:])


def test_first_digest_reuses_complete_successful_summary_destination():
    selected = {
        "provider": "custom",
        "model": "summary-fallback",
        "base_url": "https://summary-fallback.invalid/v1",
        "api_key": "synthetic-summary-key",
        "api_mode": "responses",
        "timeout": 29.0,
    }
    digest_calls: list[dict] = []
    compressor = ContextCompressor("main-model", quiet_mode=True, tail_mode="lean")

    def fake_call_llm(**kwargs):
        digest_calls.append(kwargs)
        return _response()

    token = _SUMMARY_ROUTE_RECEIPT.set(dict(selected))
    try:
        with (
            patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 10_000),
            patch("agent.auxiliary_client.call_llm", fake_call_llm),
        ):
            compressor._build_chunk_digests(_turns(1))
    finally:
        _SUMMARY_ROUTE_RECEIPT.reset(token)

    assert len(digest_calls) == 1
    assert {key: digest_calls[0].get(key) for key in selected} == selected


def test_digest_pool_workers_inherit_auxiliary_lifecycle_scope():
    import agent.auxiliary_client as aux

    cancel = threading.Event()
    seen: list[tuple] = []
    progress: list[str] = []

    def fake_call_llm(**_kwargs):
        if threading.current_thread() is not threading.main_thread():
            aux._notify_aux_progress()
            check = aux._capture_aux_cancel_check()
            seen.append((
                aux._aux_interrupt_protected(),
                callable(check) and not aux._captured_aux_cancel_requested(check),
                aux._current_aux_stream_deadline(),
                aux._aux_progress_active(),
            ))
        return _response()

    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        aux.aux_progress_hook(lambda: progress.append("tick")),
        aux.aux_stream_deadline(9876.5),
        aux.aux_interrupt_protection(cancel_event=cancel),
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 88),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch("agent.auxiliary_client._get_task_max_concurrency", return_value=2),
    ):
        compressor._build_chunk_digests(_turns(3))

    assert seen
    assert all(item == (True, True, 9876.5, True) for item in seen)
    assert progress


def test_explicit_cancellation_escapes_and_stops_later_digest_dispatch():
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    calls: list[str] = []

    def fake_call_llm(*, messages, **_kwargs):
        marker = next(marker for marker in ("MARKER-0", "MARKER-1", "MARKER-2") if marker in messages[0]["content"])
        calls.append(marker)
        if marker == "MARKER-1":
            raise AuxiliaryExplicitCancellation()
        return _response(marker)

    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 88),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch("agent.auxiliary_client._get_task_max_concurrency", return_value=1),
    ):
        with pytest.raises(AuxiliaryExplicitCancellation):
            compressor._build_chunk_digests(_turns(3))

    assert calls[:2] == ["MARKER-0", "MARKER-1"]
    assert "MARKER-2" not in calls


def test_unrelated_worker_baseexception_remains_a_local_placeholder():
    class WorkerFailure(BaseException):
        pass

    def fake_call_llm(*, messages, **_kwargs):
        if "MARKER-1" in messages[0]["content"]:
            raise WorkerFailure("local worker failure")
        return _response()

    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 88),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch("agent.auxiliary_client._get_task_max_concurrency", return_value=2),
    ):
        result = compressor._build_chunk_digests(_turns(3))

    assert "digest unavailable for segment 2/4" in result
    assert "### Segment 4/4" in result


def test_static_fallback_is_local_but_successful_augmentation_harvests():
    calls: list[dict] = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response()

    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    compressor._session_id = "synthetic-session"
    turns = _turns(3)
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 88),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch("agent.auxiliary_client._get_task_max_concurrency", return_value=1),
    ):
        fallback = compressor._build_static_fallback_summary(turns, "synthetic failure")
        assert calls == []
        successful = compressor._augment_summary_lean("successful summary", turns)

    assert "MARKER-0" in fallback
    assert "## Context Recovery" in fallback
    assert calls
    assert "## Detailed Session Log" in successful


def test_public_feasibility_skip_never_dispatches_digest_llm():
    compressor = ContextCompressor(
        "test/model", quiet_mode=True, tail_mode="lean",
        protect_first_n=2, protect_last_n=2, config_context_length=100_000,
    )
    compressor._ineffective_compression_count = 1
    messages = [{"role": "system", "content": "system"}]
    for index in range(10):
        messages.extend([
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": "short reply"},
        ])

    with patch("agent.auxiliary_client.call_llm") as digest_call:
        result = compressor.compress(messages, force=False)

    assert compressor._last_feasibility_skip is True
    digest_call.assert_not_called()
    assert len(result) < len(messages)
