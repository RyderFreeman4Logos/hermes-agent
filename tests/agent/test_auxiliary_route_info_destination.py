"""Real resolver witness for complete auxiliary fallback destinations."""

from types import SimpleNamespace
import threading
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import call_llm
from agent.context_compressor import ContextCompressor


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
    assert compressor._last_summary_route is None


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
    compressor._last_summary_route = dict(summary_route)
    turns = [
        {"role": "user", "content": f"MARKER-{index} " + chr(65 + index) * 70}
        for index in range(3)
    ]
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
