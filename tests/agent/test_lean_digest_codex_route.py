"""Unique #160 residual over admitted #165: Codex route identity + else-arm."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def test_lean_digest_siblings_use_attempt_summary_route_kwargs_without_selected_route():
    """When no selected route, sibling workers still take pin kwargs (else-arm)."""
    calls = []

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        assert task == "compression"
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="DIGEST"))]
        )

    turns = [
        {"role": "user", "content": "MARKER-A " + ("a" * 70)},
        {"role": "user", "content": "MARKER-B " + ("b" * 70)},
        {"role": "user", "content": "MARKER-C " + ("c" * 70)},
    ]
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 88),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch("agent.auxiliary_client._get_task_max_concurrency", return_value=3),
        patch(
            "agent.context_compressor.attempt_summary_route_kwargs",
            return_value={"provider": "stall-fallback", "model": "healthy-aux"},
        ),
    ):
        compressor._build_chunk_digests(turns)

    assert len(calls) == 3
    assert calls[0].get("provider") != "stall-fallback"
    assert "route_info" in calls[0]
    for call in calls[1:]:
        assert call.get("provider") == "stall-fallback"
        assert call.get("model") == "healthy-aux"


def test_lean_digest_workers_reuse_selected_codex_route_settings():
    """Parallel lean workers carry the selected entry-owned route controls."""
    calls = []

    def fake_call_llm(*, messages, task, max_tokens, **kwargs):
        assert task == "compression"
        calls.append(kwargs)
        route_info = kwargs.get("route_info")
        if route_info is not None and not route_info:
            route_info.update(
                provider="openai-codex",
                model="codex-model",
                fallback_label="fallback_chain[1](openai-codex)",
            )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="DIGEST"))]
        )
        return response

    turns = [
        {"role": "user", "content": "MARKER-A " + ("a" * 70)},
        {"role": "user", "content": "MARKER-B " + ("b" * 70)},
        {"role": "user", "content": "MARKER-C " + ("c" * 70)},
    ]
    compressor = ContextCompressor("test/model", quiet_mode=True, tail_mode="lean")
    with (
        patch("agent.context_compressor._LEAN_DIGEST_CHUNK_CHARS", 88),
        patch("agent.auxiliary_client.call_llm", fake_call_llm),
        patch("agent.auxiliary_client._get_task_max_concurrency", return_value=3),
    ):
        compressor._build_chunk_digests(turns)

    assert len(calls) == 3
    assert calls[0]["route_info"]["fallback_label"] == "fallback_chain[1](openai-codex)"
    for call in calls[1:]:
        assert call["provider"] == "openai-codex"
        assert call["model"] == "codex-model"
        assert call["route_info"] == {
            "fallback_label": "fallback_chain[1](openai-codex)"
        }
