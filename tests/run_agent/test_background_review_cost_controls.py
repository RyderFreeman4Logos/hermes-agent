"""Unit coverage for the background-review aux-model selector + routed digest.

Covers the two behaviors this change adds:
  • _resolve_review_runtime — auto/same-model → not routed (main model, warm
    cache); a configured different model → routed with resolved credentials.
  • _digest_history — compact replay used ONLY on the routed path (recent tail
    verbatim + a digest of older turns), preserving role alternation.

Pure-function / config-driven; no live model calls.
"""
import copy
import json
import threading
from typing import Any
from unittest.mock import patch

import pytest

from agent import background_review as br


def _msg(role, content, tool_calls=None):
    m = {"role": role, "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


# ---------------------------------------------------------------------------
# _resolve_review_runtime — the aux-model selector
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, provider="openai-codex", model="gpt-5.5"):
        self.provider = provider
        self.model = model
        self._credential_pool: Any = None
        self.request_overrides = {}
        self.max_tokens: int | None = None

    def _current_main_runtime(self):
        return {
            "api_key": "parent-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_app_server",
        }


def test_routing_auto_inherits_parent_and_downgrades_codex_app_server():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {"provider": "auto", "model": ""}}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"
    assert rt["model"] == "gpt-5.5"
    assert rt["api_mode"] == "codex_responses"  # downgraded so agent-loop tools dispatch


def test_routing_to_different_model_marks_routed_and_resolves_credentials():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "google/gemini-3-flash-preview",
    }}}
    fake_rp = {
        "provider": "openrouter", "api_key": "or-key",
        "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions",
        "credential_pool": "routed-pool",
        "request_overrides": {"extra_body": {"store": False}},
        "max_output_tokens": 2048,
    }
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fake_rp):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is True
    assert rt["provider"] == "openrouter"
    assert rt["model"] == "google/gemini-3-flash-preview"
    assert rt["api_key"] == "or-key"
    assert rt["credential_pool"] == "routed-pool"
    assert rt["request_overrides"] == {"extra_body": {"store": False}}
    assert rt["max_tokens"] == 2048


def test_unrouted_runtime_keeps_parent_pool_and_overrides():
    agent = _FakeAgent()
    agent._credential_pool = "parent-pool"
    agent.request_overrides = {"service_tier": "priority"}
    agent.max_tokens = 4096
    with patch("hermes_cli.config.load_config", return_value={}), patch("hermes_cli.config.load_config_readonly", return_value={}):
        rt = br._resolve_review_runtime(agent)
    assert rt["credential_pool"] == "parent-pool"
    assert rt["request_overrides"] == {"service_tier": "priority"}
    assert rt["max_tokens"] == 4096


def test_routing_same_model_as_parent_is_not_routed():
    agent = _FakeAgent(provider="openrouter", model="anthropic/claude-opus-4.8")
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "anthropic/claude-opus-4.8",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False  # same model/provider → keep full-replay path


def test_routing_resolution_failure_falls_back_to_parent():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "google/gemini-3-flash-preview",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               side_effect=RuntimeError("boom")):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"


# ---------------------------------------------------------------------------
# _digest_history — routed-path compact replay
# ---------------------------------------------------------------------------

def test_digest_under_tail_returns_full():
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    assert br._digest_history(msgs, tail=24) == msgs


def test_digest_collapses_old_keeps_tail_verbatim():
    msgs = []
    for i in range(60):
        msgs.append(_msg("user", f"u{i} " + "x" * 50))
        msgs.append(_msg("assistant", f"a{i} " + "y" * 50))
    out = br._digest_history(msgs, tail=10)
    # First message is the synthetic digest (user role → alternation preserved).
    assert out[0]["role"] == "user"
    assert out[0]["content"].startswith("[Earlier conversation digest")
    # Recent tail preserved verbatim.
    assert out[-1] == msgs[-1]
    assert len(out) == 11  # 1 digest + 10 tail


def test_digest_does_not_open_tail_on_a_tool_message():
    msgs = []
    for i in range(40):
        msgs.append(_msg("user", "u" + "x" * 50))
        msgs.append(_msg("assistant", "", tool_calls=[
            {"function": {"name": "terminal", "arguments": "{}"}}]))
        msgs.append({"role": "tool", "content": "result " + "w" * 50})
    out = br._digest_history(msgs, tail=2)
    # The verbatim tail (after the digest) must not begin on a bare tool message.
    assert out[1]["role"] != "tool"


def test_digest_records_tool_names_in_arc():
    old = [
        _msg("user", "do the thing"),
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "skill_view", "arguments": "{}"}},
            {"function": {"name": "patch", "arguments": "{}"}}]),
    ]
    msgs = old + [_msg("user", f"tail{i}") for i in range(30)]
    out = br._digest_history(msgs, tail=10)
    digest = out[0]["content"]
    assert "USER: do the thing" in digest
    assert "tools: skill_view, patch" in digest


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("duplicate_role", ["user", "assistant"])
def test_review_snapshot_is_owned_during_concurrent_alternation_repair(
    monkeypatch, tmp_path, routed, duplicate_role
):
    """Reviewer repair must not write through to a provider-bound foreground."""
    from agent import physical_attempt_diagnostics as diagnostics
    from agent.agent_runtime_helpers import repair_message_sequence
    from agent.turn_context import substitute_api_content
    from hermes_cli import config

    history = []
    for i in range(12):
        history.extend(
            [
                {"role": "user", "content": f"u{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        )

    if duplicate_role == "user":
        history.extend(
            [
                {
                    "role": "user",
                    "content": "first user",
                    "api_content": "first user with private context",
                    "display_metadata": {"nested": ["foreground"]},
                },
                {"role": "user", "content": "second user"},
                {"role": "assistant", "content": "closing assistant"},
            ]
        )
        survivor = -3
    else:
        history.extend(
            [
                {"role": "user", "content": "assistant pair follows"},
                {
                    "role": "assistant",
                    "content": "first assistant",
                    "api_content": "first assistant wire bytes",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "first", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "second assistant",
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "function": {"name": "second", "arguments": "{}"},
                        }
                    ],
                },
            ]
        )
        survivor = -2

    original = copy.deepcopy(history)

    def _wire_projection():
        projected = copy.deepcopy(history)
        for message in projected:
            substitute_api_content(message)
            message.pop("display_metadata", None)
        return projected

    before_wire = json.dumps(
        _wire_projection(), sort_keys=True, separators=(",", ":")
    ).encode()
    monkeypatch.setattr(diagnostics, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        config,
        "read_raw_config_readonly",
        lambda: {"observability": {"physical_attempt_digests": {"enabled": True}}},
    )

    def _record_prefix(retry):
        return diagnostics.start_attempt(
            {"model": "test", "messages": _wire_projection(), "tools": []},
            api_mode="chat_completions",
            route="chat_completions",
            provider="openai",
            model="test",
            role="main",
            retry=retry,
            continuation=0,
        )

    assert _record_prefix(0) is not None

    repair_done = threading.Event()
    release_review = threading.Event()
    seen = {}
    errors = []

    def _repairing_review(_agent, snapshot, prompt):
        try:
            seen["snapshot"] = snapshot
            review_history = br._digest_history(snapshot) if routed else snapshot
            request_messages = [
                *review_history,
                {"role": "user", "content": prompt},
            ]
            seen["repairs"] = repair_message_sequence(None, request_messages)
            seen["request_messages"] = request_messages
        except BaseException as exc:  # propagate worker failures to the test thread
            errors.append(exc)
        finally:
            repair_done.set()
        release_review.wait(5)

    monkeypatch.setattr(br, "_run_review_in_thread", _repairing_review)
    target, _prompt = br.spawn_background_review_thread(
        object(), history, review_memory=True
    )
    worker = threading.Thread(target=target)
    worker.start()
    try:
        assert repair_done.wait(5), "review repair did not reach the barrier"
        if errors:
            raise errors[0]
        after_wire = json.dumps(
            _wire_projection(), sort_keys=True, separators=(",", ":")
        ).encode()
        assert _record_prefix(1) is not None
    finally:
        release_review.set()
        worker.join(5)

    assert not worker.is_alive()
    assert history == original
    assert after_wire == before_wire

    starts = [
        json.loads(line)
        for line in (
            tmp_path / "observability" / "physical_attempt_digests.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(starts) == 2
    assert starts[0]["prefix_digest"] == starts[1]["prefix_digest"]

    roles = [message["role"] for message in seen["request_messages"]]
    assert seen["repairs"] > 0
    assert all(left != right for left, right in zip(roles, roles[1:]))

    snapshot = seen["snapshot"]
    assert snapshot is not history
    assert snapshot[survivor] is not history[survivor]
    if duplicate_role == "user":
        assert (
            snapshot[survivor]["display_metadata"]
            is not history[survivor]["display_metadata"]
        )
    else:
        assert (
            snapshot[survivor]["tool_calls"][0]["function"]
            is not history[survivor]["tool_calls"][0]["function"]
        )
