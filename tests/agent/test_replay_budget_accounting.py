"""Tests for provider-aware replay-field accounting in tail-budget walks (#73624).

Generic thinking fields (``reasoning`` / ``reasoning_content`` + the
``reasoning_details`` text charge) are replayed for at most the NEWEST
assistant turn unless a require-side provider echoes historical reasoning.
Anthropic strips all-but-newest for ordinary endpoints, Bedrock never replays
thinking, and strict chat-completions providers reject the field. Charging
stripped fields on every message spent 19-24% of the tail budget on bytes that
never reach the wire.

Codex sidecar fields (``codex_reasoning_items`` / ``codex_message_items``)
ARE wire-replayed on every retained turn and stay charged unconditionally
(#55572) — including native compaction checkpoints (#81747).
"""

import pytest

from agent.context_compressor import (
    _ALWAYS_REPLAYED_BUDGET_KEYS,
    _NEWEST_TURN_ONLY_BUDGET_KEYS,
    _REPLAY_BUDGET_KEYS,
    _estimate_msg_budget_tokens,
    _last_assistant_index,
)


BIG_THINKING = "deliberation " * 400  # ~1.3K tokens of stale thinking text
BIG_BLOB = [{"type": "reasoning", "encrypted_content": "x" * 4000}]


def _assistant(thinking=False, codex=False):
    msg = {"role": "assistant", "content": "done"}
    if thinking:
        msg["reasoning"] = BIG_THINKING
        msg["reasoning_content"] = BIG_THINKING
    if codex:
        msg["codex_reasoning_items"] = BIG_BLOB
    return msg


class TestChargeStaleThinking:
    @pytest.mark.parametrize(
        ("api_mode", "provider", "model", "base_url"),
        [
            ("chat_completions", "kimi-coding", "k3", "https://api.kimi.com/coding/v1"),
            ("chat_completions", "mistral", "mistral-small", "https://api.mistral.ai/v1"),
            ("anthropic_messages", "anthropic", "claude-opus-5", "https://api.anthropic.com"),
            ("bedrock_converse", "bedrock", "claude-opus-5", ""),
            ("codex_responses", "openai-codex", "gpt-5.6", "https://chatgpt.com/backend-api/codex"),
        ],
    )
    def test_reasoning_alias_pair_is_charged_once_across_transport_policies(
        self, api_mode, provider, model, base_url
    ):
        from agent.context_compressor import ContextCompressor

        compressor = ContextCompressor(
            model=model,
            provider=provider,
            base_url=base_url,
            api_mode=api_mode,
            quiet_mode=True,
            config_context_length=200_000,
        )
        long = "r" * 8_000
        single = {"role": "assistant", "content": "done", "reasoning_content": long}
        duplicate = dict(single, reasoning="short" * 400)

        # Exercise both the transport's stale-turn policy and the universally
        # charged newest-assistant policy.
        for charge in {compressor._replays_stale_thinking(), True}:
            assert _estimate_msg_budget_tokens(
                duplicate, charge_stale_thinking=charge
            ) == _estimate_msg_budget_tokens(single, charge_stale_thinking=charge)

    def test_stale_turn_thinking_not_charged(self):
        msg = _assistant(thinking=True)
        full = _estimate_msg_budget_tokens(msg, charge_stale_thinking=True)
        stale = _estimate_msg_budget_tokens(msg, charge_stale_thinking=False)
        assert stale < full
        # The delta is the thinking text — a substantial chunk, not noise.
        assert full - stale > 300

    def test_codex_sidecar_always_charged(self):
        """Wire-replayed Codex blobs (incl. native compaction checkpoints)
        must stay in the budget even for stale turns — #55572's invariant."""
        msg = _assistant(codex=True)
        full = _estimate_msg_budget_tokens(msg, charge_stale_thinking=True)
        stale = _estimate_msg_budget_tokens(msg, charge_stale_thinking=False)
        assert stale == full  # nothing thinking-only to drop
        bare = _estimate_msg_budget_tokens(
            {"role": "assistant", "content": "done"}, charge_stale_thinking=False
        )
        assert stale > bare + 500  # blob still charged on the stale path

    def test_default_is_conservative_full_charge(self):
        msg = _assistant(thinking=True)
        assert _estimate_msg_budget_tokens(msg) == _estimate_msg_budget_tokens(
            msg, charge_stale_thinking=True
        )

    def test_reasoning_details_text_skipped_on_stale_path(self):
        msg = {
            "role": "assistant",
            "content": "done",
            "reasoning_details": [
                {"type": "reasoning.text", "text": "long plan " * 300}
            ],
        }
        full = _estimate_msg_budget_tokens(msg, charge_stale_thinking=True)
        stale = _estimate_msg_budget_tokens(msg, charge_stale_thinking=False)
        assert stale < full


class TestKeyPartition:
    def test_partition_covers_replay_budget_keys_exactly(self):
        """Invariant: the two accounting classes partition _REPLAY_BUDGET_KEYS.
        A future key added to the replay budget must be classified."""
        assert set(_ALWAYS_REPLAYED_BUDGET_KEYS) | set(
            _NEWEST_TURN_ONLY_BUDGET_KEYS
        ) == set(_REPLAY_BUDGET_KEYS)
        assert not set(_ALWAYS_REPLAYED_BUDGET_KEYS) & set(
            _NEWEST_TURN_ONLY_BUDGET_KEYS
        )

    def test_codex_fields_are_always_replayed_class(self):
        assert "codex_reasoning_items" in _ALWAYS_REPLAYED_BUDGET_KEYS
        assert "codex_message_items" in _ALWAYS_REPLAYED_BUDGET_KEYS


class TestLastAssistantIndex:
    def test_finds_newest_assistant(self):
        msgs = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "tool", "content": "t"},
        ]
        assert _last_assistant_index(msgs) == 3

    def test_no_assistant_returns_minus_one(self):
        assert _last_assistant_index([{"role": "user", "content": "u"}]) == -1
        assert _last_assistant_index([]) == -1


class TestTailCutBehavior:
    """Newest-only transports must protect more real transcript when stale
    turns carry heavy thinking — #73624's symptom was the cut landing early."""

    def _compressor(self):
        from agent.context_compressor import ContextCompressor

        cc = ContextCompressor(
            model="claude-opus-5",
            quiet_mode=True,
            config_context_length=200_000,
        )
        return cc

    def test_stale_thinking_does_not_shrink_tail(self):
        cc = self._compressor()
        # Build a transcript where every assistant turn drags huge stale
        # thinking. Under the old accounting these bloat the walk and the
        # cut lands early; with newest-turn-only accounting the same budget
        # protects more messages.
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(30):
            msgs.append({"role": "user", "content": f"question {i}"})
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"answer {i}",
                    "reasoning": BIG_THINKING,
                    "reasoning_content": BIG_THINKING,
                }
            )
        budget = 3_000
        cut = cc._find_tail_cut_by_tokens(msgs, 1, token_budget=budget)

        # Compute what the OLD accounting (charge everything) would protect.
        old_accumulated = 0
        old_cut = len(msgs)
        soft = int(budget * 1.5)
        for i in range(len(msgs) - 1, 0, -1):
            t = _estimate_msg_budget_tokens(msgs[i], charge_stale_thinking=True)
            if old_accumulated + t > soft and (len(msgs) - i) >= 3:
                break
            old_accumulated += t
            old_cut = i

        # New accounting must protect at least as much transcript (lower cut
        # index = more messages in the tail), and strictly more here because
        # the stale thinking dominates each message's old-cost.
        assert cut < old_cut

    def test_require_side_reasoning_echo_charges_stale_turns(self):
        from agent.context_compressor import ContextCompressor
        from agent.message_sanitization import (
            apply_reasoning_content_policy,
            needs_reasoning_echo,
        )
        from agent.transports.chat_completions import ChatCompletionsTransport

        messages: list[dict] = [{"role": "system", "content": "sys"}]
        for i in range(30):
            messages.extend([
                {"role": "user", "content": f"question {i}"},
                {
                    "role": "assistant",
                    "content": f"answer {i}",
                    "reasoning_content": "r" * 6_000,
                },
            ])

        provider = "deepseek"
        model = "deepseek-v4-pro"
        base_url = "https://api.deepseek.com/v1"
        require_echo = needs_reasoning_echo(provider, model, base_url)
        api_message = {
            "role": "assistant",
            "content": messages[2]["content"],
        }
        apply_reasoning_content_policy(messages[2], api_message, require_echo)
        wire = ChatCompletionsTransport().convert_messages(
            [api_message], model=model
        )
        assert len(wire[0]["reasoning_content"]) == 6_000

        strict = ContextCompressor(
            model="mistral-small",
            provider="mistral",
            api_mode="chat_completions",
            quiet_mode=True,
            config_context_length=200_000,
        )
        require = ContextCompressor(
            model=model,
            provider=provider,
            base_url=base_url,
            api_mode="chat_completions",
            quiet_mode=True,
            config_context_length=200_000,
        )
        budget = 1_500

        strict_cut = strict._find_tail_cut_by_tokens(
            messages, 1, token_budget=budget
        )
        require_cut = require._find_tail_cut_by_tokens(
            messages, 1, token_budget=budget
        )
        assert require_cut > strict_cut


class TestProtectedPressureConsistency:
    def test_stale_thinking_alone_does_not_demote_tool_results(self):
        from agent.context_compressor import ContextCompressor

        def history(stale_thinking: bool) -> list[dict]:
            messages: list[dict] = [{"role": "user", "content": "start"}]
            for i in range(8):
                assistant = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": f'{{"path":"f{i}"}}',
                        },
                    }],
                }
                if stale_thinking:
                    assistant["reasoning_content"] = "r" * 6_000
                messages.extend([
                    assistant,
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{i}",
                        "content": f"UNIQUE-{i}-" + chr(65 + i) * 1_000,
                    },
                ])
            messages.extend([
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "continue"},
            ])
            return messages

        compressor = ContextCompressor(
            model="mistral-small",
            provider="mistral",
            api_mode="chat_completions",
            quiet_mode=True,
            config_context_length=128_000,
        )
        results = []
        for stale_thinking in (False, True):
            pruned, count = compressor._prune_old_tool_results(
                history(stale_thinking),
                protect_tail_count=20,
                protect_tail_tokens=3_000,
            )
            full_results = sum(
                len(message.get("content", "")) > 400
                for message in pruned
                if message.get("role") == "tool"
            )
            results.append((count, full_results))

        assert results == [(0, 8), (0, 8)]
