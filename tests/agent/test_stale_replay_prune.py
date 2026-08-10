"""Tests for stale codex_reasoning_items pruning during compaction (#71058).

Salvaged from PR #71077 (@webtecnica) with two correctness fixes:
the prune boundary is the last USER message (a Codex turn spans multiple
assistant messages whose reasoning items must replay together), and native
compaction checkpoints (type="compaction") are exempt because they carry
already-pruned history, not per-turn reasoning.
"""

from agent.context_compressor import (
    _STALE_REPLAY_PRUNE_KEYS,
    _prune_stale_reasoning_replay,
)


def _reasoning(item_id="rs_1"):
    return {"type": "reasoning", "encrypted_content": "blob-" + item_id, "id": item_id}


def _compaction():
    return {"type": "compaction", "encrypted_content": "checkpoint-blob"}


def test_prior_turn_reasoning_items_are_pruned():
    messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "a1", "codex_reasoning_items": [_reasoning("rs_a")]},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "a2", "codex_reasoning_items": [_reasoning("rs_b")]},
    ]
    pruned = _prune_stale_reasoning_replay(messages)
    assert pruned == 1
    assert "codex_reasoning_items" not in messages[1]
    # Active turn (after last user message) keeps its items.
    assert messages[3]["codex_reasoning_items"] == [_reasoning("rs_b")]


def test_multi_message_active_turn_chain_is_never_pruned():
    """A Codex turn spans assistant+tool_calls -> tool -> assistant; ALL of
    the active turn's reasoning items must survive (the #71077 review gap)."""
    messages = [
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old", "codex_reasoning_items": [_reasoning("rs_old")]},
        {"role": "user", "content": "active turn"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
            "codex_reasoning_items": [_reasoning("rs_chain1")],
        },
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
            "codex_reasoning_items": [_reasoning("rs_chain2")],
        },
        {"role": "tool", "content": "result", "tool_call_id": "c2"},
        {"role": "assistant", "content": "done", "codex_reasoning_items": [_reasoning("rs_final")]},
    ]
    pruned = _prune_stale_reasoning_replay(messages)
    assert pruned == 1  # only the old turn
    assert "codex_reasoning_items" not in messages[1]
    for idx in (3, 5, 7):
        assert messages[idx].get("codex_reasoning_items"), f"active-chain msg {idx} lost its items"


def test_compress_keeps_active_chain_before_incomplete_nudge():
    from agent.context_compressor import ContextCompressor
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE

    messages: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(11):
        messages.extend([
            {"role": "user", "content": f"old {i} " + "u" * 500},
            {
                "role": "assistant",
                "content": f"answer {i} " + "a" * 500,
                "codex_reasoning_items": [_reasoning(f"old{i}")],
            },
        ])
    messages.extend([
        {"role": "user", "content": "active task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "active_call",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }],
            "codex_reasoning_items": [_reasoning("active")],
        },
        {"role": "tool", "tool_call_id": "active_call", "content": "tool result"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "still thinking",
            "codex_reasoning_items": [_reasoning("incomplete")],
        },
        {"role": "user", "content": _CODEX_INCOMPLETE_NUDGE},
    ])
    compressor = ContextCompressor(
        model="gpt-test",
        provider="openai-codex",
        api_mode="codex_responses",
        quiet_mode=True,
        config_context_length=100_000,
        protect_first_n=3,
        protect_last_n=8,
    )
    compressor.tail_token_budget = 500
    compressor._generate_summary = lambda *args, **kwargs: "summary"

    compressed = compressor.compress(messages, current_tokens=90_000, force=True)
    active = next(
        msg
        for msg in compressed
        if any(
            call.get("id") == "active_call"
            for call in msg.get("tool_calls", [])
        )
    )

    assert active["codex_reasoning_items"] == [_reasoning("active")]
    assert any(msg.get("content") == _CODEX_INCOMPLETE_NUDGE for msg in compressed)


def test_native_compaction_checkpoints_survive_pruning():
    """type="compaction" items are cumulative context carriers — they must
    survive on stale messages even when reasoning items are stripped."""
    messages = [
        {"role": "user", "content": "turn 1"},
        {
            "role": "assistant",
            "content": "a1",
            "codex_reasoning_items": [_compaction(), _reasoning("rs_a")],
        },
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "a2"},
    ]
    pruned = _prune_stale_reasoning_replay(messages)
    assert pruned == 1
    # Reasoning stripped, checkpoint kept.
    assert messages[1]["codex_reasoning_items"] == [_compaction()]


def test_checkpoint_only_sidecar_untouched_and_uncounted():
    messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "a1", "codex_reasoning_items": [_compaction()]},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "a2"},
    ]
    pruned = _prune_stale_reasoning_replay(messages)
    assert pruned == 0
    assert messages[1]["codex_reasoning_items"] == [_compaction()]


def test_no_user_boundary_prunes_nothing():
    messages = [
        {"role": "assistant", "content": "a1", "codex_reasoning_items": [_reasoning("rs_a")]},
        {"role": "assistant", "content": "a2", "codex_reasoning_items": [_reasoning("rs_b")]},
    ]
    assert _prune_stale_reasoning_replay(messages) == 0
    assert messages[0]["codex_reasoning_items"]
    assert messages[1]["codex_reasoning_items"]


def test_non_codex_messages_untouched():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "plain"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "plain2"},
    ]
    assert _prune_stale_reasoning_replay(messages) == 0
    assert messages == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "plain"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "plain2"},
    ]


def test_prune_keys_contract():
    """codex_message_items are replayed for prefix-cache continuity and must
    NOT be in the prune set; the prune targets reasoning blobs only."""
    assert "codex_reasoning_items" in _STALE_REPLAY_PRUNE_KEYS
    assert "codex_message_items" not in _STALE_REPLAY_PRUNE_KEYS


class TestInterimMergePreservesCheckpoints:
    """Sibling site: the Codex incomplete-continuation dedup path must not
    drop checkpoints when overwriting a visually-duplicate interim message."""

    def test_prior_checkpoint_survives_overwrite(self):
        from agent.native_compaction import merge_interim_reasoning_items

        prior = [_compaction(), _reasoning("rs_old")]
        newer = [_reasoning("rs_new")]
        merged = merge_interim_reasoning_items(prior, newer)
        assert _compaction() in merged
        assert _reasoning("rs_new") in merged
        assert _reasoning("rs_old") not in merged  # newer reasoning wins

    def test_newer_checkpoint_wins_outright(self):
        from agent.native_compaction import merge_interim_reasoning_items

        prior = [{"type": "compaction", "encrypted_content": "old-ckpt"}]
        newer = [{"type": "compaction", "encrypted_content": "new-ckpt"}, _reasoning("rs_new")]
        merged = merge_interim_reasoning_items(prior, newer)
        assert merged == newer

    def test_no_prior_checkpoint_is_plain_overwrite(self):
        from agent.native_compaction import merge_interim_reasoning_items

        assert merge_interim_reasoning_items(
            [_reasoning("rs_old")], [_reasoning("rs_new")]
        ) == [_reasoning("rs_new")]

    def test_non_list_inputs_are_safe(self):
        from agent.native_compaction import merge_interim_reasoning_items

        assert merge_interim_reasoning_items(None, None) == []
        assert merge_interim_reasoning_items(None, [_reasoning("r")]) == [_reasoning("r")]
        assert merge_interim_reasoning_items([_compaction()], None) == [_compaction()]
