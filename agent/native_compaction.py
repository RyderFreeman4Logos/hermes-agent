"""Helpers for native OpenAI Responses compaction checkpoints."""

from typing import Any, Dict, List


def merge_interim_reasoning_items(
    prior_items: Any,
    new_items: Any,
) -> List[Dict[str, Any]]:
    """Merge Codex continuation items while preserving prior checkpoints.

    Newer reasoning wins, but a native compaction checkpoint captured on the
    earlier incomplete response must survive when the continuation does not
    re-emit it.
    """
    kept_checkpoints = [
        item
        for item in (prior_items if isinstance(prior_items, list) else [])
        if isinstance(item, dict) and item.get("type") == "compaction"
    ]
    new_list = list(new_items) if isinstance(new_items, list) else []
    new_has_checkpoint = any(
        isinstance(item, dict) and item.get("type") == "compaction"
        for item in new_list
    )
    if new_has_checkpoint or not kept_checkpoints:
        return new_list
    return kept_checkpoints + new_list
