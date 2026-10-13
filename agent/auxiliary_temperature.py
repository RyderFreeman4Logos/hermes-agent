"""OpenAI reasoning families reject any non-default temperature.

Kept out of ``auxiliary_client`` so that facade can only shrink.
"""
from __future__ import annotations

from typing import Optional


def is_openai_default_temperature_only(model: Optional[str]) -> bool:
    """True for OpenAI reasoning families that 400 (``unsupported_value``) on any non-default
    ``temperature``: gpt-5.x (incl. dated snapshots, ``-pro``, ``-codex``), o1/o3/o4. The
    ``gpt-5-chat`` non-reasoning line still accepts it (#51083)."""
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bare.startswith(("gpt-5", "o1", "o3", "o4")) and not bare.startswith("gpt-5-chat")
