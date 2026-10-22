"""One displayed line after a reply: measured elapsed plus exact cache hit/total.

Display only. Not the status bar, and not written into the transcript.
``total`` is this turn's prompt tokens (uncached input + cache read + cache
write, summed across the turn's requests). Cached input is already inside
that total. Missing or incomplete usage omits the cache clause.
"""

from __future__ import annotations


def _pos_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def reply_end_line(
    *,
    elapsed_s: float | None,
    cache_read: object = None,
    prompt_total: object = None,
) -> str:
    """Return the reply-end line, or ``""`` when nothing was measured."""
    parts: list[str] = []
    if isinstance(elapsed_s, (int, float)) and not isinstance(elapsed_s, bool) and elapsed_s >= 0:
        parts.append(f"{float(elapsed_s):.1f}s")
    read = _pos_int(cache_read)
    total = _pos_int(prompt_total)
    if read is not None and total is not None and total > 0:
        parts.append(f"cache {read}/{total} ({100.0 * read / total:.1f}%)")
    return " · ".join(parts)
