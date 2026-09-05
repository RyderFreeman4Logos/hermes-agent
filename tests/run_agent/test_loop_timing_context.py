"""Regression coverage for API-only loop timing context."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

from agent import conversation_loop


UTC_MINUS_7 = timezone(timedelta(hours=-7))


def test_loop_timing_is_default_on_but_config_gated():
    timing_context: Callable[..., str | None] | None = getattr(
        conversation_loop, "_loop_timing_context", None
    )
    assert callable(timing_context), "loop timing context must be available to the turn forwarder"

    agent = SimpleNamespace()
    start = datetime(2026, 8, 22, 11, 28, 0, tzinfo=UTC_MINUS_7)
    stop = start + timedelta(seconds=3)
    next_start = stop + timedelta(seconds=2)

    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        context = timing_context(agent, now=start)
        assert context is not None
        assert "Current loop start: 2026-08-22T11:28:00-07:00" in context
        assert timing_context(agent, now=stop, stop=True) is None

    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"agent": {"loop_timing_context": False}},
    ):
        assert timing_context(agent, now=next_start) == ""

    assert agent._loop_timing_last_start == next_start
    assert agent._loop_timing_last_stop == stop
