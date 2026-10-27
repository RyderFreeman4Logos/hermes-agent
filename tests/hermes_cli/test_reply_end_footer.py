"""The CLI reply panel prints elapsed time even when cache usage is unknown."""

from types import SimpleNamespace

import cli as climod
from cli import HermesCLI


def test_interrupted_reply_prints_elapsed_without_fabricating_cache(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.final_response_markdown = "raw"
    cli._stream_started = False
    cli._stream_box_opened = False
    cli._streamed_text_this_turn = ""
    cli._last_turn_interrupted = True
    cli._prompt_duration = 4.2
    cli.agent = SimpleNamespace(_reply_usage=None, _reply_usage_complete=False, _last_turn_usage={"cache_read_tokens": 800, "prompt_tokens": 1000})
    printed = []
    monkeypatch.setattr(climod, "ChatConsole", lambda: SimpleNamespace(print=printed.append))
    monkeypatch.setattr(HermesCLI, "_scrollback_box_width", lambda self: 74)
    lines = []
    monkeypatch.setattr(climod, "_cprint", lines.append)
    turn = SimpleNamespace(
        use_streaming_tts=False, box_opened=False,
        result={"interrupted": True, "final_response": "kept partial"},
    )
    cli._chat_print_response_panel(turn, "kept partial")
    plain = "".join(lines)
    assert "4.2s" in plain
    assert "cache" not in plain
    assert "0/" not in plain
