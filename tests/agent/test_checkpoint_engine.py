"""Checkpoint ContextEngine: opt-in shadow no-op (DESIGN.md §10 item 1)."""

from copy import deepcopy

from hermes_cli.config_defaults import DEFAULT_CONFIG
from plugins.context_engine import discover_context_engines, load_context_engine


def test_default_context_engine_remains_compressor():
    assert DEFAULT_CONFIG["context"]["engine"] == "compressor"


def test_checkpoint_engine_name_is_checkpoint():
    engine = load_context_engine("checkpoint")
    assert engine is not None
    assert engine.name == "checkpoint"


def test_discover_includes_checkpoint():
    names = [name for name, _desc, _available in discover_context_engines()]
    assert "checkpoint" in names


def test_shadow_compress_is_noop():
    engine = load_context_engine("checkpoint")
    assert engine is not None
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    original = deepcopy(messages)

    result = engine.compress(messages)

    assert result == original
    assert result is messages
    assert engine.compression_count == 0
