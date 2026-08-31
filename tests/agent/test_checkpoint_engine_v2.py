from agent.checkpoint_engine import CheckpointContextEngine
from plugins.context_engine import load_context_engine


def test_checkpoint_is_opt_in_shadow_noop():
    assert load_context_engine("checkpoint").name == "checkpoint"
    messages = [{"role": "user", "content": "hello"}]
    engine = CheckpointContextEngine()
    assert engine.compress(messages) is messages
    assert engine.compression_count == 0


def test_default_engine_is_compressor():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["context"]["engine"] == "compressor"
