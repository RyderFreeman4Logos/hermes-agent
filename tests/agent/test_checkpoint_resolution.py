from plugins.context_engine import load_context_engine


def test_checkpoint_plugin_receives_context_checkpoint_configuration():
    engine = load_context_engine("checkpoint", {"mode": "live", "raw_history": True})
    assert engine is not None
    assert engine.mode == "live"
