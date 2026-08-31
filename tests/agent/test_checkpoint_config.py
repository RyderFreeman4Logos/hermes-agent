from hermes_cli.config_defaults import DEFAULT_CONFIG


def test_checkpoint_defaults_are_safe_and_opt_in():
    context = DEFAULT_CONFIG["context"]
    assert context["engine"] == "compressor"
    assert context["checkpoint"] == {
        "mode": "shadow", "trace": False, "target_wire_tokens": 48000,
        "hard_max_wire_tokens": 60000, "map_concurrency": 2, "max_map_shards": 32,
    }
