from hermes_cli.config_defaults import DEFAULT_CONFIG


def test_checkpoint_defaults_are_safe_and_opt_in():
    context = DEFAULT_CONFIG["context"]
    assert context["engine"] == "compressor"
    assert context["checkpoint"]["mode"] == "shadow"
    assert context["checkpoint"]["structured_output"] == "required"
    assert "max_map_shards" not in context["checkpoint"]
