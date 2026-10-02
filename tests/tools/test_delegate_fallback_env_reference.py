"""Malformed fallback references cannot borrow provider credentials (#359)."""
import pytest

from tests.tools.test_delegate_local_alias_ownership import dispatched_child
from tests.tools.test_delegate_pool_runtime_transitions import offline as offline


@pytest.mark.parametrize("field", ["key_env", "api_key_env"])
@pytest.mark.parametrize("value", [[], {}, False, ["fixture-canary"], {"fixture-canary": 1}, 0, 23, True, None, "", "   "])
@pytest.mark.parametrize("phase", ["admission", "activation"])
@pytest.mark.parametrize("inline", [False, True])
def test_malformed_reference_never_borrows_provider_key(tmp_path, monkeypatch, caplog, field, value, phase, inline):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    hop = {"provider": "rotation-owner", "model": "bare-backup", field: value if phase == "admission" else "ROTATION_KEY"}
    if inline:
        hop["api_key"] = "fixture-inline"
    providers = {"rotation-owner": {"base_url": "https://rotation.invalid/v1", "api_key": "fixture-provider"}}
    monkeypatch.setenv("ROTATION_KEY", "fixture-launch")
    token = set_secret_scope({"ROTATION_KEY": "fixture-before"})
    caplog.set_level("INFO")
    try:
        with dispatched_child(tmp_path, hop, providers) as (child, clients):
            assert bool(child._fallback_chain) is (inline or phase == "activation")
            if phase == "activation":
                child._fallback_chain[0][field] = value
            assert child._try_activate_fallback() is inline
            assert child.model == ("bare-backup" if inline else "primary-model")
            assert clients[-1]["api_key"] == ("fixture-inline" if inline else "fixture-primary")
            assert len(clients) == (2 if inline else 1)
            assert "fixture-canary" not in caplog.text
    finally:
        reset_secret_scope(token)
