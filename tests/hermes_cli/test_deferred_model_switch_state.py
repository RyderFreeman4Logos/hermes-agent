"""Deferred route state is applied only after a committed boundary."""

import json
from types import SimpleNamespace

from agent.conversation_compression import (
    _queue_context_engine_compression_notification,
    finalize_context_engine_compression_notification,
)
from hermes_cli.model_switch import (
    ModelSwitchResult,
    apply_model_switch_after_compression,
    clear_model_switch_after_compression,
    get_model_switch_after_compression,
    restore_model_switch_after_compression,
    schedule_model_switch_after_compression,
)


class _Agent:
    def __init__(self):
        self.model = "old-model"
        self.provider = "old-provider"
        self.base_url = "https://old.example/v1"
        self.api_key = "old-key"
        self.api_mode = "chat_completions"
        self.statuses = []
        self.calls = []
        self.context_compressor = SimpleNamespace(on_session_start=self._on_session_start)

    def _emit_status(self, message):
        self.statuses.append(message)

    def _on_session_start(self, *_args, **_kwargs):
        self.calls.append(("context_hook", self.model, self.provider))

    def switch_model(self, model, provider, api_key, base_url, api_mode):
        self.calls.append(("switch", model, provider))
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.api_mode = api_mode


class _SessionDB:
    def __init__(self):
        self.row = {
            "model": "old-model",
            "model_config": json.dumps({"max_iterations": 7}),
            "system_prompt": "old prompt",
            "billing_provider": "old-provider",
            "billing_base_url": "https://old.example/v1",
            "billing_mode": "chat_completions",
        }
        self.fail_route_publication = False

    def get_session(self, _session_id):
        return dict(self.row)

    def update_session_meta(self, _session_id, model_config, model=None):
        if self.fail_route_publication and model == "new-model":
            raise RuntimeError("durable route publication failed")
        self.row["model_config"] = model_config
        if model is not None:
            self.row["model"] = model

    def update_system_prompt(self, _session_id, prompt):
        self.row["system_prompt"] = prompt

    def update_session_billing_route(
        self, _session_id, *, provider, base_url, billing_mode=None
    ):
        self.row.update(
            billing_provider=provider,
            billing_base_url=base_url,
            billing_mode=billing_mode,
        )


def _result(model="new-model", provider="new-provider"):
    return ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider=provider,
        api_key="new-key",
        base_url="https://new.example/v1",
        api_mode="responses",
        provider_label="New Provider",
    )


def test_scheduling_is_non_mutating_and_last_schedule_wins():
    agent = _Agent()
    first = _result("first", "provider-a")
    second = _result("second", "provider-b")

    assert schedule_model_switch_after_compression(agent, first) is None
    assert schedule_model_switch_after_compression(agent, second) is first
    assert (agent.model, agent.provider, agent.calls) == (
        "old-model",
        "old-provider",
        [],
    )
    assert get_model_switch_after_compression(agent) is second


def test_scheduling_persists_secret_free_descriptor():
    agent = _Agent()
    agent.session_id = "session-1"
    agent._session_db = _SessionDB()
    agent._session_init_model_config = {"max_iterations": 7}

    schedule_model_switch_after_compression(agent, _result())
    stored = json.loads(agent._session_db.row["model_config"])
    assert stored["pending_model_switch_after_compression"] == {
        "model": "new-model",
        "provider": "unresolved",
        "api_mode": "responses",
    }
    assert "new-key" not in json.dumps(stored)
    assert "new.example" not in json.dumps(stored)


def test_deferred_apply_keeps_runtime_url_but_never_persists_unsafe_route():
    agent = _Agent()
    agent.session_id = "session-1"
    agent._session_db = _SessionDB()
    agent._session_init_model_config = {"max_iterations": 7}
    result = _result()
    result.base_url = "https://user:route-marker@new.example/v1?api_key=query-marker&region=us"

    schedule_model_switch_after_compression(agent, result)
    assert apply_model_switch_after_compression(agent) == "applied"

    stored = json.loads(agent._session_db.row["model_config"])
    durable = json.dumps(stored) + json.dumps(agent._session_db.row)
    assert "route-marker" not in durable
    assert "query-marker" not in durable
    assert "base_url" not in stored
    assert agent.base_url == result.base_url
    assert agent._session_db.row["billing_base_url"] is None


def test_successful_apply_runs_once_and_orders_hook_after_switch():
    agent = _Agent()
    result = _result()
    callbacks = []
    schedule_model_switch_after_compression(
        agent,
        result,
        on_applied=lambda applied, old_model, old_provider: callbacks.append(
            (applied, old_model, old_provider, agent.model)
        ),
    )

    assert apply_model_switch_after_compression(agent) == "applied"
    assert apply_model_switch_after_compression(agent) == "none"
    assert agent.calls == [("switch", "new-model", "new-provider")]
    assert callbacks == [(result, "old-model", "old-provider", "new-model")]


def test_failed_apply_keeps_old_route_and_pending_intent():
    agent = _Agent()
    result = _result()

    def fail(*_args):
        agent.calls.append(("switch_failed", agent.model, agent.provider))
        raise RuntimeError("client rebuild failed")

    agent.switch_model = fail
    schedule_model_switch_after_compression(agent, result)
    assert apply_model_switch_after_compression(agent) == "failed"
    assert (agent.model, agent.provider) == ("old-model", "old-provider")
    assert get_model_switch_after_compression(agent) is result


def test_outer_commit_controls_application_and_orders_it_before_hook():
    agent = _Agent()
    result = _result()
    schedule_model_switch_after_compression(agent, result)

    _queue_context_engine_compression_notification(
        agent, new_session_id="child", old_session_id="parent"
    )
    assert finalize_context_engine_compression_notification(agent, committed=False) is False
    assert agent.calls == []

    _queue_context_engine_compression_notification(
        agent, new_session_id="child", old_session_id="parent"
    )
    assert finalize_context_engine_compression_notification(agent, committed=True) is True
    assert agent.calls == [
        ("switch", "new-model", "new-provider"),
        ("context_hook", "new-model", "new-provider"),
    ]


def test_clear_reports_cancelled_without_switching():
    agent = _Agent()
    result = _result()
    schedule_model_switch_after_compression(agent, result)
    assert clear_model_switch_after_compression(agent) is result
    assert get_model_switch_after_compression(agent) is None
    assert agent.calls == []


def test_restore_rehydrates_persisted_pending_intent(monkeypatch):
    agent = _Agent()
    agent.session_id = "session-1"
    agent._session_db = _SessionDB()
    agent._session_db.row["model_config"] = json.dumps(
        {
            "pending_model_switch_after_compression": {
                "model": "restored-model",
                "provider": "restored-provider",
                "api_mode": "responses",
                "reasoning_config": {"enabled": True, "effort": "low"},
            }
        }
    )

    import hermes_cli.config as cfg
    import hermes_cli.runtime_provider as rp
    config = {"providers": {"restored-provider": {
        "api": "https://restored.example.test/v1", "api_key": "restored-key",
    }}}
    monkeypatch.setattr(cfg, "load_config", lambda: config)
    monkeypatch.setattr(cfg, "load_config_readonly", lambda: config)
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)
    restored_result = restore_model_switch_after_compression(agent)
    assert restored_result is not None
    assert restored_result.base_url == "https://restored.example.test/v1"
    assert restored_result.api_key == "restored-key"
    assert get_model_switch_after_compression(agent) is restored_result
    assert restored_result.reasoning_config == {"enabled": True, "effort": "low"}


def test_restore_rehydrates_configured_custom_provider_without_mocking_switch(
    tmp_path, monkeypatch
):
    agent = _Agent()
    agent.session_id = "session-1"
    agent._session_db = _SessionDB()
    agent._session_db.row["model_config"] = json.dumps(
        {
            "pending_model_switch_after_compression": {
                "model": "team-model",
                "provider": "custom:team",
                "api_mode": "responses",
            }
        }
    )
    import hermes_cli.config as cfg
    import hermes_cli.runtime_provider as rp
    config = {"custom_providers": [{"name": "team", "base_url": "https://team.example/v1",
                                    "api_key": "test-key", "model": "team-model"}]}
    monkeypatch.setattr(cfg, "load_config", lambda: config)
    monkeypatch.setattr(cfg, "load_config_readonly", lambda: config)
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)

    restored_result = restore_model_switch_after_compression(agent)

    assert restored_result is not None
    assert restored_result.success is True
    assert restored_result.new_model == "team-model"
    assert restored_result.target_provider == "custom:team"
    assert restored_result.base_url == "https://team.example/v1"
    assert restored_result.api_key == "test-key"
    assert get_model_switch_after_compression(agent) is restored_result
