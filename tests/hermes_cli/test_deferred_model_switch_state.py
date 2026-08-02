"""Deferred model-switch state and committed compression seam tests."""

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


def test_scheduling_persists_only_a_secret_free_descriptor():
    agent = _Agent()
    agent.session_id = "session-1"
    agent._session_db = _SessionDB()
    agent._session_init_model_config = {"max_iterations": 7}
    result = _result()

    schedule_model_switch_after_compression(agent, result)

    stored = json.loads(agent._session_db.row["model_config"])
    descriptor = stored["pending_model_switch_after_compression"]
    assert descriptor == {
        "model": "new-model",
        "provider": "new-provider",
        "api_mode": "responses",
    }
    assert "new-key" not in json.dumps(stored)
    assert "new.example" not in json.dumps(stored)
    assert get_model_switch_after_compression(agent) is result


def test_successful_apply_runs_once_and_preserves_resolved_provider():
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
    assert get_model_switch_after_compression(agent) is None
    assert any("applied" in status.lower() for status in agent.statuses)


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
    assert any("client rebuild failed" in status for status in agent.statuses)


def test_durable_publication_failure_rolls_back_runtime_and_keeps_pending():
    agent = _Agent()
    agent.session_id = "session-1"
    agent._session_db = _SessionDB()
    agent._session_init_model_config = {"max_iterations": 7}
    agent._build_system_prompt = lambda _message: "new prompt"
    result = _result()
    schedule_model_switch_after_compression(agent, result)
    scheduled_config = agent._session_db.row["model_config"]
    agent._session_db.fail_route_publication = True

    assert apply_model_switch_after_compression(agent) == "failed"

    assert (agent.model, agent.provider, agent.api_key) == (
        "old-model",
        "old-provider",
        "old-key",
    )
    assert get_model_switch_after_compression(agent) is result
    assert agent._session_db.row["model"] == "old-model"
    assert agent._session_db.row["model_config"] == scheduled_config


def test_outer_commit_controls_application_and_orders_it_before_hook():
    agent = _Agent()
    result = _result()
    schedule_model_switch_after_compression(agent, result)

    _queue_context_engine_compression_notification(
        agent,
        new_session_id="child",
        old_session_id="parent",
    )
    assert finalize_context_engine_compression_notification(agent, committed=False) is False
    assert get_model_switch_after_compression(agent) is result
    assert agent.calls == []

    _queue_context_engine_compression_notification(
        agent,
        new_session_id="child",
        old_session_id="parent",
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
