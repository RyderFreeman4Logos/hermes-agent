"""Deferred model-switch state and committed compression seam tests."""

import json
from types import SimpleNamespace

import pytest

from agent.models_dev import ModelInfo
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
        self.fail_billing_publication = False
        self.writes = 0

    def get_session(self, _session_id):
        return dict(self.row)

    def update_session_meta(self, _session_id, model_config, model=None):
        self.writes += 1
        if self.fail_route_publication and model == "new-model":
            raise RuntimeError("durable route publication failed")
        self.row["model_config"] = model_config
        if model is not None:
            self.row["model"] = model

    def update_system_prompt(self, _session_id, prompt):
        self.writes += 1
        self.row["system_prompt"] = prompt

    def publish_session_route(
        self,
        _session_id,
        *,
        model_config_json,
        model,
        system_prompt,
        billing_provider,
        billing_base_url,
        billing_mode,
    ):
        self.writes += 1
        if self.fail_route_publication and model == "new-model":
            raise RuntimeError("durable route publication failed")
        if self.fail_billing_publication and billing_provider == "new-provider":
            raise RuntimeError("billing route publication failed")
        self.row.update(
            model=model,
            model_config=model_config_json,
            system_prompt=system_prompt,
            billing_provider=billing_provider,
            billing_base_url=billing_base_url,
            billing_mode=billing_mode,
        )

    def update_session_billing_route(
        self, _session_id, *, provider, base_url, billing_mode=None
    ):
        self.writes += 1
        if self.fail_billing_publication and provider == "new-provider":
            raise RuntimeError("billing route publication failed")
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


def test_scheduling_uses_destination_custom_context_without_source_fallback(
    monkeypatch,
):
    agent = _Agent()
    agent.context_compressor.context_length = 200_000
    result = _result("custom-model", "custom-provider")
    result.model_info = None
    result.base_url = "https://custom.example/v1"
    calls = []

    def resolve_destination(model, base_url, **_kwargs):
        calls.append((model, base_url))
        return 32_000

    monkeypatch.setattr(
        "hermes_cli.config.get_custom_provider_context_length",
        resolve_destination,
    )

    schedule_model_switch_after_compression(agent, result)

    assert calls == [("custom-model", "https://custom.example/v1")]
    assert result.context_length == 32_000
    assert result.context_length != agent.context_compressor.context_length


def test_scheduling_prefers_destination_metadata_context(monkeypatch):
    agent = _Agent()
    agent.context_compressor.context_length = 200_000
    result = _result()
    result.model_info = ModelInfo(
        id="new-model",
        name="New Model",
        family="test",
        provider_id="new-provider",
        context_window=64_000,
    )
    monkeypatch.setattr(
        "hermes_cli.config.get_custom_provider_context_length",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("metadata destination must not need config fallback")
        ),
    )

    schedule_model_switch_after_compression(agent, result)

    assert result.context_length == 64_000


def test_scheduling_is_strictly_in_process_and_does_not_mutate_session_db():
    agent = _Agent()
    agent.session_id = "session-1"
    agent._session_db = _SessionDB()
    agent._session_init_model_config = {"max_iterations": 7}
    result = _result()

    schedule_model_switch_after_compression(agent, result)

    assert agent._session_db.writes == 0
    assert json.loads(agent._session_db.row["model_config"]) == {"max_iterations": 7}
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
    assert agent._session_db.writes == 1


def test_billing_publication_failure_rolls_back_runtime_and_keeps_pending():
    agent = _Agent()
    agent.session_id = "session-1"
    agent._session_db = _SessionDB()
    agent._session_init_model_config = {"max_iterations": 7}
    agent._build_system_prompt = lambda _message: "new prompt"
    result = _result()
    schedule_model_switch_after_compression(agent, result)
    original_row = dict(agent._session_db.row)
    agent._session_db.fail_billing_publication = True

    assert apply_model_switch_after_compression(agent) == "failed"

    assert (agent.model, agent.provider) == ("old-model", "old-provider")
    assert get_model_switch_after_compression(agent) is result
    assert agent._session_db.row == original_row


def test_missing_session_row_aborts_publication_and_keeps_pending(tmp_path):
    from hermes_state import SessionDB

    agent = _Agent()
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        agent.session_id = "missing-session"
        agent._session_db = db
        agent._session_init_model_config = {"max_iterations": 7}
        agent._build_system_prompt = lambda _message: "new prompt"
        result = _result()
        schedule_model_switch_after_compression(agent, result)

        assert apply_model_switch_after_compression(agent) == "failed"
        assert (agent.model, agent.provider) == ("old-model", "old-provider")
        assert get_model_switch_after_compression(agent) is result
        assert db.get_session(agent.session_id) is None
    finally:
        db.close()


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


def test_post_commit_publication_error_commits_deferred_switch_forward():
    agent = _Agent()
    agent.session_id = "session-1"
    agent._session_db = _SessionDB()
    agent._session_init_model_config = {"max_iterations": 7}
    agent._build_system_prompt = lambda _message: "new prompt"
    real_publish = agent._session_db.publish_session_route

    def commit_then_raise(*args, **kwargs):
        real_publish(*args, **kwargs)
        raise OSError("injected post-commit route failure")

    agent._session_db.publish_session_route = commit_then_raise
    result = _result()
    schedule_model_switch_after_compression(agent, result)

    assert apply_model_switch_after_compression(agent) == "applied"
    assert (agent.model, agent.provider) == ("new-model", "new-provider")
    assert agent._session_db.row["model"] == "new-model"
    assert agent._session_db.writes == 1
    assert get_model_switch_after_compression(agent) is None


@pytest.mark.parametrize("readback", ["unavailable", "third-state"])
def test_indeterminate_route_publication_is_explicitly_blocked(readback):
    agent = _Agent()
    agent.session_id = "session-1"
    agent._session_db = _SessionDB()
    agent._session_init_model_config = {"max_iterations": 7}
    agent._build_system_prompt = lambda _message: "new prompt"
    agent._session_db.fail_route_publication = True
    real_get = agent._session_db.get_session
    reads = 0

    def uncertain_readback(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads <= 3:
            return real_get(*args, **kwargs)
        if readback == "unavailable":
            raise OSError("injected route readback failure")
        row = real_get(*args, **kwargs)
        row["model"] = "third-model"
        return row

    agent._session_db.get_session = uncertain_readback
    result = _result()
    schedule_model_switch_after_compression(agent, result)

    assert apply_model_switch_after_compression(agent) == "failed"
    assert (agent.model, agent.provider) == ("old-model", "old-provider")
    assert agent._session_db.row["model"] == "old-model"
    assert agent._session_db.writes == 1
    assert get_model_switch_after_compression(agent) is result
    assert any("indeterminate" in status for status in agent.statuses)


def test_status_observer_failure_does_not_rollback_committed_switch():
    agent = _Agent()
    agent._emit_status = lambda _message: (_ for _ in ()).throw(
        OSError("injected observer failure")
    )
    result = _result()
    schedule_model_switch_after_compression(agent, result)

    assert apply_model_switch_after_compression(agent) == "applied"
    assert (agent.model, agent.provider) == ("new-model", "new-provider")
    assert get_model_switch_after_compression(agent) is None
