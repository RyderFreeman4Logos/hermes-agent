"""Contract tests for the shared warm-KV heartbeat."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading

import pytest

from tools.runtime_heartbeat import HeartbeatConfigError, RuntimeHeartbeat, resolve_interval


class _Timer:
    def __init__(self, _delay, callback):
        self.callback = callback
        self.cancelled = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.callback()


@dataclass
class _Completions:
    requests: list[dict] = field(default_factory=list)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return object()


class _Agent:
    provider = "custom"
    requested_provider = "custom:localrouter"
    api_mode = "chat_completions"
    model = "test-model"
    session_id = "session-1"

    def __init__(self):
        self.completions = _Completions()
        self.chat = type("Chat", (), {"completions": self.completions})()
        self.closed = []

    def _create_request_openai_client(self, *, reason, api_kwargs):
        assert reason == "heartbeat_warm"
        return self

    def _close_request_openai_client(self, client, *, reason):
        self.closed.append((client, reason))


@pytest.fixture
def heartbeat():
    return RuntimeHeartbeat(
        config={"providers": {"custom:localrouter": 30}},
        timer_factory=_Timer,
    )


def test_interval_requires_an_exact_provider_mapping():
    config = {"providers": {"custom:localrouter": 30, "custom": 10}}

    assert resolve_interval(config, "custom:localrouter") == 30
    with pytest.raises(HeartbeatConfigError):
        resolve_interval(config, "custom:other")


def test_two_children_share_one_timer_and_last_exit_cancels(heartbeat):
    agent = _Agent()

    heartbeat.register_child(agent, "process", "one")
    first_timer = heartbeat.timer_for(agent)
    heartbeat.register_child(agent, "subagent", "two")

    assert heartbeat.timer_for(agent) is first_timer
    heartbeat.complete_child(agent, "process", "one")
    assert heartbeat.timer_for(agent) is first_timer
    assert not first_timer.cancelled

    heartbeat.complete_child(agent, "subagent", "two")
    assert first_timer.cancelled
    assert heartbeat.timer_for(agent) is None


def test_loop_stop_restarts_exact_interval_while_children_remain(heartbeat):
    agent = _Agent()
    heartbeat.register_child(agent, "process", "one")
    first_timer = heartbeat.timer_for(agent)

    heartbeat.on_loop_stop(agent, completed=False)

    second_timer = heartbeat.timer_for(agent)
    assert first_timer.cancelled
    assert second_timer is not first_timer


def test_zero_child_idle_timer_arms_after_success_and_caller_activation_cancels(heartbeat):
    agent = _Agent()
    heartbeat.capture_successful_request(
        agent,
        {
            "model": "test-model",
            "messages": [{"role": "system", "content": "prefix"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "max_tokens": 99,
        },
    )

    heartbeat.on_loop_stop(agent, completed=True)
    timer = heartbeat.timer_for(agent)
    assert timer is not None

    heartbeat.on_caller_active(agent)
    assert timer.cancelled
    assert heartbeat.timer_for(agent) is None


def test_caller_activation_cancels_an_inflight_idle_warm(heartbeat):
    agent = _Agent()
    started = threading.Event()
    release = threading.Event()

    def blocking_create(**kwargs):
        agent.completions.requests.append(kwargs)
        started.set()
        assert release.wait(timeout=1)
        return object()

    agent.completions.create = blocking_create
    heartbeat.capture_successful_request(
        agent,
        {
            "model": "test-model",
            "messages": [{"role": "system", "content": "prefix"}],
        },
    )
    heartbeat.on_loop_stop(agent, completed=True)
    timer = heartbeat.timer_for(agent)

    worker = threading.Thread(target=timer.fire)
    worker.start()
    assert started.wait(timeout=1)
    heartbeat.on_caller_active(agent)
    release.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert heartbeat.timer_for(agent) is None


def test_warm_request_isolated_from_history_and_tool_execution(heartbeat):
    agent = _Agent()
    heartbeat.capture_successful_request(
        agent,
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "cacheable prefix"},
                {"role": "user", "content": "do not replay"},
            ],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "max_tokens": 99,
            "stream": True,
        },
    )
    heartbeat.on_loop_stop(agent, completed=True)

    heartbeat.timer_for(agent).fire()

    assert len(agent.completions.requests) == 1
    request = agent.completions.requests[0]
    assert request["messages"] == [{"role": "system", "content": "cacheable prefix"}]
    assert request["tools"] == [{"type": "function", "function": {"name": "read_file"}}]
    assert request["tool_choice"] == "none"
    assert request["max_tokens"] == 1
    assert request["stream"] is False
    assert agent.closed == [(agent, "heartbeat_warm_complete")]
