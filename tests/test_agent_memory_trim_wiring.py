"""Agent teardown allocator-trim wiring."""

import threading

import hermes_cli.mem_trim as mem_trim
import run_agent
from run_agent import AIAgent


def test_agent_close_trims_after_release_once_without_force(monkeypatch):
    events = []
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "trim-close"
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.client = None
    agent._end_session_on_close = False
    agent._close_cached_request_openai_client = (
        lambda **kwargs: events.append(("release", kwargs))
    )

    monkeypatch.setattr(
        "tools.process_registry.process_registry.kill_all", lambda **_kwargs: None
    )
    monkeypatch.setattr(run_agent, "cleanup_vm", lambda _task_id: None)
    monkeypatch.setattr(run_agent, "cleanup_browser", lambda _task_id: None)
    monkeypatch.setattr(
        "tools.computer_use.release_computer_use_session", lambda _task_id: None
    )
    monkeypatch.setattr(
        mem_trim,
        "trim_memory",
        lambda **kwargs: events.append(("trim", kwargs)) or True,
    )

    agent.close()
    agent.close()

    assert events == [
        ("release", {"reason": "agent_close"}),
        ("trim", {"reason": "agent close"}),
    ]
