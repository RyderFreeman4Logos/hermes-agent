import threading

from tools.process_registry import ProcessRegistry, ProcessSession
from tui_gateway import server


def test_poller_uses_owner_observed_completion_disposition(monkeypatch):
    registry = ProcessRegistry()
    monkeypatch.setattr(
        "tools.approval.get_current_session_key",
        lambda default="": "tui-owner",
    )
    event = {
        "type": "completion",
        "session_id": "proc_observed",
        "session_key": "tui-owner",
        "started_at": 1.0,
        "command": "echo done",
        "exit_code": 0,
        "completion_reason": "exited",
        "termination_source": "",
        "output": "done",
    }
    registry._record_completion_observed(ProcessSession(
        id=event["session_id"], command=event["command"],
        session_key=event["session_key"], started_at=event["started_at"],
        notify_on_complete=True,
    ))
    registry.completion_queue.put(event)

    import tools.process_registry as process_registry_module

    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
    monkeypatch.setattr(
        server, "_notification_event_belongs_elsewhere", lambda *_args: False
    )
    monkeypatch.setattr(
        server, "_notification_event_requires_owner", lambda _event: True
    )
    monkeypatch.setattr(
        server, "_session_owns_notification_event", lambda *_args: True
    )
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("observed success must not open a turn")
        ),
    )

    stop = threading.Event()
    should_deliver = registry.completion_event_should_deliver

    def stop_after_disposition(current_event):
        result = should_deliver(current_event)
        stop.set()
        return result

    monkeypatch.setattr(
        registry, "completion_event_should_deliver", stop_after_disposition
    )
    server._notification_poller_loop(
        stop,
        event["session_key"],
        {
            "_finalized": False,
            "history_lock": threading.Lock(),
            "running": False,
            "session_key": event["session_key"],
        },
    )

    assert registry.completion_queue.empty()
