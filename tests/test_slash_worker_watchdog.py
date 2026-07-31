import inspect
import io
import sys

from tui_gateway import slash_worker


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, getppid=lambda: 999999) is True


def test_is_orphaned_false_when_direct_parent_is_unchanged():
    original_ppid = 1234
    assert slash_worker._is_orphaned(original_ppid, getppid=lambda: original_ppid) is False


def test_parent_death_watchdog_contract_has_no_create_time_plumbing():
    assert list(inspect.signature(slash_worker._is_orphaned).parameters) == [
        "original_ppid",
        "getppid",
    ]
    assert list(inspect.signature(slash_worker._start_parent_death_watchdog).parameters) == [
        "original_ppid",
    ]


def test_command_completion_calls_memory_trim(monkeypatch):
    import hermes_cli.mem_trim as mem_trim

    calls = []
    monkeypatch.setattr(sys, "argv", ["slash_worker", "--session-key", "test"])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"id": 1, "command": "/help"}\n'))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(slash_worker, "_start_parent_death_watchdog", lambda _ppid: None)
    monkeypatch.setattr(slash_worker, "_prepare_slash_worker_runtime", lambda: None)
    monkeypatch.setattr(slash_worker, "HermesCLI", lambda **_kwargs: object())
    monkeypatch.setattr(slash_worker, "_run", lambda _cli, _command: "ok")
    monkeypatch.setattr(slash_worker, "handle_spurious_eof", lambda *_args: False)
    monkeypatch.setattr(
        mem_trim, "trim_memory", lambda **kwargs: calls.append(kwargs) or True
    )

    slash_worker.main()

    assert calls == [{"reason": "slash worker command completion"}]
