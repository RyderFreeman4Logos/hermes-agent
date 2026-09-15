"""Public CLI wording for accepted background-process stop requests."""

from hermes_cli import cli_commands_mixin
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from tools import process_registry as process_registry_module
from tools.process_registry import ProcessRegistry, ProcessSession


def test_cli_stop_reports_an_accepted_stopping_request(monkeypatch, capsys):
    registry = ProcessRegistry()
    session = ProcessSession(id="proc-stopping", command="held pipe", task_id="session-a")
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "kill_process", lambda *_args, **_kwargs: {"status": "stopping"})
    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    monkeypatch.setattr(cli_commands_mixin, "_probe", lambda *_args: 0)

    CLICommandsMixin()._handle_stop_command()

    output = capsys.readouterr().out
    assert "Stop requested for 1 process(es)." in output
    assert "Stopped 1 process(es)." not in output
