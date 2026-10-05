"""Public dispatch/restart diagnostics for an owner lost before any child finishes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import sqlite3

import pytest


@pytest.mark.parametrize("split,missing_writer", [(False, False), (True, False), (True, True)])
def test_unfinished_delegation_recovery_keeps_transcript_locator(tmp_path, split, missing_writer):
    # Canonical fixture data must never appear in pytest's failure-local rendering.
    __tracebackhide__ = True
    repo = str(Path(__file__).resolve().parents[2])
    handle_path = tmp_path / "dispatch.json"
    # The source may be a read-only export without .git; owner diagnostics need a real checkout.
    owner_cwd = tmp_path / "private-owner-cwd-sentinel"
    owner_cwd.mkdir()
    subprocess.run(["git", "init", "-q", str(owner_cwd)], check=True)
    subprocess.run(["git", "-C", str(owner_cwd), "-c", "user.name=Diagnostic Fixture",
                    "-c", "user.email=fixture@example.invalid", "commit", "-q", "--allow-empty",
                    "-m", "private-commit-message-sentinel"], check=True)
    (owner_cwd / "private-owner-file").write_text("private contents", encoding="utf-8")
    subprocess.run(["git", "-C", str(owner_cwd), "branch", "-m",
                    "private-branch-sentinel"], check=True)
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": repo,
           "REPRO_HANDLE": str(handle_path), "REPRO_SPLIT": str(int(split)),
           "REPRO_MISSING": str(int(missing_writer))}
    (tmp_path / 'config.yaml').write_text('delegation:\n  independent_completions: true\n', encoding='utf-8')
    producer = r'''
import json, os, threading
from pathlib import Path
from unittest.mock import MagicMock
import tools.delegate_tool as dt
parent = MagicMock()
parent._delegate_depth = 0
parent.session_id = "diagnostic-parent"
parent._interrupt_requested = False
parent._active_children = []
parent._active_children_lock = None
started = threading.Event()
blocked = threading.Event()
def child(**kw):
    started.set()
    blocked.wait(60)
def build(**kw):
    c = MagicMock()
    c._delegate_role = "leaf"
    c._subagent_id = "diagnostic-child"
    return c
creds = dict(model="m", provider=None, base_url=None, api_key=None,
             api_mode=None, command=None, args=None)
dt._build_child_agent = build
dt._run_single_child = child
dt._resolve_delegation_credentials = lambda *a, **k: creds
if os.environ['REPRO_MISSING'] == '1':
    import tools.delegation_live_log as live
    original = live.create_live_transcripts
    def missing(*args, **kwargs):
        ident, writers, paths = original(*args, **kwargs)
        writers[0] = None
        return ident, writers, paths[1:]
    live.create_live_transcripts = missing
private_goal = 'private-task-prompt-sentinel'
private_context = 'private-task-context-sentinel Authorization: Bearer fixture-credential-sentinel'
kwargs = {'tasks': [{'goal': private_goal, 'context': private_context}, {'goal': private_goal, 'context': private_context}]} if os.environ['REPRO_SPLIT'] == '1' else {'goal': private_goal, 'context': private_context}
kwargs["context"] = private_context
handle = json.loads(dt.delegate_task(**kwargs, background=True, parent_agent=parent))
assert handle["status"] == "dispatched", handle
assert started.wait(10), "child did not start"
Path(os.environ["REPRO_HANDLE"]).write_text(json.dumps(handle), encoding="utf-8")
os._exit(0)
'''
    first = subprocess.run([sys.executable, "-c", producer], cwd=owner_cwd, env=env,
                           text=True, capture_output=True, timeout=30)
    assert first.returncode == 0, first.stdout + first.stderr
    handle = json.loads(handle_path.read_text(encoding="utf-8"))
    transcripts = handle["live_transcripts"]
    assert all(Path(path).is_file() for path in transcripts)
    # Canonical evidence remains authorized; recovery must only duplicate locators/metadata.
    private_evidence = 'private-file-content-sentinel request_body=private-request-body-sentinel'
    for path in transcripts:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(private_evidence + " Authorization: Bearer fixture-credential-sentinel\n")
    transcript_bytes = {path: Path(path).read_bytes() for path in transcripts}
    with sqlite3.connect(tmp_path / "state.db") as conn:
        canonical_before = dict(conn.execute("SELECT delegation_id, task_json FROM async_delegations"))
    owner_sha = subprocess.check_output(["git", "-C", str(owner_cwd), "rev-parse", "--short", "HEAD"],
                                        text=True).strip()
    consumer = r'''
import json, queue
from tools import async_delegation as ad
from tools.process_registry import format_process_notification
q = queue.Queue()
count = ad.restore_undelivered_completions(q)
events = [q.get_nowait() for _ in range(count)]
with ad._connect() as conn:
    durable = {row[0]: {"event_json": row[1], "result_json": row[2], "task_json": row[3]}
               for row in conn.execute("SELECT delegation_id, event_json, result_json, task_json FROM async_delegations")}
print(json.dumps([{"event": event, "message": format_process_notification(event), "durable": durable[event["delegation_id"]]} for event in events]))
'''
    second = subprocess.run([sys.executable, "-c", consumer], cwd=repo, env=env,
                            text=True, capture_output=True, timeout=20)
    if second.returncode != 0:
        # A rewritten assert would append CompletedProcess, including canonical stdout.
        raise AssertionError(f"consumer exit code {second.returncode}")
    restored = json.loads(second.stdout.strip().splitlines()[-1])
    assert len(restored) == (2 if split else 1)
    by_index = {}
    for item in restored:
        event = item['event']
        assert event["status"] == "unknown"
        assert event["last_known_status"] == "running"
        assert "running" in item["message"]
        assert not event.get("summary")
        paths = event['task_transcripts']
        assert len(paths) <= 1
        for index, path in paths.items():
            assert path in item['message']
            assert Path(path).name == f'task-{index}.log'
            assert index not in by_index
            by_index[index] = path
        for path in transcripts:
            assert (path in item['message']) == (path in paths.values())
        private_markers = (str(owner_cwd), "private-task-prompt-sentinel",
                           "private-task-context-sentinel", "fixture-credential-sentinel",
                           "private-file-content-sentinel", "private-request-body-sentinel",
                           "private-branch-sentinel", "private-commit-message-sentinel",
                           "private-owner-file", "private contents")
        def contains_private(payload):
            return any(marker in json.dumps(payload) for marker in private_markers)
        assert item["durable"]["task_json"] == canonical_before[event["delegation_id"]]
        assert event["parent_session_id"] == "diagnostic-parent"
        assert event["session_key"] == "diagnostic-parent"
        canonical = json.loads(item["durable"]["task_json"])
        assert len(event.get("goals") or []) == len(canonical.get("goals") or [])
        if canonical.get("is_batch"):
            assert f"{len(canonical['goals'])} subagent(s)" in item["message"]
        assert canonical["owner_cwd"] == str(owner_cwd)
        assert contains_private(canonical), "negative control must detect canonical private input"
        assert "private-task-prompt-sentinel" in json.dumps(canonical)
        assert "private-task-context-sentinel" in json.dumps(canonical)
        assert not contains_private({"status": "unknown", "task_transcripts": paths}), "safe control rejected"
        assert not contains_private({"event": event, "message": item["message"],
                                     "event_json": item["durable"]["event_json"],
                                     "result_json": item["durable"]["result_json"]}), "recovery privacy boundary leaked"
        assert not event.get("transcript_tails")
        assert "uncommitted file(s)" in event["git_state_hint"]
        assert "1 uncommitted file(s)" in event["git_state_hint"]
        assert owner_sha in event["git_state_hint"]
        assert "private-owner-file" not in event["git_state_hint"]
        assert "private contents" not in event["git_state_hint"]
        assert event["git_state_hint"] in item["message"]
    assert set(by_index) == ({'1'} if missing_writer else {'0', '1'} if split else {'0'})
    assert set(by_index.values()) == set(transcripts)
    assert {path: Path(path).read_bytes() for path in transcripts} == transcript_bytes


def _consumer_failure_projection_probe(tmp_path, monkeypatch):
    original_run = subprocess.run
    forced_consumer_metadata = []

    def fail_after_consumer_output(command, *args, **kwargs):
        force_failure = (
            isinstance(command, list)
            and len(command) == 3
            and command[0] == sys.executable
            and command[1] == "-c"
            and "restore_undelivered_completions" in command[2]
        )
        if force_failure:
            command = [*command[:2], f"{command[2]}\nraise SystemExit(86)\n"]
        result = original_run(command, *args, **kwargs)
        if force_failure:
            forced_consumer_metadata.append(
                (result.returncode, len(result.stdout or ""), len(result.stderr or ""))
            )
        return result

    monkeypatch.setattr(subprocess, "run", fail_after_consumer_output)
    try:
        test_unfinished_delegation_recovery_keeps_transcript_locator(
            tmp_path, split=False, missing_writer=False
        )
    except AssertionError as error:
        diagnostic = str(error)
        failure_class = "AssertionError"
    else:
        diagnostic = ""
        failure_class = "no_failure"

    private_markers = (
        "private-task-prompt-sentinel",
        "private-task-context-sentinel",
        "Authorization:",
        str(tmp_path / "private-owner-cwd-sentinel"),
    )
    with sqlite3.connect(tmp_path / "state.db") as conn:
        rows = conn.execute(
            "SELECT event_json, result_json, task_json FROM async_delegations"
        ).fetchall()
    return {
        "diagnostic_is_safe": diagnostic == "consumer exit code 86",
        "diagnostic_contains_private": any(marker in diagnostic for marker in private_markers),
        "failure_class": failure_class,
        "forced_consumer_count": len(forced_consumer_metadata),
        "forced_consumer_returncode": forced_consumer_metadata[0][0] if forced_consumer_metadata else None,
        "consumer_stdout_nonempty": (
            forced_consumer_metadata[0][1] > 0 if forced_consumer_metadata else False
        ),
        "row_count": len(rows),
        "noncanonical_data_contains_private": (
            any(marker in json.dumps(rows[0][:2]) for marker in private_markers) if rows else None
        ),
        "canonical_task_contains_private": (
            all(marker in rows[0][2] for marker in private_markers) if rows else None
        ),
    }


def test_consumer_failure_diagnostic_omits_canonical_fixture_data(tmp_path, monkeypatch):
    try:
        probe = _consumer_failure_projection_probe(tmp_path, monkeypatch)
    except Exception as error:
        probe = {"probe_exception_class": type(error).__name__}

    # The probe keeps canonical inputs/output in its own frame and returns only safe metadata;
    # pytest's failure-local rendering therefore cannot print the synthetic fixture payload.
    assert probe == {
        "diagnostic_is_safe": True,
        "diagnostic_contains_private": False,
        "failure_class": "AssertionError",
        "forced_consumer_count": 1,
        "forced_consumer_returncode": 86,
        "consumer_stdout_nonempty": True,
        "row_count": 1,
        "noncanonical_data_contains_private": False,
        "canonical_task_contains_private": True,
    }
