"""Remote session kernels (tools/code_kernel_remote.py) — hermes-agent#96873.

These tests drive execute_in_remote_kernel against a scripted fake env that
implements the same contract as docker/ssh/modal envs (run-to-completion
execute()), with canned outputs for the spawn/liveness/cell round-trips.
The REAL end-to-end behavior (actual detached processes, real files, real
kill) was verified live on Windows against a bash-backed env; these tests
pin the host-side protocol logic: spawn parsing, liveness handling,
state_lost/state_reset reporting, fail-open, and owner isolation.
"""
import base64
import contextvars
import json
import os
import shlex
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.code_kernel_remote import (
    _REGISTRY,
    _REMOTE_KERNELS,
    RemoteKernel,
    execute_in_remote_kernel,
    shutdown_all_remote_kernels,
    shutdown_remote_kernels_for_owner,
)


class ScriptedEnv:
    """Contract-faithful fake: answers env.execute() from a script table.

    Handlers are (substring, callable) pairs checked in order; the callable
    receives the command and returns the result dict.
    """

    def __init__(self, handlers):
        self.handlers = handlers
        self.commands = []

    def get_temp_dir(self):
        return "/tmp"

    def execute(self, command, cwd=None, timeout=None):
        self.commands.append(command)
        for needle, handler in self.handlers:
            if needle in command:
                return handler(command)
        return {"output": "", "returncode": 0}


class FileAwareEnv:
    """In-memory implementation of the remote file protocol.

    The host still executes the public remote-kernel path, including request
    staging, atomic rename, result polling and cleanup.  Only the remote
    process/filesystem boundary is replaced.
    """

    def __init__(self):
        self.commands = []
        self.files = {}
        self.spawn_count = 0
        self.killed = []
        self.before_submit = None
        self.before_liveness = None
        self.before_liveness_command = None
        self.liveness_response = None
        self.before_spawn = None
        self.before_kill = None
        self.cleanup_returncode = 0
        self.result_cleanup_error = None
        self.submit_error_after_apply = None
        self.result_payload = None
        self.cell_submissions = []
        self.emit_rpc = False
        self.rpc_done = threading.Event()
        self.hold_stage = None
        self.stage_entered = threading.Event()
        self.stage_release = threading.Event()
        self._held_stage = False
        self._lock = threading.Lock()

    def hold_once(self, stage):
        with self._lock:
            should_hold = self.hold_stage == stage and not self._held_stage
            if should_hold:
                self._held_stage = True
        if should_hold:
            self.stage_entered.set()
            if not self.stage_release.wait(5):
                raise AssertionError(f"timed out holding {stage}")

    def get_temp_dir(self):
        return "/tmp"

    def ship(self, _env, path, content):
        with self._lock:
            self.files[path] = content
        if path.endswith("/hermes_tools.py"):
            self.hold_once("before_launch_admission")

    def execute(self, command, cwd=None, timeout=None, **_kwargs):
        del cwd, timeout
        with self._lock:
            self.commands.append(command)
        if "nohup" in command:
            if self.before_spawn is not None:
                self.before_spawn()
            with self._lock:
                self.spawn_count += 1
                pid = str(7000 + self.spawn_count)
            return {"output": f"PID:{pid}\n", "returncode": 0}
        if "kill -0" in command:
            if self.before_liveness is not None:
                self.before_liveness()
            if self.before_liveness_command is not None:
                self.before_liveness_command(command)
            if self.liveness_response is not None:
                return self.liveness_response(command)
            return {"output": "ALIVE\n", "returncode": 0}
        if "command -v python3" in command:
            return {"output": "OK\n", "returncode": 0}
        if command.startswith("ls -1 ") and "/rpc/req_*" in command:
            with self._lock:
                paths = sorted(path for path in self.files if "/rpc/req_" in path)
            return {"output": "\n".join(paths), "returncode": 0}
        if command.startswith("cat ") and "/rpc/req_" in command:
            path = shlex.split(command)[1]
            with self._lock:
                output = self.files.get(path, "")
            self.hold_once("rpc_request_read")
            return {"output": output, "returncode": 0}
        if command.startswith("echo '") and "base64 -d >" in command:
            encoded = command.split("'", 2)[1]
            decoded = base64.b64decode(encoded).decode("utf-8")
            raw_target = command.split("base64 -d >", 1)[1].split("&&", 1)[0].strip()
            target = raw_target.removesuffix(".tmp")
            is_rpc_response = "/rpc/res_" in target
            if is_rpc_response:
                self.hold_once("rpc_response_rename")
            with self._lock:
                self.files[target] = decoded
            if is_rpc_response:
                self.rpc_done.set()
            return {"output": "", "returncode": 0}
        if command.startswith("pkill -TERM"):
            if self.before_kill is not None:
                self.before_kill(command)
            with self._lock:
                self.killed.append(command)
            return {"output": "", "returncode": 0}
        if "rm -f" in command and "/rpc/req_*" in command:
            return {"output": "", "returncode": self.cleanup_returncode}
        if command.startswith("rm -f ") and "/rpc/req_" in command:
            path = shlex.split(command)[2]
            with self._lock:
                self.files.pop(path, None)
            return {"output": "", "returncode": 0}
        if command.startswith("mv ") and "/cells/cell_req_" in command:
            if self.before_submit is not None:
                self.before_submit()
            source, target = shlex.split(command)[1:3]
            with self._lock:
                request = json.loads(self.files.pop(source))
                self.cell_submissions.append(request)
                self.files[target] = json.dumps(request)
                if self.emit_rpc:
                    rpc_dir = target.split("/cells/", 1)[0] + "/rpc"
                    rpc_path = f"{rpc_dir}/req_{request['id']}"
                    self.files[rpc_path] = json.dumps({
                        "token": "fixed-rpc-token", "seq": int(request["id"]),
                        "tool": "read_file", "args": {"path": request["code"]},
                    })
                result_path = target.replace("cell_req_", "cell_res_")
                payload = self.result_payload
                if callable(payload):
                    payload = payload(request)
                if payload is None:
                    payload = _cell(stdout=request["code"], execution_count=int(request["id"]))
                self.files[result_path] = json.dumps(payload)
            if self.emit_rpc and not self.rpc_done.wait(5):
                raise AssertionError("real RPC poller did not dispatch request")
            self.rpc_done.clear()
            if self.submit_error_after_apply is not None:
                raise self.submit_error_after_apply
            return {"output": "", "returncode": 0}
        if command.startswith("cat ") and "/cells/cell_res_" in command:
            path = shlex.split(command)[1]
            with self._lock:
                output = self.files.get(path, "")
            self.hold_once("cell_result_read")
            return {"output": output, "returncode": 0}
        if command.startswith("rm -f ") and "/cells/cell_res_" in command:
            if self.result_cleanup_error is not None:
                raise self.result_cleanup_error
            path = shlex.split(command)[2]
            with self._lock:
                self.files.pop(path, None)
            return {"output": "", "returncode": 0}
        return {"output": "", "returncode": 0}


def _spawn_ok_handlers(cell_results):
    """Handlers for a healthy kernel: spawn returns PID, liveness ALIVE,
    cat of a cell result file returns the next canned payload."""
    results = list(cell_results)

    def cat_handler(command):
        if results:
            return {"output": json.dumps(results.pop(0)), "returncode": 0}
        return {"output": "", "returncode": 0}

    return [
        ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
        ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
        ("cat ", cat_handler),
    ]


def _cell(status="ok", stdout="", execution_count=1, **kw):
    payload = {
        "id": "000001", "status": status, "stdout": stdout, "stderr": "",
        "stdout_clipped": False, "stderr_clipped": False, "traceback": "",
        "execution_count": execution_count,
    }
    payload.update(kw)
    return payload


def _run(env, code="print(1)", *, task="t1", reset=False, timeout=10,
         tools=frozenset({"read_file"})):
    return execute_in_remote_kernel(
        code, env=env, env_type="ssh", task_env_id=task,
        sandbox_tools=tools, timeout=timeout,
        max_tool_calls=5, reset=reset,
    )


class RemoteKernelBase(unittest.TestCase):
    def setUp(self):
        shutdown_all_remote_kernels()
        # No approval session key in tests → owner falls back to task id,
        # which is exactly the isolation-by-key behavior under test.
        self._ship = patch(
            "tools.code_execution_tool._ship_file_to_remote",
        )
        self._ship_mock = self._ship.start()
        self._poll = patch(
            "tools.code_execution_tool._rpc_poll_loop",
        )
        self._poll_mock = self._poll.start()

    def tearDown(self):
        self._ship.stop()
        self._poll.stop()
        shutdown_all_remote_kernels()
        from tools import code_kernel_remote
        with _REGISTRY.lock:
            self.assertEqual(getattr(code_kernel_remote, "_ACTIVE_INVOCATIONS", set()), set())


class TestSpawnAndReuse(RemoteKernelBase):
    def test_first_call_spawns_second_reuses(self):
        env = ScriptedEnv(_spawn_ok_handlers(
            [_cell(stdout="one\n"), _cell(stdout="two\n", execution_count=2)],
        ))
        first = _run(env)
        self.assertEqual(first["status"], "success", first)
        self.assertFalse(first["kernel"]["reused"])
        second = _run(env)
        self.assertTrue(second["kernel"]["reused"])
        self.assertEqual(second["kernel"]["execution_count"], 2)
        # Exactly one spawn happened.
        self.assertEqual(
            sum(1 for c in env.commands if "nohup" in c), 1,
        )

    def test_spawn_failure_fails_open(self):
        env = ScriptedEnv([
            ("nohup", lambda c: {"output": "sh: cannot fork\n", "returncode": 1}),
        ])
        self.assertIsNone(_run(env))
        self.assertEqual(len(_REMOTE_KERNELS), 0)

    def test_reset_kills_and_respawns(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env)
        result = _run(env, reset=True)
        self.assertTrue(result["kernel"].get("state_reset"))
        self.assertFalse(result["kernel"]["reused"])
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)


class TestDeathDetection(RemoteKernelBase):
    def test_dead_kernel_is_reported_and_respawned(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env)
        # Flip liveness to dead for the next probe only.
        original = env.handlers
        env.handlers = [("kill -0", lambda c: {"output": "", "returncode": 1})] \
            + [h for h in original if h[0] != "kill -0"]
        # Restore ALIVE after the respawn's own probe would run: the spawn
        # path probes liveness once — make the dead answer one-shot.
        state = {"dead_probes": 0}

        def flaky_liveness(command):
            state["dead_probes"] += 1
            if state["dead_probes"] == 1:
                return {"output": "", "returncode": 1}
            return {"output": "ALIVE\n", "returncode": 0}

        env.handlers = [("kill -0", flaky_liveness)] + \
            [h for h in original if h[0] != "kill -0"]
        result = _run(env)
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(result["kernel"].get("state_lost"))
        self.assertIn("state from earlier calls was lost",
                      result["kernel"].get("note", ""))

    def test_cell_timeout_kills_kernel_and_reports(self):
        # cat never returns a result file → cell deadline expires.
        env = ScriptedEnv([
            ("nohup", lambda c: {"output": "PID:77\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("cat ", lambda c: {"output": "", "returncode": 0}),
        ])
        result = _run(env, timeout=2)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["kernel"]["state_lost"])
        self.assertEqual(len(_REMOTE_KERNELS), 0)
        # The kernel was actually killed on the remote.
        self.assertTrue(any("kill " in c for c in env.commands))


class TestOwnershipIsolation(RemoteKernelBase):
    def test_changed_tool_set_spawns_kernel_with_fresh_stubs(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env, tools=frozenset({"read_file"}))
        _run(env, tools=frozenset({"web_search"}))

        self.assertEqual(len(_REMOTE_KERNELS), 2)
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)
        keyed_tool_sets = {key[-1] for key in _REMOTE_KERNELS}
        self.assertEqual(
            keyed_tool_sets,
            {("read_file",), ("web_search",)},
        )

    def test_delegated_children_get_their_own_remote_kernels(self):
        """Same invariant as local (#94647 review fix): the child context
        qualifier must key a DIFFERENT remote kernel."""
        from agent.delegation_context import delegated_child_context

        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env, task="conv")
        with delegated_child_context("child-9"):
            _run(env, task="conv")
        # Two distinct kernels, two spawns.
        self.assertEqual(len(_REMOTE_KERNELS), 2)
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)

    def test_owner_disposal_reaps_only_that_owner(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env, task="owner-a")
        _run(env, task="owner-b")
        self.assertEqual(len(_REMOTE_KERNELS), 2)
        shutdown_remote_kernels_for_owner("owner-a")
        self.assertEqual(len(_REMOTE_KERNELS), 1)
        remaining_owner = next(iter(_REMOTE_KERNELS))[0]
        self.assertEqual(remaining_owner, "owner-b")


class TestIdleReapAndCapEviction(RemoteKernelBase):
    """Unlike local session kernels, remote kernels had no idle-reap or
    process-wide cap: _REMOTE_KERNELS grew one entry per distinct
    (owner, env_type, task_env_id) that was never revisited, for the life
    of the gateway process."""

    def test_acquire_publishes_reserved_and_rolls_back_on_error(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell()]))
        acquired = threading.Event()
        release = threading.Event()
        errors = []

        from tools import code_kernel_remote as remote
        real_acquire = remote._acquire_remote_kernel

        def pause_after_acquire(*args, **kwargs):
            result = real_acquire(*args, **kwargs)
            acquired.set()
            release.wait(5)
            return result

        def run_cell():
            try:
                _run(env, task="reserved")
            except BaseException as exc:
                errors.append(exc)

        with patch.object(remote, "_acquire_remote_kernel", pause_after_acquire):
            worker = threading.Thread(target=run_cell)
            worker.start()
            try:
                self.assertTrue(acquired.wait(5))
                with _REGISTRY.lock:
                    attached = [kernel.attached for kernel in _REMOTE_KERNELS.values()]
                self.assertEqual(attached, [1])
            finally:
                release.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])

        shutdown_all_remote_kernels()
        failing_env = ScriptedEnv(_spawn_ok_handlers([_cell()]))
        with patch.object(remote, "_evict_over_cap_unlocked", side_effect=RuntimeError("cap read failed")):
            with self.assertRaisesRegex(RuntimeError, "cap read failed"):
                _run(failing_env, task="reservation-error")
        with _REGISTRY.lock:
            self.assertTrue(all(kernel.attached == 0 for kernel in _REMOTE_KERNELS.values()))

    def test_concurrent_same_key_spawn_uses_one_reserved_winner(self):
        from tools import code_kernel_remote as remote

        env = ScriptedEnv(_spawn_ok_handlers([]))
        spawn_barrier = threading.Barrier(2)
        spawned = []
        killed = []
        results = []
        errors = []

        def concurrent_spawn(env, env_type, owner, task_env_id, sandbox_tools, *, idle_exit, invocation):
            del invocation
            kernel = RemoteKernel(
                env=env, env_type=env_type, kernel_dir=f"/tmp/kernel-{len(spawned)}",
                pid=str(5000 + len(spawned)), rpc_token="synthetic", owner=owner,
            )
            spawned.append(kernel)
            spawn_barrier.wait(5)
            return kernel

        def acquire():
            try:
                with remote._remote_invocation("same-owner") as invocation:
                    results.append(remote._acquire_remote_kernel(
                        env, "ssh", "same-owner", "same-task", frozenset(),
                        reset=False, idle_exit=1800, invocation=invocation,
                    ))
            except BaseException as exc:
                errors.append(exc)

        with patch.object(remote, "_spawn_remote_kernel", concurrent_spawn), \
             patch.object(RemoteKernel, "kill", lambda kernel: killed.append(kernel)):
            workers = [threading.Thread(target=acquire) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(5)

        self.assertEqual(errors, [])
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(len(results), 2)
        self.assertIs(results[0][0], results[1][0])
        self.assertEqual(sorted(result[1] for result in results), [False, True])
        with _REGISTRY.lock:
            self.assertEqual(len(_REMOTE_KERNELS), 1)
            self.assertEqual(next(iter(_REMOTE_KERNELS.values())).attached, 2)
            next(iter(_REMOTE_KERNELS.values())).attached -= 2
        self.assertEqual(len(killed), 1)
        self.assertIn(killed[0], spawned)

    def test_idle_expired_kernel_is_reaped_on_next_call(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        execute_in_remote_kernel(
            "print(1)", env=env, env_type="ssh", task_env_id="stale",
            sandbox_tools=frozenset(), timeout=10, max_tool_calls=5,
            reset=False, idle_exit=1800,
        )
        self.assertEqual(len(_REMOTE_KERNELS), 1)
        # Backdate the kernel's last_used past the idle window — simulates
        # a key that is never revisited again.
        for kernel in _REMOTE_KERNELS.values():
            kernel.last_used -= 2000
        # A call for a DIFFERENT key must reap the stale entry on entry,
        # without ever touching or reviving it.
        execute_in_remote_kernel(
            "print(1)", env=env, env_type="ssh", task_env_id="fresh",
            sandbox_tools=frozenset(), timeout=10, max_tool_calls=5,
            reset=False, idle_exit=1800,
        )
        owners = {key[0] for key in _REMOTE_KERNELS}
        self.assertNotIn("stale", owners)
        self.assertIn("fresh", owners)

    def test_over_cap_evicts_least_recently_used(self):
        with patch("tools.code_kernel._lifecycle_limits", return_value=(2, 1800)):
            env = ScriptedEnv(_spawn_ok_handlers([_cell() for _ in range(10)]))
            for i in range(3):
                execute_in_remote_kernel(
                    "print(1)", env=env, env_type="ssh", task_env_id=f"owner-{i}",
                    sandbox_tools=frozenset(), timeout=10, max_tool_calls=5,
                    reset=False, idle_exit=1800,
                )
            self.assertEqual(len(_REMOTE_KERNELS), 2)
            owners = {key[0] for key in _REMOTE_KERNELS}
            self.assertNotIn("owner-0", owners)
            self.assertIn("owner-1", owners)
            self.assertIn("owner-2", owners)

    def test_equal_age_eviction_preserves_registry_insertion_order(self):
        """W13: equal-age idle victims use the registry's stable order."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        with patch("tools.code_kernel._lifecycle_limits", return_value=(3, 1800)):
            self.assertEqual(_run(env, task="equal-first")["status"], "success")
            self.assertEqual(_run(env, task="equal-second")["status"], "success")
        with _REGISTRY.lock:
            equal_time = time.monotonic() - 10
            for kernel in _REMOTE_KERNELS.values():
                kernel.last_used = equal_time
        with patch("tools.code_kernel._lifecycle_limits", return_value=(2, 1800)):
            self.assertEqual(_run(env, task="equal-trigger")["status"], "success")

        with _REGISTRY.lock:
            owners = [key[0] for key in _REMOTE_KERNELS]
        self.assertEqual(owners, ["equal-second", "equal-trigger"])
        self.assertTrue(any("7001" in command for command in env.killed))
        self.assertFalse(any("7002" in command for command in env.killed))

    def test_all_attached_kernels_may_exceed_cap_until_next_acquire(self):
        """W13: no eligible victim never kills active work or a waiter."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        entered = {name: threading.Event() for name in ("attached-a", "attached-b")}
        release = {name: threading.Event() for name in entered}

        def hold_submitted_cell():
            name = threading.current_thread().name
            if name in entered:
                entered[name].set()
                release[name].wait(5)

        env.before_submit = hold_submitted_cell
        results = {}
        errors = []

        def invoke(name):
            try:
                results[name] = _run(env, code=name, task=name)
            except BaseException as exc:
                errors.append(exc)

        workers = [threading.Thread(target=invoke, args=(name,), name=name) for name in entered]
        with patch("tools.code_kernel._lifecycle_limits", return_value=(1, 1800)):
            for worker in workers:
                worker.start()
            self.assertTrue(all(event.wait(5) for event in entered.values()))
            with _REGISTRY.lock:
                self.assertEqual(len(_REMOTE_KERNELS), 2)
                self.assertEqual([kernel.attached for kernel in _REMOTE_KERNELS.values()], [1, 1])
            self.assertEqual(env.killed, [])
            for event in release.values():
                event.set()
            for worker in workers:
                worker.join(5)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertEqual(errors, [])
            self.assertTrue(all(result["status"] == "success" for result in results.values()))
            with _REGISTRY.lock:
                self.assertEqual(len(_REMOTE_KERNELS), 2, "detach alone must not trim")
            env.before_submit = None
            self.assertEqual(_run(env, task="trim-trigger")["status"], "success")

        with _REGISTRY.lock:
            self.assertEqual([key[0] for key in _REMOTE_KERNELS], ["trim-trigger"])
        self.assertTrue(any("7001" in command for command in env.killed))
        self.assertTrue(any("7002" in command for command in env.killed))

    def test_mutex_queue_time_does_not_consume_cell_timeout(self):
        """W13: the per-cell timeout begins after an attached waiter owns K."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        first_submitted = threading.Event()
        release_first = threading.Event()

        def hold_first_cell():
            if threading.current_thread().name == "w13-first":
                first_submitted.set()
                release_first.wait(5)

        env.before_submit = hold_first_cell
        results = {}
        errors = []

        def invoke(label, timeout):
            try:
                results[label] = _run(
                    env, code=label, task="w13-timeout-owner", timeout=timeout
                )
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(target=invoke, args=("first", 10), name="w13-first")
        waiter = threading.Thread(target=invoke, args=("waiter", 1), name="w13-waiter")
        first.start()
        try:
            self.assertTrue(first_submitted.wait(5))
            waiter.start()
            deadline = time.monotonic() + 5
            while True:
                with _REGISTRY.lock:
                    attached = sum(kernel.attached for kernel in _REMOTE_KERNELS.values())
                if attached == 2:
                    break
                self.assertLess(time.monotonic(), deadline, "waiter did not reserve the active kernel")
                time.sleep(0.01)
            queue_started = time.monotonic()
            waiter.join(1.2)
            self.assertTrue(waiter.is_alive(), "waiter ran before the cell owner released K")
            self.assertGreaterEqual(time.monotonic() - queue_started, 1.0)
            self.assertEqual(env.cell_submissions, [])
        finally:
            release_first.set()
            first.join(5)
            if waiter.ident is not None:
                waiter.join(5)

        self.assertFalse(first.is_alive())
        self.assertFalse(waiter.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results["first"]["status"], "success")
        self.assertEqual(results["waiter"]["status"], "success")
        self.assertEqual(results["waiter"]["stdout"], "waiter")
        self.assertEqual([request["code"] for request in env.cell_submissions], ["first", "waiter"])

    def test_eviction_skips_kernels_with_a_running_cell(self):
        """Cap eviction must never kill a kernel mid-cell (the local-kernel
        race from hermes-agent#101861): a busy kernel stays put and a
        settled one goes instead, even if the busy one is older."""
        gate = threading.Event()

        def slow_cat(command):
            gate.wait(10)
            return {"output": json.dumps(_cell()), "returncode": 0}

        busy_env = ScriptedEnv([
            ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("cat ", slow_cat),
        ])
        with patch("tools.code_kernel._lifecycle_limits", return_value=(1, 1800)):
            worker = threading.Thread(target=_run, args=(busy_env,), kwargs={"task": "busy"})
            worker.start()
            deadline = time.monotonic() + 5
            while True:
                with _REGISTRY.lock:
                    busy_attached = any(k.attached for k in _REMOTE_KERNELS.values())
                if busy_attached:
                    break
                self.assertTrue(worker.is_alive())
                self.assertLess(time.monotonic(), deadline)
                gate.wait(0.01)
            env = ScriptedEnv(_spawn_ok_handlers([_cell()]))
            _run(env, task="settled")
            owners = {key[0] for key in _REMOTE_KERNELS}
            self.assertIn("busy", owners)
            gate.set()
            worker.join(10)
        self.assertFalse(any("kill 4242" in c for c in busy_env.commands))

    def test_zero_and_negative_capacity_normalize_to_one(self):
        """W13: configured non-positive caps retain the selected attachment."""
        for configured in (0, -2):
            with self.subTest(configured=configured):
                shutdown_all_remote_kernels()
                env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
                with patch("tools.code_execution_tool._load_config", return_value={
                    "max_session_kernels": configured,
                }):
                    _run(env, task=f"first-{configured}")
                    _run(env, task=f"second-{configured}")
                self.assertEqual(len(_REMOTE_KERNELS), 1)
                self.assertEqual(next(iter(_REMOTE_KERNELS.values())).attached, 0)

    def test_reservation_overflow_releases_attachment_and_invocation(self):
        """W14: a cap conversion failure leaves no attachment or call record."""
        from tools import code_kernel_remote as remote

        env = ScriptedEnv(_spawn_ok_handlers([_cell()]))
        with patch("tools.code_execution_tool._load_config", return_value={
            "max_session_kernels": float("inf"),
        }):
            with self.assertRaises(OverflowError):
                _run(env, task="cap-overflow")
        with _REGISTRY.lock:
            self.assertTrue(all(kernel.attached == 0 for kernel in _REMOTE_KERNELS.values()))
            self.assertEqual(getattr(remote, "_ACTIVE_INVOCATIONS", set()), set())


class TestRemoteInvocationOwnership(RemoteKernelBase):
    def test_same_kernel_cells_own_the_complete_file_rpc_namespace(self):
        """W01: cold racers adopt one K but serialize its complete namespace."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        cold_spawns = threading.Barrier(2)
        env.before_spawn = lambda: cold_spawns.wait(5)
        first_submitted = threading.Event()
        release_first = threading.Event()
        overlap = threading.Event()
        submission_count = [0]

        def before_submit():
            submission_count[0] += 1
            if submission_count[0] == 1:
                first_submitted.set()
                release_first.wait(5)
            else:
                overlap.set()

        env.before_submit = before_submit
        results = {}
        errors = []

        def invoke(label):
            try:
                results[label] = _run(env, code=label, task="shared")
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(target=invoke, args=("alpha",))
        second = threading.Thread(target=invoke, args=("beta",))
        first.start()
        second.start()
        self.assertTrue(first_submitted.wait(5))
        try:
            self.assertFalse(overlap.wait(0.2), "second cell submitted before first retired its poller")
        finally:
            release_first.set()
            first.join(5)
            second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results["alpha"]["stdout"], "alpha")
        self.assertEqual(results["beta"]["stdout"], "beta")
        self.assertEqual(env.spawn_count, 2)
        self.assertEqual(len(env.killed), 1)
        self.assertEqual(len(_REMOTE_KERNELS), 1)

    def test_owner_shutdown_cancels_identity_retry_before_republication(self):
        """W04/W06: cleanup invalidates a live invocation across acquire retry."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        self.assertEqual(_run(env, code="seed", task="owner-a")["status"], "success")
        probe_started = threading.Event()
        release_probe = threading.Event()
        probes = [0]

        def pause_reuse_probe():
            probes[0] += 1
            if probes[0] == 1:
                probe_started.set()
                release_probe.wait(5)

        env.before_liveness = pause_reuse_probe
        result = {}

        worker = threading.Thread(
            target=lambda: result.setdefault("value", _run(env, code="late", task="owner-a"))
        )
        worker.start()
        self.assertTrue(probe_started.wait(5))
        shutdown_remote_kernels_for_owner("owner-a")
        release_probe.set()
        worker.join(5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result["value"]["status"], "error")
        self.assertIn("canceled by session cleanup", result["value"]["error"])
        self.assertEqual(env.spawn_count, 1)
        self.assertEqual(len(_REMOTE_KERNELS), 0)

    def test_owner_shutdown_cancels_attached_cell_waiter(self):
        """W06/W14: a mutex waiter keeps its reservation but cannot run after cleanup."""
        from tools import code_kernel_remote as remote

        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        first_submitted = threading.Event()
        release_first = threading.Event()

        def hold_first_submit():
            if not first_submitted.is_set():
                first_submitted.set()
                release_first.wait(5)

        env.before_submit = hold_first_submit
        results = {}
        workers = [
            threading.Thread(target=lambda code=code: results.setdefault(
                code, _run(env, code=code, task="waiter-owner")
            ), name=f"w14-{code}")
            for code in ("first", "waiting")
        ]
        workers[0].start()
        self.assertTrue(first_submitted.wait(5))
        workers[1].start()
        deadline = time.monotonic() + 5
        while True:
            with _REGISTRY.lock:
                attached = sum(kernel.attached for kernel in _REMOTE_KERNELS.values())
            if attached == 2:
                break
            self.assertLess(time.monotonic(), deadline)
            threading.Event().wait(0.01)
        self.assertTrue(workers[1].is_alive())
        self.assertEqual(env.cell_submissions, [])
        with _REGISTRY.lock:
            kernel = next(iter(_REMOTE_KERNELS.values()))
        shutdown_remote_kernels_for_owner("waiter-owner")
        release_first.set()
        for worker in workers:
            worker.join(5)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(set(results), {"first", "waiting"})
        self.assertTrue(all(result["status"] == "error" for result in results.values()))
        self.assertTrue(all("canceled by session cleanup" in result["error"]
                            for result in results.values()))
        self.assertEqual(sum(c.startswith("mv ") for c in env.commands), 1)
        self.assertFalse(any("python3 script.py" in command for command in env.commands))
        self.assertEqual(kernel.attached, 0)
        self.assertEqual(len(_REMOTE_KERNELS), 0)
        self.assertEqual(getattr(remote, "_ACTIVE_INVOCATIONS", set()), set())

    def test_empty_registry_cleanup_cancels_unpublished_invocation(self):
        """W14: cleanup fences active acquisition even when the map is empty."""
        from tools import code_kernel_remote as remote

        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        env.hold_stage = "before_launch_admission"
        result = {}
        errors = []

        def invoke():
            try:
                result["value"] = _run(env, code="must-not-run", task="empty-owner")
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=invoke, name="w14-empty-map")
        worker.start()
        try:
            self.assertTrue(env.stage_entered.wait(5))
            with _REGISTRY.lock:
                self.assertEqual(_REMOTE_KERNELS, {})
                if hasattr(remote, "_ACTIVE_INVOCATIONS"):
                    self.assertEqual(len(remote._ACTIVE_INVOCATIONS), 1)
            shutdown_remote_kernels_for_owner("empty-owner")
        finally:
            env.stage_release.set()
            worker.join(5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result["value"]["status"], "error")
        self.assertIn("canceled by session cleanup", result["value"]["error"])
        self.assertEqual(env.spawn_count, 0)
        self.assertEqual(env.cell_submissions, [])
        self.assertFalse(any("python3 script.py" in command for command in env.commands))
        with _REGISTRY.lock:
            self.assertEqual(_REMOTE_KERNELS, {})
            self.assertEqual(getattr(remote, "_ACTIVE_INVOCATIONS", set()), set())

    def test_warm_rpc_dispatch_keeps_each_callers_context(self):
        """W02: the real file poller dispatches under each serialized caller context."""
        from tools.code_execution_rpc import _rpc_poll_loop

        env = FileAwareEnv()
        env.emit_rpc = True
        self._ship_mock.side_effect = env.ship
        current_call = contextvars.ContextVar("remote_test_call", default="missing")
        observed = []

        def default_dispatch(_task_id):
            return lambda _name, args: observed.append((current_call.get(), args["path"])) or "ok"

        def invoke(label):
            token = current_call.set(label)
            try:
                return _run(env, code=label, task="shared-rpc")
            finally:
                current_call.reset(token)

        with patch("tools.code_execution_tool._rpc_poll_loop", _rpc_poll_loop), \
             patch("tools.code_execution_rpc._default_dispatch", default_dispatch), \
             patch("tools.code_kernel_remote.secrets.token_urlsafe", return_value="fixed-rpc-token"):
            first = invoke("alpha")
            second = invoke("beta")

        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "success")
        self.assertEqual(observed, [("alpha", "alpha"), ("beta", "beta")])

    def _assert_warm_waiter_is_held_through(self, stage):
        """W02: one kernel owner retains namespace authority through *stage*."""
        from tools.code_execution_rpc import _rpc_poll_loop

        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        with patch("tools.code_kernel_remote.secrets.token_urlsafe", return_value="fixed-rpc-token"):
            self.assertEqual(_run(env, code="seed", task="held-rpc")["status"], "success")
        env.emit_rpc = True
        env.hold_stage = stage
        current_call = contextvars.ContextVar("remote_held_call", default="missing")
        observed = []
        results = {}
        errors = []

        def default_dispatch(_task_id):
            return lambda _name, args: observed.append((current_call.get(), args["path"])) or "ok"

        def invoke(label):
            token = current_call.set(label)
            try:
                results[label] = _run(env, code=label, task="held-rpc")
            except BaseException as exc:
                errors.append(exc)
            finally:
                current_call.reset(token)

        with patch("tools.code_execution_tool._rpc_poll_loop", _rpc_poll_loop), \
             patch("tools.code_execution_rpc._default_dispatch", default_dispatch), \
             patch("tools.code_kernel_remote.secrets.token_urlsafe", return_value="fixed-rpc-token"):
            first = threading.Thread(target=invoke, args=("alpha",))
            second = threading.Thread(target=invoke, args=("beta",))
            first.start()
            self.assertTrue(
                env.stage_entered.wait(5),
                f"first call never reached {stage}; commands={env.commands!r}; errors={errors!r}",
            )
            second.start()
            try:
                time.sleep(0.1)
                self.assertTrue(second.is_alive(), f"same-key waiter passed {stage}")
                submitted = [c for c in env.commands if c.startswith("mv ") and "/cells/cell_req_" in c]
                self.assertEqual(len(submitted), 2, "waiting call submitted before first namespace retired")
                self.assertNotIn(("beta", "beta"), observed)
            finally:
                env.stage_release.set()
                first.join(5)
                second.join(5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results["alpha"]["stdout"], "alpha")
        self.assertEqual(results["beta"]["stdout"], "beta")
        self.assertEqual(observed, [("alpha", "alpha"), ("beta", "beta")])
        submitted = [c for c in env.commands if c.startswith("mv ") and "/cells/cell_req_" in c]
        self.assertEqual(len(submitted), 3)
        first_remove = next(i for i, c in enumerate(env.commands)
                            if c.startswith("rm -f ") and "/cells/cell_res_000002" in c)
        second_submit = next(i for i, c in enumerate(env.commands)
                             if c.startswith("mv ") and "/cells/cell_req_000003" in c)
        self.assertLess(first_remove, second_submit)

    def test_warm_waiter_is_held_during_rpc_request_read(self):
        self._assert_warm_waiter_is_held_through("rpc_request_read")

    def test_warm_waiter_is_held_during_rpc_response_rename(self):
        self._assert_warm_waiter_is_held_through("rpc_response_rename")

    def test_warm_waiter_is_held_between_cell_result_read_and_removal(self):
        self._assert_warm_waiter_is_held_through("cell_result_read")

    def test_global_shutdown_retires_late_spawn_but_allows_later_call(self):
        """W05/W07: the boundary cancels old work without an owner tombstone."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        spawn_started = threading.Event()
        release_spawn = threading.Event()
        env.before_spawn = lambda: (spawn_started.set(), release_spawn.wait(5))
        result = {}
        worker = threading.Thread(
            target=lambda: result.setdefault("old", _run(env, code="old", task="global"))
        )
        worker.start()
        self.assertTrue(spawn_started.wait(5))
        shutdown_all_remote_kernels()
        release_spawn.set()
        worker.join(5)
        self.assertEqual(result["old"]["status"], "error")
        self.assertEqual(len(_REMOTE_KERNELS), 0)
        self.assertTrue(env.killed)

        env.before_spawn = None
        result["new"] = _run(env, code="new", task="global")
        self.assertEqual(result["new"]["status"], "success")

    def test_owner_shutdown_before_cold_launch_admission_preserves_other_owner(self):
        """W05: cleanup before nohup admission cancels only the old owner call."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        self.assertEqual(_run(env, code="other-seed", task="other-owner")["status"], "success")
        env.hold_stage = "before_launch_admission"
        result = {}
        errors = []

        def invoke_old_owner():
            try:
                result["old"] = _run(env, code="must-not-launch", task="old-owner")
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=invoke_old_owner)
        worker.start()
        self.assertTrue(env.stage_entered.wait(5), "cold spawn did not reach pre-launch admission")
        shutdown_remote_kernels_for_owner("old-owner")
        env.stage_release.set()
        worker.join(5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result["old"]["status"], "error")
        self.assertIn("canceled by session cleanup", result["old"]["error"])
        self.assertEqual(env.spawn_count, 1, "canceled owner admitted a nohup launch")
        self.assertFalse(any("must-not-launch" in body for body in env.files.values()))
        with _REGISTRY.lock:
            self.assertEqual({key[0] for key in _REMOTE_KERNELS}, {"other-owner"})

        env.hold_stage = None
        self.assertEqual(_run(env, code="other-after", task="other-owner")["stdout"], "other-after")
        self.assertEqual(_run(env, code="new-after", task="old-owner")["stdout"], "new-after")

    def test_owner_shutdown_retires_admitted_cold_launch_without_publication(self):
        """W05: a late PID is retired by identity and cannot publish or submit."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        self.assertEqual(_run(env, code="other-seed", task="other-owner")["status"], "success")
        launch_admitted = threading.Event()
        release_launch = threading.Event()
        env.before_spawn = lambda: (launch_admitted.set(), release_launch.wait(5))
        result = {}
        errors = []

        def invoke_old_owner():
            try:
                result["old"] = _run(env, code="must-not-submit", task="old-owner")
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=invoke_old_owner)
        worker.start()
        self.assertTrue(launch_admitted.wait(5), "cold spawn did not enter the admitted launch")
        shutdown_remote_kernels_for_owner("old-owner")
        with _REGISTRY.lock:
            self.assertEqual({key[0] for key in _REMOTE_KERNELS}, {"other-owner"})
        release_launch.set()
        worker.join(5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result["old"]["status"], "error")
        self.assertIn("canceled by session cleanup", result["old"]["error"])
        self.assertTrue(any("7002" in command for command in env.killed))
        self.assertFalse(any("must-not-submit" in body for body in env.files.values()))
        with _REGISTRY.lock:
            self.assertEqual({key[0] for key in _REMOTE_KERNELS}, {"other-owner"})

        env.before_spawn = None
        self.assertEqual(_run(env, code="other-after", task="other-owner")["stdout"], "other-after")
        self.assertEqual(_run(env, code="new-after", task="old-owner")["stdout"], "new-after")

    def _assert_owner_shutdown_cancels_cold_loser_adoption(self, held_stage):
        """W06: bind cleanup to one of the loser's two adoption boundaries."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        loser_at_launch = threading.Event()
        release_loser_launch = threading.Event()
        winner_at_submit = threading.Event()
        release_winner = threading.Event()
        held = threading.Event()
        release_held = threading.Event()

        def before_spawn():
            if threading.current_thread().name == "w06-loser":
                loser_at_launch.set()
                release_loser_launch.wait(5)

        def before_submit():
            if threading.current_thread().name == "w06-winner":
                winner_at_submit.set()
                release_winner.wait(5)

        def before_kill(command):
            if held_stage == "loser_retirement" and "7002" in command:
                held.set()
                release_held.wait(5)

        def before_liveness(command):
            if (held_stage == "winner_liveness"
                    and threading.current_thread().name == "w06-loser"
                    and "7001" in command):
                held.set()
                release_held.wait(5)

        env.before_spawn = before_spawn
        env.before_submit = before_submit
        env.before_kill = before_kill
        env.before_liveness_command = before_liveness
        results = {}
        errors = []

        def invoke(label):
            try:
                results[label] = _run(env, code=label, task="adoption-owner")
            except BaseException as exc:
                errors.append(exc)

        loser = threading.Thread(target=invoke, args=("loser",), name="w06-loser")
        winner = threading.Thread(target=invoke, args=("winner",), name="w06-winner")
        loser.start()
        self.assertTrue(loser_at_launch.wait(5), "loser did not hold its cold launch")
        winner.start()
        self.assertTrue(winner_at_submit.wait(5), "winner did not publish and reach its cell")
        with _REGISTRY.lock:
            winner_kernel = next(iter(_REMOTE_KERNELS.values()))
            self.assertEqual(winner_kernel.pid, "7001")
            self.assertEqual(winner_kernel.attached, 1)
        release_loser_launch.set()
        self.assertTrue(held.wait(5), f"loser did not reach {held_stage}")

        shutdown_remote_kernels_for_owner("adoption-owner")
        release_held.set()
        release_winner.set()
        loser.join(5)
        winner.join(5)

        self.assertFalse(loser.is_alive())
        self.assertFalse(winner.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual({result["status"] for result in results.values()}, {"error"})
        self.assertTrue(all("canceled by session cleanup" in result["error"]
                            for result in results.values()))
        self.assertEqual(env.spawn_count, 2)
        self.assertTrue(any("7001" in command for command in env.killed))
        self.assertTrue(any("7002" in command for command in env.killed))
        self.assertEqual(sum(command.startswith("mv ") for command in env.commands), 1)
        self.assertEqual(winner_kernel.attached, 0)
        with _REGISTRY.lock:
            self.assertEqual(len(_REMOTE_KERNELS), 0)

        env.before_spawn = None
        env.before_submit = None
        env.before_kill = None
        env.before_liveness_command = None
        self.assertEqual(_run(env, code="after", task="adoption-owner")["stdout"], "after")
        self.assertEqual(env.spawn_count, 3)

    def test_owner_shutdown_while_cold_loser_retires_cancels_adoption(self):
        self._assert_owner_shutdown_cancels_cold_loser_adoption("loser_retirement")

    def test_owner_shutdown_at_winner_liveness_cancels_cold_loser_adoption(self):
        self._assert_owner_shutdown_cancels_cold_loser_adoption("winner_liveness")

    def test_global_shutdown_cancels_multi_owner_work_at_acquire_retry(self):
        """W07: a global boundary fences retries and active cells for every owner."""
        from tools import code_kernel_remote as remote

        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        self.assertEqual(_run(env, code="seed-a", task="owner-a")["status"], "success")
        self.assertEqual(_run(env, code="seed-b", task="owner-b")["status"], "success")
        first_probe = threading.Event()
        release_first_probe = threading.Event()
        retry_probe = threading.Event()
        release_retry_probe = threading.Event()
        reset_submitted = threading.Event()
        release_reset = threading.Event()
        owner_b_submitted = threading.Event()
        release_owner_b = threading.Event()

        def hold_owner_a_probes(command):
            if threading.current_thread().name != "w07-retry":
                return
            if "7001" in command:
                first_probe.set()
                release_first_probe.wait(5)
            elif "7003" in command:
                retry_probe.set()
                release_retry_probe.wait(5)

        def hold_active_cells():
            name = threading.current_thread().name
            if name == "w07-reset":
                reset_submitted.set()
                release_reset.wait(5)
            elif name == "w07-owner-b":
                owner_b_submitted.set()
                release_owner_b.wait(5)

        env.before_liveness_command = hold_owner_a_probes
        env.before_submit = hold_active_cells
        results = {}
        errors = []

        def invoke(label, task, *, reset=False):
            try:
                results[label] = _run(env, code=label, task=task, reset=reset)
            except BaseException as exc:
                errors.append(exc)

        owner_b = threading.Thread(
            target=invoke, args=("active-b", "owner-b"), name="w07-owner-b"
        )
        retry = threading.Thread(
            target=invoke, args=("retry-a", "owner-a"), name="w07-retry"
        )
        reset = threading.Thread(
            target=invoke, args=("reset-a", "owner-a"), kwargs={"reset": True}, name="w07-reset"
        )
        owner_b.start()
        self.assertTrue(owner_b_submitted.wait(5), "second owner did not enter its active cell")
        retry.start()
        self.assertTrue(first_probe.wait(5), "existing owner did not hold its successful probe")
        reset.start()
        self.assertTrue(reset_submitted.wait(5), "public reset did not publish replacement K3")
        release_first_probe.set()
        self.assertTrue(retry_probe.wait(5), "old acquisition did not retry against replacement K3")
        with _REGISTRY.lock:
            self.assertEqual({key[0] for key in _REMOTE_KERNELS}, {"owner-a", "owner-b"})
            if hasattr(remote, "_ACTIVE_INVOCATIONS"):
                self.assertEqual(len(remote._ACTIVE_INVOCATIONS), 3)

        shutdown_all_remote_kernels()
        release_retry_probe.set()
        release_reset.set()
        release_owner_b.set()
        for worker in (owner_b, retry, reset):
            worker.join(5)

        self.assertTrue(all(not worker.is_alive() for worker in (owner_b, retry, reset)))
        self.assertEqual(errors, [])
        self.assertEqual(set(results), {"active-b", "retry-a", "reset-a"})
        self.assertTrue(all(result["status"] == "error" for result in results.values()))
        self.assertTrue(all("canceled by session cleanup" in result["error"]
                            for result in results.values()))
        self.assertEqual(env.spawn_count, 3, "a pre-boundary retry admitted another spawn")
        self.assertEqual(sum(command.startswith("mv ") for command in env.commands), 4)
        with _REGISTRY.lock:
            self.assertEqual(len(_REMOTE_KERNELS), 0)
            self.assertEqual(getattr(remote, "_ACTIVE_INVOCATIONS", set()), set())

        env.before_liveness_command = None
        env.before_submit = None
        self.assertEqual(_run(env, code="after-a", task="owner-a")["stdout"], "after-a")
        self.assertEqual(_run(env, code="after-b", task="owner-b")["stdout"], "after-b")
        self.assertEqual(env.spawn_count, 5)

    def test_failed_stale_rpc_cleanup_retires_before_submission(self):
        """W03: an unsafe old RPC namespace cannot arm a new authority window."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        env.cleanup_returncode = 1

        result = _run(env, code="must-not-submit", task="cleanup")

        self.assertEqual(result["status"], "error")
        self.assertIn("cleanup failed", result["error"])
        self.assertFalse(any(c.startswith("mv ") for c in env.commands))
        self.assertEqual(len(_REMOTE_KERNELS), 0)

    def test_post_submission_transport_uncertainty_is_not_replayed(self):
        """W11: an applied rename followed by failure is terminal, never fallback."""
        from tools.code_execution_tool import _execute_remote

        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        env.submit_error_after_apply = RuntimeError("rename reply lost")
        with patch("tools.code_execution_tool._load_config",
                   return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env",
                   return_value=(env, "ssh")):
            result = json.loads(_execute_remote("once", "uncertain", ["read_file"]))

        self.assertEqual(result["status"], "error")
        self.assertIn("was not replayed", result["error"])
        self.assertFalse(any("python3 script.py" in c for c in env.commands))
        self.assertEqual(env.spawn_count, 1)
        self.assertEqual(len(_REMOTE_KERNELS), 0)

    def test_known_result_survives_cleanup_failure_and_kernel_retires(self):
        """W10/W11: decoded output is authoritative even if removal fails."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        env.result_cleanup_error = RuntimeError("cleanup transport lost")

        result = _run(env, code="known", task="known")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stdout"], "known")
        self.assertTrue(result["kernel"]["ended"])
        self.assertTrue(result["kernel"]["state_lost"])
        self.assertEqual(len(_REMOTE_KERNELS), 0)

    def test_malformed_result_is_protocol_failure_and_not_reused(self):
        """W08/W11: result shape failure retires K without a second execution."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        env.result_payload = ["not", "an", "object"]

        result = _run(env, code="malformed", task="protocol")

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["kernel"]["state_lost"])
        self.assertEqual(env.spawn_count, 1)
        self.assertEqual(len(_REMOTE_KERNELS), 0)

    def test_error_and_exit_results_keep_existing_status_contract(self):
        """W08: serialized ownership preserves ordinary error and exit results."""
        cases = (
            (_cell(status="error", traceback="ValueError: bad"), "error", False),
            (_cell(status="exit", stdout="bye"), "success", True),
        )
        for index, (payload, status, ended) in enumerate(cases):
            with self.subTest(status=payload["status"]):
                shutdown_all_remote_kernels()
                env = FileAwareEnv()
                self._ship_mock.side_effect = env.ship
                env.result_payload = payload
                result = _run(env, code="status", task=f"status-{index}")
                self.assertEqual(result["status"], status)
                self.assertEqual(result["kernel"].get("ended", False), ended)
                if status == "error":
                    self.assertEqual(result["error"], "ValueError: bad")
                else:
                    self.assertEqual(result["stdout"], "bye")

    def _assert_waiter_revalidates_after_terminal_outcome(self, outcome):
        """W08: one named terminal shape retires or retains K before its waiter runs."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        first_code = f"first-{outcome}"

        def payload(request):
            if request["code"] != first_code:
                return _cell(stdout=request["code"], execution_count=int(request["id"]))
            if outcome == "protocol":
                return ["not", "an", "object"]
            if outcome == "error":
                return _cell(status="error", traceback="ValueError: bad")
            if outcome == "exit":
                return _cell(status="exit", stdout="bye")
            return _cell(stdout="ignored-by-zero-timeout")

        env.result_payload = payload
        env.hold_stage = "before_cell_submit" if outcome == "timeout" else "cell_result_read"
        first_held = threading.Event()
        release_first = threading.Event()
        original_before_submit = env.before_submit

        if outcome == "timeout":
            def hold_timeout_submit():
                if not first_held.is_set():
                    first_held.set()
                    release_first.wait(5)
            env.before_submit = hold_timeout_submit
        else:
            env.hold_stage = "cell_result_read"

        results = {}
        errors = []

        def invoke(label, timeout):
            try:
                results[label] = _run(env, code=label, task="w08-owner", timeout=timeout)
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(
            target=invoke, args=(first_code, 0 if outcome == "timeout" else 10)
        )
        second = threading.Thread(target=invoke, args=("waiting-second", 10))
        first.start()
        if outcome == "timeout":
            self.assertTrue(first_held.wait(5), "timeout call did not hold before submission")
        else:
            self.assertTrue(env.stage_entered.wait(5), f"{outcome} call did not hold its result read")
        second.start()
        deadline = time.monotonic() + 5
        while True:
            with _REGISTRY.lock:
                attached = sum(kernel.attached for kernel in _REMOTE_KERNELS.values())
            if attached == 2:
                break
            self.assertLess(time.monotonic(), deadline, "same-key waiter did not reserve K")
            time.sleep(0.01)
        if outcome == "timeout":
            release_first.set()
        else:
            env.stage_release.set()
        first.join(5)
        second.join(5)
        env.before_submit = original_before_submit

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        first_result = results[first_code]
        second_result = results["waiting-second"]
        if outcome == "timeout":
            self.assertEqual(first_result["status"], "timeout")
            self.assertTrue(first_result["kernel"]["state_lost"])
        elif outcome == "protocol":
            self.assertEqual(first_result["status"], "error")
            self.assertTrue(first_result["kernel"]["state_lost"])
        elif outcome == "error":
            self.assertEqual(first_result["status"], "error")
            self.assertEqual(first_result["error"], "ValueError: bad")
            self.assertFalse(first_result["kernel"].get("state_lost", False))
        else:
            self.assertEqual(first_result["status"], "success")
            self.assertEqual(first_result["stdout"], "bye")
            self.assertTrue(first_result["kernel"]["ended"])
        self.assertEqual(second_result["status"], "success")
        self.assertEqual(second_result["stdout"], "waiting-second")
        submitted_codes = [request["code"] for request in env.cell_submissions]
        self.assertEqual(submitted_codes.count(first_code), 1)
        self.assertEqual(submitted_codes.count("waiting-second"), 1)
        self.assertEqual(env.spawn_count, 1 if outcome == "error" else 2)

    def test_timeout_retires_before_waiting_same_key_call(self):
        self._assert_waiter_revalidates_after_terminal_outcome("timeout")

    def test_protocol_failure_retires_before_waiting_same_key_call(self):
        self._assert_waiter_revalidates_after_terminal_outcome("protocol")

    def test_ordinary_error_retains_kernel_for_waiting_same_key_call(self):
        self._assert_waiter_revalidates_after_terminal_outcome("error")

    def test_exit_retires_before_waiting_same_key_call(self):
        self._assert_waiter_revalidates_after_terminal_outcome("exit")

    def _assert_replacement_survives_old_teardown(self, replacement):
        """W09: K1 teardown cannot remove K2 while the old caller adopts it."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        self.assertEqual(_run(env, code="seed", task="replacement-owner")["status"], "success")
        old_teardown = threading.Event()
        release_old_teardown = threading.Event()
        winner_submitted = threading.Event()
        release_winner = threading.Event()
        dead_answered = [False]

        def hold_old_kill(command):
            if "7001" in command and not old_teardown.is_set():
                old_teardown.set()
                release_old_teardown.wait(5)

        def hold_winner_submit():
            if threading.current_thread().name == "w09-winner":
                winner_submitted.set()
                release_winner.wait(5)

        def liveness(command):
            if (replacement == "dead" and threading.current_thread().name == "w09-old"
                    and "7001" in command and not dead_answered[0]):
                dead_answered[0] = True
                return {"output": "", "returncode": 1}
            return {"output": "ALIVE\n", "returncode": 0}

        env.before_kill = hold_old_kill
        env.before_submit = hold_winner_submit
        env.liveness_response = liveness
        results = {}
        errors = []

        def invoke(label, *, reset=False):
            try:
                results[label] = _run(
                    env, code=label, task="replacement-owner", reset=reset
                )
            except BaseException as exc:
                errors.append(exc)

        old = threading.Thread(
            target=invoke, args=("old-caller",), kwargs={"reset": replacement == "reset"}, name="w09-old"
        )
        winner = threading.Thread(target=invoke, args=("winner",), name="w09-winner")
        old.start()
        self.assertTrue(old_teardown.wait(5), "K1 teardown did not hold")
        winner.start()
        self.assertTrue(winner_submitted.wait(5), "concurrent caller did not publish K2")
        with _REGISTRY.lock:
            k2 = next(iter(_REMOTE_KERNELS.values()))
            self.assertEqual(k2.pid, "7002")
            self.assertEqual(k2.attached, 1)
        release_old_teardown.set()
        deadline = time.monotonic() + 5
        while True:
            with _REGISTRY.lock:
                attached = k2.attached
            if attached == 2:
                break
            self.assertLess(time.monotonic(), deadline, "old caller did not wait on K2")
            time.sleep(0.01)
        release_winner.set()
        old.join(5)
        winner.join(5)

        self.assertFalse(old.is_alive())
        self.assertFalse(winner.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results["winner"]["stdout"], "winner")
        self.assertEqual(results["old-caller"]["stdout"], "old-caller")
        self.assertTrue(results["old-caller"]["kernel"]["reused"])
        flag = "state_reset" if replacement == "reset" else "state_lost"
        self.assertTrue(results["old-caller"]["kernel"][flag])
        self.assertEqual(env.spawn_count, 3)
        self.assertTrue(any("7001" in command for command in env.killed))
        self.assertTrue(any("7003" in command for command in env.killed))
        self.assertFalse(any("7002" in command for command in env.killed))
        with _REGISTRY.lock:
            self.assertIs(next(iter(_REMOTE_KERNELS.values())), k2)
            self.assertEqual(k2.attached, 0)

    def test_public_reset_old_teardown_cannot_remove_concurrent_winner(self):
        self._assert_replacement_survives_old_teardown("reset")

    def test_negative_liveness_old_teardown_cannot_remove_concurrent_winner(self):
        self._assert_replacement_survives_old_teardown("dead")

    def test_initially_absent_reset_is_consumed_once_across_cold_winner(self):
        """W09: reset=True on an empty key survives loser adoption exactly once."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        spawn_barrier = threading.Barrier(2)
        cold_spawns = [0]

        def hold_initial_cold_spawns():
            cold_spawns[0] += 1
            if cold_spawns[0] <= 2:
                spawn_barrier.wait(5)

        env.before_spawn = hold_initial_cold_spawns
        first_submitted = threading.Event()
        release_first = threading.Event()

        def hold_first_submit():
            if not first_submitted.is_set():
                first_submitted.set()
                release_first.wait(5)

        env.before_submit = hold_first_submit
        results = []
        errors = []

        def invoke(label, reset):
            try:
                results.append(_run(env, code=label, task="absent-reset", reset=reset))
            except BaseException as exc:
                errors.append(exc)

        workers = [
            threading.Thread(target=invoke, args=("reset-caller", True)),
            threading.Thread(target=invoke, args=("plain-caller", False)),
        ]
        for worker in workers:
            worker.start()
        self.assertTrue(first_submitted.wait(5))
        release_first.set()
        for worker in workers:
            worker.join(5)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(errors, [])
        self.assertEqual({result["stdout"] for result in results}, {"reset-caller", "plain-caller"})
        reset_result = next(result for result in results if result["stdout"] == "reset-caller")
        self.assertFalse(reset_result["kernel"].get("state_reset", False))
        self.assertEqual(env.spawn_count, 2)
        self.assertEqual(len(env.killed), 1)
        self.assertEqual(len(_REMOTE_KERNELS), 1)

    def _assert_late_loser_after_winner_second_cell(self, cleanup_before_release):
        """W10: a late candidate never owns or destroys the active winner."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        loser_at_launch = threading.Event()
        release_loser = threading.Event()
        second_submitted = threading.Event()
        release_second = threading.Event()

        def before_spawn():
            if threading.current_thread().name == "w10-loser":
                loser_at_launch.set()
                release_loser.wait(5)

        def before_submit():
            if threading.current_thread().name == "w10-second":
                second_submitted.set()
                release_second.wait(5)

        env.before_spawn = before_spawn
        env.before_submit = before_submit
        results = {}
        errors = []

        def invoke(label):
            try:
                results[label] = _run(env, code=label, task="late-loser-owner")
            except BaseException as exc:
                errors.append(exc)

        loser = threading.Thread(target=invoke, args=("late-loser",), name="w10-loser")
        loser.start()
        self.assertTrue(loser_at_launch.wait(5), "late candidate did not hold its launch")
        self.assertEqual(_run(env, code="winner-first", task="late-loser-owner")["stdout"], "winner-first")
        second = threading.Thread(target=invoke, args=("winner-second",), name="w10-second")
        second.start()
        self.assertTrue(second_submitted.wait(5), "winner did not begin its second cell")
        with _REGISTRY.lock:
            winner = next(iter(_REMOTE_KERNELS.values()))
            self.assertEqual((winner.pid, winner.execution_count), ("7001", 1))

        if cleanup_before_release:
            shutdown_remote_kernels_for_owner("late-loser-owner")
        release_loser.set()
        if not cleanup_before_release:
            deadline = time.monotonic() + 5
            while not any("7002" in command for command in env.killed):
                self.assertLess(time.monotonic(), deadline, "late loser was not retired by identity")
                time.sleep(0.01)
        release_second.set()
        loser.join(5)
        second.join(5)

        self.assertFalse(loser.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(any("7002" in command for command in env.killed))
        if cleanup_before_release:
            self.assertEqual({result["status"] for result in results.values()}, {"error"})
            self.assertTrue(all("canceled by session cleanup" in result["error"]
                                for result in results.values()))
            self.assertEqual(sum(request["code"] == "late-loser"
                                 for request in env.cell_submissions), 0)
            with _REGISTRY.lock:
                self.assertEqual(len(_REMOTE_KERNELS), 0)
        else:
            self.assertEqual(results["winner-second"]["stdout"], "winner-second")
            self.assertEqual(results["late-loser"]["stdout"], "late-loser")
            self.assertEqual([request["code"] for request in env.cell_submissions],
                             ["winner-first", "winner-second", "late-loser"])
            self.assertFalse(any("7001" in command for command in env.killed))
            with _REGISTRY.lock:
                self.assertIs(next(iter(_REMOTE_KERNELS.values())), winner)
                self.assertEqual(winner.attached, 0)
        self.assertEqual(env.spawn_count, 2)

    def test_late_cold_loser_adopts_after_winner_begins_second_cell(self):
        self._assert_late_loser_after_winner_second_cell(False)

    def test_owner_cleanup_before_late_loser_release_prevents_adoption(self):
        self._assert_late_loser_after_winner_second_cell(True)

    def test_live_poller_forces_permanent_retirement_before_handoff(self):
        """W10: join timeout cannot expose K to the next caller."""
        env = FileAwareEnv()
        self._ship_mock.side_effect = env.ship
        poller_started = threading.Event()
        release_poller = threading.Event()

        def stuck_poller(*_args, **_kwargs):
            poller_started.set()
            release_poller.wait(10)

        self._poll_mock.side_effect = stuck_poller
        result = _run(env, code="done", task="stuck-poller")
        self.assertTrue(poller_started.is_set())
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["kernel"]["ended"])
        self.assertEqual(len(_REMOTE_KERNELS), 0)
        release_poller.set()

    def test_cleanup_between_fallback_staging_and_admission_cancels_script(self):
        """W12: kernel-unavailable fallback shares the invocation fence."""
        from tools.code_execution_tool import _execute_remote

        env = FileAwareEnv()
        script_staged = threading.Event()
        release_staging = threading.Event()

        def hold_script(_env, path, content):
            env.ship(_env, path, content)
            if path.endswith("/script.py"):
                script_staged.set()
                release_staging.wait(5)

        self._ship_mock.side_effect = hold_script
        result = {}
        with patch("tools.code_execution_tool._load_config",
                   return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env",
                   return_value=(env, "ssh")), \
             patch("tools.code_kernel_remote.execute_in_remote_kernel", return_value=None):
            worker = threading.Thread(
                target=lambda: result.setdefault(
                    "value", json.loads(_execute_remote("fallback", "fallback-owner", ["read_file"]))
                )
            )
            worker.start()
            self.assertTrue(script_staged.wait(5))
            shutdown_remote_kernels_for_owner("fallback-owner")
            release_staging.set()
            worker.join(5)

        self.assertEqual(result["value"]["status"], "error")
        self.assertIn("canceled by session cleanup", result["value"]["error"])
        self.assertFalse(any("python3 script.py" in c for c in env.commands))


class TestDispatchIntegration(unittest.TestCase):
    """_execute_remote prefers the kernel and falls open to per-call."""

    def test_execute_remote_uses_kernel_result(self):
        from tools.code_execution_tool import _execute_remote

        fake = {
            "status": "success", "stdout": "kernel says hi\n", "stderr": "",
            "traceback": "", "tool_calls_made": 0,
            "kernel": {"reused": True, "remote": True, "execution_count": 3},
        }
        env = ScriptedEnv([
            ("command -v python3", lambda c: {"output": "OK\n", "returncode": 0}),
        ])
        with patch("tools.code_execution_tool._load_config",
                   return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env",
                   return_value=(env, "ssh")), \
             patch("tools.code_kernel_remote.execute_in_remote_kernel",
                   return_value=fake):
            result = json.loads(_execute_remote("print()", "t", ["read_file"]))
        self.assertEqual(result["status"], "success")
        self.assertIn("kernel says hi", result["output"])
        self.assertEqual(result["kernel"]["execution_count"], 3)

    def test_execute_remote_falls_open_to_per_call(self):
        from tools.code_execution_tool import _execute_remote
        from unittest.mock import MagicMock

        env = ScriptedEnv([
            ("command -v python3", lambda c: {"output": "OK\n", "returncode": 0}),
            ("python3 script.py", lambda c: {"output": "per-call ran\n",
                                             "returncode": 0}),
        ])
        with patch("tools.code_execution_tool._load_config",
                   return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env",
                   return_value=(env, "ssh")), \
             patch("tools.code_kernel_remote.execute_in_remote_kernel",
                   return_value=None), \
             patch("tools.code_execution_tool._ship_file_to_remote"), \
             patch("tools.code_execution_tool.threading.Thread",
                   return_value=MagicMock()):
            result = json.loads(_execute_remote("print()", "t", ["read_file"]))
        self.assertEqual(result["status"], "success")
        self.assertIn("per-call ran", result["output"])

    def test_spawn_unavailable_fallback_uses_unique_real_rpc_namespaces(self):
        """W12: safe fallback keeps one real RPC/result path per invocation."""
        from tools.code_execution_tool import _execute_remote
        import tools.terminal_tool as terminal_tool

        class FallbackEnv(FileAwareEnv):
            def __init__(self):
                super().__init__()
                self.fallback_dirs = []
                self.terminal_commands = []

            def execute(self, command, cwd=None, timeout=None, **kwargs):
                if "nohup" in command:
                    with self._lock:
                        self.commands.append(command)
                        self.spawn_count += 1
                    return {"output": "runner started without a usable pid\n", "returncode": 0}
                if "python3 script.py" in command:
                    with self._lock:
                        self.commands.append(command)
                    assignments = {
                        part.split("=", 1)[0]: part.split("=", 1)[1]
                        for part in shlex.split(command)
                        if part.startswith(("HERMES_RPC_DIR=", "HERMES_RPC_TOKEN="))
                    }
                    rpc_dir = assignments["HERMES_RPC_DIR"]
                    rpc_token = assignments["HERMES_RPC_TOKEN"]
                    self.fallback_dirs.append(rpc_dir.rsplit("/rpc", 1)[0])
                    request_path = f"{rpc_dir}/req_000001"
                    with self._lock:
                        self.files[request_path] = json.dumps({
                            "token": rpc_token,
                            "seq": 1,
                            "tool": "terminal",
                            "args": {"command": "printf w12-real-dispatch"},
                        })
                    if not self.rpc_done.wait(5):
                        raise AssertionError("real per-call RPC did not publish its response")
                    response_path = f"{rpc_dir}/res_000001"
                    with self._lock:
                        response = self.files[response_path]
                    self.rpc_done.clear()
                    return {"output": response, "returncode": 0}
                if command == "printf w12-real-dispatch":
                    with self._lock:
                        self.commands.append(command)
                        self.terminal_commands.append(command)
                    return {"output": "w12-dispatched", "returncode": 0}
                return super().execute(command, cwd=cwd, timeout=timeout, **kwargs)

        env = FallbackEnv()
        terminal_config = {
            "env_type": "ssh", "cwd": "/", "host_cwd": None, "timeout": 30,
            "lifetime_seconds": 300, "docker_mount_cwd_to_workspace": False,
            "docker_volumes": [], "docker_shared_container_key": "",
        }
        with terminal_tool._env_lock:
            previous_envs = dict(terminal_tool._active_environments)
            previous_activity = dict(terminal_tool._last_activity)
            terminal_tool._active_environments.clear()
            terminal_tool._last_activity.clear()
            terminal_tool._active_environments["default"] = env
        try:
            with patch("tools.code_execution_tool._load_config",
                       return_value={"timeout": 30, "max_tool_calls": 5}), \
                 patch("tools.code_execution_tool._get_or_create_env",
                       return_value=(env, "ssh")), \
                 patch("tools.terminal_tool._get_env_config", return_value=terminal_config):
                results = [
                    json.loads(_execute_remote("fallback", "w12-owner", ["terminal"]))
                    for _ in range(2)
                ]
        finally:
            with terminal_tool._env_lock:
                terminal_tool._active_environments.clear()
                terminal_tool._active_environments.update(previous_envs)
                terminal_tool._last_activity.clear()
                terminal_tool._last_activity.update(previous_activity)

        self.assertEqual([result["status"] for result in results], ["success", "success"])
        self.assertTrue(all("w12-dispatched" in result["output"] for result in results))
        self.assertEqual([result["tool_calls_made"] for result in results], [1, 1])
        self.assertEqual(env.terminal_commands, ["printf w12-real-dispatch"] * 2)
        self.assertEqual(len(env.fallback_dirs), 2)
        self.assertEqual(len(set(env.fallback_dirs)), 2)
        self.assertTrue(all("/hermes_exec_" in path for path in env.fallback_dirs))
        self.assertEqual(sum("python3 script.py" in command for command in env.commands), 2)
        self.assertEqual(len(_REMOTE_KERNELS), 0)

    def test_partial_kernel_initialization_falls_back_without_publication(self):
        """W14: four pre-submission failures leave no partial kernel state."""
        from tools.code_execution_tool import _execute_remote
        from tools import code_kernel_remote as remote

        class PartialInitEnv(FileAwareEnv):
            def __init__(self, failure_stage):
                super().__init__()
                self.failure_stage = failure_stage

            def execute(self, command, cwd=None, timeout=None, **kwargs):
                if (self.failure_stage == "mkdir" and command.startswith("mkdir -p ")
                        and "/hermes_rkernel_" in command):
                    with self._lock:
                        self.commands.append(command)
                    raise RuntimeError("synthetic kernel mkdir failure")
                if (self.failure_stage == "runner-shipping" and command.startswith("echo '")
                        and "/hermes_rkernel_" in command and "/kernel_runner.py" in command):
                    with self._lock:
                        self.commands.append(command)
                    raise RuntimeError("synthetic runner shipping failure")
                if self.failure_stage == "missing-pid" and "nohup" in command:
                    with self._lock:
                        self.commands.append(command)
                        self.spawn_count += 1
                    return {"output": "launch response omitted pid\n", "returncode": 0}
                if self.failure_stage == "liveness" and "kill -0" in command:
                    with self._lock:
                        self.commands.append(command)
                    return {"output": "", "returncode": 1}
                if "python3 script.py" in command:
                    with self._lock:
                        self.commands.append(command)
                    return {"output": f"fallback-after-{self.failure_stage}\n", "returncode": 0}
                return super().execute(command, cwd=cwd, timeout=timeout, **kwargs)

        for stage in ("mkdir", "runner-shipping", "missing-pid", "liveness"):
            with self.subTest(stage=stage):
                shutdown_all_remote_kernels()
                env = PartialInitEnv(stage)
                with patch("tools.code_execution_tool._load_config",
                           return_value={"timeout": 30, "max_tool_calls": 5}), \
                     patch("tools.code_execution_tool._get_or_create_env",
                           return_value=(env, "ssh")):
                    result = json.loads(_execute_remote("fallback", f"w14-{stage}", ["read_file"]))

                self.assertEqual(result["status"], "success", result)
                self.assertIn(f"fallback-after-{stage}", result["output"])
                self.assertEqual(result["tool_calls_made"], 0)
                self.assertEqual(sum("python3 script.py" in command
                                     for command in env.commands), 1)
                self.assertEqual(env.cell_submissions, [])
                self.assertTrue(any(command.startswith("rm -rf ")
                                    and "/hermes_rkernel_" in command
                                    for command in env.commands))
                with _REGISTRY.lock:
                    self.assertEqual(_REMOTE_KERNELS, {})
                    self.assertEqual(getattr(remote, "_ACTIVE_INVOCATIONS", set()), set())


if __name__ == "__main__":
    unittest.main()
