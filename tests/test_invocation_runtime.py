import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "_invocation_runtime.py"
)


def _load_runtime():
    if not RUNTIME_PATH.is_file():
        raise AssertionError("V1.4 invocation runtime is not implemented")
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_invocation_runtime_test", RUNTIME_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class InvocationProbeTests(unittest.TestCase):
    def test_worker_launch_source_drift_is_rejected_before_exec(self) -> None:
        runtime = _load_runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            marker = root / "worker-ran"
            worker.write_text("raise SystemExit(0)\n", encoding="utf-8")
            launch = runtime._build_worker_launch([sys.executable, str(worker)])
            worker.write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "worker launch identity changed"):
                runtime._verify_worker_launch(launch)
            request_digest = "a" * 64
            invocation_id = "inv-" + runtime._invocation_digest(
                request_digest, launch["digest"]
            )[:32]
            invocation_dir = root / "invocations" / invocation_id
            invocation_dir.mkdir(parents=True)
            runtime._create_json(
                invocation_dir / "request.json",
                {"request_digest": request_digest},
            )
            runtime._create_json(invocation_dir / "worker-launch.json", launch)
            gate_read, gate_write = os.pipe()
            environment = os.environ.copy()
            environment.update(
                {
                    runtime._WORKER_GATE_FD: str(gate_read),
                    runtime._INVOCATION_DIR: str(invocation_dir),
                }
            )
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(RUNTIME_PATH),
                        "_worker_gate",
                        "--invocation-dir",
                        str(invocation_dir),
                        "--artifact-root",
                        str(root),
                        "--worker-launch-file",
                        str(invocation_dir / "worker-launch.json"),
                    ],
                    env=environment,
                    pass_fds=(gate_read,),
                )
                os.close(gate_read)
                gate_read = None
                os.write(gate_write, b"1")
                self.assertEqual(process.wait(timeout=2), 1)
            finally:
                if gate_read is not None:
                    os.close(gate_read)
                os.close(gate_write)
            self.assertFalse(marker.exists(), "drifted worker source was executed")

    def test_symlinked_invocation_record_fails_closed(self) -> None:
        runtime = _load_runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invocation_id = "inv-" + "a" * 32
            invocation_dir = root / "invocations" / invocation_id
            invocation_dir.mkdir(parents=True)
            target = root / "request-target.json"
            target.write_text("{}\n", encoding="utf-8")
            os.symlink(target, invocation_dir / "request.json")

            with self.assertRaisesRegex(ValueError, "symlink|unsafe"):
                runtime.status(root, invocation_id)

    def test_probe_guardian_wait_is_bounded_and_closes_owner_pipe(self) -> None:
        runtime = _load_runtime()

        class Guardian:
            def __init__(self) -> None:
                self.returncode = None
                self.wait_timeouts = []

            def wait(self, timeout=None):
                self.wait_timeouts.append(timeout)
                if len(self.wait_timeouts) == 1:
                    raise subprocess.TimeoutExpired("guardian", timeout)
                self.returncode = 0
                return 0

        owner_read, owner_write = os.pipe()
        guardian = Guardian()
        try:
            timed_out = runtime._wait_probe_guardian(
                guardian,
                owner_write,
                operation_timeout_seconds=0.01,
                cleanup_timeout_seconds=0.02,
            )
            owner_write = None
            self.assertTrue(timed_out)
            self.assertEqual(guardian.wait_timeouts, [0.01, 0.02])
            self.assertEqual(os.read(owner_read, 1), b"")
        finally:
            os.close(owner_read)
            if owner_write is not None:
                os.close(owner_write)

    def test_command_timeout_terminates_the_entire_process_group(self) -> None:
        runtime = _load_runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_pid = root / "parent.pid"
            child_pid = root / "child.pid"
            program = root / "spawn_tree.py"
            program.write_text(
                "\n".join(
                    [
                        "import json",
                        "import os",
                        "import subprocess",
                        "import sys",
                        "import time",
                        "from pathlib import Path",
                        "root = Path(sys.argv[1])",
                        "(root / 'parent.pid').write_text(str(os.getpid()))",
                        "child = subprocess.Popen([",
                        "    sys.executable,",
                        "    '-c',",
                        "    'import time; time.sleep(60)',",
                        "])",
                        "(root / 'child.pid').write_text(str(child.pid))",
                        "time.sleep(60)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = runtime.probe(
                {
                    "argv": [sys.executable, str(program), str(root)],
                    "cwd": str(root),
                    "env": {},
                    "output_limit_bytes": 64 * 1024,
                    "required_gpu_uuids": [],
                },
                {
                    "operation_timeout_seconds": 1.0,
                    "command_timeout_seconds": 0.2,
                    "resource_wait_timeout_seconds": 0.2,
                    "cleanup_timeout_seconds": 2.0,
                },
                {"host_id": "local-test", "gpu_uuids": []},
            )

            self.assertEqual(result["status"], "timed_out")
            self.assertEqual(result["stop_reason"], "command_timeout")
            self.assertEqual(result["cleanup_status"], "confirmed")
            self.assertTrue(parent_pid.is_file())
            self.assertTrue(child_pid.is_file())

            pids = [
                int(parent_pid.read_text(encoding="utf-8")),
                int(child_pid.read_text(encoding="utf-8")),
            ]
            deadline = time.monotonic() + 2.0
            while any(_process_exists(pid) for pid in pids) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(
                [pid for pid in pids if _process_exists(pid)],
                [],
                "probe left a process from the timed-out process group alive",
            )

    def test_same_gpu_is_serialized_by_the_installation_lock(self) -> None:
        _load_runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            home.mkdir()
            first_cwd = root / "first"
            second_cwd = root / "second"
            first_cwd.mkdir()
            second_cwd.mkdir()
            events = root / "events.jsonl"
            hold = root / "hold.py"
            hold.write_text(
                "\n".join(
                    [
                        "import json",
                        "import os",
                        "import sys",
                        "import time",
                        "event_path, label = sys.argv[1:]",
                        "def emit(kind):",
                        "    row = json.dumps({",
                        "        'label': label,",
                        "        'kind': kind,",
                        "        'time_ns': time.time_ns(),",
                        "    }, sort_keys=True) + '\\n'",
                        "    descriptor = os.open(",
                        "        event_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600",
                        "    )",
                        "    try:",
                        "        os.write(descriptor, row.encode('utf-8'))",
                        "    finally:",
                        "        os.close(descriptor)",
                        "emit('start')",
                        "time.sleep(0.25)",
                        "emit('end')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            invoke = root / "invoke.py"
            invoke.write_text(
                "\n".join(
                    [
                        "import importlib.util",
                        "import json",
                        "import sys",
                        "from pathlib import Path",
                        "runtime_path, hold, events, label, cwd, out = sys.argv[1:]",
                        "spec = importlib.util.spec_from_file_location(",
                        "    'cuda_optimizer_runtime_subprocess', runtime_path",
                        ")",
                        "runtime = importlib.util.module_from_spec(spec)",
                        "spec.loader.exec_module(runtime)",
                        "result = runtime.probe(",
                        "    {",
                        "        'argv': [sys.executable, hold, events, label],",
                        "        'cwd': cwd,",
                        "        'env': {},",
                        "        'output_limit_bytes': 4096,",
                        "        'required_gpu_uuids': ['GPU-shared'],",
                        "    },",
                        "    {",
                        "        'operation_timeout_seconds': 3.0,",
                        "        'command_timeout_seconds': 1.0,",
                        "        'resource_wait_timeout_seconds': 2.0,",
                        "        'cleanup_timeout_seconds': 1.0,",
                        "    },",
                        "    {'host_id': 'host-a', 'gpu_uuids': ['GPU-shared']},",
                        ")",
                        "Path(out).write_text(json.dumps(result), encoding='utf-8')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            environment = os.environ.copy()
            environment["HOME"] = str(home)
            processes = []
            for label, cwd in (("first", first_cwd), ("second", second_cwd)):
                output = root / f"{label}.json"
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(invoke),
                            str(RUNTIME_PATH),
                            str(hold),
                            str(events),
                            label,
                            str(cwd),
                            str(output),
                        ],
                        env=environment,
                    )
                )
            returncodes = [process.wait(timeout=5) for process in processes]
            self.assertEqual(returncodes, [0, 0])

            for label in ("first", "second"):
                result = json.loads((root / f"{label}.json").read_text())
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["cleanup_status"], "confirmed")

            rows = [
                json.loads(line)
                for line in events.read_text(encoding="utf-8").splitlines()
            ]
            intervals = {}
            for row in rows:
                intervals.setdefault(row["label"], {})[row["kind"]] = row["time_ns"]
            self.assertEqual(set(intervals), {"first", "second"})
            ordered = sorted(intervals.values(), key=lambda item: item["start"])
            self.assertGreaterEqual(
                ordered[1]["start"],
                ordered[0]["end"],
                "commands that claimed the same GPU overlapped",
            )

    def test_cpu_only_commands_do_not_acquire_declared_gpu_locks(self) -> None:
        runtime = _load_runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = root / "hold.py"
            program.write_text(
                "import json, sys, time\n"
                "from pathlib import Path\n"
                "started = time.monotonic_ns()\n"
                "time.sleep(0.25)\n"
                "Path(sys.argv[1]).write_text(json.dumps("
                "{'start': started, 'end': time.monotonic_ns()}), encoding='utf-8')\n",
                encoding="utf-8",
            )
            limits = {
                "operation_timeout_seconds": 2.0,
                "command_timeout_seconds": 1.0,
                "resource_wait_timeout_seconds": 1.0,
                "cleanup_timeout_seconds": 1.0,
            }
            resources = {"host_id": "host-a", "gpu_uuids": ["GPU-shared"]}
            outputs = [root / "first.json", root / "second.json"]
            processes = []
            for output in outputs:
                command = {
                    "argv": [sys.executable, str(program), str(output)],
                    "cwd": str(root),
                    "env": {},
                    "output_limit_bytes": 4096,
                    "required_gpu_uuids": [],
                }
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            "import importlib.util, json, sys; "
                            "spec=importlib.util.spec_from_file_location('runtime', sys.argv[1]); "
                            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                            "m.probe(json.loads(sys.argv[2]), json.loads(sys.argv[3]), json.loads(sys.argv[4]))",
                            str(RUNTIME_PATH),
                            json.dumps(command),
                            json.dumps(limits),
                            json.dumps(resources),
                        ]
                    )
                )
            first, second = processes
            self.assertEqual(first.wait(timeout=3), 0)
            self.assertEqual(second.wait(timeout=3), 0)
            intervals = [
                json.loads(output.read_text(encoding="utf-8"))
                for output in outputs
            ]
            intervals.sort(key=lambda item: item["start"])
            self.assertLess(
                intervals[1]["start"],
                intervals[0]["end"],
                "CPU-only commands were serialized by a declared GPU lock",
            )

    def test_probe_guardian_cleans_child_when_caller_dies(self) -> None:
        _load_runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "child.pid"
            child = root / "child.py"
            child.write_text(
                "import os, sys, time\nfrom pathlib import Path\n"
                "Path(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(30)\n",
                encoding="utf-8",
            )
            caller = root / "caller.py"
            caller.write_text(
                "import importlib.util, sys\n"
                "spec=importlib.util.spec_from_file_location('runtime', sys.argv[1])\n"
                "runtime=importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)\n"
                "runtime.probe({'argv':[sys.executable,sys.argv[2],sys.argv[3]],'cwd':sys.argv[4],"
                "'env':{},'output_limit_bytes':4096,'required_gpu_uuids':[]},"
                "{'operation_timeout_seconds':20,'command_timeout_seconds':20,"
                "'resource_wait_timeout_seconds':1,'cleanup_timeout_seconds':2},"
                "{'host_id':'host-a','gpu_uuids':[]})\n",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [sys.executable, str(caller), str(RUNTIME_PATH), str(child), str(pid_file), str(root)]
            )
            deadline = time.monotonic() + 2.0
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(pid_file.exists())
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            process.terminate()
            process.wait(timeout=2)
            deadline = time.monotonic() + 3.0
            while _process_exists(child_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(_process_exists(child_pid))


if __name__ == "__main__":
    unittest.main()
