from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V14Project:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = root / "project"
        self.artifact_root = root / "artifacts"
        self.original = self.project / "original"
        self.original.mkdir(parents=True)
        write_json(self.original / "implementation.json", {"value": 1})
        self.candidate = self.project / "candidate"
        self.candidate.mkdir()
        write_json(self.candidate / "implementation.json", {"value": 2})
        self.test_suite = self.project / "test-suite.json"
        write_json(self.test_suite, {"cases": [{"id": "main"}]})
        self.correctness_reference = self.project / "correctness-reference.json"
        write_json(self.correctness_reference, {"expected_value": 1})
        self.behavior = self.project / "behavior.json"
        self.events = self.project / "driver-events.jsonl"
        self.set_behavior()
        self.driver = self.project / "driver.py"
        self.driver.write_text(
            "\n".join(
                [
                    "import argparse",
                    "import json",
                    "import os",
                    "from pathlib import Path",
                    "parser = argparse.ArgumentParser()",
                    "parser.add_argument('--request', required=True)",
                    "args = parser.parse_args()",
                    "request = json.loads(Path(args.request).read_text('utf-8'))",
                    "assert Path(request['variant']['locator']).exists()",
                    "assert Path(request['test_suite']['locator']).exists()",
                    "assert Path(request['correctness']['reference']['locator']).exists()",
                    "root = Path(__file__).resolve().parent",
                    "behavior = json.loads((root / 'behavior.json').read_text('utf-8'))",
                    "event = json.dumps({",
                    "    'execution_id': request['execution_id'],",
                    "    'operation': request['operation'],",
                    "    'mode': request['mode'],",
                    "    'role': request['role'],",
                    "}, sort_keys=True) + '\\n'",
                    "descriptor = os.open(",
                    "    root / 'driver-events.jsonl',",
                    "    os.O_WRONLY | os.O_CREAT | os.O_APPEND,",
                    "    0o600,",
                    ")",
                    "try:",
                    "    os.write(descriptor, event.encode('utf-8'))",
                    "finally:",
                    "    os.close(descriptor)",
                    "result = {",
                    "    'protocol_version': 'cuda-kernel-optimizer/driver-result-v1',",
                    "    'request_digest': request['request_digest'],",
                    "    'target_id': request['target_id'],",
                    "    'execution_id': request['execution_id'],",
                    "    'variant_digest': request['variant']['digest'],",
                    "    'role': request['role'],",
                    "    'mode': request['mode'],",
                    "    'case_id': request['case']['id'],",
                    "    'artifacts': [],",
                    "    'cleanup': {'status': 'confirmed', 'live_tasks': []},",
                    "    'driver_identity': request['driver_identity'],",
                    "    'environment': {",
                    "        'gpu_uuids': [],",
                    "        'gpu_models': [],",
                    "        'gpu_architectures': [],",
                    "        'driver_version': 'none',",
                    "        'cuda_runtime_version': 'none',",
                    "        'frameworks': {},",
                    "        'container': {'kind': 'none', 'identity': 'none'},",
                    "    },",
                    "}",
                    "if request['mode'] in {'correctness', 'combined'}:",
                    "    correctness_status = behavior.get('correctness_by_role', {}).get(",
                    "        request['role'], behavior['correctness']",
                    "    )",
                    "    correctness_metric = (",
                    "        (1.0 if correctness_status == 'passed' else 0.0)",
                    "        if request['role'] in behavior.get('correctness_by_role', {})",
                    "        else behavior['correctness_metric']",
                    "    )",
                    "    result['correctness'] = {",
                    "        'status': correctness_status,",
                    "        'metrics': {'exact_match': correctness_metric},",
                    "    }",
                    "if request['mode'] in {'measure', 'combined'}:",
                    "    samples = behavior['samples'][request['role']]",
                    "    result['measurements'] = {",
                    "        'primary': {",
                    "            'name': 'latency_ms',",
                    "            'unit': 'ms',",
                    "            'samples': samples,",
                    "        },",
                    "        'constraints': [],",
                    "    }",
                    "output = Path(request['output_path'])",
                    "temporary = output.with_suffix(output.suffix + '.tmp')",
                    "temporary.write_text(json.dumps(result, sort_keys=True), encoding='utf-8')",
                    "os.replace(temporary, output)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def set_behavior(
        self,
        *,
        correctness: str = "passed",
        correctness_metric: float | None = None,
        correctness_by_role=None,
        original_samples=None,
        candidate_samples=None,
    ) -> None:
        write_json(
            self.behavior,
            {
                "correctness": correctness,
                "correctness_metric": (
                    1.0 if correctness == "passed" else 0.0
                )
                if correctness_metric is None
                else correctness_metric,
                "correctness_by_role": (
                    {} if correctness_by_role is None else correctness_by_role
                ),
                "samples": {
                    "original": [10.0] if original_samples is None else original_samples,
                    "reference": [10.0] if original_samples is None else original_samples,
                    "candidate": [9.0] if candidate_samples is None else candidate_samples,
                },
            },
        )

    def readiness_input(self) -> dict:
        return {
            "format_version": "cuda-kernel-optimizer/readiness-input-v1",
            "operation": "check",
            "artifact_root": str(self.artifact_root),
            "target_mode": "optimization",
            "claim_layer": "workload",
            "test_suite": {
                "path": str(self.test_suite),
                "case_ids": ["main"],
            },
            "correctness": {
                "reference_path": str(self.correctness_reference),
                "method": "driver",
                "acceptance": {
                    "metric": "exact_match",
                    "operator": "greater_or_equal",
                    "value": 1.0,
                },
            },
            "original": {
                "kind": "source_snapshot",
                "path": str(self.original),
            },
            "objective": {
                "primary_metric": {
                    "name": "latency_ms",
                    "unit": "ms",
                    "direction": "lower",
                    "aggregation": "median",
                },
                "minimum_effect": {"value": 0.5, "unit": "percent"},
                "constraints": [],
            },
            "driver": {
                "command": [sys.executable, str(self.driver)],
                "request_argument": "--request",
                "execution_mode": "separate",
                "protocol_version": "cuda-kernel-optimizer/driver-v1",
                "profiler_capabilities": [],
                "side_effects": [],
                "cleanup_contract": {
                    "kind": "process_group_only",
                    "external_tasks": False,
                },
            },
            "environment_requirements": {
                "gpu_uuids": [],
                "required_tools": [],
            },
            "validity_requirements": {
                "minimum_pairs": 2,
                "confidence": 0.95,
                "bootstrap_samples": 100,
            },
            "smoke": {
                "mode": "correctness",
                "case_id": "main",
                "resources": {"host_id": "local-test", "gpu_uuids": []},
                "runtime_limits": {
                    "operation_timeout_seconds": 2.0,
                    "command_timeout_seconds": 1.0,
                    "resource_wait_timeout_seconds": 1.0,
                    "cleanup_timeout_seconds": 1.0,
                },
            },
            "scan_limits": {
                "max_files": 100,
                "max_total_bytes": 1024 * 1024,
                "max_wall_seconds": 2.0,
            },
        }

    def target_ref(self) -> dict:
        path = self.artifact_root / "target.json"
        target = json.loads(path.read_text(encoding="utf-8"))
        return {
            "id": target["id"],
            "sha256": sha256_file(path),
        }

    def baseline_input(self) -> dict:
        return {
            "format_version": "cuda-kernel-optimizer/evaluator-input-v1",
            "operation": "baseline",
            "artifact_root": str(self.artifact_root),
            "target_ref": self.target_ref(),
            "sampling_design": {
                "case_ids": ["main"],
                "samples_per_case": 1,
                "seed": 0,
            },
            "resources": {"host_id": "local-test", "gpu_uuids": []},
            "operation_timeout_seconds": 3.0,
            "command_timeout_seconds": 1.0,
            "resource_wait_timeout_seconds": 1.0,
            "cleanup_timeout_seconds": 1.0,
            "launch_deadline": time.time() + 2.0,
        }

    def experiment_input(self, baseline_ref: dict) -> dict:
        return {
            "format_version": "cuda-kernel-optimizer/evaluator-input-v1",
            "operation": "experiment",
            "artifact_root": str(self.artifact_root),
            "target_ref": self.target_ref(),
            "baseline_ref": baseline_ref,
            "source_base": {
                "kind": "source_snapshot",
                "path": str(self.original),
            },
            "candidate": {
                "kind": "source_snapshot",
                "path": str(self.candidate),
            },
            "hypothesis": "changing the implementation reduces workload latency",
            "mechanism_key": "test.fixture.latency",
            "claim_layer": "workload",
            "cheapest_falsifier": {
                "kind": "none",
                "reason": "candidate correctness is the first executable check",
            },
            "screen_design": {
                "enabled": True,
                "kind": "diagnostic_proxy",
                "reason": "a one-sample pair checks the mechanism signal",
                "claim": "the candidate changes the primary metric",
            },
            "estimated_cost": {
                "screen": {
                    "p50_seconds": 1.0,
                    "p90_seconds": 2.0,
                    "gpu_count": 0,
                    "basis": "driver smoke",
                },
                "target": {
                    "p50_seconds": 2.0,
                    "p90_seconds": 4.0,
                    "gpu_count": 0,
                    "basis": "driver smoke",
                },
            },
            "minimum_effect": {"value": 0.5, "unit": "percent"},
            "reject_if": [
                {"kind": "correctness_failed"},
                {"kind": "screen_claim_falsified"},
            ],
            "promote_if": [
                {"kind": "target_minimum_effect_met"},
                {"kind": "constraints_passed"},
            ],
            "change_scope": ["implementation.json"],
            "max_risk": "low",
        }

    def screen_input(self, experiment_ref: dict) -> dict:
        return {
            "format_version": "cuda-kernel-optimizer/evaluator-input-v1",
            "operation": "screen",
            "artifact_root": str(self.artifact_root),
            "target_ref": self.target_ref(),
            "experiment_ref": experiment_ref,
            "sampling_design": {
                "case_ids": ["main"],
                "pairs": 1,
                "seed": 0,
            },
            "resources": {"host_id": "local-test", "gpu_uuids": []},
            "operation_timeout_seconds": 4.0,
            "command_timeout_seconds": 1.0,
            "resource_wait_timeout_seconds": 1.0,
            "cleanup_timeout_seconds": 1.0,
            "launch_deadline": time.time() + 2.0,
        }

    def target_input(self, experiment_ref: dict) -> dict:
        return {
            **self.screen_input(experiment_ref),
            "operation": "target",
            "sampling_design": {
                "case_ids": ["main"],
                "pairs": 2,
                "seed": 1,
            },
        }

    def final_audit_input(self) -> dict:
        value = self.target_input({"id": "unused", "sha256": "0" * 64})
        value["operation"] = "final_audit"
        value.pop("experiment_ref")
        return value

    def knowledge_input(self) -> dict:
        return {
            "format_version": "cuda-kernel-optimizer/knowledge-input-v1",
            "operation": "query",
            "identity": {
                "gpu_architecture": "sm_120",
                "cuda_version": "13.0",
                "frameworks": {"triton": "3.5"},
                "phenomena": ["fixture-signal-with-no-card"],
                "claim_layer": "workload",
            },
            "filters": {"mechanism_keys": []},
            "limits": {"max_results": 3, "max_context_bytes": 4096},
        }

    def check(self) -> dict:
        completed = self.run_tool(
            "readiness.py",
            "check",
            self.readiness_input(),
        )
        if completed.returncode != 0:
            raise AssertionError(
                "readiness failed:\n"
                f"stdout={completed.stdout}\n"
                f"stderr={completed.stderr}"
            )
        return decode_stdout(completed)

    def baseline(self) -> dict:
        completed = self.run_tool(
            "workload_evaluate.py",
            "baseline",
            self.baseline_input(),
            wait=True,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "baseline failed:\n"
                f"stdout={completed.stdout}\n"
                f"stderr={completed.stderr}"
            )
        return decode_stdout(completed)

    def driver_events(self) -> list[dict]:
        if not self.events.exists():
            return []
        return [
            json.loads(line)
            for line in self.events.read_text(encoding="utf-8").splitlines()
        ]

    def run_tool(
        self,
        filename: str,
        operation: str,
        request: dict,
        *,
        wait: bool = False,
    ) -> subprocess.CompletedProcess:
        request_path = self.root / f"{filename}-{operation}-input.json"
        write_json(request_path, request)
        command = [
            sys.executable,
            str(SCRIPTS / filename),
            operation,
            "--request",
            str(request_path),
        ]
        if wait:
            command.append("--wait")
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )


def decode_stdout(completed: subprocess.CompletedProcess) -> dict:
    return json.loads(completed.stdout)


def decode_stderr(completed: subprocess.CompletedProcess) -> dict:
    return json.loads(completed.stderr)
