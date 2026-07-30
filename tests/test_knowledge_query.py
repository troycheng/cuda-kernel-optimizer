from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_diagnostic_knowledge import _frozen_inputs, _source_verified_frozen


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "knowledge_query.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_knowledge", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class KnowledgeQueryTests(unittest.TestCase):
    def test_query_frozen_delegates_to_identity_bound_context(self) -> None:
        frozen = _frozen_inputs()
        result = load_module().query_frozen(frozen, limit=3)
        self.assertEqual(result["promotion_authority"], "none")
        self.assertEqual(result["candidates"], [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--frozen-input",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["promotion_authority"], "none")

    def test_detached_frozen_query_rejects_candidate_bearing_input(self) -> None:
        frozen = _source_verified_frozen()
        frozen["active_evidence_results"][0][
            "adapter_implementation_sha256"
        ] = "a" * 64
        frozen["active_evidence_results"][0]["result_sha256"] = "b" * 64

        with self.assertRaisesRegex(ValueError, "Controller-owned"):
            load_module().query_frozen(frozen, limit=3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged-frozen.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--frozen-input",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Controller-owned", completed.stderr)

    def test_query_returns_small_arch_compatible_method_set(self) -> None:
        result = load_module().query(arch="sm_120", axis="compute", limit=3)
        self.assertLessEqual(len(result["methods"]), 3)
        self.assertEqual(result["arch"], "sm_120")
        self.assertNotIn(
            "compute.gemm_softmax_interleave",
            {item["id"] for item in result["methods"]},
        )
        for item in result["methods"]:
            self.assertNotIn("typical_speedup", item)
            self.assertEqual(item["applicability"], "unverified")

    def test_observed_bad_metric_ranks_matching_method_first(self) -> None:
        result = load_module().query(
            arch="sm_120",
            axis="compute",
            observed_metrics={
                "sm__pipe_tensor_op_hmma_cycles_active.pct_of_peak": 10
            },
            limit=3,
        )
        self.assertEqual(result["methods"][0]["id"], "compute.tensor_core")
        self.assertEqual(
            result["methods"][0]["applicability"], "observed_bad_trigger"
        )

    def test_unknown_arch_fails_closed_instead_of_numeric_inheritance(self) -> None:
        with self.assertRaises(ValueError):
            load_module().query(arch="sm_999", axis="memory", limit=3)

    def test_min_sm_rejects_method_even_when_feature_name_is_present(self) -> None:
        module = load_module()
        registry = {
            "arch_feature_map": {"sm_80": ["tensor_core", "tma"]},
            "methods": {
                "future": {
                    "axis": "memory",
                    "priority": 1,
                    "min_sm": 90,
                    "name": "future",
                    "required_features": ["tma"],
                }
            },
        }
        self.assertEqual(
            module._kernel_cards(registry, "sm_80", None, None, {}),
            [],
        )

    def test_registry_contains_no_transferable_speedup_claims(self) -> None:
        registry = json.loads(
            (SCRIPT.parents[1] / "references" / "method_registry.json").read_text()
        )
        self.assertFalse(
            any("typical_speedup" in method for method in registry["methods"].values())
        )

    def test_kernel_cards_respect_exact_architecture_gates(self) -> None:
        registry = json.loads(
            (SCRIPT.parents[1] / "references" / "method_registry.json").read_text()
        )
        module = load_module()
        for arch in (
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_100",
            "sm_103",
            "sm_110",
            "sm_120",
            "sm_121",
        ):
            with self.subTest(arch=arch):
                available = set(registry["arch_feature_map"][arch])
                for card in module._kernel_cards(registry, arch, None, None, {}):
                    method = registry["methods"][card["id"]]
                    self.assertLessEqual(method["min_sm"], int(arch.removeprefix("sm_")))
                    self.assertTrue(
                        set(card["required_features"]).issubset(available)
                    )

    def test_generic_cooperative_groups_is_not_cluster_routing(self) -> None:
        module = load_module()
        cluster = module.query(
            arch="sm_80",
            axis="latency",
            bottleneck="cluster",
            limit=20,
        )
        self.assertNotIn(
            "latency.cooperative_groups_sync",
            {item["id"] for item in cluster["methods"]},
        )
        generic = module.query(
            arch="sm_80",
            axis="latency",
            bottleneck="cooperative groups",
            limit=20,
        )
        self.assertIn(
            "latency.cooperative_groups_sync",
            {item["id"] for item in generic["methods"]},
        )

    def test_workload_query_routes_non_kernel_bottlenecks(self) -> None:
        result = load_module().query(
            arch="sm_120", layer="workload", bottleneck="framework", limit=2
        )
        self.assertLessEqual(len(result["methods"]), 2)
        self.assertTrue(result["methods"])
        self.assertTrue(all(item["layer"] == "workload" for item in result["methods"]))


if __name__ == "__main__":
    unittest.main()
