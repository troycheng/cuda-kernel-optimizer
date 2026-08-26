from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "cuda-kernel-optimizer"
SCRIPT = SKILL / "scripts" / "knowledge_query.py"
CARDS = SKILL / "references" / "knowledge" / "cards.json"
SOURCES = SKILL / "references" / "knowledge" / "sources.json"


def load_module():
    spec = importlib.util.spec_from_file_location("v14_knowledge_query", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def request(
    *,
    phenomena,
    max_results=5,
    max_context_bytes=8192,
    claim_layer="kernel",
    gpu_architecture="sm_120",
    cuda_version="12.9",
    frameworks=None,
    mechanism_keys=None,
):
    return {
        "format_version": "cuda-kernel-optimizer/knowledge-input-v1",
        "operation": "query",
        "identity": {
            "gpu_architecture": gpu_architecture,
            "cuda_version": cuda_version,
            "frameworks": frameworks or {"triton": "3.4.0"},
            "phenomena": phenomena,
            "claim_layer": claim_layer,
        },
        "filters": {"mechanism_keys": mechanism_keys or []},
        "limits": {
            "max_results": max_results,
            "max_context_bytes": max_context_bytes,
        },
    }


class KnowledgeQueryTests(unittest.TestCase):
    def test_registry_is_smaller_and_separates_epistemic_roles(self):
        cards = json.loads(CARDS.read_text(encoding="utf-8"))
        sources = json.loads(SOURCES.read_text(encoding="utf-8"))
        self.assertEqual(cards["schema_version"], "cuda-kernel-optimizer/knowledge-cards-v2")
        self.assertEqual(sources["schema_version"], "cuda-kernel-optimizer/knowledge-sources-v2")
        self.assertLess(len(cards["cards"]), 106)
        self.assertLess(CARDS.stat().st_size, 200_000)
        self.assertEqual(
            {card["content_kind"] for card in cards["cards"]},
            {"technical_contract", "heuristic"},
        )
        for card in cards["cards"]:
            self.assertNotIn("details", card)
            self.assertNotIn(None, card["match_terms"])
            self.assertNotIn("Does None", card["distinguishing_question"])
            if card["content_kind"] == "technical_contract":
                self.assertEqual(
                    set(card["contract"]),
                    {"proposition", "non_claims", "decision_impact"},
                )
        source_ids = {source["id"] for source in sources["sources"]}
        self.assertTrue(all(set(card["source_ids"]).issubset(source_ids) for card in cards["cards"]))

    def test_query_is_bounded_and_deduplicated_by_mechanism(self):
        result = load_module().query(
            request(phenomena=["sm__pipe_tensor_op_hmma_cycles_active.pct_of_peak"], max_results=3)
        )
        self.assertEqual(result["status"], "completed")
        self.assertLessEqual(len(result["matches"]), 3)
        self.assertLessEqual(result["context_bytes"], 8192)
        self.assertEqual(
            len({item["mechanism_key"] for item in result["matches"]}),
            len(result["matches"]),
        )
        self.assertTrue(any(item["content_kind"] == "heuristic" for item in result["matches"]))

    def test_pdl_contract_uses_the_documented_compute_capability_boundary(self):
        for architecture in ("sm_90", "sm_120", "sm_122"):
            with self.subTest(architecture=architecture):
                result = load_module().query(
                    request(
                        phenomena=[],
                        mechanism_keys=["latency.pdl_overlap"],
                        gpu_architecture=architecture,
                        frameworks={"cuda": "12.9"},
                    )
                )
                self.assertEqual(
                    result["matches"][0]["applicability"],
                    {"relation": "compatible", "mismatches": [], "limitations": []},
                )
        result = load_module().query(
            request(
                phenomena=[],
                mechanism_keys=["latency.pdl_overlap"],
                gpu_architecture="sm_89",
                frameworks={"cuda": "12.9"},
            )
        )
        self.assertEqual(
            result["matches"][0]["applicability"]["relation"],
            "incompatible",
        )

        patch_release = load_module().query(request(
            phenomena=[], mechanism_keys=["latency.pdl_overlap"],
            gpu_architecture="sm_90", cuda_version="12.9.3", frameworks={"cuda": "12.9.3"},
        ))
        newer_unreviewed = load_module().query(request(
            phenomena=[], mechanism_keys=["latency.pdl_overlap"],
            gpu_architecture="sm_130", cuda_version="13.3", frameworks={"cuda": "13.3"},
        ))
        self.assertEqual(patch_release["matches"][0]["applicability"]["relation"], "compatible")
        self.assertEqual(newer_unreviewed["matches"][0]["applicability"]["relation"], "related")

        result = load_module().query(
            request(
                phenomena=[],
                mechanism_keys=["latency.pdl_overlap"],
                gpu_architecture="future_architecture",
                frameworks={"cuda": "12.9"},
            )
        )
        self.assertEqual(len(result["matches"]), 1)
        match = result["matches"][0]
        self.assertEqual(match["content_kind"], "technical_contract")
        self.assertEqual(match["applicability"]["relation"], "related")
        self.assertIn("compute capability 9.0", match["contract"]["proposition"])
        self.assertIn("cudaGridDependencySynchronize", match["contract"]["proposition"])
        self.assertNotIn("decision", match)
        self.assertNotIn("premise_resolved", match)
        source = next(item for item in match["sources"] if item["id"] == "nvidia-cuda-pdl")
        self.assertIn("programmatic-dependent-launch", source["url"])
        self.assertTrue(source["locator"])

    def test_explicit_query_returns_identity_mismatch_without_rejecting_mechanism(self):
        result = load_module().query(
            request(
                phenomena=["an unrelated observation must not hide an explicit mechanism"],
                mechanism_keys=["workload.communication.collective"],
                claim_layer="serving",
                gpu_architecture="sm_90",
                frameworks={"nccl": "2.27.3"},
            )
        )
        self.assertEqual(len(result["matches"]), 1)
        match = result["matches"][0]
        self.assertEqual(match["applicability"]["relation"], "related")
        self.assertEqual(match["applicability"]["limitations"][0]["field"], "frameworks.nccl")
        self.assertNotIn("unsupported", json.dumps(match).lower())

    def test_contract_without_enumerated_component_versions_is_only_related(self):
        result = load_module().query(
            request(
                phenomena=[],
                mechanism_keys=["triton.compiler-hints"],
                frameworks={"triton": "3.4.0"},
            )
        )
        match = result["matches"][0]
        self.assertEqual(match["applicability"]["relation"], "related")
        self.assertEqual(
            match["applicability"]["limitations"][0]["field"],
            "frameworks.triton",
        )

    def test_triton_ocr_serving_contracts_are_source_bound_and_version_limited(self):
        cases = [
            (
                "triton.metrics-counter-semantics",
                {"tritonserver": "2.61.0"},
                "nvidia-triton-metrics",
                "batch of n as n inferences",
            ),
            (
                "triton.dynamic-batching",
                {"tritonserver": "2.61.0"},
                "nvidia-triton-batcher",
                "does not guarantee a throughput gain",
            ),
            (
                "triton.instance-group-interaction",
                {"tritonserver": "2.61.0"},
                "nvidia-triton-optimization",
                "does not add linearly",
            ),
            (
                "triton.ensemble-dataflow",
                {"tritonserver": "2.61.0"},
                "nvidia-triton-ensemble-models",
                "does not guarantee that every intermediate tensor remains in GPU memory",
            ),
            (
                "triton.response-cache-eligibility",
                {"tritonserver": "2.61.0"},
                "nvidia-triton-response-cache",
                "Visually similar OCR images",
            ),
            (
                "tensorrt.dynamic-shape-profiles",
                {"tensorrt": "11.0.0"},
                "nvidia-tensorrt-dynamic-shapes",
                "wider min-opt-max range",
            ),
            (
                "tensorrt.cuda-graph-context-shape",
                {"tensorrt": "11.0.0"},
                "nvidia-tensorrt-performance-optimization",
                "does not prove an enqueue-bound workload",
            ),
            (
                "triton.ort-tensorrt-fallback",
                {"onnxruntime": "1.23.0", "tritonserver": "2.61.0"},
                "triton-onnxruntime-backend",
                "does not mean every ONNX node executes in TensorRT",
            ),
            (
                "triton.dali-ragged-image-batching",
                {"dali": "1.50.0", "tritonserver": "2.61.0"},
                "nvidia-triton-dali-inference",
                "does not guarantee a full-service speedup",
            ),
        ]
        for mechanism_key, frameworks, source_id, bounded_text in cases:
            with self.subTest(mechanism_key=mechanism_key):
                result = load_module().query(
                    request(
                        phenomena=[],
                        mechanism_keys=[mechanism_key],
                        claim_layer="workload",
                        frameworks=frameworks,
                    )
                )
                self.assertEqual(len(result["matches"]), 1)
                match = result["matches"][0]
                self.assertEqual(match["content_kind"], "technical_contract")
                self.assertEqual(match["status"], "source_reviewed")
                self.assertEqual(match["applicability"]["relation"], "related")
                self.assertTrue(match["applicability"]["limitations"])
                self.assertIn(
                    bounded_text,
                    " ".join(
                        [match["contract"]["proposition"]]
                        + match["contract"]["non_claims"]
                    ),
                )
                source = next(
                    item for item in match["sources"] if item["id"] == source_id
                )
                self.assertTrue(source["locator"])
                self.assertTrue(source["url"].startswith("https://"))

    def test_serving_only_triton_contract_rejects_a_kernel_claim(self):
        result = load_module().query(
            request(
                phenomena=[],
                mechanism_keys=["triton.dynamic-batching"],
                claim_layer="kernel",
                frameworks={"tritonserver": "2.61.0"},
            )
        )
        applicability = result["matches"][0]["applicability"]
        self.assertEqual(applicability["relation"], "incompatible")
        self.assertEqual(applicability["mismatches"][0]["field"], "claim_layer")

    def test_documented_version_families_match_patch_releases(self):
        cutlass = load_module().query(request(
            phenomena=[], mechanism_keys=["cutlass.blackwell-architecture-boundary"],
            frameworks={"cutlass": "4.7.0"},
        ))
        nccl = load_module().query(request(
            phenomena=[], mechanism_keys=["workload.communication.collective"],
            claim_layer="serving", frameworks={"nccl": "2.30.5"},
        ))
        self.assertEqual(cutlass["matches"][0]["applicability"]["relation"], "compatible")
        self.assertEqual(nccl["matches"][0]["applicability"]["relation"], "compatible")

    def test_empty_match_is_successful_and_has_no_side_effect(self):
        result = load_module().query(request(phenomena=["not.a.known.phenomenon"]))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["context_bytes"], 2)

    def test_claim_layer_filters_workload_cards(self):
        kernel = load_module().query(request(phenomena=["GPU idle gaps"], claim_layer="kernel"))
        workload = load_module().query(request(phenomena=["GPU idle gaps"], claim_layer="workload"))
        kernel_contract = next(
            item for item in kernel["matches"] if item["id"] == "contract.nsys.time-utilization"
        )
        self.assertEqual(kernel_contract["applicability"]["relation"], "incompatible")
        self.assertTrue(
            any(item["id"] == "contract.nsys.time-utilization" for item in workload["matches"])
        )

    def test_matching_card_exposes_only_a_digest_bound_playbook_reference(self):
        result = load_module().query(
            request(phenomena=["triton.decode-attention-gqa"])
        )
        match = next(
            item
            for item in result["matches"]
            if item["id"] == "capability.triton.decode-attention-gqa"
        )
        self.assertEqual(
            match["playbook"]["path"],
            "playbooks/triton-decode-attention-gqa.md",
        )
        self.assertRegex(match["playbook"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("content", match["playbook"])

    def test_cli_only_reads_request_and_returns_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request(phenomena=["nothing.matches"])), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "query", "--request", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["matches"], [])


if __name__ == "__main__":
    unittest.main()
