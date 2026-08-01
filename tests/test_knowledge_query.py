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


def request(*, phenomena, max_results=5, max_context_bytes=8192, claim_layer="kernel"):
    return {
        "format_version": "cuda-kernel-optimizer/knowledge-input-v1",
        "operation": "query",
        "identity": {
            "gpu_architecture": "sm_120",
            "cuda_version": "12.9",
            "frameworks": {"triton": "3.4.0"},
            "phenomena": phenomena,
            "claim_layer": claim_layer,
        },
        "filters": {"mechanism_keys": []},
        "limits": {
            "max_results": max_results,
            "max_context_bytes": max_context_bytes,
        },
    }


class KnowledgeQueryTests(unittest.TestCase):
    def test_migrated_registry_keeps_all_content_kinds_and_closed_sources(self):
        cards = json.loads(CARDS.read_text(encoding="utf-8"))
        sources = json.loads(SOURCES.read_text(encoding="utf-8"))
        self.assertEqual(cards["schema_version"], "cuda-kernel-optimizer/knowledge-cards-v1")
        self.assertEqual(sources["schema_version"], "cuda-kernel-optimizer/knowledge-sources-v1")
        self.assertEqual(len(cards["cards"]), 106)
        self.assertEqual(
            {card["content_kind"] for card in cards["cards"]},
            {"capability", "diagnostic", "method", "workload_method", "local_case"},
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
        self.assertTrue(any(item["content_kind"] == "method" for item in result["matches"]))

    def test_empty_match_is_successful_and_has_no_side_effect(self):
        result = load_module().query(request(phenomena=["not.a.known.phenomenon"]))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["context_bytes"], 2)

    def test_claim_layer_filters_workload_cards(self):
        kernel = load_module().query(request(phenomena=["GPU idle gaps"], claim_layer="kernel"))
        workload = load_module().query(request(phenomena=["GPU idle gaps"], claim_layer="workload"))
        self.assertFalse(any(item["id"] == "workload.framework.launch-gaps" for item in kernel["matches"]))
        self.assertTrue(any(item["id"] == "workload.framework.launch-gaps" for item in workload["matches"]))

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
