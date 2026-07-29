#!/usr/bin/env python3
"""Freeze archive references without fabricating a runtime replay contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
REFERENCES = ROOT / "skills" / "cuda-kernel-optimizer" / "references"
V1_2_ROUTER_SNAPSHOT = (
    ROOT
    / "tests"
    / "fixtures"
    / "knowledge_replay"
    / "v1_2_router_snapshot.json"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TERMINAL_DECISIONS = {"MEASURE", "PURSUE", "REVIEW_REQUIRED", "STOP"}
_CONTROLLER_SOURCE_FILES = {
    "workload_controller_sha256": SCRIPTS / "workload_controller.py",
    "evidence_selector_sha256": SCRIPTS / "evidence_selector.py",
    "diagnostic_knowledge_sha256": SCRIPTS / "diagnostic_knowledge.py",
    "diagnostic_cards_sha256": REFERENCES / "diagnostic_cards.json",
    "case_memory_sha256": REFERENCES / "case_memory.json",
}
_V1_2_ROUTER_CARDS_SHA256 = (
    "152da6fd51bb68affc1e2903910aeb1990fc98dc0f2b1c0a600ff5d5073675e7"
)
_V1_2_ROUTER_ACTIONS_SHA256 = (
    "48538fb8723a61a6d43090c72fbc008924efcb4b70a8fc9276fe21ea76a097f4"
)
_V1_2_CARD_TO_MECHANISM = {
    "diagnostic.cross-layer.triage": "crosslayerunattributedcriticalpath",
    "diagnostic.framework.launch-gaps": "frameworklaunchgaps",
    "diagnostic.kernel.resource-or-memory": "kernelresourceormemory",
    "diagnostic.cpu-data.starvation": "cpudatainputstarvation",
    "diagnostic.transfer.h2d": "transferhostdeviceserialization",
    "diagnostic.communication.collective": "communicationcollectivecriticalpath",
    "diagnostic.io.request-path": "iorequestpath",
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _validate_controller_source_identity(
    value: object,
    *,
    verify_commit: bool = False,
) -> dict:
    required = {"source_repo_head", *_CONTROLLER_SOURCE_FILES}
    if type(value) is not dict or set(value) != required:
        raise ValueError("Controller source-state fields are invalid")
    source = copy.deepcopy(value)
    if re.fullmatch(r"[0-9a-f]{40}", source["source_repo_head"]) is None:
        raise ValueError("Controller source-state commit is invalid")
    for field, path in _CONTROLLER_SOURCE_FILES.items():
        digest = source[field]
        if (
            type(digest) is not str
            or _SHA256.fullmatch(digest) is None
            or digest != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("Controller source-state digest is invalid")
        if verify_commit:
            relative_path = path.relative_to(ROOT).as_posix()
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "show",
                    f"{source['source_repo_head']}:{relative_path}",
                ],
                check=False,
                capture_output=True,
            )
            if result.returncode != 0:
                raise ValueError("Controller source commit is unavailable")
            if hashlib.sha256(result.stdout).hexdigest() != digest:
                raise ValueError(
                    "Controller source commit does not reproduce recorded files"
                )
    return source


def _load_controller_source_identity(run_root: Path) -> dict:
    path = run_root / "source-state.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Controller scoreable source-state.json is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Controller source-state.json is invalid") from error
    return _validate_controller_source_identity(value)


def _load_v1_2_router_snapshot() -> dict:
    registry = json.loads(V1_2_ROUTER_SNAPSHOT.read_text(encoding="utf-8"))
    if (
        set(registry)
        != {
            "schema_version",
            "source_tag",
            "source_commit",
            "diagnostic_cards_sha256",
            "evidence_action_catalog_sha256",
            "cards",
            "actions",
        }
        or registry.get("schema_version")
        != "cuda-optimizer/v1-2-router-snapshot-v1"
        or registry.get("source_tag") != "v1.2.0"
        or registry.get("source_commit")
        != "bdf68c875b95f8da06937b9034dc35cd6ea930ed"
        or registry.get("diagnostic_cards_sha256")
        != "bed90de9593a1596794dc4fbdcc73e0786a0327c4c59c4d1c2ce735a380524c9"
        or registry.get("evidence_action_catalog_sha256")
        != "2a0b73c0fd6d41cfa0e0706aef68bdbfa46a1e4b82787b199e3e39a928403e2a"
        or canonical_sha256(registry.get("cards")) != _V1_2_ROUTER_CARDS_SHA256
        or canonical_sha256(registry.get("actions"))
        != _V1_2_ROUTER_ACTIONS_SHA256
    ):
        raise ValueError("frozen V1.2 router snapshot provenance is invalid")
    return registry


def _route_v1_2_cards(
    diagnosis: dict,
    execution_map: dict,
    *,
    limit: int = 3,
) -> dict:
    """Replay the frozen V1.2 router against its original seven card IDs."""
    primary = diagnosis.get("primary_category")
    categories = []
    if isinstance(primary, str) and primary != "mixed":
        categories.append(primary)
    ranked = diagnosis.get("ranked_categories", [])
    if type(ranked) is list:
        categories.extend(
            item.get("category")
            for item in ranked
            if type(item) is dict and isinstance(item.get("category"), str)
        )
    if not categories:
        categories = ["unknown"]
    elif primary == "mixed":
        categories.insert(0, "mixed")
    categories = list(dict.fromkeys(categories))
    labels = " ".join(
        str(item.get("label", "")).lower()
        for item in execution_map.get("nodes", [])
        if type(item) is dict
    )
    registry = _load_v1_2_router_snapshot()
    cards = [
        card
        for card in registry["cards"]
        if card["id"] in _V1_2_CARD_TO_MECHANISM
    ]
    if {card["id"] for card in cards} != set(_V1_2_CARD_TO_MECHANISM):
        raise ValueError("frozen V1.2 diagnostic card set is incomplete")
    ranked_cards = []
    for card in cards:
        category_rank = min(
            (
                categories.index(category)
                for category in card["categories"]
                if category in categories
            ),
            default=99,
        )
        if category_rank == 99:
            continue
        term_match = any(term in labels for term in card["match_terms"])
        ranked_cards.append(
            (
                (
                    category_rank,
                    0 if term_match else 1,
                    card["priority"],
                    card["id"],
                ),
                card,
            )
        )
    if not ranked_cards:
        fallback = next(
            card
            for card in cards
            if card["id"] == "diagnostic.cross-layer.triage"
        )
        ranked_cards = [((0, 0, 0, fallback["id"]), fallback)]
    ranked_cards.sort(key=lambda item: item[0])
    return {
        "schema_version": "cuda-optimizer/diagnostic-knowledge-context-v1",
        "categories": categories,
        "cards": [copy.deepcopy(card) for _, card in ranked_cards[:limit]],
        "promotion_authority": "none",
    }


def _load_module(filename: str, name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_json(path: Path, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _contained_path(root: Path, relative_path: str, label: str) -> Path:
    if type(relative_path) is not str or not relative_path:
        raise ValueError(f"{label} relative_path must be non-empty")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} relative_path must stay inside the run")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} source must be a regular file: {relative_path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} source escapes the run") from error
    return path


def _json_pointer(value: object, pointer: str, label: str) -> object:
    if type(pointer) is not str or not pointer.startswith("/"):
        raise ValueError(f"{label} locator must be a JSON pointer")
    current = value
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if type(current) is dict and token in current:
            current = current[token]
        elif type(current) is list and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ValueError(f"{label} locator does not resolve: {pointer}")
    return copy.deepcopy(current)


def _canonical_strings(value: object, label: str) -> list[str]:
    if (
        type(value) is not list
        or any(type(item) is not str or not item for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{label} must be a sorted unique string array")
    return list(value)


def _label_source_document(
    run_root: Path,
    source: object,
    label: str,
) -> tuple[dict, dict]:
    if type(source) is not dict or set(source) != {
        "relative_path",
        "source_sha256",
    }:
        raise ValueError(f"{label} must name one sealed JSON source")
    path = _contained_path(run_root, source["relative_path"], label)
    try:
        raw_bytes = path.read_bytes()
        raw_json = raw_bytes.decode("utf-8")
        value = json.loads(raw_json)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if source["source_sha256"] != digest:
        raise ValueError(f"{label} source digest drifted")
    if type(value) is not dict:
        raise ValueError(f"{label} must contain a JSON object")
    return value, {
        "relative_path": path.relative_to(run_root).as_posix(),
        "source_sha256": digest,
        "raw_json": raw_json,
    }


def _candidate_outcome_from_validation(value: object) -> dict:
    if type(value) is not dict:
        raise ValueError("candidate validation must be an object")
    required = {
        "accepted_mechanism_keys",
        "bootstrap_95_benefit_ci_us",
        "cheapest_valid_action_ids",
        "correctness_passed",
        "mean_benefit_us",
        "minimum_mechanism_effect_us",
        "pair_count",
        "validation_result",
        "wins",
    }
    if not required.issubset(value):
        raise ValueError("candidate validation is missing scored fields")
    promoted = _canonical_strings(
        value["accepted_mechanism_keys"],
        "candidate validation accepted_mechanism_keys",
    )
    actions = _canonical_strings(
        value["cheapest_valid_action_ids"],
        "candidate validation cheapest_valid_action_ids",
    )
    if not actions:
        raise ValueError("candidate validation has no cheapest valid action")
    interval = value["bootstrap_95_benefit_ci_us"]
    if (
        type(interval) is not list
        or len(interval) != 2
        or any(
            type(item) not in {int, float} or not math.isfinite(item)
            for item in interval
        )
        or interval[0] > interval[1]
    ):
        raise ValueError("candidate validation confidence interval is invalid")
    mean = value["mean_benefit_us"]
    threshold = value["minimum_mechanism_effect_us"]
    if (
        type(mean) not in {int, float}
        or not math.isfinite(mean)
        or type(threshold) not in {int, float}
        or not math.isfinite(threshold)
        or threshold <= 0
        or type(value["correctness_passed"]) is not bool
        or type(value["pair_count"]) is not int
        or value["pair_count"] <= 0
        or type(value["wins"]) is not int
        or not 0 <= value["wins"] <= value["pair_count"]
        or not interval[0] <= mean <= interval[1]
    ):
        raise ValueError("candidate validation measurement summary is invalid")
    result = value["validation_result"]
    if (
        result == "confirmed_above_mechanism_threshold"
        and value["correctness_passed"]
        and promoted
        and interval[0] >= threshold
    ):
        status = "promoted"
    elif (
        result == "benefit_below_mechanism_threshold"
        and value["correctness_passed"]
        and not promoted
        and interval[1] < threshold
    ):
        status = "rejected"
    else:
        raise ValueError("candidate validation result is internally inconsistent")
    return {
        "status": status,
        "reason": result,
        "correctness_passed": value["correctness_passed"],
        "mean_benefit_us": mean,
        "bootstrap_95_benefit_ci_us": copy.deepcopy(interval),
        "minimum_effect_us": threshold,
        "pair_count": value["pair_count"],
        "wins": value["wins"],
    }


def _load_machine_mapped_label(run_root: Path, label_path: Path) -> dict:
    label_path = label_path.expanduser().resolve(strict=False)
    try:
        label_path.relative_to(run_root)
    except ValueError as error:
        raise ValueError("label source must be inside the Controller run") from error
    try:
        label_raw_bytes = label_path.read_bytes()
        label_raw_json = label_raw_bytes.decode("utf-8")
        value = json.loads(label_raw_json)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("label source is invalid") from error
    fields = {
        "schema_version",
        "case_id",
        "diagnostic_decision_source",
        "candidate_validation_source",
    }
    if set(value) != fields:
        raise ValueError("label source fields are not closed")
    if value["schema_version"] != "cuda-optimizer/knowledge-replay-label-v2":
        raise ValueError("label source schema is unsupported")
    if (
        type(value["case_id"]) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value["case_id"])
        is None
    ):
        raise ValueError("label source case_id is invalid")
    decision, decision_document = _label_source_document(
        run_root,
        value["diagnostic_decision_source"],
        "diagnostic decision label evidence",
    )
    outcome, outcome_document = _label_source_document(
        run_root,
        value["candidate_validation_source"],
        "candidate validation label evidence",
    )
    diagnostic_decision = decision.get("decision")
    if diagnostic_decision not in _TERMINAL_DECISIONS:
        raise ValueError("diagnostic decision label evidence is unsupported")
    promoted = _canonical_strings(
        outcome.get("accepted_mechanism_keys"),
        "candidate validation accepted_mechanism_keys",
    )
    actions = _canonical_strings(
        outcome.get("cheapest_valid_action_ids"),
        "candidate validation cheapest_valid_action_ids",
    )
    return {
        "case_id": value["case_id"],
        "promoted_mechanism_keys": promoted,
        "cheapest_valid_action_ids": actions,
        "observed_diagnostic_decision": diagnostic_decision,
        "candidate_outcome": _candidate_outcome_from_validation(outcome),
        "source_documents": {
            "label_manifest": {
                "relative_path": label_path.relative_to(run_root).as_posix(),
                "source_sha256": hashlib.sha256(label_raw_bytes).hexdigest(),
                "raw_json": label_raw_json,
            },
            "diagnostic_decision": decision_document,
            "candidate_validation": outcome_document,
        },
    }


def _run_source_references(run_root: Path) -> list[dict]:
    fixed = (
        "source-state.json",
        "control_manifest.json",
        "state.json",
        "state_commit.json",
        "diagnosis.json",
        "diagnosis_context.json",
        "baseline/observation.json",
        "active_diagnosis/analysis_contract.json",
        "active_diagnosis/global_scan.json",
        "active_diagnosis/epoch.json",
        "active_diagnosis/evidence_catalog.json",
        "active_diagnosis/execution_map.json",
        "active_diagnosis/performance_model.json",
        "active_diagnosis/action_catalog.json",
        "active_diagnosis/selection_policy.json",
        "active_diagnosis/request_history.json",
        "active_diagnosis/completed_action_ids.json",
        "active_diagnosis/knowledge_context.json",
        "active_diagnosis/hypothesis_result.json",
        "active_diagnosis/request_set.json",
        "active_diagnosis/evidence_selection.json",
        "active_diagnosis/decision.json",
        "active_diagnosis/investment_brief.json",
    )
    paths = [run_root / relative for relative in fixed]
    for directory in (
        run_root / "readiness",
        run_root / "probes",
        run_root / "state_generations",
        run_root / "active_diagnosis" / "ledger",
        run_root / "active_diagnosis" / "evidence",
    ):
        if directory.is_dir() and not directory.is_symlink():
            paths.extend(directory.rglob("*.json"))
    records = []
    seen = set()
    for path in sorted(paths):
        if path in seen:
            continue
        seen.add(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"Controller scoreable source is missing: {path.relative_to(run_root)}"
            )
        records.append(
            {
                "relative_path": path.relative_to(run_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "locator": "whole_file",
            }
        )
    return records


def _pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _timing_provenance(
    value: object,
    *,
    relative_path: str,
    source_sha256: str,
    target_root: str,
) -> list[dict]:
    records = []

    def visit(current: object, parts: list[object]) -> None:
        if type(current) is dict:
            for key in sorted(current):
                visit(current[key], [*parts, key])
            return
        if type(current) is list:
            for index, item in enumerate(current):
                visit(item, [*parts, index])
            return
        if not parts or type(current) not in {int, float}:
            return
        field = str(parts[-1])
        if not (field.endswith("_us") or field.endswith("_seconds")):
            return
        pointer = "/" + "/".join(_pointer_token(part) for part in parts)
        records.append(
            {
                "target": f"{target_root}{pointer}",
                "relative_path": relative_path,
                "source_sha256": source_sha256,
                "locator": pointer,
                "source_value": current,
                "source_unit": "us" if field.endswith("_us") else "seconds",
                "transform": "identity",
            }
        )

    visit(value, [])
    return records


def _active_evidence_envelopes(
    controller,
    run_root: Path,
    context: dict,
    action_catalog: dict,
    contract: dict,
) -> list[dict]:
    by_action = {item["action_id"]: item for item in action_catalog["actions"]}
    contract_actions = {item["action_id"]: item for item in contract["actions"]}
    envelopes = []
    for summary in context.get("evidence_results", []):
        result = controller.load_json_object(run_root / summary["result_path"])
        action_id = summary["action_id"]
        action = by_action[action_id]
        contract_action = contract_actions[action_id]
        envelopes.append(
            {
                "action_id": action_id,
                "evidence_kind": action["evidence_kind"],
                "adapter_implementation_sha256": contract_action["adapter_sha256"],
                "result_sha256": summary["result_sha256"],
                "status": summary["status"],
                "observations": copy.deepcopy(result["observations"]),
            }
        )
    return envelopes


def extract_scoreable_controller_case(run_dir: Path, label_path: Path) -> dict:
    """Extract one live, Controller-sealed decision without consulting its label."""
    run_root = run_dir.expanduser().resolve(strict=False)
    controller_source_identity = _load_controller_source_identity(run_root)
    controller = _load_module(
        "workload_controller.py",
        "cuda_optimizer_controller_replay_extract",
    )
    state = controller.read_run_state(run_root)
    control = controller._load_frozen_control(run_root, state)
    (
        context,
        epoch,
        execution_map,
        evidence_catalog,
        action_catalog,
        selection_policy,
    ) = controller._load_active_diagnosis_context(
        control,
        run_root,
        state,
        verify_current_project_surface=False,
    )
    raw_decision = controller.load_json_object(
        run_root / "active_diagnosis" / "decision.json"
    )
    decision = controller._load_bound_diagnostic_artifacts(
        run_root,
        state,
        expected_decision=raw_decision.get("decision"),
    )
    contract = controller._load_frozen_analysis_contract(run_root, state)
    performance_model = controller.load_json_object(
        run_root / "active_diagnosis" / "performance_model.json"
    )
    ready = sorted(selection_policy["available_capability_ids"])
    contract_ids = sorted(item["action_id"] for item in contract["actions"])
    available = sorted(
        item["action_id"]
        for item in action_catalog["actions"]
        if item.get("control_scope") == "read_only"
        and item["action_id"] in contract_ids
        and set(item.get("required_capability_ids", [])).issubset(ready)
    )
    read_only_actions = sorted(
        (
            copy.deepcopy(item)
            for item in action_catalog["actions"]
            if item["action_id"] in available
        ),
        key=lambda item: item["action_id"],
    )
    active_results = _active_evidence_envelopes(
        controller,
        run_root,
        context,
        action_catalog,
        contract,
    )
    frozen = {
        "knowledge_identity": copy.deepcopy(context["knowledge_identity"]),
        "diagnosis": controller.load_json_object(run_root / "diagnosis.json"),
        "analysis_epoch": copy.deepcopy(epoch),
        "evidence_catalog": copy.deepcopy(evidence_catalog),
        "execution_map": copy.deepcopy(execution_map),
        "performance_model": copy.deepcopy(performance_model),
        "diagnostic_evidence": [],
        "active_evidence_results": active_results,
        "requested_claim": context["requested_claim"],
        "ready_capability_ids": ready,
        "contract_action_ids": contract_ids,
        "available_actions": available,
        "closed_mechanism_keys": copy.deepcopy(
            context.get("closed_mechanism_keys", [])
        ),
        "candidate_history": copy.deepcopy(context.get("candidate_history", [])),
    }
    rebuilt_context = controller._rebuild_knowledge_context(
        run_root,
        context,
        contract,
        epoch,
        execution_map,
        evidence_catalog,
        selection_policy,
        performance_model,
    )
    knowledge = controller._load_diagnostic_knowledge_module().build_knowledge_context(
        frozen,
        limit=3,
    )
    if rebuilt_context != knowledge or knowledge != context["knowledge_context"]:
        raise ValueError("Controller frozen knowledge context does not replay exactly")

    label = _load_machine_mapped_label(run_root, label_path)
    references = _run_source_references(run_root)
    map_ref = next(
        item
        for item in references
        if item["relative_path"] == "active_diagnosis/execution_map.json"
    )
    model_ref = next(
        item
        for item in references
        if item["relative_path"] == "active_diagnosis/performance_model.json"
    )
    timing = [
        *_timing_provenance(
            execution_map,
            relative_path=map_ref["relative_path"],
            source_sha256=map_ref["sha256"],
            target_root="input_snapshot.execution_map",
        ),
        *_timing_provenance(
            performance_model,
            relative_path=model_ref["relative_path"],
            source_sha256=model_ref["sha256"],
            target_root="input_snapshot.performance_model",
        ),
    ]
    case = {
        "case_id": label.pop("case_id"),
        "scoring_group": "triton",
        "replay_eligibility": {
            "status": "scoreable",
            "reason_codes": [],
            "timing_provenance": timing,
        },
        "input_snapshot": {
            "archive_identity_facts": {
                "status": "complete",
                "archive_case_directory": "controller-run",
                "source_manifest_sha256": canonical_sha256(references),
                "controller_source_identity": controller_source_identity,
                "unknown_fields": [],
            },
            **frozen,
            "read_only_actions": read_only_actions,
            "evidence_summaries": references,
        },
        "controller_decision": copy.deepcopy(decision),
        "v1_2_card_to_mechanism": copy.deepcopy(_V1_2_CARD_TO_MECHANISM),
        "label": label,
    }
    validate_scoreable_case(case, verify_source_commit=False)
    return case


def _embedded_label_document(value: object, label: str) -> tuple[dict, dict]:
    if type(value) is not dict or set(value) != {
        "relative_path",
        "source_sha256",
        "raw_json",
    }:
        raise ValueError(f"scoreable label evidence {label} is not closed")
    relative_path = value["relative_path"]
    path = Path(relative_path) if type(relative_path) is str else None
    if (
        path is None
        or not relative_path
        or path.is_absolute()
        or ".." in path.parts
        or type(value["source_sha256"]) is not str
        or _SHA256.fullmatch(value["source_sha256"]) is None
        or type(value["raw_json"]) is not str
        or hashlib.sha256(value["raw_json"].encode("utf-8")).hexdigest()
        != value["source_sha256"]
    ):
        raise ValueError(f"scoreable label evidence {label} is invalid")
    try:
        document = json.loads(value["raw_json"])
    except json.JSONDecodeError as error:
        raise ValueError(
            f"scoreable label evidence {label} is invalid JSON"
        ) from error
    if type(document) is not dict:
        raise ValueError(f"scoreable label evidence {label} is not an object")
    return document, copy.deepcopy(value)


def _validate_scoreable_label(case: dict, label: object) -> None:
    label_fields = {
        "promoted_mechanism_keys",
        "cheapest_valid_action_ids",
        "observed_diagnostic_decision",
        "candidate_outcome",
        "source_documents",
    }
    if type(label) is not dict or set(label) != label_fields:
        raise ValueError("scoreable label fields are not closed")
    promoted = _canonical_strings(
        label["promoted_mechanism_keys"],
        "scoreable label promoted_mechanism_keys",
    )
    actions = _canonical_strings(
        label["cheapest_valid_action_ids"],
        "scoreable label cheapest_valid_action_ids",
    )
    observed_decision = label["observed_diagnostic_decision"]
    if observed_decision not in _TERMINAL_DECISIONS:
        raise ValueError("scoreable observed diagnostic decision is unsupported")
    sources = label["source_documents"]
    if type(sources) is not dict or set(sources) != {
        "label_manifest",
        "diagnostic_decision",
        "candidate_validation",
    }:
        raise ValueError("scoreable label evidence sources are not closed")
    manifest, manifest_source = _embedded_label_document(
        sources["label_manifest"],
        "label_manifest",
    )
    decision, decision_source = _embedded_label_document(
        sources["diagnostic_decision"],
        "diagnostic_decision",
    )
    outcome, outcome_source = _embedded_label_document(
        sources["candidate_validation"],
        "candidate_validation",
    )
    expected_manifest_fields = {
        "schema_version",
        "case_id",
        "diagnostic_decision_source",
        "candidate_validation_source",
    }
    if (
        set(manifest) != expected_manifest_fields
        or manifest["schema_version"]
        != "cuda-optimizer/knowledge-replay-label-v2"
        or manifest["case_id"] != case.get("case_id")
        or manifest["diagnostic_decision_source"]
        != {
            "relative_path": decision_source["relative_path"],
            "source_sha256": decision_source["source_sha256"],
        }
        or manifest["candidate_validation_source"]
        != {
            "relative_path": outcome_source["relative_path"],
            "source_sha256": outcome_source["source_sha256"],
        }
        or manifest_source["relative_path"]
        in {
            decision_source["relative_path"],
            outcome_source["relative_path"],
        }
    ):
        raise ValueError("scoreable label evidence manifest does not close sources")
    if (
        decision_source["relative_path"] != "active_diagnosis/decision.json"
        or decision.get("decision") != observed_decision
        or decision != case.get("controller_decision")
    ):
        raise ValueError("scoreable diagnostic decision label evidence drifted")
    input_decision_sources = [
        source
        for source in case["input_snapshot"]["evidence_summaries"]
        if source.get("relative_path") == decision_source["relative_path"]
    ]
    if (
        len(input_decision_sources) != 1
        or input_decision_sources[0].get("sha256")
        != decision_source["source_sha256"]
    ):
        raise ValueError(
            "scoreable diagnostic decision label evidence is not input-bound"
        )
    if (
        outcome_source["relative_path"] != "validation/outcome.json"
        or _canonical_strings(
            outcome.get("accepted_mechanism_keys"),
            "candidate validation accepted_mechanism_keys",
        )
        != promoted
        or _canonical_strings(
            outcome.get("cheapest_valid_action_ids"),
            "candidate validation cheapest_valid_action_ids",
        )
        != actions
        or _candidate_outcome_from_validation(outcome)
        != label["candidate_outcome"]
    ):
        raise ValueError("scoreable candidate label evidence drifted")


def _validate_package_regression_label(label: object) -> None:
    label_fields = {
        "accepted_mechanism_keys",
        "cheapest_valid_action_ids",
        "expected_terminal_decisions",
        "field_sources",
        "label_source_sha256",
    }
    if type(label) is not dict or set(label) != label_fields:
        raise ValueError("package-regression label fields are not closed")
    for field in (
        "accepted_mechanism_keys",
        "cheapest_valid_action_ids",
        "expected_terminal_decisions",
    ):
        _canonical_strings(label[field], f"package-regression label {field}")
    if not set(label["expected_terminal_decisions"]).issubset(
        _TERMINAL_DECISIONS
    ):
        raise ValueError("package-regression label decision is unsupported")
    if (
        type(label["label_source_sha256"]) is not str
        or _SHA256.fullmatch(label["label_source_sha256"]) is None
    ):
        raise ValueError("package-regression label_source_sha256 is invalid")
    for source in label["field_sources"].values():
        if (
            type(source) is not dict
            or set(source) != {"relative_path", "source_sha256", "locator"}
            or type(source["source_sha256"]) is not str
            or _SHA256.fullmatch(source["source_sha256"]) is None
        ):
            raise ValueError("package-regression label field source is invalid")


def _validate_controller_replay_case(
    case: dict,
    *,
    eligibility_status: str,
    verify_source_commit: bool = False,
) -> None:
    if type(case) is not dict or case.get("scoring_group") != "triton":
        raise ValueError("scoreable case must belong to the Triton scoring group")
    eligibility = case.get("replay_eligibility")
    if (
        type(eligibility) is not dict
        or eligibility.get("status") != eligibility_status
        or eligibility.get("reason_codes") != []
        or type(eligibility.get("timing_provenance")) is not list
        or not eligibility["timing_provenance"]
    ):
        raise ValueError("scoreable case replay eligibility is incomplete")
    snapshot = case.get("input_snapshot")
    required = {
        "archive_identity_facts",
        "diagnosis",
        "read_only_actions",
        "evidence_summaries",
        "knowledge_identity",
        "analysis_epoch",
        "evidence_catalog",
        "execution_map",
        "performance_model",
        "diagnostic_evidence",
        "active_evidence_results",
        "requested_claim",
        "ready_capability_ids",
        "contract_action_ids",
        "available_actions",
        "closed_mechanism_keys",
        "candidate_history",
    }
    if type(snapshot) is not dict or set(snapshot) != required:
        raise ValueError("scoreable input_snapshot fields are not closed")
    forbidden = {
        "accepted_mechanism_keys",
        "cheapest_valid_action_ids",
        "expected_terminal_decisions",
        "label_source_sha256",
        "promoted_mechanism_keys",
        "observed_diagnostic_decision",
        "candidate_outcome",
        "source_documents",
    }

    def keys(value: object) -> set[str]:
        if type(value) is dict:
            return set(value).union(*(keys(item) for item in value.values()))
        if type(value) is list:
            return set().union(*(keys(item) for item in value))
        return set()

    if forbidden & keys(snapshot):
        raise ValueError("future labels leaked into scoreable input_snapshot")
    frozen_fields = {
        "knowledge_identity",
        "diagnosis",
        "analysis_epoch",
        "evidence_catalog",
        "execution_map",
        "performance_model",
        "diagnostic_evidence",
        "active_evidence_results",
        "requested_claim",
        "ready_capability_ids",
        "contract_action_ids",
        "available_actions",
        "closed_mechanism_keys",
        "candidate_history",
    }
    frozen = {field: copy.deepcopy(snapshot[field]) for field in frozen_fields}
    knowledge = _load_module(
        "diagnostic_knowledge.py",
        "cuda_optimizer_knowledge_replay_validate",
    )
    knowledge.build_knowledge_context(frozen, limit=3)
    if [
        item["action_id"] for item in snapshot["read_only_actions"]
    ] != snapshot["available_actions"]:
        raise ValueError("read_only_actions do not match available_actions")
    if eligibility_status == "scoreable":
        _validate_scoreable_label(case, case.get("label"))
    else:
        _validate_package_regression_label(case.get("label"))
    source_by_path = {}
    for source in snapshot["evidence_summaries"]:
        relative_path = (
            Path(source["relative_path"])
            if type(source) is dict
            and type(source.get("relative_path")) is str
            else None
        )
        if (
            type(source) is not dict
            or set(source) != {"relative_path", "sha256", "locator"}
            or relative_path is None
            or not source["relative_path"]
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or type(source["sha256"]) is not str
            or _SHA256.fullmatch(source["sha256"]) is None
            or source["locator"] != "whole_file"
            or source["relative_path"] in source_by_path
        ):
            raise ValueError("scoreable evidence summary is invalid")
        source_by_path[source["relative_path"]] = source
    if snapshot["archive_identity_facts"].get(
        "source_manifest_sha256"
    ) != canonical_sha256(snapshot["evidence_summaries"]):
        raise ValueError("scoreable source manifest digest drifted")
    source_identity = snapshot["archive_identity_facts"].get(
        "controller_source_identity"
    )
    if eligibility_status == "scoreable" or source_identity is not None:
        try:
            _validate_controller_source_identity(
                source_identity,
                verify_commit=verify_source_commit,
            )
        except ValueError as error:
            raise ValueError(
                f"{eligibility_status} controller source identity drifted: {error}"
            ) from error
    for record in eligibility["timing_provenance"]:
        if type(record) is not dict or set(record) != {
            "target",
            "relative_path",
            "source_sha256",
            "locator",
            "source_value",
            "source_unit",
            "transform",
        }:
            raise ValueError("scoreable timing provenance is not closed")
        source = source_by_path.get(record["relative_path"])
        if (
            source is None
            or source["sha256"] != record["source_sha256"]
            or record["transform"] != "identity"
            or type(record["source_value"]) not in {int, float}
            or not math.isfinite(record["source_value"])
        ):
            raise ValueError("scoreable timing provenance source is not sealed")
        target_root, separator, target_pointer = record["target"].partition("/")
        if separator != "/" or not target_root.startswith("input_snapshot."):
            raise ValueError("scoreable timing provenance target is invalid")
        field = target_pointer.rsplit("/", 1)[-1]
        expected_unit = (
            "us"
            if field.endswith("_us")
            else "seconds"
            if field.endswith("_seconds")
            else None
        )
        if (
            record["locator"] != f"/{target_pointer}"
            or record["source_unit"] != expected_unit
        ):
            raise ValueError("scoreable timing provenance locator is invalid")
        target_value: object = case
        for token in target_root.split("."):
            if type(target_value) is not dict or token not in target_value:
                raise ValueError("scoreable timing provenance target is missing")
            target_value = target_value[token]
        resolved_value = _json_pointer(
            target_value,
            f"/{target_pointer}",
            "scoreable timing provenance target",
        )
        if resolved_value != record["source_value"]:
            raise ValueError("scoreable timing provenance value drifted")


def validate_scoreable_case(
    case: dict,
    *,
    verify_source_commit: bool = True,
) -> None:
    _validate_controller_replay_case(
        case,
        eligibility_status="scoreable",
        verify_source_commit=verify_source_commit,
    )


def validate_package_regression_case(case: dict) -> None:
    _validate_controller_replay_case(
        case,
        eligibility_status="package_regression",
    )


def _build_scoreable_v1_2_baseline(case: dict) -> dict:
    validate_scoreable_case(case)
    snapshot = case["input_snapshot"]
    route = _route_v1_2_cards(
        snapshot["diagnosis"],
        snapshot["execution_map"],
        limit=3,
    )
    mapping = case["v1_2_card_to_mechanism"]
    ranked = []
    action_ids = []
    for card in route["cards"]:
        card_id = card["id"]
        if card_id not in mapping:
            raise ValueError(f"V1.2 card has no explicit mechanism mapping: {card_id}")
        ranked.append(mapping[card_id])
        action_id = card["preferred_actions"][0]
        if action_id not in action_ids:
            action_ids.append(action_id)
    catalog = _load_v1_2_router_snapshot()
    action_by_id = {item["action_id"]: item for item in catalog["actions"]}
    timings = snapshot["performance_model"]["action_timing_estimates"]
    next_actions = []
    for action_id in action_ids:
        action = action_by_id[action_id]
        estimate = timings.get(action_id)
        next_actions.append(
            {
                "action_id": action_id,
                "cost_class": action["cost"],
                "is_profiler": action["evidence_kind"] != "compiler_sass",
                "measured_seconds": None if estimate is None else estimate["p50_seconds"],
            }
        )
    return {
        "ranked_mechanism_keys": ranked,
        "next_actions": next_actions,
        "diagnostic_terminal_decision": {
            "status": "unavailable",
            "reason": "v1_2_controller_terminal_not_replayed",
        },
        "promotion_authority": route["promotion_authority"],
        "valid_for_ranking_scoring": True,
        "valid_for_action_id_scoring": True,
        "valid_for_measured_cost_scoring": False,
        "valid_for_terminal_scoring": False,
        "route_output_sha256": canonical_sha256(route),
    }


def _reference(root: Path, relative_path: str) -> dict:
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"required archive evidence is missing: {path}")
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "locator": "whole_file",
    }


def _references(root: Path, paths: tuple[str, ...]) -> list[dict]:
    records = [_reference(root, path) for path in paths]
    hashes = [record["sha256"] for record in records]
    if len(hashes) != len(set(hashes)) or any(value == "0" * 64 for value in hashes):
        raise ValueError("archive evidence has duplicate or placeholder SHA-256")
    return records


def _case(
    case_id: str,
    directory: Path,
    inputs: tuple[str, ...],
    labels: tuple[str, ...],
    reason_codes: list[str],
    diagnosis_text: str,
    outcome_text: str,
    *,
    scoring_group: str = "triton",
    archive_case_directory: str | None = None,
) -> dict:
    input_refs = _references(directory, inputs)
    label_refs = _references(directory, labels)
    return {
        "case_id": case_id,
        "scoring_group": scoring_group,
        "replay_eligibility": {
            "status": "partial" if scoring_group == "triton" else "rejection_only",
            "reason_codes": reason_codes,
            "timing_provenance": [],
        },
        "input_snapshot": {
            "archive_identity_facts": {
                "status": "incomplete",
                "archive_case_directory": archive_case_directory or directory.name,
                "source_manifest_sha256": canonical_sha256(input_refs),
                "unknown_fields": [
                    "knowledge_identity",
                    "analysis_epoch",
                    "execution_map",
                    "performance_model",
                ],
            },
            "diagnosis": {
                "status": "unavailable_for_runtime_replay",
                "authority": "none",
                "source_refs": input_refs,
                "note": diagnosis_text,
            },
            "read_only_actions": [
                {
                    "action_id": f"archive-protocol-{case_id.lower()}",
                    "kind": "archived_protocol_reference",
                    "availability": "historical_only",
                    "authority": "none",
                    "source_refs": input_refs,
                }
            ],
            "evidence_summaries": input_refs,
        },
        "label": {
            "historical_outcome": {
                "authority": "archived_only",
                "source_refs": label_refs,
                "note": outcome_text,
            },
            "label_source_sha256": canonical_sha256(label_refs),
        },
    }


def _extract_r01(root: Path) -> dict:
    return _case(
        "R01",
        root / "iter_156_nms_fp32_output",
        (
            "rewrite_manifest.json",
            "run_candidate_correctness.sh",
            "run_nsys.sh",
            "nsys_fp32_output_1000/mechanism_analysis.json",
        ),
        (
            "correctness_output_invariants.json",
            "correctness_semantic_iter149_vs_candidate.json",
            "correctness.sha256",
            "timing_confirmation_vs_iter149/analysis.json",
            "timing_confirmation_vs_iter149/evidence.sha256",
        ),
        [
            "aggregate_timing_only",
            "missing_execution_window",
            "missing_node_boundaries",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Later archive artifacts remain historical-only.",
    )


def _extract_r02(root: Path) -> dict:
    return _case(
        "R02",
        root / "iter_173_pdl_gap_audit",
        ("analysis.json", "run_nsys_v3.sh", "run_correctness_v3.sh"),
        (
            "correctness_v3.semantic_vs_iter161.json",
            "nsys_v3/gap_analysis.json",
            "correctness_v4.semantic_vs_iter161.json",
            "nsys_v4/gap_analysis.json",
            "correctness_v5.semantic_vs_iter161.json",
            "nsys_v5/gap_analysis.json",
        ),
        [
            "aggregate_timing_only",
            "missing_execution_window",
            "missing_node_boundaries",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Variant correctness and gap results remain historical-only.",
    )


def _extract_r03(root: Path) -> dict:
    return _case(
        "R03",
        root / "iter_182_stack_upgrade_iter161",
        ("DESIGN.md", "layer_analysis.json"),
        (
            "correctness_attempt3/analysis.json",
            "exact_attribution_primary/analysis.json",
            "exact_attribution_primary/evidence.sha256",
            "exact_deployment_primary/analysis.json",
            "exact_deployment_primary/evidence.sha256",
            "closure.sha256",
            "DECISION.md",
        ),
        [
            "missing_predecision_timing",
            "missing_execution_window",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Correctness closure remains historical-only.",
    )


def _extract_r04(root: Path) -> dict:
    return _case(
        "R04",
        root / "iter_184_fastsort_map_fused",
        ("DESIGN.md", "run_candidate_correctness.sh", "run_nsys_mechanism_gate.sh"),
        (
            "nsys_fastsort_map_1000/analysis.json",
            "DECISION.md",
            "closure.sha256",
        ),
        [
            "aggregate_timing_only",
            "missing_execution_window",
            "missing_node_boundaries",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Mechanism-gate closure remains historical-only.",
    )


def _extract_r05(root: Path) -> dict:
    return _case(
        "R05",
        root / "iter_186_iter161_vs_original_endpoint",
        ("DESIGN.md", "run_correctness.sh", "run_endpoint.sh"),
        (
            "results/correctness_decision.json",
            "results/fixed_analysis.json",
            "results/tail_analysis.json",
            "closure.sha256",
            "DECISION.md",
        ),
        [
            "missing_predecision_timing",
            "missing_execution_window",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Endpoint conclusion remains historical-only.",
    )


def _extract_r06(root: Path) -> dict:
    return _case(
        "R06",
        root / "iter_187_stack_upgrade_iter161_adapted",
        (
            "DESIGN.md",
            "run_correctness_screen.sh",
            "run_exact_forced2.sh",
            "run_endpoint_versions.sh",
        ),
        (
            "correctness_screen_attempt2/analysis.json",
            "exact_forced2_primary/analysis.json",
            "exact_forced2_primary/evidence.sha256",
            "endpoint_versions_primary/analysis.json",
            "endpoint_versions_primary/evidence.sha256",
            "closure.sha256",
            "DECISION.md",
        ),
        [
            "historical_delta_not_execution_interval",
            "label_timing_excluded",
            "missing_execution_window",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Migration conclusion remains historical-only.",
    )


_POST_AUDIT_REASON_CODES = [
    "missing_controller_epoch",
    "missing_knowledge_identity",
    "missing_controller_execution_map",
    "missing_controller_performance_model",
    "label_not_machine_mapped",
]


def _post_audit_partial_cases(root: Path) -> list[dict]:
    """Retain useful evidence found on 5090 without inventing Controller state."""
    specs = (
        (
            "R07",
            (
                "run_manifest.json",
                "preflight_check.json",
                "iter_149_decode_into_nms/nsys_candidate_1000/candidate.sqlite",
                "iter_149_decode_into_nms/nsys_candidate_1000/inputs.sha256",
                "iter_149_decode_into_nms/nsys_candidate_1000/evidence.sha256",
                "iter_156_nms_fp32_output/build/sources.sha256",
                "iter_156_nms_fp32_output/correctness_output_invariants.json",
                "iter_156_nms_fp32_output/correctness_semantic_iter149_vs_candidate.json",
                "iter_156_nms_fp32_output/correctness.sha256",
                "iter_156_nms_fp32_output/run_timing_vs_iter135.sh",
            ),
            (
                "iter_156_nms_fp32_output/timing_vs_iter149/analysis.json",
                "iter_156_nms_fp32_output/timing_confirmation_vs_iter149/analysis.json",
            ),
            "Iter149 and Iter156 artifacts contain useful kernel evidence but no Controller-sealed diagnosis.",
            "The Iter156 paired result remains historical-only.",
        ),
        (
            "R08",
            (
                "run_manifest.json",
                "preflight_check.json",
                "iter_156_nms_fp32_output/nsys_fp32_output_1000/candidate.sqlite",
                "iter_156_nms_fp32_output/nsys_fp32_output_1000/inputs.sha256",
                "iter_156_nms_fp32_output/nsys_fp32_output_1000/evidence.sha256",
                "iter_160_fastsort_store4/build/sources.sha256",
                "iter_160_fastsort_store4/correctness_output_invariants.json",
                "iter_160_fastsort_store4/correctness_semantic_iter156_vs_candidate.json",
                "iter_160_fastsort_store4/correctness.sha256",
                "iter_160_fastsort_store4/run_timing_vs_iter156.sh",
            ),
            (
                "iter_160_fastsort_store4/timing_vs_iter156/analysis.json",
                "iter_160_fastsort_store4/timing_confirmation_vs_iter156/analysis.json",
            ),
            "Iter156 and Iter160 artifacts contain useful kernel evidence but no Controller-sealed diagnosis.",
            "The Iter160 paired result remains historical-only.",
        ),
        (
            "R09",
            (
                "run_manifest.json",
                "preflight_check.json",
                "iter_160_fastsort_store4/nsys_fastsort_store4_1000/candidate.sqlite",
                "iter_160_fastsort_store4/nsys_fastsort_store4_1000/inputs.sha256",
                "iter_160_fastsort_store4/nsys_fastsort_store4_1000/evidence.sha256",
                "iter_161_group_reserve4/build/sources.sha256",
                "iter_161_group_reserve4/correctness_output_invariants.json",
                "iter_161_group_reserve4/correctness_semantic_iter160_vs_candidate.json",
                "iter_161_group_reserve4/correctness.sha256",
                "iter_161_group_reserve4/run_timing_vs_iter156.sh",
            ),
            (
                "iter_161_group_reserve4/timing_vs_iter160/analysis.json",
                "iter_161_group_reserve4/timing_confirmation_vs_iter160/analysis.json",
            ),
            "Iter160 and Iter161 artifacts contain useful kernel evidence but no Controller-sealed diagnosis.",
            "The Iter161 paired result remains historical-only.",
        ),
        (
            "R10",
            (
                "run_manifest.json",
                "preflight_check.json",
                "iter_169_aux_stream_build_gate/candidate.sha256",
                "iter_169_aux_stream_build_gate/analysis.json",
                "iter_169_aux_stream_build_gate/run_nsys_gate.sh",
                "iter_169_aux_stream_build_gate/nsys_overlap_gate/inputs.sha256",
                "iter_169_aux_stream_build_gate/nsys_overlap_gate/candidate/candidate.sqlite",
                "iter_169_aux_stream_build_gate/nsys_overlap_gate/analysis.json",
                "iter_169_aux_stream_build_gate/nsys_overlap_gate/evidence.sha256",
            ),
            ("iter_169_aux_stream_build_gate/DECISION.md",),
            "Iter169 contains a real overlap gate but no Controller-sealed diagnosis or execution map.",
            "The closed-direction decision remains historical-only.",
        ),
        (
            "R11",
            (
                "run_manifest.json",
                "preflight_check.json",
                "iter_181_persistent_counters/build_candidate/sources.sha256",
                "iter_181_persistent_counters/correctness.sha256",
                "iter_181_persistent_counters/correctness_repeat1/evidence.sha256",
                "iter_181_persistent_counters/nsys_paired_gate/02_B/profile.sqlite",
                "iter_181_persistent_counters/nsys_paired_gate/analysis.json",
                "iter_181_persistent_counters/nsys_paired_gate/inputs.sha256",
                "iter_181_persistent_counters/nsys_paired_gate/evidence.sha256",
                "iter_181_persistent_counters/run_timing_vs_iter161.sh",
            ),
            (
                "iter_181_persistent_counters/timing_vs_iter161/analysis.json",
                "iter_181_persistent_counters/RUN_RESULT.md",
                "iter_181_persistent_counters/closure.sha256",
            ),
            "Iter181 contains correctness and paired Nsys evidence but no Controller-sealed diagnosis.",
            "The formal timing and closure remain historical-only.",
        ),
        (
            "R12",
            (
                "run_manifest.json",
                "preflight_check.json",
                "iter_184_fastsort_map_fused/DESIGN.md",
                "iter_184_fastsort_map_fused/build_candidate/sources.sha256",
                "iter_184_fastsort_map_fused/correctness.sha256",
                "iter_184_fastsort_map_fused/correctness_repeat1/evidence.sha256",
                "iter_184_fastsort_map_fused/nsys_fastsort_map_1000/candidate.sqlite",
                "iter_184_fastsort_map_fused/nsys_fastsort_map_1000/analysis.json",
                "iter_184_fastsort_map_fused/nsys_fastsort_map_1000/evidence.sha256",
            ),
            (
                "iter_184_fastsort_map_fused/DECISION.md",
                "iter_184_fastsort_map_fused/closure.sha256",
            ),
            "Iter184 contains correctness and Nsys evidence but no Controller-sealed diagnosis.",
            "The sub-threshold closure remains historical-only.",
        ),
    )
    return [
        _case(
            case_id,
            root,
            inputs,
            labels,
            list(_POST_AUDIT_REASON_CODES),
            diagnosis_text,
            outcome_text,
            archive_case_directory="loop30",
        )
        for case_id, inputs, labels, diagnosis_text, outcome_text in specs
    ]


def _extract_x01(root: Path) -> dict:
    return _case(
        "X01",
        root / "iter_188_full_ensemble_original_vs_iter161",
        (
            "DESIGN.md",
            "readiness_input.json",
            "run_correctness.sh",
            "run_original_full_ensemble_perf.sh",
        ),
        (
            "correctness_attempt3_three_repeat/analysis.json",
            "correctness_attempt3_three_repeat/closure.sha256",
            "performance_attempt1_original_script/analysis.json",
            "performance_attempt1_original_script/closure.sha256",
            "closure.sha256",
            "RESULTS.md",
        ),
        [
            "correctness_gate_failed",
            "performance_result_not_promotable",
        ],
        "The complete ensemble gate is not available as a scoreable direction replay.",
        "Correctness and performance artifacts are retained only as rejection evidence.",
        scoring_group="rejection_only",
    )


def build_suite(root: Path) -> dict:
    cases = [
        extractor(root)
        for extractor in (
            _extract_r01,
            _extract_r02,
            _extract_r03,
            _extract_r04,
            _extract_r05,
            _extract_r06,
        )
    ]
    cases.extend(_post_audit_partial_cases(root))
    for case_id, relative_path, digest in (
        (
            "K01",
            "KernelBench/level1/19_ReLU.py",
            "cbbfc9409662168ee7a5d3e7f7a59bf56e0faf9d763197e7f6a41fb5942dd63a",
        ),
        (
            "K02",
            "KernelBench/level1/92_cumsum_exclusive.py",
            "ec1551c196130f5d7fae707f0750016c35b988e76aa3f1657da4347f463ced86",
        ),
    ):
        cases.append(
            {
                "case_id": case_id,
                "scoring_group": "public_kernel",
                "replay_eligibility": {
                    "status": "protocol_only",
                    "reason_codes": ["candidate_labels_not_frozen"],
                    "timing_provenance": [],
                },
                "input_snapshot": {
                    "repository": "ScalingIntelligence/KernelBench",
                    "commit": "423217d9fda91e0c2d67e4a43bf62f96f6d104f1",
                    "relative_path": relative_path,
                    "source_sha256": digest,
                },
                "label": {"label_status": "protocol_only"},
            }
        )
    counterexamples = {
        "counterexample-version-mismatch": {
            "expected_behavior": "explanation_only",
        },
        "counterexample-missing-evidence": {
            "expected_decisions": ["MEASURE"],
        },
        "counterexample-duplicate-mechanism": {
            "expected_candidate_count": 1,
        },
        "counterexample-unstable-benchmark": {
            "expected_decisions": ["REVIEW_REQUIRED", "STOP"],
        },
    }
    for case_id, expected in counterexamples.items():
        cases.append(
            {
                "case_id": case_id,
                "scoring_group": "rejection_only",
                "replay_eligibility": {
                    "status": "rejection_only",
                    "reason_codes": [case_id.removeprefix("counterexample-")],
                    "timing_provenance": [],
                },
                "input_snapshot": {"counterexample": case_id},
                "label": expected,
            }
        )
    cases.append(_extract_x01(root))
    return {
        "schema_version": "cuda-optimizer/knowledge-replay-v1",
        "cases": cases,
        "cases_sha256": canonical_sha256(cases),
    }


def build_baseline(suite: dict) -> dict:
    cases = {}
    for case in suite["cases"]:
        if case["scoring_group"] != "triton":
            continue
        if case["replay_eligibility"]["status"] == "scoreable":
            cases[case["case_id"]] = _build_scoreable_v1_2_baseline(case)
        else:
            cases[case["case_id"]] = {
                "status": "unavailable",
                "reason_codes": case["replay_eligibility"]["reason_codes"],
            }
    return {
        "schema_version": "cuda-optimizer/knowledge-replay-baseline-v1",
        "source_cases_sha256": suite["cases_sha256"],
        "baseline_cases_sha256": canonical_sha256(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-output", required=True, type=Path)
    args = parser.parse_args()
    suite = build_suite(args.archive_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    args.baseline_output.write_text(json.dumps(build_baseline(suite), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
