#!/usr/bin/env python3
"""Turn validated diagnosis artifacts into one bounded next decision."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any


DECISION_SCHEMA = "cuda-optimizer/diagnostic-decision-v1"
DECISIONS = {"MEASURE", "PURSUE", "REVIEW_REQUIRED", "STOP"}
_AUTHORIZATION_REASONS = {
    "cost_exceeds_policy",
    "perturbation_exceeds_policy",
    "risk_exceeds_policy",
    "required_capability_unavailable",
}


class ValidationError(ValueError):
    """Raised when decision inputs are not bound to one diagnosis epoch."""


def _object(value: Any, field: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field} must be an object")
    return dict(value)


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValidationError(f"{field} must be finite")
    result = float(value)
    if result < 0 or (positive and result <= 0):
        raise ValidationError(f"{field} must be {'positive' if positive else 'non-negative'}")
    return result


def _active_hypotheses(result: Mapping[str, Any]) -> list[dict]:
    root = _object(result, "hypothesis_result")
    hypothesis_set = _object(root.get("hypothesis_set"), "hypothesis_set")
    hypotheses = hypothesis_set.get("hypotheses")
    active_ids = root.get("active_hypothesis_ids")
    if type(hypotheses) is not list or type(active_ids) is not list:
        raise ValidationError("hypothesis_result is incomplete")
    active = [item for item in hypotheses if item.get("disposition") == "active"]
    if sorted(item.get("hypothesis_id") for item in active) != sorted(active_ids):
        raise ValidationError("hypothesis active ids do not match admitted content")
    if len(active) > 3:
        raise ValidationError("diagnostic decision accepts at most three active hypotheses")
    return copy.deepcopy(active)


def _rank_hypotheses(model: Mapping[str, Any], active: list[dict]) -> list[dict]:
    node_ceiling = {
        item["node_id"]: _number(item["benefit_ceiling_us"], "benefit_ceiling_us")
        for item in model.get("node_directions", [])
    }
    window = _number(model.get("window_duration_us"), "window_duration_us", positive=True)
    ranked = []
    for item in active:
        ceiling = min(
            window,
            sum(node_ceiling.get(node_id, 0.0) for node_id in item["scope_node_ids"]),
        )
        ranked.append(
            {
                "hypothesis_id": item["hypothesis_id"],
                "mechanism": item["mechanism"],
                "claim_layer": item["claim_layer"],
                "statement": item["statement"],
                "confidence": item["confidence"],
                "benefit_ceiling_us": ceiling,
                "benefit_ceiling_basis": "scoped_hot_path_active_time_capped_by_window",
                "support_evidence_ids": copy.deepcopy(item["support_evidence_ids"]),
                "oppose_evidence_ids": copy.deepcopy(item["oppose_evidence_ids"]),
                "missing_evidence_kinds": copy.deepcopy(item["missing_evidence_kinds"]),
                "falsification_question": item["falsification_question"],
            }
        )
    confidence_rank = {"direction_supported": 0, "plausible": 1, "inconclusive": 2}
    ranked.sort(
        key=lambda item: (
            -item["benefit_ceiling_us"],
            confidence_rank[item["confidence"]],
            item["hypothesis_id"],
        )
    )
    return ranked


def _cost_for_action(model: Mapping[str, Any], action: Mapping[str, Any] | None) -> dict:
    if action is None:
        return {
            "class": None,
            "p50_seconds": None,
            "p90_seconds": None,
            "basis": "unavailable",
        }
    action_id = action.get("action_id")
    estimates = model.get("action_timing_estimates", {})
    estimate = estimates.get(action_id) if isinstance(estimates, Mapping) else None
    return {
        "class": action.get("cost"),
        "p50_seconds": None if estimate is None else estimate["p50_seconds"],
        "p90_seconds": None if estimate is None else estimate["p90_seconds"],
        "basis": "unavailable" if estimate is None else estimate["basis"],
    }


def _external_summary(value: Mapping[str, Any] | None) -> dict:
    if value is None:
        return {
            "status": "not_requested",
            "providers_requested": [],
            "providers_completed": [],
            "failed_providers": [],
            "total_wait_seconds": 0.0,
            "verdicts": [],
            "advisory_only": True,
        }
    root = _object(value, "external_review")
    verdicts = []
    for item in root.get("reviews", []):
        if not isinstance(item, Mapping):
            continue
        response = item.get("response")
        verdicts.append(
            {
                "provider": item.get("provider"),
                "status": item.get("status"),
                "verdict": response.get("verdict") if isinstance(response, Mapping) else None,
            }
        )
    wait = root.get("total_wait_seconds", 0.0)
    return {
        "status": root.get("status", "unavailable"),
        "providers_requested": copy.deepcopy(root.get("providers_requested", [])),
        "providers_completed": copy.deepcopy(root.get("providers_completed", [])),
        "failed_providers": copy.deepcopy(root.get("failed_providers", [])),
        "total_wait_seconds": _number(wait, "external_review.total_wait_seconds"),
        "verdicts": verdicts,
        "advisory_only": True,
    }


def _selected_action(selection: Mapping[str, Any]) -> tuple[dict, dict] | tuple[None, None]:
    selected = selection.get("selected_request")
    if not isinstance(selected, Mapping):
        return None, None
    action = _object(selected.get("controller_action"), "selected controller_action")
    return copy.deepcopy(dict(selected)), copy.deepcopy(action)


def _blocked_authorized_action(selection: Mapping[str, Any]) -> tuple[dict, dict] | tuple[None, None]:
    candidates = []
    for item in selection.get("rejections", []):
        if not isinstance(item, Mapping) or item.get("reason") not in _AUTHORIZATION_REASONS:
            continue
        action = item.get("controller_action")
        if isinstance(action, Mapping):
            candidates.append((dict(item), dict(action)))
    if not candidates:
        return None, None
    levels = {"none": 0, "low": 1, "medium": 2, "high": 3}
    candidates.sort(
        key=lambda pair: (
            levels.get(pair[1].get("cost"), 99),
            pair[0].get("request_id", ""),
        )
    )
    return candidates[0]


def decide_next_step(
    performance_model: Mapping[str, Any],
    hypothesis_result: Mapping[str, Any],
    evidence_selection: Mapping[str, Any],
    *,
    external_review: Mapping[str, Any] | None = None,
) -> dict:
    """Return one local evidence-authoritative decision and investment brief."""
    model = _object(performance_model, "performance_model")
    if model.get("schema_version") != "cuda-optimizer/performance-model-v1":
        raise ValidationError("performance_model schema is unsupported")
    selection = _object(evidence_selection, "evidence_selection")
    active = _active_hypotheses(hypothesis_result)
    ranked = _rank_hypotheses(model, active)
    primary = ranked[0] if ranked else None
    threshold = _number(model.get("minimum_effect_us"), "minimum_effect_us", positive=True)
    maximum_ceiling = max((item["benefit_ceiling_us"] for item in ranked), default=0.0)

    decision = None
    reason = None
    next_action = None
    action = None
    checkpoint = None
    status = selection.get("status")
    if not active:
        decision, reason, checkpoint = "STOP", "no_active_hypothesis", "terminal"
    elif maximum_ceiling < threshold:
        decision, reason, checkpoint = (
            "STOP",
            "benefit_ceiling_below_minimum_effect",
            "terminal",
        )
    elif status == "selected":
        selected, action = _selected_action(selection)
        if selected is None:
            raise ValidationError("selected evidence result has no selected_request")
        decision, reason, checkpoint = "MEASURE", "discriminating_evidence_required", "after_selected_evidence"
        next_action = {
            "request_id": selected["request_id"],
            "action_id": selected["action_id"],
            "question": selected["question"],
            "target_hypothesis_ids": copy.deepcopy(selected["target_hypothesis_ids"]),
        }
    elif status == "sufficient" and all(
        item["confidence"] == "direction_supported" for item in active
    ):
        decision, reason, checkpoint = "PURSUE", "direction_supported", "after_candidate_screen"
        next_action = {
            "action_id": "implement-candidate",
            "hypothesis_id": primary["hypothesis_id"],
            "mechanism": primary["mechanism"],
            "claim_layer": primary["claim_layer"],
        }
    elif status == "evidence_gap":
        blocked, action = _blocked_authorized_action(selection)
        gap_reason = selection.get("gap_reason")
        if blocked is not None or gap_reason == "profile_budget_exhausted" or selection.get("missing_capability_ids"):
            decision, reason, checkpoint = (
                "REVIEW_REQUIRED",
                "valuable_action_outside_authorization",
                "after_authorization_decision",
            )
            if blocked is not None:
                next_action = {
                    "request_id": blocked["request_id"],
                    "action_id": blocked["action_id"],
                    "authorization_reason": blocked["reason"],
                }
            else:
                next_action = {
                    "action_id": None,
                    "authorization_reason": gap_reason or "required_capability_unavailable",
                }
        else:
            decision, reason, checkpoint = "STOP", "no_admissible_new_direction", "terminal"
    else:
        raise ValidationError("evidence selection status is inconsistent with hypotheses")

    if decision not in DECISIONS:
        raise ValidationError("decision state is unsupported")
    uncertainty = set(model.get("uncertainties", []))
    for item in active:
        uncertainty.update(item.get("missing_evidence_kinds", []))
    return {
        "schema_version": DECISION_SCHEMA,
        "epoch_id": model.get("epoch_id"),
        "decision": decision,
        "terminal_reason": reason,
        "primary_diagnosis": copy.deepcopy(primary),
        "benefit_ceiling": {
            "microseconds": maximum_ceiling,
            "minimum_effect_us": threshold,
            "qualifies": maximum_ceiling >= threshold,
            "basis": None if primary is None else primary["benefit_ceiling_basis"],
        },
        "uncertainty": sorted(uncertainty),
        "next_action": next_action,
        "cost": _cost_for_action(model, action),
        "next_checkpoint": checkpoint,
        "external_challenge": _external_summary(external_review),
    }
