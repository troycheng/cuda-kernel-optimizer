#!/usr/bin/env python3
"""Turn validated diagnosis artifacts into one bounded next decision."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
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


def _load_adaptive_investment_module():
    path = Path(__file__).with_name("adaptive_investment.py")
    name = "cuda_optimizer_adaptive_investment_diagnostic"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adaptive investment module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ADAPTIVE_INVESTMENT = _load_adaptive_investment_module()


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


def build_initial_investment_brief(
    performance_model: Mapping[str, Any],
    bootstrap_execution_seconds: float,
) -> dict:
    """Summarize the first profile without inventing an unobserved next-action cost."""
    model = _object(performance_model, "performance_model")
    if model.get("schema_version") != "cuda-optimizer/performance-model-v1":
        raise ValidationError("performance_model schema is unsupported")
    nodes = model.get("node_directions")
    if type(nodes) is not list or not nodes:
        raise ValidationError("performance_model has no modeled bottleneck")
    ranked = []
    for index, raw in enumerate(nodes):
        node = _object(raw, f"performance_model.node_directions[{index}]")
        node_id = node.get("node_id")
        layer = node.get("layer")
        basis = node.get("basis")
        if not all(type(value) is str and value for value in (node_id, layer, basis)):
            raise ValidationError("performance_model bottleneck identity is invalid")
        ranked.append(
            (
                _number(
                    node.get("benefit_ceiling_us"),
                    f"performance_model.node_directions[{index}].benefit_ceiling_us",
                ),
                index,
                node,
            )
        )
    _, _, primary = max(ranked, key=lambda item: (item[0], -item[1]))
    uncertainties = model.get("uncertainties", [])
    if type(uncertainties) is not list or any(
        type(item) is not str or not item for item in uncertainties
    ):
        raise ValidationError("performance_model uncertainties are invalid")
    return {
        "schema_version": "cuda-optimizer/initial-investment-brief-v1",
        "primary_bottleneck": {
            "node_id": primary["node_id"],
            "layer": primary["layer"],
            "removable_time_ceiling_us": _number(
                primary["benefit_ceiling_us"],
                "performance_model primary benefit_ceiling_us",
            ),
            "basis": primary["basis"],
        },
        "minimum_effect_us": _number(
            model.get("minimum_effect_us"),
            "performance_model.minimum_effect_us",
            positive=True,
        ),
        "largest_uncertainty": (
            None if not uncertainties else uncertainties[0]
        ),
        "bootstrap_execution_seconds": _number(
            bootstrap_execution_seconds,
            "bootstrap_execution_seconds",
        ),
        "cost": {
            "p50_seconds": None,
            "p90_seconds": None,
            "basis": "unavailable",
        },
        "next_checkpoint": "propose_hypotheses",
    }


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
    node_directions = {
        item["node_id"]: item for item in model.get("node_directions", [])
    }
    window = _number(model.get("window_duration_us"), "window_duration_us", positive=True)
    ranked = []
    for item in active:
        scoped = [
            node_directions[node_id]
            for node_id in item["scope_node_ids"]
            if node_id in node_directions
        ]
        if not scoped:
            raise ValidationError(
                "active hypothesis scope has no modeled hot-path node"
            )
        summed_duration = sum(
            _number(node["benefit_ceiling_us"], "benefit_ceiling_us")
            for node in scoped
        )
        intervals = []
        for node in scoped:
            if node.get("first_start_us") is None or node.get("last_end_us") is None:
                intervals = []
                break
            start = _number(node["first_start_us"], "first_start_us")
            end = _number(node["last_end_us"], "last_end_us")
            if end <= start:
                raise ValidationError("node timing envelope must be positive")
            intervals.append((start, end))
        if intervals:
            covered = 0.0
            current_start, current_end = sorted(intervals)[0]
            for start, end in sorted(intervals)[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    covered += current_end - current_start
                    current_start, current_end = start, end
            covered += current_end - current_start
            ceiling = min(window, summed_duration, covered)
            ceiling_basis = "scoped_timing_union_upper_bound"
        else:
            ceiling = min(window, summed_duration)
            ceiling_basis = "scoped_active_time_upper_bound_without_envelopes"
        ranked.append(
            {
                "hypothesis_id": item["hypothesis_id"],
                "mechanism": item["mechanism"],
                "claim_layer": item["claim_layer"],
                "statement": item["statement"],
                "confidence": item["confidence"],
                "benefit_ceiling_us": ceiling,
                "benefit_ceiling_basis": ceiling_basis,
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
            "challenges": [],
            "advisory_only": True,
        }
    root = _object(value, "external_review")
    verdicts = []
    challenges = []
    for item in root.get("reviews", []):
        if not isinstance(item, Mapping):
            continue
        response = item.get("response")
        verdict = response.get("verdict") if isinstance(response, Mapping) else None
        verdicts.append(
            {
                "provider": item.get("provider"),
                "status": item.get("status"),
                "verdict": verdict,
            }
        )
        if item.get("status") == "completed" and verdict == "challenge":
            concerns = response.get("concerns", [])
            experiments = response.get("suggested_experiments", [])
            challenges.append(
                {
                    "provider": item.get("provider"),
                    "concerns": copy.deepcopy(concerns) if type(concerns) is list else [],
                    "suggested_experiments": (
                        copy.deepcopy(experiments) if type(experiments) is list else []
                    ),
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
        "challenges": challenges,
        "advisory_only": True,
    }


def _adjudicate_external_challenge(
    summary: Mapping[str, Any], decision: str, primary: Mapping[str, Any] | None
) -> dict:
    """State how local evidence handled an advisory external challenge."""
    if not summary.get("challenges"):
        return {"status": "not_required", "evidence_ids": []}
    if decision == "MEASURE":
        return {"status": "continue_measurement", "evidence_ids": []}
    if decision == "PURSUE":
        return {
            "status": "retained_for_candidate_validation",
            "evidence_ids": [],
        }
    if decision == "REVIEW_REQUIRED":
        return {"status": "retained_for_local_review", "evidence_ids": []}
    return {"status": "no_effect_on_terminal_decision", "evidence_ids": []}


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


def _identity_digest(model: Mapping[str, Any]) -> str:
    identities = _object(model.get("identities"), "performance_model.identities")
    payload = json.dumps(
        identities, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _history_records(value: Sequence[Mapping[str, Any]] | None) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError("candidate_history must be a sequence")
    records = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValidationError(f"candidate_history[{index}] must be an object")
        records.append(copy.deepcopy(dict(item)))
    return records


def _proposal_records(
    value: Sequence[Mapping[str, Any]] | None,
    *,
    identity_digest: str,
) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError("candidate_proposals must be a sequence")
    records = []
    seen = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValidationError(f"candidate_proposals[{index}] must be an object")
        proposal = copy.deepcopy(dict(raw))
        proposal_id = proposal.get("proposal_id")
        hypothesis_id = proposal.get("hypothesis_id")
        action_id = proposal.get("action_id")
        if not all(type(item) is str and item for item in (proposal_id, hypothesis_id, action_id)):
            raise ValidationError("candidate proposal ids must be non-empty strings")
        if proposal_id in seen:
            raise ValidationError("candidate proposal ids must be unique")
        seen.add(proposal_id)
        if action_id != f"implement-{hypothesis_id}":
            raise ValidationError("candidate proposal action does not match hypothesis")
        if proposal.get("identity_digest") != identity_digest:
            continue
        if proposal.get("freshness") != "current":
            continue
        if proposal.get("basis") not in {
            "identity_matched_candidate_history",
            "user_authorized_upper_bound",
        }:
            raise ValidationError("candidate proposal cost basis is unsupported")
        p50 = _number(
            proposal.get("p50_seconds"),
            f"candidate_proposals[{index}].p50_seconds",
            positive=True,
        )
        p90 = _number(
            proposal.get("p90_seconds"),
            f"candidate_proposals[{index}].p90_seconds",
            positive=True,
        )
        if p50 > p90:
            raise ValidationError("candidate proposal P50 must not exceed P90")
        proposal["p50_seconds"] = p50
        proposal["p90_seconds"] = p90
        records.append(proposal)
    return records


def _matching_history(records: list[dict], *, hypothesis_id=None, action_id=None) -> dict:
    matches = [
        item
        for item in records
        if (hypothesis_id is not None and item.get("hypothesis_id") == hypothesis_id)
        or (action_id is not None and item.get("action_id") == action_id)
    ]
    return matches[-1] if matches else {}


def _timing_spend(model: Mapping[str, Any]) -> float:
    total = 0.0
    estimates = model.get("action_timing_estimates", {})
    if not isinstance(estimates, Mapping):
        return total
    for estimate in estimates.values():
        if not isinstance(estimate, Mapping):
            continue
        sample_count = estimate.get("sample_count")
        p50 = estimate.get("p50_seconds")
        if type(sample_count) is int and sample_count > 0 and type(p50) in {int, float}:
            total += sample_count * _number(p50, "action_timing_estimates.p50_seconds")
    return total


def _investment_inputs(
    model: Mapping[str, Any],
    authorization: Mapping[str, Any] | None,
    spend: Mapping[str, Any] | None,
) -> tuple[dict, dict]:
    if spend is None:
        elapsed = _timing_spend(model)
    else:
        elapsed = _number(
            _object(spend, "spend").get("elapsed_seconds"),
            "spend.elapsed_seconds",
        )
    if authorization is None:
        maximum = elapsed
    else:
        maximum = _number(
            _object(authorization, "authorization").get("max_seconds"),
            "authorization.max_seconds",
        )
    return {"max_seconds": maximum}, {"elapsed_seconds": elapsed}


def _direction_portfolio(
    ranked: list[dict],
    model: Mapping[str, Any],
    history: list[dict],
    threshold: float,
) -> tuple[list[dict], str]:
    identity = _identity_digest(model)
    directions = []
    for item in ranked:
        record = _matching_history(history, hypothesis_id=item["hypothesis_id"])
        bound_identity = record.get("identity_digest", identity)
        stale = type(bound_identity) is str and bound_identity != identity
        status = (
            "falsified"
            if record.get("status") == "falsified"
            else "stale"
            if stale
            else "supported"
            if item["confidence"] == "direction_supported"
            else "candidate"
        )
        directions.append(
            {
                "direction_id": item["hypothesis_id"],
                "status": status,
                "mechanism": item["mechanism"],
                "benefit": {
                    # A supported direction is not a measured candidate win.
                    # The lower bound stays zero until representative paired
                    # evaluation supplies an identity-bound interval.
                    "lower": 0.0,
                    "upper": item["benefit_ceiling_us"],
                    "basis": item["benefit_ceiling_basis"],
                    "identity_digest": identity,
                    "stale": stale,
                },
            }
        )
    return directions, identity


def _action_outcomes(action_id: str, targets: list[str]) -> list[dict]:
    return [
        {
            "outcome_id": f"{action_id}-supports",
            "supports": copy.deepcopy(targets),
            "opposes": [],
        },
        {
            "outcome_id": f"{action_id}-opposes",
            "supports": [],
            "opposes": copy.deepcopy(targets),
        },
    ]


def _investment_action(
    action_id: str,
    *,
    mechanism: str,
    targets: list[str],
    outcomes: list[dict],
    controller_action: Mapping[str, Any] | None,
    model: Mapping[str, Any],
    history: list[dict],
    action_bounds: Mapping[str, Any],
    allow_generic_bounds: bool = True,
    kind: str = "check",
) -> tuple[dict, bool]:
    controller = {} if controller_action is None else dict(controller_action)
    estimate = model.get("action_timing_estimates", {}).get(action_id, {})
    record = _matching_history(history, action_id=action_id)
    status = record.get("implementation_status", record.get("status", "available"))
    raw_p90 = record.get("p90_seconds", record.get("elapsed_seconds"))
    if raw_p90 is None and allow_generic_bounds and isinstance(estimate, Mapping):
        raw_p90 = estimate.get("p90_seconds")
    if raw_p90 is None and allow_generic_bounds:
        bound = action_bounds.get(action_id)
        if isinstance(bound, Mapping) and bound.get("basis") in {
            "identity_matched_history",
            "user_authorized_upper_bound",
        }:
            raw_p90 = bound.get("p90_seconds")
    bounded = raw_p90 is not None
    action = {
        "action_id": action_id,
        "kind": kind,
        "mechanism": mechanism,
        "cost": controller.get("cost", "low"),
        "perturbation": controller.get("perturbation", "low"),
        "risk": controller.get("risk", "none"),
        "p90_seconds": (
            _number(raw_p90, f"action {action_id} p90_seconds")
            if bounded
            else None
        ),
        "target_direction_ids": copy.deepcopy(targets),
        "outcomes": copy.deepcopy(outcomes),
        "implementation_status": "failed" if status == "failed" else "available",
    }
    return action, bounded


def _adaptive_actions(
    ranked: list[dict],
    directions: list[dict],
    selection: Mapping[str, Any],
    model: Mapping[str, Any],
    history: list[dict],
    proposals: list[dict],
    action_bounds: Mapping[str, Any],
) -> tuple[list[dict], dict[str, dict], list[dict]]:
    actions = []
    metadata = {}
    unbounded = []
    by_id = {item["direction_id"]: item for item in directions}
    if selection.get("status") == "sufficient":
        proposal_by_hypothesis = {
            item["hypothesis_id"]: item for item in proposals
        }
        candidate_order = sorted(
            ranked,
            key=lambda item: (
                -item["benefit_ceiling_us"],
                proposal_by_hypothesis.get(item["hypothesis_id"], {}).get(
                    "p90_seconds", float("inf")
                ),
                item["hypothesis_id"],
            ),
        )
        for hypothesis in candidate_order:
            direction_id = hypothesis["hypothesis_id"]
            if hypothesis["confidence"] != "direction_supported":
                continue
            stale = by_id[direction_id]["status"] == "stale"
            action_id = (
                f"refresh-{direction_id}" if stale else f"implement-{direction_id}"
            )
            record = _matching_history(history, hypothesis_id=direction_id)
            failed = (
                record.get("implementation_status", record.get("status")) == "failed"
                and record.get("identity_digest") == _identity_digest(model)
            )
            if failed and not stale:
                continue
            proposal = proposal_by_hypothesis.get(direction_id, {})
            if proposal.get("action_id") != action_id:
                proposal = {}
            candidate_cost = [proposal] if proposal else []
            action, bounded = _investment_action(
                action_id,
                mechanism=(
                    f"refresh:{hypothesis['mechanism']}"
                    if stale
                    else f"candidate:{hypothesis['mechanism']}"
                ),
                targets=[direction_id],
                outcomes=_action_outcomes(action_id, [direction_id]),
                controller_action=None,
                model=model,
                history=candidate_cost,
                action_bounds=action_bounds,
                allow_generic_bounds=False,
                kind="refresh" if stale else "check",
            )
            if stale:
                # A stale candidate bound cannot authorize a synthetic command.
                # Refresh must be represented by an executable catalog action.
                bounded = False
                action["p90_seconds"] = None
                action["implementation_status"] = "available"
                action["reason"] = "refresh_action_unavailable"
            elif not bounded:
                action["reason"] = "cost_unavailable"
            (actions if bounded else unbounded).append(action)
            metadata[action_id] = {
                "purpose": "refresh" if stale else "candidate",
                "hypothesis_id": direction_id,
                "proposal": copy.deepcopy(proposal) if proposal else None,
            }
            # Candidate implementation follows V1.1 benefit ordering. P90 and
            # stable identity only break an exact benefit-ceiling tie.
            break
        return actions, metadata, unbounded

    selected = selection.get("selected_request")
    candidates = []
    if isinstance(selected, Mapping):
        candidates.append((dict(selected), True))
    for rejected in selection.get("rejections", []):
        if isinstance(rejected, Mapping):
            candidates.append((dict(rejected), False))
    for request, is_selected in candidates:
        action_id = request.get("action_id")
        controller = request.get("controller_action")
        if type(action_id) is not str or not isinstance(controller, Mapping):
            continue
        targets = request.get("target_hypothesis_ids")
        if type(targets) is not list or not targets:
            targets = [item["hypothesis_id"] for item in ranked]
        outcomes = request.get("outcomes")
        if type(outcomes) is not list or not outcomes:
            outcomes = _action_outcomes(action_id, targets)
        action, bounded = _investment_action(
            action_id,
            mechanism=controller.get("evidence_kind", action_id),
            targets=targets,
            outcomes=outcomes,
            controller_action=controller,
            model=model,
            history=history,
            action_bounds=action_bounds,
        )
        if not is_selected:
            action["implementation_status"] = "failed"
        if not bounded:
            action["reason"] = "cost_unavailable"
        (actions if bounded else unbounded).append(action)
        metadata[action_id] = {"purpose": "evidence", "request": request}
    return actions, metadata, unbounded


def _knowledge_summary(value: Mapping[str, Any] | None) -> dict:
    if not isinstance(value, Mapping):
        return {
            "status": "unavailable",
            "candidate_count": 0,
            "rejection_count": 0,
            "advisory_only": True,
        }
    candidates = value.get("candidates", [])
    rejections = value.get("rejections", [])
    return {
        "status": value.get("knowledge_support", "context_only"),
        "candidate_count": len(candidates) if type(candidates) is list else 0,
        "rejection_count": len(rejections) if type(rejections) is list else 0,
        "advisory_only": True,
    }


def decide_next_step(
    performance_model: Mapping[str, Any],
    hypothesis_result: Mapping[str, Any],
    evidence_selection: Mapping[str, Any],
    *,
    external_review: Mapping[str, Any] | None = None,
    authorization: Mapping[str, Any] | None = None,
    spend: Mapping[str, Any] | None = None,
    wall_elapsed_seconds: float | None = None,
    candidate_history: Sequence[Mapping[str, Any]] | None = None,
    candidate_proposals: Sequence[Mapping[str, Any]] | None = None,
    knowledge_adaptation: Mapping[str, Any] | None = None,
    action_bounds: Mapping[str, Any] | None = None,
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
    history = _history_records(candidate_history)
    directions, identity = _direction_portfolio(ranked, model, history, threshold)
    proposals = _proposal_records(candidate_proposals, identity_digest=identity)
    bounds = {} if action_bounds is None else _object(action_bounds, "action_bounds")
    adaptive_actions, action_metadata, unbounded_actions = _adaptive_actions(
        ranked, directions, selection, model, history, proposals, bounds
    )
    adaptive_authorization, adaptive_spend = _investment_inputs(
        model, authorization, spend
    )
    wall_elapsed = (
        adaptive_spend["elapsed_seconds"]
        if wall_elapsed_seconds is None
        else _number(wall_elapsed_seconds, "wall_elapsed_seconds")
    )
    adaptive = _ADAPTIVE_INVESTMENT.decide_next_action(
        directions,
        adaptive_actions,
        authorization=adaptive_authorization,
        spend=adaptive_spend,
        minimum_effect=threshold,
    )

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

    local_decision = decision
    effective_selected_action = adaptive.get("selected_action")
    suppressed_action_ids = []
    if local_decision in {"MEASURE", "PURSUE"}:
        selected_investment = effective_selected_action
        blocked_investment = adaptive.get("blocked_action")
        unavailable = next(
            (
                item
                for item in unbounded_actions
                if item.get("implementation_status") != "failed"
            ),
            None,
        )
        unavailable_blocks_selected = selected_investment is None
        if unavailable is not None and isinstance(selected_investment, Mapping):
            unavailable_info = action_metadata[unavailable["action_id"]]
            selected_info = action_metadata[selected_investment["action_id"]]
            if (
                unavailable_info.get("hypothesis_id") is not None
                and selected_info.get("hypothesis_id") is not None
            ):
                rank_by_hypothesis = {
                    item["hypothesis_id"]: index
                    for index, item in enumerate(ranked)
                }
                unavailable_blocks_selected = rank_by_hypothesis[
                    unavailable_info["hypothesis_id"]
                ] < rank_by_hypothesis[selected_info["hypothesis_id"]]
        if unavailable is not None and unavailable_blocks_selected:
            if isinstance(selected_investment, Mapping):
                suppressed_action_ids.append(selected_investment["action_id"])
                effective_selected_action = None
            unavailable_info = action_metadata[unavailable["action_id"]]
            unavailable_reason = unavailable["reason"]
            decision, reason, checkpoint = (
                "REVIEW_REQUIRED",
                unavailable_reason,
                "after_authorization_decision",
            )
            if unavailable_info["purpose"] == "candidate":
                chosen = next(
                    item
                    for item in ranked
                    if item["hypothesis_id"] == unavailable_info["hypothesis_id"]
                )
                primary = chosen
                next_action = {
                    "action_id": unavailable["action_id"],
                    "hypothesis_id": chosen["hypothesis_id"],
                    "authorization_reason": unavailable_reason,
                }
            elif unavailable_info["purpose"] == "refresh":
                next_action = {
                    "action_id": unavailable["action_id"],
                    "kind": "refresh",
                    "target_hypothesis_ids": copy.deepcopy(
                        unavailable["target_direction_ids"]
                    ),
                    "authorization_reason": unavailable_reason,
                }
            else:
                request = unavailable_info["request"]
                next_action = {
                    "request_id": request.get("request_id"),
                    "action_id": unavailable["action_id"],
                    "authorization_reason": unavailable_reason,
                }
        elif adaptive["decision"] == "REVIEW_REQUIRED":
            decision, reason, checkpoint = (
                "REVIEW_REQUIRED",
                "cumulative_authorization_exceeded",
                "after_authorization_decision",
            )
            next_action = {
                "action_id": None if blocked_investment is None else blocked_investment["action_id"],
                "authorization_reason": "cumulative_time_exceeded",
            }
        elif adaptive["decision"] == "STOP":
            decision, reason, checkpoint = "STOP", adaptive["reason"], "terminal"
            next_action = None
        elif isinstance(selected_investment, Mapping):
            selected_id = selected_investment["action_id"]
            action_info = action_metadata[selected_id]
            if action_info["purpose"] == "candidate":
                chosen = next(
                    item
                    for item in ranked
                    if item["hypothesis_id"] == action_info["hypothesis_id"]
                )
                primary = chosen
                decision, reason, checkpoint = (
                    "PURSUE",
                    "direction_supported",
                    "after_candidate_screen",
                )
                next_action = {
                    "action_id": "implement-candidate",
                    "hypothesis_id": chosen["hypothesis_id"],
                    "mechanism": chosen["mechanism"],
                    "claim_layer": chosen["claim_layer"],
                }
                proposal = action_info.get("proposal")
                if isinstance(proposal, Mapping):
                    next_action.update(
                        {
                            "proposal_id": proposal["proposal_id"],
                            "proposal_digest": proposal.get("proposal_digest"),
                        }
                    )
            elif action_info["purpose"] == "refresh":
                decision, reason, checkpoint = (
                    "MEASURE",
                    "stale_bound_requires_refresh",
                    "after_selected_evidence",
                )
                next_action = {
                    "action_id": selected_id,
                    "kind": "refresh",
                    "target_hypothesis_ids": copy.deepcopy(
                        selected_investment["target_direction_ids"]
                    ),
                }
            else:
                request = action_info["request"]
                decision, reason, checkpoint = (
                    "MEASURE",
                    "discriminating_evidence_required",
                    "after_selected_evidence",
                )
                next_action = {
                    "request_id": request.get("request_id"),
                    "action_id": selected_id,
                    "question": request.get("question"),
                    "target_hypothesis_ids": copy.deepcopy(
                        selected_investment["target_direction_ids"]
                    ),
                }

    if decision not in DECISIONS:
        raise ValidationError("decision state is unsupported")
    uncertainty = set(model.get("uncertainties", []))
    for item in active:
        uncertainty.update(item.get("missing_evidence_kinds", []))
    external_summary = _external_summary(external_review)
    external_summary["local_adjudication"] = _adjudicate_external_challenge(
        external_summary, decision, primary
    )
    result_cost = _cost_for_action(model, action)
    if isinstance(effective_selected_action, Mapping):
        selected_info = action_metadata.get(effective_selected_action["action_id"], {})
        proposal = selected_info.get("proposal")
        if isinstance(proposal, Mapping):
            result_cost = {
                "class": effective_selected_action.get("cost"),
                "p50_seconds": proposal["p50_seconds"],
                "p90_seconds": proposal["p90_seconds"],
                "basis": proposal["basis"],
            }
        elif result_cost["p90_seconds"] is None:
            bound = bounds.get(effective_selected_action["action_id"])
            if isinstance(bound, Mapping) and bound.get("basis") in {
                "identity_matched_history",
                "user_authorized_upper_bound",
            }:
                result_cost = {
                    "class": effective_selected_action.get("cost"),
                    "p50_seconds": bound.get("p50_seconds"),
                    "p90_seconds": bound.get("p90_seconds"),
                    "basis": bound["basis"],
                }
    result = {
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
        "cost": result_cost,
        "next_checkpoint": checkpoint,
        "external_challenge": external_summary,
    }
    remaining = max(
        0.0,
        adaptive_authorization["max_seconds"] - adaptive_spend["elapsed_seconds"],
    )
    blocked_summary = copy.deepcopy(adaptive.get("blocked_action"))
    if blocked_summary is None and result["decision"] == "REVIEW_REQUIRED":
        blocked_id = (result.get("next_action") or {}).get("action_id")
        blocked_summary = next(
            (
                copy.deepcopy(item)
                for item in unbounded_actions
                if item["action_id"] == blocked_id
            ),
            None,
        )
    if blocked_summary is None and result["decision"] == "REVIEW_REQUIRED":
        blocked_id = (result.get("next_action") or {}).get("action_id")
        blocked_summary = next(
            (
                copy.deepcopy(item)
                for item in adaptive_actions
                if item["action_id"] == blocked_id
            ),
            None,
        )
        if blocked_summary is not None:
            blocked_summary["implementation_status"] = "available"
    result["investment_brief"] = {
        "schema_version": "cuda-optimizer/investment-brief-v1",
        "decision": result["decision"],
        "terminal_reason": result["terminal_reason"],
        "primary_diagnosis": copy.deepcopy(result["primary_diagnosis"]),
        "benefit_ceiling": copy.deepcopy(result["benefit_ceiling"]),
        "uncertainty": copy.deepcopy(result["uncertainty"]),
        "next_action": copy.deepcopy(result["next_action"]),
        "cost": copy.deepcopy(result["cost"]),
        "next_checkpoint": result["next_checkpoint"],
        "external_challenge": copy.deepcopy(result["external_challenge"]),
        "portfolio": copy.deepcopy(adaptive["portfolio"]),
        "cumulative_investment": {
            "elapsed_seconds": adaptive_spend["elapsed_seconds"],
            "wall_elapsed_seconds": wall_elapsed,
            "remaining_authorization_seconds": remaining,
            "projected_p90_seconds": (
                None
                if blocked_summary is not None
                and blocked_summary.get("p90_seconds") is None
                else adaptive["projected_spend"]["p90_seconds"]
            ),
            "bound_basis": "committed_controlled_execution",
        },
        "selected_action": copy.deepcopy(effective_selected_action),
        "blocked_action": blocked_summary,
        "skipped_action_ids": sorted(
            set(adaptive.get("skipped_actions", []))
            | {
                item["action_id"]
                for item in unbounded_actions
                if item.get("implementation_status") == "failed"
            }
            | set(suppressed_action_ids)
            | {
                item["action_id"]
                for item in proposals
                if not isinstance(effective_selected_action, Mapping)
                or item["action_id"] != effective_selected_action["action_id"]
            }
            | {
                item.get("action_id", f"implement-{item.get('hypothesis_id')}")
                for item in history
                if item.get("implementation_status", item.get("status")) == "failed"
                and item.get("identity_digest") == identity
                and item.get("hypothesis_id")
            }
        ),
        "bound_basis": {
            "identity_digest": identity,
            "benefit": "local_execution_map_timing_upper_bound",
            "knowledge_authority": "none",
        },
        "next_feedback_point": (
            result["next_checkpoint"]
            if result["decision"] == "REVIEW_REQUIRED"
            else adaptive.get("next_checkpoint")
        ),
        "knowledge_adaptation": _knowledge_summary(knowledge_adaptation),
    }
    return result
