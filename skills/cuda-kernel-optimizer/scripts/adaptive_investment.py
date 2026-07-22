#!/usr/bin/env python3
"""Choose one evidence action from locally bounded, replayable inputs."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LEVELS = {"none": 0, "low": 1, "medium": 2, "high": 3}
_DIRECTION_STATUSES = {"candidate", "supported", "stale", "falsified"}
_ACTION_KINDS = {"check", "refresh"}
_IMPLEMENTATION_STATUSES = {"available", "failed"}


class ValidationError(ValueError):
    """Raised when a decision input cannot be replayed safely."""


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a stable identifier")
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValidationError(f"{label} must be finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValidationError(f"{label} must be at least {minimum}")
    return number


def _level(value: Any, label: str) -> str:
    if value not in _LEVELS:
        raise ValidationError(f"{label} must be a supported cost level")
    return value


def _ids(value: Any, label: str) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError(f"{label} must be a non-empty identifier list")
    result = [_identifier(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValidationError(f"{label} must not contain duplicates")
    return sorted(result)


def _benefit(value: Any, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    required = {"lower", "upper", "basis", "identity_digest", "stale"}
    missing = required - set(value)
    if missing:
        raise ValidationError(f"{label} is missing fields: {sorted(missing)}")
    lower = _finite(value["lower"], f"{label}.lower")
    upper = _finite(value["upper"], f"{label}.upper")
    if lower > upper:
        raise ValidationError(f"{label}.lower must not exceed upper")
    if type(value["basis"]) is not str or not value["basis"].strip():
        raise ValidationError(f"{label}.basis must be non-empty")
    if type(value["identity_digest"]) is not str or _SHA256.fullmatch(value["identity_digest"]) is None:
        raise ValidationError(f"{label}.identity_digest must be SHA-256")
    if type(value["stale"]) is not bool:
        raise ValidationError(f"{label}.stale must be a boolean")
    # Deliberately reconstruct this allow-list: model or external gain claims do
    # not enter the decision boundary.
    return {
        "lower": lower,
        "upper": upper,
        "basis": value["basis"],
        "identity_digest": value["identity_digest"],
        "stale": value["stale"],
    }


def _directions(value: Sequence[Mapping[str, Any]], minimum_effect: float) -> list[dict]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError("directions must be a sequence")
    result = []
    seen = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValidationError(f"directions[{index}] must be an object")
        direction_id = _identifier(raw.get("direction_id"), f"directions[{index}].direction_id")
        if direction_id in seen:
            raise ValidationError("direction ids must be unique")
        seen.add(direction_id)
        status = raw.get("status")
        if status not in _DIRECTION_STATUSES:
            raise ValidationError(f"directions[{index}].status is unsupported")
        mechanism = _identifier(raw.get("mechanism"), f"directions[{index}].mechanism")
        benefit = _benefit(raw.get("benefit"), f"directions[{index}].benefit")
        stale = status == "stale" or benefit["stale"]
        reason = None
        if status == "falsified":
            reason = "falsified"
        elif benefit["upper"] < minimum_effect:
            reason = "below_minimum_effect"
        result.append(
            {
                "direction_id": direction_id,
                "status": "stale" if stale and reason is None else ("closed" if reason else status),
                "mechanism": mechanism,
                "benefit": benefit,
                "reason": reason,
            }
        )
    result.sort(key=lambda item: item["direction_id"])
    retained_mechanisms = set()
    for direction in result:
        if direction["status"] == "closed":
            continue
        if direction["mechanism"] in retained_mechanisms:
            direction["status"] = "closed"
            direction["reason"] = "exact_mechanism_repeat"
            continue
        retained_mechanisms.add(direction["mechanism"])
    return result


def _intervals(value: Any, label: str) -> dict[str, dict]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    result = {}
    for direction_id, interval in value.items():
        _identifier(direction_id, f"{label} key")
        if not isinstance(interval, Mapping):
            raise ValidationError(f"{label}.{direction_id} must be an object")
        if set(interval) != {"lower", "upper"}:
            raise ValidationError(f"{label}.{direction_id} must contain only lower and upper")
        lower = _finite(interval["lower"], f"{label}.{direction_id}.lower")
        upper = _finite(interval["upper"], f"{label}.{direction_id}.upper")
        if lower > upper:
            raise ValidationError(f"{label}.{direction_id}.lower must not exceed upper")
        result[direction_id] = {"lower": lower, "upper": upper}
    return result


def _outcomes(value: Any, label: str, known_directions: set[str]) -> list[dict]:
    if type(value) is not list or not value:
        raise ValidationError(f"{label} must be a non-empty list")
    result = []
    seen = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValidationError(f"{label}[{index}] must be an object")
        allowed = {"outcome_id", "supports", "opposes", "candidate_intervals"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValidationError(f"{label}[{index}] contains unknown fields: {sorted(unknown)}")
        outcome_id = _identifier(raw.get("outcome_id"), f"{label}[{index}].outcome_id")
        if outcome_id in seen:
            raise ValidationError(f"{label} ids must be unique")
        seen.add(outcome_id)
        supports = _ids(raw.get("supports"), f"{label}[{index}].supports") if raw.get("supports") else []
        opposes = _ids(raw.get("opposes"), f"{label}[{index}].opposes") if raw.get("opposes") else []
        if set(supports) & set(opposes):
            raise ValidationError(f"{label}[{index}] cannot support and oppose one direction")
        intervals = _intervals(raw.get("candidate_intervals"), f"{label}[{index}].candidate_intervals")
        if not (set(supports) | set(opposes) | set(intervals)) <= known_directions:
            raise ValidationError(f"{label}[{index}] references an unknown direction")
        result.append(
            {
                "outcome_id": outcome_id,
                "supports": supports,
                "opposes": opposes,
                "candidate_intervals": intervals,
            }
        )
    return sorted(result, key=lambda item: item["outcome_id"])


def _actions(value: Sequence[Mapping[str, Any]], known_directions: set[str]) -> list[dict]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError("actions must be a sequence")
    result = []
    seen = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValidationError(f"actions[{index}] must be an object")
        allowed = {
            "action_id", "kind", "mechanism", "cost", "perturbation", "risk",
            "p90_seconds", "target_direction_ids", "outcomes", "implementation_status",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValidationError(f"actions[{index}] contains unknown fields: {sorted(unknown)}")
        action_id = _identifier(raw.get("action_id"), f"actions[{index}].action_id")
        if action_id in seen:
            raise ValidationError("action ids must be unique")
        seen.add(action_id)
        kind = raw.get("kind")
        if kind not in _ACTION_KINDS:
            raise ValidationError(f"actions[{index}].kind is unsupported")
        targets = _ids(raw.get("target_direction_ids"), f"actions[{index}].target_direction_ids")
        if not set(targets) <= known_directions:
            raise ValidationError(f"actions[{index}] targets an unknown direction")
        status = raw.get("implementation_status", "available")
        if status not in _IMPLEMENTATION_STATUSES:
            raise ValidationError(f"actions[{index}].implementation_status is unsupported")
        result.append(
            {
                "action_id": action_id,
                "kind": kind,
                "mechanism": _identifier(raw.get("mechanism"), f"actions[{index}].mechanism"),
                "cost": _level(raw.get("cost"), f"actions[{index}].cost"),
                "perturbation": _level(raw.get("perturbation"), f"actions[{index}].perturbation"),
                "risk": _level(raw.get("risk"), f"actions[{index}].risk"),
                "p90_seconds": _finite(raw.get("p90_seconds"), f"actions[{index}].p90_seconds", minimum=0.0),
                "target_direction_ids": targets,
                "outcomes": _outcomes(raw.get("outcomes"), f"actions[{index}].outcomes", known_directions),
                "implementation_status": status,
            }
        )
    return sorted(result, key=lambda item: item["action_id"])


def _authorization(value: Mapping[str, Any], label: str) -> float:
    if not isinstance(value, Mapping) or set(value) != {"max_seconds"}:
        raise ValidationError(f"{label} must contain only max_seconds")
    return _finite(value["max_seconds"], f"{label}.max_seconds", minimum=0.0)


def _spend(value: Mapping[str, Any]) -> float:
    if not isinstance(value, Mapping) or set(value) != {"elapsed_seconds"}:
        raise ValidationError("spend must contain only elapsed_seconds")
    return _finite(value["elapsed_seconds"], "spend.elapsed_seconds", minimum=0.0)


def _changes_decision(action: Mapping[str, Any], active: Mapping[str, dict]) -> bool:
    for outcome in action["outcomes"]:
        if set(outcome["supports"]) | set(outcome["opposes"]):
            return True
        for direction_id, interval in outcome["candidate_intervals"].items():
            current = active.get(direction_id)
            if current and interval != {
                "lower": current["benefit"]["lower"],
                "upper": current["benefit"]["upper"],
            }:
                return True
    return False


def _evidence_gain(action: Mapping[str, Any]) -> int:
    return len(
        {
            (action["mechanism"], direction_id)
            for outcome in action["outcomes"]
            for direction_id in (
                outcome["supports"] + outcome["opposes"] + list(outcome["candidate_intervals"])
            )
        }
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = all(
        _LEVELS[left[field]] <= _LEVELS[right[field]]
        for field in ("cost", "perturbation", "risk")
    ) and left["p90_seconds"] <= right["p90_seconds"]
    covers = set(left["target_direction_ids"]) >= set(right["target_direction_ids"])
    independent_gain = _evidence_gain(left) >= _evidence_gain(right)
    strictly_better = (
        any(_LEVELS[left[field]] < _LEVELS[right[field]] for field in ("cost", "perturbation", "risk"))
        or left["p90_seconds"] < right["p90_seconds"]
        or set(left["target_direction_ids"]) > set(right["target_direction_ids"])
        or _evidence_gain(left) > _evidence_gain(right)
    )
    return no_worse and covers and independent_gain and strictly_better


def _rank(action: Mapping[str, Any]) -> tuple:
    return (
        _LEVELS[action["cost"]],
        _LEVELS[action["perturbation"]],
        _LEVELS[action["risk"]],
        action["p90_seconds"],
        -len(action["target_direction_ids"]),
        -_evidence_gain(action),
        action["action_id"],
    )


def _result(
    decision: str,
    reason: str,
    portfolio: list[dict],
    *,
    selected: dict | None = None,
    blocked: dict | None = None,
    projected: float,
    skipped: list[str],
) -> dict:
    return {
        "decision": decision,
        "reason": reason,
        "portfolio": copy.deepcopy(portfolio),
        "selected_action": copy.deepcopy(selected),
        "blocked_action": copy.deepcopy(blocked),
        "projected_spend": {"p90_seconds": projected},
        "skipped_actions": sorted(set(skipped)),
        "next_checkpoint": selected["action_id"] if selected else None,
    }


def decide_next_action(
    directions: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    *,
    authorization: Mapping[str, Any],
    spend: Mapping[str, Any],
    minimum_effect: float,
) -> dict:
    """Return one evidence-authoritative next decision without side effects."""
    minimum = _finite(minimum_effect, "minimum_effect", minimum=0.0)
    max_seconds = _authorization(authorization, "authorization")
    elapsed = _spend(spend)
    portfolio = _directions(directions, minimum)
    by_id = {item["direction_id"]: item for item in portfolio}
    normalized_actions = _actions(actions, set(by_id))

    skipped: list[str] = []
    active = {item["direction_id"]: item for item in portfolio if item["status"] != "closed"}
    for action in normalized_actions:
        if any(
            action["mechanism"] == active[direction_id]["mechanism"]
            for direction_id in action["target_direction_ids"]
            if direction_id in active
        ):
            for direction_id in action["target_direction_ids"]:
                if direction_id in active and action["mechanism"] == active[direction_id]["mechanism"]:
                    active[direction_id]["status"] = "closed"
                    active[direction_id]["reason"] = "exact_mechanism_repeat"
                    active.pop(direction_id)
            skipped.append(action["action_id"])

    eligible = []
    for action in normalized_actions:
        targets = set(action["target_direction_ids"])
        if action["action_id"] in skipped:
            continue
        if action["implementation_status"] == "failed":
            skipped.append(action["action_id"])
        elif not targets <= set(active):
            skipped.append(action["action_id"])
        elif any(active[direction_id]["status"] == "stale" for direction_id in targets) and action["kind"] != "refresh":
            skipped.append(action["action_id"])
        elif not _changes_decision(action, active):
            skipped.append(action["action_id"])
        else:
            eligible.append(action)

    undominated = [
        action for action in eligible
        if not any(other["action_id"] != action["action_id"] and _dominates(other, action) for other in eligible)
    ]
    for action in eligible:
        if action not in undominated:
            skipped.append(action["action_id"])
    if not undominated:
        return _result(
            "STOP", "no_decision_changing_action", portfolio,
            projected=elapsed, skipped=skipped,
        )

    chosen = min(undominated, key=_rank)
    projected = elapsed + chosen["p90_seconds"]
    if projected > max_seconds:
        return _result(
            "REVIEW_REQUIRED", "authorization_exceeded", portfolio,
            blocked=chosen, projected=projected, skipped=skipped,
        )
    return _result(
        "ACT", "decision_changing_action", portfolio,
        selected=chosen, projected=projected, skipped=skipped,
    )
