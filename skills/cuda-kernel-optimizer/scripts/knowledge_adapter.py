#!/usr/bin/env python3
"""Normalize bounded knowledge suggestions without creating local evidence."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_LEVELS = {"none": 0, "low": 1, "medium": 2, "high": 3}
_SCOPES = {"read_only", "project_copy"}
_CONTEXT_FIELDS = {
    "architecture",
    "software_version",
    "execution_node_ids",
    "uncovered_interval_ids",
    "available_evidence_action_ids",
    "authorized_risk",
    "authorized_scope",
}
_SUGGESTION_FIELDS = {
    "source",
    "mechanism_id",
    "statement",
    "applicability",
    "scope_node_ids",
    "unmodeled_interval_id",
    "falsification_question",
    "evidence_action",
    "risk",
    "knowledge_version",
    "freshness",
    "query_digest",
    "external_gain_pct",
}
_ACTION_FIELDS = {"action_id", "evidence_kind", "outcomes", "risk", "control_scope"}
_SENSITIVE_KEY = re.compile(
    r"command|callback|secret|token|password|promotion|success_probability",
    re.IGNORECASE,
)


class ValidationError(ValueError):
    """Raised when the local normalization context is not closed."""


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a stable identifier")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > 1024:
        raise ValidationError(f"{label} must be a non-empty bounded string")
    return value


def _ids(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if type(value) is not list or (not value and not allow_empty):
        raise ValidationError(f"{label} must be an identifier array")
    result = [_identifier(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValidationError(f"{label} must not contain duplicates")
    return sorted(result)


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _SENSITIVE_KEY.search(str(key)) is not None
            or _contains_sensitive_key(item)
            for key, item in value.items()
        )
    if type(value) is list:
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _context(value: Mapping[str, Any]) -> dict:
    if not isinstance(value, Mapping) or set(value) != _CONTEXT_FIELDS:
        raise ValidationError("context must contain only the stable adapter fields")
    risk = value["authorized_risk"]
    scope = value["authorized_scope"]
    if risk not in _LEVELS or scope not in _SCOPES:
        raise ValidationError("context authorization is unsupported")
    return {
        "architecture": _identifier(value["architecture"], "context.architecture"),
        "software_version": _text(value["software_version"], "context.software_version"),
        "execution_node_ids": _ids(value["execution_node_ids"], "context.execution_node_ids"),
        "uncovered_interval_ids": _ids(
            value["uncovered_interval_ids"],
            "context.uncovered_interval_ids",
            allow_empty=True,
        ),
        "available_evidence_action_ids": _ids(
            value["available_evidence_action_ids"],
            "context.available_evidence_action_ids",
        ),
        "authorized_risk": risk,
        "authorized_scope": scope,
    }


def _rejection(origin: str, reason: str) -> dict:
    return {"origin": origin, "reason": reason}


def _normalize(
    raw: Mapping[str, Any],
    *,
    origin: str,
    context: Mapping[str, Any],
    seen_digests: set[str],
) -> tuple[dict | None, str | None]:
    if not isinstance(raw, Mapping):
        return None, "invalid_suggestion"
    scope = raw.get("scope_node_ids")
    interval = raw.get("unmodeled_interval_id")
    falsifier = raw.get("falsification_question")
    action = raw.get("evidence_action")
    if not scope and interval is None or not falsifier or action is None:
        return None, "not_locally_falsifiable"
    if _contains_sensitive_key(raw):
        return None, "sensitive_fields"
    if set(raw) - _SUGGESTION_FIELDS:
        return None, "unknown_fields"
    required = _SUGGESTION_FIELDS - {"external_gain_pct"}
    if set(raw) - {"external_gain_pct"} != required:
        return None, "invalid_suggestion"
    try:
        source = _text(raw["source"], "suggestion.source")
        mechanism_id = _identifier(raw["mechanism_id"], "suggestion.mechanism_id")
        statement = _text(raw["statement"], "suggestion.statement")
        applicability = raw["applicability"]
        if type(applicability) is not dict or set(applicability) != {
            "architectures",
            "software_versions",
        }:
            raise ValidationError("suggestion.applicability is unsupported")
        architectures = _ids(
            applicability["architectures"], "suggestion.applicability.architectures"
        )
        versions = [
            _text(item, f"suggestion.applicability.software_versions[{index}]")
            for index, item in enumerate(applicability["software_versions"])
        ]
        if not versions or len(versions) != len(set(versions)):
            raise ValidationError("suggestion.applicability.software_versions is invalid")
        scope_ids = _ids(raw["scope_node_ids"], "suggestion.scope_node_ids", allow_empty=True)
        interval_id = raw["unmodeled_interval_id"]
        if interval_id is not None:
            interval_id = _identifier(interval_id, "suggestion.unmodeled_interval_id")
        question = _text(raw["falsification_question"], "suggestion.falsification_question")
        risk = raw["risk"]
        if risk not in _LEVELS:
            raise ValidationError("suggestion.risk is unsupported")
        if type(action) is not dict or set(action) != _ACTION_FIELDS:
            raise ValidationError("suggestion.evidence_action is unsupported")
        action_id = _identifier(action["action_id"], "suggestion.evidence_action.action_id")
        evidence_kind = _identifier(
            action["evidence_kind"], "suggestion.evidence_action.evidence_kind"
        )
        outcomes = _ids(action["outcomes"], "suggestion.evidence_action.outcomes")
        action_risk = action["risk"]
        control_scope = action["control_scope"]
        if action_risk not in _LEVELS or control_scope not in _SCOPES:
            raise ValidationError("suggestion.evidence_action authorization is unsupported")
        version = _text(raw["knowledge_version"], "suggestion.knowledge_version")
        freshness = raw["freshness"]
        digest = _identifier(raw["query_digest"], "suggestion.query_digest")
    except ValidationError:
        return None, "invalid_suggestion"
    if digest in seen_digests:
        return None, "duplicate_query_digest"
    seen_digests.add(digest)
    if (
        freshness != "current"
        or version != context["software_version"]
        or context["architecture"] not in architectures
        or context["software_version"] not in versions
    ):
        return None, "unavailable"
    if (
        not set(scope_ids).issubset(context["execution_node_ids"])
        or (interval_id is not None and interval_id not in context["uncovered_interval_ids"])
        or (not scope_ids and interval_id is None)
        or action_id not in context["available_evidence_action_ids"]
        or _LEVELS[risk] > _LEVELS[context["authorized_risk"]]
        or _LEVELS[action_risk] > _LEVELS[context["authorized_risk"]]
        or control_scope != context["authorized_scope"]
    ):
        return None, "not_locally_falsifiable"
    return {
        "mechanism_id": mechanism_id,
        "statement": statement,
        "applicability": {
            "architectures": architectures,
            "software_versions": sorted(versions),
        },
        "scope_node_ids": scope_ids,
        "unmodeled_interval_id": interval_id,
        "falsification_question": question,
        "evidence_action": {
            "action_id": action_id,
            "evidence_kind": evidence_kind,
            "outcomes": outcomes,
            "risk": action_risk,
            "control_scope": control_scope,
        },
        "risk": risk,
        "origin": origin,
        "source": source,
        "knowledge_version": version,
        "freshness": freshness,
        "confidence": "inconclusive",
        "promotion_authority": "none",
    }, None


def recommend(
    context: Mapping[str, Any],
    *,
    bundled: Sequence[Mapping[str, Any]] = (),
    searched: Sequence[Mapping[str, Any]] = (),
    external: Sequence[Mapping[str, Any]] = (),
    prior_query_digests: Sequence[str] = (),
    limit: int = 3,
) -> dict:
    """Normalize bounded suggestions; never create evidence or benefit facts."""
    normalized_context = _context(context)
    if type(limit) is not int or isinstance(limit, bool) or not 1 <= limit <= 5:
        raise ValidationError("limit must be between 1 and 5")
    if isinstance(prior_query_digests, (str, bytes)) or not isinstance(
        prior_query_digests, Sequence
    ):
        raise ValidationError("prior_query_digests must be an array")
    seen_digests = {
        _identifier(item, f"prior_query_digests[{index}]")
        for index, item in enumerate(prior_query_digests)
    }
    candidates = []
    rejections = []
    for origin, suggestions in (
        ("bundled", bundled),
        ("searched", searched),
        ("external", external),
    ):
        if isinstance(suggestions, (str, bytes)) or not isinstance(suggestions, Sequence):
            raise ValidationError(f"{origin} suggestions must be an array")
        for raw in suggestions:
            candidate, reason = _normalize(
                raw,
                origin=origin,
                context=normalized_context,
                seen_digests=seen_digests,
            )
            if candidate is not None:
                candidates.append(candidate)
            else:
                rejections.append(_rejection(origin, reason or "invalid_suggestion"))
    candidates = candidates[:limit]
    return {
        "knowledge_support": "available" if candidates else "unavailable",
        "candidates": copy.deepcopy(candidates),
        "rejections": rejections,
    }
