#!/usr/bin/env python3
"""Route a bounded set of offline diagnostic cards into active diagnosis."""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path


CARDS_PATH = Path(__file__).resolve().parents[1] / "references" / "diagnostic_cards.json"
REFERENCE_DIR = CARDS_PATH.parent
_MAX_REFERENCE_BYTES = 2 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_CASE_IDS = {"R01", "R02", "R03", "R04", "R05", "R06", "X01", "K01", "K02"}
_LAYERS = {
    "cpu",
    "gpu",
    "framework",
    "transfer",
    "communication",
    "io",
    "synchronization",
    "idle",
}
_DIAGNOSTIC_PRODUCERS = {
    "nsys_timeline": "nsys-timeline-adapter",
    "pytorch_profile": "pytorch-profile-adapter",
}
_DIAGNOSTIC_SIGNALS = {
    "nsys_timeline": {
        "launch_gap_short_context": (
            "runtime.launch_gap_short_context",
            ["cpu-submit", "gpu-kernel"],
        ),
        "gpu_idle_gap": ("runtime.gpu_idle_gap", ["gpu-kernel"]),
        "cpu_launch_overhead": ("runtime.cpu_launch_overhead", ["cpu-submit"]),
    },
    "pytorch_profile": {
        "gqa_head_ratio": (
            "framework.gqa_head_ratio",
            ["framework", "gpu-kernel"],
        ),
        "shape_fragmentation": ("framework.shape_fragmentation", ["framework"]),
        "framework_dispatch_overhead": (
            "framework.dispatch_overhead",
            ["framework", "cpu-submit"],
        ),
    },
}
_ACTIVE_ACTIONS = {
    "compiler-sass-inspection": "compiler_sass",
    "direction-experiment-project-copy": "direction_experiment",
    "ncu-targeted-kernel": "ncu_kernel",
    "nsys-global-timeline": "nsys_timeline",
    "nsys-os-runtime-slice": "os_runtime",
    "pytorch-operator-trace": "framework_trace",
}


def _closed(value: object, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strings(
    value: object, label: str, *, nonempty: bool = True, allowed: set[str] | None = None
) -> list[str]:
    if type(value) is not list or (nonempty and not value):
        raise ValueError(f"{label} must be a{' non-empty' if nonempty else ''} string list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    if allowed is not None and not set(value).issubset(allowed):
        raise ValueError(f"{label} contains unsupported values")
    return value


def _iso_date(value: object, label: str) -> datetime.date:
    try:
        parsed = datetime.date.fromisoformat(_text(value, label))
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must be a valid ISO date")
    return parsed


def _sha256(value: object, label: str) -> str:
    value = _text(value, label)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_reference(reference_dir: Path, filename: str) -> dict:
    root = Path(os.path.abspath(reference_dir))
    try:
        root_status = os.lstat(root)
    except OSError as exc:
        raise ValueError(f"knowledge reference root is unavailable: {root}") from exc
    if not stat.S_ISDIR(root_status.st_mode) or stat.S_ISLNK(root_status.st_mode):
        raise ValueError("knowledge reference root is a symlink or unsafe directory")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    root_fd = file_fd = None
    try:
        root_fd = os.open(root, directory_flags)
        file_fd = os.open(filename, flags, dir_fd=root_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_REFERENCE_BYTES:
            raise ValueError(f"knowledge reference is not a bounded regular file: {filename}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, _MAX_REFERENCE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_REFERENCE_BYTES:
                raise ValueError(f"knowledge reference exceeds size limit: {filename}")
        after = os.fstat(file_fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"knowledge reference changed while reading: {filename}")
    except OSError as exc:
        raise ValueError(
            f"knowledge reference contains a symlink or unsafe component: {filename}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if root_fd is not None:
            os.close(root_fd)
    try:
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"knowledge reference is invalid JSON: {filename}") from exc
    if type(value) is not dict:
        raise ValueError(f"knowledge reference root must be an object: {filename}")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation_identifier(value: object, label: str) -> str:
    value = _text(value, label)
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _observation_value(value: object, label: str) -> object:
    if value is None or type(value) in {bool, str}:
        return value
    if type(value) in {int, float} and math.isfinite(value):
        return value
    raise ValueError(f"{label} must be a finite scalar")


def _append_observation(observations: dict[tuple, dict], item: dict) -> None:
    key = (
        item["semantic_id"],
        tuple(item["scope"]),
        item["source_digest"],
    )
    previous = observations.get(key)
    if previous is not None and previous != item:
        raise ValueError(
            "semantic observation conflict for the same semantic_id, scope, "
            "and source_digest"
        )
    observations[key] = item


def normalize_observations(
    *,
    diagnostic_evidence: Sequence[Mapping[str, object]] = (),
    active_evidence_results: Sequence[Mapping[str, object]] = (),
) -> list[dict]:
    """Convert already-validated evidence into stable semantic observations."""
    if isinstance(diagnostic_evidence, (str, bytes)) or not isinstance(
        diagnostic_evidence, Sequence
    ):
        raise ValueError("diagnostic_evidence must be a sequence")
    if isinstance(active_evidence_results, (str, bytes)) or not isinstance(
        active_evidence_results, Sequence
    ):
        raise ValueError("active_evidence_results must be a sequence")

    normalized: dict[tuple, dict] = {}
    diagnostic_fields = {
        "kind",
        "layer",
        "summary",
        "signals",
        "producer",
        "adapter_request_sha256",
        "recorded_at",
        "subject",
        "result",
    }
    for index, raw in enumerate(diagnostic_evidence):
        item = _closed(raw, diagnostic_fields, f"diagnostic evidence {index}")
        kind = item["kind"]
        if kind not in _DIAGNOSTIC_PRODUCERS:
            raise ValueError(f"diagnostic evidence {index} has unknown producer kind")
        producer = _closed(
            item["producer"],
            {"id", "version", "implementation_sha256"},
            f"diagnostic evidence {index} producer",
        )
        if (
            producer["id"] != _DIAGNOSTIC_PRODUCERS[kind]
            or producer["version"] != "1.0.0"
        ):
            raise ValueError(f"diagnostic evidence {index} producer/version is unknown")
        implementation_sha = _sha256(
            producer["implementation_sha256"],
            f"diagnostic evidence {index} producer implementation_sha256",
        )
        request_sha = _sha256(
            item["adapter_request_sha256"],
            f"diagnostic evidence {index} adapter_request_sha256",
        )
        subject = _closed(
            item["subject"],
            {"target_sha256"},
            f"diagnostic evidence {index} subject",
        )
        _sha256(
            subject["target_sha256"],
            f"diagnostic evidence {index} subject target_sha256",
        )
        result = _closed(
            item["result"],
            {"artifact_sha256", "events_total"},
            f"diagnostic evidence {index} result",
        )
        artifact_sha = _sha256(
            result["artifact_sha256"],
            f"diagnostic evidence {index} result artifact_sha256",
        )
        if type(result["events_total"]) is not int or result["events_total"] < 1:
            raise ValueError(
                f"diagnostic evidence {index} result events_total must be positive"
            )
        if item["layer"] != "workload":
            raise ValueError(f"diagnostic evidence {index} layer is unsupported")
        _text(item["summary"], f"diagnostic evidence {index} summary")
        if (
            type(item["recorded_at"]) not in {int, float}
            or not math.isfinite(item["recorded_at"])
            or item["recorded_at"] < 0
        ):
            raise ValueError(f"diagnostic evidence {index} recorded_at is invalid")
        signals = _strings(
            item["signals"],
            f"diagnostic evidence {index} signals",
            nonempty=False,
        )
        unknown_signals = set(signals) - set(_DIAGNOSTIC_SIGNALS[kind])
        if unknown_signals:
            raise ValueError(
                f"diagnostic evidence {index} has unknown signals: "
                f"{sorted(unknown_signals)}"
            )
        source_digest = _canonical_sha256(
            {
                "kind": kind,
                "producer": {
                    "id": producer["id"],
                    "version": producer["version"],
                    "implementation_sha256": implementation_sha,
                },
                "adapter_request_sha256": request_sha,
                "report_artifact_sha256": artifact_sha,
            }
        )
        for signal in signals:
            semantic_id, scope = _DIAGNOSTIC_SIGNALS[kind][signal]
            _append_observation(
                normalized,
                {
                    "semantic_id": semantic_id,
                    "status": "present",
                    "value": True,
                    "unit": "state",
                    "scope": list(scope),
                    "aggregation": "presence",
                    "tool": {
                        "name": producer["id"],
                        "version": producer["version"],
                    },
                    "quality": "validated",
                    "source_digest": source_digest,
                },
            )

    envelope_fields = {
        "action_id",
        "evidence_kind",
        "adapter_implementation_sha256",
        "result_sha256",
        "status",
        "observations",
    }
    semantic_fields = {
        "semantic_id",
        "status",
        "value",
        "unit",
        "scope",
        "aggregation",
        "tool",
        "quality",
    }
    for index, raw in enumerate(active_evidence_results):
        envelope = _closed(raw, envelope_fields, f"active evidence result {index}")
        action_id = _observation_identifier(
            envelope["action_id"], f"active evidence result {index} action_id"
        )
        evidence_kind = _observation_identifier(
            envelope["evidence_kind"],
            f"active evidence result {index} evidence_kind",
        )
        if _ACTIVE_ACTIONS.get(action_id) != evidence_kind:
            raise ValueError(f"active evidence result {index} action identity is unknown")
        _sha256(
            envelope["adapter_implementation_sha256"],
            f"active evidence result {index} adapter identity SHA",
        )
        result_sha = _sha256(
            envelope["result_sha256"],
            f"active evidence result {index} result_sha256",
        )
        if envelope["status"] not in {
            "observed",
            "inconclusive",
            "unavailable",
            "failed",
        }:
            raise ValueError(f"active evidence result {index} status is unsupported")
        observations = envelope["observations"]
        if type(observations) is not dict:
            raise ValueError(f"active evidence result {index} observations must be an object")
        semantic_items = observations.get("semantic_observations")
        if semantic_items is None:
            continue
        if type(semantic_items) is not list:
            raise ValueError(
                f"active evidence result {index} semantic_observations must be an array"
            )
        for semantic_index, raw_semantic in enumerate(semantic_items):
            label = (
                f"active evidence result {index} semantic observation {semantic_index}"
            )
            semantic = _closed(raw_semantic, semantic_fields, label)
            semantic_id = _observation_identifier(
                semantic["semantic_id"], f"{label} semantic_id"
            )
            status = _observation_identifier(semantic["status"], f"{label} status")
            if status not in {"observed", "present", "absent", "unavailable"}:
                raise ValueError(f"{label} status is unsupported")
            value = _observation_value(semantic["value"], f"{label} value")
            unit = _text(semantic["unit"], f"{label} unit")
            scope = _strings(semantic["scope"], f"{label} scope")
            aggregation = _text(
                semantic["aggregation"], f"{label} aggregation"
            )
            tool = _closed(semantic["tool"], {"name", "version"}, f"{label} tool")
            tool_value = {
                "name": _text(tool["name"], f"{label} tool name"),
                "version": _text(tool["version"], f"{label} tool version"),
            }
            quality = _text(semantic["quality"], f"{label} quality")
            if value == "ERR_NVGPUCTRPERM":
                semantic_id = "profile.counter_access"
                status = "unavailable"
                unit = "state"
                scope = ["profile"]
                aggregation = "presence"
            _append_observation(
                normalized,
                {
                    "semantic_id": semantic_id,
                    "status": status,
                    "value": value,
                    "unit": unit,
                    "scope": list(scope),
                    "aggregation": aggregation,
                    "tool": tool_value,
                    "quality": quality,
                    "source_digest": result_sha,
                },
            )

    return [
        normalized[key]
        for key in sorted(
            normalized,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]


def validate_knowledge_package(reference_dir: Path = REFERENCE_DIR) -> dict:
    """Read and cross-check the offline knowledge references without side effects."""
    sources = _read_reference(reference_dir, "knowledge_sources.json")
    cards = _read_reference(reference_dir, "diagnostic_cards.json")
    cases = _read_reference(reference_dir, "case_memory.json")
    actions = _read_reference(reference_dir, "evidence_action_catalog.json")

    _closed(sources, {"schema_version", "as_of", "staleness_policy", "sources"}, "knowledge sources")
    if sources["schema_version"] != "cuda-optimizer/knowledge-sources-v1":
        raise ValueError("knowledge source schema is unsupported")
    source_as_of = _iso_date(sources["as_of"], "knowledge sources as_of")
    _text(sources["staleness_policy"], "knowledge sources staleness_policy")
    source_items = sources["sources"]
    if type(source_items) is not list or len(source_items) != 14:
        raise ValueError("knowledge sources must contain exactly 14 entries")
    source_fields = {
        "id",
        "title",
        "url",
        "version",
        "source_kind",
        "locator",
        "summary",
        "summary_sha256",
        "last_verified",
        "status",
    }
    source_ids: set[str] = set()
    for index, item in enumerate(source_items):
        item = _closed(item, source_fields, f"knowledge source {index}")
        source_id = _text(item["id"], f"knowledge source {index} id")
        if source_id in source_ids:
            raise ValueError(f"duplicate knowledge source id: {source_id}")
        source_ids.add(source_id)
        for field in ("title", "version", "locator", "summary"):
            _text(item[field], f"knowledge source {source_id} {field}")
        if not _text(item["url"], f"knowledge source {source_id} url").startswith(
            ("https://", "http://")
        ):
            raise ValueError(f"knowledge source {source_id} url must be HTTP(S)")
        if item["source_kind"] != "primary":
            raise ValueError(f"knowledge source {source_id} must be primary")
        if item["status"] != "verified":
            raise ValueError(f"knowledge source {source_id} status must be verified")
        expected_summary_sha = hashlib.sha256(item["summary"].encode("utf-8")).hexdigest()
        if _sha256(
            item["summary_sha256"], f"knowledge source {source_id} summary_sha256"
        ) != expected_summary_sha:
            raise ValueError(f"knowledge source {source_id} summary_sha256 does not match")
        if _iso_date(
            item["last_verified"], f"knowledge source {source_id} last_verified"
        ) > source_as_of:
            raise ValueError(f"knowledge source {source_id} is verified after as_of")

    _closed(actions, {"schema_version", "catalog_id", "actions"}, "evidence action catalog")
    if actions["schema_version"] != "cuda-optimizer/evidence-action-catalog-v1":
        raise ValueError("evidence action catalog schema is unsupported")
    action_items = actions["actions"]
    if type(action_items) is not list or not action_items:
        raise ValueError("evidence action catalog is empty")
    action_fields = {
        "action_id",
        "evidence_kind",
        "required_capability_ids",
        "cost",
        "perturbation",
        "risk",
        "control_scope",
        "repeatable",
    }
    action_scopes: dict[str, str] = {}
    for index, item in enumerate(action_items):
        item = _closed(item, action_fields, f"evidence action {index}")
        action_id = _text(item["action_id"], f"evidence action {index} action_id")
        if action_id in action_scopes:
            raise ValueError(f"duplicate evidence action id: {action_id}")
        _text(item["evidence_kind"], f"evidence action {action_id} evidence_kind")
        _strings(
            item["required_capability_ids"],
            f"evidence action {action_id} required_capability_ids",
        )
        for field in ("cost", "perturbation", "risk"):
            _text(item[field], f"evidence action {action_id} {field}")
        if item["control_scope"] not in {"read_only", "project_copy"}:
            raise ValueError(f"evidence action {action_id} control_scope is unsupported")
        if type(item["repeatable"]) is not bool:
            raise ValueError(f"evidence action {action_id} repeatable must be boolean")
        action_scopes[action_id] = item["control_scope"]

    _closed(cases, {"schema_version", "cases"}, "case memory")
    if cases["schema_version"] != "cuda-optimizer/case-memory-v1":
        raise ValueError("case memory schema is unsupported")
    case_items = cases["cases"]
    if type(case_items) is not list or len(case_items) != len(_CASE_IDS):
        raise ValueError("case memory ids are incomplete")
    case_fields = {
        "id",
        "replay_case_id",
        "replay_status",
        "replay_case_sha256",
        "identity_match",
        "predecision_evidence_sha256",
        "knowledge_identity_sha256",
        "outcome_type",
        "source_ids",
        "content_status",
    }
    case_by_id: dict[str, dict] = {}
    for index, item in enumerate(case_items):
        item = _closed(item, case_fields, f"case memory entry {index}")
        case_id = _text(item["id"], f"case memory entry {index} id")
        if case_id in case_by_id:
            raise ValueError(f"duplicate case memory id: {case_id}")
        case_by_id[case_id] = item
        if item["replay_case_id"] != case_id:
            raise ValueError("case memory ids must match replay_case_id")
        if item["replay_status"] not in {
            "scoreable",
            "partial",
            "rejection_only",
            "protocol_only",
        }:
            raise ValueError(f"case memory {case_id} replay_status is unsupported")
        _sha256(item["replay_case_sha256"], f"case memory {case_id} replay_case_sha256")
        _sha256(
            item["predecision_evidence_sha256"],
            f"case memory {case_id} predecision_evidence_sha256",
        )
        if item["knowledge_identity_sha256"] is not None:
            _sha256(
                item["knowledge_identity_sha256"],
                f"case memory {case_id} knowledge_identity_sha256",
            )
        if item["identity_match"] not in {"unknown", "analogous", "exact"}:
            raise ValueError(f"case memory {case_id} identity_match is unsupported")
        if item["outcome_type"] not in {
            "historical_outcome",
            "rejection",
            "protocol_only",
        }:
            raise ValueError(f"case memory {case_id} outcome_type is unsupported")
        referenced_sources = _strings(
            item["source_ids"], f"case memory {case_id} source_ids"
        )
        if not set(referenced_sources).issubset(source_ids):
            raise ValueError(f"case memory {case_id} references unknown source")
        if item["content_status"] not in {
            "source_verified",
            "replay_verified",
            "locally_measured",
        }:
            raise ValueError(f"case memory {case_id} content_status is unsupported")
        if (
            item["content_status"] in {"replay_verified", "locally_measured"}
            and item["replay_status"] not in {"scoreable", "rejection_only"}
        ):
            raise ValueError(
                f"case memory {case_id} {item['content_status']} requires a "
                "supporting or rejecting replay"
            )
        if item["content_status"] == "locally_measured" and (
            item["identity_match"] != "exact"
            or item["knowledge_identity_sha256"] is None
        ):
            raise ValueError(
                f"case memory {case_id} locally_measured requires exact identity "
                "and knowledge_identity_sha256"
            )
    if set(case_by_id) != _CASE_IDS:
        raise ValueError("case memory ids are incomplete")

    _closed(cards, {"schema_version", "as_of", "cards"}, "diagnostic cards")
    if cards["schema_version"] != "cuda-optimizer/diagnostic-cards-v1":
        raise ValueError("diagnostic card schema is unsupported")
    cards_as_of = _iso_date(cards["as_of"], "diagnostic cards as_of")
    card_items = cards["cards"]
    if type(card_items) is not list or len(card_items) != 7:
        raise ValueError("diagnostic cards must contain exactly 7 entries")
    card_fields = {
        "id",
        "mechanism_key",
        "status",
        "content_status",
        "categories",
        "execution_layers",
        "priority",
        "match_terms",
        "applies_when",
        "mechanism",
        "competing_explanations",
        "distinguishing_question",
        "preferred_actions",
        "cheapest_falsifier",
        "required_evidence",
        "positive_signals",
        "counter_signals",
        "invalidators",
        "source_ids",
        "case_ids",
        "last_reviewed",
    }
    card_ids: set[str] = set()
    mechanism_keys: set[str] = set()
    runtime_candidates = []
    for index, item in enumerate(card_items):
        item = _closed(item, card_fields, f"diagnostic card {index}")
        card_id = _text(item["id"], f"diagnostic card {index} id")
        mechanism_key = _text(
            item["mechanism_key"], f"diagnostic card {card_id} mechanism_key"
        )
        if card_id in card_ids or mechanism_key in mechanism_keys:
            raise ValueError("diagnostic card ids and mechanism_key values must be unique")
        card_ids.add(card_id)
        mechanism_keys.add(mechanism_key)
        if item["status"] != "routing_only":
            raise ValueError(f"diagnostic card {card_id} status must preserve routing_only")
        if item["content_status"] not in {
            "source_verified",
            "replay_verified",
            "locally_measured",
        }:
            raise ValueError(f"diagnostic card {card_id} content_status is unsupported")
        _strings(item["categories"], f"diagnostic card {card_id} categories")
        _strings(
            item["execution_layers"],
            f"diagnostic card {card_id} execution_layers",
            allowed=_LAYERS,
        )
        if type(item["priority"]) is not int or isinstance(item["priority"], bool):
            raise ValueError(f"diagnostic card {card_id} priority must be an integer")
        _strings(
            item["match_terms"],
            f"diagnostic card {card_id} match_terms",
            nonempty=False,
        )
        for field in (
            "applies_when",
            "competing_explanations",
            "required_evidence",
            "positive_signals",
            "counter_signals",
            "invalidators",
        ):
            _strings(item[field], f"diagnostic card {card_id} {field}")
        for field in ("mechanism", "distinguishing_question"):
            _text(item[field], f"diagnostic card {card_id} {field}")
        preferred_actions = _strings(
            item["preferred_actions"], f"diagnostic card {card_id} preferred_actions"
        )
        if not set(preferred_actions).issubset(action_scopes):
            raise ValueError(f"diagnostic card {card_id} references unknown action")
        falsifier = _closed(
            item["cheapest_falsifier"],
            {"action_id", "rationale"},
            f"diagnostic card {card_id} cheapest_falsifier",
        )
        falsifier_id = _text(
            falsifier["action_id"],
            f"diagnostic card {card_id} cheapest_falsifier action_id",
        )
        if falsifier_id not in action_scopes:
            raise ValueError(f"diagnostic card {card_id} references unknown action")
        if action_scopes[falsifier_id] != "read_only":
            raise ValueError(f"diagnostic card {card_id} cheapest falsifier must be read_only")
        _text(
            falsifier["rationale"],
            f"diagnostic card {card_id} cheapest_falsifier rationale",
        )
        referenced_sources = _strings(
            item["source_ids"], f"diagnostic card {card_id} source_ids"
        )
        if not set(referenced_sources).issubset(source_ids):
            raise ValueError(f"diagnostic card {card_id} references unknown source")
        referenced_cases = _strings(
            item["case_ids"], f"diagnostic card {card_id} case_ids", nonempty=False
        )
        if not set(referenced_cases).issubset(case_by_id):
            raise ValueError(f"diagnostic card {card_id} references unknown case")
        if item["content_status"] == "replay_verified":
            eligible_replays = [
                case_by_id[case_id]
                for case_id in referenced_cases
                if (
                    case_by_id[case_id]["replay_status"]
                    in {"scoreable", "rejection_only"}
                    and case_by_id[case_id]["content_status"]
                    in {"replay_verified", "locally_measured"}
                )
            ]
            if not eligible_replays:
                raise ValueError(
                    f"diagnostic card {card_id} replay_verified requires a "
                    "non-protocol supporting or rejecting replay"
                )
            runtime_candidates.append(card_id)
        if item["content_status"] == "locally_measured":
            locally_measured_cases = [
                case_by_id[case_id]
                for case_id in referenced_cases
                if (
                    case_by_id[case_id]["content_status"] == "locally_measured"
                    and case_by_id[case_id]["replay_status"]
                    in {"scoreable", "rejection_only"}
                    and case_by_id[case_id]["identity_match"] == "exact"
                    and case_by_id[case_id]["knowledge_identity_sha256"] is not None
                )
            ]
            if not locally_measured_cases:
                raise ValueError(
                    f"diagnostic card {card_id} locally_measured requires an exact "
                    "locally_measured case with a complete identity digest"
                )
            runtime_candidates.append(card_id)
        if _iso_date(
            item["last_reviewed"], f"diagnostic card {card_id} last_reviewed"
        ) > cards_as_of:
            raise ValueError(f"diagnostic card {card_id} is reviewed after as_of")

    package = {"sources": sources, "cards": cards, "cases": cases, "actions": actions}
    return {
        "status": "passed",
        "package_version": "cuda-optimizer/knowledge-package-v1",
        "source_count": len(source_items),
        "card_count": len(card_items),
        "case_count": len(case_items),
        "runtime_candidate_card_ids": sorted(runtime_candidates),
        "content_sha256": _canonical_sha256(package),
    }


def _load_cards() -> list[dict]:
    value = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "cuda-optimizer/diagnostic-cards-v1":
        raise ValueError("diagnostic card registry schema is unsupported")
    cards = value.get("cards")
    if type(cards) is not list or not cards:
        raise ValueError("diagnostic card registry is empty")
    return cards


def route_cards(
    diagnosis: Mapping[str, object],
    execution_map: Mapping[str, object],
    *,
    limit: int = 3,
) -> dict:
    if type(limit) is not int or isinstance(limit, bool) or not 1 <= limit <= 5:
        raise ValueError("limit must be between 1 and 5")
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
    ranked_cards = []
    for card in _load_cards():
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
            ((category_rank, 0 if term_match else 1, card["priority"], card["id"]), card)
        )
    if not ranked_cards:
        fallback = next(
            card for card in _load_cards() if card["id"] == "diagnostic.cross-layer.triage"
        )
        ranked_cards = [((0, 0, 0, fallback["id"]), fallback)]
    ranked_cards.sort(key=lambda item: item[0])
    return {
        "schema_version": "cuda-optimizer/diagnostic-knowledge-context-v1",
        "categories": categories,
        "cards": [copy.deepcopy(card) for _, card in ranked_cards[:limit]],
        "promotion_authority": "none",
    }
