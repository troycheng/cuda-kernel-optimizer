#!/usr/bin/env python3
"""Return a bounded, identity-compatible set of local knowledge cards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path


INPUT_VERSION = "cuda-kernel-optimizer/knowledge-input-v1"
REFERENCE_DIR = Path(__file__).resolve().parent.parent / "references"
KNOWLEDGE_DIR = REFERENCE_DIR / "knowledge"
CARDS_PATH = KNOWLEDGE_DIR / "cards.json"
SOURCES_PATH = KNOWLEDGE_DIR / "sources.json"

_COMMON_INPUT_FIELDS = {"format_version", "operation", "filters", "limits"}
_DETACHED_INPUT_FIELDS = _COMMON_INPUT_FIELDS | {"identity"}
_TARGET_INPUT_FIELDS = _COMMON_INPUT_FIELDS | {
    "artifact_root",
    "target_ref",
    "phenomena",
}
_IDENTITY_FIELDS = {
    "gpu_architecture",
    "cuda_version",
    "frameworks",
    "phenomena",
    "claim_layer",
}
_FILTER_FIELDS = {"mechanism_keys"}
_LIMIT_FIELDS = {"max_results", "max_context_bytes"}
_TARGET_REF_FIELDS = {"id", "sha256"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class KnowledgeError(ValueError):
    pass


def _closed(value, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise KnowledgeError(f"{label} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise KnowledgeError(
            f"{label} has missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    return value


def _text(value, label: str, *, maximum=4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise KnowledgeError(f"{label} must be a non-empty bounded string")
    return value


def _strings(value, label: str) -> list[str]:
    if type(value) is not list or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise KnowledgeError(f"{label} must be a string list")
    if len(value) != len(set(value)):
        raise KnowledgeError(f"{label} must not contain duplicates")
    return list(value)


def _strict_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeError(f"JSON source is invalid: {path}") from error
    if type(value) is not dict:
        raise KnowledgeError(f"JSON source must contain an object: {path}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise KnowledgeError(f"JSON source has duplicate key: {key}")
        value[key] = item
    return value


def _knowledge_json(filename: str) -> tuple[dict, str]:
    root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = file_fd = None
    try:
        root_fd = os.open(KNOWLEDGE_DIR, root_flags)
        file_fd = os.open(filename, file_flags, dir_fd=root_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise KnowledgeError(f"knowledge source is not a regular file: {filename}")
        parts = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            parts.append(chunk)
        after = os.fstat(file_fd)
    except OSError as error:
        raise KnowledgeError(f"knowledge source is unavailable or unsafe: {filename}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if root_fd is not None:
            os.close(root_fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise KnowledgeError(f"knowledge source changed while reading: {filename}")
    try:
        payload = b"".join(parts)
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeError(f"knowledge source is invalid JSON: {filename}") from error
    if type(value) is not dict:
        raise KnowledgeError(f"knowledge source must contain an object: {filename}")
    return value, hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise KnowledgeError("knowledge value is not finite JSON") from error


def _validate_request(value) -> dict:
    if type(value) is not dict:
        raise KnowledgeError("knowledge input must be an object")
    if set(value) == _DETACHED_INPUT_FIELDS:
        request = value
        identity_source = {"kind": "detached"}
        identity = _closed(request["identity"], _IDENTITY_FIELDS, "identity")
    elif set(value) == _TARGET_INPUT_FIELDS:
        request = value
        root = Path(request["artifact_root"]).expanduser().resolve()
        target_ref = _closed(
            request["target_ref"], _TARGET_REF_FIELDS, "target_ref"
        )
        try:
            payload = (root / "target.json").read_bytes()
            target = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise KnowledgeError("target record is unavailable or invalid") from error
        if (
            hashlib.sha256(payload).hexdigest() != target_ref["sha256"]
            or type(target) is not dict
            or target.get("record_type") != "target"
            or target.get("id") != target_ref["id"]
        ):
            raise KnowledgeError("target reference does not match target record")
        runtime = target.get("environment", {}).get("runtime", {})
        architectures = runtime.get("gpu_architectures", [])
        if not architectures or len(set(architectures)) != 1:
            raise KnowledgeError(
                "target knowledge query requires one exact GPU architecture"
            )
        identity = {
            "gpu_architecture": architectures[0],
            "cuda_version": runtime.get("cuda_runtime_version"),
            "frameworks": runtime.get("frameworks"),
            "phenomena": request["phenomena"],
            "claim_layer": target.get("claim_layer"),
        }
        identity_source = {
            "kind": "target",
            "target_ref": dict(target_ref),
        }
    else:
        raise KnowledgeError(
            "knowledge input must contain either identity or target_ref fields"
        )
    if request["format_version"] != INPUT_VERSION or request["operation"] != "query":
        raise KnowledgeError("knowledge input version or operation is unsupported")
    frameworks = identity["frameworks"]
    if type(frameworks) is not dict or any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in frameworks.items()
    ):
        raise KnowledgeError("identity.frameworks must map names to versions")
    phenomena = _strings(identity["phenomena"], "identity.phenomena")
    filters = _closed(request["filters"], _FILTER_FIELDS, "filters")
    mechanism_keys = _strings(
        filters["mechanism_keys"], "filters.mechanism_keys"
    )
    limits = _closed(request["limits"], _LIMIT_FIELDS, "limits")
    max_results = limits["max_results"]
    max_context = limits["max_context_bytes"]
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= 50
    ):
        raise KnowledgeError("limits.max_results must be between 1 and 50")
    if (
        isinstance(max_context, bool)
        or not isinstance(max_context, int)
        or not 256 <= max_context <= 1024 * 1024
    ):
        raise KnowledgeError(
            "limits.max_context_bytes must be between 256 and 1048576"
        )
    if identity["claim_layer"] not in {
        "diagnostic",
        "kernel",
        "workload",
        "serving",
    }:
        raise KnowledgeError("identity.claim_layer is unsupported")
    return {
        "identity": {
            "gpu_architecture": _text(
                identity["gpu_architecture"],
                "identity.gpu_architecture",
                maximum=64,
            ),
            "cuda_version": _text(
                identity["cuda_version"],
                "identity.cuda_version",
                maximum=64,
            ),
            "frameworks": dict(sorted(frameworks.items())),
            "phenomena": phenomena,
            "claim_layer": identity["claim_layer"],
        },
        "mechanism_keys": mechanism_keys,
        "max_results": max_results,
        "max_context_bytes": max_context,
        "identity_source": identity_source,
    }


def _load_knowledge() -> tuple[list[dict], dict[str, dict], dict]:
    card_document, cards_sha256 = _knowledge_json("cards.json")
    source_document, sources_sha256 = _knowledge_json("sources.json")
    if (
        card_document.get("schema_version")
        != "cuda-kernel-optimizer/knowledge-cards-v1"
        or type(card_document.get("cards")) is not list
    ):
        raise KnowledgeError("knowledge card registry version is unsupported")
    if (
        source_document.get("schema_version")
        != "cuda-kernel-optimizer/knowledge-sources-v1"
        or type(source_document.get("sources")) is not list
    ):
        raise KnowledgeError("knowledge source registry version is unsupported")
    sources = {}
    for source in source_document["sources"]:
        if type(source) is not dict:
            raise KnowledgeError("knowledge source entry must be an object")
        source_id = _text(source.get("id"), "knowledge source id", maximum=256)
        if source_id in sources:
            raise KnowledgeError("knowledge source ids must be unique")
        if source.get("status") != "verified":
            raise KnowledgeError(f"knowledge source is not verified: {source_id}")
        if _SHA256.fullmatch(str(source.get("summary_sha256", ""))) is None:
            raise KnowledgeError(f"knowledge source digest is invalid: {source_id}")
        sources[source_id] = source
    cards = []
    seen = set()
    for card in card_document["cards"]:
        if type(card) is not dict:
            raise KnowledgeError("knowledge card entry must be an object")
        card_id = _text(card.get("id"), "knowledge card id", maximum=256)
        mechanism = _text(
            card.get("mechanism_key"),
            "knowledge mechanism_key",
            maximum=256,
        )
        if card_id in seen:
            raise KnowledgeError("knowledge card ids must be unique")
        seen.add(card_id)
        source_ids = _strings(card.get("source_ids"), f"{card_id}.source_ids")
        missing = sorted(set(source_ids) - set(sources))
        if missing:
            raise KnowledgeError(
                f"knowledge card has unknown sources: {card_id}: {missing}"
            )
        if not isinstance(card.get("priority"), int):
            raise KnowledgeError(f"knowledge card priority is invalid: {card_id}")
        if mechanism:
            cards.append(card)
    provenance = {
        "cards_sha256": cards_sha256,
        "sources_sha256": sources_sha256,
    }
    return cards, sources, provenance


def _observation_ids(card: dict) -> set[str]:
    rules = card.get("observation_rules")
    if type(rules) is not dict:
        return set()
    identifiers = set()
    for group in ("positive", "counter", "invalidators"):
        entries = rules.get(group, [])
        if type(entries) is not list:
            raise KnowledgeError(
                f"knowledge card observation rules are invalid: {card.get('id')}"
            )
        for entry in entries:
            if type(entry) is dict and isinstance(entry.get("semantic_id"), str):
                identifiers.add(entry["semantic_id"])
    return identifiers


def _identity_matches(card: dict, identity: dict) -> bool:
    constraints = card.get("identity_constraints")
    if type(constraints) is not dict:
        raise KnowledgeError(
            f"knowledge card identity constraints are invalid: {card.get('id')}"
        )
    architectures = constraints.get("gpu_architecture", [])
    if architectures and identity["gpu_architecture"] not in architectures:
        return False
    cuda_versions = constraints.get("cuda_runtime_version", [])
    if cuda_versions and identity["cuda_version"] not in cuda_versions:
        return False
    framework_constraints = constraints.get("framework_versions", {})
    if type(framework_constraints) is not dict:
        raise KnowledgeError(
            f"knowledge card framework constraints are invalid: {card.get('id')}"
        )
    for name, allowed in framework_constraints.items():
        if name not in identity["frameworks"]:
            return False
        if allowed and identity["frameworks"][name] not in allowed:
            return False
    claim_layers = constraints.get("claim_layers", [])
    if (
        type(claim_layers) is not list
        or any(layer not in {"diagnostic", "kernel", "workload", "serving"} for layer in claim_layers)
    ):
        raise KnowledgeError(
            f"knowledge card claim layers are invalid: {card.get('id')}"
        )
    if claim_layers and identity["claim_layer"] not in claim_layers:
        return False
    return True


def _phenomena_match(card: dict, phenomena: list[str]) -> bool:
    if not phenomena:
        return True
    exact = _observation_ids(card)
    searchable = {
        str(term).lower() for term in card.get("match_terms", []) if term
    }
    searchable.add(str(card.get("mechanism_key", "")).lower())
    for phenomenon in phenomena:
        if phenomenon in exact or phenomenon.lower() in searchable:
            return True
    return False


def _projection(card: dict, sources: dict[str, dict]) -> dict:
    return {
        "id": card["id"],
        "mechanism_key": card["mechanism_key"],
        "status": card.get("status"),
        "priority": card["priority"],
        "mechanism": card.get("mechanism"),
        "distinguishing_question": card.get("distinguishing_question"),
        "cheapest_falsifier": card.get("cheapest_falsifier"),
        "required_evidence": card.get("required_evidence", []),
        "counter_signals": card.get("counter_signals", []),
        "invalidators": card.get("invalidators", []),
        "local_cases": card.get("local_cases", []),
        "content_kind": card.get("content_kind"),
        "sources": [
            {
                "id": source_id,
                "title": sources[source_id]["title"],
                "version": sources[source_id]["version"],
                "last_verified": sources[source_id]["last_verified"],
            }
            for source_id in card["source_ids"]
        ],
    }


def query(value) -> dict:
    request = _validate_request(value)
    cards, sources, provenance = _load_knowledge()
    mechanism_filter = set(request["mechanism_keys"])
    candidates = [
        card
        for card in cards
        if (not mechanism_filter or card["mechanism_key"] in mechanism_filter)
        and _identity_matches(card, request["identity"])
        and _phenomena_match(card, request["identity"]["phenomena"])
    ]
    candidates.sort(
        key=lambda card: (
            card["priority"],
            card["mechanism_key"],
            card["id"],
        )
    )
    matches = []
    seen_mechanisms = set()
    context_bytes = 2
    for card in candidates:
        if len(matches) >= request["max_results"]:
            break
        if card["mechanism_key"] in seen_mechanisms:
            continue
        projected = _projection(card, sources)
        proposed = matches + [projected]
        size = len(_canonical_bytes(proposed))
        if size > request["max_context_bytes"]:
            continue
        matches = proposed
        seen_mechanisms.add(card["mechanism_key"])
        context_bytes = size
    return {
        "status": "completed",
        "matches": matches,
        "context_bytes": context_bytes,
        "truncated": len(matches) < len(candidates),
        "provenance": provenance,
        "identity_source": request["identity_source"],
    }


def _emit_error(error: BaseException) -> int:
    print(
        json.dumps(
            {
                "status": "rejected",
                "error_code": "knowledge_query_invalid",
                "error": str(error)[:1024],
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Query bounded local GPU optimization knowledge."
    )
    parser.add_argument("operation", choices=("query",))
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        request = _strict_json(Path(args.request))
        if request.get("operation") != args.operation:
            raise KnowledgeError("CLI operation does not match request")
        result = query(request)
    except (KnowledgeError, OSError, ValueError) as error:
        return _emit_error(error)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
