#!/usr/bin/env python3
"""Inspect or atomically update the explicit V1.4 Champion reference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import stat
import sys
import time
from pathlib import Path


INPUT_VERSION = "cuda-kernel-optimizer/champion-input-v1"
_SHOW_FIELDS = {
    "format_version",
    "operation",
    "artifact_root",
    "target_ref",
}
_TARGET_REF_FIELDS = {"id", "sha256"}
_POINTER_FIELDS = {
    "record_type",
    "format_version",
    "target_ref",
    "selection_ref",
}
_SELECTION_REF_FIELDS = {"id", "sha256"}
_UPDATE_FIELDS = {
    "format_version",
    "operation",
    "artifact_root",
    "target_ref",
    "result_ref",
    "expected_selection_ref",
}
_RESULT_REF_FIELDS = {"invocation_id", "sha256"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_INVOCATION_ID = re.compile(r"inv-[a-z0-9-]+")


def _load_store():
    path = Path(__file__).with_name("artifact_store.py")
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_champion_store", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load artifact store: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_store()


class ChampionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _closed(value, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ChampionError("invalid_champion_input", f"{label} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise ChampionError(
            "invalid_champion_input",
            f"{label} has missing={sorted(missing)} unknown={sorted(unknown)}",
        )
    return value


def _strict_json(path) -> dict:
    try:
        value = json.loads(STORE.read_regular_bytes(path).decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ChampionError(
            "invalid_champion_input", "request is invalid JSON"
        ) from error
    if type(value) is not dict:
        raise ChampionError(
            "invalid_champion_input", "request root must be an object"
        )
    return value


def _target(root: Path, reference) -> dict:
    reference = _closed(reference, _TARGET_REF_FIELDS, "target_ref")
    path = root / "target.json"
    try:
        payload = STORE.read_regular_bytes(path)
    except (OSError, ValueError) as error:
        raise ChampionError("target_not_found", "target record is unavailable") from error
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ChampionError("target_changed", "target record digest changed")
    try:
        target = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ChampionError("target_invalid", "target record is invalid") from error
    if (
        type(target) is not dict
        or target.get("record_type") != "target"
        or target.get("id") != reference["id"]
    ):
        raise ChampionError("target_invalid", "target identity is invalid")
    return target


def _resolve_current(
    root: Path,
    target_ref: dict,
    original_variant: dict,
) -> tuple[dict, dict | None, str | None]:
    """Resolve the one current pointer to its immutable Selection record."""
    current = root / "champion" / "current.json"
    try:
        metadata = os.lstat(current)
    except FileNotFoundError:
        return original_variant, None, None
    except OSError as error:
        raise ChampionError(
            "selection_invalid", "Champion pointer is unavailable"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ChampionError(
            "selection_invalid", "Champion pointer is not a regular file"
        )
    try:
        pointer_payload = STORE.read_regular_bytes(current)
        pointer = json.loads(pointer_payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ChampionError(
            "selection_invalid", "Champion pointer is invalid"
        ) from error
    pointer = _closed(pointer, _POINTER_FIELDS, "Champion pointer")
    if (
        pointer["record_type"] != "champion_pointer"
        or pointer["format_version"]
        != "cuda-kernel-optimizer/champion-pointer-v1"
        or pointer["target_ref"] != target_ref
    ):
        raise ChampionError(
            "selection_invalid", "Champion pointer identity is inconsistent"
        )
    selection_ref = _closed(
        pointer["selection_ref"],
        _SELECTION_REF_FIELDS,
        "Champion selection_ref",
    )
    selection_id = selection_ref["id"]
    if (
        not isinstance(selection_id, str)
        or not selection_id.startswith("sel-")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in selection_id
        )
    ):
        raise ChampionError("selection_invalid", "Selection id is invalid")
    selection_path = root / "champion" / "selections" / f"{selection_id}.json"
    try:
        payload = STORE.read_regular_bytes(selection_path)
        selection = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ChampionError(
            "selection_invalid", "Selection record is unavailable or invalid"
        ) from error
    if hashlib.sha256(payload).hexdigest() != selection_ref["sha256"]:
        raise ChampionError("selection_invalid", "Selection digest changed")
    if (
        type(selection) is not dict
        or selection.get("record_type") != "champion_selection"
        or selection.get("id") != selection_id
        or selection.get("target_ref") != target_ref
        or type(selection.get("selected_variant")) is not dict
    ):
        raise ChampionError(
            "selection_invalid", "Selection record identity is inconsistent"
        )
    return (
        selection["selected_variant"],
        dict(selection_ref),
        hashlib.sha256(pointer_payload).hexdigest(),
    )


def resolve_current(
    root: Path,
    target_ref: dict,
    original_variant: dict,
) -> tuple[dict, dict | None]:
    variant, selection_ref, _ = _resolve_current(
        root,
        target_ref,
        original_variant,
    )
    return variant, selection_ref


def show(value) -> dict:
    request = _closed(value, _SHOW_FIELDS, "show input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != "show":
        raise ChampionError(
            "invalid_champion_input",
            "show input version or operation is unsupported",
        )
    root = Path(
        os.path.abspath(os.path.expanduser(os.fspath(request["artifact_root"])))
    )
    target = _target(root, request["target_ref"])
    variant, selection_ref = resolve_current(
        root,
        request["target_ref"],
        target["original"],
    )
    return {
        "status": "current",
        "target_ref": request["target_ref"],
        "selection_ref": selection_ref,
        "variant": variant,
    }


def _selection_ref_or_null(value, label: str) -> dict | None:
    if value is None:
        return None
    reference = _closed(value, _SELECTION_REF_FIELDS, label)
    selection_id = reference["id"]
    digest = reference["sha256"]
    if (
        not isinstance(selection_id, str)
        or not selection_id.startswith("sel-")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in selection_id
        )
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise ChampionError("invalid_champion_input", f"{label} is invalid")
    return dict(reference)


def _result(root: Path, reference) -> tuple[dict, dict]:
    reference = _closed(reference, _RESULT_REF_FIELDS, "result_ref")
    invocation_id = reference["invocation_id"]
    digest = reference["sha256"]
    if (
        not isinstance(invocation_id, str)
        or _INVOCATION_ID.fullmatch(invocation_id) is None
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise ChampionError("invalid_champion_input", "result_ref is invalid")
    path = root / "invocations" / invocation_id / "result.json"
    try:
        payload = STORE.read_regular_bytes(path)
    except (OSError, ValueError) as error:
        raise ChampionError("result_not_found", "invocation result is unavailable") from error
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ChampionError("result_changed", "invocation result digest changed")
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ChampionError("result_invalid", "invocation result is invalid JSON") from error
    if type(result) is not dict:
        raise ChampionError("result_invalid", "invocation result must be an object")
    return result, dict(reference)


def _valid_common_result(result: dict, target_ref: dict, operation: str) -> list[dict]:
    if (
        result.get("record_type") != "invocation_result"
        or result.get("format_version") != "cuda-kernel-optimizer/evaluator-result-v1"
        or result.get("operation") != operation
        or result.get("target_ref") != target_ref
        or result.get("cleanup_status") != "confirmed"
        or result.get("execution_status") != "succeeded"
        or result.get("measurement_validity") != "valid"
        or result.get("reference_status") != "current"
    ):
        raise ChampionError("result_invalid", "invocation result is not an eligible result")
    variants = result.get("variant_refs")
    receipt = result.get("performance_receipt")
    if (
        type(variants) is not list
        or len(variants) != 2
        or any(type(variant) is not dict for variant in variants)
        or type(receipt) is not dict
    ):
        raise ChampionError("result_invalid", "invocation result bindings are invalid")
    return variants


def _publish_selection(
    *,
    root: Path,
    target_ref: dict,
    selected_variant: dict,
    result_ref: dict,
    expected_selection_ref: dict | None,
    expected_pointer_digest: str | None,
    operation: str,
) -> dict:
    selection_id = f"sel-{secrets.token_hex(16)}"
    selection_path = root / "champion" / "selections" / f"{selection_id}.json"
    selection = {
        "record_type": "champion_selection",
        "format_version": "cuda-kernel-optimizer/champion-selection-v1",
        "id": selection_id,
        "target_ref": target_ref,
        "selected_variant": selected_variant,
        "decision": (
            "select_candidate"
            if operation == "select"
            else "restore_original"
        ),
        "comparison_result_ref": result_ref,
        "previous_selection_ref": expected_selection_ref,
        "created_at_epoch": time.time(),
    }
    STORE.create_regular_json(selection_path, selection)
    selection_ref = {
        "id": selection_id,
        "sha256": STORE.sha256_file(selection_path),
    }
    pointer = {
        "record_type": "champion_pointer",
        "format_version": "cuda-kernel-optimizer/champion-pointer-v1",
        "target_ref": target_ref,
        "selection_ref": selection_ref,
    }
    try:
        STORE.compare_and_swap_ref(
            root, "champion/current.json", expected_pointer_digest, pointer
        )
    except STORE.StaleReferenceError as error:
        raise ChampionError("stale_reference", "Champion changed before selection") from error
    return selection_ref


def _update(value, *, operation: str) -> dict:
    request = _closed(value, _UPDATE_FIELDS, f"{operation} input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != operation:
        raise ChampionError(
            "invalid_champion_input",
            f"{operation} input version or operation is unsupported",
        )
    root = Path(
        os.path.abspath(os.path.expanduser(os.fspath(request["artifact_root"])))
    )
    target = _target(root, request["target_ref"])
    if target.get("target_mode") != "optimization":
        raise ChampionError(
            "target_not_optimizable",
            "diagnostic Target cannot update Champion",
        )
    expected_selection_ref = _selection_ref_or_null(
        request["expected_selection_ref"], "expected_selection_ref"
    )
    (
        current_variant,
        current_selection_ref,
        expected_pointer_digest,
    ) = _resolve_current(
        root, request["target_ref"], target["original"]
    )
    if current_selection_ref != expected_selection_ref:
        raise ChampionError("stale_reference", "current Champion does not match expected_selection_ref")
    result, result_ref = _result(root, request["result_ref"])
    variants = _valid_common_result(
        result,
        request["target_ref"],
        "target" if operation == "select" else "final_audit",
    )
    if result.get("reference_selection_ref") != expected_selection_ref:
        raise ChampionError("result_invalid", "result selection binding does not match current Champion")

    if operation == "select":
        if (
            result.get("verdict") != "passed"
            or result.get("experiment_ref") is None
            or current_variant != variants[0]
            or result["performance_receipt"].get("status") != "valid"
            or result["performance_receipt"].get("reference") != variants[0]
            or result["performance_receipt"].get("candidate") != variants[1]
        ):
            raise ChampionError("result_invalid", "target result is not bound to current Champion")
        selected_variant = variants[1]
        status = "selected"
    else:
        performance = result["performance_receipt"]
        if (
            result.get("verdict") != "rejected"
            or result.get("restore_supported") is not True
            or variants[0] != target["original"]
            or current_variant != variants[1]
            or performance.get("status") not in {"not_run", "valid"}
            or (
                performance.get("status") == "valid"
                and (
                    performance.get("reference") != variants[0]
                    or performance.get("candidate") != variants[1]
                )
            )
        ):
            raise ChampionError("result_invalid", "final audit does not support restoring original")
        selected_variant = target["original"]
        status = "restored_original"

    selection_ref = _publish_selection(
        root=root,
        target_ref=request["target_ref"],
        selected_variant=selected_variant,
        result_ref=result_ref,
        expected_selection_ref=expected_selection_ref,
        expected_pointer_digest=expected_pointer_digest,
        operation=operation,
    )
    return {
        "status": status,
        "target_ref": request["target_ref"],
        "selection_ref": selection_ref,
        "variant": selected_variant,
    }


def select(value) -> dict:
    return _update(value, operation="select")


def restore_original(value) -> dict:
    return _update(value, operation="restore-original")


def _emit_error(error: BaseException) -> int:
    code = error.code if isinstance(error, ChampionError) else "champion_error"
    print(
        json.dumps(
            {
                "status": "rejected",
                "error_code": code,
                "error": str(error)[:1024],
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or update the explicit V1.4 Champion reference."
    )
    parser.add_argument(
        "operation",
        choices=("show", "select", "restore-original"),
    )
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        request = _strict_json(args.request)
        if request.get("operation") != args.operation:
            raise ChampionError(
                "invalid_champion_input",
                "CLI operation does not match request",
            )
        if args.operation == "show":
            result = show(request)
        elif args.operation == "select":
            result = select(request)
        else:
            result = restore_original(request)
    except (ChampionError, OSError, ValueError) as error:
        return _emit_error(error)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
