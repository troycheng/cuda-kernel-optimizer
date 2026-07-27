#!/usr/bin/env python3
"""Advisory JSON protocol for a user-supplied local reviewer CLI."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


REQUEST_SCHEMA = "cuda-workload-optimizer/review-request-v1"
SUMMARY_SCHEMA = "cuda-workload-optimizer/review-summary-v1"
RESPONSE_SCHEMA = "cuda-workload-optimizer/review-v1"
ARTIFACT_SCHEMA = "cuda-workload-optimizer/review-artifact-v1"
AGGREGATE_SCHEMA = "cuda-workload-optimizer/review-aggregate-v1"
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_PROVIDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_PROVIDER_PRIORITY = (
    "google-ai-mode",
    "github-copilot",
    "glm",
    "kimi",
    "deepseek",
    "gemini",
)
_PROVIDER_ALIASES = {
    "google-ai-mode": "google-ai-mode",
    "github-copilot": "github-copilot",
    "glm": "glm",
    "zhipu": "glm",
    "zhipu-qingyan": "glm",
    "kimi": "kimi",
    "deepseek": "deepseek",
    "gemini": "gemini",
}
_TRIGGER_TARGETS = {"ordinary": 1, "major": 2, "plateau": 3, "final": 3}
_SECRET_NAME = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|cookie|credential|password|secret|token)($|[_-])",
    re.IGNORECASE,
)
_SECRET_LOG = re.compile(
    r'''(?i)(["']?\b[A-Z0-9_]{0,128}(?:API[_-]?KEY|AUTH|COOKIE|CREDENTIAL|PASSWORD|SECRET|TOKEN)[A-Z0-9_]{0,128}\b["']?\s*[:=]\s*)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\r\n,;}]+)'''
)
_REVIEW_KINDS = {"direction", "final"}
_QUESTIONS = {"direction": "challenge_direction", "final": "challenge_final_evidence"}
_EXECUTION_LAYERS = {
    "gpu", "framework", "transfer", "cpu_data", "communication", "io", "environment",
}
_CLAIM_LAYERS = {"kernel", "runtime", "workload", "environment", "system"}
_CONFIDENCE_BUCKETS = {"low", "medium", "high", "inconclusive"}
_EVIDENCE_KINDS = {
    "timeline", "framework", "cpu_data", "transfer",
    "communication", "io", "environment", "custom",
}
_RISK_LEVELS = {"none", "low", "medium", "high"}
_EVALUATION_STATUSES = {"evaluated", "not_evaluated", "failed", "skipped", "unknown"}
_PRIMARY_STATUSES = {"confirmed_win", "inconclusive", "regression", "failed", "unknown"}
_CONSTRAINT_STATUSES = {"passed", "failed", "skipped", "unknown"}
_DIRECTION_SELECTION_STATUSES = {"selected", "evidence_gap", "blocked", "skipped", "unknown"}
_ARTIFACT_HASH_NAMES = {
    "performance_model.json", "hypothesis_result.json", "evidence_selection.json",
    "request_set.json", "knowledge_adaptation.json", "diagnosis.json",
    "change_set.json", "candidate.diff",
}
_SAFE_ENV = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONPATH",
    "TMPDIR",
}


class ReviewerError(ValueError):
    """Raised when reviewer input or output violates the advisory protocol."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object(value: Any, field: str) -> dict:
    if type(value) is not dict:
        raise ReviewerError(f"{field} must be an object")
    return value


def _closed(value: Mapping[str, Any], fields: set[str], field: str) -> None:
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ReviewerError(f"{field} contains unknown fields: {', '.join(unknown)}")


def _required(value: Mapping[str, Any], fields: set[str], field: str) -> None:
    missing = sorted(fields - set(value))
    if missing:
        raise ReviewerError(f"{field} is missing required fields: {', '.join(missing)}")


def _string(value: Any, field: str, maximum: int = 4096) -> str:
    if type(value) is not str or not value.strip():
        raise ReviewerError(f"{field} must be a non-empty string")
    if len(value) > maximum:
        raise ReviewerError(f"{field} exceeds {maximum} characters")
    return value


def _canonical_model(model: str) -> str:
    return model.strip().lower()


def _source_number(value: Any, *, minimum: float = -1.0e18) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return 0.0
    return min(1.0e18, max(minimum, float(value)))


def _source_enums(value: Any, allowed: set[str], maximum: int = 16) -> list[str]:
    if type(value) is not list:
        return []
    return list(
        dict.fromkeys(item for item in value if type(item) is str and item in allowed)
    )[:maximum]


def _source_object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _source_enum(value: Any, allowed: set[str], fallback: Any) -> Any:
    return value if type(value) is str and value in allowed else fallback


def _validate_number(value: Any, field: str, minimum: float) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= 1.0e18
    ):
        raise ReviewerError(f"{field} must be a bounded finite number")
    return float(value)


def _validate_enums(
    value: Any, field: str, allowed: set[str], maximum: int = 16
) -> list[str]:
    if (
        type(value) is not list
        or len(value) > maximum
        or len(value) != len({item for item in value if type(item) is str})
        or any(type(item) is not str or item not in allowed for item in value)
    ):
        raise ReviewerError(f"{field} contains an invalid enum")
    return list(value)


def _validate_exact(value: Any, fields: set[str], field: str) -> dict:
    result = _object(value, field)
    _closed(result, fields, field)
    _required(result, fields, field)
    return result


def _validate_review_summary(value: Mapping[str, Any]) -> dict:
    summary = _validate_exact(
        value,
        {
            "schema_version", "review_kind", "question", "performance",
            "direction", "final", "artifact_hashes",
        },
        "review_summary",
    )
    kind = summary["review_kind"]
    if (
        summary["schema_version"] != SUMMARY_SCHEMA
        or type(kind) is not str
        or kind not in _REVIEW_KINDS
        or summary["question"] != _QUESTIONS.get(kind)
    ):
        raise ReviewerError("review_summary identity is invalid")
    performance_fields = {
        "window_duration_us", "minimum_effect_us",
        "layer_benefit_upper_bound_us", "observed_layers",
        "missing_layers", "critical_path_layers", "uncertainty_kinds",
    }
    performance = _validate_exact(
        summary["performance"], performance_fields, "review_summary.performance"
    )
    for name in (
        "window_duration_us", "minimum_effect_us", "layer_benefit_upper_bound_us"
    ):
        _validate_number(performance[name], f"review_summary.performance.{name}", 0.0)
    for name, allowed in (
        ("observed_layers", _EXECUTION_LAYERS),
        ("missing_layers", _EXECUTION_LAYERS),
        ("critical_path_layers", _EXECUTION_LAYERS),
        ("uncertainty_kinds", _EVIDENCE_KINDS),
    ):
        _validate_enums(performance[name], f"review_summary.performance.{name}", allowed)
    if kind == "direction":
        names = {
            "candidate_count", "claim_layers", "confidence_buckets",
            "support_evidence_count", "oppose_evidence_count",
            "missing_evidence_count", "selection_status",
            "selected_evidence_kind", "declared_cost_upper_bound_seconds",
        }
        item = _validate_exact(summary["direction"], names, "review_summary.direction")
        if summary["final"] is not None:
            raise ReviewerError("review_summary.final must be null for direction")
        for name in (
            "candidate_count", "support_evidence_count",
            "oppose_evidence_count", "missing_evidence_count",
        ):
            if type(item[name]) is not int or not 0 <= item[name] <= 131072:
                raise ReviewerError(f"review_summary.direction.{name} is invalid")
        _validate_enums(item["claim_layers"], "review_summary.direction.claim_layers", _CLAIM_LAYERS)
        _validate_enums(
            item["confidence_buckets"],
            "review_summary.direction.confidence_buckets",
            _CONFIDENCE_BUCKETS,
        )
        if (
            type(item["selection_status"]) is not str
            or item["selection_status"] not in _DIRECTION_SELECTION_STATUSES
            or (
                item["selected_evidence_kind"] is not None
                and (
                    type(item["selected_evidence_kind"]) is not str
                    or item["selected_evidence_kind"] not in _EVIDENCE_KINDS
                )
            )
        ):
            raise ReviewerError("review_summary.direction enum is invalid")
        _validate_number(
            item["declared_cost_upper_bound_seconds"],
            "review_summary.direction.declared_cost_upper_bound_seconds",
            0.0,
        )
    else:
        names = {
            "scope_kind", "risk_level", "evaluation_status", "primary_status",
            "constraint_count", "constraints_passed", "observed_effect_pct",
            "ci_low_pct", "ci_high_pct", "diff_present",
            "candidate_diff_sha256",
        }
        item = _validate_exact(summary["final"], names, "review_summary.final")
        if summary["direction"] is not None:
            raise ReviewerError("review_summary.direction must be null for final")
        if (
            type(item["scope_kind"]) is not str
            or item["scope_kind"] not in {"project", "environment", "isolated_environment"}
            or type(item["risk_level"]) is not str
            or item["risk_level"] not in _RISK_LEVELS
            or type(item["evaluation_status"]) is not str
            or item["evaluation_status"] not in _EVALUATION_STATUSES
            or type(item["primary_status"]) is not str
            or item["primary_status"] not in _PRIMARY_STATUSES
            or type(item["constraint_count"]) is not int
            or not 0 <= item["constraint_count"] <= 1024
            or type(item["constraints_passed"]) is not bool
            or type(item["diff_present"]) is not bool
        ):
            raise ReviewerError("review_summary.final enum, count, or flag is invalid")
        for name in ("observed_effect_pct", "ci_low_pct", "ci_high_pct"):
            _validate_number(item[name], f"review_summary.final.{name}", -1.0e18)
        digest = item["candidate_diff_sha256"]
        if digest is not None and (
            type(digest) is not str or _SHA256.fullmatch(digest) is None
        ):
            raise ReviewerError("review_summary.final.candidate_diff_sha256 is invalid")
    hashes = _object(summary["artifact_hashes"], "review_summary.artifact_hashes")
    if set(hashes) - _ARTIFACT_HASH_NAMES or any(
        type(digest) is not str or _SHA256.fullmatch(digest) is None
        for digest in hashes.values()
    ):
        raise ReviewerError("review_summary.artifact_hashes is invalid")
    return copy.deepcopy(summary)


def _safe_request(summary: Mapping[str, Any]) -> dict:
    base = {
        "schema_version": REQUEST_SCHEMA,
        "review_summary": _validate_review_summary(summary),
    }
    return {**base, "request_digest": _digest(base)}


def build_review_request(
    *,
    diagnosis: Mapping[str, Any],
    change_set: Mapping[str, Any],
    redacted_diff: str,
    experiment: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
) -> dict:
    """Reduce local facts to the one closed provider request envelope."""
    diagnosis = _object(diagnosis, "diagnosis")
    change_set = _object(change_set, "change_set")
    experiment = _object(experiment, "experiment")
    hashes = _object(artifact_hashes, "artifact_hashes")
    if type(redacted_diff) is not str or len(redacted_diff.encode("utf-8")) > 256 * 1024:
        raise ReviewerError("redacted_diff must be a string of at most 262144 bytes")
    if set(hashes) - _ARTIFACT_HASH_NAMES or any(
        type(digest) is not str or _SHA256.fullmatch(digest) is None
        for digest in hashes.values()
    ):
        raise ReviewerError("artifact_hashes is invalid")
    kind = _source_enum(diagnosis.get("review_kind"), _REVIEW_KINDS, "final")
    performance_source = _source_object(diagnosis.get("performance_summary"))
    hypotheses = change_set.get("hypotheses")
    hypotheses = hypotheses if type(hypotheses) is list else []
    candidate = _source_object(change_set.get("candidate"))
    selected = _source_object(experiment.get("selected_action"))
    cost = _source_object(selected.get("cost"))
    statuses = _source_enums(
        experiment.get("constraint_statuses"), _CONSTRAINT_STATUSES, 32
    )
    direction = None
    final = None
    if kind == "direction":
        def evidence_count(name: str) -> int:
            return sum(
                min(len(item.get(name, [])), 1024)
                for item in hypotheses
                if isinstance(item, Mapping) and type(item.get(name, [])) is list
            )

        direction = {
            "candidate_count": min(len(hypotheses), 128),
            "claim_layers": _source_enums(
                [item.get("claim_layer") for item in hypotheses if isinstance(item, Mapping)],
                _CLAIM_LAYERS,
            ),
            "confidence_buckets": _source_enums(
                [item.get("confidence") for item in hypotheses if isinstance(item, Mapping)],
                _CONFIDENCE_BUCKETS,
            ),
            "support_evidence_count": evidence_count("support_evidence_ids"),
            "oppose_evidence_count": evidence_count("oppose_evidence_ids"),
            "missing_evidence_count": evidence_count("missing_evidence_kinds"),
            "selection_status": _source_enum(
                experiment.get("selection_status"),
                _DIRECTION_SELECTION_STATUSES,
                "unknown",
            ),
            "selected_evidence_kind": _source_enum(
                selected.get("evidence_kind"), _EVIDENCE_KINDS, None
            ),
            "declared_cost_upper_bound_seconds": _source_number(
                cost.get("p90_seconds", selected.get("declared_cost_upper_bound_seconds")),
                minimum=0.0,
            ),
        }
    else:
        final = {
            "scope_kind": _source_enum(
                change_set.get("scope"),
                {"project", "environment", "isolated_environment"},
                "project",
            ),
            "risk_level": _source_enum(change_set.get("risk"), _RISK_LEVELS, "none"),
            "evaluation_status": _source_enum(
                experiment.get("evaluation_status"), _EVALUATION_STATUSES, "unknown"
            ),
            "primary_status": _source_enum(
                experiment.get("primary_status"), _PRIMARY_STATUSES, "unknown"
            ),
            "constraint_count": len(statuses),
            "constraints_passed": bool(statuses) and set(statuses) == {"passed"},
            "observed_effect_pct": _source_number(candidate.get("effect_pct")),
            "ci_low_pct": _source_number(candidate.get("ci_low_pct")),
            "ci_high_pct": _source_number(candidate.get("ci_high_pct")),
            "diff_present": bool(redacted_diff),
            "candidate_diff_sha256": hashes.get("candidate.diff"),
        }
    performance = {
        name: _source_number(performance_source.get(name), minimum=0.0)
        for name in (
            "window_duration_us", "minimum_effect_us",
            "layer_benefit_upper_bound_us",
        )
    }
    for name, allowed in (
        ("observed_layers", _EXECUTION_LAYERS),
        ("missing_layers", _EXECUTION_LAYERS),
        ("critical_path_layers", _EXECUTION_LAYERS),
        ("uncertainty_kinds", _EVIDENCE_KINDS),
    ):
        source_name = "critical_path" if name == "critical_path_layers" else name
        performance[name] = _source_enums(performance_source.get(source_name), allowed)
    return _safe_request(
        {
            "schema_version": SUMMARY_SCHEMA,
            "review_kind": kind,
            "question": _QUESTIONS[kind],
            "performance": performance,
            "direction": direction,
            "final": final,
            "artifact_hashes": copy.deepcopy(hashes),
        }
    )


def request_digest(request: Mapping[str, Any]) -> str:
    """Recompute the digest without trusting the request's digest field."""
    value = copy.deepcopy(_object(request, "request"))
    value.pop("request_digest", None)
    return _digest(value)


def validate_review_request(value: Mapping[str, Any]) -> dict:
    """Reject malformed advisory input before a provider can start."""
    request = _validate_exact(
        value,
        {"schema_version", "review_summary", "request_digest"},
        "review request",
    )
    if request["schema_version"] != REQUEST_SCHEMA:
        raise ReviewerError(f"review request schema_version must be {REQUEST_SCHEMA}")
    safe = _safe_request(request["review_summary"])
    if request["request_digest"] != safe["request_digest"]:
        raise ReviewerError("review request digest is invalid")
    return safe


def validate_review_response(
    value: Mapping[str, Any], request: Mapping[str, Any]
) -> dict:
    """Validate a response that can advise but cannot execute or promote."""
    response = _object(value, "review_response")
    fields = {
        "schema_version",
        "request_digest",
        "verdict",
        "concerns",
        "suggested_experiments",
    }
    _closed(response, fields, "review_response")
    _required(response, fields, "review_response")
    if response["schema_version"] != RESPONSE_SCHEMA:
        raise ReviewerError(f"review_response.schema_version must be {RESPONSE_SCHEMA}")
    request = validate_review_request(request)
    expected_digest = request["request_digest"]
    if response["request_digest"] != expected_digest:
        raise ReviewerError("review response digest does not match request digest")
    if response["verdict"] not in {"support", "challenge", "insufficient"}:
        raise ReviewerError("review verdict must be support, challenge, or insufficient")

    concerns = response["concerns"]
    if type(concerns) is not list or len(concerns) > 32:
        raise ReviewerError("review concerns must be an array with at most 32 entries")
    for index, item in enumerate(concerns):
        concern = _object(item, f"review_response.concerns[{index}]")
        concern_fields = {"severity", "category", "message"}
        _closed(concern, concern_fields, f"review_response.concerns[{index}]")
        _required(concern, concern_fields, f"review_response.concerns[{index}]")
        if concern["severity"] not in {"low", "medium", "high"}:
            raise ReviewerError(f"review_response.concerns[{index}].severity is invalid")
        _string(concern["category"], f"review_response.concerns[{index}].category", 128)
        _string(concern["message"], f"review_response.concerns[{index}].message")

    suggestions = response["suggested_experiments"]
    if type(suggestions) is not list or len(suggestions) > 32:
        raise ReviewerError(
            "review suggested_experiments must be an array with at most 32 entries"
        )
    for index, item in enumerate(suggestions):
        _string(item, f"review_response.suggested_experiments[{index}]", 2048)
    return copy.deepcopy(response)


def _duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReviewerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_response(payload: bytes) -> dict:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_pairs,
            parse_constant=lambda token: (_raise_number(token)),
        )
    except ReviewerError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReviewerError(f"reviewer stdout must be strict JSON: {error}") from error
    return _object(value, "review_response")


def _raise_number(token: str):
    raise ReviewerError(f"reviewer JSON number must be finite: {token}")


class _BoundedCapture:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        available = max(0, self.limit - len(self.data))
        self.data.extend(chunk[:available])
        if len(chunk) > available:
            self.truncated = True


def _drain(stream, capture: _BoundedCapture) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            capture.append(chunk)
    finally:
        stream.close()


def _write_stdin(stream, payload: bytes, errors: list[str]) -> None:
    try:
        stream.write(payload)
        stream.flush()
    except (BrokenPipeError, OSError) as error:
        errors.append(f"reviewer stdin failed: {error}")
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(process) -> None:
    process_group = process.pid

    try:
        os.killpg(process_group, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + 0.25
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.01)
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _environment() -> tuple[dict, tuple[str, ...]]:
    inherited = dict(os.environ)
    values = tuple(
        value for name, value in inherited.items() if _SECRET_NAME.search(name) and value
    )
    environment = {
        name: value
        for name, value in inherited.items()
        if name in _SAFE_ENV and not _SECRET_NAME.search(name)
    }
    return environment, values


def _redact(payload: bytes, secrets: Sequence[str]) -> str:
    value = payload.decode("utf-8", errors="replace")
    value = _SECRET_LOG.sub(lambda match: f"{match.group(1)}[REDACTED]", value)
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _artifact(
    request: Mapping[str, Any],
    *,
    status: str,
    response: dict | None,
    execution: Mapping[str, Any],
) -> dict:
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "status": status,
        "request_digest": request_digest(request),
        "response": copy.deepcopy(response),
        "execution": copy.deepcopy(execution),
    }


def run_reviewer(
    config: Mapping[str, Any],
    request: Mapping[str, Any],
    run_dir: str | os.PathLike[str],
    *,
    output_limit_bytes: int = 256 * 1024,
) -> dict:
    """Run a local CLI in advisory mode and always persist a review artifact."""
    configuration = _object(config, "reviewer config")
    _closed(configuration, {"argv", "timeout_seconds"}, "reviewer config")
    _required(configuration, {"argv", "timeout_seconds"}, "reviewer config")
    argv = configuration["argv"]
    if type(argv) is not list or not argv or any(
        type(item) is not str or not item for item in argv
    ):
        raise ReviewerError("reviewer argv must be a non-empty string array")
    timeout = configuration["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ReviewerError("reviewer timeout_seconds must be numeric")
    if not math.isfinite(float(timeout)) or not 0.05 <= timeout <= 3600:
        raise ReviewerError("reviewer timeout_seconds must be between 0.05 and 3600")
    if isinstance(output_limit_bytes, bool) or not isinstance(output_limit_bytes, int):
        raise ReviewerError("output_limit_bytes must be an integer")
    if not 128 <= output_limit_bytes <= 1024 * 1024:
        raise ReviewerError("output_limit_bytes must be between 128 and 1048576")
    request = validate_review_request(request)
    expected = request["request_digest"]

    run_root = Path(run_dir).expanduser().resolve(strict=False)
    run_root.mkdir(parents=True, exist_ok=True)
    stdin_payload = _canonical_bytes(request) + b"\n"
    stdout = _BoundedCapture(output_limit_bytes)
    stderr = _BoundedCapture(output_limit_bytes)
    environment, secrets = _environment()
    started = time.monotonic()
    exit_code = None
    timed_out = False
    failure = None
    response = None
    deadline = started + float(timeout)

    with tempfile.TemporaryDirectory(prefix="reviewer-cwd-", dir=run_root) as cwd:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            readers = [
                threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
                threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
            ]
            for reader in readers:
                reader.start()
            writer_errors: list[str] = []
            writer = threading.Thread(
                target=_write_stdin,
                args=(process.stdin, stdin_payload, writer_errors),
                daemon=True,
            )
            writer.start()
            try:
                exit_code = process.wait(
                    timeout=max(0.001, deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                failure = f"reviewer exceeded {timeout} seconds"
                _stop_group(process)
                exit_code = process.returncode
            else:
                if _process_group_exists(process.pid):
                    _stop_group(process)
            writer.join(timeout=1)
            for reader in readers:
                reader.join(timeout=1)
            if failure is None and writer_errors:
                failure = writer_errors[0]
        except (FileNotFoundError, OSError) as error:
            failure = f"reviewer unavailable: {error}"

    if failure is None and exit_code != 0:
        failure = f"reviewer exited with status {exit_code}"
    if failure is None and stdout.truncated:
        failure = f"reviewer stdout exceeds {output_limit_bytes} bytes"
    if failure is None:
        try:
            response = validate_review_response(_parse_response(bytes(stdout.data)), request)
        except ReviewerError as error:
            failure = str(error)

    execution = {
        "argv_sha256": _digest(argv),
        "stdin_sha256": hashlib.sha256(stdin_payload).hexdigest(),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
        "stderr": _redact(bytes(stderr.data), secrets),
        "stderr_truncated": stderr.truncated,
        "failure": failure,
    }
    artifact = _artifact(
        request,
        status="completed" if failure is None else "unavailable",
        response=response,
        execution=execution,
    )
    _atomic_json(run_root / "review.json", artifact)
    return artifact


def run_reviewers(
    configs: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    run_dir: str | os.PathLike[str],
    *,
    total_timeout_seconds: float = 180.0,
) -> dict:
    """Run named advisory reviewers concurrently under one total wait bound."""
    if not isinstance(configs, Sequence) or isinstance(
        configs, (str, bytes, bytearray)
    ) or not configs:
        raise ReviewerError("reviewers must be a non-empty sequence")
    if len(configs) > 8:
        raise ReviewerError("reviewers must contain at most 8 providers")
    if (
        isinstance(total_timeout_seconds, bool)
        or not isinstance(total_timeout_seconds, (int, float))
        or not math.isfinite(float(total_timeout_seconds))
        or not 1 <= float(total_timeout_seconds) <= 180
    ):
        raise ReviewerError("total_timeout_seconds must be between 1 and 180")
    request = validate_review_request(request)
    expected = request["request_digest"]

    normalized = []
    run_keys = set()
    cleanup_reserve = min(4.0, float(total_timeout_seconds) * 0.25)
    provider_deadline = max(
        0.05, float(total_timeout_seconds) - cleanup_reserve
    )
    for index, raw in enumerate(configs):
        config = _object(raw, f"reviewers[{index}]")
        _closed(
            config,
            {"provider", "underlying_model", "argv", "timeout_seconds"},
            f"reviewers[{index}]",
        )
        _required(config, {"provider", "argv", "timeout_seconds"}, f"reviewers[{index}]")
        provider = _string(config["provider"], f"reviewers[{index}].provider", 64)
        if _PROVIDER.fullmatch(provider) is None:
            raise ReviewerError(f"reviewers[{index}].provider is invalid")
        underlying_model = config.get("underlying_model", "unknown")
        underlying_model = _string(
            underlying_model, f"reviewers[{index}].underlying_model", 64
        )
        if _PROVIDER.fullmatch(underlying_model) is None:
            raise ReviewerError(f"reviewers[{index}].underlying_model is invalid")
        canonical_provider = _PROVIDER_ALIASES.get(provider.lower(), provider.lower())
        run_key = (expected, canonical_provider, _canonical_model(underlying_model))
        if run_key in run_keys:
            continue
        run_keys.add(run_key)
        timeout = config["timeout_seconds"]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 1 <= float(timeout) <= 3600
        ):
            raise ReviewerError(
                f"reviewers[{index}].timeout_seconds must be between 1 and 3600"
            )
        argv = config["argv"]
        if type(argv) is not list or not argv or any(
            type(item) is not str or not item for item in argv
        ):
            raise ReviewerError(f"reviewers[{index}].argv must be a non-empty string array")
        normalized.append(
            {
                "provider": provider,
                "canonical_provider": canonical_provider,
                "underlying_model": underlying_model,
                "argv": list(argv),
                "timeout_seconds": min(
                    float(timeout), provider_deadline
                ),
            }
        )

    run_root = Path(run_dir).expanduser().resolve(strict=False)
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def execute(config: Mapping[str, Any]) -> dict:
        provider = config["provider"]
        try:
            artifact = run_reviewer(
                {
                    "argv": config["argv"],
                    "timeout_seconds": config["timeout_seconds"],
                },
                request,
                run_root / "reviewers" / _digest(
                    [
                        expected,
                        config["canonical_provider"],
                        _canonical_model(config["underlying_model"]),
                    ]
                )[:16],
            )
            execution = artifact.get("execution", {})
            return {
                "provider": provider,
                "underlying_model": config["underlying_model"],
                "status": artifact["status"],
                "response": copy.deepcopy(artifact.get("response")),
                "failure": execution.get("failure"),
                "duration_seconds": float(execution.get("duration_seconds", 0.0)),
            }
        except (OSError, ReviewerError, RuntimeError) as error:
            return {
                "provider": provider,
                "underlying_model": config["underlying_model"],
                "status": "unavailable",
                "response": None,
                "failure": str(error),
                "duration_seconds": max(0.0, time.monotonic() - started),
            }

    with ThreadPoolExecutor(max_workers=len(normalized)) as executor:
        futures = [executor.submit(execute, config) for config in normalized]
        reviews = [future.result() for future in futures]

    elapsed = max(0.0, time.monotonic() - started)
    requested = [item["provider"] for item in normalized]
    completed = [
        item["provider"] for item in reviews if item["status"] == "completed"
    ]
    failed = [
        item["provider"] for item in reviews if item["status"] != "completed"
    ]
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "completed" if completed else "unavailable",
        "request_digest": expected,
        "providers_requested": requested,
        "providers_completed": completed,
        "failed_providers": failed,
        "heterogeneous_models": _heterogeneous_models(reviews),
        "total_timeout_seconds": float(total_timeout_seconds),
        "total_wait_seconds": float(elapsed),
        "reviews": reviews,
    }
    _atomic_json(run_root / "review.json", aggregate)
    return aggregate


def _ordered_reviewer_configs(
    configs: Sequence[Mapping[str, Any]],
) -> list[dict]:
    if not isinstance(configs, Sequence) or isinstance(
        configs, (str, bytes, bytearray)
    ) or not configs:
        raise ReviewerError("reviewers must be a non-empty sequence")
    if len(configs) > 8:
        raise ReviewerError("reviewers must contain at most 8 providers")
    normalized = []
    priority = {provider: index for index, provider in enumerate(_PROVIDER_PRIORITY)}
    for index, raw in enumerate(configs):
        config = _object(raw, f"reviewers[{index}]")
        _closed(
            config,
            {"provider", "underlying_model", "argv", "timeout_seconds"},
            f"reviewers[{index}]",
        )
        _required(config, {"provider", "argv", "timeout_seconds"}, f"reviewers[{index}]")
        provider = _string(config["provider"], f"reviewers[{index}].provider", 64)
        if _PROVIDER.fullmatch(provider) is None:
            raise ReviewerError(f"reviewers[{index}].provider is invalid")
        canonical = _PROVIDER_ALIASES.get(
            provider.lower(),
            provider.lower(),
        )
        underlying_model = config.get("underlying_model", "unknown")
        underlying_model = _string(
            underlying_model, f"reviewers[{index}].underlying_model", 64
        )
        if _PROVIDER.fullmatch(underlying_model) is None:
            raise ReviewerError(f"reviewers[{index}].underlying_model is invalid")
        argv = config["argv"]
        if type(argv) is not list or not argv or any(
            type(item) is not str or not item for item in argv
        ):
            raise ReviewerError(f"reviewers[{index}].argv must be a non-empty string array")
        timeout = config["timeout_seconds"]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 1 <= float(timeout) <= 3600
        ):
            raise ReviewerError(
                f"reviewers[{index}].timeout_seconds must be between 1 and 3600"
            )
        normalized.append(
            {
                "provider": provider,
                "canonical_provider": canonical,
                "underlying_model": underlying_model,
                "argv": list(argv),
                "timeout_seconds": float(timeout),
            }
        )
    normalized.sort(
        key=lambda item: priority.get(
            item["canonical_provider"],
            len(priority),
        )
    )
    distinct = []
    run_keys = set()
    for item in normalized:
        run_key = (
            item["canonical_provider"],
            _canonical_model(item["underlying_model"]),
        )
        if run_key in run_keys:
            continue
        run_keys.add(run_key)
        distinct.append(item)
    return [
        {
            "provider": item["provider"],
            "underlying_model": item["underlying_model"],
            "argv": item["argv"],
            "timeout_seconds": item["timeout_seconds"],
        }
        for item in distinct
    ]


def select_reviewer_configs(
    configs: Sequence[Mapping[str, Any]], trigger: str
) -> list[dict]:
    """Select the highest-priority configured providers for one trigger."""
    if trigger not in _TRIGGER_TARGETS:
        raise ReviewerError("review trigger is unsupported")
    return _ordered_reviewer_configs(configs)[: _TRIGGER_TARGETS[trigger]]


def _heterogeneous_models(reviews: Sequence[Mapping[str, Any]]) -> list[str]:
    models = []
    model_keys = set()
    for item in reviews:
        model = item["underlying_model"]
        model_key = _canonical_model(model)
        if (
            item["status"] == "completed"
            and model_key not in {"auto", "unknown"}
            and model_key not in model_keys
        ):
            models.append(model)
            model_keys.add(model_key)
    return models


def run_prioritized_reviewers(
    configs: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    run_dir: str | os.PathLike[str],
    *,
    trigger: str,
    total_timeout_seconds: float = 180.0,
) -> dict:
    """Fill a trigger-sized advisory panel with bounded priority fallback."""
    if trigger not in _TRIGGER_TARGETS:
        raise ReviewerError("review trigger is unsupported")
    if (
        isinstance(total_timeout_seconds, bool)
        or not isinstance(total_timeout_seconds, (int, float))
        or not math.isfinite(float(total_timeout_seconds))
        or not 1 <= float(total_timeout_seconds) <= 180
    ):
        raise ReviewerError("total_timeout_seconds must be between 1 and 180")
    request = validate_review_request(request)
    expected = request["request_digest"]
    ordered = _ordered_reviewer_configs(configs)
    target = _TRIGGER_TARGETS[trigger]
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cursor = 0
    reviews = []
    while cursor < len(ordered):
        completed_count = sum(item["status"] == "completed" for item in reviews)
        if completed_count >= target:
            break
        remaining = float(total_timeout_seconds) - (time.monotonic() - started)
        if remaining < 1:
            break
        width = target - completed_count
        batch = ordered[cursor : cursor + width]
        cursor += len(batch)
        if not batch:
            break
        artifact = run_reviewers(
            batch,
            request,
            run_root / "batches" / f"{len(reviews):04d}",
            total_timeout_seconds=min(180.0, remaining),
        )
        reviews.extend(copy.deepcopy(artifact["reviews"]))
    elapsed = max(0.0, time.monotonic() - started)
    requested = [item["provider"] for item in reviews]
    completed = [
        item["provider"] for item in reviews if item["status"] == "completed"
    ]
    failed = [
        item["provider"] for item in reviews if item["status"] != "completed"
    ]
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "completed" if completed else "unavailable",
        "request_digest": expected,
        "trigger": trigger,
        "target_completed_provider_count": target,
        "providers_requested": requested,
        "providers_completed": completed,
        "failed_providers": failed,
        "heterogeneous_models": _heterogeneous_models(reviews),
        "total_timeout_seconds": float(total_timeout_seconds),
        "total_wait_seconds": elapsed,
        "reviews": reviews,
    }
    _atomic_json(run_root / "review.json", aggregate)
    return aggregate


def write_skipped_review(
    request: Mapping[str, Any],
    run_dir: str | os.PathLike[str],
    *,
    reason: str = "reviewer not configured",
) -> dict:
    """Record that no reviewer was configured without changing the decision path."""
    request = validate_review_request(request)
    if reason not in {"reviewer not configured", "controller_managed"}:
        raise ReviewerError("skipped review reason is invalid")
    artifact = _artifact(
        request,
        status="skipped",
        response=None,
        execution={"failure": None, "reason": reason},
    )
    _atomic_json(Path(run_dir).expanduser().resolve(strict=False) / "review.json", artifact)
    return artifact
