#!/usr/bin/env python3
"""CPU/static installation audit for the frozen V1.4 skill surface."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import secrets
import stat
import sys
from pathlib import Path, PurePosixPath


PRODUCTION_MODULES = (
    "_invocation_runtime.py",
    "artifact_store.py",
    "champion.py",
    "compiler_evidence.py",
    "execution_map.py",
    "experiment_design.py",
    "knowledge_query.py",
    "paired_stats.py",
    "profile_ncu.py",
    "profile_nsys.py",
    "profile_pytorch.py",
    "readiness.py",
    "sass_check.py",
    "self_check.py",
    "version_audit.py",
    "workload_adapter.py",
    "workload_evaluate.py",
)
DRIVER_TEMPLATES = (
    "workload_driver.py",
    "workload_driver_request.schema.json",
    "workload_driver_result.schema.json",
)
ROOT_FILES = ("LICENSE", "NOTICE", "SKILL.md")
ROOT_DIRECTORIES = ("agents", "examples", "references", "scripts", "templates")
AGENT_FILES = ("openai.yaml",)
EXAMPLE_FILES = ("walkthrough.md",)
REFERENCE_FILES = (
    "compatibility.md",
    "environment_readiness.md",
    "ncu_metrics_guide.md",
    "nonstationary_serving_evidence.md",
    "optimizer_limits.md",
    "performance_iteration.md",
    "research_augmentation.md",
    "sass_signatures.json",
    "serving_evidence_protocol.md",
    "systems_and_ir_coverage.md",
    "version_stack_audit.md",
)
ALLOWED_PRODUCTION_DEPENDENCIES = {
    "_invocation_runtime": frozenset({"artifact_store"}),
    "artifact_store": frozenset(),
    "champion": frozenset({"artifact_store"}),
    "compiler_evidence": frozenset(
        {"_invocation_runtime", "artifact_store", "workload_adapter"}
    ),
    "execution_map": frozenset(),
    "experiment_design": frozenset(),
    "knowledge_query": frozenset(),
    "paired_stats": frozenset(),
    "profile_ncu": frozenset(
        {
            "_invocation_runtime",
            "artifact_store",
            "execution_map",
            "version_audit",
            "workload_adapter",
        }
    ),
    "profile_nsys": frozenset(
        {
            "_invocation_runtime",
            "artifact_store",
            "execution_map",
            "version_audit",
            "workload_adapter",
        }
    ),
    "profile_pytorch": frozenset(
        {
            "_invocation_runtime",
            "artifact_store",
            "execution_map",
            "version_audit",
            "workload_adapter",
        }
    ),
    "readiness": frozenset(
        {"_invocation_runtime", "artifact_store", "version_audit", "workload_adapter"}
    ),
    "sass_check": frozenset(
        {"_invocation_runtime", "artifact_store", "workload_adapter"}
    ),
    "self_check": frozenset({"version_audit"}),
    "version_audit": frozenset(),
    "workload_adapter": frozenset({"artifact_store"}),
    "workload_evaluate": frozenset(
        {
            "_invocation_runtime",
            "artifact_store",
            "experiment_design",
            "paired_stats",
            "workload_adapter",
        }
    ),
}
LEGACY_ENTRIES = frozenset(
    {
        "orchestrate.py", "workload_controller.py", "run_control.py", "run_iteration.py",
        "state.py", "direction_guard.py", "decision.py", "diagnostic_decision.py",
        "evidence_controller.py", "evidence_ledger.py", "planner_boundary.py",
        "planner_admission.py", "iteration_guard.py", "strategy_memory.py",
    }
)


def _regular_names(directory: Path, suffix: str | None = None) -> set[str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"missing or unsafe directory: {directory}")
    names = set()
    for path in directory.iterdir():
        if path.name == "__pycache__" and path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe package entry: {path}")
        if suffix is None or path.name.endswith(suffix):
            names.add(path.name)
    return names


def _exact_files(directory: Path, expected: tuple[str, ...], label: str) -> None:
    actual = _regular_names(directory)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        raise ValueError(f"{label} differs; missing={missing}, extra={extra}")


def _exact_mixed_entries(
    directory: Path,
    *,
    files: tuple[str, ...],
    directories: tuple[str, ...],
    label: str,
) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"missing or unsafe directory: {directory}")
    expected = set(files) | set(directories)
    actual = {path.name for path in directory.iterdir()}
    if actual != expected:
        raise ValueError(
            f"{label} differs; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for name in files:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe {label} file: {path}")
    for name in directories:
        path = directory / name
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"unsafe {label} directory: {path}")


def _imports(source: str, filename: Path) -> set[str]:
    try:
        tree = ast.parse(source, filename=str(filename))
    except SyntaxError as error:
        raise ValueError(f"invalid production module: {filename.name}") from error
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            result.add(node.module.split(".", 1)[0])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_load_sibling"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.endswith(".py")
        ):
            result.add(Path(node.args[0].value).stem)
    return result


def _validate_dependency_graph(scripts: Path) -> None:
    known = {Path(name).stem for name in PRODUCTION_MODULES}
    if set(ALLOWED_PRODUCTION_DEPENDENCIES) != known:
        raise ValueError("production dependency policy does not cover the exact surface")
    graph = {}
    for name in PRODUCTION_MODULES:
        module = Path(name).stem
        source = (scripts / name).read_text(encoding="utf-8")
        imports = _imports(source, scripts / name)
        if module != "_invocation_runtime" and "subprocess" in imports:
            raise ValueError(
                f"subprocess ownership must remain in _invocation_runtime: {name}"
            )
        graph[module] = imports.intersection(known)
    for source, dependencies in graph.items():
        unexpected = dependencies - ALLOWED_PRODUCTION_DEPENDENCIES[source]
        if unexpected:
            raise ValueError(
                f"unexpected production dependency: {source} -> {sorted(unexpected)[0]}"
            )
    visiting, visited = set(), set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"production dependency cycle at {node}")
        if node in visited:
            return
        visiting.add(node)
        for child in graph[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for module in graph:
        visit(module)


def _collect_source_ids(value, result: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "source_id" and isinstance(item, str):
                result.add(item)
            elif key in {"source_ids", "sources"} and isinstance(item, list):
                result.update(entry for entry in item if isinstance(entry, str))
            else:
                _collect_source_ids(item, result)
    elif isinstance(value, list):
        for item in value:
            _collect_source_ids(item, result)


def _validate_knowledge(root: Path) -> None:
    knowledge = root / "references" / "knowledge"
    expected = {"cards.json", "sources.json", "playbooks"}
    actual = {path.name for path in knowledge.iterdir()} if knowledge.is_dir() else set()
    if actual != expected or (knowledge / "playbooks").is_symlink():
        raise ValueError("knowledge directory must contain only cards, sources, and playbooks")
    manifests = (knowledge / "cards.json", knowledge / "sources.json")
    if any(path.is_symlink() or not path.is_file() for path in manifests):
        raise ValueError("unsafe knowledge manifest")
    cards = json.loads(manifests[0].read_text(encoding="utf-8"))
    sources = json.loads(manifests[1].read_text(encoding="utf-8"))
    source_items = sources.get("sources", sources) if isinstance(sources, dict) else sources
    if not isinstance(source_items, list):
        raise ValueError("knowledge sources must be a list")
    source_ids = {item.get("id") for item in source_items if isinstance(item, dict)}
    if not source_ids or None in source_ids or len(source_ids) != len(source_items):
        raise ValueError("knowledge sources must have unique ids")
    for item in source_items:
        if (
            item.get("status") != "verified"
            or not isinstance(item.get("summary_sha256"), str)
            or len(item["summary_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["summary_sha256"])
        ):
            raise ValueError("knowledge source status or digest is invalid")
    referenced = set()
    _collect_source_ids(cards, referenced)
    if not referenced.issubset(source_ids):
        raise ValueError("knowledge cards reference an unknown source")
    playbooks = knowledge / "playbooks"
    if playbooks.is_symlink() or not playbooks.is_dir() or not any(playbooks.iterdir()):
        raise ValueError("knowledge playbooks are missing")
    for path in playbooks.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix != ".md":
            raise ValueError("knowledge playbooks contain an unsafe entry")
    referenced_playbooks = {}

    def collect_playbooks(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "playbook" and isinstance(item, str):
                    reference = PurePosixPath(item)
                    digest = value.get("playbook_sha256")
                    if (
                        reference.is_absolute()
                        or len(reference.parts) != 2
                        or reference.parts[0] != "playbooks"
                        or reference.suffix != ".md"
                        or any(part in {"", ".", ".."} for part in reference.parts)
                        or not isinstance(digest, str)
                        or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)
                    ):
                        raise ValueError("knowledge playbook reference is invalid")
                    previous = referenced_playbooks.setdefault(item, digest)
                    if previous != digest:
                        raise ValueError("knowledge playbook digest is inconsistent")
                elif key == "playbook_sha256" and "playbook" not in value:
                    raise ValueError("knowledge playbook digest has no reference")
                else:
                    collect_playbooks(item)
        elif isinstance(value, list):
            for item in value:
                collect_playbooks(item)

    collect_playbooks(cards)
    available_playbooks = {
        f"playbooks/{path.name}" for path in playbooks.iterdir() if path.is_file()
    }
    if set(referenced_playbooks) != available_playbooks:
        raise ValueError("knowledge cards and playbooks are not a closed set")
    for relative, expected_digest in referenced_playbooks.items():
        path = knowledge / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
            raise ValueError("knowledge playbook digest does not match")


def _probe_runtime_lock(scripts: Path) -> None:
    runtime_path = scripts / "_invocation_runtime.py"
    spec = importlib.util.spec_from_file_location("v14_self_check_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load invocation runtime")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    root = runtime._lock_root()
    if not isinstance(root, Path) or root.is_symlink() or not root.is_dir():
        raise ValueError("runtime lock root is unsafe")
    probe = root / f".self-check-{secrets.token_hex(8)}"
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, b"ok")
        os.fsync(fd)
    finally:
        os.close(fd)
    if probe.read_bytes() != b"ok":
        raise ValueError("runtime lock root is not readable and writable")
    probe.unlink()


def check_installation(skill_dir: Path | str) -> dict:
    root = Path(skill_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"missing or unsafe skill directory: {root}")
    _exact_mixed_entries(
        root,
        files=ROOT_FILES,
        directories=ROOT_DIRECTORIES,
        label="skill root",
    )
    _exact_files(root / "agents", AGENT_FILES, "agent metadata")
    _exact_files(root / "examples", EXAMPLE_FILES, "examples")
    _exact_mixed_entries(
        root / "references",
        files=REFERENCE_FILES,
        directories=("knowledge",),
        label="references",
    )
    scripts = root / "scripts"
    actual_scripts = _regular_names(scripts, ".py")
    expected_scripts = set(PRODUCTION_MODULES)
    if actual_scripts != expected_scripts:
        raise ValueError(
            f"unexpected production scripts; missing={sorted(expected_scripts - actual_scripts)}, "
            f"extra={sorted(actual_scripts - expected_scripts)}"
        )
    if actual_scripts.intersection(LEGACY_ENTRIES):
        raise ValueError("legacy Controller, state, or entrypoint remains installed")
    production_lines = sum(
        (scripts / name).read_text(encoding="utf-8").count("\n")
        for name in PRODUCTION_MODULES
    )
    if production_lines >= 14_671:
        raise ValueError("production Python exceeds the frozen V1.4 size boundary")
    _validate_dependency_graph(scripts)
    _exact_files(root / "templates", DRIVER_TEMPLATES, "driver templates")
    _validate_knowledge(root)
    _probe_runtime_lock(scripts)
    return {
        "schema_version": "cuda-kernel-optimizer/self-check-v1",
        "status": "passed",
        "checks": [
            "installation_surface", "production_surface", "dependency_graph",
            "driver_templates", "knowledge_closure", "runtime_lock_root",
            "legacy_removal",
        ],
        "gpu_checks_run": False,
        "network_checks_run": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check the frozen V1.4 installation surface.")
    parser.add_argument("--skill-dir", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(check_installation(args.skill_dir), sort_keys=True))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
