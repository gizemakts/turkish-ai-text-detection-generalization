# -*- coding: utf-8 -*-
"""Static semantic-contract audit for the V2 CE/SupCon protocol-faithful rerun.

This module performs no training or inference. It verifies that the dedicated
runner delegates every scientific operation to the bound migrated primary
implementation and only owns V2 guarding, checkpoint persistence, and QA
reporting. It may write a review report only when explicitly invoked with
``--write-report``.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".gitignore").is_file():
            return candidate
    raise RuntimeError("project root not found")


ROOT = _root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from turkish_ai_detection.data.study import sha256_file, write_json  # noqa: E402

CONTRACT = ROOT / "configs" / "external_cross_corpus.json"
PRIMARY = ROOT / "src" / "turkish_ai_detection" / "experiments" / "primary" / "run_fit.py"
SUPCON = ROOT / "src" / "turkish_ai_detection" / "experiments" / "model_development" / "supcon" / "train.py"
RUNNER = ROOT / "src" / "turkish_ai_detection" / "experiments" / "external_cross_corpus" / "rerun_fit.py"

REQUIRED_PRIMARY_DELEGATES = {
    "build_model",
    "make_seeded_trainer_class",
    "training_arguments",
    "make_dataset",
    "score",
    "metric_record",
    "pair_hash",
}
FORBIDDEN_LOCAL_REDEFINITIONS = REQUIRED_PRIMARY_DELEGATES | {
    "supervised_contrastive_loss",
    "encode_head_tail",
}
PERMITTED_LOCAL_RESPONSIBILITIES = {
    "verify_v2_contract",
    "verify_structural_identity",
    "save_terminal_payload",
    "reload_and_verify_payload",
    "numerical_qa",
    "run_one",
    "main",
}


def _called_primary_attributes(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "primary":
                found.add(node.attr)
    return found


def assert_semantic_equivalence(contract_path: Path = CONTRACT) -> dict:
    cfg = json.loads(contract_path.read_text(encoding="utf-8"))
    bindings = cfg["internal_training_bindings"]
    checks: dict[str, object] = {}

    checks["primary_source_hash"] = (
        sha256_file(PRIMARY) == bindings["current_primary_runner"]["sha256"]
    )
    checks["supcon_source_hash"] = (
        sha256_file(SUPCON) == bindings["current_supcon_implementation"]["sha256"]
    )
    checks["dataset_hash"] = (
        sha256_file(ROOT / bindings["dataset"]["path"]) == bindings["dataset"]["sha256"]
    )
    checks["fold_hash"] = (
        sha256_file(ROOT / bindings["folds"]["path"]) == bindings["folds"]["sha256"]
    )
    checks["historical_scientific_config_hash"] = (
        sha256_file(ROOT / bindings["primary_historical_config"]["path"])
        == bindings["primary_historical_config"]["sha256"]
    )
    runner_binding = bindings["dedicated_protocol_faithful_rerun_runner"]
    audit_binding = bindings["semantic_equivalence_audit_implementation"]
    checks["dedicated_runner_hash"] = (
        runner_binding.get("sha256") is None
        or sha256_file(RUNNER) == runner_binding["sha256"]
    )
    checks["semantic_audit_implementation_hash"] = (
        audit_binding.get("sha256") is None
        or sha256_file(Path(__file__)) == audit_binding["sha256"]
    )

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    local_functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    delegated = _called_primary_attributes(tree)
    checks["no_scientific_function_redefinition"] = not bool(
        local_functions & FORBIDDEN_LOCAL_REDEFINITIONS
    )
    checks["all_scientific_functions_delegated"] = sorted(
        REQUIRED_PRIMARY_DELEGATES - delegated
    ) == []
    checks["permitted_local_responsibilities_present"] = sorted(
        PERMITTED_LOCAL_RESPONSIBILITIES - local_functions
    ) == []
    checks["runner_imports_primary_module"] = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "turkish_ai_detection.experiments.primary"
        and any(alias.name == "run_fit" and alias.asname == "primary" for alias in node.names)
        for node in tree.body
    )

    failures = sorted(key for key, value in checks.items() if value is not True)
    if failures:
        raise RuntimeError(f"semantic-equivalence audit failed: {failures}")
    return {
        "schema_version": "external-cross-corpus-v2-semantic-audit-v1",
        "status": "PASS_STATIC_SEMANTIC_EQUIVALENCE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": contract_path.relative_to(ROOT).as_posix(),
        "contract_sha256": sha256_file(contract_path),
        "runner_path": RUNNER.relative_to(ROOT).as_posix(),
        "runner_sha256": sha256_file(RUNNER),
        "primary_reference_sha256": sha256_file(PRIMARY),
        "supcon_reference_sha256": sha256_file(SUPCON),
        "checks": checks,
        "scope": "Static delegation/identity audit; independent human review remains required.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    report = assert_semantic_equivalence(args.contract.resolve())
    if args.write_report is not None:
        write_json(args.write_report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
