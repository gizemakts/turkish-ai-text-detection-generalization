"""Small command-line interface for validation and workflow discovery."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from turkish_ai_detection.paths import PROJECT_ROOT
from turkish_ai_detection.validation.environment import (
    data_availability_messages,
    require_study_data,
    validate_package,
)
from turkish_ai_detection.validation.scientific import validate_all

WORKFLOWS = {
    "primary": Path("pipelines/03_primary/01_train.py"),
    "generator-holdout": Path("pipelines/04_robustness/generator_holdout.py"),
    "domain-shift": Path("pipelines/04_robustness/domain_shift.py"),
    "artifact-sensitivity": Path("pipelines/04_robustness/artifact_sensitivity.py"),
    "near-duplicate-sensitivity": Path("pipelines/04_robustness/near_duplicate_sensitivity.py"),
    "literature-comparison": Path("pipelines/05_literature_comparison/01_train.py"),
    "calibration": Path("pipelines/06_calibration/01_calibrate.py"),
}


def _validate() -> int:
    failures = validate_package() + validate_all()
    if failures:
        print("Validation: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Validation: PASS")
    print("  Configured scientific files available to this validator passed its checks.")
    for message in data_availability_messages():
        print(f"  NOTE: {message}")
    return 0


def _status() -> int:
    print("Turkish AI-Generated Text Detection Study")
    print("  Included materials: research code, AI-generated texts, aggregate results, configurations and partitions")
    print("  Scientific packages: primary, literature comparison, generator holdout,")
    print("    domain shift, artifact sensitivity, calibration, near-duplicate sensitivity")
    print("  NEWSPAPER: paired-corpus, author-disjoint and domain-transfer analyses")
    print("  Excluded materials: human-written source texts, model weights and per-document predictions")
    print("  Release metadata: MIT license and software citation metadata are included")
    return 0


def _run(name: str, execute: bool, extra: list[str]) -> int:
    path = PROJECT_ROOT / WORKFLOWS[name]
    if not execute:
        print(f"Workflow: {path.relative_to(PROJECT_ROOT)}")
        print("Dry run only. Use --execute with the required study data and model assets to run the workflow.")
        return 0
    available, message = require_study_data()
    if not available:
        print(message)
        return 2
    return subprocess.call([sys.executable, str(path), *extra], cwd=PROJECT_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="turkish_ai_detection", description="Turkish AI-Generated Text Detection Study research utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="read-only package and finalized-result validation")
    sub.add_parser("status", help="show the current scientific status")
    run = sub.add_parser("run", help="locate or explicitly execute a workflow")
    run.add_argument("workflow", choices=sorted(WORKFLOWS))
    run.add_argument("--execute", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if args.command == "validate":
        return _validate()
    if args.command == "status":
        return _status()
    return _run(args.workflow, args.execute, extra)


if __name__ == "__main__":
    raise SystemExit(main())

