"""Explicit historical-exact versus current-migrated reproduction policy.

This module never rewrites a freeze and never treats semantic equivalence as
byte identity. The six migrated workflows below have preserved historical
expected hashes, but their exact historical config/source bundles are absent.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


EXACT_UNAVAILABLE = "EXACT_HISTORICAL_REPRODUCTION_UNAVAILABLE"
CURRENT_MIGRATED = "CURRENT_MIGRATED_REPRODUCTION"


@dataclass(frozen=True)
class WorkflowBinding:
    config: str
    freeze: str
    freeze_config_key: str
    freeze_runner_key: str
    historical_config_sha256: str


WORKFLOWS = {
    "generator_holdout": WorkflowBinding(
        "configs/generator_holdout.json",
        "provenance/experiments/generator_holdout/PREEXECUTION_FREEZE.json",
        "config", "runner", "a606e0d775351cdc77eb43308a0cf10cf1b1a1037c2d0b5d6ddef4f21705be38",
    ),
    "domain_shift": WorkflowBinding(
        "configs/domain_shift.json",
        "provenance/experiments/domain_shift/PREEXECUTION_FREEZE.json",
        "config", "runner", "bf5fd8a4c099c05c42cb433d0968764a5d8ca19bf0bcec8b514bd0eece7d6fb9",
    ),
    "artifact_sensitivity": WorkflowBinding(
        "configs/artifact_sensitivity.json",
        "provenance/experiments/artifact_sensitivity/PREEXECUTION_FREEZE.json",
        "config", "runner", "eaea667604f6649aaed8332dd0e8b6d55f2d47e959adede9957fb9453f5d744d",
    ),
    "literature_comparison": WorkflowBinding(
        "configs/literature_comparison.json",
        "provenance/experiments/literature_comparison/PREEXECUTION_FREEZE.json",
        "config", "runner", "86e435fb05d9c11a7285364c145077f19de46e7b7214149cd3a1acf799d3545e",
    ),
    "calibration": WorkflowBinding(
        "configs/calibration.json",
        "provenance/experiments/calibration/PREEXECUTION_FREEZE.json",
        "rq4_config", "rq4_runner", "8184e8f792969e5f4e1ec5f2bdf2d424bfa140fbfe16692f43b67c5e8c582797",
    ),
    "near_duplicate_sensitivity": WorkflowBinding(
        "configs/near_duplicate_sensitivity.json",
        "provenance/experiments/near_duplicate_sensitivity/SPLIT_PREEXECUTION_FREEZE.json",
        "config", "scientific_runner", "65f8847823c282d027b1265a8d51741038a195d632eeb086504461fdf09d0a8b",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_exact_reproduction_status(root: Path, workflow: str) -> dict[str, str]:
    binding = WORKFLOWS[workflow]
    freeze_path = root / binding.freeze
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    config_record = freeze["files"][binding.freeze_config_key]
    runner_record = freeze["files"][binding.freeze_runner_key]
    if config_record["sha256"] != binding.historical_config_sha256:
        raise RuntimeError(f"{workflow}: immutable historical config hash is inconsistent")
    for record in (config_record, runner_record):
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"]).lower()):
            raise RuntimeError(f"{workflow}: malformed historical SHA-256")

    index = (root / "provenance/history/frozen_configs/README.md").read_text(encoding="utf-8")
    if binding.historical_config_sha256 not in index or "UNRECOVERABLE" not in index:
        raise RuntimeError(f"{workflow}: historical unavailability is not documented")
    return {
        "status": EXACT_UNAVAILABLE,
        "historical_config_path": str(config_record["path"]),
        "historical_config_sha256": str(config_record["sha256"]),
        "historical_runner_path": str(runner_record["path"]),
        "historical_runner_sha256": str(runner_record["sha256"]),
    }


def validate_current_migrated_reproduction(root: Path, workflow: str, runner: Path) -> dict[str, str]:
    """Validate current inputs without claiming historical byte identity."""
    binding = WORKFLOWS[workflow]
    config = root / binding.config
    freeze = root / binding.freeze
    audit = root / "docs/audits/active_config_semantic_equivalence_20261006.csv"
    for path, label in ((config, "active config"), (runner, "current runner"), (freeze, "historical freeze"), (audit, "semantic-equivalence audit")):
        if not path.is_file():
            raise RuntimeError(f"{workflow}: missing {label}: {path}")

    with audit.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("active_config") == binding.config.removeprefix("configs/")]
    if not rows or any(row.get("status") != "PASS" for row in rows):
        raise RuntimeError(f"{workflow}: active-config semantic equivalence is not documented as PASS")

    historical = historical_exact_reproduction_status(root, workflow)
    return {
        "reproduction_mode": CURRENT_MIGRATED,
        "historical_exact_reproduction": historical["status"],
        "current_config_path": binding.config,
        "current_config_sha256": sha256_file(config),
        "current_runner_path": runner.resolve().relative_to(root.resolve()).as_posix(),
        "current_runner_sha256": sha256_file(runner),
        "semantic_equivalence_audit": audit.relative_to(root).as_posix(),
        "semantic_equivalence_audit_sha256": sha256_file(audit),
        "historical_freeze_path": binding.freeze,
        "historical_freeze_sha256": sha256_file(freeze),
    }


def expected_run_hashes(
    run_record: dict,
    freeze: dict,
    current_policy: dict[str, str],
    *,
    config_key: str,
    runner_key: str,
) -> tuple[str, str]:
    """Return strict expected hashes for a historical or migrated run record.

    Unlabelled completed records predate the architecture migration and remain
    bound to the immutable freeze's recorded hashes. Newly produced records
    must declare CURRENT_MIGRATED_REPRODUCTION and bind to current bytes.
    """
    mode = run_record.get("reproduction_mode")
    if mode is None:
        return (
            str(freeze["files"][config_key]["sha256"]),
            str(freeze["files"][runner_key]["sha256"]),
        )
    if mode == CURRENT_MIGRATED:
        if run_record.get("historical_exact_reproduction") != EXACT_UNAVAILABLE:
            raise RuntimeError("migrated run does not disclose historical-exact unavailability")
        return current_policy["current_config_sha256"], current_policy["current_runner_sha256"]
    raise RuntimeError(f"unknown reproduction mode: {mode}")
