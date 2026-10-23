# -*- coding: utf-8 -*-
"""Post-run, pre-unblinding integrity verifier for COMBINED_V2_PRIMARY_ABLATION_V1.

Does NOT compute or print AUROC, AUPRC, pAUC, fold metrics, score summaries,
or CE-vs-SupCon deltas. It verifies completeness, run identity, row counts,
fold coverage, finiteness, and paired alignment only.
"""
from __future__ import annotations

import sys

import json
from pathlib import Path

import numpy as np
import pandas as pd

def _resolve_project_root() -> Path:
    for _candidate in Path(__file__).resolve().parents:
        if (_candidate / ".gitignore").is_file():
            return _candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study project root could not be resolved")

ROOT = _resolve_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from turkish_ai_detection.experiments.primary.integrity import (  # noqa: E402
    load_active_folds, verify_fit_integrity,
)
CONFIG = ROOT / "configs" / "primary_evaluation.json"
RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/primary/run_fit.py"
OUT = ROOT / "results" / "primary"

SEEDS = [42, 137, 314, 2026, 3407]
OUTERS = [0, 1, 2, 3, 4]
OBJECTIVES = ["ce", "supcon"]


def req(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def main() -> int:
    req(CONFIG.is_file(), f"missing config: {CONFIG}")
    req(RUNNER.is_file(), f"missing runner: {RUNNER}")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = load_active_folds(ROOT, cfg)
    req(cfg["seeds"] == SEEDS, "seed plan does not match active configuration")
    req(cfg["outer_folds"] == OUTERS, "outer-fold plan does not match active configuration")
    req(cfg["planned_fits"] == 50, "planned fit count changed")

    pooled = {}
    verified = 0

    for objective in OBJECTIVES:
        for seed in SEEDS:
            parts = []
            for outer in OUTERS:
                base = OUT / objective / f"seed_{seed}"
                pred = base / f"outer_{outer}_predictions.csv"
                sidecar = base / f"outer_{outer}_run.json"

                req(pred.is_file(), f"missing prediction: {objective}/{seed}/{outer}")
                req(sidecar.is_file(), f"missing sidecar: {objective}/{seed}/{outer}")

                run = json.loads(sidecar.read_text(encoding="utf-8"))
                verify_fit_integrity(
                    root=ROOT, cfg=cfg, config_path=CONFIG, folds=folds,
                    objective=objective, seed=seed, outer=outer,
                    sidecar=run, prediction_path=pred,
                )

                frame = pd.read_csv(pred)
                needed = {
                    "text_id", "pair_id", "label", "domain", "generator",
                    "outer_fold", "seed", "objective", "condition",
                    "raw_decision_score",
                }
                req(needed <= set(frame.columns), f"schema mismatch: {objective}/{seed}/{outer}")
                req(len(frame) == 600, f"row count != 600: {objective}/{seed}/{outer}")
                req(frame.text_id.nunique() == 600, f"text IDs not unique: {objective}/{seed}/{outer}")
                req(frame.pair_id.nunique() == 300, f"pair count != 300: {objective}/{seed}/{outer}")
                req(set(frame.outer_fold.astype(int)) == {outer}, f"fold column mismatch: {objective}/{seed}/{outer}")
                req(set(frame.seed.astype(int)) == {seed}, f"seed column mismatch: {objective}/{seed}/{outer}")
                req(set(frame.objective.astype(str)) == {objective}, f"objective column mismatch: {objective}/{seed}/{outer}")
                req(set(frame.condition.astype(str)) == {"A"}, f"condition column mismatch: {objective}/{seed}/{outer}")
                req(np.isfinite(frame.raw_decision_score.to_numpy(float)).all(), f"non-finite score: {objective}/{seed}/{outer}")

                counts = frame.label.astype(str).value_counts().to_dict()
                req(counts.get("HUMAN", 0) == 300 and counts.get("AI", 0) == 300,
                    f"class balance mismatch: {objective}/{seed}/{outer}")

                parts.append(frame)
                verified += 1

            oof = pd.concat(parts, ignore_index=True).sort_values("text_id").reset_index(drop=True)
            req(len(oof) == 3000, f"pooled rows != 3000: {objective}/{seed}")
            req(oof.text_id.nunique() == 3000, f"pooled text IDs != 3000: {objective}/{seed}")
            req(oof.pair_id.nunique() == 1500, f"pooled pairs != 1500: {objective}/{seed}")
            req(set(oof.outer_fold.astype(int)) == set(OUTERS), f"fold coverage mismatch: {objective}/{seed}")
            pooled[(objective, seed)] = oof

    req(verified == 50, f"verified fit count != 50: {verified}")

    ref = pooled[("ce", SEEDS[0])][
        ["text_id", "pair_id", "label", "domain", "generator", "outer_fold"]
    ].reset_index(drop=True)

    for key, frame in pooled.items():
        cur = frame[
            ["text_id", "pair_id", "label", "domain", "generator", "outer_fold"]
        ].reset_index(drop=True)
        req(cur.equals(ref), f"paired OOF alignment failed: {key}")

    report = {
        "schema_version": "combined-v2-primary-ablation-completion-integrity-v1",
        "status": "PASS",
        "results_unblinded": False,
        "scientific_metrics_computed": False,
        "verified_fits": 50,
        "verified_predictions": 50,
        "rows_per_fit": 600,
        "rows_per_seed_objective_oof": 3000,
        "pairs_per_seed_objective_oof": 1500,
        "sidecar_identity_verified": True,
        "outer_test_training_use_verified_zero": True,
        "calibration_absent": True,
        "thresholding_absent": True,
        "finite_raw_scores_verified_without_revealing_values": True,
        "paired_oof_alignment_verified": True,
        "next_step": "Run src/turkish_ai_detection/experiments/primary/summarize_statistics.py for the prespecified one-time unblinding."
    }

    report_path = ROOT / "provenance" / "experiments" / "primary" / "RUN_INTEGRITY.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
