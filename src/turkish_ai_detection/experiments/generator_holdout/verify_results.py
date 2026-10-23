# -*- coding: utf-8 -*-
"""Post-run pre-unblinding integrity verifier for GENERATOR_HOLDOUT_V1."""
from __future__ import annotations

import sys
import json, sys
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
SCRIPTS = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from turkish_ai_detection.data.study import resolve_frozen_record_path, sha256_file, write_json  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import expected_run_hashes, validate_current_migrated_reproduction  # noqa: E402

CONFIG = ROOT / "configs" / "generator_holdout.json"
RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/generator_holdout/run_fit.py"
FREEZE = ROOT / "provenance" / "experiments" / "generator_holdout" / "PREEXECUTION_FREEZE.json"
OUT = ROOT / "results" / "generator_holdout"
REPORT = ROOT / "provenance" / "experiments" / "generator_holdout" / "RUN_INTEGRITY.json"

GENERATORS = ["Claude", "GPT", "Gemini", "Kimi", "Qwen"]
SEEDS = [42, 314, 2026]
OBJECTIVES = ["ce", "supcon"]


def slug(x):
    return x.lower().replace(" ", "_")


def req(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    req(freeze.get("status") == "FROZEN_READY_TO_RUN", "invalid freeze")
    policy = validate_current_migrated_reproduction(ROOT, "generator_holdout", RUNNER)

    verified = 0
    frames = {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            heldout_seen = set()
            for generator in GENERATORS:
                base = OUT / objective / f"seed_{seed}"
                pred = base / f"holdout_{slug(generator)}_predictions.csv"
                side = base / f"holdout_{slug(generator)}_run.json"
                req(pred.is_file() and side.is_file(), f"missing fit {objective}/{seed}/{generator}")
                run = json.loads(side.read_text(encoding="utf-8"))
                req(run.get("status") == "COMPLETE", f"incomplete {objective}/{seed}/{generator}")
                req(run["prediction_sha256"] == sha256_file(pred), f"prediction hash mismatch {objective}/{seed}/{generator}")
                expected_config, expected_runner = expected_run_hashes(run, freeze, policy, config_key="config", runner_key="runner")
                req(run["runner_sha256"] == expected_runner, f"runner mismatch {objective}/{seed}/{generator}")
                req(run["config_sha256"] == expected_config, f"config mismatch {objective}/{seed}/{generator}")
                req(run["preexecution_freeze_sha256"] == sha256_file(FREEZE), f"freeze mismatch {objective}/{seed}/{generator}")
                req(run["heldout_generator_training_rows"] == 0, f"heldout training leakage {objective}/{seed}/{generator}")
                req(run["test_rows_used_for_training"] == 0, f"test training leakage {objective}/{seed}/{generator}")
                req(run["heldout_generator_used_for_model_selection"] is False, f"heldout model-selection leakage {objective}/{seed}/{generator}")
                req(run["calibration_used"] is False and run["threshold_used"] is False, "unexpected calibration/threshold")

                f = pd.read_csv(pred).sort_values("text_id").reset_index(drop=True)
                req(len(f) == 600 and f.text_id.nunique() == 600 and f.pair_id.nunique() == 300,
                    f"shape mismatch {objective}/{seed}/{generator}")
                req(set(f.generator.astype(str)) == {generator}, f"generator purity failure {objective}/{seed}/{generator}")
                req(set(f.heldout_generator.astype(str)) == {generator}, f"heldout column failure {objective}/{seed}/{generator}")
                req(set(f.seed.astype(int)) == {seed}, f"seed column failure {objective}/{seed}/{generator}")
                req(set(f.objective.astype(str)) == {objective}, f"objective column failure {objective}/{seed}/{generator}")
                req(np.isfinite(f.raw_decision_score.to_numpy(float)).all(), f"nonfinite score {objective}/{seed}/{generator}")
                frames[(objective, seed, generator)] = f
                heldout_seen |= set(f.pair_id.astype(str))
                verified += 1
            req(len(heldout_seen) == 1500, f"five holdouts do not cover 1500 pairs: {objective}/{seed}")

    req(verified == 30, f"verified fit count != 30: {verified}")

    keys = ["text_id", "pair_id", "label", "domain", "generator"]
    for seed in SEEDS:
        for generator in GENERATORS:
            ref = frames[("ce", seed, generator)][keys]
            cur = frames[("supcon", seed, generator)][keys]
            req(cur.equals(ref), f"CE/SupCon paired alignment failed {seed}/{generator}")

    report = {
        "schema_version": "combined-v2-generator-holdout-completion-integrity-v1",
        "status": "PASS",
        "results_unblinded": False,
        "scientific_metrics_computed": False,
        "verified_fits": 30,
        "verified_predictions": 30,
        "rows_per_fit": 600,
        "pairs_per_fit": 300,
        "five_holdouts_cover_all_1500_pairs_per_seed_objective": True,
        "heldout_generator_training_rows_verified_zero": True,
        "heldout_generator_model_selection_use_verified_false": True,
        "test_training_use_verified_zero": True,
        "calibration_absent": True,
        "thresholding_absent": True,
        "prediction_hashes_verified": True,
        "freeze_runner_config_bindings_verified": True,
        "paired_ce_supcon_alignment_verified": True,
        "finite_raw_scores_verified_without_revealing_values": True,
        "freeze_sha256": sha256_file(FREEZE),
        "active_migrated_config_sha256": policy["current_config_sha256"],
        "active_migrated_runner_sha256": policy["current_runner_sha256"],
        "historical_exact_reproduction": policy["historical_exact_reproduction"],
        "next_step": "Run src/turkish_ai_detection/experiments/generator_holdout/summarize_statistics.py for one-time OOD unblinding."
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
