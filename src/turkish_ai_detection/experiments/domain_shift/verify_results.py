# -*- coding: utf-8 -*-
"""Post-run pre-unblinding integrity verifier for DOMAIN_SHIFT_V1."""
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

CONFIG = ROOT / "configs" / "domain_shift.json"
RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/domain_shift/run_fit.py"
FREEZE = ROOT / "provenance" / "experiments" / "domain_shift" / "PREEXECUTION_FREEZE.json"
OUT = ROOT / "results" / "domain_shift"
REPORT = ROOT / "provenance" / "experiments" / "domain_shift" / "RUN_INTEGRITY.json"

DIRECTIONS = {
    "abstract_to_newspaper": ("ABSTRACT", "NEWSPAPER", 1000, 500),
    "newspaper_to_abstract": ("NEWSPAPER", "ABSTRACT", 2000, 1000),
}
SEEDS = [42, 314, 2026]
OBJECTIVES = ["ce", "supcon"]


def req(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    req(freeze.get("status") == "FROZEN_READY_TO_RUN", "invalid freeze")
    policy = validate_current_migrated_reproduction(ROOT, "domain_shift", RUNNER)

    frames = {}
    verified = 0
    for objective in OBJECTIVES:
        for seed in SEEDS:
            all_target_pairs = set()
            for direction, (_train_domain, test_domain, expected_rows, expected_pairs) in DIRECTIONS.items():
                base = OUT / objective / f"seed_{seed}"
                pred = base / f"{direction}_predictions.csv"
                side = base / f"{direction}_run.json"
                req(pred.is_file() and side.is_file(), f"missing fit {objective}/{seed}/{direction}")
                run = json.loads(side.read_text(encoding="utf-8"))
                req(run.get("status") == "COMPLETE", f"incomplete {objective}/{seed}/{direction}")
                req(run["prediction_sha256"] == sha256_file(pred), f"prediction hash mismatch {objective}/{seed}/{direction}")
                expected_config, expected_runner = expected_run_hashes(run, freeze, policy, config_key="config", runner_key="runner")
                req(run["runner_sha256"] == expected_runner, f"runner mismatch {objective}/{seed}/{direction}")
                req(run["config_sha256"] == expected_config, f"config mismatch {objective}/{seed}/{direction}")
                req(run["preexecution_freeze_sha256"] == sha256_file(FREEZE), f"freeze mismatch {objective}/{seed}/{direction}")
                req(run["heldout_domain_training_rows"] == 0, f"target-domain training leakage {objective}/{seed}/{direction}")
                req(run["heldout_domain_used_for_model_selection"] is False, f"target-domain model-selection leakage {objective}/{seed}/{direction}")
                req(run["test_rows_used_for_training"] == 0, f"test training leakage {objective}/{seed}/{direction}")
                req(run["calibration_used"] is False and run["threshold_used"] is False, "unexpected calibration/threshold")

                f = pd.read_csv(pred).sort_values("text_id").reset_index(drop=True)
                req(len(f) == expected_rows and f.text_id.nunique() == expected_rows and f.pair_id.nunique() == expected_pairs,
                    f"shape mismatch {objective}/{seed}/{direction}")
                req(set(f.domain.astype(str)) == {test_domain}, f"target domain purity failure {objective}/{seed}/{direction}")
                req(set(f.test_domain.astype(str)) == {test_domain}, f"test_domain column failure {objective}/{seed}/{direction}")
                req(set(f.direction.astype(str)) == {direction}, f"direction column failure {objective}/{seed}/{direction}")
                req(set(f.seed.astype(int)) == {seed}, f"seed column failure {objective}/{seed}/{direction}")
                req(set(f.objective.astype(str)) == {objective}, f"objective column failure {objective}/{seed}/{direction}")
                req(np.isfinite(f.raw_decision_score.to_numpy(float)).all(), f"nonfinite score {objective}/{seed}/{direction}")

                frames[(objective, seed, direction)] = f
                req(not (all_target_pairs & set(f.pair_id.astype(str))), f"direction target overlap {objective}/{seed}")
                all_target_pairs |= set(f.pair_id.astype(str))
                verified += 1
            req(len(all_target_pairs) == 1500, f"two target domains do not cover all 1500 pairs: {objective}/{seed}")

    req(verified == 12, f"verified fit count != 12: {verified}")

    keys = ["text_id", "pair_id", "label", "domain", "generator"]
    for seed in SEEDS:
        for direction in DIRECTIONS:
            ref = frames[("ce", seed, direction)][keys]
            cur = frames[("supcon", seed, direction)][keys]
            req(cur.equals(ref), f"CE/SupCon target alignment failed {seed}/{direction}")

    report = {
        "schema_version": "combined-v2-domain-shift-completion-integrity-v1",
        "status": "PASS",
        "results_unblinded": False,
        "scientific_metrics_computed": False,
        "verified_fits": 12,
        "verified_predictions": 12,
        "direction_target_rows": {
            "abstract_to_newspaper": 1000,
            "newspaper_to_abstract": 2000
        },
        "two_target_domains_cover_all_1500_pairs_per_seed_objective": True,
        "heldout_domain_training_rows_verified_zero": True,
        "heldout_domain_model_selection_use_verified_false": True,
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
        "next_step": "Run src/turkish_ai_detection/experiments/domain_shift/summarize_statistics.py for one-time domain-shift unblinding."
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
