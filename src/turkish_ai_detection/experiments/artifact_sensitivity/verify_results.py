# -*- coding: utf-8 -*-
"""Post-run pre-unblinding integrity verifier for Condition B."""
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

CONFIG = ROOT / "configs" / "artifact_sensitivity.json"
RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/artifact_sensitivity/run_fit.py"
FREEZE = ROOT / "provenance" / "experiments" / "artifact_sensitivity" / "PREEXECUTION_FREEZE.json"
OUT = ROOT / "results" / "artifact_sensitivity"
REPORT = ROOT / "provenance" / "experiments" / "artifact_sensitivity" / "RUN_INTEGRITY.json"
SEEDS=[42,314,2026]
OUTERS=[0,1,2,3,4]
OBJECTIVES=["ce","supcon"]


def req(ok,msg):
    if not ok: raise RuntimeError(msg)


def main():
    freeze=json.loads(FREEZE.read_text(encoding="utf-8"))
    req(freeze.get("status")=="FROZEN_READY_TO_RUN","invalid freeze")
    policy=validate_current_migrated_reproduction(ROOT,"artifact_sensitivity",RUNNER)

    frames={}
    verified=0
    for obj in OBJECTIVES:
        for seed in SEEDS:
            all_pairs=set()
            for outer in OUTERS:
                base=OUT/obj/f"seed_{seed}"
                pred=base/f"outer_{outer}_predictions.csv"
                side=base/f"outer_{outer}_run.json"
                req(pred.is_file() and side.is_file(),f"missing {obj}/{seed}/{outer}")
                run=json.loads(side.read_text(encoding="utf-8"))
                req(run.get("status")=="COMPLETE",f"incomplete {obj}/{seed}/{outer}")
                req(run["prediction_sha256"]==sha256_file(pred),f"prediction hash {obj}/{seed}/{outer}")
                expected_config,expected_runner=expected_run_hashes(run,freeze,policy,config_key="config",runner_key="runner")
                req(run["runner_sha256"]==expected_runner,f"runner binding {obj}/{seed}/{outer}")
                req(run["config_sha256"]==expected_config,f"config binding {obj}/{seed}/{outer}")
                req(run["preexecution_freeze_sha256"]==sha256_file(FREEZE),f"freeze binding {obj}/{seed}/{outer}")
                req(run["outer_test_rows_used_for_training"]==0,f"test training leakage {obj}/{seed}/{outer}")
                req(run["outer_test_used_for_model_selection"] is False,f"test model-selection leakage {obj}/{seed}/{outer}")
                req(run["calibration_used"] is False and run["threshold_used"] is False,"unexpected calibration/threshold")
                f=pd.read_csv(pred).sort_values("text_id").reset_index(drop=True)
                req(len(f)==600 and f.text_id.nunique()==600 and f.pair_id.nunique()==300,f"shape {obj}/{seed}/{outer}")
                req(set(f.condition.astype(str))=={"B"},f"condition {obj}/{seed}/{outer}")
                req(set(f.outer_fold.astype(int))=={outer},f"outer {obj}/{seed}/{outer}")
                req(set(f.seed.astype(int))=={seed},f"seed {obj}/{seed}/{outer}")
                req(set(f.objective.astype(str))=={obj},f"objective {obj}/{seed}/{outer}")
                req(np.isfinite(f.raw_decision_score.to_numpy(float)).all(),f"nonfinite {obj}/{seed}/{outer}")
                req(not (all_pairs & set(f.pair_id.astype(str))),f"OOF pair overlap {obj}/{seed}")
                all_pairs |= set(f.pair_id.astype(str))
                frames[(obj,seed,outer)] = f
                verified += 1
            req(len(all_pairs)==1500,f"OOF does not cover 1500 pairs {obj}/{seed}")

    req(verified==30,f"verified fits != 30: {verified}")
    keys=["text_id","pair_id","label","domain","generator"]
    for seed in SEEDS:
        for outer in OUTERS:
            req(
                frames[("ce",seed,outer)][keys].equals(frames[("supcon",seed,outer)][keys]),
                f"CE/SupCon alignment failed {seed}/{outer}"
            )

    report={
        "schema_version":"combined-v2-condition-b-completion-integrity-v1",
        "status":"PASS",
        "results_unblinded":False,
        "scientific_metrics_computed":False,
        "verified_fits":30,
        "verified_predictions":30,
        "rows_per_fit":600,
        "pairs_per_fit":300,
        "five_outer_folds_cover_all_1500_pairs_per_seed_objective":True,
        "outer_test_training_use_verified_zero":True,
        "outer_test_model_selection_use_verified_false":True,
        "calibration_absent":True,
        "thresholding_absent":True,
        "prediction_hashes_verified":True,
        "freeze_runner_config_bindings_verified":True,
        "paired_ce_supcon_alignment_verified":True,
        "finite_raw_scores_verified_without_revealing_values":True,
        "freeze_sha256":sha256_file(FREEZE),
        "active_migrated_config_sha256":policy["current_config_sha256"],
        "active_migrated_runner_sha256":policy["current_runner_sha256"],
        "historical_exact_reproduction":policy["historical_exact_reproduction"],
        "next_step":"Run src/turkish_ai_detection/experiments/artifact_sensitivity/summarize_statistics.py for one-time Condition-B unblinding."
    }
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    write_json(REPORT,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
