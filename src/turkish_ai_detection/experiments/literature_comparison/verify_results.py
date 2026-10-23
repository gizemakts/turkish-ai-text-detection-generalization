# -*- coding: utf-8 -*-
"""Post-run pre-unblinding verifier for LIT2026_REIMPL."""
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
SCRIPTS=ROOT/"src"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from turkish_ai_detection.data.study import resolve_frozen_record_path, sha256_file, write_json  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import expected_run_hashes, validate_current_migrated_reproduction  # noqa: E402

CONFIG=ROOT/"configs"/"literature_comparison.json"
RUNNER=ROOT/"src"/"turkish_ai_detection/experiments/literature_comparison/run_fit.py"
FREEZE=ROOT/"provenance"/"experiments"/"literature_comparison"/"PREEXECUTION_FREEZE.json"
OUT=ROOT/"results"/"literature_comparison"
REPORT=ROOT/"provenance"/"experiments"/"literature_comparison"/"RUN_INTEGRITY.json"
SEEDS=[42,314,2026]
OUTERS=[0,1,2,3,4]

def req(ok,msg):
    if not ok: raise RuntimeError(msg)

def main():
    fr=json.loads(FREEZE.read_text(encoding="utf-8"))
    req(fr.get("status")=="FROZEN_READY_TO_RUN","invalid freeze")
    policy=validate_current_migrated_reproduction(ROOT,"literature_comparison",RUNNER)

    frames={}
    verified=0
    val_hash_by_outer={}
    for seed in SEEDS:
        all_pairs=set()
        for outer in OUTERS:
            base=OUT/f"seed_{seed}"
            pred=base/f"outer_{outer}_predictions.csv"
            side=base/f"outer_{outer}_run.json"
            req(pred.is_file() and side.is_file(),f"missing seed={seed}/outer={outer}")
            run=json.loads(side.read_text(encoding="utf-8"))
            req(run.get("status")=="COMPLETE",f"incomplete seed={seed}/outer={outer}")
            req(run["prediction_sha256"]==sha256_file(pred),f"prediction hash seed={seed}/outer={outer}")
            expected_config,expected_runner=expected_run_hashes(run,fr,policy,config_key="config",runner_key="runner")
            req(run["runner_sha256"]==expected_runner,f"runner binding seed={seed}/outer={outer}")
            req(run["config_sha256"]==expected_config,f"config binding seed={seed}/outer={outer}")
            req(run["preexecution_freeze_sha256"]==sha256_file(FREEZE),f"freeze binding seed={seed}/outer={outer}")
            req(run["outer_test_rows_used_for_training"]==0,f"outer test training use seed={seed}/outer={outer}")
            req(run["outer_test_used_for_model_selection"] is False,f"outer test model selection seed={seed}/outer={outer}")
            req(run["train_pair_count"]==1050 and run["validation_pair_count"]==150 and run["test_pair_count"]==300,"70/10/20 pair counts")
            req(run["train_validation_overlap"]==0 and run["train_test_overlap"]==0 and run["validation_test_overlap"]==0,"split overlap")
            if outer not in val_hash_by_outer:
                val_hash_by_outer[outer]=run["validation_pair_sha256"]
            req(val_hash_by_outer[outer]==run["validation_pair_sha256"],f"validation split changed across seeds outer={outer}")

            f=pd.read_csv(pred).sort_values("text_id").reset_index(drop=True)
            req(len(f)==600 and f.text_id.nunique()==600 and f.pair_id.nunique()==300,f"shape seed={seed}/outer={outer}")
            req(set(f.method.astype(str))=={"LIT2026_REIMPL"},"method column")
            req(set(f.condition.astype(str))=={"A"},"condition column")
            req(np.isfinite(f.raw_decision_score.to_numpy(float)).all(),"nonfinite scores")
            req(set(f.predicted_label.astype(int)).issubset({0,1}),"invalid predicted labels")
            req(not (all_pairs & set(f.pair_id.astype(str))),f"OOF overlap seed={seed}")
            all_pairs |= set(f.pair_id.astype(str))
            frames[(seed,outer)] = f
            verified += 1
        req(len(all_pairs)==1500,f"OOF does not cover all pairs seed={seed}")
    req(verified==15,f"verified fits !=15: {verified}")

    keys=["text_id","pair_id","label","domain","generator"]
    for outer in OUTERS:
        ref=frames[(SEEDS[0],outer)][keys]
        for seed in SEEDS[1:]:
            req(frames[(seed,outer)][keys].equals(ref),f"test alignment failed outer={outer}, seed={seed}")

    report={
        "schema_version":"combined-v2-lit2026-completion-integrity-v1",
        "status":"PASS",
        "results_unblinded":False,
        "scientific_metrics_computed":False,
        "verified_fits":15,
        "verified_predictions":15,
        "rows_per_fit":600,
        "pairs_per_fit":300,
        "five_outer_folds_cover_all_1500_pairs_per_seed":True,
        "validation_split_identical_across_seeds":True,
        "outer_test_training_use_verified_zero":True,
        "outer_test_model_selection_use_verified_false":True,
        "prediction_hashes_verified":True,
        "freeze_runner_config_bindings_verified":True,
        "finite_raw_scores_verified_without_revealing_values":True,
        "freeze_sha256":sha256_file(FREEZE),
        "active_migrated_config_sha256":policy["current_config_sha256"],
        "active_migrated_runner_sha256":policy["current_runner_sha256"],
        "historical_exact_reproduction":policy["historical_exact_reproduction"],
        "next_step":"Run src/turkish_ai_detection/experiments/literature_comparison/summarize_statistics.py for one-time literature-reimplementation unblinding."
    }
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    write_json(REPORT,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
