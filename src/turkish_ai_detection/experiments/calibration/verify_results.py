# -*- coding: utf-8 -*-
"""Post-run, pre-unblinding integrity verifier for COMBINED_V2_CALIBRATION_V1.

Does not fit a calibrator, choose thresholds, or compute Brier/log-loss/ECE/FPR/TPR.
"""
from __future__ import annotations
import hashlib, json, sys
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
from turkish_ai_detection.data.study import resolve_frozen_record_path  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import expected_run_hashes, validate_current_migrated_reproduction  # noqa: E402
CONFIG=ROOT/"configs"/"calibration.json"
FREEZE=ROOT/"provenance"/"experiments"/"calibration"/"PREEXECUTION_FREEZE.json"
RUNNER=ROOT/"src"/"turkish_ai_detection"/"experiments"/"calibration"/"run_fit.py"
OUT=ROOT/"results"/"calibration"
FOLDS=ROOT/"data"/"study"/"nested_folds.csv"
REPORT=ROOT/"provenance"/"experiments"/"calibration"/"RUN_INTEGRITY.json"


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def req(ok:bool,msg:str):
    if not ok: raise RuntimeError(msg)

def main()->int:
    req(CONFIG.is_file(),"missing RQ4 config"); req(FREEZE.is_file(),"missing RQ4 freeze")
    cfg=json.loads(CONFIG.read_text(encoding="utf-8")); freeze=json.loads(FREEZE.read_text(encoding="utf-8")); folds=pd.read_csv(FOLDS)
    req(freeze.get("status")=="FROZEN_READY_TO_RUN","invalid freeze status")
    req(freeze.get("calibration_results_unblinded_before_freeze") is False,"pre-result blinding not certified")
    req(freeze.get("scientific_rq4_outputs_present_before_freeze") is False,"clean RQ4 state not certified")
    # Verify every frozen file binding that still exists as an immutable source/code dependency.
    for key,rec in freeze["files"].items():
        if key not in {"rq4_config", "rq4_runner"}:
            resolve_frozen_record_path(ROOT, rec["path"], rec["sha256"])
    policy=validate_current_migrated_reproduction(ROOT,"calibration",RUNNER)

    # Verify frozen primary outer inputs without using their values for metrics.
    primary_records={(r["objective"],int(r["seed"]),int(r["outer_fold"])):r for r in freeze["source_primary_outer_artifacts"]}
    req(len(primary_records)==50,"frozen primary source cardinality != 50")
    primary_meta={}
    for key,rec in primary_records.items():
        pred=ROOT/rec["prediction"]["path"]; side=ROOT/rec["sidecar"]["path"]
        req(sha256_file(pred)==rec["prediction"]["sha256"],f"primary prediction changed: {key}")
        req(sha256_file(side)==rec["sidecar"]["sha256"],f"primary sidecar changed: {key}")
        frame=pd.read_csv(pred)
        need={"text_id","pair_id","label","domain","generator","outer_fold","seed","objective","condition","raw_decision_score"}
        req(need<=set(frame.columns),f"primary schema mismatch: {key}"); req(len(frame)==600 and frame.text_id.nunique()==600 and frame.pair_id.nunique()==300,f"primary shape mismatch: {key}")
        req(np.isfinite(frame.raw_decision_score.to_numpy(float)).all(),f"primary nonfinite score: {key}")
        primary_meta[key]=frame[["text_id","pair_id","label","domain","generator","outer_fold"]].sort_values("text_id").reset_index(drop=True)

    verified=0; pooled={}
    for objective in cfg["objectives"]:
        for seed in cfg["seeds"]:
            for outer in cfg["outer_folds"]:
                col=f"inner_fold_outer{outer}"; outer_train=set(folds.loc[folds.outer_fold.ne(outer),"pair_id"]); outer_test=set(folds.loc[folds.outer_fold.eq(outer),"pair_id"])
                parts=[]
                for inner in cfg["inner_folds"]:
                    base=OUT/"inner_oof"/objective/f"seed_{seed}"; pred=base/f"outer_{outer}_inner_{inner}_predictions.csv"; side=base/f"outer_{outer}_inner_{inner}_run.json"
                    req(pred.is_file(),f"missing inner prediction: {objective}/{seed}/{outer}/{inner}"); req(side.is_file(),f"missing inner sidecar: {objective}/{seed}/{outer}/{inner}")
                    run=json.loads(side.read_text(encoding="utf-8")); req(run.get("status")=="COMPLETE",f"incomplete inner fit: {objective}/{seed}/{outer}/{inner}")
                    req(run.get("experiment")==cfg["experiment"],"experiment mismatch"); req(run.get("objective")==objective,"objective mismatch"); req(int(run.get("seed"))==seed,"seed mismatch"); req(int(run.get("outer_fold"))==outer and int(run.get("inner_fold"))==inner,"fold mismatch")
                    expected_config,expected_runner=expected_run_hashes(run,freeze,policy,config_key="rq4_config",runner_key="rq4_runner")
                    req(run.get("config_sha256")==expected_config,"config hash mismatch"); req(run.get("runner_sha256")==expected_runner,"runner hash mismatch"); req(run.get("preexecution_freeze_sha256")==sha256_file(FREEZE),"freeze hash mismatch")
                    req(run.get("prediction_sha256")==sha256_file(pred),"inner prediction hash mismatch"); req(run.get("scientific_outer_test_rows_read")==0,"outer test was read by inner runner"); req(run.get("calibration_fit") is False and run.get("threshold_fit") is False,"unexpected calibration/threshold fit inside inner runner")
                    frame=pd.read_csv(pred); need={"text_id","pair_id","label","domain","generator","outer_fold","inner_fold","seed","objective","condition","raw_decision_score"}
                    req(need<=set(frame.columns),"inner schema mismatch"); req(len(frame)==600 and frame.text_id.nunique()==600 and frame.pair_id.nunique()==300,"inner shape mismatch"); req(np.isfinite(frame.raw_decision_score.to_numpy(float)).all(),"inner nonfinite score")
                    held=set(frame.pair_id); expected=set(folds.loc[folds.outer_fold.ne(outer)&folds[col].eq(inner),"pair_id"]); req(held==expected,"inner heldout pair identity mismatch"); req(not (held&outer_test),"outer leakage in heldout")
                    counts=frame.label.astype(str).value_counts().to_dict(); req(counts.get("HUMAN",0)==300 and counts.get("AI",0)==300,"inner class imbalance")
                    parts.append(frame); verified+=1
                oof=pd.concat(parts,ignore_index=True); req(len(oof)==2400 and oof.text_id.nunique()==2400 and oof.pair_id.nunique()==1200,"outer-training OOF coverage mismatch"); req(set(oof.pair_id)==outer_train,"OOF does not exactly cover outer training pairs")
                pooled[(objective,seed,outer)]=oof[["text_id","pair_id","label","domain","generator","outer_fold"]].sort_values("text_id").reset_index(drop=True)
    req(verified==200,f"verified inner fit count != 200: {verified}")

    # CE/SupCon OOF heldout identity must be exactly paired.
    for seed in cfg["seeds"]:
        for outer in cfg["outer_folds"]:
            req(pooled[("ce",seed,outer)].equals(pooled[("supcon",seed,outer)]),f"CE/SupCon inner OOF alignment failed: {seed}/{outer}")
            req(primary_meta[("ce",seed,outer)].equals(primary_meta[("supcon",seed,outer)]),f"CE/SupCon primary outer alignment failed: {seed}/{outer}")

    final_dir=OUT/"final_statistics"; req(not final_dir.exists() or not any(final_dir.iterdir()),"RQ4 final statistics already exist before pre-unblinding verifier")
    report={"schema_version":"combined-v2-calibration-completion-integrity-v1","status":"PASS","results_unblinded":False,"scientific_metrics_computed":False,"verified_inner_fits":200,"verified_inner_predictions":200,"inner_rows_per_fit":600,"inner_pairs_per_fit":300,"oof_rows_per_outer_seed_objective":2400,"oof_pairs_per_outer_seed_objective":1200,"primary_outer_inputs_verified":50,"primary_outer_rows_per_fit":600,"prediction_hashes_verified":True,"freeze_file_bindings_verified":True,"outer_test_inner_training_use_verified_zero":True,"calibrator_not_fit_by_verifier":True,"threshold_not_fit_by_verifier":True,"finite_scores_verified_without_revealing_values":True,"ce_supcon_oof_alignment_verified":True,"freeze_sha256":sha256_file(FREEZE),"active_migrated_config_sha256":policy["current_config_sha256"],"active_migrated_runner_sha256":policy["current_runner_sha256"],"historical_exact_reproduction":policy["historical_exact_reproduction"],"next_step":"Run src/turkish_ai_detection/experiments/calibration/summarize_statistics.py for the prespecified one-time RQ4 unblinding."}
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
