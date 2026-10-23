# -*- coding: utf-8 -*-
"""One-time RQ4 statistics/unblinding for COMBINED_V2_CALIBRATION_V1.

Requires RUN_INTEGRITY.json status PASS. Fits Platt calibrators
only on outer-training inner-OOF scores, chooses NP-inspired thresholds only on
OOF HUMAN probabilities, applies both to frozen primary outer-test raw scores,
and then computes the prespecified calibration/operating-point statistics.
"""
from __future__ import annotations
import hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

def _resolve_project_root() -> Path:
    for _candidate in Path(__file__).resolve().parents:
        if (_candidate / ".gitignore").is_file():
            return _candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study project root could not be resolved")

ROOT = _resolve_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SCRIPTS=ROOT/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from turkish_ai_detection.data.study import np_umbrella_k, resolve_frozen_record_path  # noqa: E402
from turkish_ai_detection.evaluation.metrics import calibrated_score, fit_platt  # noqa: E402
from turkish_ai_detection.features.core import ece_equal_width  # noqa: E402

CONFIG=ROOT/"configs"/"calibration.json"
FREEZE=ROOT/"provenance"/"experiments"/"calibration"/"PREEXECUTION_FREEZE.json"
VERIFY=ROOT/"provenance"/"experiments"/"calibration"/"RUN_INTEGRITY.json"
OUT=ROOT/"results"/"calibration"
FINAL=OUT/"final_statistics"


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def req(ok:bool,msg:str):
    if not ok: raise RuntimeError(msg)

def atomic_csv(df:pd.DataFrame,p:Path):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+".tmp"); df.to_csv(tmp,index=False); tmp.replace(p)

def label_array(frame:pd.DataFrame)->np.ndarray:
    return frame.label.astype(str).str.upper().eq("AI").astype(int).to_numpy()

def threshold_np(prob:np.ndarray,y:np.ndarray,alpha:float,delta:float)->tuple[float,int]:
    human=np.sort(np.asarray(prob)[np.asarray(y)==0])[::-1]
    k=np_umbrella_k(len(human),alpha=alpha,delta=delta)
    return float(human[k]),int(k)

def metric_bundle(frame:pd.DataFrame, alphas:list[float])->dict[str,float]:
    y=label_array(frame); p=frame.calibrated_study_ai_score.to_numpy(float)
    out={"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),"ece_10_equal_width":float(ece_equal_width(y,p,10))}
    for a in alphas:
        tag=f"{a:.2f}"; pred=frame[f"binary_prediction_alpha_{tag}"].to_numpy(int)
        human=(y==0); ai=(y==1)
        out[f"realized_fpr_alpha_{tag}"]=float(np.mean(pred[human]))
        out[f"tpr_alpha_{tag}"]=float(np.mean(pred[ai]))
    return out

def reliability_bins(frame:pd.DataFrame, objective:str, seed:int, bins:int=10)->pd.DataFrame:
    y=label_array(frame); p=frame.calibrated_study_ai_score.to_numpy(float); idx=np.minimum((p*bins).astype(int),bins-1)
    rows=[]
    for b in range(bins):
        m=idx==b; n=int(m.sum())
        rows.append({"objective":objective,"method":"CE_ONLY_MATCHED" if objective=="ce" else "SUPCON","seed":seed,"bin":b,"lower":b/bins,"upper":1.0 if b==bins-1 else (b+1)/bins,"n":n,"mean_predicted":float(p[m].mean()) if n else np.nan,"observed_ai_rate":float(y[m].mean()) if n else np.nan})
    return pd.DataFrame(rows)

def main()->int:
    req(CONFIG.is_file() and FREEZE.is_file() and VERIFY.is_file(),"missing RQ4 config/freeze/verifier")
    cfg=json.loads(CONFIG.read_text(encoding="utf-8")); freeze=json.loads(FREEZE.read_text(encoding="utf-8")); verify=json.loads(VERIFY.read_text(encoding="utf-8"))
    req(verify.get("status")=="PASS" and verify.get("results_unblinded") is False,"RQ4 pre-unblinding integrity has not passed")
    req(verify.get("verified_inner_fits")==200,"RQ4 verifier fit count mismatch")
    for key,rec in freeze["files"].items():
        resolve_frozen_record_path(ROOT, rec["path"], rec["sha256"])
    for rec in freeze["source_primary_outer_artifacts"]:
        pred=ROOT/rec["prediction"]["path"]; side=ROOT/rec["sidecar"]["path"]
        req(pred.is_file() and sha256_file(pred)==rec["prediction"]["sha256"],"frozen primary prediction changed after verification")
        req(side.is_file() and sha256_file(side)==rec["sidecar"]["sha256"],"frozen primary sidecar changed after verification")
    if FINAL.exists() and any(FINAL.iterdir()): raise RuntimeError("RQ4 final_statistics already exists; refusing overwrite")
    FINAL.mkdir(parents=True,exist_ok=True)

    alphas=[float(x) for x in cfg["operating_points"]["nominal_fpr_alpha"]]; delta=float(cfg["operating_points"]["delta"])
    combined=[]; param_rows=[]
    for objective in cfg["objectives"]:
        method=cfg["objective_labels"][objective]
        for seed in cfg["seeds"]:
            for outer in cfg["outer_folds"]:
                parts=[]
                for inner in cfg["inner_folds"]:
                    p=OUT/"inner_oof"/objective/f"seed_{seed}"/f"outer_{outer}_inner_{inner}_predictions.csv"
                    side=OUT/"inner_oof"/objective/f"seed_{seed}"/f"outer_{outer}_inner_{inner}_run.json"
                    req(p.is_file() and side.is_file(),f"missing verified inner artifact: {objective}/{seed}/{outer}/{inner}")
                    run=json.loads(side.read_text(encoding="utf-8"))
                    req(run.get("prediction_sha256")==sha256_file(p),f"inner prediction changed after verification: {objective}/{seed}/{outer}/{inner}")
                    req(run.get("preexecution_freeze_sha256")==sha256_file(FREEZE),f"inner freeze binding changed: {objective}/{seed}/{outer}/{inner}")
                    parts.append(pd.read_csv(p))
                oof=pd.concat(parts,ignore_index=True).sort_values("text_id").reset_index(drop=True)
                req(len(oof)==2400 and oof.text_id.nunique()==2400,"invalid OOF calibration frame")
                y_train=label_array(oof); raw_train=oof.raw_decision_score.to_numpy(float)
                calibrator=fit_platt(raw_train,y_train); oof_prob=calibrated_score(calibrator,raw_train)

                thresholds={}; ks={}
                for a in alphas:
                    t,k=threshold_np(oof_prob,y_train,a,delta); thresholds[a]=t; ks[a]=k

                outer_path=ROOT/"results"/"primary"/objective/f"seed_{seed}"/f"outer_{outer}_predictions.csv"
                outer_df=pd.read_csv(outer_path).sort_values("text_id").reset_index(drop=True)
                req(len(outer_df)==600 and outer_df.text_id.nunique()==600,"invalid frozen primary outer frame")
                prob=calibrated_score(calibrator,outer_df.raw_decision_score.to_numpy(float))
                result=outer_df[["text_id","pair_id","label","domain","generator","outer_fold","seed","objective","condition","raw_decision_score"]].copy()
                result["method"]=method; result["calibrated_study_ai_score"]=prob
                for a in alphas:
                    tag=f"{a:.2f}"; result[f"threshold_alpha_{tag}"]=thresholds[a]; result[f"binary_prediction_alpha_{tag}"]=(prob>thresholds[a]).astype(int)
                combined.append(result)
                row={"objective":objective,"method":method,"seed":seed,"outer_fold":outer,"calibration":"Platt logistic","calibration_fit_rows":2400,"calibration_fit_humans":1200,"calibration_fit_ai":1200,"coefficient":float(calibrator.coef_[0,0]),"intercept":float(calibrator.intercept_[0]),"outer_test_rows_used_for_calibration":0,"outer_test_rows_used_for_threshold":0}
                for a in alphas:
                    tag=f"{a:.2f}"; row[f"threshold_alpha_{tag}"]=thresholds[a]; row[f"k_exceed_alpha_{tag}"]=ks[a]
                param_rows.append(row)

    predictions=pd.concat(combined,ignore_index=True)
    atomic_csv(predictions,FINAL/"calibrated_outer_predictions.csv")
    params=pd.DataFrame(param_rows); atomic_csv(params,FINAL/"calibration_parameters_and_thresholds.csv")

    per_seed=[]; reliability=[]; frames={}
    for objective in cfg["objectives"]:
        method=cfg["objective_labels"][objective]
        for seed in cfg["seeds"]:
            frame=predictions[(predictions.objective==objective)&(predictions.seed.astype(int)==int(seed))].copy().sort_values(["pair_id","label"]).reset_index(drop=True)
            req(len(frame)==3000 and frame.pair_id.nunique()==1500,"invalid pooled seed frame")
            frames[(objective,int(seed))]=frame
            metrics=metric_bundle(frame,alphas)
            for metric,value in metrics.items(): per_seed.append({"objective":objective,"method":method,"seed":int(seed),"metric":metric,"value":value})
            reliability.append(reliability_bins(frame,objective,int(seed),10))
    per_seed_df=pd.DataFrame(per_seed); atomic_csv(per_seed_df,FINAL/"per_seed_calibration_operating_metrics.csv")
    summary=per_seed_df.groupby(["objective","method","metric"],as_index=False).agg(mean=("value","mean"),sd=("value","std"),n_seeds=("value","count")); atomic_csv(summary,FINAL/"method_calibration_operating_summary.csv")

    ce=per_seed_df[per_seed_df.objective.eq("ce")][["seed","metric","value"]].rename(columns={"value":"ce_value"})
    su=per_seed_df[per_seed_df.objective.eq("supcon")][["seed","metric","value"]].rename(columns={"value":"supcon_value"})
    deltas=ce.merge(su,on=["seed","metric"],validate="one_to_one"); deltas["delta_supcon_minus_ce"]=deltas.supcon_value-deltas.ce_value; atomic_csv(deltas,FINAL/"paired_seed_deltas.csv")

    rel=pd.concat(reliability,ignore_index=True); atomic_csv(rel,FINAL/"reliability_bins_per_seed.csv")
    rel_summary=rel.groupby(["objective","method","bin","lower","upper"],as_index=False).agg(mean_n=("n","mean"),mean_predicted=("mean_predicted","mean"),sd_mean_predicted=("mean_predicted","std"),mean_observed_ai_rate=("observed_ai_rate","mean"),sd_observed_ai_rate=("observed_ai_rate","std")); atomic_csv(rel_summary,FINAL/"reliability_bins_method_summary.csv")

    metrics=sorted(per_seed_df.metric.unique().tolist()); observed=deltas.groupby("metric")["delta_supcon_minus_ce"].mean().to_dict()
    pair_ids=sorted(frames[("ce",int(cfg["seeds"][0]))].pair_id.astype(str).unique().tolist()); n_pairs=len(pair_ids); req(n_pairs==1500,"bootstrap pair count != 1500")
    prepared={}
    for key,frame in frames.items():
        f=frame.copy(); f["pair_id"]=f.pair_id.astype(str); f=f.set_index(["pair_id","label"]).sort_index()
        # Fixed 2 rows/pair; labels retained for robust metric reconstruction.
        order=[]
        for pid in pair_ids:
            sub=f.loc[pid].reset_index(); req(len(sub)==2,"pair does not contain two rows"); order.append(sub)
        g=pd.concat(order,ignore_index=True); prepared[key]=g
    rng=np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"])); draws=[]
    # row indices for pair i are 2*i and 2*i+1 after preparation.
    for draw in range(int(cfg["statistics"]["bootstrap_replicates"])):
        pick=rng.integers(0,n_pairs,size=n_pairs); row_idx=np.empty(n_pairs*2,dtype=int); row_idx[0::2]=2*pick; row_idx[1::2]=2*pick+1
        seed_metric={}
        for objective in cfg["objectives"]:
            for seed in cfg["seeds"]:
                sample=prepared[(objective,int(seed))].iloc[row_idx].copy(); seed_metric[(objective,int(seed))]=metric_bundle(sample,alphas)
        for metric in metrics:
            d=np.mean([seed_metric[("supcon",int(seed))][metric]-seed_metric[("ce",int(seed))][metric] for seed in cfg["seeds"]])
            draws.append({"draw":draw,"metric":metric,"delta_supcon_minus_ce":float(d)})
    draws_df=pd.DataFrame(draws); atomic_csv(draws_df,FINAL/"paired_pair_bootstrap_draws.csv")
    boot=[]
    for metric in metrics:
        vals=draws_df.loc[draws_df.metric.eq(metric),"delta_supcon_minus_ce"].to_numpy(float)
        lo,hi=np.quantile(vals,[0.025,0.975]); boot.append({"metric":metric,"observed_paired_mean_delta_supcon_minus_ce":float(observed[metric]),"bootstrap_mean":float(vals.mean()),"ci95_lower":float(lo),"ci95_upper":float(hi),"bootstrap_replicates":len(vals),"bootstrap_unit":"pair_id","bootstrap_seed":int(cfg["statistics"]["bootstrap_seed"]),"seed_resampling":False})
    boot_df=pd.DataFrame(boot); atomic_csv(boot_df,FINAL/"paired_pair_bootstrap_summary.csv")

    outputs={}
    for p in sorted(FINAL.glob("*.csv")):
        outputs[p.name]={"sha256":sha256_file(p),"bytes":p.stat().st_size}
    manifest={"schema_version":"combined-v2-calibration-final-statistics-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"COMPLETE","results_are_now_unblinded":True,"verified_inner_fits":int(verify["verified_inner_fits"]),"source_primary_outer_inputs":int(verify["primary_outer_inputs_verified"]),"freeze_sha256":sha256_file(FREEZE),"config_sha256":sha256_file(CONFIG),"runner_sha256":freeze["files"]["rq4_runner"]["sha256"],"statistics_sha256":freeze["files"]["statistics_module"]["sha256"],"preunblinding_integrity_sha256":sha256_file(VERIFY),"calibration":"Platt logistic on outer-training inner-OOF only","ece":"10 equal-width bins","operating_point_rule":"training-side NP-inspired order-statistic threshold on calibrated OOF HUMAN probabilities","nominal_fpr_alpha":alphas,"delta":delta,"bootstrap":{"unit":"pair_id","replicates":int(cfg["statistics"]["bootstrap_replicates"]),"seed":int(cfg["statistics"]["bootstrap_seed"]),"seeds_resampled":False,"point_estimate":"observed paired mean; bootstrap quantiles for CI"},"outputs":outputs}
    (FINAL/"final_statistics_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"COMPLETE","results_are_now_unblinded":True,"verified_inner_fits":int(verify["verified_inner_fits"]),"final_statistics_dir":str(FINAL),"manifest_sha256":sha256_file(FINAL/"final_statistics_manifest.json")},ensure_ascii=False,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
