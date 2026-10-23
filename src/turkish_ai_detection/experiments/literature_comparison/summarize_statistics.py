# -*- coding: utf-8 -*-
"""One-time LIT2026 unblinding and same-data comparison to primary CE/SupCon."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)

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
from turkish_ai_detection.evaluation.metrics import _pauc  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import expected_run_hashes, validate_current_migrated_reproduction  # noqa: E402

CONFIG=ROOT/"configs"/"literature_comparison.json"
RUNNER=ROOT/"src"/"turkish_ai_detection/experiments/literature_comparison/run_fit.py"
STATS=Path(__file__)
FREEZE=ROOT/"provenance"/"experiments"/"literature_comparison"/"PREEXECUTION_FREEZE.json"
OUT=ROOT/"results"/"literature_comparison"
RESULTS=OUT/"final_statistics"

PRIMARY_OUT=ROOT/"results"/"primary"
PRIMARY_MANIFEST=PRIMARY_OUT/"final_statistics"/"final_statistics_manifest.json"
SEEDS=[42,314,2026]
OUTERS=[0,1,2,3,4]
METRICS=["roc_auc","average_precision","standardized_pauc_003"]

def metric_triplet(frame):
    y=frame.label.astype(str).eq("AI").astype(int).to_numpy()
    s=frame.raw_decision_score.to_numpy(float)
    return {
        "roc_auc":float(roc_auc_score(y,s)),
        "average_precision":float(average_precision_score(y,s)),
        "standardized_pauc_003":float(_pauc(y,s)),
    }

def full_metrics(frame):
    y=frame.label.astype(str).eq("AI").astype(int).to_numpy()
    s=frame.raw_decision_score.to_numpy(float)
    if "predicted_label" in frame:
        p=frame.predicted_label.astype(int).to_numpy()
    else:
        p=(s>0).astype(int)
    x=metric_triplet(frame)
    x.update({
        "accuracy":float(accuracy_score(y,p)),
        "balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "precision_ai":float(precision_score(y,p,zero_division=0)),
        "recall_ai":float(recall_score(y,p,zero_division=0)),
        "f1_ai":float(f1_score(y,p,zero_division=0)),
        "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
    })
    return x

def verify_freeze():
    fr=json.loads(FREEZE.read_text(encoding="utf-8"))
    if fr.get("status")!="FROZEN_READY_TO_RUN": raise RuntimeError("invalid freeze")
    policy=validate_current_migrated_reproduction(ROOT,"literature_comparison",RUNNER)
    return fr,policy

def load_lit(fr,policy):
    out={}
    for seed in SEEDS:
        parts=[]
        for outer in OUTERS:
            base=OUT/f"seed_{seed}"
            pred=base/f"outer_{outer}_predictions.csv"
            side=base/f"outer_{outer}_run.json"
            if not pred.is_file() or not side.is_file(): raise RuntimeError(f"missing LIT {seed}/{outer}")
            run=json.loads(side.read_text(encoding="utf-8"))
            if run["prediction_sha256"]!=sha256_file(pred): raise RuntimeError(f"LIT hash {seed}/{outer}")
            expected_config,expected_runner=expected_run_hashes(run,fr,policy,config_key="config",runner_key="runner")
            if run["runner_sha256"]!=expected_runner or run["config_sha256"]!=expected_config: raise RuntimeError(f"LIT runner/config {seed}/{outer}")
            parts.append(pd.read_csv(pred))
        f=pd.concat(parts,ignore_index=True).sort_values("text_id").reset_index(drop=True)
        if len(f)!=3000 or f.text_id.nunique()!=3000 or f.pair_id.nunique()!=1500:
            raise RuntimeError(f"LIT pooled OOF mismatch seed={seed}")
        out[seed]=f
    return out

def load_primary():
    if sha256_file(PRIMARY_MANIFEST)!="ed30975ecb16f49715352f6b11ef1c29aa6e4df2e0d2a0bf55f93a69fcc2187a":
        raise RuntimeError("primary manifest lineage changed")
    out={}
    for obj in ("ce","supcon"):
        for seed in SEEDS:
            parts=[]
            for outer in OUTERS:
                base=PRIMARY_OUT/obj/f"seed_{seed}"
                pred=base/f"outer_{outer}_predictions.csv"
                side=base/f"outer_{outer}_run.json"
                if not pred.is_file() or not side.is_file(): raise RuntimeError(f"missing primary {obj}/{seed}/{outer}")
                run=json.loads(side.read_text(encoding="utf-8"))
                if run["prediction_sha256"]!=sha256_file(pred): raise RuntimeError(f"primary hash {obj}/{seed}/{outer}")
                parts.append(pd.read_csv(pred))
            f=pd.concat(parts,ignore_index=True).sort_values("text_id").reset_index(drop=True)
            if len(f)!=3000 or f.text_id.nunique()!=3000: raise RuntimeError(f"primary pooled mismatch {obj}/{seed}")
            out[(obj,seed)]=f
    return out

def alignment(lit,primary):
    keys=["text_id","pair_id","label","domain","generator"]
    for seed in SEEDS:
        ref=lit[seed][keys]
        for obj in ("ce","supcon"):
            if not primary[(obj,seed)][keys].equals(ref):
                raise RuntimeError(f"same-data alignment failed: {obj}/{seed}")

def observed(lit,primary):
    rows=[]
    for seed in SEEDS:
        for method,frame in [
            ("LIT2026_REIMPL",lit[seed]),
            ("CE_ONLY_MATCHED",primary[("ce",seed)]),
            ("SUPCON",primary[("supcon",seed)]),
        ]:
            m=full_metrics(frame)
            rows.append({"seed":seed,"method":method,**m})
    per=pd.DataFrame(rows)
    summ=[]
    for method,g in per.groupby("method",sort=True):
        for metric in ["roc_auc","average_precision","standardized_pauc_003","accuracy","balanced_accuracy","precision_ai","recall_ai","f1_ai","macro_f1"]:
            vals=g[metric].to_numpy(float)
            summ.append({
                "method":method,"metric":metric,
                "mean_across_3_seeds":float(vals.mean()),
                "sd_across_3_seeds":float(vals.std(ddof=1)),
            })
    deltas=[]
    for seed in SEEDS:
        l=per[(per.seed==seed)&(per.method=="LIT2026_REIMPL")].iloc[0]
        c=per[(per.seed==seed)&(per.method=="CE_ONLY_MATCHED")].iloc[0]
        s=per[(per.seed==seed)&(per.method=="SUPCON")].iloc[0]
        row={"seed":seed}
        for m in METRICS:
            row[f"lit_minus_ce_{m}"]=l[m]-c[m]
            row[f"lit_minus_supcon_{m}"]=l[m]-s[m]
        deltas.append(row)
    return per,pd.DataFrame(summ),pd.DataFrame(deltas)

def metric_from(y,s):
    return {
        "roc_auc":float(roc_auc_score(y,s)),
        "average_precision":float(average_precision_score(y,s)),
        "standardized_pauc_003":float(_pauc(y,s)),
    }

def bootstrap(cfg,lit,primary):
    B=int(cfg["statistics"]["bootstrap_replicates"])
    rng=np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))
    ref=lit[SEEDS[0]]
    pairs=np.asarray(sorted(ref.pair_id.unique()))
    idx_map={p:np.flatnonzero(ref.pair_id.to_numpy()==p) for p in pairs}
    if len(pairs)!=1500 or any(len(v)!=2 for v in idx_map.values()): raise RuntimeError("invalid pair structure")
    yfull=ref.label.astype(str).eq("AI").astype(int).to_numpy()
    scores={}
    for seed in SEEDS:
        scores[("lit",seed)]=lit[seed].raw_decision_score.to_numpy(float)
        scores[("ce",seed)]=primary[("ce",seed)].raw_decision_score.to_numpy(float)
        scores[("supcon",seed)]=primary[("supcon",seed)].raw_decision_score.to_numpy(float)
    rows=[]
    for b in range(1,B+1):
        sampled=rng.choice(pairs,size=len(pairs),replace=True)
        idx=np.concatenate([idx_map[p] for p in sampled])
        y=yfull[idx]
        dce={m:[] for m in METRICS}
        dsc={m:[] for m in METRICS}
        for seed in SEEDS:
            lm=metric_from(y,scores[("lit",seed)][idx])
            cm=metric_from(y,scores[("ce",seed)][idx])
            sm=metric_from(y,scores[("supcon",seed)][idx])
            for m in METRICS:
                dce[m].append(lm[m]-cm[m])
                dsc[m].append(lm[m]-sm[m])
        row={"bootstrap_id":b}
        for m in METRICS:
            row[f"lit_minus_ce_{m}"]=float(np.mean(dce[m]))
            row[f"lit_minus_supcon_{m}"]=float(np.mean(dsc[m]))
        rows.append(row)
    draws=pd.DataFrame(rows)
    summary=[]
    for m in METRICS:
        for q in ("lit_minus_ce","lit_minus_supcon"):
            vals=draws[f"{q}_{m}"].to_numpy(float)
            summary.append({
                "quantity":q,"metric":m,"bootstrap_unit":"pair_id",
                "replicates":B,"bootstrap_seed":int(cfg["statistics"]["bootstrap_seed"]),
                "bootstrap_mean":float(vals.mean()),
                "ci95_lower":float(np.quantile(vals,.025)),
                "ci95_upper":float(np.quantile(vals,.975)),
                "inference_scope":cfg["statistics"]["inference_scope"],
            })
    return draws,pd.DataFrame(summary)

def main():
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))
    fr,policy=verify_freeze()
    lit=load_lit(fr,policy)
    primary=load_primary()
    alignment(lit,primary)
    per,summary,deltas=observed(lit,primary)
    draws,boot_summary=bootstrap(cfg,lit,primary)

    RESULTS.mkdir(parents=True,exist_ok=True)
    context=pd.DataFrame([
        {"source":"Renklier & Sarıtaş 2026","dataset":"paper's own 11,990-text corpus","metric":"test_roc_auc","value":0.9931,"directly_comparable":False},
        {"source":"Renklier & Sarıtaş 2026","dataset":"paper's own group-aware 5-fold CV","metric":"mean_roc_auc","value":0.9966,"directly_comparable":False},
    ])
    outputs={
        "per_seed_same_data_metrics.csv":per,
        "method_same_data_summary.csv":summary,
        "paired_seed_deltas.csv":deltas,
        "paired_pair_bootstrap_draws.csv":draws,
        "paired_pair_bootstrap_summary.csv":boot_summary,
        "published_context_only.csv":context,
    }
    for name,frame in outputs.items():
        frame.to_csv(RESULTS/name,index=False)

    manifest={
        "schema_version":"combined-v2-lit2026-final-statistics-v1",
        "status":"COMPLETE","unblinded":True,"fits_verified":15,
        "seeds":SEEDS,"outer_folds":OUTERS,
        "literature_doi":cfg["authoritative_literature_source"]["doi"],
        "published_score_directly_comparable":False,
        "statistics_config":cfg["statistics"],
        "config_sha256":sha256_file(CONFIG),
        "runner_sha256":sha256_file(RUNNER),
        "statistics_sha256":sha256_file(STATS),
        "freeze_sha256":sha256_file(FREEZE),
        "primary_reference_manifest_sha256":sha256_file(PRIMARY_MANIFEST),
        "outputs":{name:sha256_file(RESULTS/name) for name in outputs},
    }
    write_json(RESULTS/"final_statistics_manifest.json",manifest)
    print(json.dumps({
        "status":"COMPLETE","fits_verified":15,
        "results_directory":str(RESULTS),
        "results_are_now_unblinded":True,
    },ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
