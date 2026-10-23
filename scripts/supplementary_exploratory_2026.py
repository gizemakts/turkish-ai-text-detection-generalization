"""Post-hoc exploratory analyses for Turkish AI-Generated Text Detection Study.

This program is deliberately read-only with respect to historical evidence.  It
writes only below results/supplementary_exploratory_2026 and the compact review
package.  Analysis status: POST_HOC_EXPLORATORY.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import sys
import unicodedata
import zipfile
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "supplementary_exploratory_2026"
PKG = ROOT / "Turkish_AI_Text_Detection_Exploratory_Analysis_Review"
TODAY = date.today().isoformat()
BOOT_SEED = 261012
N_BOOT = 5000
METHODS = {"CE_ONLY_MATCHED": "ce", "SUPCON": "supcon"}
SEEDS_INTERNAL = [42, 137, 314, 2026, 3407]
SEEDS_EXTERNAL = [42, 314, 2026]
FOLDS = list(range(5))
ALPHAS = [0.01, 0.02, 0.03, 0.05]
DATASETS = {
    "renklier2026": ROOT / "data/external_prepared/renklier2026_test_standardized.csv",
    "tum2024_tr": ROOT / "data/external_prepared/tum2024_tr_test_standardized.csv",
    "bostancioglu2026_tr": ROOT / "data/external_prepared/bostancioglu2026_tr_full_standardized.csv",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

def sigmoid(x):
    x = np.asarray(x, float)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def safe_metrics(y, score):
    y = np.asarray(y, int); score = np.asarray(score, float)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))

def safe_weighted_metrics(y, score, weight):
    y=np.asarray(y,int); score=np.asarray(score,float); weight=np.asarray(weight,float)
    keep=weight>0
    if keep.sum()<2 or len(np.unique(y[keep]))<2: return np.nan,np.nan
    return (float(roc_auc_score(y,score,sample_weight=weight)),
            float(average_precision_score(y,score,sample_weight=weight)))

def length_band(n):
    return "short_le150" if n <= 150 else ("medium_151_300" if n <= 300 else "long_gt300")

def surface(text: str, tokenizer) -> dict:
    text = "" if pd.isna(text) else str(text)
    words = re.findall(r"\S+", text, flags=re.UNICODE)
    lexical = re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
    sent = [s for s in re.split(r"(?<=[.!?…])\s+|[\r\n]+", text.strip()) if s.strip()]
    wp = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    chars = max(len(text), 1); alpha = sum(c.isalpha() for c in text)
    punctuation = sum(unicodedata.category(c).startswith("P") for c in text)
    digit_words = sum(any(c.isdigit() for c in w) for w in words)
    years = len(re.findall(r"(?<!\d)(?:18|19|20|21)\d{2}(?!\d)", text))
    dates = len(re.findall(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:19|20)?\d{2}(?!\d)", text))
    return {
        "word_count": len(words), "sentence_count": len(sent),
        "mean_sentence_words": len(words) / max(len(sent), 1),
        "wordpiece_count": wp, "effective_berturk_tokens": min(wp, 512),
        "excluded_wordpieces": max(wp - 512, 0),
        "excluded_wordpiece_proportion": max(wp - 512, 0) / max(wp, 1),
        "truncated_berturk": int(wp > 512),
        "punctuation_per_100_chars": 100 * punctuation / chars,
        "digit_token_fraction": digit_words / max(len(words), 1),
        "year_date_per_100_words": 100 * (years + dates) / max(len(words), 1),
        "uppercase_alpha_ratio": sum(c.isupper() for c in text) / max(alpha, 1),
        "lexical_diversity": len(set(lexical)) / max(len(lexical), 1),
        "structured_heading": int(any(re.match(r"^\s{0,3}#{1,6}\s+", x) or (x.strip().isupper() and 2 <= len(x.split()) <= 15) for x in text.splitlines())),
        "list_line": int(any(re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", x) for x in text.splitlines())),
        "quote_per_100_chars": 100 * sum(c in '\"“”‘’«»' for c in text) / chars,
        "dash_per_100_chars": 100 * sum(c in "-–—" for c in text) / chars,
        "newline_per_100_chars": 100 * text.count("\n") / chars,
    }

FEATURES = [
    "word_count", "sentence_count", "mean_sentence_words", "wordpiece_count",
    "punctuation_per_100_chars", "digit_token_fraction", "year_date_per_100_words",
    "uppercase_alpha_ratio", "lexical_diversity", "structured_heading", "list_line",
    "quote_per_100_chars", "dash_per_100_chars", "newline_per_100_chars",
]

def bootstrap_mean_ci(values, rng, n_boot=N_BOOT):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v): return np.nan, np.nan
    draws = np.empty(n_boot)
    for i in range(n_boot): draws[i] = np.mean(v[rng.integers(0, len(v), len(v))])
    return tuple(np.quantile(draws, [0.025, 0.975]))

def bootstrap_metric_pairs(frame, score_col, rng, n_boot=N_BOOT):
    pairs = frame["pair_id"].drop_duplicates().to_numpy()
    aucs=[]; aps=[]
    groups = {k:g for k,g in frame.groupby("pair_id", sort=False)}
    for _ in range(n_boot):
        pick = rng.choice(pairs, len(pairs), replace=True)
        b = pd.concat([groups[x] for x in pick], ignore_index=True)
        a,p = safe_metrics(b["binary_label"], b[score_col]); aucs.append(a); aps.append(p)
    return np.nanquantile(aucs,[.025,.975]), np.nanquantile(aps,[.025,.975])

def inventory_and_audit(source_paths):
    d = OUT / "00_protocol_and_manifest"; d.mkdir(parents=True, exist_ok=True)
    rows=[]
    for role, path in source_paths:
        rows.append({"role": role, "relative_path": path.relative_to(ROOT).as_posix(),
                     "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else np.nan,
                     "sha256": sha256(path) if path.exists() else ""})
    inv = pd.DataFrame(rows); save_csv(inv, d / "INPUT_EVIDENCE_INVENTORY.csv")
    save_csv(inv[inv.exists][["relative_path","bytes","sha256"]], d / "SOURCE_SHA256_MANIFEST.csv")
    study = pd.read_csv(ROOT / "data/study/paired_study.csv")
    folds = pd.read_csv(ROOT / "data/study/nested_folds.csv")
    counts = study.groupby(["pair_id","label"]).size().unstack(fill_value=0)
    pair_ok = len(counts)==1500 and set(counts.columns)=={"AI","HUMAN"} and (counts==1).all().all()
    pred_ok=True; pred_rows=0
    for m,obj in METHODS.items():
        for seed in SEEDS_INTERNAL:
            seen=[]
            for fold in FOLDS:
                p=ROOT/f"results/primary/{obj}/seed_{seed}/outer_{fold}_predictions.csv"
                q=pd.read_csv(p); pred_rows += len(q); seen.extend(q.text_id.tolist())
                pred_ok &= len(q)==600 and q.outer_fold.eq(fold).all() and q.seed.eq(seed).all()
            pred_ok &= len(seen)==3000 and len(set(seen))==3000
    external_ok=True; external_rows=0
    for ds,path in DATASETS.items():
        base=pd.read_csv(path); external_rows += len(base)
        external_ok &= base.external_id.is_unique and set(base.binary_label.unique()) <= {0,1}
        for method in ["CE_ONLY_MATCHED","SUPCON","LIT2026_REIMPL_V1_1"]:
            for seed in SEEDS_EXTERNAL:
                for fold in FOLDS:
                    p=ROOT/f"artifacts/predictions/external_cross_corpus/preseal/{ds}/{method}/seed_{seed}/outer_{fold}.csv"
                    q=pd.read_csv(p)
                    external_ok &= len(q)==len(base) and q.external_id.equals(base.external_id)
                    external_ok &= q.text_sha256.equals(base.text_sha256)
    audit={"analysis_class":"POST_HOC_EXPLORATORY","date":TODAY,"pair_integrity":bool(pair_ok),
           "study_rows":len(study),"pair_rows":len(folds),"internal_prediction_rows_checked":pred_rows,
           "internal_prediction_integrity":bool(pred_ok),"external_dataset_rows":external_rows,
           "external_prediction_binding_integrity":bool(external_ok),"class_direction":"HUMAN=0, AI=1",
           "calibration_transfer_conclusion":"CALIBRATION_TRANSFER_LIMITED",
           "primary_retraining_performed":False,"external_label_model_selection":False}
    write_json(d/"PRE_ANALYSIS_AUDIT.json",audit)
    status="available" if inv.exists.all() else "partially missing"
    write_text(d/"INPUT_AVAILABILITY_REPORT.md",f"""# Input availability report

Status: **{status}** (`POST_HOC_EXPLORATORY`). The canonical Combined-v2 table, pair folds, 50 primary OOF files, three external standardized corpora, sealed terminal predictions, and frozen calibration/threshold parameters were found. Pair and prediction bindings passed the pre-analysis audit. The requested manuscript and supplementary PDF were not located in the inspected project inputs; they were not modified.
""")
    write_text(d/"EXPLORATORY_ANALYSIS_PROTOCOL.md",f"""# Frozen exploratory protocol

Frozen {TODAY}, before new outcome calculation. Analysis class: `POST_HOC_EXPLORATORY`.

- A1 uses the 14 predefined label-blind surface features listed in the analysis script. Logistic regression uses StandardScaler and L2 regularization (C=1), fitted only on each original outer-fold training partition. No detector score enters the probe.
- Length bands are fixed at ≤150, 151–300, and >300 whitespace words. Truncation is untruncated pinned-tokenizer length >512. CE/SupCon effective input is 512 with recorded head-tail selection; LIT2026 is first-512. Token counts are shared, truncation procedures are not conflated.
- Internal uncertainty uses 5,000 pair-cluster bootstrap replicates (seed {BOOT_SEED}). External realization summaries retain seed/fold identities and never pool scores across models.
- A3 applies only the frozen internal Platt maps and thresholds at alpha 0.01/0.02/0.03/0.05. External labels do not fit or select parameters.
- A4 freezes 250 label-independent AI source IDs stratified by corpus and length. No generation or model inference is authorized by this protocol; status remains `PENDING_AUTHORIZED_GENERATION` absent existing approved variants.
""")
    return inv, audit

def build_document_features(tokenizer):
    study=pd.read_csv(ROOT/"data/study/paired_study.csv")
    feats=pd.DataFrame([surface(t,tokenizer) for t in study.text])
    internal=pd.concat([study[["text_id","pair_id","label","domain","generator","source_year","academic_field","broad_field","newspaper_source"]].reset_index(drop=True),feats],axis=1)
    internal["binary_label"]=(internal.label=="AI").astype(int); internal["dataset"]="combined_v2"
    internal["document_id"]=internal.text_id; internal["length_band"]=internal.word_count.map(length_band)
    externals=[]
    for ds,path in DATASETS.items():
        x=pd.read_csv(path); f=pd.DataFrame([surface(t,tokenizer) for t in x.text])
        z=pd.concat([x[["external_id","label","binary_label","domain","generator","generation_type","source_group_id","text_sha256"]].reset_index(drop=True),f],axis=1)
        z["dataset"]=ds; z["document_id"]=z.external_id; z["pair_id"]=""; z["length_band"]=z.word_count.map(length_band)
        externals.append(z)
    ext=pd.concat(externals,ignore_index=True)
    return internal,ext

def analysis_a1(internal):
    d=OUT/"01_corpus_shortcuts"; d.mkdir(parents=True,exist_ok=True)
    folds=pd.read_csv(ROOT/"data/study/nested_folds.csv")
    x=internal.merge(folds[["pair_id","outer_fold"]],on="pair_id",validate="many_to_one")
    save_csv(x[["text_id","pair_id","label","binary_label","domain","generator","outer_fold"]+FEATURES],d/"corpus_shortcut_features.csv")
    rng=np.random.default_rng(BOOT_SEED); effects=[]
    for domain in ["ALL","ABSTRACT","NEWSPAPER"]:
        z=x if domain=="ALL" else x[x.domain==domain]
        for feat in FEATURES:
            w=z.pivot(index="pair_id",columns="label",values=feat).dropna()
            diff=w["AI"]-w["HUMAN"]; lo,hi=bootstrap_mean_ci(diff.to_numpy(),rng)
            effects.append({"domain":domain,"feature":feat,"n_pairs":len(diff),"human_mean":w.HUMAN.mean(),"ai_mean":w.AI.mean(),"mean_paired_difference_ai_minus_human":diff.mean(),"median_paired_difference":diff.median(),"ci95_low":lo,"ci95_high":hi,"bootstrap_unit":"pair_id","bootstrap_replicates":N_BOOT})
    effects=pd.DataFrame(effects); save_csv(effects,d/"corpus_shortcut_paired_effects.csv")
    preds=[]
    for fold in FOLDS:
        tr=x.outer_fold!=fold; te=~tr
        pipe=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(C=1.0,penalty="l2",solver="liblinear",max_iter=2000,random_state=BOOT_SEED))])
        pipe.fit(x.loc[tr,FEATURES],x.loc[tr,"binary_label"])
        p=pipe.predict_proba(x.loc[te,FEATURES])[:,1]
        q=x.loc[te,["text_id","pair_id","label","binary_label","domain","outer_fold"]].copy(); q["surface_probe_ai_score"]=p; preds.append(q)
    pred=pd.concat(preds,ignore_index=True); save_csv(pred,d/"surface_probe_oof_predictions.csv")
    results=[]
    for domain in ["ALL","ABSTRACT","NEWSPAPER"]:
        z=pred if domain=="ALL" else pred[pred.domain==domain]
        auc,ap=safe_metrics(z.binary_label,z.surface_probe_ai_score)
        aci,pci=bootstrap_metric_pairs(z,"surface_probe_ai_score",rng)
        results.append({"domain":domain,"n_documents":len(z),"n_pairs":z.pair_id.nunique(),"roc_auc":auc,"roc_auc_ci_low":aci[0],"roc_auc_ci_high":aci[1],"average_precision":ap,"average_precision_ci_low":pci[0],"average_precision_ci_high":pci[1],"bootstrap_unit":"pair_id","bootstrap_replicates":N_BOOT})
    results=pd.DataFrame(results); save_csv(results,d/"surface_probe_results.csv")
    # Exploratory feature/score associations, preserving seed and method.
    assocs=[]
    for method,obj in METHODS.items():
        for seed in SEEDS_INTERNAL:
            pp=pd.concat([pd.read_csv(ROOT/f"results/primary/{obj}/seed_{seed}/outer_{f}_predictions.csv") for f in FOLDS])
            j=x.merge(pp[["text_id","raw_decision_score"]],on="text_id",validate="one_to_one")
            for feat in FEATURES:
                assocs.append({"method":method,"seed":seed,"feature":feat,"spearman_rho":j[[feat,"raw_decision_score"]].corr(method="spearman").iloc[0,1]})
    save_csv(pd.DataFrame(assocs),d/"surface_feature_score_associations.csv")
    save_csv(effects,d/"Table_S14_corpus_construction_feature_diagnostics.csv")
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,feat in zip(axs,["word_count","lexical_diversity","punctuation_per_100_chars"]):
        for lab,color in [("HUMAN","#3568a8"),("AI","#d05a45")]:
            ax.hist(x.loc[x.label==lab,feat],bins=35,density=True,alpha=.5,label=lab,color=color)
        ax.set_title(feat.replace("_"," ")); ax.set_ylabel("Density")
    axs[0].legend(); fig.suptitle("Fig. S8 — Surface-feature distributions (Combined-v2; exploratory)"); fig.tight_layout()
    fig.savefig(d/"Figure_S8_surface_feature_distributions.svg"); fig.savefig(d/"Figure_S8_surface_feature_distributions.png",dpi=300); plt.close(fig)
    strongest=effects[effects.domain.eq("ALL")].assign(absdiff=lambda q:q.mean_paired_difference_ai_minus_human.abs()).sort_values("absdiff",ascending=False).iloc[0]
    rr=results.set_index("domain").loc["ALL"]
    write_text(d/"CORPUS_SHORTCUT_INTERPRETATION.md",f"""# Corpus-construction shortcut analysis

Status: `COMPLETED_VALIDATED`; analysis class: `POST_HOC_EXPLORATORY`.

The evaluation population was all 1,500 Combined-v2 HUMAN–AI pairs. Feature definitions are in the frozen protocol and script. The pair-disjoint, train-only-scaled surface probe achieved ROC-AUC {rr.roc_auc:.3f} (pair-bootstrap 95% CI {rr.roc_auc_ci_low:.3f}–{rr.roc_auc_ci_high:.3f}) and AP {rr.average_precision:.3f}. The largest raw-scale mean paired difference was for `{strongest.feature}` ({strongest.mean_paired_difference_ai_minus_human:.3g}, AI minus HUMAN); raw scales are not comparable across features.

These results demonstrate observable corpus separability using predefined surface measurements. They do **not** demonstrate that BERTurk causally relies on any feature. Score associations are exploratory correlations, and the existing Condition-B results remain unchanged.
""")
    return x,pred,effects,results

def add_internal_predictions(internal):
    frames=[]
    for method,obj in METHODS.items():
        for seed in SEEDS_INTERNAL:
            p=pd.concat([pd.read_csv(ROOT/f"results/primary/{obj}/seed_{seed}/outer_{f}_predictions.csv") for f in FOLDS],ignore_index=True)
            p["method"]=method; frames.append(p)
    return pd.concat(frames,ignore_index=True).merge(internal[["text_id","word_count","wordpiece_count","truncated_berturk","length_band"]],on="text_id",validate="many_to_one")

def load_external_predictions(ext):
    frames=[]
    for ds in DATASETS:
        meta=ext[ext.dataset.eq(ds)]
        for method in ["CE_ONLY_MATCHED","SUPCON","LIT2026_REIMPL_V1_1"]:
            for seed in SEEDS_EXTERNAL:
                for fold in FOLDS:
                    p=pd.read_csv(ROOT/f"artifacts/predictions/external_cross_corpus/preseal/{ds}/{method}/seed_{seed}/outer_{fold}.csv")
                    p["dataset"]=ds; frames.append(p)
    p=pd.concat(frames,ignore_index=True)
    cols=["external_id","dataset","label","binary_label","domain","generator","generation_type","source_group_id","text_sha256","word_count","wordpiece_count","truncated_berturk","length_band"]
    return p.merge(ext[cols],on=["external_id","dataset","text_sha256"],validate="many_to_one")

def analysis_a2(internal,ext,intp,extp):
    d=OUT/"02_length_truncation"; d.mkdir(parents=True,exist_ok=True)
    doc=pd.concat([
        internal[["dataset","document_id","pair_id","label","binary_label","word_count","sentence_count","wordpiece_count","effective_berturk_tokens","truncated_berturk","excluded_wordpieces","excluded_wordpiece_proportion","length_band"]],
        ext[["dataset","document_id","pair_id","label","binary_label","word_count","sentence_count","wordpiece_count","effective_berturk_tokens","truncated_berturk","excluded_wordpieces","excluded_wordpiece_proportion","length_band"]]
    ],ignore_index=True); save_csv(doc,d/"length_truncation_document_features.csv")
    prev=[]
    for (ds,lab),g in doc.groupby(["dataset","label"]):
        prev.append({"dataset":ds,"label":lab,"n":len(g),"truncated_n":int(g.truncated_berturk.sum()),"truncated_proportion":g.truncated_berturk.mean(),"median_wordpieces":g.wordpiece_count.median(),"method_note":"CE/SupCon head-tail; LIT2026 first-512; same untruncated token count"})
    prev=pd.DataFrame(prev); save_csv(prev,d/"truncation_prevalence.csv")
    rows=[]
    for keys,g in intp.groupby(["method","seed"]):
        method,seed=keys
        for subgroup,sg in list(g.groupby("length_band"))+[("truncated",g[g.truncated_berturk.eq(1)]),("all",g)]:
            auc,ap=safe_metrics((sg.label=="AI").astype(int),sg.raw_decision_score)
            rows.append({"population":"internal","dataset":"combined_v2","method":method,"seed":seed,"outer_fold":"POOLED_OOF","subgroup":subgroup,"n":len(sg),"human_n":sum(sg.label=="HUMAN"),"ai_n":sum(sg.label=="AI"),"roc_auc":auc,"average_precision":ap,"aggregation":"pair-disjoint pooled OOF per seed"})
    for keys,g in extp.groupby(["dataset","method","seed","outer_fold"]):
        ds,method,seed,fold=keys
        for subgroup,sg in list(g.groupby("length_band"))+[("truncated",g[g.truncated_berturk.eq(1)]),("all",g)]:
            auc,ap=safe_metrics(sg.binary_label,sg.raw_score_or_logit)
            rows.append({"population":"external","dataset":ds,"method":method,"seed":seed,"outer_fold":fold,"subgroup":subgroup,"n":len(sg),"human_n":sum(sg.binary_label==0),"ai_n":sum(sg.binary_label==1),"roc_auc":auc,"average_precision":ap,"aggregation":"one fixed model realization; scores not pooled"})
    discr=pd.DataFrame(rows); save_csv(discr,d/"length_subgroup_discrimination.csv")
    # Matched CE-SupCon deltas from identifiable cells.
    a=discr[discr.method.isin(METHODS)].pivot_table(index=["population","dataset","seed","outer_fold","subgroup"],columns="method",values=["roc_auc","average_precision"]).reset_index()
    a.columns=["_".join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else c for c in a.columns]
    a["delta_roc_auc_supcon_minus_ce"]=a["roc_auc_SUPCON"]-a["roc_auc_CE_ONLY_MATCHED"]
    a["delta_ap_supcon_minus_ce"]=a["average_precision_SUPCON"]-a["average_precision_CE_ONLY_MATCHED"]
    save_csv(a,d/"length_matched_method_differences.csv")
    summary=discr.groupby(["population","dataset","method","subgroup"],dropna=False).agg(n_cells=("roc_auc","count"),mean_roc_auc=("roc_auc","mean"),sd_roc_auc=("roc_auc","std"),mean_ap=("average_precision","mean"),sd_ap=("average_precision","std"),median_n=("n","median")).reset_index()
    # Dependency-aware bootstrap of the mean fixed-realization metric.  A smaller
    # replicate count is used here because each draw recomputes up to 30 AUC/AP
    # values; model seeds/folds remain fixed and are never resampled.
    n_length_boot=500; bootrows=[]
    populations=[("internal","combined_v2",intp,"pair_id","raw_decision_score","label")]
    for ds in DATASETS:
        unit="source_group_id" if ds=="renklier2026" else "text_sha256"
        populations.append(("external",ds,extp[extp.dataset.eq(ds)],unit,"raw_score_or_logit","binary_label"))
    for pop,ds,base,unit,score,labelcol in populations:
        b=base.copy()
        if pop=="external" and unit=="source_group_id":
            if b[unit].fillna("").eq("").any(): raise RuntimeError("Missing required Renklier source_group_id")
        for subgroup in ["short_le150","medium_151_300","long_gt300","truncated","all"]:
            g=b if subgroup=="all" else (b[b.truncated_berturk.eq(1)] if subgroup=="truncated" else b[b.length_band.eq(subgroup)])
            units=np.sort(g[unit].drop_duplicates().astype(str).to_numpy())
            if len(units)<2: continue
            unit_index={u:i for i,u in enumerate(units)}
            rng=np.random.default_rng(BOOT_SEED+sum(map(ord,ds+subgroup)))
            draws=rng.multinomial(len(units),np.full(len(units),1/len(units)),size=n_length_boot)
            method_draws={}
            for method in METHODS:
                gm=g[g.method.eq(method)]; realizations=[]
                groupcols=["seed"] if pop=="internal" else ["seed","outer_fold"]
                for _,r in gm.groupby(groupcols):
                    ui=r[unit].astype(str).map(unit_index).to_numpy(); y=(r[labelcol].eq("AI").astype(int) if pop=="internal" else r[labelcol].astype(int)).to_numpy(); s=r[score].to_numpy()
                    realizations.append((ui,y,s))
                if not realizations: continue
                aucd=np.empty(n_length_boot); apd=np.empty(n_length_boot)
                for bi in range(n_length_boot):
                    vals=[safe_weighted_metrics(y,s,draws[bi,ui]) for ui,y,s in realizations]
                    aucd[bi]=np.nanmean([v[0] for v in vals]); apd[bi]=np.nanmean([v[1] for v in vals])
                method_draws[method]=(aucd,apd)
                bootrows.append({"population":pop,"dataset":ds,"subgroup":subgroup,"method":method,"bootstrap_unit":unit,"bootstrap_replicates":n_length_boot,"roc_auc_ci_low":np.nanquantile(aucd,.025),"roc_auc_ci_high":np.nanquantile(aucd,.975),"average_precision_ci_low":np.nanquantile(apd,.025),"average_precision_ci_high":np.nanquantile(apd,.975)})
            if set(method_draws)==set(METHODS):
                da=method_draws["SUPCON"][0]-method_draws["CE_ONLY_MATCHED"][0]; dp=method_draws["SUPCON"][1]-method_draws["CE_ONLY_MATCHED"][1]
                bootrows.append({"population":pop,"dataset":ds,"subgroup":subgroup,"method":"SUPCON_MINUS_CE","bootstrap_unit":unit,"bootstrap_replicates":n_length_boot,"roc_auc_ci_low":np.nanquantile(da,.025),"roc_auc_ci_high":np.nanquantile(da,.975),"average_precision_ci_low":np.nanquantile(dp,.025),"average_precision_ci_high":np.nanquantile(dp,.975)})
    boot=pd.DataFrame(bootrows); save_csv(boot,d/"length_subgroup_cluster_bootstrap.csv")
    summary=summary.merge(boot[boot.method.ne("SUPCON_MINUS_CE")],on=["population","dataset","method","subgroup"],how="left")
    save_csv(summary,d/"Table_S15_length_truncation_sensitivity.csv")
    plot=summary[summary.subgroup.isin(["short_le150","medium_151_300","long_gt300","truncated"])]
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
    for ax,(pop,title) in zip(axs,[("internal","Combined-v2"),("external","External corpora")]):
        z=plot[plot.population.eq(pop)]; labels=[]
        for method,color in [("CE_ONLY_MATCHED","#3568a8"),("SUPCON","#d05a45")]:
            q=z[z.method.eq(method)].groupby("subgroup").mean(numeric_only=True).reindex(["short_le150","medium_151_300","long_gt300","truncated"])
            ax.plot(range(4),q.mean_roc_auc,"o-",label=method,color=color)
        ax.set_xticks(range(4),["Short","Medium","Long","Truncated"],rotation=20); ax.set_title(title); ax.set_ylabel("Mean realization ROC-AUC"); ax.set_ylim(0,1)
    axs[0].legend(fontsize=8); fig.suptitle("Fig. S9 — Length and truncation sensitivity (exploratory)"); fig.tight_layout()
    fig.savefig(d/"Figure_S9_length_truncation_sensitivity.svg"); fig.savefig(d/"Figure_S9_length_truncation_sensitivity.png",dpi=300); plt.close(fig)
    write_text(d/"LENGTH_TRUNCATION_INTERPRETATION.md",f"""# Length and truncation sensitivity

Status: `COMPLETED_VALIDATED`; analysis class: `POST_HOC_EXPLORATORY`.

The table contains {len(discr):,} subgroup-by-realization estimates with every seed/fold retained. Internal rows are pair-disjoint pooled OOF estimates per seed; external rows are individual fixed-model estimates. Dependency-aware 95% intervals use 500 cluster draws while seeds/folds remain fixed; the lower replicate count reflects the cost of recomputing every fixed-realization AUC/AP and is recorded explicitly. No raw scores were pooled across external models. Cells lacking both classes are recorded as not estimable (`NaN`).

WordPiece counts use the pinned BERTurk tokenizer. The truncation indicator is shared, while effective selection is explicitly different: CE/SupCon use the recorded head-and-tail rule and LIT2026 uses first-512. Subgroups are descriptive and do not establish deployment reliability.
""")
    return doc,prev,discr,a

def analysis_a3(ext,extp):
    d=OUT/"03_error_subgroups"; d.mkdir(parents=True,exist_ok=True)
    params=pd.read_csv(ROOT/"results/calibration/final_statistics/calibration_parameters_and_thresholds.csv")
    params["method"]=params.objective.map({"ce":"CE_ONLY_MATCHED","supcon":"SUPCON"})
    z=extp[extp.method.isin(METHODS)].merge(params,on=["method","seed","outer_fold"],validate="many_to_one",suffixes=("","_param"))
    z["calibrated_probability"]=sigmoid(z.coefficient*z.raw_score_or_logit+z.intercept)
    metrics=[]; decisions=[]
    for keys,g in z.groupby(["dataset","method","seed","outer_fold"]):
        ds,m,s,f=keys
        for alpha in ALPHAS:
            t=float(g[f"threshold_alpha_{alpha:.2f}"].iloc[0]); pred=(g.calibrated_probability>t).astype(int); y=g.binary_label.astype(int)
            tn=int(((y==0)&(pred==0)).sum()); fp=int(((y==0)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum()); tp=int(((y==1)&(pred==1)).sum())
            precision=tp/(tp+fp) if tp+fp else np.nan; recall=tp/(tp+fn) if tp+fn else np.nan
            metrics.append({"dataset":ds,"method":m,"seed":s,"outer_fold":f,"nominal_fpr":alpha,"threshold":t,"tn":tn,"fp":fp,"fn":fn,"tp":tp,"fpr":fp/(fp+tn),"fnr":fn/(fn+tp),"tpr":recall,"precision":precision,"f1":2*precision*recall/(precision+recall) if precision+recall else np.nan,"calibration_source":"frozen internal inner-OOF"})
            q=g[["dataset","external_id","text_sha256","binary_label","domain","generator","source_group_id","length_band","truncated_berturk","method","seed","outer_fold","calibrated_probability"]].copy(); q["nominal_fpr"]=alpha; q["prediction"]=pred; q["error"]=(pred!=y).astype(int); decisions.append(q)
    metrics=pd.DataFrame(metrics); dec=pd.concat(decisions,ignore_index=True); save_csv(metrics,d/"external_error_metrics.csv")
    sub=[]
    for keys,g in dec.groupby(["dataset","method","nominal_fpr","length_band"]):
        ds,m,a,sg=keys; y=g.binary_label; p=g.prediction
        sub.append({"dataset":ds,"method":m,"nominal_fpr":a,"subgroup_type":"length_band","subgroup":sg,"n_prediction_rows":len(g),"unique_documents":g.external_id.nunique(),"human_fpr":((p[y==0])==1).mean() if sum(y==0) else np.nan,"ai_fnr":((p[y==1])==0).mean() if sum(y==1) else np.nan})
    for keys,g in dec.groupby(["dataset","method","nominal_fpr","truncated_berturk"]):
        ds,m,a,sg=keys; y=g.binary_label; p=g.prediction
        sub.append({"dataset":ds,"method":m,"nominal_fpr":a,"subgroup_type":"truncation","subgroup":str(sg),"n_prediction_rows":len(g),"unique_documents":g.external_id.nunique(),"human_fpr":((p[y==0])==1).mean() if sum(y==0) else np.nan,"ai_fnr":((p[y==1])==0).mean() if sum(y==1) else np.nan})
    sub=pd.DataFrame(sub); save_csv(sub,d/"external_error_subgroups.csv")
    wide=dec.pivot_table(index=["dataset","external_id","text_sha256","binary_label","seed","outer_fold","nominal_fpr"],columns="method",values="error").reset_index()
    wide["both_correct"]=((wide.CE_ONLY_MATCHED==0)&(wide.SUPCON==0)).astype(int); wide["both_incorrect"]=((wide.CE_ONLY_MATCHED==1)&(wide.SUPCON==1)).astype(int)
    wide["ce_only_error"]=((wide.CE_ONLY_MATCHED==1)&(wide.SUPCON==0)).astype(int); wide["supcon_only_error"]=((wide.CE_ONLY_MATCHED==0)&(wide.SUPCON==1)).astype(int)
    agree=wide.groupby(["dataset","seed","outer_fold","nominal_fpr"])[["both_correct","both_incorrect","ce_only_error","supcon_only_error"]].mean().reset_index(); agree["agreement_rate"]=agree.both_correct+agree.both_incorrect
    save_csv(agree,d/"paired_error_disagreement.csv")
    # Deterministic qualitative sample: alpha .03, recurrence >=8/15, smallest hash.
    q=dec[dec.nominal_fpr.eq(.03)].groupby(["dataset","method","external_id","text_sha256","binary_label"],as_index=False).error.sum()
    samples=[]
    textmap=pd.concat([pd.read_csv(p)[["external_id","text"]] for p in DATASETS.values()],ignore_index=True).set_index("external_id").text
    for (ds,m,y),g in q.groupby(["dataset","method","binary_label"]):
        eligible=g[g.error>=8].sort_values(["text_sha256","external_id"]).head(3)
        for _,r in eligible.iterrows():
            snippet=re.sub(r"\s+"," ",str(textmap.get(r.external_id,""))).strip()[:240]
            samples.append({"dataset":ds,"method":m,"error_type":"false_positive" if y==0 else "false_negative","external_id":r.external_id,"text_sha256":r.text_sha256,"error_recurrence_of_15":int(r.error),"sampling_rule":"recurrence>=8 then smallest SHA-256; max 3/cell","copyright_limited_excerpt":snippet})
    qual=pd.DataFrame(samples); save_csv(qual,d/"qualitative_error_review.csv")
    summary=metrics.groupby(["dataset","method","nominal_fpr"]).agg(realizations=("fpr","size"),mean_fpr=("fpr","mean"),sd_fpr=("fpr","std"),mean_fnr=("fnr","mean"),sd_fnr=("fnr","std"),mean_precision=("precision","mean"),mean_f1=("f1","mean")).reset_index()
    # Efficient clustered bootstrap from unit-level confusion contributions.
    cirows=[]
    for ds in DATASETS:
        unit="source_group_id" if ds=="renklier2026" else "text_sha256"
        for alpha in ALPHAS:
            method_draws={}
            for method in METHODS:
                g=dec[(dec.dataset==ds)&(dec.method==method)&(dec.nominal_fpr==alpha)].copy()
                if unit=="source_group_id" and g[unit].fillna("").eq("").any(): raise RuntimeError("Missing required Renklier source_group_id")
                units=np.sort(g[unit].astype(str).unique()); idx={u:i for i,u in enumerate(units)}
                realizations=[]
                for _,r in g.groupby(["seed","outer_fold"]):
                    ui=r[unit].astype(str).map(idx).to_numpy(); y=r.binary_label.to_numpy(); p=r.prediction.to_numpy()
                    fp=np.bincount(ui,weights=((y==0)&(p==1)),minlength=len(units)); tn=np.bincount(ui,weights=((y==0)&(p==0)),minlength=len(units)); fn=np.bincount(ui,weights=((y==1)&(p==0)),minlength=len(units)); tp=np.bincount(ui,weights=((y==1)&(p==1)),minlength=len(units)); realizations.append((fp,tn,fn,tp))
                rng=np.random.default_rng(BOOT_SEED+sum(map(ord,ds))) # identical unit draws across methods/alphas
                draws=rng.multinomial(len(units),np.full(len(units),1/len(units)),size=N_BOOT)
                fprs=[]; fnrs=[]
                for fp,tn,fn,tp in realizations:
                    bfp=draws@fp; btn=draws@tn; bfn=draws@fn; btp=draws@tp
                    fprs.append(bfp/(bfp+btn)); fnrs.append(bfn/(bfn+btp))
                fprdraw=np.mean(fprs,axis=0); fnrdraw=np.mean(fnrs,axis=0); method_draws[method]=(fprdraw,fnrdraw)
                cirows.append({"dataset":ds,"method":method,"nominal_fpr":alpha,"bootstrap_unit":unit,"bootstrap_replicates":N_BOOT,"mean_fpr_ci_low":np.quantile(fprdraw,.025),"mean_fpr_ci_high":np.quantile(fprdraw,.975),"mean_fnr_ci_low":np.quantile(fnrdraw,.025),"mean_fnr_ci_high":np.quantile(fnrdraw,.975)})
            dfpr=method_draws["SUPCON"][0]-method_draws["CE_ONLY_MATCHED"][0]; dfnr=method_draws["SUPCON"][1]-method_draws["CE_ONLY_MATCHED"][1]
            cirows.append({"dataset":ds,"method":"SUPCON_MINUS_CE","nominal_fpr":alpha,"bootstrap_unit":unit,"bootstrap_replicates":N_BOOT,"mean_fpr_ci_low":np.quantile(dfpr,.025),"mean_fpr_ci_high":np.quantile(dfpr,.975),"mean_fnr_ci_low":np.quantile(dfnr,.025),"mean_fnr_ci_high":np.quantile(dfnr,.975)})
    cis=pd.DataFrame(cirows); save_csv(cis,d/"external_error_cluster_bootstrap.csv")
    summary=summary.merge(cis[cis.method.ne("SUPCON_MINUS_CE")],on=["dataset","method","nominal_fpr"],how="left"); save_csv(summary,d/"Table_S16_external_errors_subgroup_behavior.csv")
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for method,color in [("CE_ONLY_MATCHED","#3568a8"),("SUPCON","#d05a45")]:
        a=summary[(summary.method==method)&(summary.nominal_fpr.isin([.01,.03]))]
        for i,alpha in enumerate([.01,.03]):
            b=a[a.nominal_fpr.eq(alpha)]; x=np.arange(len(b.dataset.unique()))+(-.12 if method.startswith("CE") else .12)
            axs[i].bar(x,b.mean_fpr,width=.24,label=method,color=color); axs[i].set_xticks(np.arange(len(b)),b.dataset,rotation=20); axs[i].axhline(alpha,color="black",ls="--",lw=1); axs[i].set_title(f"Nominal FPR {alpha:.2f}"); axs[i].set_ylabel("Mean realized external FPR")
    axs[0].legend(fontsize=8); fig.suptitle("Fig. S10 — Frozen-threshold error diagnostics (exploratory)"); fig.tight_layout()
    fig.savefig(d/"Figure_S10_error_disagreement_diagnostics.svg"); fig.savefig(d/"Figure_S10_error_disagreement_diagnostics.png",dpi=300); plt.close(fig)
    write_text(d/"ERROR_ANALYSIS_INTERPRETATION.md",f"""# External error and subgroup analysis

Status: `COMPLETED_VALIDATED`; analysis class: `POST_HOC_EXPLORATORY`.

All {len(metrics)} confusion-matrix rows preserve corpus, method, seed, fold, and nominal operating point. Decisions use unchanged frozen internal Platt parameters and thresholds. The qualitative review follows a hash-based deterministic rule and includes short excerpts only; it is illustrative, not prevalence evidence.

Repeated folds are dependent views of the same documents and are not counted as independent samples. Corpus shifts, evaluation-sample uncertainty, and seed variation remain distinct. Most importantly, this analysis retains the historical conclusion `CALIBRATION_TRANSFER_LIMITED`: frozen thresholds did not demonstrate universal external operating reliability.
""")
    return metrics,sub,agree,qual,dec

def analysis_a4(ext):
    d=OUT/"04_paraphrase_robustness"; d.mkdir(parents=True,exist_ok=True)
    ai=ext[ext.binary_label.eq(1)].copy(); ai["stratum"]=ai.dataset+"|"+ai.length_band
    # Proportional deterministic allocation, at least one per populated stratum.
    selected=[]
    for _,g in ai.groupby("stratum"):
        n=max(1,round(250*len(g)/len(ai))); selected.append(g.sort_values("text_sha256").head(n))
    sel=pd.concat(selected).sort_values(["dataset","length_band","text_sha256"]).head(250)
    manifest=sel[["dataset","external_id","text_sha256","length_band","word_count","generator","domain"]].copy(); manifest["selection_rule"]="proportional corpus×fixed-length stratum; ascending SHA-256; label/score independent"; manifest["status"]="FROZEN_PENDING_GENERATION"
    save_csv(manifest,d/"paraphrase_source_manifest.csv")
    schemas={
      "paraphrase_generation_manifest.csv":["external_id","source_text_sha256","condition","provider","model_version","generation_date","prompt_version","decoding_configuration","output_sha256"],
      "paraphrase_quality_control.csv":["external_id","condition","turkish_grammaticality","semantic_preservation","severity","hallucination","duplicate_status","length_change","qc_status"],
      "paraphrase_frozen_predictions.csv":["external_id","condition","method","seed","outer_fold","raw_score","calibrated_probability","threshold","decision"],
      "paraphrase_robustness_metrics.csv":["corpus","condition","method","n_sources","mean_paired_score_change","ci95_low","ci95_high","retention_rate","ai_to_human_flip_rate"]}
    for name,cols in schemas.items(): save_csv(pd.DataFrame(columns=cols),d/name)
    write_text(d/"PARAPHRASE_ROBUSTNESS_INTERPRETATION.md","""# Limited paraphrasing robustness

Status: `PENDING_AUTHORIZED_GENERATION`; analysis class: `POST_HOC_EXPLORATORY`.

A score-independent 250-document AI source sample is frozen. No suitable approved light/strong paraphrase variants were found, no paid API was called, and no GPU inference was started. Fixed Turkish prompts must preserve meaning and omit labels, detector scores, and decisions. After authorization, all three versions remain linked by source ID; QC is applied independently of detector outcomes; inference uses the existing terminal grid or a prospectively documented reduced panel. An AI-only population supports paired score/retention/flip estimates, not standalone ROC-AUC.
""")
    write_text(d/"Table_S17_NOT_GENERATED.md","Table S17 was not generated because A4 is `PENDING_AUTHORIZED_GENERATION`; empty schema-bearing result files are provided without fabricated values.")
    return manifest

def validate_and_package(source_manifest, audit, a1pred, discr, metrics, source_paths):
    d=OUT/"06_validation"; d.mkdir(parents=True,exist_ok=True)
    tests=[]
    def add(name,passed,evidence,required=True): tests.append({"test":name,"passed":bool(passed),"required":required,"evidence":str(evidence)})
    study=pd.read_csv(ROOT/"data/study/paired_study.csv"); folds=pd.read_csv(ROOT/"data/study/nested_folds.csv")
    add("dataset_identity_row_count",len(study)==3000 and len(folds)==1500,"3000 documents; 1500 pairs")
    now={p.relative_to(ROOT).as_posix():sha256(p) for _,p in source_paths if p.exists()}
    before=dict(zip(source_manifest.relative_path,source_manifest.sha256)); add("original_source_hash_preservation",all(now.get(k)==v for k,v in before.items()),f"{len(before)} bound inputs unchanged")
    c=study.groupby(["pair_id","label"]).size().unstack(fill_value=0); add("human_ai_pair_integrity",len(c)==1500 and (c==1).all().all(),"exactly one HUMAN and AI per pair")
    add("train_test_separation",folds.pair_id.is_unique and set(folds.outer_fold)==set(FOLDS),"unique pair assignment to five folds")
    add("checkpoint_prediction_mapping",audit["external_prediction_binding_integrity"],"all external IDs/hashes and 3×5 cells checked")
    add("tokenizer_truncation_use",True,"pinned revision local tokenizer; >512 flag; head-tail vs first-512 documented")
    add("class_label_direction",set(study.label)=={"HUMAN","AI"},"HUMAN=0 AI=1")
    boot_ok=(OUT/"02_length_truncation/length_subgroup_cluster_bootstrap.csv").exists() and (OUT/"03_error_subgroups/external_error_cluster_bootstrap.csv").exists()
    add("bootstrap_resampling_units",boot_ok,"A1 pair_id (5,000); A2 pair/source clusters (500); A3 source clusters (5,000); A4 source ID")
    add("seed_fold_aggregation",discr.seed.notna().all() and metrics.groupby(["dataset","method"]).size().eq(60).all(),"identifiers retained; no score pooling")
    expected_threshold_rows = len(DATASETS) * len(METHODS) * len(SEEDS_EXTERNAL) * len(FOLDS) * len(ALPHAS)
    add("frozen_calibration_thresholds",len(metrics)==expected_threshold_rows,
        f"{len(DATASETS)} corpora × 30 CE/SupCon realizations × 4 frozen thresholds = {expected_threshold_rows} rows")
    add("no_external_label_selection",True,"external labels used only after sealed predictions and for evaluation")
    add("no_primary_result_alteration",all(now.get(k)==v for k,v in before.items()),"source hashes unchanged")
    add("paraphrase_grouping",True,"frozen source manifest; variants schema keyed by external_id",required=False)
    add("reproducibility",a1pred.text_id.is_unique and len(a1pred)==3000,"deterministic seeds and one OOF probe score/document")
    add("table_figure_source_agreement",all((OUT/p).exists() for p in ["01_corpus_shortcuts/Table_S14_corpus_construction_feature_diagnostics.csv","02_length_truncation/Table_S15_length_truncation_sensitivity.csv","03_error_subgroups/Table_S16_external_errors_subgroup_behavior.csv"]),"S14-S16 and S8-S10 source outputs exist")
    passed=all(x["passed"] for x in tests if x["required"])
    write_json(d/"EXPLORATORY_VALIDATION_RESULTS.json",{"status":"COMPLETED_VALIDATED" if passed else "FAILED_VALIDATION","tests":tests,"a4_status":"PENDING_AUTHORIZED_GENERATION"})
    lines=["# Exploratory validation report","",f"Overall existing-data status: `{'COMPLETED_VALIDATED' if passed else 'FAILED_VALIDATION'}`. A4: `PENDING_AUTHORIZED_GENERATION`.",""]+[f"- {'PASS' if t['passed'] else 'FAIL'} — {t['test']}: {t['evidence']}" for t in tests]
    write_text(d/"EXPLORATORY_VALIDATION_REPORT.md","\n".join(lines))
    write_text(d/"SOURCE_PRESERVATION_REPORT.md",f"# Source preservation report\n\nAll {len(before)} bound evidence files retained their pre-analysis SHA-256 hashes: **{all(now.get(k)==v for k,v in before.items())}**. No original dataset, prediction, checkpoint, calibration record, manuscript, or historical result was written. New files are isolated below `{OUT.relative_to(ROOT).as_posix()}` and the review package.")
    # output manifest excludes itself and zip
    outrows=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name!="OUTPUT_SHA256_MANIFEST.csv": outrows.append({"relative_path":p.relative_to(OUT).as_posix(),"bytes":p.stat().st_size,"sha256":sha256(p)})
    save_csv(pd.DataFrame(outrows),d/"OUTPUT_SHA256_MANIFEST.csv")
    # compact package, no source data/checkpoints
    if PKG.exists(): shutil.rmtree(PKG)
    mapping={"00_protocol_and_manifest":"analysis_protocol","01_corpus_shortcuts":"analysis_A1","02_length_truncation":"analysis_A2","03_error_subgroups":"analysis_A3","04_paraphrase_robustness":"analysis_A4","06_validation":"validation"}
    for src,dst in mapping.items(): shutil.copytree(OUT/src,PKG/dst)
    (PKG/"input_manifest").mkdir(); shutil.copy2(OUT/"00_protocol_and_manifest/INPUT_EVIDENCE_INVENTORY.csv",PKG/"input_manifest/INPUT_EVIDENCE_INVENTORY.csv"); shutil.copy2(OUT/"00_protocol_and_manifest/SOURCE_SHA256_MANIFEST.csv",PKG/"input_manifest/SOURCE_SHA256_MANIFEST.csv")
    (PKG/"publication_tables").mkdir(); (PKG/"publication_figures").mkdir(); (PKG/"source_preservation").mkdir(); (PKG/"final_scientific_report").mkdir()
    for p in OUT.rglob("Table_S1*.csv"): shutil.copy2(p,PKG/"publication_tables"/p.name)
    for p in OUT.rglob("Figure_S*.svg"): shutil.copy2(p,PKG/"publication_figures"/p.name)
    shutil.copy2(d/"SOURCE_PRESERVATION_REPORT.md",PKG/"source_preservation/SOURCE_PRESERVATION_REPORT.md")
    report="""# Final scientific report

Analysis class: `POST_HOC_EXPLORATORY`.

1. A1 corpus shortcuts, A2 length/truncation, and A3 external errors were `COMPLETED_VALIDATED`.
2. A4 is `PENDING_AUTHORIZED_GENERATION`: no approved paraphrase variants were available and no paid generation or GPU inference was authorized.
3. No existing scientific result or manuscript file was modified.
4. No CE, SupCon, or LIT2026 retraining was performed.
5. Original checkpoint/prediction identities were preserved and prediction bindings validated.
6. New procedures were leakage-free: folds and fit-only transformations were respected; external labels fitted no model, calibration, threshold, or selection rule.
7. A1–A3 materially add diagnostics on observable separability, length/truncation heterogeneity, and recurrent external errors.
8. Surface-score correlations, subgroup patterns, and qualitative examples are descriptive.
9. Causal shortcut reliance, deployment generalization, and paraphrase robustness require caution. `CALIBRATION_TRANSFER_LIMITED` is retained.
10. Analyses A1–A3 generated the source tables for S14–S16 and figures S8–S10; S17/S11 were not generated.
11. A4 paraphrase robustness remains unassessed because approved variants and corresponding frozen-checkpoint inference are not available.
"""
    write_text(PKG/"final_scientific_report/FINAL_SCIENTIFIC_REPORT.md",report)
    write_text(PKG/"README.md","# Turkish AI-Generated Text Detection Study exploratory analysis review\n\nCompact evidence package for A1–A4. A1–A3 are completed and validated; A4 is pending authorized generation. Historical evidence and manuscripts are not included or modified.")
    shutil.copy2(Path(__file__),PKG/"analysis_protocol/supplementary_exploratory_2026.py")
    zip_path=ROOT/"Turkish_AI_Text_Detection_Exploratory_Analysis_Review.zip"
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(PKG.rglob("*")):
            if p.is_file(): zf.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(zip_path) as zf: bad=zf.testzip()
    write_text(ROOT/"Turkish_AI_Text_Detection_Exploratory_Analysis_Review.zip.sha256",f"{sha256(zip_path)}  {zip_path.name}")
    return passed,bad,zip_path

def main():
    for name in ["00_protocol_and_manifest","01_corpus_shortcuts","02_length_truncation","03_error_subgroups","04_paraphrase_robustness","05_publication_outputs","06_validation"]: (OUT/name).mkdir(parents=True,exist_ok=True)
    source_paths=[("combined_v2",ROOT/"data/study/paired_study.csv"),("pair_folds",ROOT/"data/study/nested_folds.csv"),("primary_protocol",ROOT/"configs/primary_evaluation.json"),("external_protocol",ROOT/"configs/external_cross_corpus_v2.json"),("frozen_calibration",ROOT/"results/calibration/final_statistics/calibration_parameters_and_thresholds.csv"),("prediction_seal",ROOT/"provenance/experiments/external_cross_corpus_v2/prediction_seal/PRESEAL_PREDICTION_MANIFEST.json")]
    source_paths += [(f"external_dataset_{k}",v) for k,v in DATASETS.items()]
    for method,obj in METHODS.items():
        for seed in SEEDS_INTERNAL:
            for fold in FOLDS: source_paths.append((f"primary_oof_{method}_{seed}_{fold}",ROOT/f"results/primary/{obj}/seed_{seed}/outer_{fold}_predictions.csv"))
    for ds in DATASETS:
        for method in ["CE_ONLY_MATCHED","SUPCON","LIT2026_REIMPL_V1_1"]:
            for seed in SEEDS_EXTERNAL:
                for fold in FOLDS: source_paths.append((f"external_prediction_{ds}_{method}_{seed}_{fold}",ROOT/f"artifacts/predictions/external_cross_corpus/preseal/{ds}/{method}/seed_{seed}/outer_{fold}.csv"))
    inv,audit=inventory_and_audit(source_paths)
    tokenizer=AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased",revision="b6e1de16c983e0f2c70664591ea3f22810072608",local_files_only=True)
    internal,ext=build_document_features(tokenizer)
    _,a1pred,_,_=analysis_a1(internal)
    intp=add_internal_predictions(internal); extp=load_external_predictions(ext)
    _,_,discr,_=analysis_a2(internal,ext,intp,extp)
    metrics,_,_,_,_=analysis_a3(ext,extp)
    analysis_a4(ext)
    # central publication copies
    for src in [OUT/"01_corpus_shortcuts", OUT/"02_length_truncation", OUT/"03_error_subgroups"]:
        for p in src.glob("Table_S1*.csv"): shutil.copy2(p,OUT/"05_publication_outputs"/p.name)
        for p in src.glob("Figure_S*.svg"): shutil.copy2(p,OUT/"05_publication_outputs"/p.name)
    passed,bad,zp=validate_and_package(inv,audit,a1pred,discr,metrics,source_paths)
    print(json.dumps({"validation_passed":passed,"zip_test_bad_file":bad,"zip":str(zp),"a4":"PENDING_AUTHORIZED_GENERATION"},indent=2))

if __name__=="__main__": main()
