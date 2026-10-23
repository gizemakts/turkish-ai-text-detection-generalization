"""Supplementary exploratory analyses, cluster-based uncertainty, and source validation."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
import re
import shutil
import sys
import unicodedata
import zipfile
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

VERSION = "20261012"
STATUS = "POST_HOC_EXPLORATORY"
BOOT_SEED = 261012
METHODS = ["CE_ONLY_MATCHED", "SUPCON", "LIT2026_REIMPL_V1_1"]
TURKISH_AI_DETECTION_METHODS = ["CE_ONLY_MATCHED", "SUPCON"]
DATASET_LABELS = {
    "combined_v2": "Combined-v2",
    "renklier2026": "Renklier2026",
    "tum2024_tr": "TUM2024_TR",
    "bostancioglu2026_tr": "Bostancioglu2026_TR",
}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text.rstrip()+"\n",encoding="utf-8")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def require_project(root: Path) -> None:
    required=["data/study/paired_study.csv","data/study/nested_folds.csv",
              "results/supplementary_exploratory_2026/01_corpus_shortcuts/surface_probe_results.csv",
              "configs/primary_evaluation.json"]
    missing=[p for p in required if not (root/p).is_file()]
    if missing: raise RuntimeError(f"Explicit --project-root is not a Turkish AI-Generated Text Detection Study project; missing: {missing}")


def sigmoid(x):
    x=np.asarray(x,float)
    return np.where(x>=0,1/(1+np.exp(-x)),np.exp(x)/(1+np.exp(x)))


def metric(y, score, weight=None):
    y=np.asarray(y,int); score=np.asarray(score,float)
    if weight is None: weight=np.ones(len(y))
    weight=np.asarray(weight,float); keep=weight>0
    if keep.sum()<2 or len(np.unique(y[keep]))<2: return np.nan,np.nan
    return (float(roc_auc_score(y,score,sample_weight=weight)),
            float(average_precision_score(y,score,sample_weight=weight)))


def cluster_draws(units, n_boot, seed):
    """Multinomial cluster multiplicities; every row in a unit gets one weight."""
    unique=np.sort(pd.Series(units).astype(str).unique())
    index={u:i for i,u in enumerate(unique)}
    row_index=pd.Series(units).astype(str).map(index).to_numpy()
    rng=np.random.default_rng(seed)
    draws=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=n_boot)
    return unique,row_index,draws


def normalize_surface(text: str, tokenizer) -> dict:
    text="" if pd.isna(text) else str(text)
    words=re.findall(r"\S+",text,flags=re.UNICODE)
    lexical=re.findall(r"[^\W\d_]+",text.casefold(),flags=re.UNICODE)
    sent=[s for s in re.split(r"(?<=[.!?…])\s+|[\r\n]+",text.strip()) if s.strip()]
    wp=len(tokenizer.encode(text,add_special_tokens=True,truncation=False,verbose=False))
    chars=max(len(text),1); alpha=sum(c.isalpha() for c in text)
    return {
      "word_count":len(words),"sentence_count":len(sent),"mean_sentence_words":len(words)/max(len(sent),1),
      "wordpiece_count":wp,"punctuation_per_100_chars":100*sum(unicodedata.category(c).startswith("P") for c in text)/chars,
      "digit_token_fraction":sum(any(c.isdigit() for c in w) for w in words)/max(len(words),1),
      "year_date_per_100_words":100*(len(re.findall(r"(?<!\d)(?:18|19|20|21)\d{2}(?!\d)",text))+len(re.findall(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:19|20)?\d{2}(?!\d)",text)))/max(len(words),1),
      "uppercase_alpha_ratio":sum(c.isupper() for c in text)/max(alpha,1),"lexical_diversity":len(set(lexical))/max(len(lexical),1),
      "structured_heading":int(any(re.match(r"^\s{0,3}#{1,6}\s+",x) or (x.strip().isupper() and 2<=len(x.split())<=15) for x in text.splitlines())),
      "list_line":int(any(re.match(r"^\s*(?:[-*+]|\d+[.)])\s+",x) for x in text.splitlines())),
      "quote_per_100_chars":100*sum(c in '\"“”‘’«»' for c in text)/chars,
      "dash_per_100_chars":100*sum(c in "-–—" for c in text)/chars,"newline_per_100_chars":100*text.count("\n")/chars,
    }


FEATURES=["word_count","sentence_count","mean_sentence_words","wordpiece_count","punctuation_per_100_chars",
          "digit_token_fraction","year_date_per_100_words","uppercase_alpha_ratio","lexical_diversity",
          "structured_heading","list_line","quote_per_100_chars","dash_per_100_chars","newline_per_100_chars"]


def pair_boot_ci(frame, score, n_boot, seed):
    pairs=np.sort(frame.pair_id.unique()); groups={k:g for k,g in frame.groupby("pair_id",sort=False)}
    rng=np.random.default_rng(seed); auc=[]; ap=[]
    for _ in range(n_boot):
        b=pd.concat([groups[k] for k in rng.choice(pairs,len(pairs),replace=True)],ignore_index=True)
        a,p=metric(b.binary_label,b[score]); auc.append(a); ap.append(p)
    return np.quantile(auc,[.025,.975]),np.quantile(ap,[.025,.975])


def a1(root, out, tokenizer, normalize_text, n_boot):
    d=out/"amended_A1"; d.mkdir(parents=True,exist_ok=True)
    study=pd.read_csv(root/"data/study/paired_study.csv"); folds=pd.read_csv(root/"data/study/nested_folds.csv")
    source=study.text.fillna("").astype(str); norm=source.map(normalize_text)
    rows=[]
    for i,(s,n) in enumerate(zip(source,norm)):
        rows.append({"text_id":study.text_id.iloc[i],"domain":study.domain.iloc[i],"exact_equal":s==n,
          "source_sha256":hashlib.sha256(s.encode()).hexdigest(),"condition_a_sha256":hashlib.sha256(n.encode()).hexdigest(),
          "source_chars":len(s),"condition_a_chars":len(n),"source_newlines":s.count("\n")+s.count("\r"),"condition_a_newlines":n.count("\n"),
          "source_tabs":s.count("\t"),"condition_a_tabs":n.count("\t"),
          "source_controls":sum(unicodedata.category(c) in {"Cc","Cf","Cs","Co","Cn"} for c in s),
          "condition_a_controls":sum(unicodedata.category(c) in {"Cc","Cf","Cs","Co","Cn"} for c in n)})
    eq=pd.DataFrame(rows); save_csv(eq,d/"condition_a_text_equivalence_rows.csv")
    sf=pd.DataFrame([normalize_surface(t,tokenizer) for t in source]); nf=pd.DataFrame([normalize_surface(t,tokenizer) for t in norm])
    diffs=[]
    for f in FEATURES:
        delta=nf[f]-sf[f]
        diffs.append({"feature":f,"changed_documents":int((delta.abs()>1e-12).sum()),"max_absolute_change":float(delta.abs().max()),"mean_change_condition_a_minus_source":float(delta.mean())})
    diff=pd.DataFrame(diffs); save_csv(diff,d/"condition_a_feature_change_summary.csv")
    assessment="EQUIVALENT" if eq.exact_equal.all() and diff.changed_documents.eq(0).all() else "NON_EQUIVALENT"
    # Separate sensitivity probe; never replaces the historical source-text result.
    z=pd.concat([study[["text_id","pair_id","label","domain"]],nf],axis=1).merge(folds[["pair_id","outer_fold"]],on="pair_id",validate="many_to_one")
    z["binary_label"]=(z.label=="AI").astype(int); pred=[]
    for fold in range(5):
        tr=z.outer_fold.ne(fold); te=~tr
        model=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(C=1,solver="liblinear",max_iter=2000,random_state=BOOT_SEED))])
        model.fit(z.loc[tr,FEATURES],z.loc[tr,"binary_label"])
        q=z.loc[te,["text_id","pair_id","label","binary_label","domain","outer_fold"]].copy(); q["condition_a_surface_score"]=model.predict_proba(z.loc[te,FEATURES])[:,1]; pred.append(q)
    pred=pd.concat(pred,ignore_index=True); save_csv(pred,d/"condition_a_surface_probe_oof_predictions.csv")
    sens=[]
    for domain in ["ALL","ABSTRACT","NEWSPAPER"]:
        q=pred if domain=="ALL" else pred[pred.domain.eq(domain)]; auc,ap=metric(q.binary_label,q.condition_a_surface_score); aci,pci=pair_boot_ci(q,"condition_a_surface_score",n_boot,BOOT_SEED+len(domain))
        sens.append({"input_representation":"Condition-A-normalized sensitivity","domain":domain,"roc_auc":auc,"roc_auc_ci_low":aci[0],"roc_auc_ci_high":aci[1],"average_precision":ap,"average_precision_ci_low":pci[0],"average_precision_ci_high":pci[1],"n_documents":len(q),"n_pairs":q.pair_id.nunique(),"protocol":"same mixed-domain pair-disjoint outer folds; train-only scaling; C=1 L2 logistic"})
    sens=pd.DataFrame(sens); save_csv(sens,d/"condition_a_surface_probe_results.csv")
    original=pd.read_csv(root/"results/supplementary_exploratory_2026/01_corpus_shortcuts/surface_probe_results.csv")
    original.insert(0,"input_representation","Source text (original preserved A1)"); original["protocol"]="same mixed-domain pair-disjoint probe; subgroup scores are evaluations of this probe, not domain-specific fits"
    perf=pd.concat([original,sens],ignore_index=True)
    effects=pd.read_csv(root/"results/supplementary_exploratory_2026/01_corpus_shortcuts/corpus_shortcut_paired_effects.csv")
    effects.insert(0,"panel","A_feature_differences_source_text"); perf.insert(0,"panel","B_surface_probe_diagnostic_performance")
    # One machine-readable table with unioned columns and explicit panels.
    table=pd.concat([effects,perf],ignore_index=True,sort=False); save_csv(table,d/"Table_S14_amended.csv")
    figdata=pd.concat([study[["text_id","pair_id","label","domain"]],sf[["word_count","lexical_diversity","punctuation_per_100_chars"]]],axis=1)
    save_csv(figdata,d/"Figure_S8_source_data.csv")
    fig,axs=plt.subplots(2,4,figsize=(13,6.5),gridspec_kw={"width_ratios":[1,1,1,.9]})
    colors={"HUMAN":"#3568a8","AI":"#d05a45"}
    for ri,domain in enumerate(["ABSTRACT","NEWSPAPER"]):
        q=figdata[figdata.domain.eq(domain)]
        for ci,f in enumerate(["word_count","lexical_diversity","punctuation_per_100_chars"]):
            vals=[q.loc[q.label.eq(l),f].to_numpy() for l in ["HUMAN","AI"]]
            parts=axs[ri,ci].violinplot(vals,positions=[0,1],showmedians=True,showextrema=False,widths=.8)
            for body,c in zip(parts["bodies"],[colors["HUMAN"],colors["AI"]]): body.set_facecolor(c); body.set_alpha(.65)
            axs[ri,ci].set_xticks([0,1],["HUMAN","AI"]); axs[ri,ci].set_title(f.replace("_"," ")); axs[ri,ci].set_ylabel(domain if ci==0 else "")
        r=original[original.domain.eq(domain)].iloc[0]; axs[ri,3].errorbar([0],[r.roc_auc],yerr=[[r.roc_auc-r.roc_auc_ci_low],[r.roc_auc_ci_high-r.roc_auc]],fmt="o",color="#4a4a4a",capsize=4)
        axs[ri,3].set_xlim(-.6,.6); axs[ri,3].set_ylim(.85,1.005); axs[ri,3].set_xticks([0],["Mixed-domain\nprobe"]); axs[ri,3].set_title("OOF ROC-AUC (95% CI)")
    fig.suptitle("Figure S8. Source-text surface separability by domain (post-hoc exploratory)")
    fig.text(.5,.01,"Subgroup values evaluate one mixed-domain pair-disjoint probe; high separability does not demonstrate causal reliance by BERTurk.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.96]); fig.savefig(d/"Figure_S8_amended.svg"); fig.savefig(d/"Figure_S8_amended.png",dpi=300); plt.close(fig)
    changed=int((~eq.exact_equal).sum()); affected=diff.loc[diff.changed_documents.gt(0),"feature"].tolist()
    write_text(d/"A1_INPUT_NORMALIZATION_AUDIT.md",f"""# A1 input normalization audit

Result: `{assessment}`. Analysis class: `{STATUS}`.

The original A1 extractor used `data/study/paired_study.csv:text` directly. The primary runner uses `condition_text(data, 'A')`, which calls the bound `normalize_text` implementation before head-tail tokenization. That implementation applies NFC, replaces disallowed control/format/private/unassigned characters, normalizes CR/LF, collapses horizontal whitespace, replaces newline runs with spaces, and trims ends.

Across 3,000 documents, {changed} source strings differed byte-for-byte from effective Condition-A text. Affected A1 features: {', '.join(affected) if affected else 'none'}. Row-level hashes and formatting counts are provided. The original A1 ROC-AUC 0.96504178 is preserved and is now explicitly described as **source-text surface separability**. A separately labeled leakage-free Condition-A-normalized sensitivity probe was run on the same original pair-disjoint folds; it does not replace the original result and is not evidence of causal BERTurk feature reliance.
""")
    return {"assessment":assessment,"changed_texts":changed,"affected_features":affected,"original":original,"sensitivity":sens}


def higher_bootstrap(root, discr, doc, n_boot):
    """Higher-replicate sensitivity for CE/SupCon; fixed realizations, matched draws."""
    # Reconstruct score-level inputs from historical predictions and document features.
    study=pd.read_csv(root/"data/study/paired_study.csv")
    intframes=[]
    for method,obj in [("CE_ONLY_MATCHED","ce"),("SUPCON","supcon")]:
        for seed in [42,137,314,2026,3407]:
            p=pd.concat([pd.read_csv(root/f"results/primary/{obj}/seed_{seed}/outer_{f}_predictions.csv") for f in range(5)])
            p["method"]=method; intframes.append(p)
    intp=pd.concat(intframes).merge(doc[doc.dataset.eq("combined_v2")][["document_id","length_band","truncated_berturk"]],left_on="text_id",right_on="document_id",validate="many_to_one")
    extframes=[]
    paths={"renklier2026":"renklier2026_test_standardized.csv","tum2024_tr":"tum2024_tr_test_standardized.csv","bostancioglu2026_tr":"bostancioglu2026_tr_full_standardized.csv"}
    for ds,name in paths.items():
        meta=pd.read_csv(root/"data/external_prepared"/name)
        lf=doc[doc.dataset.eq(ds)][["document_id","length_band","truncated_berturk"]]
        for method in TURKISH_AI_DETECTION_METHODS:
            for seed in [42,314,2026]:
                for fold in range(5):
                    p=pd.read_csv(root/f"artifacts/predictions/external_cross_corpus/preseal/{ds}/{method}/seed_{seed}/outer_{fold}.csv")
                    p["dataset"]=ds; extframes.append(p.merge(meta[["external_id","binary_label","source_group_id","text_sha256"]],on=["external_id","text_sha256"],validate="one_to_one").merge(lf,left_on="external_id",right_on="document_id",validate="many_to_one"))
    extp=pd.concat(extframes,ignore_index=True)
    output=[]
    populations=[("internal","combined_v2",intp,"pair_id","raw_decision_score","label")]
    for ds in paths:
        populations.append(("external",ds,extp[extp.dataset.eq(ds)],"source_group_id" if ds=="renklier2026" else "text_sha256","raw_score_or_logit","binary_label"))
    for pop,ds,base,unit,score,labelcol in populations:
        for subgroup in ["short_le150","medium_151_300","long_gt300","truncated","all"]:
            g=base if subgroup=="all" else (base[base.truncated_berturk.eq(1)] if subgroup=="truncated" else base[base.length_band.eq(subgroup)])
            units=np.sort(g[unit].astype(str).unique())
            if len(units)<2: continue
            idx={u:i for i,u in enumerate(units)}; rng=np.random.default_rng(BOOT_SEED+sum(map(ord,ds+subgroup)))
            draws=rng.multinomial(len(units),np.full(len(units),1/len(units)),size=n_boot)
            bymethod={}
            for method in TURKISH_AI_DETECTION_METHODS:
                gm=g[g.method.eq(method)]; real=[]; keys=["seed"] if pop=="internal" else ["seed","outer_fold"]
                for _,r in gm.groupby(keys):
                    ui=r[unit].astype(str).map(idx).to_numpy(); y=(r[labelcol].eq("AI").astype(int) if pop=="internal" else r[labelcol].astype(int)).to_numpy(); real.append((ui,y,r[score].to_numpy()))
                aucd=np.full(n_boot,np.nan); apd=np.full(n_boot,np.nan)
                for bi in range(n_boot):
                    vals=[metric(y,s,draws[bi,ui]) for ui,y,s in real]
                    va=np.array([v[0] for v in vals]); vp=np.array([v[1] for v in vals])
                    if np.isfinite(va).any(): aucd[bi]=np.nanmean(va)
                    if np.isfinite(vp).any(): apd[bi]=np.nanmean(vp)
                bymethod[method]=(aucd,apd)
                output.append({"population":pop,"dataset":ds,"subgroup":subgroup,"method":method,"bootstrap_unit":unit,"bootstrap_replicates":n_boot,"unique_units":len(units),"roc_auc_ci_low":np.nanquantile(aucd,.025) if np.isfinite(aucd).any() else np.nan,"roc_auc_ci_high":np.nanquantile(aucd,.975) if np.isfinite(aucd).any() else np.nan,"average_precision_ci_low":np.nanquantile(apd,.025) if np.isfinite(apd).any() else np.nan,"average_precision_ci_high":np.nanquantile(apd,.975) if np.isfinite(apd).any() else np.nan})
            if set(bymethod)==set(TURKISH_AI_DETECTION_METHODS):
                da=bymethod["SUPCON"][0]-bymethod["CE_ONLY_MATCHED"][0]; dp=bymethod["SUPCON"][1]-bymethod["CE_ONLY_MATCHED"][1]
                output.append({"population":pop,"dataset":ds,"subgroup":subgroup,"method":"SUPCON_MINUS_CE","bootstrap_unit":unit,"bootstrap_replicates":n_boot,"unique_units":len(units),"roc_auc_ci_low":np.nanquantile(da,.025) if np.isfinite(da).any() else np.nan,"roc_auc_ci_high":np.nanquantile(da,.975) if np.isfinite(da).any() else np.nan,"average_precision_ci_low":np.nanquantile(dp,.025) if np.isfinite(dp).any() else np.nan,"average_precision_ci_high":np.nanquantile(dp,.975) if np.isfinite(dp).any() else np.nan})
    return pd.DataFrame(output)


def a2(root,out,n_boot):
    d=out/"amended_A2"; d.mkdir(parents=True,exist_ok=True)
    discr=pd.read_csv(root/"results/supplementary_exploratory_2026/02_length_truncation/length_subgroup_discrimination.csv")
    oldboot=pd.read_csv(root/"results/supplementary_exploratory_2026/02_length_truncation/length_subgroup_cluster_bootstrap.csv")
    doc=pd.read_csv(root/"results/supplementary_exploratory_2026/02_length_truncation/length_truncation_document_features.csv")
    high=higher_bootstrap(root,discr,doc,n_boot); save_csv(high,d/"length_subgroup_cluster_bootstrap_2000.csv")
    comp=oldboot.merge(high,on=["population","dataset","subgroup","method"],how="inner",suffixes=("_500","_2000"))
    for col in ["roc_auc_ci_low","roc_auc_ci_high","average_precision_ci_low","average_precision_ci_high"]: comp[f"absolute_endpoint_change_{col}"]=(comp[f"{col}_2000"]-comp[f"{col}_500"]).abs()
    save_csv(comp,d/"bootstrap_500_vs_2000_comparison.csv")
    # Counts and estimability are document-level; contributing cells are fixed.
    meta=[]; paths={"renklier2026":"renklier2026_test_standardized.csv","tum2024_tr":"tum2024_tr_test_standardized.csv","bostancioglu2026_tr":"bostancioglu2026_tr_full_standardized.csv"}
    internal=pd.read_csv(root/"data/study/paired_study.csv")[["text_id","pair_id","label"]].merge(doc[doc.dataset.eq("combined_v2")],left_on="text_id",right_on="document_id",validate="one_to_one")
    sources={"combined_v2":internal}
    for ds,name in paths.items():
        sources[ds]=pd.read_csv(root/"data/external_prepared"/name).merge(doc[doc.dataset.eq(ds)],left_on="external_id",right_on="document_id",validate="one_to_one")
    for ds,base in sources.items():
        for subgroup in ["short_le150","medium_151_300","long_gt300","truncated","all"]:
            g=base if subgroup=="all" else (base[base.truncated_berturk.eq(1)] if subgroup=="truncated" else base[base.length_band.eq(subgroup)])
            lab=g.label_x if "label_x" in g else g.label
            unit="pair_id_x" if ds=="combined_v2" and "pair_id_x" in g else ("pair_id" if ds=="combined_v2" else ("source_group_id" if ds=="renklier2026" else "text_sha256"))
            meta.append({"dataset":ds,"subgroup":subgroup,"total_n":len(g),"human_n":int(lab.astype(str).str.lower().eq("human").sum()),"ai_n":int(lab.astype(str).str.lower().eq("ai").sum()),"unique_resampling_units":g[unit].astype(str).nunique(),"resampling_unit":unit.replace("_x","")})
    meta=pd.DataFrame(meta)
    summary=discr.groupby(["population","dataset","method","subgroup"],as_index=False).agg(mean_roc_auc=("roc_auc","mean"),sd_roc_auc_across_cells=("roc_auc","std"),mean_average_precision=("average_precision","mean"),sd_ap_across_cells=("average_precision","std"),contributing_realizations=("roc_auc","count"),requested_cells=("seed","size"),contributing_seeds=("seed","nunique"))
    summary=summary.merge(meta,on=["dataset","subgroup"],how="left"); summary["roc_auc_estimable"]=(summary.human_n.gt(0)&summary.ai_n.gt(0))
    # Prefer 2,000-replicate amended CI for CE/SupCon; retain 500-replicate LIT interval, explicitly labeled.
    h=high[high.method.isin(TURKISH_AI_DETECTION_METHODS)]; b=oldboot[oldboot.method.eq("LIT2026_REIMPL_V1_1")]
    ci=pd.concat([h,b],ignore_index=True,sort=False)
    summary=summary.merge(ci[["population","dataset","subgroup","method","bootstrap_replicates","roc_auc_ci_low","roc_auc_ci_high","average_precision_ci_low","average_precision_ci_high"]],on=["population","dataset","subgroup","method"],how="left")
    summary["realization_structure"]=np.where(summary.population.eq("internal"),"5 seeds; pooled pair-disjoint OOF per seed","3 seeds × 5 dependent fold realizations; not 15 independent seeds")
    save_csv(summary,d/"Table_S15_amended.csv")
    figdata=summary[summary.subgroup.isin(["short_le150","medium_151_300","long_gt300","truncated"])].copy(); save_csv(figdata,d/"Figure_S9_source_data.csv")
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharey=False); order=["combined_v2","renklier2026","tum2024_tr","bostancioglu2026_tr"]
    suborder=["short_le150","medium_151_300","long_gt300","truncated"]; labels=["Short","Medium","Long","Truncated"]
    colors={"CE_ONLY_MATCHED":"#3568a8","SUPCON":"#d05a45","LIT2026_REIMPL_V1_1":"#4b9b69"}
    for ax,ds in zip(axs.flat,order):
        q=figdata[figdata.dataset.eq(ds)]
        for mi,m in enumerate(METHODS):
            r=q[q.method.eq(m)].set_index("subgroup").reindex(suborder); x=np.arange(4)+(mi-1)*.18
            y=r.mean_roc_auc.to_numpy(float); lo=r.roc_auc_ci_low.to_numpy(float); hi=r.roc_auc_ci_high.to_numpy(float)
            ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),fmt="o-",capsize=2,label=m,color=colors[m],lw=1)
        ax.set_xticks(range(4),[f"{lab}\nN={int(meta[(meta.dataset==ds)&(meta.subgroup==sg)].total_n.iloc[0])}\nH/A={int(meta[(meta.dataset==ds)&(meta.subgroup==sg)].human_n.iloc[0])}/{int(meta[(meta.dataset==ds)&(meta.subgroup==sg)].ai_n.iloc[0])}" for lab,sg in zip(labels,suborder)],fontsize=7)
        ax.set_title(DATASET_LABELS[ds]); ax.set_ylabel("Mean ROC-AUC"); ax.grid(axis="y",alpha=.2)
        if ds=="combined_v2": ax.set_ylim(.995,1.0001)
        else: ax.set_ylim(0,1.02)
    axs[0,0].legend(fontsize=7); fig.suptitle("Figure S9. Corpus-specific length and truncation sensitivity")
    fig.text(.5,.01,"Internal: five seed-level pooled OOF estimates. External: three seeds × five dependent folds; intervals cluster evaluation units with realizations fixed.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.96]); fig.savefig(d/"Figure_S9_amended.svg"); fig.savefig(d/"Figure_S9_amended.png",dpi=300); plt.close(fig)
    maxchange=comp.filter(like="absolute_endpoint_change").max().max() if len(comp) else np.nan
    write_text(d/"A2_BOOTSTRAP_AND_SUBGROUP_VALIDATION.md",f"""# A2 bootstrap and subgroup validation

Status: `A2_POSTHOC_AMENDED`; analysis class: `{STATUS}`.

The original 500-replicate implementation was inspected. Internal resampling uses `pair_id`; Renklier uses `source_group_id`; TUM and Bostancioglu use `text_sha256`. One multiplicity is assigned to every row in a sampled unit and identical draws are used for matched CE/SupCon comparisons. Seeds and folds remain fixed.

A 2,000-replicate sensitivity analysis was computationally feasible and was run for CE/SupCon. Original 500-replicate intervals remain archived; they were not overwritten. The largest absolute endpoint change across directly comparable intervals was {maxchange:.6f}. The machine-readable comparison supports interpretation review rather than a hard-coded conclusion.

Table S15 now reports total/HUMAN/AI counts, unique units, contributing seeds/realizations, and estimability. Single-class ROC-AUC remains `NaN`. BERTurk untruncated WordPiece counts use the same pinned tokenizer vocabulary, but truncation semantics are not interchangeable: CE/SupCon select first 128 + last 382 WordPieces, whereas LIT2026 uses the first 512 tokenizer tokens. Accordingly, token length/truncation prevalence is comparable as an input-length diagnostic; retained token content is method-specific.
""")
    return summary,high,comp


def load_decisions(root, doc):
    params=pd.read_csv(root/"results/calibration/final_statistics/calibration_parameters_and_thresholds.csv")
    params["method"]=params.objective.map({"ce":"CE_ONLY_MATCHED","supcon":"SUPCON"})
    paths={"renklier2026":"renklier2026_test_standardized.csv","tum2024_tr":"tum2024_tr_test_standardized.csv","bostancioglu2026_tr":"bostancioglu2026_tr_full_standardized.csv"}; frames=[]
    for ds,name in paths.items():
        meta=pd.read_csv(root/"data/external_prepared"/name); lf=doc[doc.dataset.eq(ds)][["document_id","length_band","truncated_berturk"]]
        for method in TURKISH_AI_DETECTION_METHODS:
            for seed in [42,314,2026]:
                for fold in range(5):
                    p=pd.read_csv(root/f"artifacts/predictions/external_cross_corpus/preseal/{ds}/{method}/seed_{seed}/outer_{fold}.csv").merge(meta,on=["external_id","text_sha256"],validate="one_to_one").merge(lf,left_on="external_id",right_on="document_id",validate="many_to_one")
                    par=params[(params.method==method)&(params.seed==seed)&(params.outer_fold==fold)].iloc[0]; prob=sigmoid(par.coefficient*p.raw_score_or_logit+par.intercept)
                    for alpha in [.01,.02,.03,.05]:
                        q=p[["external_id","text_sha256","binary_label","source_group_id","length_band","truncated_berturk"]].copy(); q["dataset"]=ds; q["method"]=method; q["seed"]=seed; q["outer_fold"]=fold; q["nominal_fpr"]=alpha; q["prediction"]=(prob>par[f"threshold_alpha_{alpha:.2f}"]).astype(int); frames.append(q)
    return pd.concat(frames,ignore_index=True)


def rate_cluster_ci(g, unit, n_boot, seed):
    units=np.sort(g[unit].astype(str).unique()); idx={u:i for i,u in enumerate(units)}; rng=np.random.default_rng(seed); draws=rng.multinomial(len(units),np.full(len(units),1/len(units)),size=n_boot)
    fprs=[]; fnrs=[]
    for _,r in g.groupby(["seed","outer_fold"]):
        ui=r[unit].astype(str).map(idx).to_numpy(); y=r.binary_label.to_numpy(); p=r.prediction.to_numpy()
        fp=np.bincount(ui,weights=(y==0)&(p==1),minlength=len(units)); hn=np.bincount(ui,weights=y==0,minlength=len(units)); fn=np.bincount(ui,weights=(y==1)&(p==0),minlength=len(units)); an=np.bincount(ui,weights=y==1,minlength=len(units))
        fprs.append(np.divide(draws@fp,draws@hn,out=np.full(n_boot,np.nan),where=(draws@hn)>0)); fnrs.append(np.divide(draws@fn,draws@an,out=np.full(n_boot,np.nan),where=(draws@an)>0))
    return np.nanmean(fprs,axis=0),np.nanmean(fnrs,axis=0),len(units)


def a3(root,out,n_boot=2000):
    d=out/"amended_A3"; d.mkdir(parents=True,exist_ok=True)
    doc=pd.read_csv(root/"results/supplementary_exploratory_2026/02_length_truncation/length_truncation_document_features.csv"); dec=load_decisions(root,doc)
    overall=pd.read_csv(root/"results/supplementary_exploratory_2026/03_error_subgroups/Table_S16_external_errors_subgroup_behavior.csv"); overall.insert(0,"panel","A_overall_frozen_threshold_results")
    rows=[]
    for ds in ["renklier2026","tum2024_tr","bostancioglu2026_tr"]:
        unit="source_group_id" if ds=="renklier2026" else "text_sha256"
        for method in TURKISH_AI_DETECTION_METHODS:
            base=dec[(dec.dataset==ds)&(dec.method==method)&(dec.nominal_fpr==.03)]
            for stype,subgroup,g in [("length_band",sg,base[base.length_band.eq(sg)]) for sg in ["short_le150","medium_151_300","long_gt300"]]+[("truncation","truncated",base[base.truncated_berturk.eq(1)]),("truncation","not_truncated",base[base.truncated_berturk.eq(0)])]:
                docs=g.drop_duplicates("external_id"); h=int((docs.binary_label==0).sum()); a=int((docs.binary_label==1).sum()); fpr=((g.prediction[g.binary_label==0])==1).mean() if h else np.nan; fnr=((g.prediction[g.binary_label==1])==0).mean() if a else np.nan
                fd,nd,nu=rate_cluster_ci(g,unit,n_boot,BOOT_SEED+sum(map(ord,ds+subgroup))) if len(g) else (np.array([np.nan]),np.array([np.nan]),0)
                rows.append({"panel":"B_subgroup_results_at_frozen_0.03","dataset":ds,"method":method,"nominal_fpr":.03,"subgroup_type":stype,"subgroup":subgroup,"human_n":h,"ai_n":a,"total_n":h+a,"unique_resampling_units":nu,"dependent_realizations":g.groupby(["seed","outer_fold"]).ngroups,"fpr":fpr,"fpr_ci_low":np.nanquantile(fd,.025) if h and np.isfinite(fd).any() else np.nan,"fpr_ci_high":np.nanquantile(fd,.975) if h and np.isfinite(fd).any() else np.nan,"fnr":fnr,"fnr_ci_low":np.nanquantile(nd,.025) if a and np.isfinite(nd).any() else np.nan,"fnr_ci_high":np.nanquantile(nd,.975) if a and np.isfinite(nd).any() else np.nan,"estimability":"FPR"+("+FNR" if a else " only; no AI") if h else ("FNR only; no HUMAN" if a else "not estimable")})
    subgroup=pd.DataFrame(rows); save_csv(subgroup,d/"external_error_subgroups_amended.csv")
    table=pd.concat([overall,subgroup],ignore_index=True,sort=False); save_csv(table,d/"Table_S16_amended.csv")
    # Matched disagreement at alpha=.03, separately for HUMAN false positives and AI false negatives.
    q=dec[dec.nominal_fpr.eq(.03)].pivot_table(index=["dataset","external_id","binary_label","seed","outer_fold"],columns="method",values="prediction").reset_index(); dis=[]
    for keys,g in q.groupby(["dataset","binary_label","seed","outer_fold"]):
        ds,y,s,f=keys; truth=int(y); ce=(g.CE_ONLY_MATCHED!=truth); su=(g.SUPCON!=truth)
        dis.append({"dataset":ds,"error_type":"false_positive" if y==0 else "false_negative","seed":s,"outer_fold":f,"shared_error":np.mean(ce&su),"ce_only_error":np.mean(ce&~su),"supcon_only_error":np.mean(~ce&su),"both_correct":np.mean(~ce&~su),"n_documents":len(g)})
    dis=pd.DataFrame(dis); save_csv(dis,d/"matched_error_disagreement_alpha_0.03.csv")
    overall03=overall[overall.nominal_fpr.eq(.03)][["dataset","method","mean_fpr","mean_fnr","mean_fpr_ci_low","mean_fpr_ci_high","mean_fnr_ci_low","mean_fnr_ci_high"]]
    figsource=dis.groupby(["dataset","error_type"],as_index=False)[["shared_error","ce_only_error","supcon_only_error","both_correct","n_documents"]].mean(); save_csv(figsource,d/"Figure_S10_disagreement_source_data.csv"); save_csv(overall03,d/"Figure_S10_rate_source_data.csv")
    fig,axs=plt.subplots(2,3,figsize=(13,7)); dsorder=["renklier2026","tum2024_tr","bostancioglu2026_tr"]
    for ci,ds in enumerate(dsorder):
        r=overall03[overall03.dataset.eq(ds)]; x=np.array([0,1]); width=.34
        for mi,m in enumerate(TURKISH_AI_DETECTION_METHODS):
            z=r[r.method.eq(m)].iloc[0]; y=np.array([z.mean_fpr,z.mean_fnr]); lo=np.array([z.mean_fpr_ci_low,z.mean_fnr_ci_low]); hi=np.array([z.mean_fpr_ci_high,z.mean_fnr_ci_high]); axs[0,ci].bar(x+(mi-.5)*width,y,width,label=m); axs[0,ci].errorbar(x+(mi-.5)*width,y,yerr=np.vstack([y-lo,hi-y]),fmt="none",color="black",capsize=2)
        axs[0,ci].set_xticks(x,["FPR","FNR"]); axs[0,ci].set_title(DATASET_LABELS[ds]+" — frozen α=.03"); axs[0,ci].set_ylim(0,1)
        b=figsource[figsource.dataset.eq(ds)].set_index("error_type").reindex(["false_positive","false_negative"]); bottom=np.zeros(2)
        for col,color,label in [("shared_error","#7b5ea7","Shared"),("ce_only_error","#3568a8","CE-only"),("supcon_only_error","#d05a45","SupCon-only")]: axs[1,ci].bar([0,1],b[col],bottom=bottom,color=color,label=label); bottom+=b[col].to_numpy()
        axs[1,ci].set_xticks([0,1],["HUMAN FP","AI FN"]); axs[1,ci].set_ylabel("Mean document fraction"); axs[1,ci].set_ylim(0,1)
    axs[0,0].legend(fontsize=7); axs[1,0].legend(fontsize=7); fig.suptitle("Figure S10. Corpus-specific errors and matched CE/SupCon disagreement")
    fig.text(.5,.01,"Frozen internal thresholds; three seeds × five dependent folds. Realizations are matched, not independent documents. CALIBRATION_TRANSFER_LIMITED.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.96]); fig.savefig(d/"Figure_S10_amended.svg"); fig.savefig(d/"Figure_S10_amended.png",dpi=300); plt.close(fig)
    return table,subgroup,dis


def a4(root,out):
    d=out/"A4_corrected_sample"; d.mkdir(parents=True,exist_ok=True)
    oldpath=root/"results/supplementary_exploratory_2026/04_paraphrase_robustness/paraphrase_source_manifest.csv"; old=pd.read_csv(oldpath)
    duphash=old.groupby("text_sha256").filter(lambda g:len(g)>1).text_sha256.unique(); duplicate_rows=old[old.text_sha256.isin(duphash)].sort_values(["text_sha256","external_id"])
    keep_drop=[]
    for h,g in duplicate_rows.groupby("text_sha256"):
        ids=sorted(g.external_id); keep_drop.extend(ids[1:])
    v2=old[~old.external_id.isin(keep_drop)].copy(); replacements=[]
    paths={"renklier2026":"renklier2026_test_standardized.csv","tum2024_tr":"tum2024_tr_test_standardized.csv","bostancioglu2026_tr":"bostancioglu2026_tr_full_standardized.csv"}
    # Every dropped row is replaced within the exact original corpus×length stratum.
    doc=pd.read_csv(root/"results/supplementary_exploratory_2026/02_length_truncation/length_truncation_document_features.csv")
    for drop_id in keep_drop:
        dr=old[old.external_id.eq(drop_id)].iloc[0]; meta=pd.read_csv(root/"data/external_prepared"/paths[dr.dataset]); meta=meta[meta.binary_label.eq(1)].merge(doc[doc.dataset.eq(dr.dataset)][["document_id","length_band","word_count"]],left_on="external_id",right_on="document_id",validate="one_to_one")
        eligible=meta[meta.length_band.eq(dr.length_band) & ~meta.external_id.isin(old.external_id) & ~meta.text_sha256.isin(v2.text_sha256)].sort_values(["text_sha256","external_id"])
        if eligible.empty: raise RuntimeError(f"No unique same-stratum replacement for {drop_id}")
        r=eligible.iloc[0]; row={c:"" for c in old.columns}; row.update({"dataset":dr.dataset,"external_id":r.external_id,"text_sha256":r.text_sha256,"length_band":r.length_band,"word_count":r.word_count,"generator":r.generator,"domain":r.domain,"selection_rule":"deterministic same corpus×length replacement; smallest eligible SHA-256; no scores/outcomes","status":"FROZEN_PENDING_GENERATION"}); v2=pd.concat([v2,pd.DataFrame([row])],ignore_index=True); replacements.append({"removed_external_id":drop_id,"duplicate_hash":dr.text_sha256,"replacement_external_id":r.external_id,"replacement_hash":r.text_sha256,"dataset":dr.dataset,"length_band":dr.length_band})
    v2=v2.sort_values(["dataset","length_band","text_sha256","external_id"]).reset_index(drop=True); v2["manifest_version"]="v2_20261012"; v2["immutable_status"]="FROZEN_PENDING_AUTHORIZED_GENERATION"
    if len(v2)!=250 or v2.text_sha256.nunique()!=250: raise RuntimeError("A4 v2 is not 250 unique contents")
    save_csv(v2,d/"paraphrase_source_manifest_v2.csv"); rep=pd.DataFrame(replacements); save_csv(rep,d/"A4_replacements.csv")
    freeze=pd.DataFrame([{"path":"paraphrase_source_manifest_v2.csv","rows":len(v2),"unique_external_ids":v2.external_id.nunique(),"unique_text_sha256":v2.text_sha256.nunique(),"sha256":sha256(d/"paraphrase_source_manifest_v2.csv"),"status":"A4_PENDING_AUTHORIZED_GENERATION"}]); save_csv(freeze,d/"A4_FROZEN_SAMPLE_SHA256.csv")
    write_text(d/"A4_SAMPLE_DEDUPLICATION_REPORT.md",f"""# A4 sample deduplication report

Status: `A4_PENDING_AUTHORIZED_GENERATION`.

The immutable original manifest remains at `{oldpath.relative_to(root).as_posix()}` (SHA-256 `{sha256(oldpath)}`). It contained 250 IDs but 248 unique content hashes. The duplicate group was `{duphash[0]}` with IDs {', '.join(duplicate_rows.external_id)}. The lexicographically first ID was retained; two others were replaced deterministically by the smallest eligible, previously unselected content hashes within the exact TUM2024_TR × long_gt300 stratum. Detector scores, errors, and decisions were not loaded by the selection function.

The v2 manifest contains 250 unique IDs and 250 unique text SHA-256 values. No paraphrases were generated, no API was called, and no model inference occurred.
""")
    return v2,rep,freeze


def preservation(root,out,amend_manifest):
    d=out/"source_preservation"; d.mkdir(parents=True,exist_ok=True); checks=[]
    for _,r in amend_manifest.iterrows():
        p=root/r.relative_path; checks.append({"scope":"prior_exploratory_artifact","path":r.relative_path,"expected_sha256":r.sha256_before,"observed_sha256":sha256(p) if p.exists() else "","match":p.exists() and sha256(p)==r.sha256_before})
    freeze=json.loads((root/"provenance/experiments/external_cross_corpus_v2/PREEXECUTION_FREEZE.json").read_text(encoding="utf-8"))
    for b in freeze.get("bindings",[]):
        p=root/b["path"]; checks.append({"scope":"historical_external_preexecution_binding","path":b["path"],"expected_sha256":b["sha256"],"observed_sha256":sha256(p) if p.exists() else "","match":p.exists() and sha256(p)==b["sha256"]})
    # Current checkpoint bytes versus recorded pre-external/current bindings.
    for objective in ["ce","supcon"]:
        for seed in [42,314,2026]:
            for fold in range(5):
                side=root/f"artifacts/checkpoints/external_cross_corpus/protocol_faithful_rerun/{objective}_seed_{seed}_outer_{fold}/run.json"; j=json.loads(side.read_text(encoding="utf-8")); p=root/j["payload"]["model"]
                checks.append({"scope":"current_terminal_checkpoint_binding","path":p.relative_to(root).as_posix(),"expected_sha256":j["payload"]["model_sha256"],"observed_sha256":sha256(p),"match":sha256(p)==j["payload"]["model_sha256"]})
    lit=json.loads((root/"provenance/experiments/external_cross_corpus_v2/LIT2026_CHECKPOINT_BINDING.json").read_text(encoding="utf-8"))
    for b in lit["bindings"]:
        p=root/b["current_model"]; checks.append({"scope":"current_lit_checkpoint_binding_not_pre_migration_proof","path":b["current_model"],"expected_sha256":b["current_model_sha256"],"observed_sha256":sha256(p),"match":sha256(p)==b["current_model_sha256"]})
    checks=pd.DataFrame(checks); save_csv(checks,d/"SOURCE_PRESERVATION_CHECKS.csv")
    write_text(d/"SOURCE_PRESERVATION_SCOPE_REPORT.md",f"""# Source preservation scope report

1. **Current amendment-run preservation:** {int(checks[checks.scope.eq('prior_exploratory_artifact')].match.sum())}/{sum(checks.scope.eq('prior_exploratory_artifact'))} specifically listed prior exploratory artifacts match their before hashes.
2. **Historical freeze agreement:** {int(checks[checks.scope.eq('historical_external_preexecution_binding')].match.sum())}/{sum(checks.scope.eq('historical_external_preexecution_binding'))} available bindings in the external pre-execution freeze match current bytes. This is limited to the manifest's enumerated files.
3. **Checkpoint byte identity:** all 30 current terminal CE/SupCon payloads and 15 current LIT2026 payloads were compared to their recorded current bindings. The LIT binding explicitly states that current hashes are **not proof of pre-migration historical byte identity**; that stronger claim is not made.
4. **Complete scientific reproducibility:** `NOT_INDEPENDENTLY_VERIFIABLE` from this amendment alone. Training was not repeated, the repository has no Git metadata, and the review ZIP intentionally omits checkpoints and source datasets.

Accordingly, this report does not claim that every historical project file is unchanged. It makes only the scoped claims backed by `SOURCE_PRESERVATION_CHECKS.csv`.
""")
    return checks


def validations(root,out,a1r,a2table,a3table,a4v2,preserve,amend_manifest):
    d=out/"validation"; d.mkdir(parents=True,exist_ok=True); tests=[]
    def add(name,status,evidence): tests.append({"check":name,"status":status,"evidence":evidence})
    # Actual tokenizer/config/implementation verification.
    cfg=json.loads((root/"configs/primary_evaluation.json").read_text(encoding="utf-8")); bert=(root/"src/turkish_ai_detection/experiments/baseline/berturk.py").read_text(encoding="utf-8"); lit=(root/"src/turkish_ai_detection/experiments/literature_comparison/run_fit.py").read_text(encoding="utf-8")
    tok_ok=cfg["backbone"]["revision"]=="b6e1de16c983e0f2c70664591ea3f22810072608" and "inspected_ids[:HEAD_WORDPIECES] + inspected_ids[-TAIL_WORDPIECES:]" in bert and "MAX_MODEL_TOKENS = 512" in bert
    lit_ok="max_length=512" in lit.replace(" ","") or ("max_length" in lit and "512" in lit)
    add("tokenizer_truncation_use","PASS" if tok_ok and lit_ok else "FAIL",f"config revision/head-tail source verified={tok_ok}; LIT first-512 source verified={lit_ok}")
    # External label separation: verify every preseal header and parameter fit counters plus frozen contract.
    pred=list((root/"artifacts/predictions/external_cross_corpus/preseal").rglob("*.csv")); forbidden={"label","binary_label","domain","generator","generation_type"}; headers_ok=all(not (forbidden & set(pd.read_csv(p,nrows=0).columns)) for p in pred)
    pars=pd.read_csv(root/"results/supplementary_exploratory_2026/../calibration/final_statistics/calibration_parameters_and_thresholds.csv") if False else pd.read_csv(root/"results/calibration/final_statistics/calibration_parameters_and_thresholds.csv")
    param_ok=pars.outer_test_rows_used_for_calibration.eq(0).all() and pars.outer_test_rows_used_for_threshold.eq(0).all()
    contract=json.loads((root/"configs/external_cross_corpus.json").read_text(encoding="utf-8")); contract_ok=contract["evaluation"]["prediction_first_label_later"] and contract["evaluation"]["external_adaptation"] is False
    add("external_label_separation","PASS" if headers_ok and param_ok and contract_ok else "FAIL",f"{len(pred)} preseal files exclude label/subgroup columns; calibration counters zero={param_ok}; frozen prediction-first contract={contract_ok}. Scope is stored computation path, not an omniscient process claim.")
    # Controlled cluster fixture: rows of a cluster must have identical multiplicity; methods reuse draws.
    units=np.array(["A","A","B","C","C"]); _,ri,draw=cluster_draws(units,50,123); weights=draw[:,ri]; fixture_ok=np.all(weights[:,0]==weights[:,1]) and np.all(weights[:,3]==weights[:,4]) and np.array_equal(draw,cluster_draws(units,50,123)[2])
    add("bootstrap_resampling_units","PASS" if fixture_ok else "FAIL","controlled fixture keeps all members of A/C together and reproduces identical matched draws")
    # Publication values: selected recomputation from raw source and exact source-data hashes/specs.
    orig=pd.read_csv(root/"results/supplementary_exploratory_2026/01_corpus_shortcuts/surface_probe_oof_predictions.csv"); auc=roc_auc_score(orig.binary_label,orig.surface_probe_ai_score); t14=pd.read_csv(out/"amended_A1/Table_S14_amended.csv"); recorded=float(t14[(t14.panel=="B_surface_probe_diagnostic_performance")&(t14.input_representation.fillna("").str.startswith("Source"))&(t14.domain=="ALL")].roc_auc.iloc[0]); table_ok=abs(auc-recorded)<1e-12 and len(a2table)==55 and len(a3table)>0
    fig_ok=all((out/p).exists() and (out/p).stat().st_size>0 for p in ["amended_A1/Figure_S8_source_data.csv","amended_A2/Figure_S9_source_data.csv","amended_A3/Figure_S10_rate_source_data.csv","amended_A3/Figure_S10_disagreement_source_data.csv"])
    add("table_numeric_consistency","PASS" if table_ok else "FAIL",f"A1 raw-prediction ROC-AUC={auc:.12f} equals S14; S15 rows={len(a2table)}; S16 generated from saved panels")
    add("figure_source_consistency","PASS" if fig_ok else "FAIL","each amended figure has saved machine-readable plotted series; generator reads those same frames in memory")
    a4_ok=len(a4v2)==250 and a4v2.external_id.nunique()==250 and a4v2.text_sha256.nunique()==250
    add("paraphrase_unique_content_grouping","PASS" if a4_ok else "FAIL",f"rows/IDs/content hashes={len(a4v2)}/{a4v2.external_id.nunique()}/{a4v2.text_sha256.nunique()}; no generation columns populated")
    old_ok=preserve[preserve.scope.eq("prior_exploratory_artifact")].match.all(); freeze_ok=preserve[preserve.scope.eq("historical_external_preexecution_binding")].match.all(); ck_ok=preserve[preserve.scope.str.contains("checkpoint_binding")].match.all()
    add("prior_artifact_preservation","PASS" if old_ok else "FAIL","only amendment-manifest prior files asserted")
    add("historical_freeze_manifest_agreement","PASS" if freeze_ok else "FAIL","enumerated available external pre-execution bindings only")
    add("current_checkpoint_byte_bindings","PASS" if ck_ok else "FAIL","30 CE/SupCon + 15 LIT current payload bindings; not proof of LIT pre-migration bytes")
    # Immutable calibration/threshold input compared with the prior package source manifest.
    source_manifest=pd.read_csv(root/"results/supplementary_exploratory_2026/00_protocol_and_manifest/SOURCE_SHA256_MANIFEST.csv"); cal="results/calibration/final_statistics/calibration_parameters_and_thresholds.csv"; expected=source_manifest[source_manifest.relative_path.eq(cal)].sha256.iloc[0]; cal_ok=sha256(root/cal)==expected
    add("frozen_calibration_threshold_preservation","PASS" if cal_ok else "FAIL",f"current SHA-256 equals original exploratory input manifest: {cal_ok}")
    # Static prohibition checks on amendment AST/source.
    src=Path(__file__).read_text(encoding="utf-8"); forbidden_tokens=["Auto"+"Model","Trainer"+"(","requests"+"."+"post(","openai"+"."]; prohibited=not any(x in src for x in forbidden_tokens)
    add("no_training_generation_gpu_inference","PASS" if prohibited else "FAIL","static amendment-source scan plus output scope; no model loading/forward path")
    status="PASS" if all(t["status"]=="PASS" for t in tests) else "FAILED_VALIDATION"
    write_json(d/"FINAL_REVALIDATION_RESULTS.json",{"status":status,"analysis_class":STATUS,"tests":tests,"manual_or_scope_limits":["Complete historical byte equivalence: NOT_INDEPENDENTLY_VERIFIABLE","Causal shortcut reliance: NOT_INDEPENDENTLY_VERIFIABLE","A1–A3 publication integration: MANUAL_REVIEW_REQUIRED"],"analysis_statuses":{"A1":"A1_POSTHOC_AMENDED","A2":"A2_POSTHOC_AMENDED","A3":"A3_POSTHOC_AMENDED","A4":"A4_PENDING_AUTHORIZED_GENERATION"}})
    write_text(d/"FINAL_REVALIDATION_REPORT.md","# Final revalidation report\n\nOverall automated result: `"+status+"`.\n\n"+"\n".join(f"- {t['status']} — `{t['check']}`: {t['evidence']}" for t in tests)+"\n\nManual review remains required for scientific interpretation and publication integration. Complete historical equivalence is not independently verifiable from the available repository alone.")
    write_json(d/"BOOTSTRAP_FIXTURE_LOG.json",{"units":units.tolist(),"first_five_draws":draw[:5].tolist(),"row_weights_first_five":weights[:5].tolist(),"passed":bool(fixture_ok)})
    return status,tests


def package(root,out,package,amend_manifest,status):
    if package.exists(): shutil.rmtree(package)
    mapping={"amended_A1":"amended_A1","amended_A2":"amended_A2","amended_A3":"amended_A3","A4_corrected_sample":"A4_corrected_sample","validation":"validation","source_preservation":"source_preservation","00_amendment_manifest":"audit_response"}
    for src,dst in mapping.items(): shutil.copytree(out/src,package/dst)
    if (out/"audit_response").is_dir():
        for p in (out/"audit_response").glob("*"):
            if p.is_file(): shutil.copy2(p,package/"audit_response"/p.name)
    for n in ["publication_tables","publication_figures","reproducibility","final_scientific_report"]: (package/n).mkdir(parents=True,exist_ok=True)
    for p in out.rglob("Table_S*_amended.csv"): shutil.copy2(p,package/"publication_tables"/p.name)
    for p in out.rglob("Figure_S*_amended.svg"): shutil.copy2(p,package/"publication_figures"/p.name)
    shutil.copy2(Path(__file__),package/"reproducibility"/Path(__file__).name)
    test_path=root/"tests/unit/test_exploratory_amendment.py"
    if test_path.is_file(): shutil.copy2(test_path,package/"reproducibility"/test_path.name)
    smoke_path=root/f"results/supplementary_exploratory_2026/amendment_{VERSION}_smoke/SMOKE_TEST_RESULT.json"
    if smoke_path.is_file(): shutil.copy2(smoke_path,package/"reproducibility"/smoke_path.name)
    write_text(package/"reproducibility/REPLAY.md",f"""# Replay instructions

Python 3.13 with pandas, NumPy, scikit-learn, matplotlib, and transformers is required. The pinned BERTurk tokenizer revision must already be available locally. The ZIP alone is insufficient: original datasets, predictions, configurations, and provenance bindings must exist in an explicit Turkish AI-Generated Text Detection Study project root.

Full amendment (CPU statistical work only; no model inference):

`python supplementary_exploratory_validation.py --project-root "<PROJECT_ROOT>" --output-dir "results/supplementary_exploratory_2026/amendment_20261012_replay" --bootstrap-replicates 2000`

Safe entry-point smoke test:

`python supplementary_exploratory_validation.py --project-root "<PROJECT_ROOT>" --output-dir "<TEMP_OUTPUT_DIR>" --smoke-test`

The command refuses a root missing canonical inputs. Smoke mode reads schemas/configuration, tests helper invariants, writes only its explicit output directory, and performs no bootstrap, training, generation, model loading, GPU inference, or mutation of completed results.
""")
    final="""# Final scientific amendment report

- `A1_POSTHOC_AMENDED`: source text is non-equivalent to Condition-A-normalized text; original A1 is preserved and relabeled, with a separate normalized sensitivity.
- `A2_POSTHOC_AMENDED`: external display is corpus-specific; sparse-cell counts/estimability are explicit; a 2,000-replicate CE/SupCon interval sensitivity is supplied alongside the original 500-replicate intervals.
- `A3_POSTHOC_AMENDED`: Table S16 now has separate overall and subgroup panels; Figure S10 shows corpus-specific FPR/FNR and matched shared/CE-only/SupCon-only errors at frozen alpha=.03.
- `A4_PENDING_AUTHORIZED_GENERATION`: the v2 selection has 250 unique content hashes. No paraphrasing or inference occurred.

No primary result, checkpoint, calibration mapping, or threshold was changed. No primary model was retrained. `CALIBRATION_TRANSFER_LIMITED` remains the operative conclusion. The amendment analyses remain exploratory and do not replace the original model evaluations.
"""
    write_text(package/"final_scientific_report/FINAL_SCIENTIFIC_AMENDMENT_REPORT.md",final)
    write_text(package/"README.md",f"# Turkish AI-Generated Text Detection Study amended exploratory review ({VERSION})\n\nAutomated validation status: `{status}`. A1–A3 are exploratory analyses; no A4 paraphrase generation or inference was performed. See `reproducibility/REPLAY.md` for the explicit-root entry point.")
    # Package manifest then ZIP (manifest excludes itself by definition).
    rows=[]
    for p in sorted(package.rglob("*")):
        if p.is_file(): rows.append({"relative_path":p.relative_to(package).as_posix(),"bytes":p.stat().st_size,"sha256":sha256(p)})
    save_csv(pd.DataFrame(rows),package/"validation/PACKAGE_SHA256_MANIFEST.csv")
    zp=root/f"Turkish_AI_Text_Detection_Exploratory_Amended_Review_{VERSION}.zip"
    if zp.exists(): zp.unlink()
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob("*")):
            if p.is_file(): z.write(p,p.relative_to(root))
    with zipfile.ZipFile(zp) as z: bad=z.testzip()
    write_text(root/f"{zp.name}.sha256",f"{sha256(zp)}  {zp.name}")
    return zp,bad


def changed_manifest(root,out):
    rows=[]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name!="CHANGED_FILES_SHA256.csv": rows.append({"relative_path":p.relative_to(out).as_posix(),"bytes":p.stat().st_size,"sha256_after":sha256(p),"change_type":"new_versioned_amendment_artifact"})
    for p in [Path(__file__),root/"tests/unit/test_exploratory_amendment.py"]:
        if p.is_file(): rows.append({"relative_path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256_after":sha256(p),"change_type":"new_amendment_code_or_test"})
    save_csv(pd.DataFrame(rows),out/"validation/CHANGED_FILES_SHA256.csv")


def reports(out,a1r,a2comp,a4rep):
    write_text(out/"audit_response/TARGETED_AMENDMENT_REPORT.md",f"""# Targeted amendment report

Condition-A text equivalence: `{a1r['assessment']}` ({a1r['changed_texts']} changed strings). The supplementary A1–A3 tables and figures were regenerated with explicit source and subgroup definitions. A2 includes a 2,000-replicate uncertainty sensitivity analysis. Input and result validation covers source/configuration consistency, fixtures, numerical checks and hashes. The A4 candidate manifest contains {len(a4rep)} revised duplicate identifiers and remains unexecuted; no robustness inference is reported.
""")
    write_text(out/"audit_response/BEFORE_AFTER_COMPARISON.md","""# Before/after comparison

| Area | Before | Amendment | Change class |
|---|---|---|---|
| A1 input | Source text not distinguished from Condition A | Executable 3,000-row normalization audit; separate normalized probe | New diagnostic statistic; original preserved |
| Table S14 | Feature effects only | Feature-effects and probe-performance panels | Presentation plus existing metrics |
| Figure S8 | Domains pooled | ABSTRACT/NEWSPAPER panels; mixed-probe label | Presentation only |
| A2 uncertainty | 500 replicates | Original retained plus 2,000-replicate CE/SupCon sensitivity | Recomputed uncertainty sensitivity |
| Table/Figure S15/S9 | Sparse metadata; external corpora averaged in figure | Counts/units/estimability and four corpus-specific panels | Presentation plus CI sensitivity |
| Table/Figure S16/S10 | Overall thresholds; FPR-focused plot | Overall + subgroup panels; FPR/FNR + matched disagreement | New subgroup uncertainty and presentation |
| Validation | Several existence/hard-coded checks | Executable source/config/header/fixture/numeric/hash checks | Validation methodology |
| Preservation | Broad within-run wording | Current-run, freeze, checkpoint, reproducibility scopes separated | Claim correction |
| Replay | Script-relative root | Mandatory explicit root/output; smoke test | Reproducibility correction |
| A4 | 250 IDs / 248 hashes | 250 IDs / 250 hashes, versioned freeze | Deterministic sample correction |

No primary discrimination, calibration, or threshold statistic was replaced.
""")
    write_text(out/"audit_response/REMAINING_LIMITATIONS.md","""# Remaining limitations

- A1 surface separability is associative and cannot prove causal BERTurk reliance.
- A1 subgroup scores still evaluate one mixed-domain probe, not separately trained domain models.
- A2 sparse/single-class cells remain non-estimable; 2,000-replicate sensitivity reduces but does not eliminate Monte Carlo error.
- External folds are dependent realizations of the same corpora; intervals condition on the stored model grid.
- LIT2026 and CE/SupCon share tokenizer vocabulary but retain different token positions after truncation.
- Stored-path audits support external-label separation but cannot prove facts about unrecorded activity; scope is explicit.
- Current LIT checkpoint hashes do not prove pre-migration historical byte identity.
- Complete training reproducibility was not retested and is `NOT_INDEPENDENTLY_VERIFIABLE` here.
- A4 has no robustness results until separately authorized generation, QC, and frozen inference occur.
""")


def smoke(root,out):
    out.mkdir(parents=True,exist_ok=True); study=pd.read_csv(root/"data/study/paired_study.csv",nrows=3); cfg=json.loads((root/"configs/primary_evaluation.json").read_text(encoding="utf-8")); units=np.array(["A","A","B"]); _,ri,draw=cluster_draws(units,5,7)
    result={"status":"PASS","project_root":str(root),"rows_read":len(study),"backbone":cfg["backbone"]["identifier"],"cluster_members_together":bool(np.all(draw[:,ri][:,0]==draw[:,ri][:,1])),"training":False,"generation":False,"gpu_inference":False}
    write_json(out/"SMOKE_TEST_RESULT.json",result); print(json.dumps(result,indent=2)); return


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",required=True); ap.add_argument("--output-dir",default=f"results/supplementary_exploratory_2026/amendment_{VERSION}"); ap.add_argument("--package-dir",default=f"Turkish_AI_Text_Detection_Exploratory_Amended_Review_{VERSION}"); ap.add_argument("--bootstrap-replicates",type=int,default=2000); ap.add_argument("--smoke-test",action="store_true"); ap.add_argument("--package-only",action="store_true"); args=ap.parse_args()
    root=Path(args.project_root).resolve(); require_project(root); out=Path(args.output_dir); out=out if out.is_absolute() else root/out
    if args.smoke_test: return smoke(root,out)
    if args.package_only:
        result=json.loads((out/"validation/FINAL_REVALIDATION_RESULTS.json").read_text(encoding="utf-8")); manifest=pd.read_csv(out/"00_amendment_manifest/AMENDMENT_MANIFEST.csv"); changed_manifest(root,out); package_dir=Path(args.package_dir); package_dir=package_dir if package_dir.is_absolute() else root/package_dir; zp,bad=package(root,out,package_dir,manifest,result["status"]); print(json.dumps({"validation":result["status"],"zip":str(zp),"zip_test_bad_file":bad,"sha256":sha256(zp),"mode":"package_only"},indent=2)); return
    if args.bootstrap_replicates<2000: raise ValueError("amendment sensitivity requires at least 2,000 replicates")
    sys.path.insert(0,str(root/"src")); from turkish_ai_detection.features.core import normalize_text
    tok=AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased",revision="b6e1de16c983e0f2c70664591ea3f22810072608",local_files_only=True)
    manifest=pd.read_csv(out/"00_amendment_manifest/AMENDMENT_MANIFEST.csv")
    a1r=a1(root,out,tok,normalize_text,args.bootstrap_replicates)
    a2table,a2high,a2comp=a2(root,out,args.bootstrap_replicates)
    a3table,a3sub,a3dis=a3(root,out,args.bootstrap_replicates)
    a4v2,a4rep,a4freeze=a4(root,out)
    preserve=preservation(root,out,manifest)
    status,tests=validations(root,out,a1r,a2table,a3table,a4v2,preserve,manifest)
    reports(out,a1r,a2comp,a4rep); changed_manifest(root,out)
    package_dir=Path(args.package_dir); package_dir=package_dir if package_dir.is_absolute() else root/package_dir
    zp,bad=package(root,out,package_dir,manifest,status)
    print(json.dumps({"validation":status,"zip":str(zp),"zip_test_bad_file":bad,"sha256":sha256(zp),"statuses":{"A1":"A1_POSTHOC_AMENDED","A2":"A2_POSTHOC_AMENDED","A3":"A3_POSTHOC_AMENDED","A4":"A4_PENDING_AUTHORIZED_GENERATION"}},indent=2))


if __name__=="__main__": main()
