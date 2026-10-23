#!/usr/bin/env python
"""Protocol-faithful M4 TF-IDF baseline evaluated on external corpora."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from turkish_ai_detection.features.core import normalize_text, turkish_lower

C_GRID=(.01,.03,.1,.3,1.,3.,10.)
EXTERNAL={"Renklier2026":"renklier2026_test_standardized.csv","TUM2024_TR":"tum2024_tr_test_standardized.csv","Bostancioglu2026_TR":"bostancioglu2026_tr_full_standardized.csv"}
def clean(x): return turkish_lower(normalize_text(str(x)))
def vectors():
    return (TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,norm="l2",lowercase=False),TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=2,sublinear_tf=True,norm="l2",lowercase=False))
def fit_transform(train, other):
    w,c=vectors(); return sparse.hstack([w.fit_transform(train),c.fit_transform(train)],format="csr"),sparse.hstack([w.transform(other),c.transform(other)],format="csr"),(w,c)
def transform(v, texts): return sparse.hstack([v[0].transform(texts),v[1].transform(texts)],format="csr")
def pauc(y,s): return roc_auc_score(y,s,max_fpr=.03)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description="M4 recipe: word 1-2 + char_wb 2-5, min_df=2, sublinear TF, L2 liblinear; inner-4-fold C tuning.")
    ap.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2]); ap.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parents[1]/"results"/"tfidf_external"); ap.add_argument("--execute",action="store_true"); ap.add_argument("--save-predictions",action="store_true")
    a=ap.parse_args(); root=a.project_root.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    study_p=root/"data/study/paired_study.csv"; folds_p=root/"data/study/nested_folds.csv"
    ext_paths={k:root/"data/external_prepared"/v for k,v in EXTERNAL.items()}
    for p in [study_p,folds_p,*ext_paths.values()]:
        if not p.exists(): raise FileNotFoundError(p)
    study=pd.read_csv(study_p); folds=pd.read_csv(folds_p)
    req={"pair_id","text","label"};
    if not req<=set(study): raise ValueError(f"Study missing {req-set(study)}")
    study=study.merge(folds,on="pair_id",validate="many_to_one")
    if len(study)!=3000 or study.pair_id.nunique()!=1500: raise ValueError("Unexpected Combined-v2 population")
    if not a.execute:
        print("VALIDATED ONLY: inputs and 1,500 pair-disjoint outer assignments found. Add --execute to fit M4."); return
    study["clean_text"]=study.text.map(clean); study["y"]=(study.label.astype(str).str.upper()=="AI").astype(int)
    external={}
    for name,p in ext_paths.items():
        d=pd.read_csv(p)
        if not {"external_id","text","binary_label"}<=set(d): raise ValueError(f"Bad external schema: {p}")
        d["clean_text"]=d.text.map(clean); external[name]=d
    fold_results=[]; pred_rows=[]; tuning=[]
    for outer in range(5):
        tr=study[study.outer_fold!=outer].copy()
        if tr.pair_id.nunique()*2!=len(tr): raise ValueError("Pair integrity failure")
        inner_col=f"inner_fold_outer{outer}"
        inner_oof={c:np.full(len(tr),np.nan) for c in C_GRID}
        for inner in range(4):
            va=tr[inner_col].eq(inner).to_numpy(); itr=tr.loc[~va]; iva=tr.loc[va]
            Xtr,Xva,_=fit_transform(itr.clean_text,iva.clean_text)
            for c in C_GRID:
                m=LogisticRegression(C=c,penalty="l2",solver="liblinear",max_iter=5000,random_state=42).fit(Xtr,itr.y)
                inner_oof[c][va]=m.decision_function(Xva)
        scores={c:pauc(tr.y,values) for c,values in inner_oof.items()}; best=min(C_GRID,key=lambda c:(-scores[c],c))
        tuning += [dict(outer_fold=outer,C=c,inner_oof_std_pauc=scores[c],selected=(c==best)) for c in C_GRID]
        Xtr,_,vec=fit_transform(tr.clean_text,[tr.clean_text.iloc[0]])
        model=LogisticRegression(C=best,penalty="l2",solver="liblinear",max_iter=5000,random_state=42).fit(Xtr,tr.y)
        for corpus,d in external.items():
            score=model.decision_function(transform(vec,d.clean_text)); y=d.binary_label.astype(int).to_numpy()
            fold_results.append(dict(corpus=corpus,outer_fold=outer,selected_C=best,n_train=len(tr),n_external=len(d),roc_auc=roc_auc_score(y,score),average_precision=average_precision_score(y,score),standardized_partial_roc_auc_max_fpr_003=pauc(y,score)))
            if a.save_predictions:
                pred_rows.extend(dict(corpus=corpus,outer_fold=outer,external_id=i,raw_score=float(s)) for i,s in zip(d.external_id,score))
    fr=pd.DataFrame(fold_results); fr.to_csv(out/"tfidf_external_fold_metrics.csv",index=False,float_format="%.12f")
    metrics=["roc_auc","average_precision","standardized_partial_roc_auc_max_fpr_003"]
    agg=fr.groupby("corpus")[metrics].agg(["mean","std","min","max"]); agg.columns=[f"{x}_{y}" for x,y in agg.columns]; agg.reset_index().to_csv(out/"tfidf_external_summary.csv",index=False,float_format="%.12f")
    pd.DataFrame(tuning).to_csv(out/"tfidf_inner_tuning.csv",index=False,float_format="%.12f")
    if pred_rows: pd.DataFrame(pred_rows).to_csv(out/"tfidf_external_predictions.csv",index=False)
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"analysis_label":"exploratory; external corpora not used for tuning","recipe":"Exact documented M4 Condition-A recipe","external_fit_count":0,"aggregation":"metrics computed per independently fitted outer model; no score pooling","training_population":"Combined-v2 rows whose pair-disjoint outer_fold differs from fitted fold","script_sha256":sha(Path(__file__)),"inputs":[{"path":str(p.relative_to(root)),"sha256":sha(p)} for p in [study_p,folds_p,*ext_paths.values()]]}
    (out/"provenance.json").write_text(json.dumps(prov,indent=2),encoding="utf-8"); print(f"PASS: {out}")
if __name__=="__main__": main()
