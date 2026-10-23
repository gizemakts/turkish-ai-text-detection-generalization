#!/usr/bin/env python
"""Re-fit finalized Combined-v2-only M4 outer models and score Meric2."""
from __future__ import annotations
import argparse, hashlib, json, re, sys, unicodedata
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x): return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",str(x))).strip().casefold()
def thash(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def vecs(): return TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,norm="l2",lowercase=False),TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=2,sublinear_tf=True,norm="l2",lowercase=False)
def main():
 ap=argparse.ArgumentParser(); here=Path(__file__).resolve(); ap.add_argument("--project-root",type=Path,default=here.parents[4]); ap.add_argument("--dataset",type=Path,required=True,help="Path to an authorized local Meric2 copy; the dataset is not distributed here"); ap.add_argument("--output",type=Path,default=here.parents[1]/"predictions/TFIDF_M4_predictions.csv"); ap.add_argument("--validate-only",action="store_true"); a=ap.parse_args(); root=a.project_root.resolve(); a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists(): raise FileExistsError(f"Refusing overwrite: {a.output}")
 prov=json.loads((root/"supplementary_analyses/results/tfidf_external/provenance.json").read_text());
 if prov.get("external_fit_count")!=0 or prov.get("recipe")!="Exact documented M4 Condition-A recipe": raise RuntimeError("Task 2 provenance mismatch")
 tuning=pd.read_csv(root/"supplementary_analyses/results/tfidf_external/tfidf_inner_tuning.csv"); selected=tuning[tuning.selected.astype(str).str.lower().eq("true")].set_index("outer_fold").C.to_dict()
 if set(selected)!=set(range(5)): raise RuntimeError("Missing finalized Task 2 selected C")
 sys.path.insert(0,str(root/"src")); from turkish_ai_detection.features.core import normalize_text,turkish_lower
 study=pd.read_csv(root/"data/study/paired_study.csv").merge(pd.read_csv(root/"data/study/nested_folds.csv")[["pair_id","outer_fold"]],on="pair_id",validate="many_to_one"); study["model_text"]=study.text.map(normalize_text).map(turkish_lower); study["y"]=(study.label.str.upper()=="AI").astype(int)
 source=pd.read_csv(a.dataset).reset_index(names="original_row_index"); manifest=pd.read_csv(here.parents[1]/"MERIC2_ELIGIBLE_MANIFEST.csv"); ext=manifest.merge(source[["original_row_index","review"]],on="original_row_index",validate="one_to_one");
 if not ext.apply(lambda r:thash(r.review)==r.normalized_text_sha256,axis=1).all(): raise RuntimeError("Source/manifest mismatch")
 if a.validate_only: print(f"PASS_TFIDF_INPUT_VALIDATION rows={len(ext)} selected_C={selected}"); return
 ext["model_text"]=ext.review.map(normalize_text).map(turkish_lower); rows=[]
 for fold in range(5):
  tr=study[study.outer_fold!=fold]; w,c=vecs(); X=sparse.hstack([w.fit_transform(tr.model_text),c.fit_transform(tr.model_text)],format="csr"); Xe=sparse.hstack([w.transform(ext.model_text),c.transform(ext.model_text)],format="csr"); model=LogisticRegression(C=float(selected[fold]),penalty="l2",solver="liblinear",max_iter=5000,random_state=42).fit(X,tr.y); score=model.decision_function(Xe)
  q=ext[["text_id","normalized_text_sha256","binary_label"]].copy(); q["method"]="TFIDF_M4"; q["seed"]=""; q["outer_fold"]=fold; q["raw_logit_score"]=score; rows.append(q); print(f"COMPLETE TFIDF outer {fold}")
 pd.concat(rows).to_csv(a.output,index=False)
 meta={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"COMPLETE_COMBINED_V2_ONLY_TFIDF","rows":sum(map(len,rows)),"meric2_used_for_fit_tune_or_selection":False,"selected_C":selected,"tfidf_external_provenance_sha256":sha(root/"supplementary_analyses/results/tfidf_external/provenance.json"),"tfidf_tuning_sha256":sha(root/"supplementary_analyses/results/tfidf_external/tfidf_inner_tuning.csv"),"prediction_sha256":sha(a.output)}; a.output.with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print("COMPLETE_TFIDF")
if __name__=="__main__": main()
