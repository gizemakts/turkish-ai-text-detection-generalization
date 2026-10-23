#!/usr/bin/env python
"""Manual, resumable inference over the verifier-authorized 30-cell grid."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, unicodedata, re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
SEEDS=(42,314,2026); FOLDS=range(5); METHODS=("CE_ONLY_MATCHED","SUPCON")
def canon(x): return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",str(x))).strip().casefold()
def thash(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_existing(target,side,d,dataset_hash,manifest_hash,method,seed,fold):
 if not target.exists() or not side.exists(): raise RuntimeError(f"Partial cell requires manual review; nothing overwritten: {target}")
 old=json.loads(side.read_text()); q=pd.read_csv(target)
 required={"text_id","normalized_text_sha256","binary_label","method","seed","outer_fold","raw_logit_score"}
 identity=(old.get("method")==method and int(old.get("seed",-1))==seed and int(old.get("outer_fold",-1))==fold and q.method.nunique()==1 and q.method.iloc[0]==method and q.seed.nunique()==1 and int(q.seed.iloc[0])==seed and q.outer_fold.nunique()==1 and int(q.outer_fold.iloc[0])==fold)
 content=q[["text_id","normalized_text_sha256","binary_label"]].reset_index(drop=True).equals(d[["text_id","normalized_text_sha256","binary_label"]].reset_index(drop=True)) if required<=set(q) else False
 valid=(old.get("status")=="COMPLETE" and old.get("dataset_sha256")==dataset_hash and old.get("eligible_manifest_sha256")==manifest_hash and len(q)==len(d) and sha(target)==old.get("prediction_sha256") and identity and content and np.isfinite(q.raw_logit_score).all())
 if not valid: raise RuntimeError(f"Invalid existing cell; nothing overwritten: {target}")
 return q
def main():
 ap=argparse.ArgumentParser(); here=Path(__file__).resolve(); ap.add_argument("--project-root",type=Path,default=here.parents[4]); ap.add_argument("--dataset",type=Path,required=True,help="Path to an authorized local Meric2 copy; the dataset is not distributed here"); ap.add_argument("--output-dir",type=Path,default=here.parents[1]/"predictions" ); ap.add_argument("--smoke-cells",type=int,default=0,help="0 means full authorized 30-cell grid; otherwise run first N cells"); ap.add_argument("--validate-only",action="store_true"); a=ap.parse_args(); root=a.project_root.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
 audit=pd.read_csv(here.parents[1]/"CHECKPOINT_ELIGIBILITY_AUDIT.csv");
 audit=audit[audit.seed.astype(int).isin(SEEDS)]
 if len(audit)!=30 or not audit.eligible.astype(str).str.lower().eq("true").all(): raise RuntimeError("Eligibility audit has not passed for all 30 verifier-authorized cells")
 source=pd.read_csv(a.dataset).reset_index(names="original_row_index"); manifest=pd.read_csv(here.parents[1]/"MERIC2_ELIGIBLE_MANIFEST.csv"); d=manifest.merge(source[["original_row_index","review"]],on="original_row_index",validate="one_to_one")
 if not (d.apply(lambda r:thash(r.review)==r.normalized_text_sha256,axis=1)).all(): raise RuntimeError("Source/manifest text hash mismatch")
 cells=[(m,s,f) for m in METHODS for s in SEEDS for f in FOLDS]; cells=cells[:a.smoke_cells or None]
 dataset_hash=sha(a.dataset); manifest_hash=sha(here.parents[1]/"MERIC2_ELIGIBLE_MANIFEST.csv")
 if a.validate_only:
  complete=missing=0
  for method,seed,fold in cells:
   target=out/method/f"seed_{seed}"/f"outer_{fold}.csv"; side=target.with_suffix(".json")
   if not target.exists() and not side.exists(): missing+=1; continue
   validate_existing(target,side,d,dataset_hash,manifest_hash,method,seed,fold); complete+=1
  print(f"PASS_TRANSFORMER_INPUT_AND_EXISTING_CELL_VALIDATION rows={len(d)} authorized_cells={len(cells)} complete_valid={complete} missing={missing}"); return
 sys.path.insert(0,str(root/"src")); from turkish_ai_detection.experiments.external_cross_corpus import inference as inf
 for method,seed,fold in cells:
  target=out/method/f"seed_{seed}"/f"outer_{fold}.csv"; side=target.with_suffix(".json"); target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() or side.exists():
   validate_existing(target,side,d,dataset_hash,manifest_hash,method,seed,fold)
   print(f"SKIP COMPLETE {method} {seed} {fold}"); continue
  model,tok,ck=inf.load_ce_supcon(method,seed,fold); scores=inf.score_ce_supcon(model,tok,d.review.astype(str).tolist()); del model
  q=d[["text_id","normalized_text_sha256","binary_label"]].copy(); q["method"]=method; q["seed"]=seed; q["outer_fold"]=fold; q["raw_logit_score"]=scores
  tmp=target.with_suffix(".csv.tmp"); q.to_csv(tmp,index=False); os.replace(tmp,target)
  meta={"status":"COMPLETE","created_utc":datetime.now(timezone.utc).isoformat(),"dataset_sha256":dataset_hash,"eligible_manifest_sha256":manifest_hash,"authorized_seed_grid":list(SEEDS),"method":method,"seed":seed,"outer_fold":fold,"rows":len(q),"checkpoint":ck,"prediction_sha256":sha(target),"meric2_used_for_fit_tune_or_selection":False}
  st=side.with_suffix(".json.tmp"); st.write_text(json.dumps(meta,indent=2),encoding="utf-8"); os.replace(st,side); print(f"COMPLETE {method} {seed} {fold}")
if __name__=="__main__": main()
