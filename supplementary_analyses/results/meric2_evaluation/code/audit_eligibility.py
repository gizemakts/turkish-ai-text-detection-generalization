#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

EXPECTED_MERIC="e9ab81061c4cb388889cb37b2e9dda3d30928a083a3227ad8641cc7baad97ec3"
SEEDS=(42,314,2026); FOLDS=range(5); OBJECTIVES=("ce","supcon")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def canon(x): return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",str(x))).strip().casefold()
def thash(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def strata(n):
 return pd.cut(n,[-np.inf,25,32,40,48,60,np.inf],labels=["le25","26_32","33_40","41_48","49_60","gt60"])
def features(text):
 t=str(text); tok=re.findall(r"\b\w+\b",t,flags=re.UNICODE); n=max(len(tok),1); chars=max(len(t),1)
 return {"word_count":len(tok),"character_count":len(t),"mean_token_length":sum(map(len,tok))/n,"type_token_ratio":len(set(x.casefold() for x in tok))/n,"digit_per_100_words":100*sum(ch.isdigit() for ch in t)/n,"punctuation_per_100_words":100*sum(ch in ".,;:!?()[]-/%" for ch in t)/n,"uppercase_per_100_chars":100*sum(ch.isupper() for ch in t)/chars,"nonascii_per_100_chars":100*sum(ord(ch)>127 for ch in t)/chars}
def main():
 ap=argparse.ArgumentParser(); here=Path(__file__).resolve(); ap.add_argument("--project-root",type=Path,default=here.parents[4]); ap.add_argument("--dataset",type=Path,required=True,help="Path to an authorized local Meric2 copy; the dataset is not distributed here"); ap.add_argument("--output-dir",type=Path,default=here.parents[1]); a=ap.parse_args(); root=a.project_root.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
 if sha(a.dataset)!=EXPECTED_MERIC: raise RuntimeError("Meric2 SHA-256 mismatch")
 d=pd.read_csv(a.dataset); 
 if list(d.columns)!=["review","label"] or len(d)!=11600: raise RuntimeError("Unexpected Meric2 schema/count")
 d=d.reset_index(names="original_row_index"); d["label_norm"]=d.label.astype(str).str.strip().str.lower(); d["missing_text"]=d.review.isna()|d.review.astype(str).str.strip().eq(""); d["valid_label"]=d.label_norm.isin(["human","ai"]); d["normalized_text_sha256"]=d.review.fillna("").map(thash)
 valid=d[~d.missing_text&d.valid_label].copy(); valid["duplicate_within_label"]=valid.duplicated(["label_norm","normalized_text_sha256"]); clean=valid[~valid.duplicate_within_label].copy()
 refs={"Combined-v2":root/"data/study/paired_study.csv","Renklier2026":root/"data/external_prepared/renklier2026_test_standardized.csv","TUM2024_TR":root/"data/external_prepared/tum2024_tr_test_standardized.csv","Bostancioglu2026_TR":root/"data/external_prepared/bostancioglu2026_tr_full_standardized.csv"}; overlap={}; refhash=set()
 for name,p in refs.items():
  x=pd.read_csv(p,usecols=["text"]); hs=set(x.text.dropna().map(thash)); overlap[name]=int(clean.normalized_text_sha256.isin(hs).sum()); refhash|=hs
 clean["overlap_excluded"]=clean.normalized_text_sha256.isin(refhash); eligible=clean[~clean.overlap_excluded].copy(); eligible["text_id"]="MERIC2_"+eligible.original_row_index.astype(str).str.zfill(5); eligible["binary_label"]=eligible.label_norm.map({"human":0,"ai":1}); feat=pd.DataFrame([features(x) for x in eligible.review],index=eligible.index); eligible=pd.concat([eligible,feat],axis=1); eligible["length_stratum"]=strata(eligible.word_count).astype(str)
 manifest_cols=["text_id","original_row_index","normalized_text_sha256","label_norm","binary_label","length_stratum",*feat.columns]; eligible[manifest_cols].to_csv(out/"MERIC2_ELIGIBLE_MANIFEST.csv",index=False)
 # Deterministic exact-word-count balance; the rank pairing is not a source pairing.
 matched=[]
 for wc,g in eligible.groupby("word_count",sort=True):
  groups={k:v.sort_values("normalized_text_sha256") for k,v in g.groupby("binary_label")}; n=min(len(groups.get(0,[])),len(groups.get(1,[])))
  if n==0: continue
  for y in (0,1):
   q=groups[y].head(n).copy(); q["length_match_id"]=[f"LM_WC{wc}_{i:05d}" for i in range(n)]; matched.append(q)
 pd.concat(matched)[manifest_cols+["length_match_id"]].to_csv(out/"MERIC2_LENGTH_MATCHED_MANIFEST.csv",index=False)
 pilot_run=root/"results/meric2_ce_pilot_20261013/training_complete.json"; pilot=json.loads(pilot_run.read_text()); pilot_hash=pilot["checkpoint_sha256"]
 rows=[]
 for obj in OBJECTIVES:
  for seed in SEEDS:
   for fold in FOLDS:
    base=root/f"artifacts/checkpoints/external_cross_corpus/protocol_faithful_rerun/{obj}_seed_{seed}_outer_{fold}"; runp=base/"run.json"; run=json.loads(runp.read_text()); model=base/"checkpoint/model.safetensors"; cfg=base/"checkpoint/config.json"; ref=root/f"results/primary/{obj}/seed_{seed}/outer_{fold}_run.json"; refj=json.loads(ref.read_text()); mh=sha(model)
    checks={"status":run.get("status")=="SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN","identity":run.get("objective")==obj and run.get("seed")==seed and run.get("outer_fold")==fold,"model_hash":mh==run.get("payload",{}).get("model_sha256"),"config_hash":sha(cfg)==run.get("payload",{}).get("config_sha256"),"roundtrip":run.get("checkpoint_roundtrip",{}).get("status")=="PASS_CHECKPOINT_ROUNDTRIP","combined_only_reference":refj.get("dataset_sha256")==sha(root/"data/study/paired_study.csv") and refj.get("fold_sha256")==sha(root/"data/study/nested_folds.csv") and run.get("train_pair_sha256")==refj.get("train_pair_sha256") and run.get("test_pair_sha256")==refj.get("test_pair_sha256"),"not_pilot_checkpoint":mh!=pilot_hash,"no_meric_marker":EXPECTED_MERIC not in json.dumps(run).lower() and "meric2" not in json.dumps(run).lower()}
    rows.append({"method":"CE_ONLY_MATCHED" if obj=="ce" else "SUPCON","objective":obj,"seed":seed,"outer_fold":fold,"run_path":str(runp),"model_path":str(model),"model_sha256":mh,"revision":run.get("revision"),"created_utc":run.get("created_utc"),**checks,"eligible":all(checks.values())})
 audit=pd.DataFrame(rows); audit.to_csv(out/"CHECKPOINT_ELIGIBILITY_AUDIT.csv",index=False)
 if len(audit)!=30 or not audit.eligible.all(): raise RuntimeError("Authorized-grid checkpoint eligibility failed; transformer inference is blocked")
 summary={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS_ELIGIBLE_FIXED_COMBINED_V2_CHECKPOINTS_AUTHORIZED_3SEED_GRID","authorized_transformer_seeds":list(SEEDS),"excluded_seeds_due_to_verifier_grid":[137,3407],"expected_transformer_cells":30,"analysis_config_sha256":sha(out/"configs/analysis_config.json"),"audit_script_sha256":sha(Path(__file__)),"dataset_sha256":sha(a.dataset),"combined_v2_sha256":sha(root/"data/study/paired_study.csv"),"fold_manifest_sha256":sha(root/"data/study/nested_folds.csv"),"raw_rows":len(d),"missing_text":int(d.missing_text.sum()),"invalid_labels":int((~d.valid_label).sum()),"duplicate_excess":int(valid.duplicate_within_label.sum()),"eligible_rows":len(eligible),"eligible_by_label":eligible.label_norm.value_counts().to_dict(),"word_count":eligible.groupby("label_norm").word_count.agg(["mean","median","min","max"]).to_dict("index"),"overlap":overlap,"overlap_excluded":int(clean.overlap_excluded.sum()),"length_matched_rows":sum(len(x) for x in matched),"length_matched_by_label":pd.concat(matched).label_norm.value_counts().to_dict(),"physical_checkpoint_cells":len(audit),"eligible_checkpoint_cells":int(audit.eligible.sum()),"pilot":{"used_meric2_for_training":True,"model":"CE seed 42 outer 0 augmented pilot only","meric2_training_rows":2400,"checkpoint_sha256":pilot_hash,"excluded_from_new_grid":pilot_hash not in set(audit.model_sha256),"prior_outcome_known":"no measurable improvement; this makes the new analysis post-hoc exploratory"},"conclusion":"Contamination excluded for the 30 verifier-authorized frozen terminal cells through Combined-v2 primary-reference hashes, pair hashes, payload hashes, and exclusion of the distinct Meric2-augmented pilot checkpoint."}
 (out/"ELIGIBILITY_AUDIT.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"); print("PASS_ELIGIBILITY_AUDIT")
if __name__=="__main__": main()
