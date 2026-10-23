#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score
METRICS=("roc_auc","average_precision","standardized_partial_roc_auc_max_fpr_003")
FEATURES=("word_count","character_count","mean_token_length","type_token_ratio","digit_per_100_words","punctuation_per_100_words","uppercase_per_100_chars","nonascii_per_100_chars")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(y,s): return {"roc_auc":roc_auc_score(y,s),"average_precision":average_precision_score(y,s),"standardized_partial_roc_auc_max_fpr_003":roc_auc_score(y,s,max_fpr=.03)}
def main():
 ap=argparse.ArgumentParser(); here=Path(__file__).resolve(); ap.add_argument("--output-dir",type=Path,default=here.parents[1]); a=ap.parse_args(); out=a.output_dir.resolve(); predroot=out/"predictions"
 man=pd.read_csv(out/"MERIC2_ELIGIBLE_MANIFEST.csv"); matched=pd.read_csv(out/"MERIC2_LENGTH_MATCHED_MANIFEST.csv"); matched_ids=set(matched.text_id)
 rows=[]
 authorized_seeds=(42,314,2026)
 transformer_paths=[predroot/m/f"seed_{s}"/f"outer_{f}.csv" for m in ("CE_ONLY_MATCHED","SUPCON") for s in authorized_seeds for f in range(5)]
 for p in transformer_paths:
  side=p.with_suffix(".json");
  if not side.exists(): raise RuntimeError(f"Missing transformer sidecar: {p}")
  meta=json.loads(side.read_text()); q=pd.read_csv(p)
  if meta.get("status")!="COMPLETE" or meta.get("prediction_sha256")!=sha(p) or len(q)!=len(man): raise RuntimeError(f"Incomplete/invalid transformer cell: {p}")
  if q.method.nunique()!=1 or q.method.iloc[0]!=meta["method"] or q.seed.nunique()!=1 or int(q.seed.iloc[0])!=int(meta["seed"]) or q.outer_fold.nunique()!=1 or int(q.outer_fold.iloc[0])!=int(meta["outer_fold"]): raise RuntimeError(f"Transformer identity mismatch: {p}")
  rows.append(q)
 if len(rows)!=30: raise RuntimeError(f"Expected 30 verifier-authorized transformer cells, found {len(rows)}")
 tp=predroot/"TFIDF_M4_predictions.csv";
 if not tp.exists() or not tp.with_suffix(".json").exists(): raise FileNotFoundError("TF-IDF predictions incomplete")
 tm=json.loads(tp.with_suffix(".json").read_text())
 if tm.get("status")!="COMPLETE_COMBINED_V2_ONLY_TFIDF" or tm.get("prediction_sha256")!=sha(tp): raise RuntimeError("TF-IDF prediction provenance mismatch")
 rows.append(pd.read_csv(tp)); pred=pd.concat(rows,ignore_index=True); pred=pred.merge(man,on=["text_id","normalized_text_sha256","binary_label"],validate="many_to_one")
 expected=(30+5)*len(man)
 if len(pred)!=expected or pred.raw_logit_score.isna().any(): raise RuntimeError(f"Prediction grid invalid: {len(pred)} vs {expected}")
 pred["population"]=np.where(pred.text_id.isin(matched_ids),"LENGTH_MATCHED","FULL_ONLY")
 per=[]
 for population,ids in [("FULL",set(man.text_id)),("LENGTH_MATCHED",matched_ids)]:
  q=pred[pred.text_id.isin(ids)]
  for keys,g in q.groupby(["method","seed","outer_fold"],dropna=False): per.append({"population":population,"method":keys[0],"seed":keys[1],"outer_fold":keys[2],"n":len(g),"n_human":int((g.binary_label==0).sum()),"n_ai":int((g.binary_label==1).sum()),**metric(g.binary_label,g.raw_logit_score)})
 per=pd.DataFrame(per); per.to_csv(out/"MAIN_AND_MATCHED_PER_CELL_METRICS.csv",index=False,float_format="%.12f")
 seed=[]
 for (pop,method,seedv),g in per[per.method.ne("TFIDF_M4")].groupby(["population","method","seed"]): seed.append({"population":pop,"method":method,"seed":seedv,"n_folds":len(g),**{m:g[m].mean() for m in METRICS}})
 seed=pd.DataFrame(seed); seed.to_csv(out/"TRANSFORMER_SEED_LEVEL_METRICS.csv",index=False,float_format="%.12f")
 agg=[]
 for (pop,method),g in seed.groupby(["population","method"]):
  row={"population":pop,"method":method,"aggregation_unit":"seed after averaging five dependent folds","n_units":len(g)}
  for m in METRICS: row|={f"{m}_mean":g[m].mean(),f"{m}_sample_sd":g[m].std(ddof=1),f"{m}_min":g[m].min(),f"{m}_max":g[m].max()}
  agg.append(row)
 for pop,g in per[per.method.eq("TFIDF_M4")].groupby("population"):
  row={"population":pop,"method":"TFIDF_M4","aggregation_unit":"five dependent outer-fold fits; descriptive","n_units":len(g)}
  for m in METRICS: row|={f"{m}_mean":g[m].mean(),f"{m}_sample_sd":g[m].std(ddof=1),f"{m}_min":g[m].min(),f"{m}_max":g[m].max()}
  agg.append(row)
 agg=pd.DataFrame(agg); agg.to_csv(out/"MAIN_AND_MATCHED_AGGREGATE_METRICS.csv",index=False,float_format="%.12f")
 length=[]
 for keys,g in pred.groupby(["method","seed","outer_fold","length_stratum"],dropna=False):
  if g.binary_label.nunique()<2: continue
  length.append({"method":keys[0],"seed":keys[1],"outer_fold":keys[2],"length_stratum":keys[3],"n":len(g),"n_human":int((g.binary_label==0).sum()),"n_ai":int((g.binary_label==1).sum()),**metric(g.binary_label,g.raw_logit_score)})
 pd.DataFrame(length).to_csv(out/"LENGTH_STRATIFIED_PER_CELL_METRICS.csv",index=False,float_format="%.12f")
 comps=[]
 for pop,frame in [("FULL",man),("LENGTH_MATCHED",matched)]:
  for f in FEATURES:
   h=frame.loc[frame.binary_label.eq(0),f]; ai=frame.loc[frame.binary_label.eq(1),f]; pooled=np.sqrt(((len(h)-1)*h.var(ddof=1)+(len(ai)-1)*ai.var(ddof=1))/(len(h)+len(ai)-2))
   comps.append({"population":pop,"feature":f,"human_n":len(h),"ai_n":len(ai),"human_mean":h.mean(),"ai_mean":ai.mean(),"human_median":h.median(),"ai_median":ai.median(),"ai_minus_human":ai.mean()-h.mean(),"standardized_mean_difference":(ai.mean()-h.mean())/pooled if pooled>0 else np.nan})
 pd.DataFrame(comps).to_csv(out/"LEXICAL_SURFACE_CLASS_COMPARISON.csv",index=False,float_format="%.12f")
 cor=[]
 for keys,g in pred.groupby(["method","seed","outer_fold"],dropna=False):
  for subset,y in [("ALL",None),("HUMAN",0),("AI",1)]:
   z=g if y is None else g[g.binary_label.eq(y)]
   for f in FEATURES:
    rho,p=spearmanr(z.raw_logit_score,z[f]); cor.append({"method":keys[0],"seed":keys[1],"outer_fold":keys[2],"subset":subset,"feature":f,"n":len(z),"spearman_rho":rho,"p_unadjusted_descriptive":p})
 pd.DataFrame(cor).to_csv(out/"SCORE_SURFACE_RELATIONSHIPS.csv",index=False,float_format="%.12f")
 full=agg[agg.population.eq("FULL")]; sens=agg[agg.population.eq("LENGTH_MATCHED")]; lines=["# Meric2 fixed-model exploratory stress test","","Status: COMPLETE only after all prediction validations above passed.","","Meric2 was not used to fit, tune, calibrate, select, or modify any evaluated model. The distinct Meric2-augmented CE pilot checkpoint was excluded. All results are post-hoc exploratory because that pilot outcome and earlier dataset audits were known.","","Transformer evaluation is restricted to the external inference verifier's authorized seeds 42, 314, and 2026 (five folds per seed; 30 CE/SupCon cells). Seeds 137 and 3407 are not used. The five TF-IDF outer models are unchanged.","","## Full population",full.to_markdown(index=False,floatfmt=".6f"),"","## Exact-word-count matched sensitivity",sens.to_markdown(index=False,floatfmt=".6f"),"","Primary metrics were calculated separately for each independently trained realization. Transformer folds were averaged within seed before three-seed summaries; folds were not treated as independent experimental replicates. TF-IDF fold variability is descriptive because its outer fits overlap in training data.","","The exact-word-count matches are artificial balance blocks, not HUMAN–AI source pairs. Observation-level independence and prompt lineage cannot be verified, so no causal or source-paired inference is claimed and no nominal sample-bootstrap interval is presented as definitive.","","Surface differences and score correlations are associational. This experiment cannot identify historical effects or generation-prompt effects."]
 (out/"SCIENTIFIC_INTERPRETATION.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print("COMPLETE_FINALIZATION")
if __name__=="__main__": main()
