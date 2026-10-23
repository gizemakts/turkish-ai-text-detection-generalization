"""Finalize five-seed external statistics after manual checkpoint inference."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"results/robustness_analyses"
SEEDS=[42,137,314,2026,3407]; FOLDS=range(5); METHODS=["CE_ONLY_MATCHED","SUPCON"]
EXT={"renklier2026":ROOT/"data/external_prepared/renklier2026_test_standardized.csv","tum2024_tr":ROOT/"data/external_prepared/tum2024_tr_test_standardized.csv","bostancioglu2026_tr":ROOT/"data/external_prepared/bostancioglu2026_tr_full_standardized.csv"}
def auc(y,s,w=None): return roc_auc_score(y,s,sample_weight=w)
def load():
 z=[]
 for ds,path in EXT.items():
  meta=pd.read_csv(path)
  for m in METHODS:
   for seed in SEEDS:
    for fold in FOLDS:
     p=ROOT/f"artifacts/predictions/external_cross_corpus/preseal/{ds}/{m}/seed_{seed}/outer_{fold}.csv"
     if not p.is_file(): raise FileNotFoundError(f"manual inference incomplete: {p}")
     q=pd.read_csv(p); assert len(q)==len(meta) and q.external_id.equals(meta.external_id) and q.text_sha256.equals(meta.text_sha256)
     q=q.merge(meta[["external_id","binary_label","source_group_id","text_sha256"]].rename(columns={"text_sha256":"meta_hash"}),on="external_id",validate="one_to_one"); q["corpus"]=ds; z.append(q)
 return pd.concat(z,ignore_index=True)
def main():
 d=load(); cells=[]
 for k,g in d.groupby(["corpus","method","seed","outer_fold"]): cells.append((*k,auc(g.binary_label,g.raw_score_or_logit)))
 cells=pd.DataFrame(cells,columns=["corpus","method","seed","outer_fold","roc_auc"])
 seed=cells.groupby(["corpus","method","seed"]).roc_auc.mean().reset_index(name="fold_mean_roc_auc"); seed["status"]="COMPLETE_5SEED"; seed.to_csv(OUT/"EXTERNAL_CE_SUPCON_SEED_LEVEL.csv",index=False)
 summ=seed.groupby(["corpus","method"]).fold_mean_roc_auc.agg(["mean","std","min","max","count"]).reset_index(); summ.columns=["corpus","method","mean_roc_auc","sample_sd","range_min","range_max","n_seeds"]; summ["status"]="COMPLETE_5SEED"; summ.to_csv(OUT/"EXTERNAL_CE_SUPCON_5SEED_SUMMARY.csv",index=False)
 old=pd.read_csv(OUT/"external/EXTERNAL_CE_SUPCON_SEED_LEVEL_3SEED.csv"); oldsum=old[old.method.isin(METHODS)].groupby(["corpus","method"]).roc_auc.mean().rename("original_3seed_mean").reset_index(); comp=oldsum.merge(summ,on=["corpus","method"]); comp["expanded_5seed_mean"]=comp.mean_roc_auc; comp["absolute_change"]=comp.expanded_5seed_mean-comp.original_3seed_mean
 # ordering and effect-sign conclusions are evaluated per corpus from matched means
 effects3=old[old.method.isin(METHODS)].pivot(index=["corpus","seed"],columns="method",values="roc_auc"); e3=(effects3.SUPCON-effects3.CE_ONLY_MATCHED).groupby("corpus").mean()
 effects5=seed.pivot(index=["corpus","seed"],columns="method",values="fold_mean_roc_auc"); e5=(effects5.SUPCON-effects5.CE_ONLY_MATCHED).groupby("corpus").mean()
 comp["sign_or_order_changed"]=comp.corpus.map(lambda c: bool(np.sign(e3[c])!=np.sign(e5[c]))); comp["conclusion_changed"]=comp.sign_or_order_changed.map({True:"YES_ORDER_REVERSED",False:"NO_DIRECTIONAL_ORDER_CHANGE"}); comp["status"]="COMPLETE_5SEED"; comp[["corpus","method","original_3seed_mean","expanded_5seed_mean","absolute_change","sign_or_order_changed","conclusion_changed","status"]].to_csv(OUT/"EXTERNAL_CE_SUPCON_3SEED_VS_5SEED.csv",index=False)
 # Conditional fixed-model cluster bootstrap; model identities remain fixed.
 br=[]; rng=np.random.default_rng(20261014)
 for ds,g0 in d.groupby("corpus"):
  unit="source_group_id" if ds=="renklier2026" else "meta_hash"; units=np.sort(g0[unit].astype(str).unique()); ui={u:i for i,u in enumerate(units)}; draws=rng.multinomial(len(units),np.full(len(units),1/len(units)),size=5000); md={}
  for m in METHODS:
   reals=[]
   for _,g in g0[g0.method.eq(m)].groupby(["seed","outer_fold"]): reals.append((g[unit].astype(str).map(ui).to_numpy(),g.binary_label.to_numpy(),g.raw_score_or_logit.to_numpy()))
   vals=np.empty(5000)
   for i,w in enumerate(draws): vals[i]=np.mean([auc(y,s,w[ix]) for ix,y,s in reals])
   md[m]=vals; point=cells[(cells.corpus==ds)&(cells.method==m)].roc_auc.mean(); br.append({"corpus":ds,"estimand":m,"point_estimate":point,"ci95_low":np.quantile(vals,.025),"ci95_high":np.quantile(vals,.975),"bootstrap_unit":unit,"replicates":5000,"conditional_on_fixed_models":True,"status":"COMPLETE_5SEED"})
  vals=md["SUPCON"]-md["CE_ONLY_MATCHED"]; point=float((effects5.loc[ds,"SUPCON"]-effects5.loc[ds,"CE_ONLY_MATCHED"]).mean()); br.append({"corpus":ds,"estimand":"SUPCON_MINUS_CE","point_estimate":point,"ci95_low":np.quantile(vals,.025),"ci95_high":np.quantile(vals,.975),"bootstrap_unit":unit,"replicates":5000,"conditional_on_fixed_models":True,"status":"COMPLETE_5SEED"})
 pd.DataFrame(br).to_csv(OUT/"EXTERNAL_CE_SUPCON_5SEED_BOOTSTRAP.csv",index=False)
 eff=(effects5.SUPCON-effects5.CE_ONLY_MATCHED).rename("effect").reset_index(); fig,ax=plt.subplots(figsize=(8,4));
 for i,(ds,g) in enumerate(eff.groupby("corpus")):
  g=g.sort_values("seed"); ax.scatter(g.effect,[i]*len(g),s=50,label=ds)
  for j,r in enumerate(g.itertuples()): ax.annotate(str(r.seed),(r.effect,i),xytext=(0,7+7*(j%2)),textcoords="offset points",fontsize=7,ha="center")
 ax.axvline(0,color="black",lw=.8); ax.set_yticks(range(3),sorted(eff.corpus.unique())); ax.margins(y=.2); ax.set_xlabel("Seed-level fold-mean ROC-AUC: SupCon - CE"); fig.tight_layout(); fig.savefig(OUT/"figures/EXTERNAL_5SEED_SEED_LEVEL_EFFECTS.png",dpi=220)
if __name__=="__main__": main()
