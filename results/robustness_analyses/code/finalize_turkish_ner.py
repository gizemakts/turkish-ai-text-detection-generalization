"""Frozen NEWSPAPER cue summaries and author-disjoint surface-probe ablation."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/"results/robustness_analyses"
import run_lightweight_analyses as light

CUE_FEATURES=(
 [f"temporal_{k}_{s}" for k in light.PAT for s in ("count","per_100_words")]
 + ["temporal_merged_count","temporal_merged_per_100_words","temporal_presence"]
 + [f"{k}_{s}" for k in ("person","organization","location_gpe","named_entity_total") for s in ("count","per_100_words")]
 + ["entity_temporal_document_proxy","temporal_near_entity_count","temporal_near_entity_per_100_words",
    "entity_near_temporal_count","entity_near_temporal_per_100_words"]
)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def probe(old, allf, af):
 # Exact pair-disjoint reference reproduction.
 pred=[]
 for fold in light.FOLDS:
  tr=old.outer_fold.ne(fold); te=~tr
  model=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(C=1,penalty="l2",solver="liblinear",max_iter=2000,random_state=261012))])
  model.fit(old.loc[tr,light.OLD_FEATURES],old.loc[tr,"binary_label"])
  q=old.loc[te,["binary_label","domain"]].copy(); q["score"]=model.predict_proba(old.loc[te,light.OLD_FEATURES])[:,1]; pred.append(q)
 pred=pd.concat(pred); q=pred[pred.domain.eq("NEWSPAPER")]; ref_auc=roc_auc_score(q.binary_label,q.score)
 if abs(ref_auc-0.998772)>1e-10: raise RuntimeError(f"existing reference failed: {ref_auc}")
 pd.DataFrame([{"target":"existing_NEWSPAPER_surface_probe","expected_roc_auc":0.998772,"reproduced_roc_auc":ref_auc,"absolute_difference":abs(ref_auc-0.998772),"tolerance":1e-10,"status":"PASS"}]).to_csv(OUT/"surface_probe/EXISTING_SURFACE_PROBE_REPRODUCTION.csv",index=False)

 x=old.drop(columns=["outer_fold"]).merge(af[["pair_id","author","author_fold"]],on="pair_id",how="left",validate="many_to_one").merge(allf[["text_id"]+CUE_FEATURES],on="text_id",validate="one_to_one")
 removed=["year_date_per_100_words"]
 p2=[f for f in light.OLD_FEATURES if f not in removed]
 variants={"BASELINE_EXISTING_FULL_SURFACE_AUTHOR_DISJOINT":light.OLD_FEATURES,"P1_CUE_ONLY":CUE_FEATURES,"P2_EXISTING_MINUS_CUE":p2,"P3_FULL":p2+CUE_FEATURES}
 preds=[]; folds=[]; checks=[]
 for name,features in variants.items():
  for fold in light.FOLDS:
   tr=x.domain.eq("ABSTRACT")|(x.domain.eq("NEWSPAPER")&x.author_fold.ne(fold)); te=x.domain.eq("NEWSPAPER")&x.author_fold.eq(fold)
   train_auth=set(x.loc[tr & x.domain.eq("NEWSPAPER"),"author"].dropna()); test_auth=set(x.loc[te,"author"].dropna()); overlap=sorted(train_auth & test_auth)
   if overlap: raise RuntimeError(f"author leakage fold {fold}: {overlap}")
   model=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(C=1,penalty="l2",solver="liblinear",max_iter=2000,random_state=light.BOOT_SEED))])
   model.fit(x.loc[tr,features],x.loc[tr,"binary_label"])
   q=x.loc[te,["text_id","pair_id","binary_label","author","author_fold"]].copy(); q["score"]=model.predict_proba(x.loc[te,features])[:,1]; q["variant"]=name; preds.append(q)
   folds.append({"variant":name,"author_fold":fold,"roc_auc":roc_auc_score(q.binary_label,q.score),"n_test_documents":len(q),"n_test_authors":len(test_auth),"n_train_documents":int(tr.sum()),"n_features":len(features)})
   checks.append({"variant":name,"author_fold":fold,"train_newspaper_authors":";".join(sorted(train_auth)),"test_authors":";".join(sorted(test_auth)),"author_overlap_count":len(overlap),"scaler_fit_scope":"training_fold_only","classifier_fit_scope":"training_fold_only","status":"PASS"})
 pr=pd.concat(preds,ignore_index=True)
 if any(len(g)!=1000 or g.text_id.nunique()!=1000 for _,g in pr.groupby("variant")): raise RuntimeError("OOF accounting failure")
 pr.to_csv(OUT/"surface_probe/NEWSPAPER_SURFACE_PROBE_OOF_PREDICTIONS.csv",index=False)
 pd.DataFrame(folds).to_csv(OUT/"surface_probe/NEWSPAPER_SURFACE_PROBE_FOLD_LEVEL.csv",index=False)
 pd.DataFrame(checks).to_csv(OUT/"surface_probe/NEWSPAPER_SURFACE_PROBE_SPLIT_VALIDATION.csv",index=False)

 authors=sorted(pr.author.unique()); amap={a:i for i,a in enumerate(authors)}
 rng=np.random.default_rng(light.BOOT_SEED); draws=rng.multinomial(len(authors),np.full(len(authors),1/len(authors)),size=light.NBOOT)
 rows=[]; boots=[]; boot_by={}
 for name,g in pr.groupby("variant",sort=False):
  ix=g.author.map(amap).to_numpy(); vals=np.array([roc_auc_score(g.binary_label,g.score,sample_weight=w[ix]) for w in draws]); boot_by[name]=vals
  rows.append({"variant":name,"roc_auc":roc_auc_score(g.binary_label,g.score),"ci95_low":np.quantile(vals,.025),"ci95_high":np.quantile(vals,.975),"n_documents":len(g),"n_authors":len(authors),"n_features":len(variants[name]),"split":"author-disjoint; ABSTRACT + other-eight-author NEWSPAPER training","ner_completeness":"COMPLETE_STANZA_TR_STARLANG","scaling":"fit on training fold only"})
  boots.extend({"variant":name,"replicate":i,"roc_auc":v,"bootstrap_unit":"newspaper_author","bootstrap_seed":light.BOOT_SEED} for i,v in enumerate(vals))
 res=pd.DataFrame(rows); res.to_csv(OUT/"NEWSPAPER_SURFACE_PROBE_ABLATION.csv",index=False); pd.DataFrame(boots).to_csv(OUT/"NEWSPAPER_SURFACE_PROBE_ABLATION_BOOTSTRAP.csv",index=False)
 point=dict(zip(res.variant,res.roc_auc)); contrasts=[]
 for label,a,b in [("P1_minus_baseline","P1_CUE_ONLY","BASELINE_EXISTING_FULL_SURFACE_AUTHOR_DISJOINT"),("P2_minus_baseline","P2_EXISTING_MINUS_CUE","BASELINE_EXISTING_FULL_SURFACE_AUTHOR_DISJOINT"),("P3_minus_P2","P3_FULL","P2_EXISTING_MINUS_CUE")]:
  vals=boot_by[a]-boot_by[b]; contrasts.append({"contrast":label,"estimate":point[a]-point[b],"ci95_low":np.quantile(vals,.025),"ci95_high":np.quantile(vals,.975),"bootstrap_unit":"newspaper_author","interpretation":"descriptive paired contrast"})
 pd.DataFrame(contrasts).to_csv(OUT/"surface_probe/NEWSPAPER_SURFACE_PROBE_CONTRASTS.csv",index=False)
 return res, ref_auc, removed

def run(allf, resources, stanza_version):
 import matplotlib; matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 stamp=datetime.now(timezone.utc).astimezone().isoformat()
 afpath=ROOT/"provenance/experiments/author_disjoint_newspaper_20261013/author_disjoint_folds.csv"
 af=pd.read_csv(afpath)[["pair_id","author","author_fold"]]
 news=allf[allf.domain.eq("NEWSPAPER")].copy().drop(columns=["author","author_fold"],errors="ignore").merge(af,on="pair_id",validate="many_to_one")
 if not (len(news)==1000 and news.text_id.nunique()==1000 and news.pair_id.nunique()==500 and news.author.nunique()==10): raise RuntimeError("NEWSPAPER accounting failure")
 if not news[CUE_FEATURES].notna().all().all(): raise RuntimeError("missing cue features")
 news.to_csv(OUT/"NEWSPAPER_CUE_FEATURES_PER_TEXT.csv",index=False)

 rng=np.random.default_rng(light.BOOT_SEED); rows=[]
 for feature in CUE_FEATURES:
  w=news.pivot(index="pair_id",columns="label",values=feature).dropna(); diff=(w.AI-w.HUMAN).to_numpy(); auth=news[news.label.eq("HUMAN")].set_index("pair_id").loc[w.index,"author"].to_numpy()
  pairboot=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(light.NBOOT)])
  au=sorted(set(auth)); author_means=np.array([diff[auth==a].mean() for a in au]); authorboot=np.array([author_means[rng.integers(0,len(au),len(au))].mean() for _ in range(light.NBOOT)])
  tmp=news[[feature,"binary_label","word_count","author"]]; D=pd.get_dummies(tmp.author,drop_first=True,dtype=float); X=np.column_stack([np.ones(len(tmp)),tmp.binary_label,np.log1p(tmp.word_count),D.to_numpy()]); coef=np.linalg.lstsq(X,tmp[feature].to_numpy(float),rcond=None)[0][1]
  rows.append({"feature":feature,"human_n":len(w),"ai_n":len(w),"human_mean":w.HUMAN.mean(),"ai_mean":w.AI.mean(),"human_median":w.HUMAN.median(),"ai_median":w.AI.median(),"mean_paired_difference_ai_minus_human":diff.mean(),"median_paired_difference":np.median(diff),"paired_standardized_effect":diff.mean()/diff.std(ddof=1) if diff.std(ddof=1)>0 else np.nan,"pair_boot_ci_low":np.quantile(pairboot,.025),"pair_boot_ci_high":np.quantile(pairboot,.975),"author_cluster_ci_low":np.quantile(authorboot,.025),"author_cluster_ci_high":np.quantile(authorboot,.975),"length_author_adjusted_label_coefficient":coef,"bootstrap_replicates":light.NBOOT,"primary_bootstrap_unit":"newspaper_author (10 clusters)"})
 comp=pd.DataFrame(rows); comp.to_csv(OUT/"NEWSPAPER_CUE_CLASS_COMPARISON.csv",index=False)

 oldpath=ROOT/"Turkish_AI_Text_Detection_Exploratory_Analysis_Review/analysis_A1/corpus_shortcut_features.csv"; old=pd.read_csv(oldpath)
 res,ref_auc,removed=probe(old,allf,af)
 selected=["person_per_100_words","organization_per_100_words","location_gpe_per_100_words","named_entity_total_per_100_words","temporal_merged_per_100_words","temporal_near_entity_per_100_words"]
 cidx=comp.set_index("feature"); md=["# NEWSPAPER cue analysis","",f"Completed: `{stamp}`.","",f"Stanza `{stanza_version}` with Turkish `boun` tokenization and `starlang` NER was applied to all 1,000 NEWSPAPER texts (500 HUMAN–AI pairs; 10 newspaper authors). The frozen deterministic temporal definitions and 20-token entity–temporal proximity window were preserved.","","## Selected density results","","| Feature | HUMAN mean | AI mean | AI − HUMAN | 95% author-cluster interval |","|---|---:|---:|---:|---:|"]
 for f in selected:
  r=cidx.loc[f]; md.append(f"| {f} | {r.human_mean:.6f} | {r.ai_mean:.6f} | {r.mean_paired_difference_ai_minus_human:+.6f} | [{r.author_cluster_ci_low:.6f}, {r.author_cluster_ci_high:.6f}] |")
 md.extend(["","Intervals resample the ten newspaper-author clusters. Pair-level intervals and descriptive length/author-adjusted coefficients are in the CSV. These are associational corpus-construction differences; proxy fields are not ground-truth factual-event labels, and class differences do not establish that BERTurk used the cues."])
 (OUT/"NEWSPAPER_CUE_ANALYSIS.md").write_text("\n".join(md)+"\n",encoding="utf-8")
 md=["# NEWSPAPER surface-probe ablation","",f"Completed: `{stamp}`.","","Primary estimates use five established author-disjoint folds. Each held-out fold contains two newspaper authors; no held-out newspaper author occurs in training. ABSTRACT texts and the other eight newspaper authors form each training set. Standardization and logistic-regression fitting occur inside each training fold.","",f"The original full-surface pair-disjoint result was exactly reproduced at ROC-AUC `{ref_auc:.6f}`. The author-disjoint baseline uses the same original surface features under the primary author-disjoint split.","","| Probe | ROC-AUC | 95% ten-author cluster interval |","|---|---:|---:|"]
 for _,r in res.iterrows(): md.append(f"| {r['variant']} | {r.roc_auc:.6f} | [{r.ci95_low:.6f}, {r.ci95_high:.6f}] |")
 md.extend(["",f"P2 removes only `{removed[0]}`, the sole original feature directly duplicated by the frozen temporal/cue family. It retains `digit_token_fraction` because that broader feature is not a direct duplicate.","","Interpretation: high P1 means the cue family itself carries label information; high P2 means other surface features retain strong separability; P3 > P2 indicates incremental information from the cue family. P3 approximately equal to P2 does not establish independence or non-use because correlated or redundant signals may remain."])
 (OUT/"NEWSPAPER_SURFACE_PROBE_ABLATION.md").write_text("\n".join(md)+"\n",encoding="utf-8")

 show=res.set_index("variant").loc[["P1_CUE_ONLY","P2_EXISTING_MINUS_CUE","P3_FULL"]].reset_index(); y=np.arange(3)
 fig,ax=plt.subplots(figsize=(7.2,3.8)); ax.errorbar(show.roc_auc,y,xerr=[show.roc_auc-show.ci95_low,show.ci95_high-show.roc_auc],fmt="o",color="#285f8f",ecolor="#6f8fae",capsize=3); ax.set_yticks(y,["P1: cue only","P2: surface minus cue overlap","P3: surface + cues"]); ax.set_xlim(max(.5,show.ci95_low.min()-.03),1.002); ax.axvline(.5,color="#777",ls="--",lw=.8); ax.set_xlabel("Author-disjoint ROC-AUC (95% author-cluster interval)"); ax.set_title("NEWSPAPER surface-probe ablation"); fig.tight_layout(); fig.savefig(OUT/"figures/NEWSPAPER_SURFACE_PROBE_ABLATION_COMPLETE.png",dpi=300); plt.close(fig)

 resource_hashes=OUT/"provenance/TURKISH_NER_RESOURCE_SHA256.csv"; spec=OUT/"newspaper_cues/TURKISH_NER_MODEL_SPEC.json"
 inputs=[ROOT/"data/study/paired_study.csv",oldpath,afpath,OUT/"ANALYSIS_PLAN_FROZEN.md",Path(__file__),HERE/"complete_turkish_ner.py",spec,resource_hashes]
 prov={"completed_at":stamp,"analysis_status":"COMPLETE","stanza_version":stanza_version,"language":"tr","tokenize_package":"boun","ner_package":"starlang","download_source":"stanfordnlp/stanza-tr release v1.14.0 via Stanza downloader/Hugging Face Hub","resource_path":str(Path(resources).resolve()),"resource_file_count":len(pd.read_csv(resource_hashes)),"bootstrap_seed":light.BOOT_SEED,"bootstrap_replicates":light.NBOOT,"input_files":[{"path":str(p.resolve()),"sha256":sha(p)} for p in inputs],"sample_accounting":{"newspaper_documents":len(news),"pairs":news.pair_id.nunique(),"authors":news.author.nunique(),"exclusions":0},"validation":{"all_cue_features_nonmissing":True,"author_disjoint_all_folds":True,"scaling_fit_training_only":True,"existing_surface_probe_reproduced":ref_auc,"label_dependent_feature_engineering":False},"removed_existing_features_for_p2":removed}
 (OUT/"newspaper_cues/NEWSPAPER_CUE_PROVENANCE.json").write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":
 cache=OUT/"newspaper_cues/ALL_TEXT_CUE_FEATURES_WITH_NER.csv"
 spec=json.loads((OUT/"newspaper_cues/TURKISH_NER_MODEL_SPEC.json").read_text(encoding="utf-8"))
 run(pd.read_csv(cache),OUT/"resources/stanza_tr_1.14.0",spec["library_version"])
 print("COMPLETE_FINALIZATION_FROM_NER_CACHE")
