"""Manual Turkish-NER completion. Never downloads models itself."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, re, sys, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"results/robustness_analyses"; CODE=OUT/"code/run_lightweight_analyses.py"
spec=importlib.util.spec_from_file_location("light",CODE); light=importlib.util.module_from_spec(spec); spec.loader.exec_module(light)
MAP={"PER":"person","PERSON":"person","ORG":"organization","ORGANIZATION":"organization","LOC":"location_gpe","LOCATION":"location_gpe","GPE":"location_gpe"}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resource_manifest(root):
 rows=[]
 for p in sorted(Path(root).rglob("*")):
  if p.is_file(): rows.append({"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)})
 pd.DataFrame(rows).to_csv(OUT/"provenance/TURKISH_NER_RESOURCE_SHA256.csv",index=False); return rows
def token_index(spans,pos):
 for i,(a,b) in enumerate(spans):
  if a<=pos<b:return i
 return min(range(len(spans)),key=lambda i:abs(spans[i][0]-pos)) if spans else 0
def extract(nlp, frame):
 rows=[]
 for r in frame.itertuples(index=False):
  text=unicodedata.normalize("NFC",str(r.text)); doc=nlp(text); ents=[]
  for e in doc.ents:
   typ=MAP.get(str(e.type).upper())
   if typ: ents.append((int(e.start_char),int(e.end_char),typ))
  by,temp=light.merged_spans(text); toks=[m.span() for m in re.finditer(r"\S+",text)]; n=len(toks); et=[(a,b,t,token_index(toks,a)) for a,b,t in ents]; tt=[(a,b,token_index(toks,a)) for a,b in temp]
  z={"text_id":r.text_id,"pair_id":r.pair_id,"label":r.label,"binary_label":int(r.label=="AI"),"domain":r.domain,"generator":r.generator,"author":r.author,"word_count":n,"text_sha256":hashlib.sha256(text.encode()).hexdigest(),"ner_status":"COMPLETE_STANZA_TR_STARLANG","ner_spans_json":json.dumps(ents,ensure_ascii=False),"temporal_spans_json":json.dumps(temp),"cue_spans_json":json.dumps(sorted([(a,b) for a,b,_ in ents]+temp))}
  for typ in light.PAT: z[f"temporal_{typ}_count"]=len(by[typ]); z[f"temporal_{typ}_per_100_words"]=100*len(by[typ])/max(n,1)
  z["temporal_merged_count"]=len(temp); z["temporal_merged_per_100_words"]=100*len(temp)/max(n,1); z["temporal_presence"]=int(bool(temp))
  for typ in ("person","organization","location_gpe"):
   c=sum(e[2]==typ for e in et); z[f"{typ}_count"]=c; z[f"{typ}_per_100_words"]=100*c/max(n,1)
  z["named_entity_total_count"]=len(et); z["named_entity_total_per_100_words"]=100*len(et)/max(n,1); z["entity_temporal_document_proxy"]=int(bool(et) and bool(tt))
  near_t=sum(any(abs(ti-ei)<=20 for *_,ei in et) for *_,ti in tt); near_e=sum(any(abs(ei-ti)<=20 for *_,ti in tt) for *_,ei in et)
  z["temporal_near_entity_count"]=near_t; z["temporal_near_entity_per_100_words"]=100*near_t/max(n,1); z["entity_near_temporal_count"]=near_e; z["entity_near_temporal_per_100_words"]=100*near_e/max(n,1); rows.append(z)
 return pd.DataFrame(rows)
def summarize_and_probe(allf):
 import matplotlib; matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 news=allf[allf.domain.eq("NEWSPAPER")].copy(); af=pd.read_csv(ROOT/"provenance/experiments/author_disjoint_newspaper_20261013/author_disjoint_folds.csv")[["pair_id","author_fold"]]; news=news.drop(columns=["author_fold"],errors="ignore").merge(af,on="pair_id",validate="many_to_one"); news.to_csv(OUT/"NEWSPAPER_CUE_FEATURES_PER_TEXT.csv",index=False)
 feats=[c for c in allf if (c.endswith("_count") or c.endswith("_per_100_words") or c in ["temporal_presence","entity_temporal_document_proxy"])]
 rng=np.random.default_rng(light.BOOT_SEED); comp=[]
 for f in feats:
  w=news.pivot(index="pair_id",columns="label",values=f).dropna(); diff=(w.AI-w.HUMAN).to_numpy(); auth=news[news.label.eq("HUMAN")].set_index("pair_id").loc[w.index,"author"].to_numpy(); pb=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(light.NBOOT)]); au=sorted(set(auth)); am=np.array([diff[auth==a].mean() for a in au]); ab=np.array([am[rng.integers(0,len(am),len(am))].mean() for _ in range(light.NBOOT)])
  tmp=news[[f,"binary_label","word_count","author"]]; D=pd.get_dummies(tmp.author,drop_first=True,dtype=float); X=np.column_stack([np.ones(len(tmp)),tmp.binary_label,np.log1p(tmp.word_count),D]); coef=np.linalg.lstsq(X,tmp[f],rcond=None)[0][1]
  comp.append({"feature":f,"human_n":len(w),"ai_n":len(w),"human_mean":w.HUMAN.mean(),"ai_mean":w.AI.mean(),"human_median":w.HUMAN.median(),"ai_median":w.AI.median(),"mean_paired_difference_ai_minus_human":diff.mean(),"median_paired_difference":np.median(diff),"paired_standardized_effect":diff.mean()/diff.std(ddof=1) if diff.std(ddof=1)>0 else np.nan,"pair_boot_ci_low":np.quantile(pb,.025),"pair_boot_ci_high":np.quantile(pb,.975),"author_cluster_ci_low":np.quantile(ab,.025),"author_cluster_ci_high":np.quantile(ab,.975),"length_author_adjusted_label_coefficient":coef,"bootstrap_replicates":light.NBOOT})
 compdf=pd.DataFrame(comp); compdf.to_csv(OUT/"NEWSPAPER_CUE_CLASS_COMPARISON.csv",index=False)
 old=pd.read_csv(ROOT/"Turkish_AI_Text_Detection_Exploratory_Analysis_Review/analysis_A1/corpus_shortcut_features.csv"); x=old.drop(columns=["outer_fold"]).merge(af,on="pair_id",how="left",validate="many_to_one").merge(allf[["text_id"]+feats],on="text_id",validate="one_to_one")
 p2=[f for f in light.OLD_FEATURES if f!="year_date_per_100_words"]; variants={"P1_CUE_ONLY":feats,"P2_EXISTING_MINUS_CUE":p2,"P3_FULL":p2+feats}; preds=[]
 for name,fs in variants.items():
  for fold in light.FOLDS:
   tr=x.domain.eq("ABSTRACT")|(x.domain.eq("NEWSPAPER")&x.author_fold.ne(fold)); te=x.domain.eq("NEWSPAPER")&x.author_fold.eq(fold); pipe=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(C=1,penalty="l2",solver="liblinear",max_iter=2000,random_state=light.BOOT_SEED))]); pipe.fit(x.loc[tr,fs],x.loc[tr,"binary_label"]); q=x.loc[te,["text_id","pair_id","binary_label","author","author_fold"]].copy(); q["score"]=pipe.predict_proba(x.loc[te,fs])[:,1]; q["variant"]=name; preds.append(q)
 pr=pd.concat(preds); authors=sorted(pr.author.unique()); amap={a:i for i,a in enumerate(authors)}; draws=rng.multinomial(10,np.full(10,.1),size=light.NBOOT); rr=[]; bb=[]
 for name,g in pr.groupby("variant"):
  ix=g.author.map(amap).to_numpy(); vals=np.array([roc_auc_score(g.binary_label,g.score,sample_weight=w[ix]) for w in draws]); rr.append({"variant":name,"roc_auc":roc_auc_score(g.binary_label,g.score),"ci95_low":np.quantile(vals,.025),"ci95_high":np.quantile(vals,.975),"n_documents":len(g),"n_authors":10,"split":"author-disjoint; ABSTRACT + other-eight-author NEWSPAPER training","ner_completeness":"COMPLETE"}); bb += [{"variant":name,"replicate":i,"roc_auc":v,"bootstrap_unit":"author","bootstrap_seed":light.BOOT_SEED} for i,v in enumerate(vals)]
 res=pd.DataFrame(rr); res.to_csv(OUT/"NEWSPAPER_SURFACE_PROBE_ABLATION.csv",index=False); pd.DataFrame(bb).to_csv(OUT/"NEWSPAPER_SURFACE_PROBE_ABLATION_BOOTSTRAP.csv",index=False)
 top=compdf.sort_values("paired_standardized_effect",key=lambda s:s.abs(),ascending=False).head(10); fig,ax=plt.subplots(figsize=(9,5)); ax.barh(top.feature,top.mean_paired_difference_ai_minus_human,color="#6857a5"); ax.axvline(0,color="black",lw=.8); ax.set_xlabel("Mean paired difference (AI − HUMAN)"); ax.set_title("Named-entity, temporal, and event-proxy cue differences"); fig.tight_layout(); fig.savefig(OUT/"figures/NEWSPAPER_CUE_DIFFERENCES_COMPLETE.png",dpi=220); plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4)); ax.errorbar(res.roc_auc,np.arange(len(res)),xerr=[res.roc_auc-res.ci95_low,res.ci95_high-res.roc_auc],fmt="o"); ax.set_yticks(np.arange(len(res)),res.variant); ax.set_xlim(0,1.01); ax.set_xlabel("Author-cluster ROC-AUC"); ax.set_title("Complete NEWSPAPER surface-probe ablation"); fig.tight_layout(); fig.savefig(OUT/"figures/NEWSPAPER_SURFACE_PROBE_ABLATION_COMPLETE.png",dpi=220); plt.close(fig)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--resources",type=Path,required=True); p.add_argument("--validate-only",action="store_true"); p.add_argument("--smoke-test",action="store_true"); p.add_argument("--run-full",action="store_true"); a=p.parse_args(); import stanza
 assert stanza.__version__=="1.14.0", stanza.__version__
 if a.validate_only: print(json.dumps({"status":"PASS_CODE_AND_VERSION","models_present":a.resources.exists()})); return
 if not a.resources.exists(): raise FileNotFoundError("manual Stanza model download required")
 resource_manifest(a.resources); nlp=stanza.Pipeline("tr",dir=str(a.resources),processors={"tokenize":"boun","ner":"starlang"},download_method=None,use_gpu=False,verbose=False); data=pd.read_csv(ROOT/"data/study/paired_study.csv")
 if a.smoke_test:
  smoke=pd.DataFrame([
   {"text_id":"SMOKE_1","pair_id":"SMOKE_1","label":"HUMAN","domain":"NEWSPAPER","generator":"HUMAN","author":"SMOKE","text":"Recep Tayyip Erdoğan bugün Ankara'da Türkiye Büyük Millet Meclisi üyeleriyle görüştü."},
   {"text_id":"SMOKE_2","pair_id":"SMOKE_2","label":"AI","domain":"NEWSPAPER","generator":"SMOKE","author":"SMOKE","text":"Orhan Pamuk pazartesi günü İstanbul Üniversitesi tarafından düzenlenen toplantıya katıldı."},
  ])
  out=extract(nlp,smoke); out.to_csv(OUT/"newspaper_cues/NER_SMOKE_TEST.csv",index=False)
  detected={k:int(out[f"{k}_count"].sum()) for k in ("person","organization","location_gpe")}
  if not all(v>0 for v in detected.values()): raise RuntimeError(f"smoke test did not detect all required Turkish entity types: {detected}")
  print(json.dumps({"status":"PASS_NER_SMOKE_TEST","detected":detected},ensure_ascii=False)); return
 if a.run_full:
  allf=extract(nlp,data); allf.to_csv(OUT/"newspaper_cues/ALL_TEXT_CUE_FEATURES_WITH_NER.csv",index=False)
  import finalize_turkish_ner as finalizer
  finalizer.run(allf,a.resources,stanza.__version__)
  print("COMPLETE_NER_AND_FULL_ABLATION"); return
 p.error("choose a mode")
if __name__=="__main__": main()
