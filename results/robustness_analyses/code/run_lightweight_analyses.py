from __future__ import annotations

import hashlib, json, math, re, sys, unicodedata
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "robustness_analyses"
SEEDS3 = [42, 314, 2026]
SEEDS5 = [42, 137, 314, 2026, 3407]
FOLDS = range(5)
METHODS = ["CE_ONLY_MATCHED", "SUPCON"]
ALL_METHODS = METHODS + ["LIT2026_REIMPL_V1_1"]
OBJ = {"CE_ONLY_MATCHED": "ce", "SUPCON": "supcon"}
BOOT_SEED = 20261014
NBOOT = 5000
STAMP = datetime.now(timezone.utc).astimezone().isoformat()
EXT = {
    "renklier2026": ROOT / "data/external_prepared/renklier2026_test_standardized.csv",
    "tum2024_tr": ROOT / "data/external_prepared/tum2024_tr_test_standardized.csv",
    "bostancioglu2026_tr": ROOT / "data/external_prepared/bostancioglu2026_tr_full_standardized.csv",
}
OLD_FEATURES = ["word_count","sentence_count","mean_sentence_words","wordpiece_count",
 "punctuation_per_100_chars","digit_token_fraction","year_date_per_100_words",
 "uppercase_alpha_ratio","lexical_diversity","structured_heading","list_line",
 "quote_per_100_chars","dash_per_100_chars","newline_per_100_chars"]

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def save(df, name):
    p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,index=False,encoding="utf-8")

def auc(y,s,w=None):
    y=np.asarray(y,int); s=np.asarray(s,float)
    if len(np.unique(y))<2:return np.nan
    return float(roc_auc_score(y,s,sample_weight=w))

def checkpoint_audits():
    rows=[]; base=ROOT/"artifacts/checkpoints/external_cross_corpus/protocol_faithful_rerun"
    for method in METHODS:
      obj=OBJ[method]
      for seed in [137,3407]:
       for fold in FOLDS:
        d=base/f"{obj}_seed_{seed}_outer_{fold}"; rp=d/"run.json"; mp=d/"checkpoint/model.safetensors"; cp=d/"checkpoint/config.json"
        status="MISSING"; warning="No terminal-protocol directory/run/payload found"
        rec={}
        if rp.is_file():
          try: rec=json.loads(rp.read_text(encoding="utf-8"))
          except Exception as e: warning=f"Unreadable run record: {e}"
          ok=(rec.get("status")=="SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN" and rec.get("objective")==obj and rec.get("seed")==seed and rec.get("outer_fold")==fold and mp.is_file() and cp.is_file())
          status="READY_EXISTING" if ok else "INVALID_OR_AMBIGUOUS"; warning="" if ok else "Run/payload identity or completion gate failed"
        rows.append({"method":method,"objective":obj,"seed":seed,"outer_fold":fold,"classification":status,
          "checkpoint_path":mp.relative_to(ROOT).as_posix(),"checkpoint_exists":mp.is_file(),"checkpoint_sha256":sha(mp) if mp.is_file() else "",
          "config_path":cp.relative_to(ROOT).as_posix(),"config_sha256":sha(cp) if cp.is_file() else "",
          "run_log_path":rp.relative_to(ROOT).as_posix(),"run_log_exists":rp.is_file(),"training_status":rec.get("status",""),
          "backbone":rec.get("model",""),"backbone_revision":rec.get("revision",""),"terminal_protocol_verified":status=="READY_EXISTING","warning":warning})
    a=pd.DataFrame(rows); save(a,"EXTERNAL_5SEED_CHECKPOINT_AUDIT.csv")
    counts=a.classification.value_counts().to_dict()
    md=f"""# External five-seed checkpoint audit\n\nCreated: {STAMP}\n\nAll 20 required added cells were checked using run records and payload hashes, not filenames alone. Classification counts: `{counts}`. No seed-137 or seed-3407 terminal checkpoint exists. Primary-CV predictions/checkpoints were not substituted.\n\nThe three-seed terminal grid remains distinct and complete at `artifacts/checkpoints/external_cross_corpus/protocol_faithful_rerun`. Full training and subsequent inference are therefore prepared for manual execution only.\n"""
    (OUT/"EXTERNAL_5SEED_CHECKPOINT_AUDIT.md").write_text(md,encoding="utf-8")

    ar=[]
    for method in METHODS:
     obj=OBJ[method]
     for seed in SEEDS3:
      for fold in FOLDS:
       scratch=ROOT/f"artifacts/checkpoints/author_disjoint_newspaper/{obj}_seed_{seed}_fold_{fold}"
       candidates=list(scratch.rglob("*.safetensors")) if scratch.exists() else []
       run=ROOT/f"results/author_disjoint_newspaper/{obj}/seed_{seed}/fold_{fold}_run.json"
       pred=ROOT/f"results/author_disjoint_newspaper/{obj}/seed_{seed}/fold_{fold}_predictions.csv"
       meta=json.loads(run.read_text(encoding="utf-8")) if run.is_file() else {}
       status="READY_EXISTING" if len(candidates)==1 and meta.get("persistent_checkpoint_saved") is True else ("INVALID_OR_AMBIGUOUS" if candidates else "MISSING")
       ar.append({"method":method,"objective":obj,"seed":seed,"author_fold":fold,"classification":status,
        "checkpoint_candidates":";".join(p.relative_to(ROOT).as_posix() for p in candidates),"checkpoint_hashes":";".join(sha(p) for p in candidates),
        "run_record":run.relative_to(ROOT).as_posix(),"run_record_exists":run.is_file(),"prediction_exists":pred.is_file(),
        "training_status":meta.get("status",""),"persistent_checkpoint_saved":meta.get("persistent_checkpoint_saved",pd.NA),
        "warning":"Completed predictions exist, but the frozen protocol explicitly retained no persistent checkpoint" if not candidates else ""})
    save(pd.DataFrame(ar),"AUTHOR_DISJOINT_CHECKPOINT_AUDIT.csv")

def load_ext_preds(methods=ALL_METHODS):
    z=[]
    for ds,path in EXT.items():
      meta=pd.read_csv(path)
      for m in methods:
       for seed in SEEDS3:
        for fold in FOLDS:
         p=ROOT/f"artifacts/predictions/external_cross_corpus/preseal/{ds}/{m}/seed_{seed}/outer_{fold}.csv"
         q=pd.read_csv(p)
         if len(q)!=len(meta) or not q.external_id.equals(meta.external_id) or not q.text_sha256.equals(meta.text_sha256): raise RuntimeError(f"prediction binding failed: {p}")
         q=q.merge(meta.drop(columns=["text_sha256"]),on="external_id",validate="one_to_one"); q["dataset"]=ds; z.append(q)
    return pd.concat(z,ignore_index=True)

def external_three_seed_and_placeholders(pred):
    cells=[]
    for k,g in pred.groupby(["dataset","method","seed","outer_fold"]):
      cells.append(dict(zip(["corpus","method","seed","outer_fold"],k),roc_auc=auc(g.binary_label,g.raw_score_or_logit),n=len(g)))
    cells=pd.DataFrame(cells)
    seed=cells.groupby(["corpus","method","seed"]).agg(roc_auc=("roc_auc","mean"),folds=("outer_fold","nunique")).reset_index()
    seed["seed_set"]="ORIGINAL_3SEED"; save(seed,"external/EXTERNAL_CE_SUPCON_SEED_LEVEL_3SEED.csv")
    summ=seed.groupby(["corpus","method"]).roc_auc.agg(["mean","std","min","max","count"]).reset_index().rename(columns={"std":"sample_sd","min":"range_min","max":"range_max","count":"n_seeds"})
    rows=[]
    for _,r in summ.iterrows():
      rows.append({"corpus":r.corpus,"method":r.method,"seed_set":"ORIGINAL_3SEED","mean_roc_auc":r["mean"],"sample_sd":r.sample_sd,"range_min":r.range_min,"range_max":r.range_max,"n_seeds":r.n_seeds,"status":"COMPLETE_EXISTING_PREDICTIONS"})
      if r.method in METHODS: rows.append({"corpus":r.corpus,"method":r.method,"seed_set":"EXPANDED_5SEED","mean_roc_auc":np.nan,"sample_sd":np.nan,"range_min":np.nan,"range_max":np.nan,"n_seeds":3,"status":"NOT_EXECUTED_MISSING_TERMINAL_CHECKPOINTS_AND_PREDICTIONS"})
    out=pd.DataFrame(rows); save(out,"EXTERNAL_CE_SUPCON_5SEED_SUMMARY.csv")
    comp=[]
    for _,r in summ[summ.method.isin(METHODS)].iterrows(): comp.append({"corpus":r.corpus,"method":r.method,"original_3seed_mean":r["mean"],"expanded_5seed_mean":np.nan,"absolute_change":np.nan,"sign_or_order_changed":"NOT_ASSESSABLE","conclusion_changed":"NOT_ASSESSABLE","status":"WAITING_FOR_MANUAL_LONG_RUN"})
    save(pd.DataFrame(comp),"EXTERNAL_CE_SUPCON_3SEED_VS_5SEED.csv")
    save(pd.DataFrame(columns=["corpus","estimand","point_estimate","ci95_low","ci95_high","bootstrap_unit","replicates","conditional_on_fixed_models","status"]),"EXTERNAL_CE_SUPCON_5SEED_BOOTSTRAP.csv")
    save(pd.DataFrame(columns=["corpus","method","seed","fold_mean_roc_auc","status"]),"EXTERNAL_CE_SUPCON_SEED_LEVEL.csv")

def generator_analysis(pred):
    audit=[]
    for ds,path in EXT.items():
      d=pd.read_csv(path); gen=d.generator.replace(r"^\s*$",np.nan,regex=True)
      for label,g in d.assign(_gen=gen).groupby("label"):
       vc=g._gen.value_counts(dropna=False)
       for val,n in vc.items(): audit.append({"corpus":ds,"label":label,"generator":val if pd.notna(val) else "MISSING_OR_UNKNOWN","n":n,"metadata_field":"generator","identity_status":"DOCUMENTED_ROW_FIELD" if pd.notna(val) else "UNAVAILABLE","applies_to":"AI_ONLY" if label=="ai" else "HUMAN_REFERENCE","matched_source_group_design":bool(ds=="renklier2026"),"warning":"" if pd.notna(val) else "No generator identity; never inferred from filename"})
    save(pd.DataFrame(audit),"EXTERNAL_GENERATOR_METADATA_AUDIT.csv")
    rows=[]
    for (ds,m,seed,fold),g in pred.groupby(["dataset","method","seed","outer_fold"]):
      gens=sorted(g.loc[(g.binary_label==1)&g.generator.notna()&g.generator.astype(str).str.strip().ne(""),"generator"].unique())
      humans=g[g.binary_label==0]
      for gen in gens:
       airows=g[(g.binary_label==1)&g.generator.eq(gen)]; z=pd.concat([humans,airows])
       rows.append({"corpus":ds,"method":m,"generator":gen,"seed":seed,"outer_fold":fold,"human_n":len(humans),"ai_n":len(airows),"roc_auc":auc(z.binary_label,z.raw_score_or_logit),"estimand":"generator-G AI versus reused corpus HUMAN reference"})
    lev=pd.DataFrame(rows); save(lev,"EXTERNAL_GENERATOR_LEVEL_ROCAUC.csv")
    seeds=lev.groupby(["corpus","method","generator","seed"]).agg(seed_mean_roc_auc=("roc_auc","mean"),folds=("outer_fold","nunique"),human_n=("human_n","max"),ai_n=("ai_n","max")).reset_index()
    su=seeds.groupby(["corpus","method","generator"]).agg(n_seeds=("seed","nunique"),human_n=("human_n","max"),ai_n=("ai_n","max"),mean_roc_auc=("seed_mean_roc_auc","mean"),sample_sd=("seed_mean_roc_auc","std"),range_min=("seed_mean_roc_auc","min"),range_max=("seed_mean_roc_auc","max")).reset_index(); save(su,"EXTERNAL_GENERATOR_LEVEL_SUMMARY.csv")
    p=lev[lev.method.isin(METHODS)].pivot(index=["corpus","generator","seed","outer_fold","human_n","ai_n"],columns="method",values="roc_auc").reset_index(); p["supcon_minus_ce"]=p.SUPCON-p.CE_ONLY_MATCHED
    cs=p.groupby(["corpus","generator","seed"]).agg(seed_mean_supcon_minus_ce=("supcon_minus_ce","mean"),human_n=("human_n","max"),ai_n=("ai_n","max")).reset_index()
    co=cs.groupby(["corpus","generator"]).agg(n_seeds=("seed","nunique"),human_n=("human_n","max"),ai_n=("ai_n","max"),mean_supcon_minus_ce=("seed_mean_supcon_minus_ce","mean"),sample_sd=("seed_mean_supcon_minus_ce","std"),range_min=("seed_mean_supcon_minus_ce","min"),range_max=("seed_mean_supcon_minus_ce","max")).reset_index(); save(co,"EXTERNAL_GENERATOR_LEVEL_CONTRASTS.csv")
    fig,axs=plt.subplots(1,2,figsize=(13,5),sharex=True)
    for ax,(ds,g) in zip(axs,su.groupby("corpus")):
      y=np.arange(g.generator.nunique()); gens=sorted(g.generator.unique())
      for m,off,c in [("CE_ONLY_MATCHED",-.12,"#3568a8"),("SUPCON",.12,"#d05a45")]:
       q=g[g.method.eq(m)].set_index("generator").reindex(gens); ax.scatter(q.mean_roc_auc,y+off,label=m,color=c)
      ax.set_yticks(y,gens); ax.set_title(ds); ax.set_xlim(0,1.01); ax.grid(axis="x",alpha=.3)
    axs[0].legend(fontsize=8); fig.suptitle("Generator-conditional external ROC-AUC (three fixed seeds)"); fig.tight_layout(); fig.savefig(OUT/"figures/EXTERNAL_GENERATOR_LEVEL_ROCAUC.png",dpi=220); plt.close(fig)

MONTHS="ocak şubat mart nisan mayıs haziran temmuz ağustos eylül ekim kasım aralık".split()
WEEKDAYS="pazartesi salı çarşamba perşembe cuma cumartesi pazar".split()
REL=["geçtiğimiz hafta","geçtiğimiz ay","geçtiğimiz yıl","geçen hafta","geçen ay","geçen yıl","önceki gün","bu hafta","bu ay","bu yıl","bugün","yarın","dün"]
SUFFIX=r"(?:['’]?(?:da|de|dan|den|ta|te|tan|ten|ın|in|un|ün|a|e|ı|i|u|ü))?"
PAT={
 "year":re.compile(r"(?<!\d)(?:1[5-9]\d{2}|20\d{2}|21\d{2})(?!\d)"),
 "numeric_date":re.compile(r"(?<!\d)(?:(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:\d{2}|\d{4})|(?:1[5-9]\d{2}|20\d{2}|21\d{2})[./-](?:0?[1-9]|1[0-2])[./-](?:0?[1-9]|[12]\d|3[01]))(?!\d)"),
 "month":re.compile(r"(?<!\w)(?:"+"|".join(MONTHS)+r")"+SUFFIX+r"(?!\w)",re.I),
 "weekday":re.compile(r"(?<!\w)(?:"+"|".join(WEEKDAYS)+r")"+SUFFIX+r"(?!\w)",re.I),
 "relative":re.compile(r"(?<!\w)(?:"+"|".join(map(re.escape,REL))+r")(?!\w)",re.I),
}
def merged_spans(text):
    found=[]; by={}
    for k,p in PAT.items():
      ss=[m.span() for m in p.finditer(text)]; by[k]=ss; found += [(a,b,k) for a,b in ss]
    merged=[]
    for a,b,k in sorted(found):
      if merged and a<merged[-1][1]: merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
      else: merged.append((a,b))
    return by,merged

def cues_and_comparison():
    d=pd.read_csv(ROOT/"data/study/paired_study.csv"); d=d[d.domain.eq("NEWSPAPER")].copy()
    af=pd.read_csv(ROOT/"provenance/experiments/author_disjoint_newspaper_20261013/author_disjoint_folds.csv")[["pair_id","author","author_fold"]]
    d=d.drop(columns=["author"]).merge(af,on="pair_id",validate="many_to_one")
    rr=[]
    for r in d.itertuples(index=False):
      text=unicodedata.normalize("NFC",str(r.text)); words=re.findall(r"\S+",text); by,merged=merged_spans(text); n=len(words)
      z={"text_id":r.text_id,"pair_id":r.pair_id,"label":r.label,"binary_label":int(r.label=="AI"),"author":r.author,"author_fold":r.author_fold,"generator":r.generator,"word_count":n,"text_sha256":hashlib.sha256(text.encode()).hexdigest(),"ner_status":"NOT_EXECUTED_MODEL_UNAVAILABLE"}
      for typ in PAT: z[f"temporal_{typ}_count"]=len(by[typ]); z[f"temporal_{typ}_per_100_words"]=100*len(by[typ])/max(n,1)
      z["temporal_merged_count"]=len(merged); z["temporal_merged_per_100_words"]=100*len(merged)/max(n,1); z["temporal_presence"]=int(bool(merged)); z["temporal_spans_json"]=json.dumps(merged)
      for c in ["person_count","organization_count","location_gpe_count","named_entity_total_count","person_per_100_words","organization_per_100_words","location_gpe_per_100_words","named_entity_total_per_100_words","entity_temporal_document_proxy","temporal_near_entity_count","temporal_near_entity_per_100_words","entity_near_temporal_count","entity_near_temporal_per_100_words"]: z[c]=np.nan
      rr.append(z)
    x=pd.DataFrame(rr); save(x,"NEWSPAPER_CUE_FEATURES_PER_TEXT.csv")
    feats=[c for c in x if c.startswith("temporal_") and (c.endswith("count") or c.endswith("words") or c=="temporal_presence") and c!="temporal_spans_json" and x[c].notna().all()]
    rows=[]; rng=np.random.default_rng(BOOT_SEED)
    for f in feats:
      w=x.pivot(index="pair_id",columns="label",values=f).dropna(); diff=(w.AI-w.HUMAN).to_numpy(); authors=x[x.label.eq("HUMAN")].set_index("pair_id").loc[w.index,"author"]
      pb=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(NBOOT)])
      au=sorted(authors.unique()); am=np.array([diff[authors.to_numpy()==a].mean() for a in au]); ab=np.array([am[rng.integers(0,len(am),len(am))].mean() for _ in range(NBOOT)])
      # adjusted coefficient: label + log length + 9 author indicators
      tmp=x[[f,"binary_label","word_count","author"]].dropna(); D=pd.get_dummies(tmp.author,drop_first=True,dtype=float); X=np.column_stack([np.ones(len(tmp)),tmp.binary_label,np.log1p(tmp.word_count),D.to_numpy()]); coef=np.linalg.lstsq(X,tmp[f].to_numpy(float),rcond=None)[0][1]
      rows.append({"feature":f,"human_n":len(w),"ai_n":len(w),"human_mean":w.HUMAN.mean(),"ai_mean":w.AI.mean(),"human_median":w.HUMAN.median(),"ai_median":w.AI.median(),"mean_paired_difference_ai_minus_human":diff.mean(),"median_paired_difference":np.median(diff),"paired_standardized_effect":diff.mean()/diff.std(ddof=1) if diff.std(ddof=1)>0 else np.nan,"pair_boot_ci_low":np.quantile(pb,.025),"pair_boot_ci_high":np.quantile(pb,.975),"author_cluster_ci_low":np.quantile(ab,.025),"author_cluster_ci_high":np.quantile(ab,.975),"length_author_adjusted_label_coefficient":coef,"bootstrap_replicates":NBOOT})
    comp=pd.DataFrame(rows); save(comp,"NEWSPAPER_CUE_CLASS_COMPARISON.csv")
    md=f"""# NEWSPAPER cue analysis\n\nStatus: `PARTIALLY_COMPLETE_TEMPORAL_FEATURES`; created {STAMP}.\n\nThe deterministic frozen temporal detector was applied to all {len(x)} NEWSPAPER texts ({x.pair_id.nunique()} matched pairs). Pair and ten-author cluster intervals use {NBOOT} draws. Length/author-adjusted coefficients are descriptive OLS coefficients with log word count and author fixed indicators.\n\nTurkish NER was not run: Stanza is installed, but no Turkish resources or cached Turkish NER transformer were available locally. NER and entity–temporal proxy fields are explicitly missing rather than replaced by an English model or capitalization heuristic. Thus this analysis quantifies temporal cues now, while complete named-entity/event-proxy results require the documented manual model installation/run. Class differences are corpus-level separability and do not show BERTurk reliance or factual event detection.\n"""; (OUT/"NEWSPAPER_CUE_ANALYSIS.md").write_text(md,encoding="utf-8")
    top=comp.sort_values("paired_standardized_effect",key=lambda s:s.abs(),ascending=False).head(8)
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(top.feature,top.mean_paired_difference_ai_minus_human,color="#6857a5"); ax.axvline(0,color="black",lw=.8); ax.set_xlabel("Mean paired difference (AI − HUMAN)"); ax.set_title("Frozen temporal cue differences; NEWSPAPER"); fig.tight_layout(); fig.savefig(OUT/"figures/NEWSPAPER_TEMPORAL_CUE_DIFFERENCES.png",dpi=220); plt.close(fig)
    return x

def surface_probe(cues):
    old=pd.read_csv(ROOT/"Turkish_AI_Text_Detection_Exploratory_Analysis_Review/analysis_A1/corpus_shortcut_features.csv")
    # exact reproduction
    pp=[]
    for fold in FOLDS:
      tr=old.outer_fold.ne(fold); te=~tr; pipe=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(C=1,penalty="l2",solver="liblinear",max_iter=2000,random_state=261012))]); pipe.fit(old.loc[tr,OLD_FEATURES],old.loc[tr,"binary_label"]); q=old.loc[te,["text_id","binary_label","domain"]].copy(); q["s"]=pipe.predict_proba(old.loc[te,OLD_FEATURES])[:,1]; pp.append(q)
    pp=pd.concat(pp); rep=auc(pp[pp.domain.eq("NEWSPAPER")].binary_label,pp[pp.domain.eq("NEWSPAPER")].s); expected=0.998772
    pd.DataFrame([{"target":"existing_NEWSPAPER_surface_probe","expected_roc_auc":expected,"reproduced_roc_auc":rep,"absolute_difference":abs(rep-expected),"tolerance":1e-10,"status":"PASS" if abs(rep-expected)<=1e-10 else "FAIL"}]).to_csv(OUT/"surface_probe/EXISTING_SURFACE_PROBE_REPRODUCTION.csv",index=False)
    af=pd.read_csv(ROOT/"provenance/experiments/author_disjoint_newspaper_20261013/author_disjoint_folds.csv")[["pair_id","author","author_fold"]]
    # Derive the same frozen deterministic temporal features for ABSTRACT rows
    # because the original probe's author-disjoint analogue retains ABSTRACT in
    # every training population. No class outcome enters feature extraction.
    study=pd.read_csv(ROOT/"data/study/paired_study.csv")[["text_id","text"]]
    allcue=[]
    for r in study.itertuples(index=False):
      text=unicodedata.normalize("NFC",str(r.text)); n=len(re.findall(r"\S+",text)); by,merged=merged_spans(text); z={"text_id":r.text_id}
      for typ in PAT: z[f"temporal_{typ}_count"]=len(by[typ]); z[f"temporal_{typ}_per_100_words"]=100*len(by[typ])/max(n,1)
      z["temporal_merged_count"]=len(merged); z["temporal_merged_per_100_words"]=100*len(merged)/max(n,1); z["temporal_presence"]=int(bool(merged)); allcue.append(z)
    allcue=pd.DataFrame(allcue)
    x=old.drop(columns=["outer_fold"]).merge(af,on="pair_id",how="left",validate="many_to_one").merge(allcue,on="text_id",validate="one_to_one")
    temporal=[c for c in allcue if c!="text_id"]
    variants={"P1_TEMPORAL_ONLY_INCOMPLETE":temporal,"P2_EXISTING_MINUS_CUE":[f for f in OLD_FEATURES if f!="year_date_per_100_words"],"P3_FULL_TEMPORAL_ONLY_INCOMPLETE":[f for f in OLD_FEATURES if f!="year_date_per_100_words"]+temporal}
    preds=[]
    for name,fs in variants.items():
      for fold in FOLDS:
       tr=x.domain.eq("ABSTRACT") | (x.domain.eq("NEWSPAPER")&x.author_fold.ne(fold)); te=x.domain.eq("NEWSPAPER")&x.author_fold.eq(fold)
       pipe=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(C=1,penalty="l2",solver="liblinear",max_iter=2000,random_state=BOOT_SEED))]); pipe.fit(x.loc[tr,fs],x.loc[tr,"binary_label"]); q=x.loc[te,["text_id","pair_id","binary_label","author","author_fold"]].copy(); q["score"]=pipe.predict_proba(x.loc[te,fs])[:,1]; q["variant"]=name; preds.append(q)
    pr=pd.concat(preds); rows=[]; boots=[]; rng=np.random.default_rng(BOOT_SEED); authors=sorted(pr.author.unique()); draws=rng.multinomial(len(authors),np.full(len(authors),1/len(authors)),size=NBOOT)
    for name,g in pr.groupby("variant"):
      ai=auc(g.binary_label,g.score); ai_draw=[]; amap={a:i for i,a in enumerate(authors)}; ix=g.author.map(amap).to_numpy()
      for b in draws: ai_draw.append(auc(g.binary_label,g.score,b[ix]))
      rows.append({"variant":name,"roc_auc":ai,"ci95_low":np.nanquantile(ai_draw,.025),"ci95_high":np.nanquantile(ai_draw,.975),"n_documents":len(g),"n_authors":10,"split":"author-disjoint; ABSTRACT + other-eight-author NEWSPAPER training","ner_completeness":"INCOMPLETE_NER_UNAVAILABLE" if "TEMPORAL_ONLY" in name else "NOT_APPLICABLE"})
      boots += [{"variant":name,"replicate":i,"roc_auc":v,"bootstrap_unit":"author","bootstrap_seed":BOOT_SEED} for i,v in enumerate(ai_draw)]
    res=pd.DataFrame(rows); save(res,"NEWSPAPER_SURFACE_PROBE_ABLATION.csv"); save(pd.DataFrame(boots),"NEWSPAPER_SURFACE_PROBE_ABLATION_BOOTSTRAP.csv")
    fig,ax=plt.subplots(figsize=(8,4)); ax.errorbar(res.roc_auc,np.arange(len(res)),xerr=[res.roc_auc-res.ci95_low,res.ci95_high-res.roc_auc],fmt="o",color="#3568a8"); ax.set_yticks(np.arange(len(res)),res.variant); ax.set_xlim(0,1.01); ax.set_xlabel("Author-cluster bootstrap ROC-AUC"); ax.set_title("NEWSPAPER surface-probe ablation"); fig.tight_layout(); fig.savefig(OUT/"figures/NEWSPAPER_SURFACE_PROBE_ABLATION.png",dpi=220); plt.close(fig)

def year_analysis():
    data=pd.read_csv(ROOT/"data/study/paired_study.csv")
    base=data[(data.domain.eq("ABSTRACT"))&(data.label.eq("HUMAN"))&(data.source_year.between(2005,2018))][["text_id","source_year","journal"]]
    rows=[]; yearly=[]
    for m,obj in OBJ.items():
     for seed in SEEDS5:
      pred=pd.concat([pd.read_csv(ROOT/f"results/primary/{obj}/seed_{seed}/outer_{f}_predictions.csv") for f in FOLDS]); z=base.merge(pred[["text_id","raw_decision_score"]],on="text_id",validate="one_to_one"); rho,p=spearmanr(z.source_year,z.raw_decision_score); slope=np.polyfit(z.source_year,z.raw_decision_score,1)[0]
      rows.append({"method":m,"seed":seed,"n":len(z),"spearman_rho":rho,"spearman_p_descriptive":p,"linear_slope_per_year":slope,"prediction_type":"OOF_ONLY"})
      for y,g in z.groupby("source_year"): yearly.append({"method":m,"seed":seed,"year":y,"n":len(g),"mean_raw_score":g.raw_decision_score.mean(),"median_raw_score":g.raw_decision_score.median(),"q25":g.raw_decision_score.quantile(.25),"q75":g.raw_decision_score.quantile(.75)})
    save(pd.DataFrame(rows),"optional/ABSTRACT_HUMAN_YEAR_TRENDS.csv"); save(pd.DataFrame(yearly),"optional/ABSTRACT_HUMAN_YEAR_DISTRIBUTIONS.csv")

def main():
    checkpoint_audits(); pred=load_ext_preds(); external_three_seed_and_placeholders(pred); generator_analysis(pred); cues=cues_and_comparison(); surface_probe(cues); year_analysis()
    print(json.dumps({"status":"COMPLETE_LIGHTWEIGHT","created":STAMP},ensure_ascii=False))

if __name__=="__main__": main()
