# -*- coding: utf-8 -*-
"""One-time RQ5 statistics/unblinding with near-duplicate-cluster bootstrap."""
from __future__ import annotations

import sys
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
def _resolve_project_root() -> Path:
    for _candidate in Path(__file__).resolve().parents:
        if (_candidate / ".gitignore").is_file():
            return _candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study project root could not be resolved")

ROOT = _resolve_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SCRIPTS=ROOT/'src'
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from turkish_ai_detection.evaluation.metrics import _pauc  # noqa: E402
CONFIG=ROOT/'configs'/'near_duplicate_sensitivity.json'; FREEZE=ROOT/'provenance'/'experiments'/'near_duplicate_sensitivity'/'SPLIT_PREEXECUTION_FREEZE.json'; VERIFY=ROOT/'provenance'/'experiments'/'near_duplicate_sensitivity'/'RUN_INTEGRITY.json'; SPLIT=ROOT/'data'/'study'/'cluster_aware_folds_v1.csv'; OUT=ROOT/'results'/'near_duplicate_sensitivity'; HIST=ROOT/'results'/'primary'; FINAL=OUT/'final_statistics'
def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def req(x,m):
    if not x: raise RuntimeError(m)
def metric(y,s): return {'roc_auc':float(roc_auc_score(y,s)),'average_precision':float(average_precision_score(y,s)),'standardized_pauc_003':float(_pauc(y,s))}
def load_pooled(base, objective, seed, include_cluster=False):
    parts=[]
    for outer in range(5): parts.append(pd.read_csv(base/objective/f'seed_{seed}'/f'outer_{outer}_predictions.csv'))
    f=pd.concat(parts,ignore_index=True).sort_values('text_id').reset_index(drop=True); req(len(f)==3000 and f.text_id.nunique()==3000,'OOF completeness')
    if include_cluster: req('near_duplicate_cluster_id' in f.columns,'cluster id missing')
    return f
def main():
    req(VERIFY.is_file(),'missing pre-unblinding verifier'); vr=json.loads(VERIFY.read_text(encoding='utf-8')); req(vr.get('status')=='PASS' and vr.get('results_unblinded') is False,'integrity verifier not PASS/blinded')
    fr=json.loads(FREEZE.read_text(encoding='utf-8')); cfg=json.loads(CONFIG.read_text(encoding='utf-8')); plan=cfg['planned_scientific_evaluation_after_split_freeze']; seeds=[int(x) for x in plan['seeds']]; split=pd.read_csv(SPLIT)
    req(not FINAL.exists() or not any(FINAL.iterdir()),'RQ5 statistics already exist; one-time unblinding refuses overwrite')
    pooled={}; rows=[]
    for source,base in [('historical',HIST),('cluster_aware',OUT)]:
        for objective in ('ce','supcon'):
            for seed in seeds:
                f=load_pooled(base,objective,seed,include_cluster=(source=='cluster_aware'))
                if source=='historical': f['near_duplicate_cluster_id']=f.pair_id.map(split.set_index('pair_id').near_duplicate_cluster_id.astype(str).to_dict())
                pooled[(source,objective,seed)]=f
                y=f.label.astype(str).eq('AI').astype(int).to_numpy(); m=metric(y,f.raw_decision_score.to_numpy(float)); rows.append({'split':source,'objective':objective,'seed':seed,**m})
    ref=pooled[('cluster_aware','ce',seeds[0])][['text_id','pair_id','label','near_duplicate_cluster_id']]
    for k,f in pooled.items(): req(f[['text_id','pair_id','label','near_duplicate_cluster_id']].reset_index(drop=True).equals(ref.reset_index(drop=True)),f'alignment failed {k}')
    per_seed=pd.DataFrame(rows)
    contrasts=[]
    metrics=['roc_auc','average_precision','standardized_pauc_003']
    for seed in seeds:
        hce=per_seed[(per_seed.split=='historical')&(per_seed.objective=='ce')&(per_seed.seed==seed)].iloc[0]; hsc=per_seed[(per_seed.split=='historical')&(per_seed.objective=='supcon')&(per_seed.seed==seed)].iloc[0]; cce=per_seed[(per_seed.split=='cluster_aware')&(per_seed.objective=='ce')&(per_seed.seed==seed)].iloc[0]; csc=per_seed[(per_seed.split=='cluster_aware')&(per_seed.objective=='supcon')&(per_seed.seed==seed)].iloc[0]
        for met in metrics:
            contrasts += [
              {'seed':seed,'comparison':'CE_cluster_minus_historical','metric':met,'delta':float(cce[met]-hce[met])},
              {'seed':seed,'comparison':'SUPCON_cluster_minus_historical','metric':met,'delta':float(csc[met]-hsc[met])},
              {'seed':seed,'comparison':'SUPCON_minus_CE_within_cluster','metric':met,'delta':float(csc[met]-cce[met])},
              {'seed':seed,'comparison':'difference_in_differences','metric':met,'delta':float((csc[met]-cce[met])-(hsc[met]-hce[met]))},]
    contrast_df=pd.DataFrame(contrasts)
    seed_summary=per_seed.groupby(['split','objective'])[metrics].agg(['mean','std']).reset_index(); seed_summary.columns=['_'.join([str(x) for x in c if str(x)]) for c in seed_summary.columns]
    contrast_summary=contrast_df.groupby(['comparison','metric']).delta.agg(['mean','std']).reset_index().rename(columns={'mean':'mean_paired_delta_across_seeds','std':'sd_paired_delta_across_seeds'})
    cluster_ids=np.asarray(sorted(ref.near_duplicate_cluster_id.astype(str).unique())); req(len(cluster_ids)==1499,'expected 1499 bootstrap clusters')
    ids=ref.near_duplicate_cluster_id.astype(str).to_numpy(); index_map={c:np.flatnonzero(ids==c) for c in cluster_ids}; yfull=ref.label.astype(str).eq('AI').astype(int).to_numpy(); scores={(src,obj,seed):pooled[(src,obj,seed)].raw_decision_score.to_numpy(float) for src in ('historical','cluster_aware') for obj in ('ce','supcon') for seed in seeds}
    B=int(plan['bootstrap_replicates']); rng=np.random.default_rng(int(plan['bootstrap_seed'])); draws=[]
    for b in range(1,B+1):
        sampled=rng.choice(cluster_ids,size=len(cluster_ids),replace=True); idx=np.concatenate([index_map[c] for c in sampled]); y=yfull[idx]
        acc={(comp,met):[] for comp in ['CE_cluster_minus_historical','SUPCON_cluster_minus_historical','SUPCON_minus_CE_within_cluster','difference_in_differences'] for met in metrics}
        for seed in seeds:
            hm={o:metric(y,scores[('historical',o,seed)][idx]) for o in ('ce','supcon')}; cm={o:metric(y,scores[('cluster_aware',o,seed)][idx]) for o in ('ce','supcon')}
            for met in metrics:
                vals={'CE_cluster_minus_historical':cm['ce'][met]-hm['ce'][met],'SUPCON_cluster_minus_historical':cm['supcon'][met]-hm['supcon'][met],'SUPCON_minus_CE_within_cluster':cm['supcon'][met]-cm['ce'][met],'difference_in_differences':(cm['supcon'][met]-cm['ce'][met])-(hm['supcon'][met]-hm['ce'][met])}
                for comp,v in vals.items(): acc[(comp,met)].append(v)
        row={'bootstrap_id':b}
        for (comp,met),v in acc.items(): row[f'{comp}__{met}']=float(np.mean(v))
        draws.append(row)
    draw_df=pd.DataFrame(draws); boot=[]
    obs={(r.comparison,r.metric):r.mean_paired_delta_across_seeds for r in contrast_summary.itertuples()}
    for comp in ['CE_cluster_minus_historical','SUPCON_cluster_minus_historical','SUPCON_minus_CE_within_cluster','difference_in_differences']:
        for met in metrics:
            v=draw_df[f'{comp}__{met}'].to_numpy(float); boot.append({'comparison':comp,'metric':met,'bootstrap_unit':'near_duplicate_cluster_id','clusters':1499,'replicates':B,'bootstrap_seed':int(plan['bootstrap_seed']),'observed_mean_paired_delta':float(obs[(comp,met)]),'bootstrap_mean_delta':float(v.mean()),'ci95_lower':float(np.quantile(v,.025)),'ci95_upper':float(np.quantile(v,.975))})
    FINAL.mkdir(parents=True,exist_ok=True); per_seed.to_csv(FINAL/'per_seed_pooled_metrics.csv',index=False); seed_summary.to_csv(FINAL/'method_seed_summary.csv',index=False); contrast_df.to_csv(FINAL/'per_seed_contrasts.csv',index=False); contrast_summary.to_csv(FINAL/'contrast_seed_summary.csv',index=False); draw_df.to_csv(FINAL/'cluster_bootstrap_draws.csv',index=False); pd.DataFrame(boot).to_csv(FINAL/'cluster_bootstrap_summary.csv',index=False)
    manifest={'schema_version':'combined-v2-near-duplicate-final-statistics-v1','status':'COMPLETE','unblinded':True,'fits_verified':50,'bootstrap_unit':'near_duplicate_cluster_id','bootstrap_clusters':1499,'seeds':seeds,'config_sha256':sha(CONFIG),'freeze_sha256':sha(FREEZE),'integrity_report_sha256':sha(VERIFY),'outputs':{p.name:sha(p) for p in FINAL.iterdir() if p.is_file()}}
    mp=FINAL/'final_statistics_manifest.json'; mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'status':'COMPLETE','results_are_now_unblinded':True,'fits_verified':50,'final_statistics_dir':str(FINAL),'manifest_sha256':sha(mp)},ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
