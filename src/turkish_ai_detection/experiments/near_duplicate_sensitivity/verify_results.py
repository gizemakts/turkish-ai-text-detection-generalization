# -*- coding: utf-8 -*-
"""Post-run pre-unblinding integrity verifier for RQ5. Computes no scientific metrics."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import numpy as np, pandas as pd
def _resolve_project_root() -> Path:
    for _candidate in Path(__file__).resolve().parents:
        if (_candidate / ".gitignore").is_file():
            return _candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study project root could not be resolved")

ROOT = _resolve_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
CONFIG=ROOT/'configs'/'near_duplicate_sensitivity.json'; FREEZE=ROOT/'provenance'/'experiments'/'near_duplicate_sensitivity'/'SPLIT_PREEXECUTION_FREEZE.json'; SPLIT=ROOT/'data'/'study'/'cluster_aware_folds_v1.csv'; OUT=ROOT/'results'/'near_duplicate_sensitivity'; REPORT=ROOT/'provenance'/'experiments'/'near_duplicate_sensitivity'/'RUN_INTEGRITY.json'
if str(ROOT/'src') not in sys.path: sys.path.insert(0,str(ROOT/'src'))
from turkish_ai_detection.data.study import resolve_frozen_record_path  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import expected_run_hashes, validate_current_migrated_reproduction  # noqa: E402
RUNNER=ROOT/'src'/'turkish_ai_detection'/'experiments'/'near_duplicate_sensitivity'/'run_fit.py'
def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def req(x,m):
    if not x: raise RuntimeError(m)
def main():
    req(FREEZE.is_file(),'missing scientific freeze'); fr=json.loads(FREEZE.read_text(encoding='utf-8')); cfg=json.loads(CONFIG.read_text(encoding='utf-8')); split=pd.read_csv(SPLIT)
    req(fr.get('status')=='FROZEN_READY_TO_RUN','invalid freeze status'); req(fr.get('rq5_results_unblinded_before_freeze') is False,'blinding not certified')
    for k,r in fr['files'].items():
        if k not in {'config','scientific_runner'}: resolve_frozen_record_path(ROOT,r['path'],r['sha256'])
    policy=validate_current_migrated_reproduction(ROOT,'near_duplicate_sensitivity',RUNNER)
    source={(r['objective'],int(r['seed']),int(r['outer_fold'])):r for r in fr['source_historical_primary_artifacts']}; req(len(source)==50,'historical source cardinality')
    for key,r in source.items(): req(sha(ROOT/r['prediction']['path'])==r['prediction']['sha256'],f'historical prediction changed {key}')
    verified=0; pooled={}
    for objective in ('ce','supcon'):
        for seed in cfg['planned_scientific_evaluation_after_split_freeze']['seeds']:
            parts=[]
            for outer in range(5):
                base=OUT/objective/f'seed_{seed}'; pred=base/f'outer_{outer}_predictions.csv'; side=base/f'outer_{outer}_run.json'; req(pred.is_file() and side.is_file(),f'missing RQ5 fit {objective}/{seed}/{outer}')
                run=json.loads(side.read_text(encoding='utf-8')); req(run.get('status')=='COMPLETE','incomplete run'); req(run.get('scientific_metrics_computed') is False,'metrics computed in runner'); req(run.get('outer_test_rows_used_for_training')==0 and run.get('outer_test_used_for_model_selection') is False,'outer leakage')
                expected_config,expected_runner=expected_run_hashes(run,fr,policy,config_key='config',runner_key='scientific_runner')
                req(run.get('runner_sha256')==expected_runner,'runner binding'); req(run.get('config_sha256')==expected_config,'config binding'); req(run.get('cluster_aware_split_sha256')==fr['files']['cluster_aware_split']['sha256'],'split binding'); req(run.get('preexecution_freeze_sha256')==sha(FREEZE),'freeze binding'); req(run.get('prediction_sha256')==sha(pred),'prediction hash')
                frame=pd.read_csv(pred); need={'text_id','pair_id','label','domain','generator','near_duplicate_cluster_id','outer_fold','seed','objective','condition','raw_decision_score'}; req(need<=set(frame.columns),'schema'); req(len(frame)==600 and frame.text_id.nunique()==600 and frame.pair_id.nunique()==300,'shape'); req(np.isfinite(frame.raw_decision_score.to_numpy(float)).all(),'nonfinite')
                exp=set(split.loc[split.outer_fold.eq(outer),'pair_id']); req(set(frame.pair_id)==exp,'test pair identity'); req(frame.near_duplicate_cluster_id.astype(str).equals(frame.pair_id.map(split.set_index('pair_id').near_duplicate_cluster_id.astype(str).to_dict()).astype(str)),'cluster id mismatch')
                parts.append(frame); verified+=1
            allf=pd.concat(parts,ignore_index=True).sort_values('text_id').reset_index(drop=True); req(len(allf)==3000 and allf.text_id.nunique()==3000 and allf.pair_id.nunique()==1500,'OOF completeness'); pooled[(objective,seed)]=allf[['text_id','pair_id','label','near_duplicate_cluster_id']]
    req(verified==50,'fit count')
    for seed in cfg['planned_scientific_evaluation_after_split_freeze']['seeds']: req(pooled[('ce',seed)].equals(pooled[('supcon',seed)]),f'CE/SupCon alignment {seed}')
    final=OUT/'final_statistics'; req(not final.exists() or not any(final.iterdir()),'final statistics already exist')
    rec={'schema_version':'combined-v2-near-duplicate-completion-integrity-v1','status':'PASS','results_unblinded':False,'scientific_metrics_computed':False,'verified_fits':50,'verified_predictions':50,'rows_per_fit':600,'pairs_per_fit':300,'pooled_rows_per_seed_objective':3000,'pooled_pairs_per_seed_objective':1500,'historical_primary_inputs_verified':50,'prediction_hashes_verified':True,'freeze_file_bindings_verified':True,'cluster_disjoint_train_test_verified':True,'finite_scores_verified_without_revealing_values':True,'ce_supcon_alignment_verified':True,'freeze_sha256':sha(FREEZE),'active_migrated_config_sha256':policy['current_config_sha256'],'active_migrated_runner_sha256':policy['current_runner_sha256'],'historical_exact_reproduction':policy['historical_exact_reproduction'],'next_step':'Run src/turkish_ai_detection/experiments/near_duplicate_sensitivity/summarize_statistics.py for the prespecified one-time RQ5 unblinding.'}
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(rec,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
