# -*- coding: utf-8 -*-
"""Single-fit runner for COMBINED_V2_NEAR_DUPLICATE_SENSITIVITY_V1.

Uses the primary CE/SupCon training recipe unchanged. The only
scientific change is replacement of the historical pair-aware outer split by
the prespecified cluster-aware outer split. No metrics/calibration/thresholds
are computed by this runner.
"""
from __future__ import annotations

import sys
import argparse, gc, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

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
from turkish_ai_detection.data.study import sha256_file, write_json  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import validate_current_migrated_reproduction  # noqa: E402
from turkish_ai_detection.evaluation.metrics import condition_text  # noqa: E402
from turkish_ai_detection.experiments.baseline.berturk import MODEL_NAME, REVISION, encode_head_tail  # noqa: E402
from turkish_ai_detection.experiments.model_development.supcon.train import BoundedCollator, DATASET_SHA256, DATA_PATH, build_model  # noqa: E402
from turkish_ai_detection.experiments.primary.run_fit import (ALLOWED_OUTERS, ALLOWED_SEEDS, atomic_csv, make_dataset,
    make_seeded_trainer_class, pair_hash, score, training_arguments)  # noqa: E402

EXPERIMENT='COMBINED_V2_NEAR_DUPLICATE_SENSITIVITY_V1'; CONDITION='A'
CONFIG=ROOT/'configs'/'near_duplicate_sensitivity.json'
SPLIT=ROOT/'data'/'study'/'cluster_aware_folds_v1.csv'
FREEZE=ROOT/'provenance'/'experiments'/'near_duplicate_sensitivity'/'SPLIT_PREEXECUTION_FREEZE.json'
OUT=ROOT/'results'/'near_duplicate_sensitivity'; STATE=ROOT/'artifacts'/'checkpoints'/'near_duplicate_sensitivity'

def utc_now(): return datetime.now(timezone.utc).isoformat()

def verify_freeze():
    if not FREEZE.is_file(): raise RuntimeError('RQ5 execution locked until SPLIT_PREEXECUTION_FREEZE.json exists')
    freeze=json.loads(FREEZE.read_text(encoding='utf-8'))
    if freeze.get('status')!='FROZEN_READY_TO_RUN': raise RuntimeError('invalid RQ5 scientific freeze status')
    freeze['_current_reproduction']=validate_current_migrated_reproduction(ROOT,'near_duplicate_sensitivity',Path(__file__))
    if freeze['files']['cluster_aware_split']['sha256']!=sha256_file(SPLIT): raise RuntimeError('RQ5 split changed after freeze')
    if freeze.get('scientific_rq5_outputs_present_before_freeze') is not False: raise RuntimeError('freeze does not certify clean RQ5 state')
    if freeze.get('rq5_results_unblinded_before_freeze') is not False: raise RuntimeError('freeze does not certify RQ5 result blinding')
    return freeze

def run_one(objective:int|str, seed:int, outer:int):
    import torch, transformers
    from transformers import AutoTokenizer, set_seed
    objective=str(objective)
    if not torch.cuda.is_available(): raise RuntimeError('RQ5 scientific fits require CUDA')
    if objective not in {'ce','supcon'}: raise ValueError(objective)
    if seed not in ALLOWED_SEEDS: raise ValueError(f'seed not prespecified: {seed}')
    if outer not in ALLOWED_OUTERS: raise ValueError(f'outer fold not prespecified: {outer}')
    freeze=verify_freeze(); cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    if sha256_file(DATA_PATH)!=cfg['dataset_sha256'] or sha256_file(DATA_PATH)!=DATASET_SHA256: raise RuntimeError('dataset identity mismatch')
    if sha256_file(SPLIT)!=freeze['files']['cluster_aware_split']['sha256']: raise RuntimeError('cluster split identity mismatch')
    data=pd.read_csv(DATA_PATH).copy(); folds=pd.read_csv(SPLIT)
    if len(data)!=3000 or data.text_id.nunique()!=3000 or data.pair_id.nunique()!=1500: raise RuntimeError('Combined-v2 identity/count changed')
    need={'pair_id','domain','generator','stratum','near_duplicate_cluster_id','cluster_pair_count','is_multi_pair_cluster','outer_fold'}
    if not need<=set(folds.columns) or len(folds)!=1500 or folds.pair_id.nunique()!=1500: raise RuntimeError('cluster split schema/count mismatch')
    data['model_text']=condition_text(data,CONDITION)
    test_pairs=set(folds.loc[folds.outer_fold.eq(outer),'pair_id']); train_pairs=set(folds.loc[folds.outer_fold.ne(outer),'pair_id'])
    if len(test_pairs)!=300 or len(train_pairs)!=1200 or train_pairs&test_pairs: raise RuntimeError('invalid cluster-aware train/test partition')
    cluster_map=folds.set_index('pair_id')['near_duplicate_cluster_id'].astype(str).to_dict()
    train_clusters={cluster_map[p] for p in train_pairs}; test_clusters={cluster_map[p] for p in test_pairs}
    if train_clusters & test_clusters: raise RuntimeError('near-duplicate cluster split across train/test')
    train=data[data.pair_id.isin(train_pairs)].copy(); test=data[data.pair_id.isin(test_pairs)].copy()
    if len(train)!=2400 or len(test)!=600: raise RuntimeError('unexpected RQ5 row counts')
    base=OUT/objective/f'seed_{seed}'; pred_path=base/f'outer_{outer}_predictions.csv'; side=base/f'outer_{outer}_run.json'; tag=f'{objective}_seed_{seed}_outer_{outer}'
    binding={'experiment':EXPERIMENT,'objective':objective,'seed':int(seed),'outer_fold':int(outer),'condition':CONDITION,
      'dataset_sha256':sha256_file(DATA_PATH),'cluster_aware_split_sha256':sha256_file(SPLIT),'config_sha256':sha256_file(CONFIG),
      'runner_sha256':sha256_file(Path(__file__)),'preexecution_freeze_sha256':sha256_file(FREEZE),
      'reproduction_mode':freeze['_current_reproduction']['reproduction_mode'],
      'historical_exact_reproduction':freeze['_current_reproduction']['historical_exact_reproduction'],
      'train_pair_count':1200,'test_pair_count':300,
      'train_pair_sha256':pair_hash(train_pairs),'test_pair_sha256':pair_hash(test_pairs),'train_test_pair_overlap':0,
      'train_cluster_count':int(len(train_clusters)),'test_cluster_count':int(len(test_clusters)),'train_test_cluster_overlap':0}
    if pred_path.exists() or side.exists():
        if not pred_path.exists() or not side.exists(): raise RuntimeError(f'incomplete resume state: {tag}')
        saved=json.loads(side.read_text(encoding='utf-8')); expected={**binding,'prediction_sha256':sha256_file(pred_path)}
        if {k:saved.get(k) for k in expected}!=expected: raise RuntimeError(f'resume binding mismatch: {tag}')
        return saved
    set_seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,revision=REVISION)
    encodings=[encode_head_tail(tokenizer,t) for t in data.model_text.astype(str)]; collator=BoundedCollator(tokenizer)
    model=build_model(); TrainerClass=make_seeded_trainer_class(objective,seed)
    trainer=TrainerClass(model=model,args=training_arguments(STATE/tag,seed),train_dataset=make_dataset(train,encodings),data_collator=collator)
    trainer.train()
    if trainer.last_group_audit is None: raise RuntimeError('no completed true-32 objective group observed')
    raw=score(model,test,encodings,collator)
    pred=test[['text_id','pair_id','label','domain','generator']].copy(); pred['near_duplicate_cluster_id']=pred.pair_id.map(cluster_map)
    pred['outer_fold']=int(outer); pred['seed']=int(seed); pred['objective']=objective; pred['condition']=CONDITION; pred['raw_decision_score']=raw
    if len(pred)!=600 or pred.text_id.nunique()!=600 or pred.pair_id.nunique()!=300 or pred.near_duplicate_cluster_id.isna().any(): raise RuntimeError('invalid RQ5 prediction shape')
    if not np.isfinite(pred.raw_decision_score.to_numpy(float)).all(): raise RuntimeError('non-finite RQ5 score')
    atomic_csv(pred,pred_path)
    record={**binding,'schema_version':'combined-v2-near-duplicate-single-fit-v1','created_utc':utc_now(),'status':'COMPLETE','model':MODEL_NAME,'revision':REVISION,
      'calibration_used':False,'threshold_used':False,'outer_test_rows_used_for_training':0,'outer_test_used_for_model_selection':False,
      'scientific_metrics_computed':False,'last_group_audit':trainer.last_group_audit,'prediction_sha256':sha256_file(pred_path),
      'torch_version':torch.__version__,'transformers_version':transformers.__version__,'cuda_version':torch.version.cuda,'gpu_model':torch.cuda.get_device_name(0)}
    write_json(side,record); del trainer,model; gc.collect(); torch.cuda.empty_cache(); return record

def main():
    p=argparse.ArgumentParser(); p.add_argument('--objective',required=True,choices=['ce','supcon']); p.add_argument('--seed',required=True,type=int); p.add_argument('--outer-fold',required=True,type=int); a=p.parse_args()
    r=run_one(a.objective,a.seed,a.outer_fold); print(json.dumps({'status':r['status'],'experiment':r['experiment'],'objective':r['objective'],'seed':r['seed'],'outer_fold':r['outer_fold'],'prediction_sha256':r['prediction_sha256'],'scientific_metrics_computed':False},ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
