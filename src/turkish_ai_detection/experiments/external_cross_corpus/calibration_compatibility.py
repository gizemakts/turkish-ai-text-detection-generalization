# -*- coding: utf-8 -*-
"""Internal-only, pre-external, per-realization calibration-transfer gate.

Run --validate-only while protocol is a draft. The full diagnostic is allowed
ONLY after final 30 CE/SupCon reruns and frozen code/config binding. No external
corpus, external prediction or external label is loaded by this module.

Conservative conditional reuse rule fixed before final reruns:
  * the historical and final *same-realization* internal raw scores must satisfy
    the already-recorded QA closeness criterion (rtol=1e-4, atol=1e-5);
  * frozen-Platt-mapped probabilities must independently satisfy that criterion;
  * all 4 frozen threshold decisions must agree exactly for all 600 rows.
A failure means no mapping transfer for that cell (not a failure of the model).
Passing means a conditional, descriptive transfer; it does not prove external
calibration or operating-FPR performance. Never refit Platt/thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = next((p for p in Path(__file__).resolve().parents if (p / '.gitignore').is_file()), None)
if ROOT is None:
    raise RuntimeError('Turkish AI-Generated Text Detection Study project root not found')

CONFIG_REL = 'configs/external_cross_corpus.json'
FREEZE_REL = 'provenance/experiments/external_cross_corpus_v2/PREEXECUTION_FREEZE.json'
OUTPUT_REL = 'provenance/experiments/external_cross_corpus_v2/CALIBRATION_COMPATIBILITY.json'
RUNNER_REL = 'src/turkish_ai_detection/experiments/external_cross_corpus/rerun_fit.py'
METHODS = {'ce': 'CE_ONLY_MATCHED', 'supcon': 'SUPCON'}
SEEDS = (42, 314, 2026)
FOLDS = (0, 1, 2, 3, 4)
ALPHAS = ('0.01', '0.02', '0.03', '0.05')
RTOL = 1e-4
ATOL = 1e-5
ROWS = 600
IDENTITY_COLS = ('text_id', 'pair_id', 'label', 'domain', 'generator')


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha(path: Path) -> str:
    need(path.is_file(), f'missing required file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def binding(path: Path) -> dict[str, str]:
    return {'path': relative(path), 'sha256': sha(path)}


def read_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding='utf-8'))
    need(isinstance(obj, dict), f'JSON object required: {path}')
    return obj


def iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from iter_dicts(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_dicts(v)


def verify_v2_runtime_freeze(config_path: Path, freeze_path: Path,
                             runner_path: Path) -> tuple[dict, dict]:
    from turkish_ai_detection.experiments.external_cross_corpus.freeze_v2 import verify_freeze

    freeze = verify_freeze(
        config_path,
        freeze_path,
        required_code_path=runner_path,
    )
    return freeze, {
        'freeze': binding(freeze_path),
        'runner': binding(runner_path),
    }


def validate_contract(config_path: Path) -> tuple[dict, dict]:
    cfg = read_json(config_path)
    need(cfg.get('protocol_id') == 'EXTERNAL_CROSS_CORPUS_V2', 'protocol ID changed')
    cal = cfg.get('calibration_transfer')
    need(isinstance(cal, dict), 'calibration_transfer missing')
    need(cal.get('status') == 'SECONDARY_CONDITIONAL', 'calibration must be conditional')
    need(set(cal.get('applies_to', [])) == set(METHODS.values()), 'method set changed')
    need(cal.get('method') == 'Platt logistic', 'calibrator type changed')
    need(cal.get('fit_source') == 'outer-training inner-OOF raw scores only',
         'calibrator fit source changed')
    need(cal.get('automatic_or_optional_regeneration') == 'PROHIBITED',
         'calibration regeneration must remain prohibited')
    need(cal.get('incompatible_cell_value') == 'NA_INCOMPATIBLE_MAPPING',
         'incompatible value changed')
    need(cal.get('compatibility_decision', {}).get('external_data_use') == 'PROHIBITED',
         'external data must not affect mapping decision')
    need(tuple(float(v) for v in cal.get('nominal_fpr_alpha', [])) ==
         tuple(float(a) for a in ALPHAS), 'frozen alpha levels changed')
    need(cal.get('decision_rule') ==
         'AI iff calibrated_probability > realization-specific threshold',
         'decision rule changed')
    grid = cfg.get('model_grid', {})
    need(tuple(grid.get('seeds', [])) == SEEDS and
         tuple(grid.get('outer_folds', [])) == FOLDS,
         'fixed realization grid changed')
    for field in ('existing_parameters', 'existing_manifest', 'existing_integrity'):
        record = cal.get(field)
        need(isinstance(record, dict) and record.get('path') and record.get('sha256'),
             f'missing calibration identity binding: {field}')
    for _, item in cal.get('implementation_bindings', {}).items():
        if isinstance(item, dict) and 'path' in item:
            need('sha256' in item, 'implementation hash binding missing')
    need(cfg.get('evaluation', {}).get('labels_and_subgroups_join_only_after_complete_manifest_sealed') is True,
         'label-blind sequencing altered')
    return cfg, cal


def validate_only(config_path: Path) -> dict:
    cfg, cal = validate_contract(config_path)
    return {
        'status': 'PASS_CALIBRATION_COMPATIBILITY_STATIC_CONTRACT',
        'config': binding(config_path),
        'implementation': binding(Path(__file__).resolve()),
        'methods': list(METHODS.values()),
        'realization_grid': {'seeds': list(SEEDS), 'folds': list(FOLDS), 'cells': 30},
        'numeric_comparison': {'raw_score': {'rtol': RTOL, 'atol': ATOL},
                               'mapped_probability': {'rtol': RTOL, 'atol': ATOL},
                               'threshold_decisions': 'EXACT_ALL_FOUR_PER_ROW'},
        'numerical_gate_purpose': 'CALIBRATION_REUSE_ONLY_NOT_TRAINING_ACCEPTANCE',
        'mapping_source': cal['existing_parameters']['path'],
        'historical_calibration_refit': False,
        'external_data_read': False,
        'prediction_data_read': False,
        'final_reruns_performed': False,
        'needs_30_final_internal_predictions_before_full_execution': True,
    }


def verify_runtime(config_path: Path, cal: dict) -> tuple[dict, dict]:
    freeze_path = ROOT / FREEZE_REL
    need(freeze_path.is_file(), 'full compatibility gate blocked before PREEXECUTION_FREEZE')
    cfg = read_json(config_path)
    need(cfg.get('status') == 'FROZEN_READY_TO_RUN' and cfg.get('execution_permitted') is True,
         'active configuration is not frozen for scientific execution')
    for field in ('existing_parameters','existing_manifest','existing_integrity'):
        rec = cal[field]
        need(binding(ROOT / rec['path']) == {'path': rec['path'], 'sha256': rec['sha256']},
             f'historical calibration identity mismatch: {field}')
    for name, rec in cal['implementation_bindings'].items():
        if isinstance(rec, dict) and rec.get('path') and rec.get('sha256'):
            need(sha(ROOT / rec['path']) == rec['sha256'],
                 f'calibration implementation binding changed: {name}')
    runner = ROOT / RUNNER_REL
    freeze, sources = verify_v2_runtime_freeze(config_path, freeze_path, runner)
    return freeze, sources


def select_calibration_grid(path: Path) -> dict[tuple[str,int,int], dict]:
    tab = pd.read_csv(path)
    need(len(tab) == 50, 'historical calibration table must have 50 rows')
    need({'objective','method','seed','outer_fold','coefficient','intercept',
          'calibration','calibration_fit_rows','outer_test_rows_used_for_calibration',
          'outer_test_rows_used_for_threshold'} <= set(tab.columns),
         'calibration columns missing')
    got = {}
    for record in tab.to_dict(orient='records'):
        method = record['method']
        obj = record['objective']
        need(METHODS.get(obj) == method, 'historical calibration method/objective mismatch')
        key = (obj, int(record['seed']), int(record['outer_fold']))
        need(key not in got, f'duplicate calibration cell {key}')
        need(record['calibration'] == 'Platt logistic', f'calibration method changed: {key}')
        need(int(record['calibration_fit_rows']) == 2400, f'calibration fit rows changed: {key}')
        need(int(record['outer_test_rows_used_for_calibration']) == 0 and
             int(record['outer_test_rows_used_for_threshold']) == 0,
             f'outer-test calibration leakage recorded: {key}')
        for field in ('coefficient','intercept',*(f'threshold_alpha_{a}' for a in ALPHAS)):
            need(field in record and math.isfinite(float(record[field])),
                 f'non-finite calibration value {key}: {field}')
        for alpha in ALPHAS:
            value = float(record[f'threshold_alpha_{alpha}'])
            need(0 <= value <= 1, f'calibration threshold not probability: {key}')
        got[key] = record
    for obj in METHODS:
        need(sum(key[0] == obj for key in got) == 25,
             f'missing historical calibration grid for {obj}')
        for seed in SEEDS:
            for fold in FOLDS:
                need((obj,seed,fold) in got, f'missing required calibration cell {obj}/{seed}/{fold}')
    return got


def score_comparison(old: np.ndarray, new: np.ndarray, row: dict) -> dict:
    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    need(old.shape == new.shape == (ROWS,), 'internal score shape mismatch')
    need(np.isfinite(old).all() and np.isfinite(new).all(), 'non-finite internal raw score')
    coef = float(row['coefficient'])
    intercept = float(row['intercept'])
    p_old = expit(coef * old + intercept)
    p_new = expit(coef * new + intercept)
    raw_close = bool(np.allclose(old, new, rtol=RTOL, atol=ATOL))
    prob_close = bool(np.allclose(p_old, p_new, rtol=RTOL, atol=ATOL))
    changed = {}
    for alpha in ALPHAS:
        threshold = float(row[f'threshold_alpha_{alpha}'])
        changed[alpha] = int(np.count_nonzero((p_old > threshold) != (p_new > threshold)))
    decisions_match = all(n == 0 for n in changed.values())
    reusable = bool(raw_close and prob_close and decisions_match)
    return {
        'status': ('COMPATIBLE_CONDITIONAL_INTERNAL_GATE' if reusable else
                   'NA_INCOMPATIBLE_MAPPING'),
        'raw_allclose': raw_close,
        'mapped_probability_allclose': prob_close,
        'all_four_threshold_decisions_identical': decisions_match,
        'decision_change_counts_by_alpha': changed,
        'max_abs_raw_difference': float(np.max(np.abs(new - old))),
        'max_abs_calibrated_difference': float(np.max(np.abs(p_new - p_old))),
        'mapping_reuse_permitted': reusable,
    }


def strict_internal_labels(values: pd.Series, cell: str) -> np.ndarray:
    """Respect the retained HUMAN/AI contract; reject mixed/unknown labels."""
    labels = values.astype("string")
    need(labels.notna().all(), f"missing internal label {cell}")
    observed = set(labels.tolist())
    if observed == {"HUMAN", "AI"}:
        result = labels.map({"HUMAN": 0, "AI": 1}).to_numpy(dtype=np.int64)
    elif observed == {"0", "1"}:
        result = labels.astype(np.int64).to_numpy()
    else:
        raise RuntimeError(f"unsupported internal labels {cell}: {sorted(observed)}")
    need(np.count_nonzero(result == 0) == ROWS // 2 and
         np.count_nonzero(result == 1) == ROWS // 2,
         f"internal label balance changed {cell}")
    return result


def compare_cell(obj: str, seed: int, fold: int, cal_row: dict,
                 config: Path, runner: Path) -> dict:
    old_dir = ROOT / 'results' / 'primary' / obj / f'seed_{seed}'
    old_pred = old_dir / f'outer_{fold}_predictions.csv'
    old_sidecar = old_dir / f'outer_{fold}_run.json'
    new_dir = (ROOT / 'artifacts' / 'checkpoints' / 'external_cross_corpus' /
               'protocol_faithful_rerun' / f'{obj}_seed_{seed}_outer_{fold}')
    new_sidecar = new_dir / 'run.json'
    new_pred = new_dir / 'internal_predictions.csv'
    old_run = read_json(old_sidecar)
    new_run = read_json(new_sidecar)
    key = f'{obj}/seed_{seed}/outer_{fold}'
    need(old_run.get('objective') == obj and old_run.get('seed') == seed and
         old_run.get('outer_fold') == fold, f'historical identity mismatch {key}')
    need(sha(old_pred) == old_run['prediction_sha256'], f'historical internal prediction hash failed {key}')
    need(new_run.get('status') == 'SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN',
         f'not a scientific rerun {key}')
    for name, val in [('objective',obj),('seed',seed),('outer_fold',fold)]:
        need(new_run.get(name) == val, f'new sidecar {name} mismatch {key}')
    need(new_run.get('contract_sha256') == sha(config) and
         new_run.get('runner_sha256') == sha(runner),
         f'wrong code/config lineage {key}')
    need(new_run.get('structural_reference_sidecar_sha256') == sha(old_sidecar) and
         new_run.get('structural_reference_prediction_sha256') == sha(old_pred),
         f'historical reference hash mismatch {key}')
    for hashfield in ('train_pair_sha256','test_pair_sha256'):
        need(old_run.get(hashfield) == new_run.get(hashfield),
             f'pair identity mismatch {key}: {hashfield}')
    need(new_run.get('internal_prediction_path') == relative(new_pred) and
         new_run.get('internal_prediction_sha256') == sha(new_pred),
         f'new internal prediction hash/path mismatch {key}')
    need(new_run.get('checkpoint_roundtrip',{}).get('status') == 'PASS_CHECKPOINT_ROUNDTRIP',
         f'new checkpoint roundtrip not PASS {key}')
    payload = new_run.get('payload',{})
    for path_field, hash_field in [('model','model_sha256'),('config','config_sha256')]:
        p = ROOT / payload.get(path_field, 'MISSING')
        need(p.is_file() and payload.get(hash_field) == sha(p),
             f'new checkpoint payload changed {key}: {path_field}')

    old = pd.read_csv(old_pred)
    new = pd.read_csv(new_pred)
    need(len(old) == len(new) == ROWS, f'internal row count mismatch {key}')
    need(all(old[c].tolist() == new[c].tolist() for c in IDENTITY_COLS),
         f'internal row order/identity mismatch {key}')
    for df in (old,new):
        need(df.text_id.is_unique and df.pair_id.nunique() == ROWS // 2,
             f'paired test identity mismatch {key}')
        strict_internal_labels(df.label, key)
        need(df.seed.astype(int).eq(seed).all() and
             df.outer_fold.astype(int).eq(fold).all() and
             df.objective.eq(obj).all(), f'internal cell metadata mismatch {key}')
    details = score_comparison(old.raw_decision_score.to_numpy(float),
                               new.raw_decision_score.to_numpy(float), cal_row)
    return {
        'objective': obj, 'method': METHODS[obj], 'seed': seed, 'outer_fold': fold,
        'historical_prediction': binding(old_pred),
        'new_internal_prediction': binding(new_pred),
        'new_run_record': binding(new_sidecar),
        'calibration_row_identity': {'objective': obj, 'method': METHODS[obj],
                                     'seed': seed, 'outer_fold': fold},
        **details,
    }


def execute(config_path: Path) -> dict:
    cfg, cal = validate_contract(config_path)
    _, sources = verify_runtime(config_path, cal)
    destination = ROOT / OUTPUT_REL
    need(not destination.exists(), 'refusing to overwrite calibration compatibility evidence')
    grid = select_calibration_grid(ROOT / cal['existing_parameters']['path'])
    cells = []
    for obj in METHODS:
        for seed in SEEDS:
            for fold in FOLDS:
                cells.append(compare_cell(obj,seed,fold,grid[(obj,seed,fold)],
                                          config_path,ROOT / RUNNER_REL))
    compatible = [c for c in cells if c['mapping_reuse_permitted']]
    incompatible = [c for c in cells if not c['mapping_reuse_permitted']]
    result = {
        'schema_version': 'external-cross-corpus-v2-internal-calibration-compatibility-v1',
        'status': 'COMPLETE_INTERNAL_CALIBRATION_COMPATIBILITY_REVIEW',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': binding(config_path),
        'implementation': binding(Path(__file__).resolve()),
        'sources': sources,
        'historical_parameters': binding(ROOT / cal['existing_parameters']['path']),
        'decision_rule': {'raw_score_allclose': {'rtol':RTOL,'atol':ATOL},
                          'mapped_probability_allclose': {'rtol':RTOL,'atol':ATOL},
                          'four_frozen_threshold_decisions':'EXACT_MATCH_ALL_600_ROWS',
                          'aggregation':'AND_ALL_THREE'},
        'cells': cells,
        'n_expected': 30,
        'n_reusable_conditional': len(compatible),
        'n_incompatible': len(incompatible),
        'compatible_cells': [{'method':c['method'],'seed':c['seed'],'outer_fold':c['outer_fold']} for c in compatible],
        'incompatible_cells': [{'method':c['method'],'seed':c['seed'],'outer_fold':c['outer_fold']} for c in incompatible],
        'mapping_regeneration_performed': False,
        'external_data_or_predictions_accessed': False,
        'scope': 'Internal QA ONLY. Conditional mapping transfer, not an external FPR guarantee.',
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    return {'status': result['status'], 'manifest': binding(destination),
            'n_expected':30,'n_reusable_conditional':len(compatible),
            'n_incompatible':len(incompatible),'external_data_read':False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / CONFIG_REL)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    path = path.resolve()
    result = validate_only(path) if args.validate_only else execute(path)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
