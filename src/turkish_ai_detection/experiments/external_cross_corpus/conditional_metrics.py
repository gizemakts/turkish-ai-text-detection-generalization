# -*- coding: utf-8 -*-
"""Prespecified conditional calibration/operating metrics AFTER complete seal.

No refitting, threshold selection or external adaptation. Cell eligibility is
fixed by the pre-external INTERNAL calibration compatibility manifest.

--validate-only reads only source/config metadata. Full execution revalidates
all sealed label-blind predictions and INTERNAL calibration evidence before
opening a single external label. LIT2026 has no compatible frozen mapping.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss


def find_root() -> Path:
    for path in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (path / '.gitignore').is_file():
            return path
    raise RuntimeError('project root not found')


ROOT = find_root()
sys.path.insert(0, str(ROOT / 'src'))
from turkish_ai_detection.experiments.external_cross_corpus import statistics as stats  # noqa: E402
from turkish_ai_detection.experiments.external_cross_corpus import calibration_compatibility as cal  # noqa: E402

CONFIG_REL = 'configs/external_cross_corpus.json'
REPORT_REL = 'provenance/experiments/external_cross_corpus_v2/CALIBRATION_COMPATIBILITY.json'
RESULT_REL = 'results/external_cross_corpus/conditional_metrics'
ALPHAS = ('0.01', '0.02', '0.03', '0.05')
METHODS = ('CE_ONLY_MATCHED', 'SUPCON')
NA = 'NA_INCOMPATIBLE_MAPPING'
NO_LIT_MAPPING = 'NOT_DEFINED_NO_LIT2026_FROZEN_MAPPING'


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def bind(path: Path) -> dict[str, str]:
    return {'path': stats.rel(path), 'sha256': stats.sha256_file(path)}


def validate_contract(config_path: Path) -> tuple[dict, dict]:
    cfg, sources = stats.verify_config(config_path)
    _, cal_block = cal.validate_contract(config_path)
    evaluation = cfg['evaluation']
    need(evaluation.get('conditional_secondary_operational_metrics') ==
         ['realized_fpr', 'tpr', 'fnr'], 'conditional operating metrics contract changed')
    need(evaluation.get('calibration_metrics_when_mapping_valid') ==
         ['brier', 'log_loss', 'ece_10_equal_width'], 'calibration metrics contract changed')
    need(evaluation.get('descriptive_fpr_transfer_quantity') ==
         'realized_fpr minus nominal_alpha', 'descriptive FPR shift formula changed')
    need(evaluation.get('lit2026_threshold_metrics') ==
         'NONE_NO_VALID_FROZEN_THRESHOLD', 'LIT threshold rule changed')
    need(cfg['calibration_transfer']['lit2026'].startswith('No model-specific frozen Platt/low-FPR mapping'),
         'LIT calibration prohibition changed')
    need(cal_block.get('incompatible_cell_value') == NA, 'incompatibility marker changed')
    need(tuple(f'{float(v):.2f}' for v in cal_block['nominal_fpr_alpha']) == ALPHAS,
         'nominal alpha levels changed')
    need(tuple(stats.DATASET_IDS) == ('renklier2026', 'tum2024_tr', 'bostancioglu2026_tr'),
         'primary corpora changed')
    need(cfg.get('model_grid', {}).get('realizations_per_method') == 15,
         'model grid changed')
    return cfg, sources


def validate_only(config_path: Path) -> dict[str, Any]:
    cfg, bindings = validate_contract(config_path)
    return {
        'status': 'PASS_POSTSEAL_CONDITIONAL_METRICS_STATIC_CONTRACT',
        'config': bind(config_path),
        'implementation': bind(Path(__file__).resolve()),
        'datasets': list(stats.DATASET_IDS),
        'calibrated_methods': list(METHODS),
        'grid_per_method': 15,
        'nominal_fpr_alphas': [float(a) for a in ALPHAS],
        'operating_metrics': cfg['evaluation']['conditional_secondary_operational_metrics'],
        'calibration_metrics': cfg['evaluation']['calibration_metrics_when_mapping_valid'],
        'ece_definition': '10 equal-width bins on [0,1]; left-closed/right-open, last bin includes 1; sample-weighted |mean(p)-frequency|',
        'eligibility_rule': '30-cell internal compatibility report; revalidated against original internal predictions before external label access',
        'incompatible_marker': NA,
        'lit2026_calibration': NO_LIT_MAPPING,
        'partial_grid_policy': 'No full 15-cell method aggregate when any of its 15 cells incompatible',
        'requires_complete_prediction_seal': True,
        'external_data_or_predictions_read': False,
        'freeze_required_for_full_execution_only': True,
        'expected_rows_per_corpus': {k: int(v['rows']) for k, v in bindings.items()},
    }


def report_grid_and_verify(report: dict, cfg: dict, config_path: Path,
                           freeze: dict) -> tuple[dict, dict]:
    """Validate all 30 internal gate decisions, *before* external label access."""
    need(report.get('status') == 'COMPLETE_INTERNAL_CALIBRATION_COMPATIBILITY_REVIEW',
         'internal calibration review missing or incomplete')
    need(report.get('config') == bind(config_path), 'internal calibration report config mismatch')
    module = Path(cal.__file__).resolve()
    need(report.get('implementation') == bind(module), 'internal calibration report implementation mismatch')
    need(stats.freeze_contains_binding(freeze, stats.rel(module), stats.sha256_file(module)),
         'freeze does not bind internal calibration code')
    need(report.get('mapping_regeneration_performed') is False, 'calibration was regenerated')
    need(report.get('external_data_or_predictions_accessed') is False,
         'internal gate accessed external data')
    frozen_cal = cfg['calibration_transfer']['existing_parameters']
    cal_path = ROOT / frozen_cal['path']
    need(bind(cal_path) == {'path': frozen_cal['path'], 'sha256': frozen_cal['sha256']},
         'historical Platt parameters changed')
    need(report.get('historical_parameters') == bind(cal_path),
         'internal report wrong Platt parameter binding')
    need(report.get('n_expected') == 30 and isinstance(report.get('cells'), list)
         and len(report['cells']) == 30, 'internal compatibility report must have 30 cells')
    need(report.get('decision_rule', {}).get('aggregation') == 'AND_ALL_THREE',
         'internal compatibility rule changed')
    need(report['decision_rule'].get('raw_score_allclose') == {'rtol': cal.RTOL, 'atol': cal.ATOL}
         and report['decision_rule'].get('mapped_probability_allclose') ==
         {'rtol': cal.RTOL, 'atol': cal.ATOL}, 'gate numerical tolerance changed')
    need(report['decision_rule'].get('four_frozen_threshold_decisions') ==
         'EXACT_MATCH_ALL_600_ROWS', 'gate threshold decision rule changed')

    grid = cal.select_calibration_grid(cal_path)
    cell_map = {}
    for cell in report['cells']:
        method = cell.get('method')
        obj = cell.get('objective')
        seed = cell.get('seed')
        fold = cell.get('outer_fold')
        need(method in METHODS and cal.METHODS.get(obj) == method and
             seed in cal.SEEDS and fold in cal.FOLDS,
             f'invalid calibration gate cell: {method}/{seed}/{fold}')
        key = (method, seed, fold)
        need(key not in cell_map, f'duplicate calibration compatibility cell {key}')
        need(cell.get('calibration_row_identity') ==
             {'objective': obj, 'method': method, 'seed': seed, 'outer_fold': fold},
             f'calibration row identity mismatch {key}')
        old_path = ROOT / f'results/primary/{obj}/seed_{seed}/outer_{fold}_predictions.csv'
        new_path = stats.internal_prediction_path(method, seed, fold)
        new_run = new_path.with_name('run.json')
        for name, path in [('historical_prediction', old_path),
                           ('new_internal_prediction', new_path),
                           ('new_run_record', new_run)]:
            need(path.is_file() and cell.get(name) == bind(path),
                 f'internal compatibility evidence changed: {name} {key}')
        run = stats.read_json(new_run)
        need(run.get('status') == 'SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN' and
             run.get('internal_prediction_sha256') == stats.sha256_file(new_path),
             f'non-final internal prediction detected: {key}')
        # No external data: recompute original gate with exact frozen Platt row.
        old = pd.read_csv(old_path)
        new = pd.read_csv(new_path)
        need(len(old) == len(new) == 600 and
             all(old[col].tolist() == new[col].tolist() for col in cal.IDENTITY_COLS),
             f'internal row identity changed: {key}')
        recalculated = cal.score_comparison(old.raw_decision_score.to_numpy(float),
                                            new.raw_decision_score.to_numpy(float),
                                            grid[(obj, seed, fold)])
        for name, value in recalculated.items():
            observed = cell.get(name)
            if isinstance(value, float):
                need(isinstance(observed, (float, int)) and
                     bool(np.isclose(observed, value, rtol=1e-12, atol=1e-12)),
                     f'modified internal gate numeric result {name} {key}')
            else:
                need(observed == value, f'modified internal gate result {name} {key}')
        cell_map[key] = bool(recalculated['mapping_reuse_permitted'])

    expected = {(method, seed, fold) for method in METHODS
                for seed in stats.SEEDS for fold in stats.OUTER_FOLDS}
    need(set(cell_map) == expected, 'incomplete compatibility grid')
    good = sum(cell_map.values())
    need(report.get('n_reusable_conditional') == good
         and report.get('n_incompatible') == 30 - good,
         'internal report counts inconsistent')
    need({(x['method'], x['seed'], x['outer_fold'])
          for x in report.get('compatible_cells', [])} ==
         {k for k, v in cell_map.items() if v}, 'compatible cells list inconsistent')
    need({(x['method'], x['seed'], x['outer_fold'])
          for x in report.get('incompatible_cells', [])} ==
         {k for k, v in cell_map.items() if not v}, 'incompatible cells list inconsistent')
    return cell_map, grid


def check_prelabel_guards(config_path: Path, cfg: dict) -> tuple[dict, dict, dict]:
    # Full check of all 135 sealed prediction CSV and sidecar hashes (no labels).
    guards = stats.verify_execution_guards(config_path)
    freeze = stats.read_json(ROOT / stats.FREEZE_REL)
    own = Path(__file__).resolve()
    need(stats.freeze_contains_binding(freeze, stats.rel(own), stats.sha256_file(own)),
         'conditional_metrics.py missing from final PREEXECUTION_FREEZE')
    report_path = ROOT / REPORT_REL
    need(report_path.is_file(), 'internal calibration compatibility gate has not run')
    report = stats.read_json(report_path)
    cell_map, grid = report_grid_and_verify(report, cfg, config_path, freeze)
    guards['compatibility_report'] = bind(report_path)
    return guards, cell_map, grid


def ece_10_equal_width(y: np.ndarray, p: np.ndarray) -> float:
    """Ten equal bins, with 1.0 assigned to final bin; no adaptive binning."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    need(len(y) == len(p) and len(y) > 0, 'ECE arrays invalid')
    need(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
         'ECE needs finite probabilities in [0,1]')
    bins = np.minimum((p * 10).astype(int), 9)
    counts = np.bincount(bins, minlength=10).astype(float)
    p_sums = np.bincount(bins, weights=p, minlength=10)
    y_sums = np.bincount(bins, weights=y, minlength=10)
    used = counts > 0
    return float(np.sum(np.abs(p_sums[used] - y_sums[used])) / len(y))


def calibration_scores(y: np.ndarray, raw: np.ndarray, coef: float,
                       intercept: float) -> tuple[np.ndarray, dict[str, float]]:
    need(set(np.unique(y)) == {0, 1}, 'calibration requires both classes')
    need(len(y) == len(raw) and np.isfinite(raw).all(), 'bad raw decision scores')
    p = expit(float(coef) * raw + float(intercept))
    need(np.isfinite(p).all(), 'non-finite calibrated probability')
    return p, {
        'brier': float(np.mean((p - y) ** 2)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'ece_10_equal_width': ece_10_equal_width(y, p),
    }


def operating_point(y: np.ndarray, p: np.ndarray, threshold: float,
                    nominal_alpha: float) -> dict[str, Any]:
    need(0 <= threshold <= 1 and np.isfinite(threshold), 'bad frozen threshold')
    pred = p > threshold  # Historical strict decision rule.
    pos = y == 1
    neg = y == 0
    tp = int(np.sum(pred & pos))
    fn = int(np.sum(~pred & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum(~pred & neg))
    need(fn + tp > 0 and fp + tn > 0, 'need positive and negative examples')
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'realized_fpr': fpr, 'tpr': tpr, 'fnr': fn / (tp + fn),
            'realized_fpr_minus_nominal_alpha': fpr - nominal_alpha}


def aggregate_full_grid(frame: pd.DataFrame, metric_cols: list[str],
                        status_col: str = 'status') -> pd.DataFrame:
    out = []
    grouping = ['dataset_id', 'method']
    if 'alpha' in frame.columns:
        grouping.append('alpha')
    for key, part in frame.groupby(grouping, sort=False, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(grouping, values))
        n = int((part[status_col] == 'COMPATIBLE_CONDITIONAL_INTERNAL_GATE').sum())
        rec.update({'n_expected_cells': 15, 'n_compatible_cells': n,
                    'n_incompatible_cells': 15 - n,
                    'status': 'COMPLETE_CONDITIONAL_15_CELL_GRID' if n == 15 else
                    NA})
        for col in metric_cols:
            rec[f'mean_{col}'] = (float(pd.to_numeric(part[col]).mean()) if n == 15 else NA)
        out.append(rec)
    return pd.DataFrame(out)


def execute(config_path: Path) -> dict:
    cfg, sources = validate_contract(config_path)
    guards, cell_map, grid = check_prelabel_guards(config_path, cfg)
    output_root = ROOT / RESULT_REL
    destinations = {
        'calibration_cells': output_root / 'calibration_cells.csv',
        'operating_cells': output_root / 'operating_cells.csv',
        'calibration_summary': output_root / 'calibration_summary.csv',
        'operating_summary': output_root / 'operating_summary.csv',
        'status': output_root / 'mapping_status.csv',
    }
    manifest_path = output_root / 'CONDITIONAL_METRICS_MANIFEST.json'
    for path in (*destinations.values(), manifest_path):
        need(not path.exists(), f'no overwrite of conditional metrics: {path}')

    cal_rows, op_rows, status_rows = [], [], []
    for dataset_id in stats.DATASET_IDS:
        source = sources[dataset_id]
        ext = stats.load_labeled_external(
            source, cfg['external_data_bindings']['label_contract'],
            cfg['statistics'][f'{dataset_id}_unit'])
        y = ext['binary_label'].to_numpy(int)
        need(len(y) == int(source['rows']) and
             int(np.sum(y == 0)) == int(source['human']) and
             int(np.sum(y == 1)) == int(source['ai']),
             f'external population identity changed: {dataset_id}')
        for method in stats.METHODS:
            for seed in stats.SEEDS:
                for fold in stats.OUTER_FOLDS:
                    base = {'dataset_id': dataset_id, 'population': 'PRIMARY_RELEASED',
                            'method': method, 'seed': seed, 'outer_fold': fold}
                    if method == 'LIT2026_REIMPL_V1_1':
                        status_rows.append({**base, 'status': NO_LIT_MAPPING})
                        continue
                    status = ('COMPATIBLE_CONDITIONAL_INTERNAL_GATE'
                              if cell_map[(method, seed, fold)] else NA)
                    status_rows.append({**base, 'status': status})
                    if status == NA:
                        cal_rows.append({**base, 'status': status,
                                         'brier': NA, 'log_loss': NA, 'ece_10_equal_width': NA})
                        for alpha in ALPHAS:
                            op_rows.append({**base, 'alpha': float(alpha), 'status': status,
                                            'threshold': NA, 'realized_fpr': NA,
                                            'tpr': NA, 'fnr': NA,
                                            'realized_fpr_minus_nominal_alpha': NA,
                                            'tn': NA, 'fp': NA, 'fn': NA, 'tp': NA})
                        continue
                    # Seal was fully revalidated BEFORE the first external label.
                    raw = stats.load_prediction(dataset_id, method, seed, fold, ext)
                    objective = 'ce' if method == 'CE_ONLY_MATCHED' else 'supcon'
                    row = grid[(objective, seed, fold)]
                    p, metrics = calibration_scores(y, raw, row['coefficient'],
                                                    row['intercept'])
                    cal_rows.append({**base, 'status': status, **metrics})
                    for alpha in ALPHAS:
                        thresh = float(row[f'threshold_alpha_{alpha}'])
                        op_rows.append({**base, 'status': status, 'alpha': float(alpha),
                                        'threshold': thresh,
                                        **operating_point(y, p, thresh, float(alpha))})

    tables = {'calibration_cells': pd.DataFrame(cal_rows),
              'operating_cells': pd.DataFrame(op_rows),
              'status': pd.DataFrame(status_rows)}
    need(len(tables['calibration_cells']) == 90 and
         len(tables['operating_cells']) == 360 and
         len(tables['status']) == 135, 'conditional analysis grid incomplete')
    tables['calibration_summary'] = aggregate_full_grid(
        tables['calibration_cells'], ['brier', 'log_loss', 'ece_10_equal_width'])
    tables['operating_summary'] = aggregate_full_grid(
        tables['operating_cells'], ['realized_fpr', 'tpr', 'fnr',
                                    'realized_fpr_minus_nominal_alpha'])
    for name, path in destinations.items():
        stats.atomic_csv(tables[name], path)
    manifest = {
        'schema_version': 'external-cross-corpus-v2-conditional-postseal-metrics-v1',
        'status': 'COMPLETE_POSTSEAL_CONDITIONAL_METRICS',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': bind(config_path),
        'implementation': bind(Path(__file__).resolve()),
        'guards': guards,
        'historical_calibration': bind(ROOT / cfg['calibration_transfer']['existing_parameters']['path']),
        'conditional_only': True,
        'mapping_regenerated': False,
        'thresholds_regenerated': False,
        'external_adaptation_performed': False,
        'lit2026_calibration': NO_LIT_MAPPING,
        'compatibility_failures_not_silently_dropped': True,
        'method_level_estimates_only_when_all_15_cells_compatible': True,
        'primary_population_unchanged': True,
        'no_cross_corpus_pooling': True,
        'calibration_ece': '10_equal_width_left_closed_last_closed',
        'decision_rule': 'AI iff calibrated_probability > cell_frozen_threshold',
        'output_tables': {k: bind(v) for k, v in destinations.items()},
    }
    stats.atomic_json(manifest_path, manifest)
    return {'status': manifest['status'], 'manifest': bind(manifest_path),
            'table_rows': {k: len(v) for k, v in tables.items()},
            'external_primary_corpora_pooled': False,
            'mapping_regenerated': False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / CONFIG_REL)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    result = validate_only(config_path) if args.validate_only else execute(config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
