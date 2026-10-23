# -*- coding: utf-8 -*-
"""Prespecified post-seal descriptive subgroup AUROC and corpus text profiles.

This script MUST NOT be executed without a binding PREEXECUTION_FREEZE and a
fully verified label-blind prediction seal. --validate-only reads config/code
metadata only; it does not open external data or prediction payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / '.gitignore').is_file():
            return parent
    raise RuntimeError('project root not found via .gitignore')


ROOT = find_root()
sys.path.insert(0, str(ROOT / 'src'))
from turkish_ai_detection.experiments.external_cross_corpus import statistics as stats  # noqa: E402

CONFIG_REL = 'configs/external_cross_corpus.json'
RESULT_REL = 'results/external_cross_corpus/secondary_audits'
MODEL_ID = 'dbmdz/bert-base-turkish-cased'
MODEL_REVISION = 'b6e1de16c983e0f2c70664591ea3f22810072608'
EXPECTED_SUBGROUPS = {
    'renklier2026': ('domain', 'scenario', 'generator'),
    'tum2024_tr': ('generator', 'original_split'),
    'bostancioglu2026_tr': (),
}
FEATURES = (
    'whitespace_word_count', 'unicode_character_count',
    'berturk_untruncated_token_count', 'proportion_over_512_berturk_tokens',
    'lowercase_letter_fraction', 'uppercase_letter_fraction',
    'unicode_punctuation_density', 'unicode_digit_density',
    'newline_density', 'markdown_heading_indicator', 'list_line_indicator',
)
EXPECTED_FEATURE_DEFINITIONS = {'whitespace_word_count': 'len(text.split()) using Python str.split with no explicit separator',
 'unicode_character_count': 'len(text), counting Python Unicode code points',
 'berturk_untruncated_token_count': 'len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) '
                                    'using the pinned BERTurk revision',
 'proportion_over_512_berturk_tokens': 'Arithmetic mean of int(berturk_untruncated_token_count > 512)',
 'lowercase_letter_fraction': 'sum(ch.islower() for ch in text) / sum(ch.isalpha() for ch in text); 0.0 if '
                              'denominator is zero',
 'uppercase_letter_fraction': 'sum(ch.isupper() for ch in text) / sum(ch.isalpha() for ch in text); 0.0 if '
                              'denominator is zero',
 'unicode_punctuation_density': "sum(unicodedata.category(ch).startswith('P') for ch in text) / "
                                'max(len(text), 1)',
 'unicode_digit_density': 'sum(ch.isdecimal() for ch in text) / max(len(text), 1)',
 'newline_density': "text.count('\\n') / max(len(text), 1), with no newline normalization",
 'markdown_heading_indicator': '1 iff any text.splitlines() line matches Unicode-aware Python regex '
                               '^\\s{0,3}#{1,6}\\s+; else 0',
 'list_line_indicator': '1 iff any text.splitlines() line matches Unicode-aware Python regex '
                        '^\\s*(?:[-*+]|[0-9]+[.)])\\s+; else 0'}
HEADING = re.compile(r'^\s{0,3}#{1,6}\s+', re.UNICODE)
LIST_LINE = re.compile(r'^\s*(?:[-*+]|[0-9]+[.)])\s+', re.UNICODE)
MISSING = '<MISSING_METADATA>'


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def binding(path: Path) -> dict[str, str]:
    return {'path': stats.rel(path), 'sha256': stats.sha256_file(path)}


def validate_contract(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    cfg, sources = stats.verify_config(config_path)
    sub = cfg.get('subgroups')
    shift = cfg.get('dataset_shift_audit')
    require(isinstance(sub, dict) and isinstance(shift, dict), 'missing descriptive contracts')
    require(sub.get('status') == 'SECONDARY_DESCRIPTIVE', 'subgroup status changed')
    require(sub.get('minimum_for_auroc') == 'at least 20 human and 20 AI rows',
            'subgroup size threshold changed')
    for dataset_id, fields in EXPECTED_SUBGROUPS.items():
        require(tuple(sub.get(dataset_id, [])) == fields, f'subgroups changed: {dataset_id}')
    require(sub.get('generator_rule') == (
        "For generator-specific discrimination, compare that generator's AI rows "
        "with the corpus human reference rows; label this one-generator-versus-common-human "
        "and do not call a family seen unless exact model identity is documented."
    ), 'generator common-HUMAN rule changed')
    require(shift.get('status') == 'DESCRIPTIVE', 'shift status changed')
    require(shift.get('stratify_by') == ['dataset', 'label'], 'shift grouping changed')
    require(shift.get('text_normalization') == 'NONE; operate on the prepared text string',
            'shift text contract changed')
    require(shift.get('features') == EXPECTED_FEATURE_DEFINITIONS,
            'shift formulas changed from the reviewed exact definitions')
    require(shift.get('performance_guided_population_changes') is False,
            'performance-guided population changes prohibited')
    require(shift.get('causal_interpretation') is False, 'causal claims prohibited')
    internal = cfg.get('internal_training_bindings', {})
    require(internal.get('backbone') == MODEL_ID, 'model name mismatch')
    require(internal.get('backbone_revision') == MODEL_REVISION, 'tokenizer revision mismatch')
    return cfg, sources


def validate_only(config_path: Path) -> dict[str, Any]:
    cfg, sources = validate_contract(config_path)
    return {
        'status': 'PASS_EXTERNAL_SECONDARY_AUDITS_STATIC_CONTRACT',
        'config': binding(config_path),
        'implementation': binding(Path(__file__).resolve()),
        'dataset_ids': list(stats.DATASET_IDS),
        'subgroups': {k: list(v) for k, v in EXPECTED_SUBGROUPS.items()},
        'subgroup_min_human_and_ai': 20,
        'generator_contrast': 'ONE_GENERATOR_VERSUS_COMMON_HUMAN',
        'shift_features': list(FEATURES),
        'tokenizer_revision': MODEL_REVISION,
        'source_rows': {k: int(v['rows']) for k, v in sources.items()},
        'label_or_prediction_files_read': False,
        'requires_verified_complete_seal_before_execution': True,
        'primary_populations_modified': False,
    }


def verify_runtime(config_path: Path) -> dict[str, Any]:
    # Uses exactly the AUROC statistics module's full 135-cell seal checks,
    # sidecar hashes, config/freeze bindings and historical overlap integrity.
    guards = stats.verify_execution_guards(config_path)
    freeze = stats.read_json(ROOT / stats.FREEZE_REL)
    own = Path(__file__).resolve()
    require(stats.freeze_contains_binding(freeze, stats.rel(own), stats.sha256_file(own)),
            'secondary_audits.py is not bound to final PREEXECUTION_FREEZE')
    return guards


def load_corpus(source: dict[str, Any], label_contract: dict[str, Any],
                fields: tuple[str, ...]) -> pd.DataFrame:
    path = ROOT / source['path']
    require(path.is_file() and stats.sha256_file(path) == source['sha256'],
            f'prepared dataset hash mismatch: {path}')
    # Preserve literal prepared text and literal missing/NA-like strings.
    frame = pd.read_csv(path, keep_default_na=False, na_filter=False,
                        dtype={'external_id': str, 'text_sha256': str, 'text': str})
    required = {'external_id', 'text_sha256', 'text', *fields}
    require(required <= set(frame.columns),
            f'missing required columns: {sorted(required - set(frame.columns))} in {path}')
    require(len(frame) == int(source['rows']), f'row count changed: {path}')
    frame['external_id'] = frame['external_id'].astype(str)
    frame['text_sha256'] = frame['text_sha256'].astype(str).str.lower()
    require(frame['external_id'].is_unique, f'duplicate external IDs: {path}')
    require(not frame['external_id'].eq('').any(), 'empty external_id')
    require(frame['text'].map(lambda v: isinstance(v, str)).all(), 'non-text source payload')
    frame['binary_label'] = stats.resolve_binary_label(frame, label_contract).to_numpy()
    require(set(frame['binary_label']) == {0, 1}, f'missing label class: {path}')
    require(int((frame['binary_label'] == 0).sum()) == int(source['human']),
            f'HUMAN count mismatch: {path}')
    require(int((frame['binary_label'] == 1).sum()) == int(source['ai']),
            f'AI count mismatch: {path}')
    return frame


def subgroup_masks(frame: pd.DataFrame, field: str):
    y = frame['binary_label'].to_numpy(int)
    vals = frame[field].astype(str).str.strip().to_numpy()
    if field == 'generator':
        values = sorted(set(vals[y == 1]) | ({''} if '' in vals[y == 1] else set()))
    else:
        values = sorted(set(vals))
    for value in values:
        missing = not bool(value)
        # Generator-specific AUROC NEVER compares AI subtypes against a
        # generator-specific HUMAN subset. All corpus HUMANs are the reference.
        if field == 'generator':
            mask = (y == 0) | ((y == 1) & (vals == value))
            contrast = 'ONE_GENERATOR_VERSUS_COMMON_HUMAN'
        else:
            mask = vals == value
            contrast = 'WITHIN_GROUP_HUMAN_VERSUS_AI'
        human = int(((y == 0) & mask).sum())
        ai = int(((y == 1) & mask).sum())
        status = ('MISSING_METADATA_NOT_ASSESSED' if missing else
                  'ELIGIBLE' if min(human, ai) >= 20 else
                  'INSUFFICIENT_CLASS_COUNTS')
        yield (value if not missing else MISSING, mask, contrast,
               human, ai, status)


def analyze_subgroups(dataset_id: str, frame: pd.DataFrame,
                      score_map: dict[tuple[str, int, int], np.ndarray]):
    statuses, cell_rows, summaries = [], [], []
    y = frame['binary_label'].to_numpy(dtype=int)
    fields = EXPECTED_SUBGROUPS[dataset_id]
    if not fields:
        statuses.append({'dataset_id': dataset_id, 'field': 'NONE_PRESPECIFIED',
                         'value': '', 'contrast': '', 'human_rows': int((y == 0).sum()),
                         'ai_rows': int((y == 1).sum()), 'status': 'NO_SUBGROUPS_PRESPECIFIED'})
    for field in fields:
        for value, mask, contrast, human, ai, status in subgroup_masks(frame, field):
            prefix = {'dataset_id': dataset_id, 'field': field, 'value': value,
                      'contrast': contrast, 'human_rows': human, 'ai_rows': ai}
            statuses.append({**prefix, 'status': status})
            if status != 'ELIGIBLE':
                continue
            for method in stats.METHODS:
                seed_means = []
                for seed in stats.SEEDS:
                    fold_values = []
                    for outer in stats.OUTER_FOLDS:
                        auc = float(roc_auc_score(y[mask], score_map[(method, seed, outer)][mask]))
                        cell_rows.append({**prefix, 'method': method,
                                          'seed': seed, 'outer_fold': outer, 'auroc': auc})
                        fold_values.append(auc)
                    seed_means.append(float(np.mean(fold_values)))
                summaries.append({**prefix, 'method': method,
                                  'mean_seed_mean_auroc': float(np.mean(seed_means)),
                                  'sample_sd_seed_means': float(np.std(seed_means, ddof=1)),
                                  'seed_42_mean': seed_means[0],
                                  'seed_314_mean': seed_means[1],
                                  'seed_2026_mean': seed_means[2],
                                  'inference': 'DESCRIPTIVE_NO_BOOTSTRAP_CI'})
    return statuses, cell_rows, summaries


def row_features(text: str, tokenizer) -> dict[str, float | int]:
    # Exact formulas from the locked config. No Unicode/whitespace/newline
    # normalization and NO tokenizer truncation.
    size = len(text)
    alpha = sum(ch.isalpha() for ch in text)
    tokens = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    return {
        'whitespace_word_count': len(text.split()),
        'unicode_character_count': size,
        'berturk_untruncated_token_count': tokens,
        'proportion_over_512_berturk_tokens': int(tokens > 512),
        'lowercase_letter_fraction': sum(ch.islower() for ch in text) / alpha if alpha else 0.0,
        'uppercase_letter_fraction': sum(ch.isupper() for ch in text) / alpha if alpha else 0.0,
        'unicode_punctuation_density': sum(unicodedata.category(ch).startswith('P') for ch in text) / max(size, 1),
        'unicode_digit_density': sum(ch.isdecimal() for ch in text) / max(size, 1),
        'newline_density': text.count('\n') / max(size, 1),
        'markdown_heading_indicator': int(any(HEADING.match(line) for line in text.splitlines())),
        'list_line_indicator': int(any(LIST_LINE.match(line) for line in text.splitlines())),
    }


def summarize_text_profile(dataset_id: str, frame: pd.DataFrame, tokenizer):
    # Only aggregate feature summaries are retained, never text or row-level
    # group payloads. Primary/sensitivity membership is never changed.
    feature_frame = pd.DataFrame([row_features(text, tokenizer) for text in frame['text']])
    y = frame['binary_label'].to_numpy(int)
    output = []
    for cls, label in ((0, 'HUMAN'), (1, 'AI')):
        mask = y == cls
        for feat in FEATURES:
            arr = feature_frame.loc[mask, feat].to_numpy(dtype=float)
            require(len(arr) > 0 and np.isfinite(arr).all(), f'invalid shift feature: {feat}')
            output.append({'dataset_id': dataset_id, 'label': label,
                           'label_binary': cls, 'rows': int(mask.sum()),
                           'feature': feat, 'mean': float(np.mean(arr)),
                           'median': float(np.median(arr)),
                           'sample_sd': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                           'minimum': float(np.min(arr)),
                           'maximum': float(np.max(arr))})
    return output


def execute(config_path: Path) -> dict[str, Any]:
    cfg, sources = validate_contract(config_path)
    guards = verify_runtime(config_path)  # MANDATORY before reading ANY external source
    root = ROOT / RESULT_REL
    outputs = {
        'subgroup_status': root / 'subgroup_status.csv',
        'subgroup_cells': root / 'subgroup_cell_auroc.csv',
        'subgroup_summary': root / 'subgroup_summary.csv',
        'text_profiles': root / 'dataset_label_text_profiles.csv',
    }
    manifest_path = root / 'SECONDARY_AUDITS_MANIFEST.json'
    require(all(not p.exists() for p in (*outputs.values(), manifest_path)),
            'refusing to overwrite secondary audit results')

    # Load tokenizer only when explicitly executing post-seal analysis.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                use_fast=True)
    label_contract = cfg['external_data_bindings']['label_contract']
    status_rows, cell_rows, summary_rows, profile_rows = [], [], [], []
    source_bindings = []
    for dataset_id in stats.DATASET_IDS:
        source = sources[dataset_id]
        frame = load_corpus(source, label_contract, EXPECTED_SUBGROUPS[dataset_id])
        source_bindings.append({'dataset_id': dataset_id, **binding(ROOT / source['path'])})
        # Source and every prediction must have identical order, IDs and hashes.
        identity = frame[['external_id', 'text_sha256']]
        score_map = {
            (method, seed, outer): stats.load_prediction(dataset_id, method, seed, outer, identity)
            for method in stats.METHODS
            for seed in stats.SEEDS
            for outer in stats.OUTER_FOLDS
        }
        a, b, c = analyze_subgroups(dataset_id, frame, score_map)
        status_rows.extend(a)
        cell_rows.extend(b)
        summary_rows.extend(c)
        profile_rows.extend(summarize_text_profile(dataset_id, frame, tokenizer))

    dataframes = {
        'subgroup_status': pd.DataFrame(status_rows),
        'subgroup_cells': pd.DataFrame(cell_rows, columns=[
            'dataset_id', 'field', 'value', 'contrast', 'human_rows', 'ai_rows',
            'method', 'seed', 'outer_fold', 'auroc']),
        'subgroup_summary': pd.DataFrame(summary_rows, columns=[
            'dataset_id', 'field', 'value', 'contrast', 'human_rows', 'ai_rows',
            'method', 'mean_seed_mean_auroc', 'sample_sd_seed_means',
            'seed_42_mean', 'seed_314_mean', 'seed_2026_mean', 'inference']),
        'text_profiles': pd.DataFrame(profile_rows),
    }
    for name, path in outputs.items():
        stats.atomic_csv(dataframes[name], path)
    manifest = {
        'schema_version': 'external-secondary-descriptive-audits-v1',
        'status': 'COMPLETE_POSTSEAL_SECONDARY_DESCRIPTIVE_AUDITS',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': binding(config_path),
        'implementation': binding(Path(__file__).resolve()),
        'guards': guards,
        'sources': source_bindings,
        'model': MODEL_ID, 'tokenizer_revision': MODEL_REVISION,
        'subgroup_rule': 'at least 20 HUMAN and 20 AI; generator AI versus common corpus HUMAN',
        'subgroups_secondary_descriptive': True,
        'no_causal_shift_claims': True,
        'text_normalization': 'NONE',
        'population_membership_modified': False,
        'text_profiles_external_corpora_only': True,
        'outputs': {name: binding(path) for name, path in outputs.items()},
    }
    stats.atomic_json(manifest_path, manifest)
    return {'status': manifest['status'], 'manifest': binding(manifest_path),
            'rows': {name: len(df) for name, df in dataframes.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / CONFIG_REL)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    path = path.resolve()
    result = validate_only(path) if args.validate_only else execute(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
