# -*- coding: utf-8 -*-
"""Prepare frozen M1/M2 document-level raw features for Combined v2.

The ABSTRACT feature cache may be reused only after byte hashes, code hash,
and every condition-specific input text are verified. NEWSPAPER features are
computed afresh. Checkpoints are hash-bound, atomically replaced, and contain
only document-local features; fold-derived transforms are never cached here.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import os
import sys
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

from turkish_ai_detection.data.study import sha256_file, write_json
from turkish_ai_detection.data.stanza import verify_stanza_resources
from turkish_ai_detection.features.core import (
    BertPLLScorer, ZeyrekFeatureExtractor, artifact_neutral_context_text,
    normalize_text, sentence_features_v6, statistical_style_raw,
    surface_features,
)

DATASET_SHA256 = "ea5797ac19348957472626d1a23b740de531cd19577f613e1a73894e34d66b44"
CORE_SHA256 = "c634da0aecbca737f1b44e2ddb8b3cc494b57f675fda0a0e8ebc55d37a076870"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MLM_REVISION = "b6e1de16c983e0f2c70664591ea3f22810072608"
REUSE_ROOT_DEFAULT = ROOT / "artifacts" / "feature_cache"
REUSE_PREPROCESSED_SHA256 = "1adecdba1dc5bbda32834988fc6c4bc722d6cda1ac5118a77ba4df9b27a69421"
REUSE_HASHES = {
    "A": {
        "sentence": "eb3b0952424e0d3b3bc936ba1ace6cd3a235fb73ac7d8bca004c5dd4cdefc91a",
        "style": "c638557245241d3f19daf5955c3eb3fd7e6ce60428f09e12a7a2c25341f57763",
        "morphology": "7d25f6c0ada7e62e089548c811cca024a87b348233b4956e5e9e5e140e12f945",
        "pll": "41425554594723040186f8a10247e4e7f055e7b39324e10474a9f3e553c82cdf",
    },
    "B": {
        "sentence": "da3b3e55596442fae27c370502d53908c2b18045408773aede4433c3631a41e2",
        "style": "66a70fe85e567ce327f166ec5bb3e3d8bc39217517ad3966ee2e4908d47c48e1",
        "morphology": "eced4c545678780daea6ac3678c43dc754359b839f21000a442170aece1043ae",
        "pll": "92aa65ee9cb674fd4fc04d3285f2d0b1c3800edeffb2269ecba6ad4dc929b34e",
    },
}
SOURCE_NAMES = {
    "sentence": "abstract_sentence_raw_{condition}.csv",
    "style": "abstract_sss_surface_raw_{condition}.csv",
    "morphology": "abstract_mes_raw_{condition}.csv",
    "pll": "abstract_tbs_raw_{condition}.csv",
}
STAGE_COLUMNS = {
    "sentence": ["sentence_count", "sentence_word_count", "mean_sentence_length", "std_sentence_length", "min_sentence_length", "max_sentence_length", "sentence_lengths", "sentence_texts", "sentence_spans", "removed_false_boundary_count", "removed_false_boundaries"],
    "style": ["lexical_token_count", "unique_word_count", "ttr", "hapax_count", "hapax_ratio", "sentence_length_cv", "bigram_entropy", "stopword_count", "stopword_density", "conjunction_count", "conjunction_density", "surface_digit_per100", "surface_paren_per100", "surface_percent_per100", "surface_curly_apostrophe_per100", "surface_ascii_apostrophe_per100", "surface_colon_per100", "surface_slash_per100", "surface_semicolon_per100", "surface_hyphen_per100", "surface_comma_per100", "surface_curly_quote_per100", "surface_upper_abbrev_per100"],
    "morphology": ["morph_token_count", "morph_analyzed_tokens", "morph_unknown_tokens", "morph_analysis_coverage", "morph_ambiguous_tokens", "morph_ambiguity_rate", "mean_morph_candidates", "mes_analyzed_token_count", "morph_type_count", "morph_ttr", "suffix_tag_entropy", "suffix_surface_entropy", "suffix_tag_type_count", "suffix_surface_type_count", "fiilimsi_expected_mass", "fiilimsi_expected_density", "fiilimsi_diversity_ratio", "fiilimsi_pattern_entropy", "fiilimsi_effective_patterns", "vowel_harmony_eligible_count", "vowel_harmony_eligible_ratio", "vowel_back_ratio", "vowel_front_ratio", "vowel_mixed_ratio"],
    "pll": ["bert_wordpiece_count", "pll_scored_wordpiece_count", "pll_window_count", "pseudo_log_likelihood_mean", "pseudo_perplexity", "wordpiece_logprob_variance", "wordpiece_logprob_std", "log_pseudo_perplexity"],
}


def text_sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def condition_text(data: pd.DataFrame, condition: str) -> pd.Series:
    return data.text.map(normalize_text if condition == "A" else artifact_neutral_context_text)


def old_to_new_id(sample_id: object) -> str:
    value = str(sample_id); number = int(value.rsplit("_", 1)[1])
    return f"ABS_{number:04d}_{'H' if '_human_' in value else 'A'}"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def append_log(path: Path, event: dict) -> None:
    event = {"utc": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def stage_binding(stage: str, condition: str, data: pd.DataFrame, texts: pd.Series) -> dict:
    aggregate = hashlib.sha256()
    for text_id, value in zip(data.text_id, texts):
        aggregate.update(str(text_id).encode() + b"\0" + str(value).encode("utf-8") + b"\0")
    return {
        "schema_version": "combined-v2-feature-checkpoint-v2", "stage": stage, "condition": condition,
        "dataset_sha256": DATASET_SHA256, "turkish_ai_detection_core_sha256": CORE_SHA256,
        "feature_script_sha256": sha256_file(Path(__file__)), "ordered_condition_text_sha256": aggregate.hexdigest(),
        "model": MODEL_NAME if stage == "pll" else None, "model_revision": MLM_REVISION if stage == "pll" else None,
    }


def validate_reuse(data: pd.DataFrame, texts: pd.Series, condition: str, reuse_root: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    processed = reuse_root / "data" / "ABSTRACT" / "PROCESSED_V2"
    core_path = reuse_root / "src" / "turkish_ai_detection" / "features" / "core.py"; preprocessed = processed / "abstract_dataset_preprocessed_v2.csv"
    if sha256_file(core_path) != CORE_SHA256: raise RuntimeError("reuse rejected: feature core hash mismatch")
    if sha256_file(preprocessed) != REUSE_PREPROCESSED_SHA256: raise RuntimeError("reuse rejected: preprocessed data hash mismatch")
    source = pd.read_csv(preprocessed); source["text_id"] = source.sample_id.map(old_to_new_id)
    text_col = "text_A" if condition == "A" else "text_B_turkish_ai_detection"
    abstract = data.loc[data.domain.eq("ABSTRACT"), ["text_id"]].copy(); abstract["combined_text"] = texts.loc[abstract.index].to_numpy()
    matched = abstract.merge(source[["text_id", text_col]], on="text_id", how="left", validate="one_to_one")
    exact = matched.combined_text.astype(str).eq(matched[text_col].astype(str))
    if len(matched) != 2000 or not exact.all(): raise RuntimeError(f"reuse rejected: exact text matches={int(exact.sum())}/2000")
    text_map = matched.set_index("text_id").combined_text
    frames, source_hashes = {}, {}
    for stage, template in SOURCE_NAMES.items():
        path = processed / template.format(condition=condition); observed = sha256_file(path); expected = REUSE_HASHES[condition][stage]
        if observed != expected: raise RuntimeError(f"reuse rejected for {stage}: {observed} != {expected}")
        frame = pd.read_csv(path).rename(columns={"sample_id": "historical_sample_id"})
        frame["text_id"] = frame.historical_sample_id.map(old_to_new_id); frame = frame.drop(columns="historical_sample_id")
        if len(frame) != 2000 or not frame.text_id.is_unique: raise RuntimeError(f"reuse rejected: malformed {stage}")
        frame["condition_text_sha256"] = frame.text_id.map(text_map).map(text_sha256); frame["feature_origin"] = "VERIFIED_CORRECTED_ABSTRACT_REUSE"
        frames[stage] = frame; source_hashes[stage] = {"path": str(path), "sha256": observed}
    return frames, {"status": "PASS", "exact_text_matches": 2000, "core_sha256": CORE_SHA256, "preprocessed_sha256": REUSE_PREPROCESSED_SHA256, "sources": source_hashes}


def load_or_initialize(stage, condition, data, texts, reuse, work):
    partial = work / f"{stage}_{condition}.partial.csv"; binding_path = work / f"{stage}_{condition}.partial.binding.json"
    binding = stage_binding(stage, condition, data, texts)
    if partial.exists() or binding_path.exists():
        if not partial.exists() or not binding_path.exists(): raise RuntimeError(f"invalid {stage} checkpoint pair")
        if json.loads(binding_path.read_text(encoding="utf-8")) != binding: raise RuntimeError(f"{stage} checkpoint binding mismatch")
        frame = pd.read_csv(partial); required = {"text_id", "condition_text_sha256", "feature_origin", *STAGE_COLUMNS[stage]}
        if required - set(frame.columns) or not frame.text_id.is_unique: raise RuntimeError(f"invalid {stage} checkpoint schema")
        expected_hash = dict(zip(data.text_id.astype(str), texts.map(text_sha256)))
        if not frame.apply(lambda row: expected_hash.get(str(row.text_id)) == row.condition_text_sha256, axis=1).all(): raise RuntimeError(f"{stage} checkpoint text mismatch")
        rows = frame.to_dict("records")
    else:
        rows = reuse.to_dict("records"); write_json(binding_path, binding); atomic_csv(pd.DataFrame(rows), partial)
    return rows, {str(row["text_id"]) for row in rows}, partial, binding_path, binding


def checkpoint(rows, partial, binding_path, binding):
    frame = pd.DataFrame(rows).drop_duplicates("text_id", keep="last").sort_values("text_id", kind="stable")
    atomic_csv(frame, partial); write_json(binding_path, binding)


def finish_stage(stage, condition, rows, data, texts, partial, binding_path, work):
    frame = pd.DataFrame(rows).drop_duplicates("text_id", keep="last").sort_values("text_id", kind="stable")
    expected = set(data.text_id.astype(str)); observed = set(frame.text_id.astype(str))
    if len(frame) != 3000 or observed != expected: raise RuntimeError(f"{stage} incomplete: rows={len(frame)}, missing={len(expected-observed)}")
    expected_hash = dict(zip(data.text_id.astype(str), texts.map(text_sha256)))
    if not frame.apply(lambda row: expected_hash[str(row.text_id)] == row.condition_text_sha256, axis=1).all(): raise RuntimeError(f"{stage} final text mismatch")
    numeric = frame[STAGE_COLUMNS[stage]].select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy(float)).all(): raise RuntimeError(f"{stage} has non-finite numeric values")
    path = work / f"{stage}_{condition}.csv"; atomic_csv(frame, path); partial.unlink(missing_ok=True); binding_path.unlink(missing_ok=True)
    return path


def record_error(errors, stage, text_id, exc):
    errors.append({"stage": stage, "text_id": text_id, "exception_type": type(exc).__name__, "message": str(exc)})


def run_sentence(data, texts, condition, reuse, work, log_path, errors):
    rows, done, partial, binding_path, binding = load_or_initialize("sentence", condition, data, texts, reuse, work); pending = data.loc[~data.text_id.astype(str).isin(done)]
    if len(pending):
        import stanza
        locked = verify_stanza_resources(ROOT / "resources" / "stanza_tr_tokenizer_lock.json")
        nlp = stanza.Pipeline(locked["language"], processors=locked["processor"], package={locked["processor"]: locked["package"]}, use_gpu=True, verbose=False, download_method=None)
        for count, row in enumerate(pending.itertuples(), 1):
            try:
                value = texts.loc[row.Index]; result = sentence_features_v6(value, nlp, letter_words_only=condition == "B")
                result.update(text_id=row.text_id, condition_text_sha256=text_sha256(value), feature_origin="FRESH_COMPUTE"); rows.append(result)
            except Exception as exc: record_error(errors, "sentence", row.text_id, exc)
            if count % 25 == 0: checkpoint(rows, partial, binding_path, binding); append_log(log_path, {"stage": "sentence", "completed_total": len(rows), "fresh_completed": count, "errors": len(errors)})
        checkpoint(rows, partial, binding_path, binding)
    return finish_stage("sentence", condition, rows, data, texts, partial, binding_path, work)


def run_style(data, texts, condition, reuse, work, log_path, errors):
    sentence = pd.read_csv(work / f"sentence_{condition}.csv").set_index("text_id")
    rows, done, partial, binding_path, binding = load_or_initialize("style", condition, data, texts, reuse, work)
    stopwords = {x.strip() for x in (ROOT / "resources" / "turkish_stopwords_v1.txt").read_text(encoding="utf-8").splitlines() if x.strip()}; conjunctions = {x.strip() for x in (ROOT / "resources" / "conjunctions_primary_v1.txt").read_text(encoding="utf-8").splitlines() if x.strip()}
    pending = data.loc[~data.text_id.astype(str).isin(done)]
    for count, row in enumerate(pending.itertuples(), 1):
        try:
            value = texts.loc[row.Index]; sent = sentence.loc[row.text_id]; result = statistical_style_raw(value, sent.mean_sentence_length, sent.std_sentence_length, stopwords, conjunctions); result.update(surface_features(value)); result.update(text_id=row.text_id, condition_text_sha256=text_sha256(value), feature_origin="FRESH_COMPUTE"); rows.append(result)
        except Exception as exc: record_error(errors, "style", row.text_id, exc)
        if count % 100 == 0: checkpoint(rows, partial, binding_path, binding); append_log(log_path, {"stage": "style", "completed_total": len(rows), "fresh_completed": count, "errors": len(errors)})
    checkpoint(rows, partial, binding_path, binding)
    return finish_stage("style", condition, rows, data, texts, partial, binding_path, work)


def run_morphology(data, texts, condition, reuse, work, log_path, errors):
    rows, done, partial, binding_path, binding = load_or_initialize("morphology", condition, data, texts, reuse, work)
    logging.getLogger().setLevel(logging.ERROR); logging.getLogger("zeyrek").setLevel(logging.ERROR); extractor = ZeyrekFeatureExtractor(); pending = data.loc[~data.text_id.astype(str).isin(done)]
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        for count, row in enumerate(pending.itertuples(), 1):
            try:
                value = texts.loc[row.Index]
                with contextlib.redirect_stderr(devnull): result = extractor.extract(value)
                result.update(text_id=row.text_id, condition_text_sha256=text_sha256(value), feature_origin="FRESH_COMPUTE"); rows.append(result)
            except Exception as exc: record_error(errors, "morphology", row.text_id, exc)
            if count % 25 == 0: checkpoint(rows, partial, binding_path, binding); append_log(log_path, {"stage": "morphology", "completed_total": len(rows), "fresh_completed": count, "errors": len(errors)})
    checkpoint(rows, partial, binding_path, binding)
    return finish_stage("morphology", condition, rows, data, texts, partial, binding_path, work)


def run_pll(data, texts, condition, reuse, work, log_path, errors, batch_size):
    rows, done, partial, binding_path, binding = load_or_initialize("pll", condition, data, texts, reuse, work)
    scorer = BertPLLScorer(MODEL_NAME, revision=MLM_REVISION); pending = data.loc[~data.text_id.astype(str).isin(done)]
    for count, row in enumerate(pending.itertuples(), 1):
        try:
            value = texts.loc[row.Index]
            try: result = scorer.score(value, batch_size) if condition == "A" else scorer.score_letter_wordpieces(value, batch_size)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or not scorer.device.startswith("cuda"): raise
                scorer.torch.cuda.empty_cache(); gc.collect(); retry = max(4, batch_size // 2); result = scorer.score(value, retry) if condition == "A" else scorer.score_letter_wordpieces(value, retry)
            result["log_pseudo_perplexity"] = float(np.log(result["pseudo_perplexity"])); result.update(text_id=row.text_id, condition_text_sha256=text_sha256(value), feature_origin="FRESH_COMPUTE"); rows.append(result)
        except Exception as exc: record_error(errors, "pll", row.text_id, exc)
        if count % 10 == 0: checkpoint(rows, partial, binding_path, binding); append_log(log_path, {"stage": "pll", "completed_total": len(rows), "fresh_completed": count, "errors": len(errors), "device": scorer.device})
    checkpoint(rows, partial, binding_path, binding)
    return finish_stage("pll", condition, rows, data, texts, partial, binding_path, work)


def assemble(data, condition, work, reuse_audit, errors):
    paths = [work / f"{stage}_{condition}.csv" for stage in ("sentence", "style", "morphology", "pll")]; missing = [str(path) for path in paths if not path.exists()]
    if missing: raise FileNotFoundError(f"raw feature stages missing: {missing}")
    output = data[["text_id"]].copy(); origins = {}
    for path in paths:
        part = pd.read_csv(path); origins[path.stem.rsplit("_", 1)[0]] = {str(k): int(v) for k, v in part.feature_origin.value_counts().items()}; part = part.drop(columns=["condition_text_sha256", "feature_origin"]); columns = [c for c in part.columns if c == "text_id" or c not in output.columns]; output = output.merge(part[columns], on="text_id", how="left", validate="one_to_one")
    numeric = output.drop(columns="text_id").select_dtypes(include=[np.number]); expected = set(data.text_id.astype(str)); observed = set(output.text_id.astype(str)); missing_cells = int(output.drop(columns="text_id").isna().sum().sum()); nonfinite = int((~np.isfinite(numeric.to_numpy(float))).sum())
    audit = {"schema_version": "combined-v2-raw-feature-audit-v2", "condition": condition, "status": "PASS" if len(output) == 3000 and output.text_id.is_unique and observed == expected and missing_cells == 0 and nonfinite == 0 and not errors else "FAIL", "dataset_sha256": DATASET_SHA256, "row_count": int(len(output)), "unique_text_ids": int(output.text_id.nunique()), "missing_ids": sorted(expected-observed), "extra_ids": sorted(observed-expected), "missing_cells": missing_cells, "nonfinite_numeric_cells": nonfinite, "stage_origins": origins, "verified_reuse": reuse_audit, "errors": errors, "stage_hashes": {path.name: sha256_file(path) for path in paths}}
    audit_path = ROOT / "results" / "validation" / "feature_extraction" / f"raw_feature_audit_{condition}.json"; error_path = ROOT / "results" / "validation" / "feature_extraction" / f"raw_feature_errors_{condition}.json"; write_json(audit_path, audit); write_json(error_path, {"condition": condition, "error_count": len(errors), "errors": errors})
    if audit["status"] != "PASS": raise RuntimeError(f"raw feature audit failed; see {audit_path}")
    final_path = ROOT / "data" / "study" / f"raw_features_{condition}.csv"; atomic_csv(output, final_path)
    provenance = {"schema_version": "combined-v2-raw-features-v2", "created_utc": datetime.now(timezone.utc).isoformat(), "condition": condition, "dataset_sha256": DATASET_SHA256, "output_sha256": sha256_file(final_path), "audit_sha256": sha256_file(audit_path), "document_local_only": True, "fold_fitted_transformations_included": False, "model": MODEL_NAME, "revision": MLM_REVISION, "turkish_ai_detection_core_sha256": CORE_SHA256}
    write_json(ROOT / "provenance" / "experiments" / "feature_extraction" / f"raw_features_{condition}.json", provenance); return final_path, audit_path


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--condition", choices=["A", "B"], default="A"); parser.add_argument("--stage", choices=["sentence", "style", "morphology", "pll", "assemble", "all"], default="all"); parser.add_argument("--approve-compute", action="store_true"); parser.add_argument("--mask-batch-size", type=int, default=32); parser.add_argument("--reuse-root", type=Path, default=REUSE_ROOT_DEFAULT); args = parser.parse_args()
    if args.stage in {"morphology", "pll", "all"} and not args.approve_compute: raise RuntimeError("feature compute guard: pass --approve-compute for morphology/PLL stages")
    dataset_path = ROOT / "data" / "study" / "paired_study.csv"
    if sha256_file(dataset_path) != DATASET_SHA256 or sha256_file(ROOT / "src" / "turkish_ai_detection/features/core.py") != CORE_SHA256: raise RuntimeError("dataset or frozen feature implementation hash mismatch")
    data = pd.read_csv(dataset_path); texts = condition_text(data, args.condition); work = ROOT / "data" / "study" / "raw_features"; work.mkdir(parents=True, exist_ok=True); log_dir = ROOT / "results" / "validation" / "feature_extraction" / "logs"; log_dir.mkdir(parents=True, exist_ok=True); log_path = log_dir / f"condition_{args.condition}.jsonl"; errors = []
    reuse_frames, reuse_audit = validate_reuse(data, texts, args.condition, args.reuse_root.resolve()); write_json(ROOT / "provenance" / "experiments" / "feature_extraction" / f"raw_feature_reuse_{args.condition}.json", reuse_audit); append_log(log_path, {"event": "START", "condition": args.condition, "stage_request": args.stage, "verified_reuse_rows_per_stage": 2000})
    stages = ["sentence", "style", "morphology", "pll", "assemble"] if args.stage == "all" else [args.stage]
    if "sentence" in stages: run_sentence(data, texts, args.condition, reuse_frames["sentence"], work, log_path, errors)
    if "style" in stages: run_style(data, texts, args.condition, reuse_frames["style"], work, log_path, errors)
    if "morphology" in stages: run_morphology(data, texts, args.condition, reuse_frames["morphology"], work, log_path, errors)
    if "pll" in stages: run_pll(data, texts, args.condition, reuse_frames["pll"], work, log_path, errors, args.mask_batch_size)
    if errors: write_json(ROOT / "results" / "validation" / "feature_extraction" / f"raw_feature_errors_{args.condition}.json", {"condition": args.condition, "error_count": len(errors), "errors": errors}); raise RuntimeError(f"feature extraction recorded {len(errors)} errors; assembly forbidden")
    if "assemble" in stages:
        final_path, audit_path = assemble(data, args.condition, work, reuse_audit, errors); append_log(log_path, {"event": "COMPLETE", "condition": args.condition, "output": str(final_path), "output_sha256": sha256_file(final_path), "audit": str(audit_path)}); print(final_path); print(audit_path)
    return 0


if __name__ == "__main__": raise SystemExit(main())
