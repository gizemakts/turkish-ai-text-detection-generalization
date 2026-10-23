# -*- coding: utf-8 -*-
"""Exploratory cturan/Uunan feature and frozen-score error analysis.

No record-level scores exist in the source result packages. GPU inference is
therefore guarded by an explicit --execute-inference flag. No training,
calibration, threshold fitting, sampling, or checkpoint selection is present.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "cross_corpus_error_analysis_20261013"
PROTOCOL = OUT / "ANALYSIS_PROTOCOL.md"
EXPECTED_PROTOCOL_SHA256 = "0410218b2dfb3286ee974e73f9445812abce0f955ff9fa778be1cd13784e57a2"
CT_OUT = ROOT / "results" / "cturan_external_exploratory_20261013"
UU_OUT = ROOT / "results" / "uunan_external_exploratory_20261013"
CT_SOURCE = ROOT.parent / "Turkish AI Detection_Yeni_Veri_Setleri" / "Yeni_Adaylar_Tam_20261013" / "cturan_turkish_synthetic_corpus" / "tr_corpus.jsonl"
UU_DIR = ROOT.parent / "Turkish AI Detection_Yeni_Veri_Setleri" / "Yeni_Adaylar_Tam_20261013" / "Uunan_turkish_synthetic_corpus"
UU_PARQUETS = (
    UU_DIR / "data" / "train-00000-of-00002.parquet",
    UU_DIR / "data" / "train-00001-of-00002.parquet",
)
CT_HELPER = ROOT / "scripts" / "cturan_external_exploratory_20261013.py"

SEED = 20261013
BOOTSTRAPS = 500
BANDS = (
    "01_very_short_le1000",
    "02_short_1001_2500",
    "03_medium_2501_4000",
    "04_long_4001_6000",
    "05_very_long_gt6000",
)
CELLS = ((42, 0), (314, 1), (2026, 2))
METHODS = (("CE_ONLY_MATCHED", "ce"), ("SUPCON", "supcon"))
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
NUMBERED_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*•]\s+|(?:\d+(?:\.\d+)*)[.)]?\s+|[IVXLCDM]+[.)]\s+)", re.IGNORECASE)

FEATURES = (
    "character_count", "whitespace_word_count", "lexical_token_count",
    "sentence_count", "sentence_words_mean", "sentence_words_sd",
    "sentence_words_cv", "sentence_chars_mean", "sentence_chars_sd",
    "paragraph_count", "paragraph_words_mean", "paragraph_words_sd",
    "paragraph_words_cv", "short_line_rate", "heading_line_count",
    "heading_line_rate", "numbered_section_count", "title_like_first_line",
    "colon_line_rate", "digit_character_ratio", "punctuation_character_ratio",
    "newline_per_1000_chars", "type_token_ratio", "root_ttr", "hapax_ratio",
    "mean_token_characters", "repeated_sentence_fraction",
    "repeated_paragraph_fraction", "repeated_word_3gram_fraction",
    "repeated_word_5gram_fraction", "max_sentence_multiplicity",
    "opening_template_frequency", "closing_template_frequency",
    "first_segment_ttr", "last_segment_ttr", "first_segment_digit_ratio",
    "last_segment_digit_ratio", "first_segment_punctuation_ratio",
    "last_segment_punctuation_ratio", "first_segment_mean_token_chars",
    "last_segment_mean_token_chars", "first_last_ttr_difference",
    "first_last_mean_token_chars_difference", "wordpiece_count_with_special_tokens",
    "truncated_512", "omitted_wordpiece_count", "omitted_wordpiece_fraction",
    "retained_wordpiece_fraction",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def structural_normalize(value: str) -> str:
    text = unicodedata.normalize("NFC", value)
    text = "".join(
        char if char in "\n\t\r" or unicodedata.category(char) not in {"Cc", "Cf", "Cs", "Co", "Cn"} else " "
        for char in text
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [PROTOCOL, CT_SOURCE, *UU_PARQUETS, CT_HELPER]
    for directory in (CT_OUT, UU_OUT):
        required.extend([
            directory / "SAMPLE_MANIFEST.csv",
            directory / "CHECKPOINT_INVENTORY.csv",
            directory / "MODEL_SCORE_SUMMARY.csv",
            directory / "LENGTH_STRATIFIED_RESULTS.csv",
            directory / "CE_SUPCON_COMPARISON.csv",
        ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing required inputs: {missing}")
    observed_protocol = sha256_file(PROTOCOL)
    if EXPECTED_PROTOCOL_SHA256 == "TO_BE_FROZEN" or observed_protocol != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"analysis protocol hash mismatch: {observed_protocol}")

    ct_manifest = pd.read_csv(CT_OUT / "SAMPLE_MANIFEST.csv")
    uu_manifest = pd.read_csv(UU_OUT / "SAMPLE_MANIFEST.csv")
    for name, manifest in (("cturan", ct_manifest), ("Uunan", uu_manifest)):
        if len(manifest) != 2000 or manifest.sample_id.nunique() != 2000:
            raise RuntimeError(f"{name} manifest identity/count failure")
        if manifest.normalized_text_sha256.nunique() != 2000:
            raise RuntimeError(f"{name} manifest normalized duplicates")
        if set(manifest.length_band) != set(BANDS):
            raise RuntimeError(f"{name} length bands changed")
    ct_inventory = pd.read_csv(CT_OUT / "CHECKPOINT_INVENTORY.csv")
    uu_inventory = pd.read_csv(UU_OUT / "CHECKPOINT_INVENTORY.csv")
    identity = [
        "method", "objective", "seed", "outer_fold", "checkpoint_sha256",
        "platt_coefficient", "platt_intercept", "frozen_threshold_alpha_0_03",
    ]
    if not ct_inventory[identity].reset_index(drop=True).equals(uu_inventory[identity].reset_index(drop=True)):
        raise RuntimeError("cturan/Uunan checkpoint or threshold identity differs")
    if len(ct_inventory) != 6 or not ct_inventory.identity_checks_passed.all():
        raise RuntimeError("six verified checkpoint cells were not found")
    return ct_manifest, uu_manifest, ct_inventory


def retrieve_cturan(manifest: pd.DataFrame) -> dict[str, str]:
    targets = dict(zip(manifest.line_number.astype(int), manifest.sample_id))
    found: dict[str, str] = {}
    with CT_SOURCE.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number not in targets:
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str):
                raise RuntimeError(f"cturan target line lacks text: {line_number}")
            found[targets[line_number]] = text
            if len(found) == len(targets):
                break
    return found


def retrieve_uunan(manifest: pd.DataFrame) -> dict[str, str]:
    import pyarrow.parquet as pq

    found: dict[str, str] = {}
    for path in UU_PARQUETS:
        part = manifest[manifest.source_file.eq(path.name)]
        targets = dict(zip(part.source_file_row_number.astype(int), part.sample_id))
        local_row = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096, columns=["text"], use_threads=True):
            for value in batch.column(0).to_pylist():
                local_row += 1
                if local_row in targets:
                    if not isinstance(value, str):
                        raise RuntimeError(f"Uunan target row lacks text: {path.name}/{local_row}")
                    found[targets[local_row]] = value
            if len(found) >= len(part) and all(sample in found for sample in part.sample_id):
                break
    return found


def verify_and_order_texts(corpus: str, manifest: pd.DataFrame, found: dict[str, str]) -> list[str]:
    if set(found) != set(manifest.sample_id):
        missing = set(manifest.sample_id) - set(found)
        raise RuntimeError(f"{corpus} frozen text retrieval incomplete: {len(missing)} missing")
    ordered = []
    for row in manifest.itertuples(index=False):
        text = found[row.sample_id]
        raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        normalized_hash = hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()
        if raw_hash != row.text_sha256 or normalized_hash != row.normalized_text_sha256:
            raise RuntimeError(f"{corpus} frozen text hash mismatch: {row.sample_id}")
        ordered.append(text)
    return ordered


def safe_mean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()) if len(array) else 0.0


def safe_sd(values) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def safe_cv(values) -> float:
    mean = safe_mean(values)
    return safe_sd(values) / mean if mean else 0.0


def ratio_category(text: str, prefix: str) -> float:
    if not text:
        return 0.0
    if prefix == "digit":
        return sum(char.isdigit() for char in text) / len(text)
    return sum(unicodedata.category(char).startswith("P") for char in text) / len(text)


def repeated_ngram_fraction(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    return sum(counts[gram] - 1 for gram in counts if counts[gram] > 1) / len(grams)


def document_features(raw_text: str) -> dict:
    text = structural_normalize(raw_text)
    tokens_original = TOKEN_RE.findall(text)
    tokens = [token.casefold() for token in tokens_original]
    token_counts = Counter(tokens)
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    paragraphs = [part.strip() for part in PARAGRAPH_SPLIT_RE.split(text) if part.strip()]
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    sentence_words = [len(TOKEN_RE.findall(sentence)) for sentence in sentences]
    sentence_chars = [len(sentence) for sentence in sentences]
    paragraph_words = [len(TOKEN_RE.findall(paragraph)) for paragraph in paragraphs]
    normalized_sentences = [canonical(sentence) for sentence in sentences]
    normalized_paragraphs = [canonical(paragraph) for paragraph in paragraphs]
    sentence_counts = Counter(normalized_sentences)
    paragraph_counts = Counter(normalized_paragraphs)
    headings = []
    for line in lines:
        letters = [char for char in line if char.isalpha()]
        all_upper = bool(letters) and all(char.isupper() for char in letters)
        headings.append(
            len(line) <= 120
            and (line.endswith(":") or all_upper or line.startswith("#") or bool(NUMBERED_RE.match(line)))
        )
    first_line_title = False
    if lines:
        second_length = len(lines[1]) if len(lines) > 1 else math.inf
        first_line_title = len(lines[0]) <= 120 and not re.search(r"[.!?]\s*$", lines[0]) and len(lines[0]) < second_length

    segment_n = min(len(tokens), max(20, int(math.ceil(len(tokens) * 0.10)))) if tokens else 0
    first_tokens = tokens[:segment_n]
    last_tokens = tokens[-segment_n:] if segment_n else []
    char_window = min(len(text), max(200, int(math.ceil(len(text) * 0.10)))) if text else 0
    first_chars = text[:char_window]
    last_chars = text[-char_window:] if char_window else ""

    def ttr(values: list[str]) -> float:
        return len(set(values)) / len(values) if values else 0.0

    first_ttr, last_ttr = ttr(first_tokens), ttr(last_tokens)
    first_mean_chars, last_mean_chars = safe_mean(map(len, first_tokens)), safe_mean(map(len, last_tokens))
    character_count = len(text)
    return {
        "character_count": character_count,
        "whitespace_word_count": len(text.split()),
        "lexical_token_count": len(tokens),
        "sentence_count": len(sentences),
        "sentence_words_mean": safe_mean(sentence_words),
        "sentence_words_sd": safe_sd(sentence_words),
        "sentence_words_cv": safe_cv(sentence_words),
        "sentence_chars_mean": safe_mean(sentence_chars),
        "sentence_chars_sd": safe_sd(sentence_chars),
        "paragraph_count": len(paragraphs),
        "paragraph_words_mean": safe_mean(paragraph_words),
        "paragraph_words_sd": safe_sd(paragraph_words),
        "paragraph_words_cv": safe_cv(paragraph_words),
        "short_line_rate": sum(len(line) <= 120 for line in lines) / len(lines) if lines else 0.0,
        "heading_line_count": int(sum(headings)),
        "heading_line_rate": sum(headings) / len(lines) if lines else 0.0,
        "numbered_section_count": int(sum(bool(NUMBERED_RE.match(line)) for line in lines)),
        "title_like_first_line": int(first_line_title),
        "colon_line_rate": sum(":" in line for line in lines) / len(lines) if lines else 0.0,
        "digit_character_ratio": ratio_category(text, "digit"),
        "punctuation_character_ratio": ratio_category(text, "punctuation"),
        "newline_per_1000_chars": text.count("\n") * 1000.0 / character_count if character_count else 0.0,
        "type_token_ratio": ttr(tokens),
        "root_ttr": len(set(tokens)) / math.sqrt(len(tokens)) if tokens else 0.0,
        "hapax_ratio": sum(value == 1 for value in token_counts.values()) / len(token_counts) if token_counts else 0.0,
        "mean_token_characters": safe_mean(map(len, tokens)),
        "repeated_sentence_fraction": sum(value - 1 for value in sentence_counts.values() if value > 1) / len(sentences) if sentences else 0.0,
        "repeated_paragraph_fraction": sum(value - 1 for value in paragraph_counts.values() if value > 1) / len(paragraphs) if paragraphs else 0.0,
        "repeated_word_3gram_fraction": repeated_ngram_fraction(tokens, 3),
        "repeated_word_5gram_fraction": repeated_ngram_fraction(tokens, 5),
        "max_sentence_multiplicity": max(sentence_counts.values(), default=0),
        "_opening_key": " ".join(tokens[:8]),
        "_closing_key": " ".join(tokens[-8:]),
        "first_segment_ttr": first_ttr,
        "last_segment_ttr": last_ttr,
        "first_segment_digit_ratio": ratio_category("".join(first_tokens), "digit"),
        "last_segment_digit_ratio": ratio_category("".join(last_tokens), "digit"),
        "first_segment_punctuation_ratio": ratio_category(first_chars, "punctuation"),
        "last_segment_punctuation_ratio": ratio_category(last_chars, "punctuation"),
        "first_segment_mean_token_chars": first_mean_chars,
        "last_segment_mean_token_chars": last_mean_chars,
        "first_last_ttr_difference": first_ttr - last_ttr,
        "first_last_mean_token_chars_difference": first_mean_chars - last_mean_chars,
    }


def build_feature_frame(manifests: dict[str, pd.DataFrame], texts: dict[str, list[str]]) -> pd.DataFrame:
    from transformers import AutoTokenizer
    from turkish_ai_detection.experiments.primary import run_fit as primary

    rows = []
    ordered_raw = []
    for corpus in ("cturan", "Uunan"):
        manifest = manifests[corpus]
        for index, raw_text in enumerate(texts[corpus]):
            row = document_features(raw_text)
            row.update({
                "corpus": corpus,
                "sample_id": manifest.iloc[index].sample_id,
                "length_band": manifest.iloc[index].length_band,
            })
            rows.append(row)
            ordered_raw.append(raw_text)
    frame = pd.DataFrame(rows)
    opening_counts = Counter(frame._opening_key)
    closing_counts = Counter(frame._closing_key)
    frame["opening_template_frequency"] = frame._opening_key.map(opening_counts).astype(int)
    frame["closing_template_frequency"] = frame._closing_key.map(closing_counts).astype(int)
    frame = frame.drop(columns=["_opening_key", "_closing_key"])

    tokenizer = AutoTokenizer.from_pretrained(primary.MODEL_NAME, revision=primary.REVISION)
    wordpiece_lengths = []
    for start in range(0, len(ordered_raw), 32):
        batch = [structural_normalize(text) for text in ordered_raw[start:start + 32]]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False, padding=False)
        wordpiece_lengths.extend(map(len, encoded["input_ids"]))
    lengths = np.asarray(wordpiece_lengths, dtype=int)
    frame["wordpiece_count_with_special_tokens"] = lengths
    frame["truncated_512"] = (lengths > 512).astype(int)
    frame["omitted_wordpiece_count"] = np.maximum(0, lengths - 512)
    frame["omitted_wordpiece_fraction"] = np.maximum(0, lengths - 512) / np.maximum(lengths, 1)
    frame["retained_wordpiece_fraction"] = np.minimum(lengths, 512) / np.maximum(lengths, 1)
    missing = set(FEATURES) - set(frame.columns)
    if missing or not np.isfinite(frame[list(FEATURES)].to_numpy(float)).all():
        raise RuntimeError(f"invalid feature matrix; missing={sorted(missing)}")
    return frame


def obtain_record_scores(
    inventory: pd.DataFrame,
    manifests: dict[str, pd.DataFrame],
    texts: dict[str, list[str]],
    execute_inference: bool,
) -> pd.DataFrame:
    cache = OUT / "_RECORD_LEVEL_MODEL_SCORES.csv"
    if cache.is_file():
        scores = pd.read_csv(cache)
    else:
        if not execute_inference:
            raise RuntimeError(
                "record-level scores are absent; rerun with --execute-inference to score only the frozen 4,000 texts with the six fixed checkpoints"
            )
        helper = load_module(CT_HELPER, "cross_corpus_frozen_scoring")
        parts = []
        for corpus in ("cturan", "Uunan"):
            scored = helper.score_dataset(inventory, texts[corpus])
            scored.insert(0, "sample_id", manifests[corpus].sample_id.to_numpy())
            scored.insert(0, "corpus", corpus)
            scored["length_band"] = manifests[corpus].length_band.to_numpy()
            parts.append(scored)
        scores = pd.concat(parts, ignore_index=True)
        temporary = cache.with_suffix(".csv.tmp")
        scores.to_csv(temporary, index=False)
        os.replace(temporary, cache)
    expected = {(corpus, sample_id) for corpus, manifest in manifests.items() for sample_id in manifest.sample_id}
    observed = set(zip(scores.corpus, scores.sample_id))
    required_score_columns = {
        f"{kind}_{objective}_s{seed}_f{fold}"
        for kind in ("raw", "prob", "flag")
        for _method, objective in METHODS
        for seed, fold in CELLS
    }
    if observed != expected or len(scores) != 4000 or not required_score_columns.issubset(scores.columns):
        raise RuntimeError("record-level score cache identity/schema failure")
    return scores


def pooled_smd(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    denominator = math.sqrt(((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1)) / max(len(left) + len(right) - 2, 1))
    return float((right.mean() - left.mean()) / denominator) if denominator else 0.0


def cliff_delta(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu
    left, right = np.asarray(left, float), np.asarray(right, float)
    u = float(mannwhitneyu(right, left, alternative="two-sided").statistic)
    return 2.0 * u / (len(left) * len(right)) - 1.0


def stratified_boot_indices(frame: pd.DataFrame, replicates: int, seed_offset: int = 0) -> np.ndarray:
    rng = np.random.default_rng(SEED + seed_offset)
    groups = [np.flatnonzero(frame.length_band.to_numpy() == band) for band in BANDS]
    rows = []
    for _ in range(replicates):
        rows.append(np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups if len(group)]))
    return np.asarray(rows, dtype=int)


def ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_two_group(left: np.ndarray, right: np.ndarray, seed_offset: int) -> dict:
    rng = np.random.default_rng(SEED + seed_offset)
    left_index = rng.integers(0, len(left), size=(BOOTSTRAPS, len(left)))
    right_index = rng.integers(0, len(right), size=(BOOTSTRAPS, len(right)))
    left_draw = left[left_index]
    right_draw = right[right_index]
    differences = right_draw.mean(axis=1) - left_draw.mean(axis=1)
    left_var = left_draw.var(axis=1, ddof=1)
    right_var = right_draw.var(axis=1, ddof=1)
    pooled = np.sqrt(((len(left) - 1) * left_var + (len(right) - 1) * right_var) / max(len(left) + len(right) - 2, 1))
    smd = np.divide(differences, pooled, out=np.zeros_like(differences), where=pooled > 0)
    diff_low, diff_high = ci(differences)
    smd_low, smd_high = ci(smd)
    return {
        "mean_difference_ci_low": diff_low, "mean_difference_ci_high": diff_high,
        "standardized_mean_difference_ci_low": smd_low,
        "standardized_mean_difference_ci_high": smd_high,
    }


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def length_design(frame: pd.DataFrame, include_bands: bool = True) -> np.ndarray:
    columns = [np.ones(len(frame)), np.log1p(frame.character_count.to_numpy(float))]
    if include_bands:
        for band in BANDS[1:]:
            columns.append(frame.length_band.eq(band).astype(float).to_numpy())
    return np.column_stack(columns)


def corpus_feature_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus_rows, controlled_rows = [], []
    ct = frame[frame.corpus.eq("cturan")].reset_index(drop=True)
    uu = frame[frame.corpus.eq("Uunan")].reset_index(drop=True)
    for feature_index, feature in enumerate(FEATURES):
        left, right = ct[feature].to_numpy(float), uu[feature].to_numpy(float)
        uncertainty = bootstrap_two_group(left, right, 1000 + feature_index)
        corpus_rows.append({
            "feature": feature, "cturan_n": len(left), "uunan_n": len(right),
            "cturan_mean": left.mean(), "uunan_mean": right.mean(),
            "cturan_median": np.median(left), "uunan_median": np.median(right),
            "mean_difference_uunan_minus_cturan": right.mean() - left.mean(),
            "median_difference_uunan_minus_cturan": np.median(right) - np.median(left),
            "standardized_mean_difference": pooled_smd(left, right),
            "cliff_delta_uunan_greater": cliff_delta(left, right),
            **uncertainty,
            "interpretation": "source-feature association only; not AI-authorship evidence",
        })
        for band_index, band in enumerate(BANDS):
            band_left = ct.loc[ct.length_band.eq(band), feature].to_numpy(float)
            band_right = uu.loc[uu.length_band.eq(band), feature].to_numpy(float)
            uncertainty = bootstrap_two_group(band_left, band_right, 2000 + feature_index * 10 + band_index)
            controlled_rows.append({
                "analysis": "within_length_band", "feature": feature, "length_band": band,
                "cturan_n": len(band_left), "uunan_n": len(band_right),
                "cturan_mean": band_left.mean(), "uunan_mean": band_right.mean(),
                "mean_difference_uunan_minus_cturan": band_right.mean() - band_left.mean(),
                "standardized_mean_difference": pooled_smd(band_left, band_right),
                "cliff_delta_uunan_greater": cliff_delta(band_left, band_right),
                **uncertainty,
            })
        combined = pd.concat([ct, uu], ignore_index=True)
        residual = residualize(combined[feature].to_numpy(float), length_design(combined))
        residual_left, residual_right = residual[:len(ct)], residual[len(ct):]
        uncertainty = bootstrap_two_group(residual_left, residual_right, 3000 + feature_index)
        controlled_rows.append({
            "analysis": "residualized_log_length_and_band", "feature": feature, "length_band": "ALL",
            "cturan_n": len(residual_left), "uunan_n": len(residual_right),
            "cturan_mean": residual_left.mean(), "uunan_mean": residual_right.mean(),
            "mean_difference_uunan_minus_cturan": residual_right.mean() - residual_left.mean(),
            "standardized_mean_difference": pooled_smd(residual_left, residual_right),
            "cliff_delta_uunan_greater": cliff_delta(residual_left, residual_right),
            **uncertainty,
        })
    return pd.DataFrame(corpus_rows), pd.DataFrame(controlled_rows)


def rank_columns(matrix: np.ndarray) -> np.ndarray:
    return pd.DataFrame(matrix).rank(method="average").to_numpy(float)


def correlation_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    denominator = np.sqrt((x * x).sum(axis=0)[:, None] * (y * y).sum(axis=0)[None, :])
    return np.divide(x.T @ y, denominator, out=np.zeros((x.shape[1], y.shape[1])), where=denominator > 0)


def association_block(
    frame: pd.DataFrame,
    score_names: list[str],
    scope: str,
    seed_offset: int,
) -> list[dict]:
    x_rank = rank_columns(frame[list(FEATURES)].to_numpy(float))
    y_rank = rank_columns(frame[score_names].to_numpy(float))
    design = length_design(frame, include_bands=scope == "ALL")
    x_partial = np.column_stack([residualize(x_rank[:, index], design) for index in range(x_rank.shape[1])])
    y_partial = np.column_stack([residualize(y_rank[:, index], design) for index in range(y_rank.shape[1])])
    point = correlation_matrix(x_rank, y_rank)
    partial_point = correlation_matrix(x_partial, y_partial)
    boot_indices = stratified_boot_indices(frame, BOOTSTRAPS, seed_offset) if scope == "ALL" else np.asarray([
        np.random.default_rng(SEED + seed_offset + replicate).integers(0, len(frame), size=len(frame))
        for replicate in range(BOOTSTRAPS)
    ])
    boot = np.empty((BOOTSTRAPS, len(FEATURES), len(score_names)), dtype=np.float32)
    boot_partial = np.empty_like(boot)
    for replicate, indices in enumerate(boot_indices):
        boot[replicate] = correlation_matrix(x_rank[indices], y_rank[indices])
        boot_partial[replicate] = correlation_matrix(x_partial[indices], y_partial[indices])
    rows = []
    for feature_index, feature in enumerate(FEATURES):
        for score_index, score_name in enumerate(score_names):
            low, high = ci(boot[:, feature_index, score_index])
            partial_low, partial_high = ci(boot_partial[:, feature_index, score_index])
            rows.append({
                "corpus": frame.corpus.iloc[0], "scope": scope, "n": len(frame),
                "score_target": score_name, "feature": feature,
                "spearman_rho": point[feature_index, score_index],
                "spearman_ci_low": low, "spearman_ci_high": high,
                "partial_rank_rho_length_controlled": partial_point[feature_index, score_index],
                "partial_rank_ci_low": partial_low, "partial_rank_ci_high": partial_high,
                "bootstrap_replicates": BOOTSTRAPS,
            })
    return rows


def feature_score_associations(analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score_names = [f"prob_{objective}_s{seed}_f{fold}" for _method, objective in METHODS for seed, fold in CELLS]
    score_names += ["prob_ce_mean", "prob_supcon_mean"]
    for corpus_index, corpus in enumerate(("cturan", "Uunan")):
        subset = analysis[analysis.corpus.eq(corpus)].reset_index(drop=True)
        rows.extend(association_block(subset, score_names, "ALL", 4000 + corpus_index * 100))
        for band_index, band in enumerate(BANDS):
            band_frame = subset[subset.length_band.eq(band)].reset_index(drop=True)
            if len(band_frame) < 30:
                continue
            rows.extend(association_block(band_frame, score_names, band, 4100 + corpus_index * 100 + band_index))
    return pd.DataFrame(rows)


def ce_supcon_analysis(analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for corpus_index, corpus in enumerate(("cturan", "Uunan")):
        corpus_frame = analysis[analysis.corpus.eq(corpus)].reset_index(drop=True)
        scopes = [("ALL", corpus_frame)] + [
            (band, corpus_frame[corpus_frame.length_band.eq(band)].reset_index(drop=True)) for band in BANDS
        ]
        for scope_index, (scope, frame) in enumerate(scopes):
            for method_index, (method, objective) in enumerate(METHODS):
                score = frame[f"prob_{objective}_mean"].to_numpy(float)
                low_cut, high_cut = np.quantile(score, [0.25, 0.75])
                low_mask, high_mask = score <= low_cut, score >= high_cut
                for feature_index, feature in enumerate(FEATURES):
                    low_values = frame.loc[low_mask, feature].to_numpy(float)
                    high_values = frame.loc[high_mask, feature].to_numpy(float)
                    uncertainty = bootstrap_two_group(
                        low_values, high_values,
                        5000 + corpus_index * 10000 + scope_index * 1000 + method_index * 100 + feature_index,
                    )
                    rows.append({
                        "analysis": "high_vs_low_method_mean_score", "corpus": corpus,
                        "scope": scope, "method": method, "feature": feature,
                        "low_n": len(low_values), "high_n": len(high_values),
                        "low_feature_mean": low_values.mean(), "high_feature_mean": high_values.mean(),
                        "high_minus_low_mean": high_values.mean() - low_values.mean(),
                        "standardized_mean_difference": pooled_smd(low_values, high_values),
                        "low_score_cut": low_cut, "high_score_cut": high_cut,
                        **uncertainty,
                    })
            delta_name = "prob_supcon_minus_ce_mean"
            block = association_block(frame.assign(corpus=corpus), [delta_name], scope, 7000 + corpus_index * 100 + scope_index)
            for item in block:
                rows.append({
                    "analysis": "feature_association_with_supcon_minus_ce_probability",
                    "corpus": corpus, "scope": scope, "method": "SUPCON_MINUS_CE",
                    "feature": item["feature"], "n": item["n"],
                    "partial_rank_rho_length_controlled": item["partial_rank_rho_length_controlled"],
                    "partial_rank_ci_low": item["partial_rank_ci_low"],
                    "partial_rank_ci_high": item["partial_rank_ci_high"],
                    "bootstrap_replicates": BOOTSTRAPS,
                })
        for group in ("both_ai", "ce_only_ai", "supcon_only_ai", "both_not_ai"):
            group_frame = corpus_frame[corpus_frame.majority_decision_group.eq(group)]
            for feature in FEATURES:
                rows.append({
                    "analysis": "majority_decision_group_feature_summary",
                    "corpus": corpus, "scope": "ALL", "method": "CE_VS_SUPCON",
                    "decision_group": group, "feature": feature, "n": len(group_frame),
                    "feature_mean": group_frame[feature].mean() if len(group_frame) >= 20 else np.nan,
                    "feature_median": group_frame[feature].median() if len(group_frame) >= 20 else np.nan,
                    "estimability": "reported" if len(group_frame) >= 20 else "suppressed_n_lt_20",
                })
    return pd.DataFrame(rows)


def build_findings(
    corpus_features: pd.DataFrame,
    controlled: pd.DataFrame,
    associations: pd.DataFrame,
    ce_analysis: pd.DataFrame,
    analysis: pd.DataFrame,
) -> str:
    residual = controlled[controlled.analysis.eq("residualized_log_length_and_band")].copy()
    residual["magnitude"] = residual.standardized_mean_difference.abs()
    top_source = residual.nlargest(8, "magnitude")
    source_lines = "\n".join(
        f"- `{row.feature}`: uzunluk/bant artıklarında Uunan−cturan SMD {row.standardized_mean_difference:+.3f} (%95 bootstrap {row.standardized_mean_difference_ci_low:+.3f}–{row.standardized_mean_difference_ci_high:+.3f})."
        for row in top_source.itertuples(index=False)
    )
    overall_assoc = associations[
        associations.scope.eq("ALL")
        & associations.score_target.isin(["prob_ce_mean", "prob_supcon_mean"])
    ].copy()
    overall_assoc["magnitude"] = overall_assoc.partial_rank_rho_length_controlled.abs()
    top_assoc = overall_assoc.nlargest(10, "magnitude")
    association_lines = "\n".join(
        f"- {row.corpus}, `{row.score_target}`, `{row.feature}`: length-controlled rank ilişki {row.partial_rank_rho_length_controlled:+.3f} (%95 bootstrap {row.partial_rank_ci_low:+.3f}–{row.partial_rank_ci_high:+.3f})."
        for row in top_assoc.itertuples(index=False)
    )
    delta = ce_analysis[
        ce_analysis.analysis.eq("feature_association_with_supcon_minus_ce_probability")
        & ce_analysis.scope.eq("ALL")
    ].copy()
    delta["magnitude"] = delta.partial_rank_rho_length_controlled.abs()
    delta_lines = "\n".join(
        f"- {row.corpus}, `{row.feature}`: SupCon−CE olasılık farkıyla partial-rank {row.partial_rank_rho_length_controlled:+.3f} (%95 bootstrap {row.partial_rank_ci_low:+.3f}–{row.partial_rank_ci_high:+.3f})."
        for row in delta.nlargest(6, "magnitude").itertuples(index=False)
    )
    rates = analysis.groupby("corpus")[["ce_majority_ai", "supcon_majority_ai"]].mean()
    return f"""# Cross-Corpus Error Analysis — Scientific Findings

Durum: **EXPLORATORY_COMPLETE_NO_CAUSAL_CLAIM**

## Soruya kısa yanıt

Turkish AI Detection'in cturan ve Uunan arasındaki farklı işaretleme davranışı, aşağıdaki kaynak-yapı özellikleri ve model skoru ilişkileriyle birlikte gözlenmektedir; ancak bunların hiçbiri korpus farkının nedeni veya gerçek AI yazarlığının göstergesi olarak doğrulanmış değildir.

## Uzunluk kontrolünden sonra en büyük kaynak–özellik farkları

{source_lines}

Bu liste bir kaynak profili tarif eder. Özelliklerin cturan/Uunan kaynağını ayırması, HUMAN/AI ayrımına genellenemez.

## Model skorlarıyla en güçlü gözlenen ilişkiler

{association_lines}

Bu ilişkiler aynı özelliklerin korpuslar arasında farklı dağılım göstermesi, BERTurk truncation düzeni veya gözlenmeyen konu/üretici farklarıyla karışmış olabilir.

## CE ve SupCon farkıyla ilişkili özellikler

{delta_lines}

Çoğunluk kararıyla AI işaretleme oranları cturan'da CE {rates.loc['cturan','ce_majority_ai']:.2%}, SupCon {rates.loc['cturan','supcon_majority_ai']:.2%}; Uunan'da CE {rates.loc['Uunan','ce_majority_ai']:.2%}, SupCon {rates.loc['Uunan','supcon_majority_ai']:.2%} olarak gözlenmiştir. Bunlar kaynak beyanına dayalı keşifsel kapsama ölçüleridir; doğrulanmış duyarlılık değildir.

## Mevcut kanıtla açıklanamayanlar

- Özellikler ile skorlar arasındaki ilişki, modelin bu özelliği nedensel olarak kullandığını göstermez.
- Korpus üreticisi/modeli, prompt kaynağı, konu bileşimi, kalite filtreleri ve oluşturma ardışıklığı kayıt düzeyinde mevcut değildir; etkileri ayrıştırılamaz.
- Kaynaklarda HUMAN örnek yoktur. Hangi özelliklerin gerçek AI yazarlığını ayırdığı, hata oranı, FPR, specificity veya ROC-AUC bu analizden çıkarılamaz.
- Uzunluk bantları ve log-uzunluk kontrolü, konu/biçem/üretici karışmasını ortadan kaldırmaz.
- BERTurk 512-token truncation göstergeleri ilişkiyi betimler; alternatif truncation çalıştırılmadığı için etkisinin nedensel payı ölçülmez.
- Çok sayıda keşifsel özellik ve alt grup incelenmiştir; sıralamalar bağımsız doğrulama gerektirir.

Yeni model, kalibrasyon veya eşik üretilmedi; mevcut örnek ve checkpoint kimlikleri değiştirilmedi.
"""


def limitations_text() -> str:
    return """# Limitations

1. **Kaynak etiketi, yazarlık etiketi değildir.** Her iki korpus yalnız kaynak tarafından sentetik olarak bildirilmiştir. Kaynak farkını açıklayan özellikler HUMAN/AI ayrımı olarak yorumlanamaz.
2. **Nedensellik yoktur.** Özellik–skor korelasyonları müdahale, karşı-olgusal varyant veya model içi mekanizma kanıtı değildir.
3. **Gözlenmeyen karıştırıcılar vardır.** Üretici model, prompt, konu, kalite filtresi ve belge üretim geçmişi kayıt düzeyinde bulunmaz.
4. **Sabit stratifiye örnekler popülasyon-ağırlıklı değildir.** cturan ve Uunan uzunluk kotası farklıdır; bant-içi analiz bunu azaltır fakat temsil sorununu çözmez.
5. **Kayıt düzeyi skorlar yeniden inference gerektirir.** Skorlar aynı altı dondurulmuş checkpoint ve aynı metinlerde yeniden üretilir; eski agregat dosyalar değiştirilmez. Ortam kaynaklı çok küçük sayısal farklar mümkündür.
6. **Tokenizer truncation gözlemseldir.** WordPiece kaybı ölçülür fakat alternatif input penceresi denenmez.
7. **Cümle/paragraf sezgiseldir.** Regex tabanlı Türkçe cümle, başlık ve bölüm tanımları dilbilimsel altın standart değildir.
8. **Bağımlılık kısmen ele alınır.** Aynı belge için altı skor ortak bootstrap çekilişiyle taşınır; checkpoint'ler yalnız üç seçilmiş seed/fold hücresidir ve daha geniş model popülasyonunu temsil etmez.
9. **Çoklu keşif vardır.** 49 özellik, iki korpus, modeller ve bantlar taranır. P-değeri/confirmatory significance kullanılmaz; bootstrap aralıkları seçim çokluğunu düzeltmez.
10. **Lisans/provenance sınırı.** Uunan `other` lisanslıdır; analiz onu eğitim verisi olarak onaylamaz. cturan ve Uunan üretim beyanları bağımsız doğrulanmamıştır.
11. **Ham metin incelemesi raporlanmaz.** Gizlilik/koruma nedeniyle nitel örnekler verilmez; bazı şablon örüntülerinin semantik yorumu sınırlıdır.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute-inference",
        action="store_true",
        help="Explicitly authorize inference for the fixed 4,000 documents and six existing checkpoints; never trains.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ct_manifest, uu_manifest, inventory = validate_inputs()
    manifests = {"cturan": ct_manifest, "Uunan": uu_manifest}

    score_cache = OUT / "_RECORD_LEVEL_MODEL_SCORES.csv"
    if not score_cache.is_file() and not args.execute_inference:
        raise RuntimeError(
            "record-level predictions are absent. No long scan or inference was started. "
            "Review ANALYSIS_PROTOCOL.md, then explicitly rerun with --execute-inference."
        )

    ct_texts = verify_and_order_texts("cturan", ct_manifest, retrieve_cturan(ct_manifest))
    uu_texts = verify_and_order_texts("Uunan", uu_manifest, retrieve_uunan(uu_manifest))
    texts = {"cturan": ct_texts, "Uunan": uu_texts}
    features = build_feature_frame(manifests, texts)
    scores = obtain_record_scores(inventory, manifests, texts, args.execute_inference)
    analysis = features.merge(scores, on=["corpus", "sample_id", "length_band"], validate="one_to_one")

    ce_columns = [f"prob_ce_s{seed}_f{fold}" for seed, fold in CELLS]
    supcon_columns = [f"prob_supcon_s{seed}_f{fold}" for seed, fold in CELLS]
    ce_flag_columns = [f"flag_ce_s{seed}_f{fold}" for seed, fold in CELLS]
    supcon_flag_columns = [f"flag_supcon_s{seed}_f{fold}" for seed, fold in CELLS]
    analysis["prob_ce_mean"] = analysis[ce_columns].mean(axis=1)
    analysis["prob_supcon_mean"] = analysis[supcon_columns].mean(axis=1)
    analysis["prob_supcon_minus_ce_mean"] = analysis.prob_supcon_mean - analysis.prob_ce_mean
    analysis["ce_majority_ai"] = analysis[ce_flag_columns].astype(int).sum(axis=1).ge(2)
    analysis["supcon_majority_ai"] = analysis[supcon_flag_columns].astype(int).sum(axis=1).ge(2)
    analysis["majority_decision_group"] = np.select(
        [
            analysis.ce_majority_ai & analysis.supcon_majority_ai,
            analysis.ce_majority_ai & ~analysis.supcon_majority_ai,
            ~analysis.ce_majority_ai & analysis.supcon_majority_ai,
        ],
        ["both_ai", "ce_only_ai", "supcon_only_ai"],
        default="both_not_ai",
    )

    corpus_features, controlled = corpus_feature_tables(analysis)
    associations = feature_score_associations(analysis)
    ce_analysis = ce_supcon_analysis(analysis)
    corpus_features.to_csv(OUT / "CORPUS_FEATURE_COMPARISON.csv", index=False)
    controlled.to_csv(OUT / "LENGTH_CONTROLLED_FEATURES.csv", index=False)
    associations.to_csv(OUT / "FEATURE_SCORE_ASSOCIATIONS.csv", index=False)
    ce_analysis.to_csv(OUT / "CE_SUPCON_ERROR_ANALYSIS.csv", index=False)
    features.to_csv(OUT / "_DOCUMENT_FEATURES.csv", index=False)
    (OUT / "SCIENTIFIC_FINDINGS.md").write_text(
        build_findings(corpus_features, controlled, associations, ce_analysis, analysis), encoding="utf-8"
    )
    (OUT / "LIMITATIONS.md").write_text(limitations_text(), encoding="utf-8")
    metadata = {
        "status": "COMPLETE_EXPLORATORY_NO_TRAINING",
        "created_utc": utc_now(),
        "analysis_protocol_sha256": sha256_file(PROTOCOL),
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "documents": len(analysis),
        "cturan_manifest_sha256": sha256_file(CT_OUT / "SAMPLE_MANIFEST.csv"),
        "uunan_manifest_sha256": sha256_file(UU_OUT / "SAMPLE_MANIFEST.csv"),
        "checkpoint_inventory_sha256": sha256_file(CT_OUT / "CHECKPOINT_INVENTORY.csv"),
        "record_score_cache_sha256": sha256_file(OUT / "_RECORD_LEVEL_MODEL_SCORES.csv"),
        "new_models_trained": 0,
        "new_calibrations_fit": 0,
        "new_thresholds_fit": 0,
    }
    (OUT / "_RUN_METADATA.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del texts, ct_texts, uu_texts
    gc.collect()
    print(json.dumps({
        "status": metadata["status"],
        "documents": metadata["documents"],
        "new_models_trained": 0,
        "output": str(OUT),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
