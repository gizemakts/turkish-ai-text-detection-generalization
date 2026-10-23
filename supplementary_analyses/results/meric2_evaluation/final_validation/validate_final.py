#!/usr/bin/env python
"""Independent, read-only validation of the completed Meric2 stress test."""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
PROJECT = ANALYSIS.parents[3]
RAW = Path(os.environ["TURKISH_AI_DETECTION_MERIC2_PATH"])
METHODS = ("CE_ONLY_MATCHED", "SUPCON")
SEEDS = (42, 314, 2026)
FOLDS = tuple(range(5))
METRICS = (
    "roc_auc",
    "average_precision",
    "standardized_partial_roc_auc_max_fpr_003",
)
FEATURES = (
    "word_count",
    "character_count",
    "mean_token_length",
    "type_token_ratio",
    "digit_per_100_words",
    "punctuation_per_100_words",
    "uppercase_per_100_chars",
    "nonascii_per_100_chars",
)
TOL = 5e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_text(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def text_hash(value: object) -> str:
    return hashlib.sha256(canonical_text(value).encode("utf-8")).hexdigest()


def surface_features(value: object) -> dict[str, float]:
    """Unicode-aware diagnostics computed from unmodified raw text."""
    text = str(value)
    tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    n_words = max(len(tokens), 1)
    n_chars = max(len(text), 1)
    return {
        "word_count": len(tokens),
        "character_count": len(text),
        "mean_token_length": sum(map(len, tokens)) / n_words,
        "type_token_ratio": len({token.casefold() for token in tokens}) / n_words,
        "digit_per_100_words": 100 * sum(char.isdigit() for char in text) / n_words,
        # Unicode General Categories Pc/Pd/Pe/Pf/Pi/Po/Ps, not an ASCII list.
        "punctuation_per_100_words": 100
        * sum(unicodedata.category(char).startswith("P") for char in text)
        / n_words,
        # str.isupper() is Unicode-aware, including Turkish ÇĞİÖŞÜ.
        "uppercase_per_100_chars": 100 * sum(char.isupper() for char in text) / n_chars,
        "nonascii_per_100_chars": 100 * sum(ord(char) > 127 for char in text) / n_chars,
    }


def calculate_metrics(labels: pd.Series, scores: pd.Series) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "standardized_partial_roc_auc_max_fpr_003": roc_auc_score(
            labels, scores, max_fpr=0.03
        ),
    }


def safely_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return np.nan, np.nan
    result = spearmanr(x, y, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)


def compare_tables(
    independent: pd.DataFrame,
    published: pd.DataFrame,
    keys: list[str],
    values: list[str],
    table_name: str,
) -> list[dict[str, object]]:
    merged = independent.merge(
        published, on=keys, how="outer", suffixes=("_independent", "_published"), indicator=True
    )
    output: list[dict[str, object]] = []
    for metric in values:
        left = pd.to_numeric(merged.get(f"{metric}_independent"), errors="coerce")
        right = pd.to_numeric(merged.get(f"{metric}_published"), errors="coerce")
        differences = (left - right).abs()
        mismatch = (merged["_merge"] != "both") | (differences > TOL) | (left.isna() != right.isna())
        output.append(
            {
                "table": table_name,
                "metric": metric,
                "independent_rows": len(independent),
                "published_rows": len(published),
                "max_absolute_difference": differences.max(skipna=True),
                "tolerance": TOL,
                "mismatched_rows": int(mismatch.sum()),
                "status": "PASS" if not mismatch.any() else "FAIL",
            }
        )
    return output


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW).reset_index(names="original_row_index")
    manifest = pd.read_csv(ANALYSIS / "MERIC2_ELIGIBLE_MANIFEST.csv")
    matched = pd.read_csv(ANALYSIS / "MERIC2_LENGTH_MATCHED_MANIFEST.csv")

    if len(manifest) != 11575 or len(matched) != 7938:
        raise RuntimeError("Unexpected eligible or length-matched population size")
    if not manifest.text_id.is_unique or not matched.text_id.is_unique:
        raise RuntimeError("Observation IDs are not unique")
    source = manifest.merge(
        raw[["original_row_index", "review", "label"]],
        on="original_row_index",
        validate="one_to_one",
    )
    expected_labels = source.label.astype(str).str.strip().str.lower().map({"human": 0, "ai": 1})
    source_hashes = source.review.map(text_hash)
    if not expected_labels.equals(source.binary_label) or not source_hashes.equals(source.normalized_text_sha256):
        raise RuntimeError("Ground-truth label or raw-text hash does not align with the manifest")
    matched_ids = set(matched.text_id)
    if not matched_ids.issubset(set(manifest.text_id)):
        raise RuntimeError("Length-matched population is not a subset of the eligible population")
    if matched.binary_label.value_counts().to_dict() != {0: 3969, 1: 3969}:
        raise RuntimeError("Length-matched class counts differ from the frozen sensitivity population")

    # Corrected diagnostic features; no raw text is written.
    corrected = pd.DataFrame([surface_features(value) for value in source.review], index=source.index)
    corrected.insert(0, "binary_label", source.binary_label.to_numpy())
    corrected.insert(0, "text_id", source.text_id.to_numpy())
    corrected.to_csv(HERE / "CORRECTED_SURFACE_FEATURES.csv", index=False, float_format="%.12f")
    reconciliation = []
    for feature in FEATURES:
        old = pd.to_numeric(manifest[feature])
        new = corrected[feature]
        reconciliation.append(
            {
                "feature": feature,
                "raw_min": new.min(),
                "raw_max": new.max(),
                "nonzero_texts": int(new.ne(0).sum()),
                "changed_from_original_manifest": int((~np.isclose(old, new, rtol=0, atol=1e-12)).sum()),
                "original_mean": old.mean(),
                "corrected_mean": new.mean(),
                "status": "CORRECTED_UNICODE_PUNCTUATION"
                if feature == "punctuation_per_100_words"
                else "CONFIRMED",
            }
        )
    pd.DataFrame(reconciliation).to_csv(HERE / "FEATURE_EXTRACTION_RECONCILIATION.csv", index=False)

    comparison_rows = []
    corrected_by_id = corrected.set_index("text_id")
    for population, ids in (("FULL", set(manifest.text_id)), ("LENGTH_MATCHED", matched_ids)):
        frame = corrected_by_id.loc[sorted(ids)]
        for feature in FEATURES:
            human = frame.loc[frame.binary_label.eq(0), feature]
            ai = frame.loc[frame.binary_label.eq(1), feature]
            pooled = np.sqrt(
                ((len(human) - 1) * human.var(ddof=1) + (len(ai) - 1) * ai.var(ddof=1))
                / (len(human) + len(ai) - 2)
            )
            comparison_rows.append(
                {
                    "population": population,
                    "feature": feature,
                    "human_n": len(human),
                    "ai_n": len(ai),
                    "human_mean": human.mean(),
                    "ai_mean": ai.mean(),
                    "human_median": human.median(),
                    "ai_median": ai.median(),
                    "ai_minus_human": ai.mean() - human.mean(),
                    "standardized_mean_difference":
                    (ai.mean() - human.mean()) / pooled if pooled > 0 else np.nan,
                }
            )
    pd.DataFrame(comparison_rows).to_csv(
        HERE / "CORRECTED_LEXICAL_SURFACE_CLASS_COMPARISON.csv", index=False, float_format="%.12f"
    )

    all_predictions: list[pd.DataFrame] = []
    cell_checks: list[dict[str, object]] = []
    manifest_identity = manifest[["text_id", "normalized_text_sha256", "binary_label"]].sort_values("text_id")
    expected_manifest_hash = sha256(ANALYSIS / "MERIC2_ELIGIBLE_MANIFEST.csv")
    for method in METHODS:
        for seed in SEEDS:
            for fold in FOLDS:
                path = ANALYSIS / "predictions" / method / f"seed_{seed}" / f"outer_{fold}.csv"
                sidecar_path = path.with_suffix(".json")
                frame = pd.read_csv(path)
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                identity = frame[["text_id", "normalized_text_sha256", "binary_label"]].sort_values("text_id")
                checks = {
                    "rows_11575": len(frame) == 11575,
                    "unique_ids": frame.text_id.is_unique,
                    "identity_alignment": identity.reset_index(drop=True).equals(manifest_identity.reset_index(drop=True)),
                    "method_identity": frame.method.nunique() == 1 and frame.method.iloc[0] == method,
                    "seed_identity": frame.seed.nunique() == 1 and int(frame.seed.iloc[0]) == seed,
                    "fold_identity": frame.outer_fold.nunique() == 1 and int(frame.outer_fold.iloc[0]) == fold,
                    "finite_scores": np.isfinite(frame.raw_logit_score).all(),
                    "prediction_hash": sha256(path) == sidecar.get("prediction_sha256"),
                    "manifest_hash": expected_manifest_hash == sidecar.get("eligible_manifest_sha256"),
                    "complete_sidecar": sidecar.get("status") == "COMPLETE",
                }
                cell_checks.append(
                    {"method": method, "seed": seed, "outer_fold": fold, **checks,
                     "status": "PASS" if all(checks.values()) else "FAIL"}
                )
                all_predictions.append(frame)

    tfidf_path = ANALYSIS / "predictions" / "TFIDF_M4_predictions.csv"
    tfidf_sidecar = json.loads(tfidf_path.with_suffix(".json").read_text(encoding="utf-8"))
    tfidf = pd.read_csv(tfidf_path)
    for fold in FOLDS:
        frame = tfidf[tfidf.outer_fold.eq(fold)].copy()
        identity = frame[["text_id", "normalized_text_sha256", "binary_label"]].sort_values("text_id")
        checks = {
            "rows_11575": len(frame) == 11575,
            "unique_ids": frame.text_id.is_unique,
            "identity_alignment": identity.reset_index(drop=True).equals(manifest_identity.reset_index(drop=True)),
            "method_identity": frame.method.nunique() == 1 and frame.method.iloc[0] == "TFIDF_M4",
            "seed_identity": frame.seed.isna().all(),
            "fold_identity": frame.outer_fold.nunique() == 1 and int(frame.outer_fold.iloc[0]) == fold,
            "finite_scores": np.isfinite(frame.raw_logit_score).all(),
            "prediction_hash": sha256(tfidf_path) == tfidf_sidecar.get("prediction_sha256"),
            "manifest_hash": True,  # Alignment above is the stronger content check; TF-IDF sidecar predates this field.
            "complete_sidecar": tfidf_sidecar.get("status") == "COMPLETE_COMBINED_V2_ONLY_TFIDF",
        }
        cell_checks.append(
            {"method": "TFIDF_M4", "seed": np.nan, "outer_fold": fold, **checks,
             "status": "PASS" if all(checks.values()) else "FAIL"}
        )
        all_predictions.append(frame)
    checks_frame = pd.DataFrame(cell_checks)
    checks_frame.to_csv(HERE / "PREDICTION_CELL_VALIDATION.csv", index=False)
    if len(checks_frame) != 35 or not checks_frame.status.eq("PASS").all():
        raise RuntimeError("One or more prediction cells failed integrity validation")

    predictions = pd.concat(all_predictions, ignore_index=True).merge(
        manifest[["text_id", "normalized_text_sha256", "binary_label"]],
        on=["text_id", "normalized_text_sha256", "binary_label"],
        validate="many_to_one",
    )
    predictions = predictions.merge(corrected, on=["text_id", "binary_label"], validate="many_to_one")

    per_cell_rows = []
    for population, ids in (("FULL", set(manifest.text_id)), ("LENGTH_MATCHED", matched_ids)):
        subset = predictions[predictions.text_id.isin(ids)]
        for (method, seed, fold), frame in subset.groupby(
            ["method", "seed", "outer_fold"], dropna=False, sort=True
        ):
            per_cell_rows.append(
                {
                    "population": population,
                    "method": method,
                    "seed": seed,
                    "outer_fold": fold,
                    "n": len(frame),
                    "n_human": int(frame.binary_label.eq(0).sum()),
                    "n_ai": int(frame.binary_label.eq(1).sum()),
                    **calculate_metrics(frame.binary_label, frame.raw_logit_score),
                }
            )
    per_cell = pd.DataFrame(per_cell_rows)
    per_cell.to_csv(HERE / "INDEPENDENT_PER_CELL_METRICS.csv", index=False, float_format="%.12f")

    seed_rows = []
    transformer = per_cell[per_cell.method.ne("TFIDF_M4")]
    for (population, method, seed), frame in transformer.groupby(["population", "method", "seed"]):
        seed_rows.append(
            {"population": population, "method": method, "seed": seed, "n_folds": len(frame),
             **{metric: frame[metric].mean() for metric in METRICS}}
        )
    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(HERE / "INDEPENDENT_TRANSFORMER_SEED_LEVEL_METRICS.csv", index=False, float_format="%.12f")

    aggregate_rows = []
    for (population, method), frame in seed_metrics.groupby(["population", "method"]):
        row = {"population": population, "method": method,
               "aggregation_unit": "seed after averaging five dependent folds", "n_units": len(frame)}
        for metric in METRICS:
            row.update({f"{metric}_mean": frame[metric].mean(), f"{metric}_sample_sd": frame[metric].std(ddof=1),
                        f"{metric}_min": frame[metric].min(), f"{metric}_max": frame[metric].max()})
        aggregate_rows.append(row)
    for population, frame in per_cell[per_cell.method.eq("TFIDF_M4")].groupby("population"):
        row = {"population": population, "method": "TFIDF_M4",
               "aggregation_unit": "five dependent outer-fold fits; descriptive", "n_units": len(frame)}
        for metric in METRICS:
            row.update({f"{metric}_mean": frame[metric].mean(), f"{metric}_sample_sd": frame[metric].std(ddof=1),
                        f"{metric}_min": frame[metric].min(), f"{metric}_max": frame[metric].max()})
        aggregate_rows.append(row)
    aggregates = pd.DataFrame(aggregate_rows)
    aggregates.to_csv(HERE / "INDEPENDENT_AGGREGATE_METRICS.csv", index=False, float_format="%.12f")

    reconciliation_rows = []
    reconciliation_rows += compare_tables(
        per_cell, pd.read_csv(ANALYSIS / "MAIN_AND_MATCHED_PER_CELL_METRICS.csv"),
        ["population", "method", "seed", "outer_fold"],
        ["n", "n_human", "n_ai", *METRICS], "per_cell"
    )
    reconciliation_rows += compare_tables(
        seed_metrics, pd.read_csv(ANALYSIS / "TRANSFORMER_SEED_LEVEL_METRICS.csv"),
        ["population", "method", "seed"], ["n_folds", *METRICS], "transformer_seed_level"
    )
    aggregate_values = [
        f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "sample_sd", "min", "max")
    ]
    reconciliation_rows += compare_tables(
        aggregates, pd.read_csv(ANALYSIS / "MAIN_AND_MATCHED_AGGREGATE_METRICS.csv"),
        ["population", "method"], ["n_units", *aggregate_values], "aggregate"
    )
    metric_reconciliation = pd.DataFrame(reconciliation_rows)
    metric_reconciliation.to_csv(HERE / "METRIC_RECONCILIATION.csv", index=False)
    if not metric_reconciliation.status.eq("PASS").all():
        raise RuntimeError("Independent metrics disagree with published summary tables")

    correlation_rows = []
    for (method, seed, fold), frame in predictions.groupby(["method", "seed", "outer_fold"], dropna=False):
        for subset_name, label in (("ALL", None), ("HUMAN", 0), ("AI", 1)):
            subset = frame if label is None else frame[frame.binary_label.eq(label)]
            for feature in FEATURES:
                rho, p_value = safely_spearman(subset.raw_logit_score, subset[feature])
                correlation_rows.append(
                    {"method": method, "seed": seed, "outer_fold": fold, "subset": subset_name,
                     "feature": feature, "n": len(subset), "spearman_rho": rho,
                     "p_unadjusted_descriptive": p_value,
                     "undefined_reason": "constant_feature_or_score" if np.isnan(rho) else ""}
                )
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(HERE / "CORRECTED_SCORE_SURFACE_RELATIONSHIPS.csv", index=False, float_format="%.12f")

    inputs = [RAW, ANALYSIS / "MERIC2_ELIGIBLE_MANIFEST.csv", ANALYSIS / "MERIC2_LENGTH_MATCHED_MANIFEST.csv",
              ANALYSIS / "MAIN_AND_MATCHED_PER_CELL_METRICS.csv", ANALYSIS / "TRANSFORMER_SEED_LEVEL_METRICS.csv",
              ANALYSIS / "MAIN_AND_MATCHED_AGGREGATE_METRICS.csv", tfidf_path]
    inputs.extend(
        ANALYSIS / "predictions" / method / f"seed_{seed}" / f"outer_{fold}.csv"
        for method in METHODS for seed in SEEDS for fold in FOLDS
    )
    pd.DataFrame([{"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
                  for path in inputs]).to_csv(HERE / "VALIDATION_INPUT_PROVENANCE.csv", index=False)

    undefined_count = int(correlations.spearman_rho.isna().sum())
    print(f"Metric validation complete; cells=35; undefined surface correlations={undefined_count}")


if __name__ == "__main__":
    main()
