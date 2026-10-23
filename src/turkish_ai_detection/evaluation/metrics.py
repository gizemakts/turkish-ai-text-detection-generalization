# -*- coding: utf-8 -*-
"""Leakage-safe Combined v2 evaluation for unchanged M1--M4 definitions.

M3/M4 can run directly from the combined dataset.  M1/M2 require a Combined
v2 raw-feature cache produced by the unchanged feature extractors; this script
fails explicitly when that prerequisite is absent.  M5 has a separate GPU
runner because it is compute-heavy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from turkish_ai_detection.data.study import np_umbrella_k, sha256_file, write_json  # noqa: E402
from turkish_ai_detection.features.core import (  # noqa: E402
    PRIMARY_FEATURES,
    M2_FEATURES,
    SURFACE_FEATURES,
    TurkishAIDetectionFeatureTransformer,
    artifact_neutral_text,
    clopper_pearson_interval,
    ece_equal_width,
    normalize_text,
    surface_features,
    surface_model_matrix,
    turkish_lower,
)

C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["label"].astype(str).str.upper().eq("AI").astype(int).to_numpy()


def fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    return model.fit(np.asarray(scores).reshape(-1, 1), y)


def calibrated_score(calibrator: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


def threshold_from_training_oof(probabilities: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    human = np.sort(np.asarray(probabilities)[np.asarray(y) == 0])[::-1]
    k = np_umbrella_k(len(human), alpha=0.03, delta=0.05)
    return float(human[k]), int(k)


def make_logistic(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=float(C), penalty="l2", solver="liblinear", max_iter=5000, random_state=42)),
    ])


def make_tfidf(C: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True, norm="l2")),
        ("char", TfidfVectorizer(lowercase=False, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, norm="l2")),
    ])
    return Pipeline([
        ("tfidf", features),
        ("lr", LogisticRegression(C=float(C), solver="liblinear", max_iter=5000, random_state=42)),
    ])


def make_tfidf_features() -> FeatureUnion:
    """Unchanged M4 feature union, separated so deterministic features can be cached across C."""
    return FeatureUnion([
        ("word", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True, norm="l2")),
        ("char", TfidfVectorizer(lowercase=False, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, norm="l2")),
    ])


def make_sparse_logistic(C: float) -> LogisticRegression:
    return LogisticRegression(C=float(C), solver="liblinear", max_iter=5000, random_state=42)


def _pauc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score, max_fpr=0.03))


def _best_c(rows: list[dict]) -> float:
    return float(sorted(rows, key=lambda row: (-row["std_pauc_003"], row["C"]))[0]["C"])


def condition_text(data: pd.DataFrame, condition: str) -> pd.Series:
    if condition == "A":
        return data["text"].map(normalize_text)
    if condition == "B":
        return data["text"].map(artifact_neutral_text)
    raise ValueError("condition must be A or B")


def surface_table(data: pd.DataFrame, condition: str) -> pd.DataFrame:
    values = pd.DataFrame([surface_features(text) for text in condition_text(data, condition)], index=data.index)
    return pd.concat([data.reset_index(drop=True), values.reset_index(drop=True)], axis=1)


def _splits(folds: pd.DataFrame, outer: int):
    train_pairs = set(folds.loc[folds["outer_fold"].ne(outer), "pair_id"])
    test_pairs = set(folds.loc[folds["outer_fold"].eq(outer), "pair_id"])
    if train_pairs & test_pairs:
        raise RuntimeError("outer-fold pair leakage")
    inner_col = f"inner_fold_outer{outer}"
    return train_pairs, test_pairs, inner_col


def evaluate_m3(data: pd.DataFrame, folds: pd.DataFrame, condition: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    raw = surface_table(data, condition)
    predictions, fold_metrics, tuning, fit_audit = [], [], [], []
    for outer in range(5):
        train_pairs, test_pairs, inner_col = _splits(folds, outer)
        train = raw[raw["pair_id"].isin(train_pairs)].copy()
        test = raw[raw["pair_id"].isin(test_pairs)].copy()
        y_train, y_test = labels(train), labels(test)
        x_train = surface_model_matrix(train)
        candidate_rows = []
        for C in C_GRID:
            inner_oof = np.full(len(train), np.nan)
            for inner in range(4):
                validation_pairs = set(folds.loc[folds["outer_fold"].ne(outer) & folds[inner_col].eq(inner), "pair_id"])
                validation = train["pair_id"].isin(validation_pairs).to_numpy()
                model = make_logistic(C).fit(x_train.iloc[np.flatnonzero(~validation)], y_train[~validation])
                inner_oof[validation] = model.decision_function(x_train.iloc[np.flatnonzero(validation)])
            row = {"model": "M3", "condition": condition, "outer_fold": outer, "C": C, "std_pauc_003": _pauc(y_train, inner_oof)}
            candidate_rows.append(row); tuning.append(row)
        best = _best_c(candidate_rows)
        inner_decision = np.full(len(train), np.nan)
        for inner in range(4):
            validation_pairs = set(folds.loc[folds["outer_fold"].ne(outer) & folds[inner_col].eq(inner), "pair_id"])
            validation = train["pair_id"].isin(validation_pairs).to_numpy()
            fit_pairs = set(train.loc[~validation, "pair_id"])
            if fit_pairs & validation_pairs or validation_pairs & test_pairs or fit_pairs & test_pairs:
                raise RuntimeError("M3 inner/outer leakage")
            model = make_logistic(best).fit(x_train.iloc[np.flatnonzero(~validation)], y_train[~validation])
            inner_decision[validation] = model.decision_function(x_train.iloc[np.flatnonzero(validation)])
            fit_audit.append({"model": "M3", "condition": condition, "outer_fold": outer, "inner_fold": inner, "role": "inner_model", "fit_pair_count": len(fit_pairs), "validation_pair_count": len(validation_pairs), "outer_evaluation_overlap": 0})
        calibrator = fit_platt(inner_decision, y_train)
        inner_probability = calibrated_score(calibrator, inner_decision)
        threshold, k = threshold_from_training_oof(inner_probability, y_train)
        final_model = make_logistic(best).fit(x_train, y_train)
        decision = final_model.decision_function(surface_model_matrix(test))
        probability = calibrated_score(calibrator, decision)
        prediction = (probability > threshold).astype(int)
        predictions.extend(_prediction_rows(test, outer, "M3", condition, decision, probability, prediction, threshold))
        fold_metrics.append(_fold_metric_row(y_test, probability, prediction, outer, "M3", condition, best, threshold, k))
        fit_audit.append({"model": "M3", "condition": condition, "outer_fold": outer, "inner_fold": None, "role": "calibration_and_threshold", "fit_pair_count": len(train_pairs), "validation_pair_count": len(train_pairs), "outer_evaluation_overlap": 0, "k_exceed": k})
    return pd.DataFrame(predictions), pd.DataFrame(fold_metrics), pd.DataFrame(tuning), fit_audit


def evaluate_m4(data: pd.DataFrame, folds: pd.DataFrame, condition: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    raw = data.copy(); raw["model_text"] = condition_text(raw, condition).map(turkish_lower)
    predictions, fold_metrics, tuning, fit_audit = [], [], [], []
    for outer in range(5):
        train_pairs, test_pairs, inner_col = _splits(folds, outer)
        train = raw[raw["pair_id"].isin(train_pairs)].copy()
        test = raw[raw["pair_id"].isin(test_pairs)].copy()
        y_train, y_test = labels(train), labels(test)
        split_cache = []
        for inner in range(4):
            validation_pairs = set(folds.loc[folds["outer_fold"].ne(outer) & folds[inner_col].eq(inner), "pair_id"])
            validation = train["pair_id"].isin(validation_pairs).to_numpy()
            vectorizer = make_tfidf_features()
            x_fit = vectorizer.fit_transform(train.loc[~validation, "model_text"])
            x_validation = vectorizer.transform(train.loc[validation, "model_text"])
            split_cache.append((inner, validation_pairs, validation, x_fit, x_validation))
        candidate_rows = []
        for C in C_GRID:
            inner_oof = np.full(len(train), np.nan)
            for inner, validation_pairs, validation, x_fit, x_validation in split_cache:
                model = make_sparse_logistic(C).fit(x_fit, y_train[~validation])
                inner_oof[validation] = model.decision_function(x_validation)
            row = {"model": "M4", "condition": condition, "outer_fold": outer, "C": C, "std_pauc_003": _pauc(y_train, inner_oof)}
            candidate_rows.append(row); tuning.append(row)
        best = _best_c(candidate_rows)
        inner_decision = np.full(len(train), np.nan)
        for inner, validation_pairs, validation, x_fit, x_validation in split_cache:
            fit_pairs = set(train.loc[~validation, "pair_id"])
            if fit_pairs & validation_pairs or validation_pairs & test_pairs or fit_pairs & test_pairs:
                raise RuntimeError("M4 inner/outer leakage")
            model = make_sparse_logistic(best).fit(x_fit, y_train[~validation])
            inner_decision[validation] = model.decision_function(x_validation)
            fit_audit.append({"model": "M4", "condition": condition, "outer_fold": outer, "inner_fold": inner, "role": "inner_model_and_tfidf_fit", "fit_pair_count": len(fit_pairs), "validation_pair_count": len(validation_pairs), "outer_evaluation_overlap": 0})
        calibrator = fit_platt(inner_decision, y_train)
        inner_probability = calibrated_score(calibrator, inner_decision)
        threshold, k = threshold_from_training_oof(inner_probability, y_train)
        final_vectorizer = make_tfidf_features()
        x_train = final_vectorizer.fit_transform(train["model_text"])
        x_test = final_vectorizer.transform(test["model_text"])
        final_model = make_sparse_logistic(best).fit(x_train, y_train)
        decision = final_model.decision_function(x_test)
        probability = calibrated_score(calibrator, decision)
        prediction = (probability > threshold).astype(int)
        predictions.extend(_prediction_rows(test, outer, "M4", condition, decision, probability, prediction, threshold))
        fold_metrics.append(_fold_metric_row(y_test, probability, prediction, outer, "M4", condition, best, threshold, k))
        fit_audit.append({"model": "M4", "condition": condition, "outer_fold": outer, "inner_fold": None, "role": "outer_model_tfidf_calibration_threshold", "fit_pair_count": len(train_pairs), "validation_pair_count": len(train_pairs), "outer_evaluation_overlap": 0, "k_exceed": k})
        print(f"M4 condition={condition} completed outer_fold={outer}", flush=True)
    return pd.DataFrame(predictions), pd.DataFrame(fold_metrics), pd.DataFrame(tuning), fit_audit


def evaluate_m1_m2(data: pd.DataFrame, folds: pd.DataFrame, condition: str, model_name: str, raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    if not raw_path.exists():
        raise FileNotFoundError(f"{model_name} requires unchanged raw features at {raw_path}; no fallback or feature substitution is allowed")
    raw_features = pd.read_csv(raw_path)
    required_raw = {
        "text_id", "hapax_ratio", "sentence_length_cv", "stopword_density",
        "conjunction_density", "bigram_entropy", "lexical_token_count",
        "morph_ttr", "mes_analyzed_token_count", "suffix_tag_entropy",
        "vowel_back_ratio", "vowel_front_ratio", "vowel_mixed_ratio",
        "fiilimsi_diversity_ratio", "fiilimsi_expected_mass",
        "fiilimsi_expected_density", "fiilimsi_pattern_entropy",
        "log_pseudo_perplexity", "wordpiece_logprob_variance",
    }
    missing_columns = required_raw - set(raw_features.columns)
    if missing_columns:
        raise ValueError(f"Combined raw-feature cache missing columns: {sorted(missing_columns)}")
    metadata = data[["text_id", "pair_id", "label", "domain", "generator"]]
    raw = metadata.merge(raw_features, on="text_id", how="left", validate="one_to_one")
    if len(raw) != len(data) or raw[list(required_raw - {"text_id"})].isna().any().any():
        raise ValueError("Combined raw-feature join is incomplete")
    features = PRIMARY_FEATURES if model_name == "M1" else M2_FEATURES
    predictions, fold_metrics, tuning, fit_audit = [], [], [], []
    for outer in range(5):
        train_pairs, test_pairs, inner_col = _splits(folds, outer)
        train = raw[raw["pair_id"].isin(train_pairs)].copy(); test = raw[raw["pair_id"].isin(test_pairs)].copy()
        # Historical transformer expects lower-case labels.
        train["label"] = train["label"].str.lower(); test["label"] = test["label"].str.lower()
        y_train, y_test = labels(train), labels(test)
        candidates = []
        for C in C_GRID:
            oof = np.full(len(train), np.nan)
            for inner in range(4):
                validation_pairs = set(folds.loc[folds["outer_fold"].ne(outer) & folds[inner_col].eq(inner), "pair_id"])
                validation = train["pair_id"].isin(validation_pairs).to_numpy()
                transform = TurkishAIDetectionFeatureTransformer().fit(train.loc[~validation])
                model = make_logistic(C).fit(transform.transform(train.loc[~validation])[features], y_train[~validation])
                oof[validation] = model.decision_function(transform.transform(train.loc[validation])[features])
            row = {"model": model_name, "condition": condition, "outer_fold": outer, "C": C, "std_pauc_003": _pauc(y_train, oof)}
            candidates.append(row); tuning.append(row)
        best = _best_c(candidates); inner_decision = np.full(len(train), np.nan)
        for inner in range(4):
            validation_pairs = set(folds.loc[folds["outer_fold"].ne(outer) & folds[inner_col].eq(inner), "pair_id"])
            validation = train["pair_id"].isin(validation_pairs).to_numpy()
            transform = TurkishAIDetectionFeatureTransformer().fit(train.loc[~validation])
            model = make_logistic(best).fit(transform.transform(train.loc[~validation])[features], y_train[~validation])
            inner_decision[validation] = model.decision_function(transform.transform(train.loc[validation])[features])
            fit_audit.append({"model": model_name, "condition": condition, "outer_fold": outer, "inner_fold": inner, "role": "inner_model_and_reference_fit", "outer_evaluation_overlap": 0})
        calibrator = fit_platt(inner_decision, y_train); inner_probability = calibrated_score(calibrator, inner_decision)
        threshold, k = threshold_from_training_oof(inner_probability, y_train)
        transform = TurkishAIDetectionFeatureTransformer().fit(train)
        model = make_logistic(best).fit(transform.transform(train)[features], y_train)
        decision = model.decision_function(transform.transform(test)[features]); probability = calibrated_score(calibrator, decision); prediction = (probability > threshold).astype(int)
        predictions.extend(_prediction_rows(test.assign(label=test["label"].str.upper()), outer, model_name, condition, decision, probability, prediction, threshold))
        fold_metrics.append(_fold_metric_row(y_test, probability, prediction, outer, model_name, condition, best, threshold, k))
        fit_audit.append({"model": model_name, "condition": condition, "outer_fold": outer, "inner_fold": None, "role": "outer_reference_model_calibration_threshold", "fit_pair_count": len(train_pairs), "validation_pair_count": len(train_pairs), "outer_evaluation_overlap": 0, "k_exceed": k})
        print(f"{model_name} condition={condition} completed outer_fold={outer}", flush=True)
    return pd.DataFrame(predictions), pd.DataFrame(fold_metrics), pd.DataFrame(tuning), fit_audit


def _prediction_rows(test, outer, model, condition, decision, probability, prediction, threshold):
    rows = []
    for position, row in enumerate(test.itertuples(index=False)):
        rows.append({"text_id": row.text_id, "pair_id": row.pair_id, "label": row.label, "domain": row.domain, "generator": row.generator, "outer_fold": outer, "model": model, "condition": condition, "raw_decision_score": float(decision[position]), "calibrated_study_ai_score": float(probability[position]), "binary_prediction": int(prediction[position]), "binary_decision_threshold": float(threshold)})
    return rows


def _fold_metric_row(y, probability, prediction, outer, model, condition, best_c, threshold, k):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"model": model, "condition": condition, "outer_fold": outer, "roc_auc": float(roc_auc_score(y, probability)), "std_pauc_003": _pauc(y, probability), "fpr": float(fp / (fp + tn)), "tpr": float(tp / (tp + fn)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "brier": float(brier_score_loss(y, probability)), "log_loss": float(log_loss(y, probability, labels=[0, 1])), "ece_10": ece_equal_width(y, probability, 10), "best_C": best_c, "binary_decision_threshold": threshold, "k_exceed": k}


def metric_row(group: pd.DataFrame, scope: str, scope_value: str) -> dict:
    y = group["label"].astype(str).str.upper().eq("AI").astype(int).to_numpy()
    probability = group["calibrated_study_ai_score"].to_numpy(float)
    prediction = group["binary_prediction"].to_numpy(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    fpr_ci = clopper_pearson_interval(fp, fp + tn)
    tpr_ci = clopper_pearson_interval(tp, tp + fn)
    return {"model": group["model"].iloc[0], "condition": group["condition"].iloc[0], "scope": scope, "scope_value": scope_value, "n": int(len(group)), "roc_auc": float(roc_auc_score(y, probability)), "std_pauc_003": _pauc(y, probability), "fpr": float(fp / (fp + tn)), "fpr_ci95_lower": fpr_ci[0], "fpr_ci95_upper": fpr_ci[1], "tpr": float(tp / (tp + fn)), "tpr_ci95_lower": tpr_ci[0], "tpr_ci95_upper": tpr_ci[1], "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "brier": float(brier_score_loss(y, probability)), "log_loss": float(log_loss(y, probability, labels=[0, 1])), "ece_10": ece_equal_width(y, probability, 10)}


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [metric_row(predictions, "ALL_DATA", "ALL")]
    for domain, group in predictions.groupby("domain", sort=True):
        rows.append(metric_row(group, "DOMAIN", str(domain)))
    for generator, group in predictions.groupby("generator", sort=True):
        rows.append(metric_row(group, "GENERATOR", str(generator)))
    return pd.DataFrame(rows)


def update_baseline_table(output_dir: Path) -> None:
    metric_files = sorted(output_dir.glob("M[1-5]_*_pooled_metrics.csv"))
    rows = []
    for path in metric_files:
        metrics = pd.read_csv(path)
        all_row = metrics[(metrics.scope == "ALL_DATA")].iloc[0]
        abstract = metrics[(metrics.scope == "DOMAIN") & (metrics.scope_value == "ABSTRACT")].iloc[0]
        newspaper = metrics[(metrics.scope == "DOMAIN") & (metrics.scope_value == "NEWSPAPER")].iloc[0]
        rows.append({"Model": all_row.model, "Condition": all_row.condition, "Overall AUC": all_row.roc_auc, "Abstract AUC": abstract.roc_auc, "Newspaper AUC": newspaper.roc_auc, "FPR": all_row.fpr, "TPR": all_row.tpr, "pAUC@0.03": all_row.std_pauc_003})
    status_path = output_dir / "model_status.json"
    statuses = json.loads(status_path.read_text(encoding="utf-8"))["models"] if status_path.exists() else {}
    present = {(row["Model"], row["Condition"]) for row in rows}
    for model in ("M1", "M2", "M3", "M4", "M5"):
        if not any(key[0] == model for key in present):
            rows.append({"Model": model, "Condition": "A", "Overall AUC": np.nan, "Abstract AUC": np.nan, "Newspaper AUC": np.nan, "FPR": np.nan, "TPR": np.nan, "pAUC@0.03": np.nan, "Status": statuses.get(model, {}).get("status", "NOT_RUN")})
    table = pd.DataFrame(rows).sort_values(["Model", "Condition"], kind="stable")
    table.to_csv(output_dir / "baseline_comparison.csv", index=False)
    display = table.fillna("").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in display.to_numpy().tolist()]
    (output_dir / "baseline_comparison.md").write_text("\n".join([header, separator] + body) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["M1", "M2", "M3", "M4"], required=True)
    parser.add_argument("--condition", choices=["A", "B"], default="A")
    parser.add_argument("--data", default="data/study/paired_study.csv")
    parser.add_argument("--folds", default="data/study/nested_folds.csv")
    parser.add_argument("--raw-features", default=None)
    args = parser.parse_args()
    data_path = ROOT / args.data; folds_path = ROOT / args.folds
    data = pd.read_csv(data_path); folds = pd.read_csv(folds_path)
    if args.model == "M3":
        prediction, fold_metric, tuning, fit_audit = evaluate_m3(data, folds, args.condition)
    elif args.model == "M4":
        prediction, fold_metric, tuning, fit_audit = evaluate_m4(data, folds, args.condition)
    else:
        default_raw = ROOT / "data" / "study" / f"raw_features_{args.condition}.csv"
        prediction, fold_metric, tuning, fit_audit = evaluate_m1_m2(data, folds, args.condition, args.model, Path(args.raw_features) if args.raw_features else default_raw)
    output_dir = ROOT / "results" / "model_comparison" / "baseline"; output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.condition}"
    prediction.to_csv(output_dir / f"{stem}_outer_oof.csv", index=False)
    fold_metric.to_csv(output_dir / f"{stem}_fold_metrics.csv", index=False)
    tuning.to_csv(output_dir / f"{stem}_inner_tuning.csv", index=False)
    pooled = summarize(prediction); pooled.to_csv(output_dir / f"{stem}_pooled_metrics.csv", index=False)
    write_json(output_dir / f"{stem}_fit_isolation_audit.json", {"status": "PASS", "outer_evaluation_data_used_for_fitting": False, "records": fit_audit})
    provenance = {"schema_version": "combined-v2-model-run-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "model": args.model, "condition": args.condition, "dataset_sha256": sha256_file(data_path), "fold_sha256": sha256_file(folds_path), "outputs": {name: sha256_file(output_dir / name) for name in [f"{stem}_outer_oof.csv", f"{stem}_fold_metrics.csv", f"{stem}_inner_tuning.csv", f"{stem}_pooled_metrics.csv", f"{stem}_fit_isolation_audit.json"]}, "score_terminology": {"raw_decision_score": "uncalibrated model decision function", "calibrated_study_ai_score": "Platt-calibrated score for this study distribution; not a real-world posterior probability", "binary_decision_threshold": "training-side inner-OOF NP order-statistic threshold"}}
    write_json(ROOT / "provenance" / "experiments" / "model_comparison" / f"{stem}_run.json", provenance)
    update_baseline_table(output_dir)
    print(pooled.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
