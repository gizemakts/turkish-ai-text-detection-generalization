"""Read-only post-hoc AP/pAUC audit for the three primary external corpora.

This program never trains a model. It validates and scores existing predictions,
then writes only to its own new output directory. Publication outputs are emitted
only if the prespecified ROC-AUC reconciliation succeeds at six decimals.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = (42, 137, 314, 2026, 3407)
FOLDS = tuple(range(5))
METHOD_DIRS = {
    "CE_ONLY_MATCHED": "CE_ONLY_MATCHED",
    "CE + 0.1 × SupCon": "SUPCON",
}
CORPORA = {
    "Renklier2026": ("renklier2026", ROOT / "data/external_prepared/renklier2026_test_standardized.csv"),
    "TUM2024_TR": ("tum2024_tr", ROOT / "data/external_prepared/tum2024_tr_test_standardized.csv"),
    "Bostancioglu2026_TR": ("bostancioglu2026_tr", ROOT / "data/external_prepared/bostancioglu2026_tr_full_standardized.csv"),
}
EXPECTED_AUC = {
    ("Renklier2026", "CE_ONLY_MATCHED"): 0.752517,
    ("Renklier2026", "CE + 0.1 × SupCon"): 0.790117,
    ("Renklier2026", "TF-IDF M4"): 0.828799,
    ("TUM2024_TR", "CE_ONLY_MATCHED"): 0.635333,
    ("TUM2024_TR", "CE + 0.1 × SupCon"): 0.636981,
    ("TUM2024_TR", "TF-IDF M4"): 0.686205,
    ("Bostancioglu2026_TR", "CE_ONLY_MATCHED"): 0.755326,
    ("Bostancioglu2026_TR", "CE + 0.1 × SupCon"): 0.846698,
    ("Bostancioglu2026_TR", "TF-IDF M4"): 0.909896,
}
M4_DIR = ROOT / "supplementary_analyses/results/tfidf_external"
M4_PREDICTIONS = M4_DIR / "tfidf_external_predictions.csv"
M4_FOLD_METRICS = M4_DIR / "tfidf_external_fold_metrics.csv"
M4_TUNING = M4_DIR / "tfidf_inner_tuning.csv"
M4_PROVENANCE = M4_DIR / "provenance.json"
M4_SCRIPT = ROOT / "supplementary_analyses/scripts/external_tfidf_baseline.py"
PREDICTION_ROOT = ROOT / "artifacts/predictions/external_cross_corpus/preseal"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def metric_record(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    require(np.isfinite(score).all(), "non-finite model score")
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "standardized_partial_roc_auc_max_fpr_003": float(
            roc_auc_score(y, score, max_fpr=0.03)
        ),
    }


def input_record(path: Path, role: str) -> dict[str, object]:
    require(path.is_file(), f"missing input: {path}")
    return {
        "role": role,
        "path": rel(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    require(not path.exists(), f"refusing to overwrite output: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    require(not path.exists(), f"refusing to overwrite output: {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def validate_external_dataset(display: str, path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path)
    required = {"external_id", "text_sha256", "label", "binary_label"}
    require(required <= set(frame.columns), f"{display}: external schema mismatch")
    require(frame["external_id"].notna().all(), f"{display}: null external_id")
    require(frame["external_id"].astype(str).is_unique, f"{display}: duplicate external_id")
    require(frame["text_sha256"].notna().all(), f"{display}: null text_sha256")
    require(frame["binary_label"].isin([0, 1]).all(), f"{display}: invalid binary label")
    expected_binary = frame["label"].astype(str).str.lower().eq("ai").astype(int)
    require(expected_binary.equals(frame["binary_label"].astype(int)), f"{display}: label mapping mismatch")
    counts = frame["binary_label"].astype(int).value_counts().to_dict()
    duplicate_hash_mask = frame["text_sha256"].duplicated(keep=False)
    audit = {
        "corpus": display,
        "path": rel(path),
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "human": int(counts.get(0, 0)),
        "ai": int(counts.get(1, 0)),
        "ai_prevalence": float(counts.get(1, 0) / len(frame)),
        "unique_external_id": int(frame["external_id"].nunique()),
        "unique_text_sha256": int(frame["text_sha256"].nunique()),
        "rows_in_repeated_text_sha256_groups": int(duplicate_hash_mask.sum()),
        "positive_class": "AI (binary_label=1)",
    }
    return frame, audit


def validate_transformer_prediction(
    corpus_key: str,
    external: pd.DataFrame,
    method: str,
    method_dir: str,
    seed: int,
    fold: int,
    inputs: list[dict[str, object]],
) -> tuple[np.ndarray, dict[str, object]]:
    base = PREDICTION_ROOT / corpus_key / method_dir / f"seed_{seed}"
    csv_path = base / f"outer_{fold}.csv"
    json_path = base / f"outer_{fold}.json"
    require(csv_path.is_file(), f"missing prediction: {rel(csv_path)}")
    require(json_path.is_file(), f"missing sidecar: {rel(json_path)}")
    pred = pd.read_csv(csv_path)
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    expected_columns = ["external_id", "text_sha256", "method", "seed", "outer_fold", "raw_score_or_logit"]
    require(list(pred.columns) == expected_columns, f"{rel(csv_path)}: unexpected columns")
    require(len(pred) == len(external), f"{rel(csv_path)}: row count mismatch")
    require(pred["external_id"].is_unique, f"{rel(csv_path)}: duplicate external_id")
    require(pred["external_id"].astype(str).tolist() == external["external_id"].astype(str).tolist(), f"{rel(csv_path)}: external_id/order mismatch")
    require(pred["text_sha256"].astype(str).tolist() == external["text_sha256"].astype(str).tolist(), f"{rel(csv_path)}: text hash mismatch")
    require(pred["method"].eq(method_dir).all(), f"{rel(csv_path)}: method mismatch")
    require(pred["seed"].eq(seed).all() and pred["outer_fold"].eq(fold).all(), f"{rel(csv_path)}: realization mismatch")
    require(sidecar.get("external_labels_read") is False, f"{rel(json_path)}: labels entered inference")
    require(sidecar.get("external_adaptation") is False, f"{rel(json_path)}: external adaptation reported")
    require(sidecar.get("score_contract", "").find("logit_1-logit_0") >= 0, f"{rel(json_path)}: score contract mismatch")
    require(sidecar.get("prediction", {}).get("sha256") == sha256(csv_path), f"{rel(json_path)}: prediction hash mismatch")
    inputs.extend([input_record(csv_path, "transformer_prediction"), input_record(json_path, "transformer_prediction_sidecar")])
    score = pred["raw_score_or_logit"].to_numpy(float)
    return score, {
        "prediction_path": rel(csv_path),
        "prediction_sha256": sha256(csv_path),
        "sidecar_path": rel(json_path),
        "sidecar_sha256": sha256(json_path),
        "score_direction": "higher=AI; logit_AI-logit_HUMAN",
        "external_fit_or_adaptation": False,
        "external_label_model_selection": False,
    }


def aggregate(realizations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["roc_auc", "average_precision", "standardized_partial_roc_auc_max_fpr_003"]
    neural = realizations[realizations.method.ne("TF-IDF M4")]
    seed_means = neural.groupby(["corpus", "method", "seed"], as_index=False)[metrics].mean()
    neural_summary = seed_means.groupby(["corpus", "method"], as_index=False)[metrics].mean()
    m4_summary = realizations[realizations.method.eq("TF-IDF M4")].groupby(["corpus", "method"], as_index=False)[metrics].mean()
    return seed_means, pd.concat([neural_summary, m4_summary], ignore_index=True)


def fmt(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    require(OUT.is_dir(), "output directory missing")
    managed = [
        "realization_metrics.csv", "seed_level_metrics.csv", "method_corpus_summary.csv",
        "dataset_validation.csv", "prediction_validation.csv", "roc_auc_reconciliation.csv",
        "Table_S23_Panel_B.csv",
        "Table_S23_Panel_A_M4_fold_metrics_preserved.csv",
        "Table_S23_Panel_A_M4_inner_selection_preserved.csv", "TABLE_S23_SOURCE.md",
        "INPUT_METHOD_MANIFEST.json",
        "VALIDATION_STATUS.json", "SHA256_MANIFEST.csv",
    ]
    require(not any((OUT / name).exists() for name in managed), "managed output already exists")

    input_manifest: list[dict[str, object]] = []
    datasets: dict[str, pd.DataFrame] = {}
    dataset_audits: list[dict[str, object]] = []
    for display, (key, path) in CORPORA.items():
        frame, audit = validate_external_dataset(display, path)
        datasets[display] = frame
        dataset_audits.append(audit)
        input_manifest.append(input_record(path, "external_evaluation_dataset"))

    # M4 provenance and score semantics are frozen in the original implementation.
    for path, role in [
        (M4_PREDICTIONS, "m4_external_predictions"),
        (M4_FOLD_METRICS, "m4_original_fold_metrics"),
        (M4_TUNING, "m4_original_inner_tuning"),
        (M4_PROVENANCE, "m4_original_provenance"),
        (M4_SCRIPT, "m4_original_implementation"),
    ]:
        input_manifest.append(input_record(path, role))
    m4_prov = json.loads(M4_PROVENANCE.read_text(encoding="utf-8"))
    require(m4_prov.get("external_fit_count") == 0, "M4 provenance reports external fitting")
    require(m4_prov.get("script_sha256") == sha256(M4_SCRIPT), "M4 script hash mismatch")
    bound = {Path(item["path"]).as_posix(): item["sha256"] for item in m4_prov["inputs"]}
    for _display, (_key, path) in CORPORA.items():
        require(bound.get(rel(path)) == sha256(path), f"M4 dataset binding mismatch: {rel(path)}")

    realization_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for display, (corpus_key, _path) in CORPORA.items():
        external = datasets[display]
        y = external["binary_label"].astype(int).to_numpy()
        for method, method_dir in METHOD_DIRS.items():
            for seed in SEEDS:
                for fold in FOLDS:
                    score, audit = validate_transformer_prediction(
                        corpus_key, external, method, method_dir, seed, fold, input_manifest
                    )
                    values = metric_record(y, score)
                    realization_rows.append({
                        "corpus": display, "method": method, "seed": seed,
                        "outer_fold": fold, "realization_structure": "seed_x_outer_fold",
                        "n": len(y), "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
                        "ai_prevalence": float(y.mean()), **values,
                    })
                    validation_rows.append({"corpus": display, "method": method, "seed": seed, "outer_fold": fold, "rows": len(y), "id_match": True, "text_sha256_match": True, "no_missing_score": True, "no_external_adaptation": True, "no_external_label_selection": True, **audit})

    m4 = pd.read_csv(M4_PREDICTIONS)
    require(set(m4.columns) == {"corpus", "outer_fold", "external_id", "raw_score"}, "M4 prediction schema mismatch")
    for display in CORPORA:
        external = datasets[display]
        y_lookup = external.set_index(external.external_id.astype(str))["binary_label"].astype(int)
        expected_ids = set(external.external_id.astype(str))
        for fold in FOLDS:
            pred = m4[(m4.corpus == display) & (m4.outer_fold == fold)].copy()
            ids = pred.external_id.astype(str)
            require(len(pred) == len(external), f"M4 {display} fold {fold}: row count mismatch")
            require(ids.is_unique, f"M4 {display} fold {fold}: duplicate external_id")
            require(set(ids) == expected_ids, f"M4 {display} fold {fold}: external_id mismatch")
            y = y_lookup.loc[ids].to_numpy(int)
            score = pred.raw_score.to_numpy(float)
            values = metric_record(y, score)
            realization_rows.append({
                "corpus": display, "method": "TF-IDF M4", "seed": "",
                "outer_fold": fold, "realization_structure": "dependent_outer_fold_estimator",
                "n": len(y), "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
                "ai_prevalence": float(y.mean()), **values,
            })
            validation_rows.append({
                "corpus": display, "method": "TF-IDF M4", "seed": "", "outer_fold": fold,
                "rows": len(y), "id_match": True, "text_sha256_match": "not_in_prediction; dataset_sha256_bound",
                "no_missing_score": True, "no_external_adaptation": True,
                "no_external_label_selection": True, "prediction_path": rel(M4_PREDICTIONS),
                "prediction_sha256": sha256(M4_PREDICTIONS), "sidecar_path": rel(M4_PROVENANCE),
                "sidecar_sha256": sha256(M4_PROVENANCE),
                "score_direction": "higher=AI; LogisticRegression decision_function for AI=1",
                "external_fit_or_adaptation": False, "external_label_model_selection": False,
            })

    realization = pd.DataFrame(realization_rows)
    require(len(realization) == 165, f"unexpected realization-metric rows: {len(realization)}")

    # The saved M4 predictions must reproduce the original Task-2 fold table,
    # including its positive-class and standardized-pAUC convention.
    original_m4 = pd.read_csv(M4_FOLD_METRICS).sort_values(["corpus", "outer_fold"]).reset_index(drop=True)
    recomputed_m4 = (
        realization[realization.method.eq("TF-IDF M4")]
        .sort_values(["corpus", "outer_fold"])
        .reset_index(drop=True)
    )
    require(
        original_m4[["corpus", "outer_fold"]].equals(recomputed_m4[["corpus", "outer_fold"]]),
        "M4 fold identity differs from the original Task-2 fold table",
    )
    for column in ("roc_auc", "average_precision", "standardized_partial_roc_auc_max_fpr_003"):
        require(
            np.allclose(original_m4[column].to_numpy(float), recomputed_m4[column].to_numpy(float), rtol=0.0, atol=5e-12),
            f"M4 recomputed {column} differs from the original Task-2 fold table",
        )
    seed_means, summary = aggregate(realization)

    reconciliation_rows: list[dict[str, object]] = []
    reconciliation_ok = True
    for (corpus, method), expected in EXPECTED_AUC.items():
        observed = float(summary[(summary.corpus == corpus) & (summary.method == method)].iloc[0].roc_auc)
        passed = round(observed, 6) == round(expected, 6)
        reconciliation_ok &= passed
        reconciliation_rows.append({
            "corpus": corpus, "method": method, "expected_roc_auc_6dp": f"{expected:.6f}",
            "observed_roc_auc": f"{observed:.12f}", "observed_roc_auc_6dp": f"{observed:.6f}",
            "difference": f"{observed - expected:.12f}", "status": "PASS" if passed else "FAIL",
        })

    realization.to_csv(OUT / "realization_metrics.csv", index=False, float_format="%.12f")
    seed_means.to_csv(OUT / "seed_level_metrics.csv", index=False, float_format="%.12f")
    summary.to_csv(OUT / "method_corpus_summary.csv", index=False, float_format="%.12f")
    write_csv(OUT / "dataset_validation.csv", dataset_audits, list(dataset_audits[0]))
    write_csv(OUT / "prediction_validation.csv", validation_rows, list(validation_rows[0]))
    write_csv(OUT / "roc_auc_reconciliation.csv", reconciliation_rows, list(reconciliation_rows[0]))

    if not reconciliation_ok:
        status = {"status": "FAIL_ROC_AUC_RECONCILIATION_PUBLICATION_OUTPUTS_WITHHELD", "created_utc": datetime.now(timezone.utc).isoformat(), "realization_rows": len(realization)}
        write_text(OUT / "VALIDATION_STATUS.json", json.dumps(status, indent=2))
        raise RuntimeError("ROC-AUC reconciliation failed; summary table not generated")

    # Panel B, one row per corpus, after successful reconciliation only.
    panel_rows: list[dict[str, object]] = []
    for display in CORPORA:
        audit = next(row for row in dataset_audits if row["corpus"] == display)
        values = {row.method: row for _, row in summary[summary.corpus == display].iterrows()}
        panel_rows.append({
            "Corpus": display,
            "AI prevalence": f"{float(audit['ai_prevalence']):.6f}",
            "CE AP": fmt(values["CE_ONLY_MATCHED"].average_precision),
            "SupCon AP": fmt(values["CE + 0.1 × SupCon"].average_precision),
            "M4 AP": fmt(values["TF-IDF M4"].average_precision),
            "CE standardized pAUC": fmt(values["CE_ONLY_MATCHED"].standardized_partial_roc_auc_max_fpr_003),
            "SupCon standardized pAUC": fmt(values["CE + 0.1 × SupCon"].standardized_partial_roc_auc_max_fpr_003),
            "M4 standardized pAUC": fmt(values["TF-IDF M4"].standardized_partial_roc_auc_max_fpr_003),
        })
    write_csv(OUT / "Table_S23_Panel_B.csv", panel_rows, list(panel_rows[0]))

    # Preserve the two existing M4 Panel-A evidence tables byte-for-byte.
    shutil.copyfile(M4_FOLD_METRICS, OUT / "Table_S23_Panel_A_M4_fold_metrics_preserved.csv")
    shutil.copyfile(M4_TUNING, OUT / "Table_S23_Panel_A_M4_inner_selection_preserved.csv")
    require(sha256(M4_FOLD_METRICS) == sha256(OUT / "Table_S23_Panel_A_M4_fold_metrics_preserved.csv"), "Panel A fold copy changed")
    require(sha256(M4_TUNING) == sha256(OUT / "Table_S23_Panel_A_M4_inner_selection_preserved.csv"), "Panel A tuning copy changed")

    table_lines = [
        "# Supplementary Table S23 - source data", "",
        "Panel A presents the TF-IDF M4 external fold measurements and inner cross-validation selections in `Table_S23_Panel_A_M4_fold_metrics_preserved.csv` and `Table_S23_Panel_A_M4_inner_selection_preserved.csv`. In each outer fold, M4 selected `C=10` using only outer-training observations and four-fold inner cross-validation for standardized pAUC at FPR <= 0.03.", "",
        "## Panel B. Exploratory external AP and low-FPR discrimination", "",
        "| Corpus | AI prevalence | CE AP | SupCon AP | M4 AP | CE standardized pAUC | SupCon standardized pAUC | M4 standardized pAUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in panel_rows:
        table_lines.append("| " + " | ".join(str(row[col]) for col in panel_rows[0]) + " |")
    table_lines += [
        "", "**Caption.** Supplementary Table S23. Panel A reports fold-specific external results and inner model selection for the TF-IDF M4 baseline. Panel B compares post-hoc exploratory average precision and standardized partial ROC-AUC (FPR <= 0.03) for CE_ONLY_MATCHED, CE + 0.1 × SupCon, and TF-IDF M4 on the same three external evaluation populations.", "",
        "**Note.** AI was the positive class. Metrics were calculated separately for each realization without pooling scores. CE and SupCon values summarize 25 realizations per corpus by first averaging five outer folds within each seed and then averaging the five seed means. M4 values average five dependent outer-fold estimators; those estimators must not be interpreted as five independent seeds. Standardized pAUC was computed with `roc_auc_score(y_true, y_score, max_fpr=0.03)`, for which random ranking equals 0.5. AP must be interpreted relative to corpus-specific AI prevalence. Evaluation on a common population supports descriptive comparison but does not make the different training-realization structures inferentially equivalent. No confidence intervals or significance tests were computed.",
    ]
    write_text(OUT / "TABLE_S23_SOURCE.md", "\n".join(table_lines))

    manifest = {
        "schema_version": "external-ap-pauc-s23-posthoc-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "PASS_ROC_AUC_RECONCILIATION",
        "analysis_type": "post-hoc exploratory; saved predictions only; no training",
        "positive_class": "AI (binary_label=1)",
        "score_contracts": {
            "CE_ONLY_MATCHED": "higher=AI; logit_AI-logit_HUMAN; fixed terminal checkpoint inference",
            "CE + 0.1 × SupCon": "higher=AI; logit_AI-logit_HUMAN; fixed terminal checkpoint inference",
            "TF-IDF M4": "higher=AI; LogisticRegression decision_function with AI=1; external_fit_count=0",
        },
        "metrics": {
            "average_precision": "sklearn.metrics.average_precision_score(y_true, y_score)",
            "standardized_partial_roc_auc": "sklearn.metrics.roc_auc_score(y_true, y_score, max_fpr=0.03)",
            "pauc_random_reference": 0.5,
        },
        "aggregation": {
            "CE_and_SupCon": "mean 5 folds within each seed, then mean 5 seed means",
            "TF-IDF_M4": "mean 5 dependent outer-fold estimator metrics; not treated as independent seeds",
            "score_pooling": False,
        },
        "m4_definition_reconciliation": "PASS: recomputed fold-level ROC-AUC, AP, and standardized pAUC match the original Task-2 fold table within 5e-12 absolute tolerance",
        "input_files": input_manifest,
    }
    write_text(OUT / "INPUT_METHOD_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    status = {
        "status": "PASS",
        "roc_auc_reconciliation": "9/9 PASS at six decimals",
        "realization_metric_rows": 165,
        "transformer_prediction_cells": 150,
        "m4_prediction_cells": 15,
        "table_s23_source": "TABLE_S23_SOURCE.md",
        "model_training_performed": False,
    }
    write_text(OUT / "VALIDATION_STATUS.json", json.dumps(status, indent=2))

    # Hash every output except this self-referential manifest.
    hash_rows = []
    for path in sorted(OUT.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            hash_rows.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(OUT / "SHA256_MANIFEST.csv", hash_rows, ["path", "bytes", "sha256"])
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
