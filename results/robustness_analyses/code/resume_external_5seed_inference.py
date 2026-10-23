"""Audit and safely resume only missing five-seed external inference cells.

Completed artifacts are read and authenticated, never rewritten. Any partial or
invalid cell stops the resume before GPU inference begins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/robustness_analyses"
PRESEAL = ROOT / "artifacts/predictions/external_cross_corpus/preseal"
RUNNER = OUT / "code/external_5seed_extension.py"
CHECKPOINT_REPORT = OUT / "EXTERNAL_5SEED_POSTTRAIN_VERIFICATION.csv"
INFERENCE_CODE = ROOT / "src/turkish_ai_detection/experiments/external_cross_corpus/inference.py"
FREEZE = ROOT / "provenance/experiments/external_cross_corpus_v2/PREEXECUTION_FREEZE.json"
METHODS = ("CE_ONLY_MATCHED", "SUPCON")
SEEDS = (137, 3407)
FOLDS = tuple(range(5))
DATASETS = {
    "renklier2026": ROOT / "data/external_prepared/renklier2026_test_standardized.csv",
    "tum2024_tr": ROOT / "data/external_prepared/tum2024_tr_test_standardized.csv",
    "bostancioglu2026_tr": ROOT / "data/external_prepared/bostancioglu2026_tr_full_standardized.csv",
}
DATASET_HASHES = {
    "renklier2026": "1ec79d25ae852321cd5659691643eda58e407bcc17e169e6559413587ed8acb5",
    "tum2024_tr": "8ab15a4ace28674d20ce94faf4a63e1c5885254051ac2a795bd2033c961e557a",
    "bostancioglu2026_tr": "dfcde6dadf0c3f7bea719e4b54a61cea8ffb7c29a47adb0413de22cba39b578c",
}
EXPECTED_COLUMNS = ["external_id", "text_sha256", "method", "seed", "outer_fold", "raw_score_or_logit"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def paths(dataset: str, method: str, seed: int, fold: int) -> tuple[Path, Path]:
    base = PRESEAL / dataset / method / f"seed_{seed}" / f"outer_{fold}"
    return base.with_suffix(".csv"), base.with_suffix(".json")


def checkpoint_index() -> dict[tuple[str, int, int], dict]:
    if not CHECKPOINT_REPORT.is_file():
        raise RuntimeError("missing PASS_ALL_20 checkpoint report; run checkpoint verification first")
    frame = pd.read_csv(CHECKPOINT_REPORT)
    if len(frame) != 20 or set(frame.status) != {"PASS_AUTHENTIC_COMPLETE_READABLE"}:
        raise RuntimeError("checkpoint verification report is not a clean 20-cell PASS")
    return {(str(r.method), int(r.seed), int(r.outer_fold)): r._asdict() for r in frame.itertuples(index=False)}


def audit_cell(dataset: str, method: str, seed: int, fold: int, meta: pd.DataFrame, checkpoints: dict) -> dict:
    csv_path, json_path = paths(dataset, method, seed, fold)
    csv_tmp, json_tmp = csv_path.with_suffix(".csv.tmp"), json_path.with_suffix(".json.tmp")
    base = {"dataset": dataset, "method": method, "seed": seed, "outer_fold": fold,
            "prediction_csv": rel(csv_path), "sidecar_json": rel(json_path)}
    present = {"csv": csv_path.is_file(), "json": json_path.is_file(), "csv_tmp": csv_tmp.exists(), "json_tmp": json_tmp.exists()}
    base.update(present)
    if not present["csv"] and not present["json"] and not present["csv_tmp"] and not present["json_tmp"]:
        return {**base, "classification": "MISSING", "reason": "no final or temporary output exists"}
    if not present["csv"] or not present["json"] or present["csv_tmp"] or present["json_tmp"]:
        return {**base, "classification": "PARTIAL_OR_INVALID", "reason": "CSV/JSON pair incomplete or temporary artifact present"}
    try:
        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        checks = [
            (sidecar.get("schema_version") == "external-cross-corpus-preseal-inference-v1", "sidecar schema"),
            (sidecar.get("status") == "COMPLETE_LABEL_BLIND_PRESEAL_INFERENCE", "completion status"),
            (sidecar.get("dataset_id") == dataset, "dataset identity"),
            (sidecar.get("method") == method, "method identity"),
            (sidecar.get("seed") == seed, "seed identity"),
            (sidecar.get("outer_fold") == fold, "fold identity"),
            (sidecar.get("rows") == len(meta), "sidecar row count"),
            (sidecar.get("input_projection_exact") == ["external_id", "text", "text_sha256"], "input projection"),
            (sidecar.get("output_columns_exact") == EXPECTED_COLUMNS, "output schema"),
            (sidecar.get("external_labels_read") is False, "label blindness"),
            (sidecar.get("external_subgroups_read") is False, "subgroup blindness"),
            (sidecar.get("performance_metrics_computed") is False, "no preseal metrics"),
            (sidecar.get("external_adaptation") is False, "no external adaptation"),
            (sidecar.get("calibration_applied") is False, "no calibration"),
            (sidecar.get("external_source", {}).get("path") == rel(DATASETS[dataset]), "dataset path binding"),
            (sidecar.get("external_source", {}).get("sha256") == DATASET_HASHES[dataset] == sha(DATASETS[dataset]), "dataset hash binding"),
            (sidecar.get("inference_implementation", {}).get("path") == rel(INFERENCE_CODE), "inference path binding"),
            (sidecar.get("inference_implementation", {}).get("sha256") == sha(INFERENCE_CODE), "inference hash binding"),
            (sidecar.get("preexecution_freeze", {}).get("path") == rel(FREEZE), "freeze path binding"),
            (sidecar.get("preexecution_freeze", {}).get("sha256") == sha(FREEZE), "freeze hash binding"),
            (sidecar.get("prediction", {}).get("path") == rel(csv_path), "prediction path binding"),
            (sidecar.get("prediction", {}).get("sha256") == sha(csv_path), "prediction hash binding"),
        ]
        failed = [name for ok, name in checks if not ok]
        if failed:
            raise RuntimeError("failed sidecar checks: " + "; ".join(failed))

        checkpoint = sidecar.get("checkpoint", {})
        verified = checkpoints[(method, seed, fold)]
        checkpoint_checks = {
            "model_path": "model_path", "model_sha256": "model_sha256",
            "metadata_path": "config_path", "metadata_sha256": "config_sha256",
            "run_record_path": "run_record_path", "internal_prediction_path": "internal_prediction_path",
        }
        failed_checkpoint = [left for left, right in checkpoint_checks.items() if str(checkpoint.get(left)) != str(verified[right])]
        run_path = ROOT / checkpoint["run_record_path"]
        internal_path = ROOT / checkpoint["internal_prediction_path"]
        if checkpoint.get("run_record_sha256") != sha(run_path): failed_checkpoint.append("run_record_sha256")
        if checkpoint.get("internal_prediction_sha256") != sha(internal_path): failed_checkpoint.append("internal_prediction_sha256")
        if failed_checkpoint:
            raise RuntimeError("failed checkpoint bindings: " + "; ".join(failed_checkpoint))

        prediction = pd.read_csv(csv_path)
        if list(prediction.columns) != EXPECTED_COLUMNS: raise RuntimeError("prediction columns differ from sealed schema")
        if len(prediction) != len(meta): raise RuntimeError("prediction row count mismatch")
        if not prediction.external_id.equals(meta.external_id): raise RuntimeError("external_id/order mismatch")
        if not prediction.text_sha256.equals(meta.text_sha256): raise RuntimeError("text_sha256/order mismatch")
        if not prediction.method.eq(method).all(): raise RuntimeError("prediction method mismatch")
        if not prediction.seed.eq(seed).all(): raise RuntimeError("prediction seed mismatch")
        if not prediction.outer_fold.eq(fold).all(): raise RuntimeError("prediction fold mismatch")
        if not np.isfinite(prediction.raw_score_or_logit.to_numpy(float)).all(): raise RuntimeError("non-finite score")
        return {**base, "classification": "COMPLETE_VALID", "reason": "all prediction, sidecar, dataset, checkpoint, and hash bindings passed",
                "rows": len(prediction), "prediction_sha256": sha(csv_path), "sidecar_sha256": sha(json_path)}
    except Exception as exc:
        return {**base, "classification": "PARTIAL_OR_INVALID", "reason": str(exc)}


def audit() -> pd.DataFrame:
    checkpoints = checkpoint_index()
    metadata = {}
    for dataset, path in DATASETS.items():
        if sha(path) != DATASET_HASHES[dataset]: raise RuntimeError(f"dataset hash changed: {dataset}")
        metadata[dataset] = pd.read_csv(path, usecols=["external_id", "text_sha256"])
    rows = [audit_cell(ds, method, seed, fold, metadata[ds], checkpoints)
            for ds in DATASETS for method in METHODS for seed in SEEDS for fold in FOLDS]
    frame = pd.DataFrame(rows)
    if len(frame) != 60: raise RuntimeError("audit grid is not 60 cells")
    frame.to_csv(OUT / "EXTERNAL_5SEED_INFERENCE_RESUME_AUDIT.csv", index=False)
    counts = frame.classification.value_counts().to_dict()
    summary = {"status": "AUDIT_COMPLETE", "expected_cells": 60,
               "complete_valid": int(counts.get("COMPLETE_VALID", 0)),
               "missing": int(counts.get("MISSING", 0)),
               "partial_or_invalid": int(counts.get("PARTIAL_OR_INVALID", 0))}
    (OUT / "EXTERNAL_5SEED_INFERENCE_RESUME_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return frame


def resume() -> None:
    frame = audit()
    bad = frame[frame.classification.eq("PARTIAL_OR_INVALID")]
    if len(bad):
        raise RuntimeError(f"resume blocked: {len(bad)} partial/invalid cells; inspect EXTERNAL_5SEED_INFERENCE_RESUME_AUDIT.csv")
    missing = frame[frame.classification.eq("MISSING")]
    print(f"Skipping {sum(frame.classification.eq('COMPLETE_VALID'))} authenticated completed cells; running {len(missing)} missing cells.")
    for index, row in enumerate(missing.itertuples(index=False), 1):
        command = [sys.executable, str(RUNNER), "--infer", "--dataset-id", row.dataset,
                   "--method", row.method, "--seed", str(row.seed), "--outer-fold", str(row.outer_fold)]
        print(f"[{index:02d}/{len(missing):02d}] {row.dataset} {row.method} seed={row.seed} fold={row.outer_fold}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        refreshed = audit()
        current = refreshed[(refreshed.dataset.eq(row.dataset)) & (refreshed.method.eq(row.method)) &
                            (refreshed.seed.eq(row.seed)) & (refreshed.outer_fold.eq(row.outer_fold))]
        if len(current) != 1 or current.iloc[0].classification != "COMPLETE_VALID":
            raise RuntimeError(f"newly generated cell failed validation: {row.dataset}/{row.method}/{row.seed}/{row.outer_fold}")
    final = audit()
    if not final.classification.eq("COMPLETE_VALID").all():
        raise RuntimeError("resume ended without 60 COMPLETE_VALID cells")
    print("PASS_ALL_60_EXTERNAL_INFERENCE_CELLS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.audit == args.resume: parser.error("choose exactly one of --audit or --resume")
    if args.audit: audit()
    else: resume()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
