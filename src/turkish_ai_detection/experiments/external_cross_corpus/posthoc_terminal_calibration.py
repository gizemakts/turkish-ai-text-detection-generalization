"""Post-hoc supplementary calibration and threshold-transfer analysis.

The ``prepare`` phase is label-blind: it reads only the existing internal
inner-OOF artifacts and the sealed prediction manifest, then atomically freezes
the 30 mappings and thresholds.  The ``evaluate`` phase is the only phase that
joins external labels.  Neither phase runs inference or changes primary files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from turkish_ai_detection.data.study import np_umbrella_k
from turkish_ai_detection.features.core import ece_equal_width

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".gitignore").is_file())
CONFIG = ROOT / "configs" / "posthoc_terminal_calibration.json"
OUT = ROOT / "results" / "posthoc_terminal_calibration"
SEAL = ROOT / "provenance/experiments/external_cross_corpus_v2/prediction_seal/PRESEAL_PREDICTION_MANIFEST.json"
INNER = ROOT / "results/calibration/inner_oof"
PRIMARY = ROOT / "results/primary"
EXTERNAL = ROOT / "data/external_prepared"
METHODS = {"CE_ONLY_MATCHED": "ce", "SUPCON": "supcon"}
SEEDS = (42, 314, 2026)
FOLDS = tuple(range(5))
INNERS = tuple(range(4))
ALPHAS = (0.01, 0.02, 0.03, 0.05)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["label"].astype(str).str.upper().eq("AI").astype(int).to_numpy()


def fit_platt(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    require(np.isfinite(scores).all(), "non-finite internal calibration scores")
    require(set(np.unique(y)) == {0, 1}, "internal calibration lacks a class")
    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000).fit(
        scores.reshape(-1, 1), y
    )


def probabilities(coefficient: float, intercept: float, scores: np.ndarray) -> np.ndarray:
    z = coefficient * np.asarray(scores, dtype=float) + intercept
    require(np.isfinite(z).all(), "non-finite calibration linear predictor")
    z = np.clip(z, -745.0, 745.0)
    p = 1.0 / (1.0 + np.exp(-z))
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "invalid calibrated probability")
    return p


def fit_one(method: str, seed: int, outer: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    objective = METHODS[method]
    parts = []
    source_bindings = []
    for inner in INNERS:
        pred = INNER / objective / f"seed_{seed}" / f"outer_{outer}_inner_{inner}_predictions.csv"
        side = pred.with_name(pred.stem.replace("_predictions", "_run") + ".json")
        require(pred.is_file() and side.is_file(), f"missing inner-OOF artifact: {pred}")
        sidecar = read_json(side)
        require(sidecar.get("status") == "COMPLETE", f"incomplete inner run: {side}")
        require(sidecar.get("objective") == objective and int(sidecar.get("seed")) == seed,
                f"wrong seed/objective binding: {side}")
        require(int(sidecar.get("outer_fold")) == outer and int(sidecar.get("inner_fold")) == inner,
                f"wrong fold binding: {side}")
        require(int(sidecar.get("outer_test_fit_overlap", -1)) == 0 and
                int(sidecar.get("outer_test_heldout_overlap", -1)) == 0,
                f"inner split leakage recorded: {side}")
        require(sidecar.get("prediction_sha256") == sha(pred), f"inner prediction hash mismatch: {pred}")
        frame = pd.read_csv(pred)
        required = {"text_id", "pair_id", "label", "raw_decision_score", "outer_fold", "inner_fold", "seed", "objective"}
        require(required <= set(frame.columns), f"inner prediction schema mismatch: {pred}")
        require(len(frame) == 600 and frame.text_id.is_unique, f"inner prediction cardinality mismatch: {pred}")
        require(frame.outer_fold.eq(outer).all() and frame.inner_fold.eq(inner).all() and
                frame.seed.eq(seed).all() and frame.objective.eq(objective).all(),
                f"inner prediction identity mismatch: {pred}")
        parts.append(frame)
        source_bindings.extend([{"path": rel(pred), "sha256": sha(pred)},
                                {"path": rel(side), "sha256": sha(side)}])
    oof = pd.concat(parts, ignore_index=True)
    require(len(oof) == 2400 and oof.text_id.is_unique, f"OOF coverage mismatch: {method}/{seed}/{outer}")
    y = labels(oof)
    cal = fit_platt(oof.raw_decision_score.to_numpy(float), y)
    p = probabilities(float(cal.coef_[0, 0]), float(cal.intercept_[0]), oof.raw_decision_score.to_numpy(float))
    thresholds = {}
    for alpha in ALPHAS:
        human = np.sort(p[y == 0])[::-1]
        k = np_umbrella_k(len(human), alpha=alpha, delta=0.05)
        require(k < len(human), f"threshold index infeasible: {alpha}")
        thresholds[f"{alpha:.2f}"] = {"threshold": float(human[k]), "k_exceed": int(k), "negative_n": int(len(human))}
    row = {
        "method": method, "objective": objective, "seed": seed, "outer_fold": outer,
        "calibration_method": "Platt logistic", "coefficient": float(cal.coef_[0, 0]),
        "intercept": float(cal.intercept_[0]), "calibration_rows": len(oof),
        "calibration_human": int((y == 0).sum()), "calibration_ai": int((y == 1).sum()),
        "outer_test_rows_used_for_calibration": 0, "external_rows_used_for_calibration": 0,
        "thresholds": thresholds, "source_bindings": source_bindings,
    }
    return row, source_bindings


def verify_seal_label_blind() -> dict[str, Any]:
    seal = read_json(SEAL)
    require(seal.get("status") == "SEALED_COMPLETE_LABEL_BLIND_PREDICTION_GRID", "sealed grid incomplete")
    require(int(seal.get("grid", {}).get("sealed_cells", -1)) == 135, "sealed grid is not 135 cells")
    for cell in seal["cells"]:
        pred = ROOT / cell["prediction"]["path"]
        side = ROOT / cell["sidecar"]["path"]
        require(sha(pred) == cell["prediction"]["sha256"], f"sealed prediction changed: {pred}")
        require(sha(side) == cell["sidecar"]["sha256"], f"sealed sidecar changed: {side}")
        require(read_json(side).get("external_labels_read") is False, f"label-blind sidecar failed: {side}")
    return seal


def prepare() -> None:
    require(not (OUT / "PARAMETER_MANIFEST.json").exists(), "supplementary parameters already frozen; refusing overwrite")
    cfg = read_json(CONFIG)
    seal = verify_seal_label_blind()
    rows = []
    bindings = [{"path": rel(CONFIG), "sha256": sha(CONFIG)}, {"path": rel(SEAL), "sha256": sha(SEAL)}]
    for method in METHODS:
        for seed in SEEDS:
            for outer in FOLDS:
                row, sources = fit_one(method, seed, outer)
                rows.append(row)
                bindings.extend(sources)
    params = pd.DataFrame(rows)
    atomic_csv(params.drop(columns=["thresholds", "source_bindings"]), OUT / "calibration_parameters.csv")
    atomic_json(OUT / "calibration_parameters_detail.json", {"rows": rows})
    manifest = {
        "schema_version": "posthoc-terminal-calibration-parameter-manifest-v1",
        "status": "FROZEN_BEFORE_EXTERNAL_LABEL_JOIN",
        "analysis_status": "SUPPLEMENTARY_CALIBRATION_READY_FOR_EXTERNAL_EVALUATION",
        "config": {"path": rel(CONFIG), "sha256": sha(CONFIG)},
        "implementation": {"path": rel(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
        "sealed_prediction_manifest": {"path": rel(SEAL), "sha256": sha(SEAL)},
        "realizations": 30, "external_labels_read": False,
        "external_performance_used_for_parameters": False,
        "leakage_assessment": {
            "inner_oof_rows_per_realization": 2400,
            "inner_fit_rows_per_fold": 1800,
            "inner_heldout_rows_per_fold": 600,
            "outer_test_rows_used_for_calibration": 0,
            "outer_test_rows_used_for_threshold": 0,
            "external_rows_used_for_calibration": 0,
            "all_inner_sidecars_report_zero_outer_test_overlap": True,
            "all_inner_prediction_hashes_verified": True
        },
        "source_bindings": bindings,
        "disclosure": cfg["disclosure"],
    }
    manifest["outputs"] = {
        "calibration_parameters": {
            "path": rel(OUT / "calibration_parameters.csv"),
            "sha256": sha(OUT / "calibration_parameters.csv")
        },
        "calibration_parameters_detail": {
            "path": rel(OUT / "calibration_parameters_detail.json"),
            "sha256": sha(OUT / "calibration_parameters_detail.json")
        }
    }
    atomic_json(OUT / "PARAMETER_MANIFEST.json", manifest)


def metric_row(dataset: str, method: str, seed: int, outer: int, frame: pd.DataFrame,
               detail: dict[str, Any]) -> dict[str, Any]:
    y = frame.binary_label.to_numpy(int)
    p = probabilities(detail["coefficient"], detail["intercept"], frame.raw_score.to_numpy(float))
    out = {"dataset": dataset, "method": method, "seed": seed, "outer_fold": outer,
           "n": len(frame), "human_n": int((y == 0).sum()), "ai_n": int((y == 1).sum()),
           "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
           "ece_10_equal_width": float(ece_equal_width(y, p, 10))}
    for alpha in ALPHAS:
        tag = f"{alpha:.2f}"
        pred = (p > detail["thresholds"][tag]["threshold"]).astype(int)
        tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
        out.update({f"threshold_{tag}": detail["thresholds"][tag]["threshold"],
                    f"tn_{tag}": tn, f"fp_{tag}": fp, f"fn_{tag}": fn, f"tp_{tag}": tp,
                    f"fpr_{tag}": fp / (fp + tn) if fp + tn else math.nan,
                    f"tpr_{tag}": tp / (tp + fn) if tp + fn else math.nan,
                    f"fnr_{tag}": fn / (tp + fn) if tp + fn else math.nan,
                    f"fpr_minus_nominal_{tag}": fp / (fp + tn) - alpha if fp + tn else math.nan})
    return out


def evaluate() -> None:
    manifest = read_json(OUT / "PARAMETER_MANIFEST.json")
    require(manifest.get("status") == "FROZEN_BEFORE_EXTERNAL_LABEL_JOIN", "parameter manifest is not frozen")
    require(manifest.get("external_performance_used_for_parameters") is False, "invalid parameter manifest")
    seal = verify_seal_label_blind()
    details = { (r["method"], int(r["seed"]), int(r["outer_fold"])): r
                for r in read_json(OUT / "calibration_parameters_detail.json")["rows"] }
    metric_rows = []
    for dataset in ("renklier2026", "tum2024_tr", "bostancioglu2026_tr"):
        source = {"renklier2026": "renklier2026_test_standardized.csv",
                  "tum2024_tr": "tum2024_tr_test_standardized.csv",
                  "bostancioglu2026_tr": "bostancioglu2026_tr_full_standardized.csv"}[dataset]
        labeled = pd.read_csv(EXTERNAL / source)
        require(set(labeled.binary_label.unique()) == {0, 1}, f"external labels incomplete: {dataset}")
        for method in METHODS:
            for seed in SEEDS:
                for outer in FOLDS:
                    cell = next(c for c in seal["cells"] if c["dataset_id"] == dataset and
                                c["method"] == method and c["seed"] == seed and c["outer_fold"] == outer)
                    pred = pd.read_csv(ROOT / cell["prediction"]["path"])
                    require(sha(ROOT / cell["prediction"]["path"]) == cell["prediction"]["sha256"], "sealed prediction hash changed")
                    joined = pred.merge(labeled[["external_id", "binary_label", "source_group_id", "text_sha256"]],
                                        on="external_id", how="inner", validate="one_to_one")
                    require(len(joined) == len(pred), f"external label join incomplete: {dataset}")
                    joined = joined.rename(columns={"raw_score_or_logit": "raw_score"})
                    metric_rows.append(metric_row(dataset, method, seed, outer, joined,
                                                  details[(method, seed, outer)]))
    metrics = pd.DataFrame(metric_rows)
    atomic_csv(metrics, OUT / "external_metrics_per_realization.csv")
    summaries = []
    for (dataset, method), group in metrics.groupby(["dataset", "method"], sort=True):
        row = {"dataset": dataset, "method": method, "realizations": len(group)}
        for col in ["brier", "log_loss", "ece_10_equal_width"] + [f"{x}_{a:.2f}" for a in ALPHAS for x in ("fpr", "tpr", "fnr", "fpr_minus_nominal")]:
            row[f"{col}_mean"] = float(group[col].mean()); row[f"{col}_sd"] = float(group[col].std(ddof=1))
        summaries.append(row)
    atomic_csv(pd.DataFrame(summaries), OUT / "external_metrics_method_summary.csv")
    pairs = metrics.pivot_table(index=["dataset", "seed", "outer_fold"], columns="method",
                                values=["brier", "log_loss", "ece_10_equal_width"] +
                                [f"{x}_{a:.2f}" for a in ALPHAS for x in ("fpr", "tpr", "fnr")])
    delta = pd.DataFrame({"dataset": pairs.index.get_level_values(0),
                          "seed": pairs.index.get_level_values(1), "outer_fold": pairs.index.get_level_values(2)})
    for col in pairs.columns.levels[0]:
        delta[f"{col}_SUPCON_minus_CE"] = pairs[(col, "SUPCON")].to_numpy() - pairs[(col, "CE_ONLY_MATCHED")].to_numpy()
    atomic_csv(delta, OUT / "matched_ce_supcon_differences.csv")
    atomic_json(OUT / "ANALYSIS_STATUS.json", {
        "status": "SUPPLEMENTARY_CALIBRATION_COMPLETE",
        "external_labels_read_after_parameter_freeze": True,
        "external_predictions_rerun": False, "external_prediction_seal_modified": False,
        "lit2026": "UNAVAILABLE_BY_DESIGN",
        "limitations": [
            "Metrics are exploratory and post-hoc; nominal internal FPR does not guarantee external FPR.",
            "Realization summaries describe five dependent outer folds per seed, not independent texts.",
            "Bootstrap uncertainty is not reported where the existing protocol does not define a valid paired resampling estimand."
        ]
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "evaluate"))
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else evaluate)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
