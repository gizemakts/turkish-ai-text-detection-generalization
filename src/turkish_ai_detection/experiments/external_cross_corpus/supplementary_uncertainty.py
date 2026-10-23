"""Independent validation and conditional paired cluster bootstrap analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from turkish_ai_detection.data.study import np_umbrella_k
from turkish_ai_detection.features.core import ece_equal_width

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".gitignore").is_file())
BASE = ROOT / "results/posthoc_terminal_calibration"
OUT = ROOT / "results/posthoc_terminal_calibration_uncertainty"
CONFIG = ROOT / "configs/posthoc_terminal_calibration.json"
SEAL = ROOT / "provenance/experiments/external_cross_corpus_v2/prediction_seal/PRESEAL_PREDICTION_MANIFEST.json"
INNER = ROOT / "results/calibration/inner_oof"
EXTERNAL = ROOT / "data/external_prepared"
METHODS = ("CE_ONLY_MATCHED", "SUPCON")
OBJECTIVES = {"CE_ONLY_MATCHED": "ce", "SUPCON": "supcon"}
SEEDS = (42, 314, 2026)
FOLDS = tuple(range(5))
INNERS = tuple(range(4))
ALPHAS = (0.01, 0.02, 0.03, 0.05)
DATASETS = {
    "renklier2026": ("renklier2026_test_standardized.csv", "source_group_id"),
    "tum2024_tr": ("tum2024_tr_test_standardized.csv", "text_sha256"),
    "bostancioglu2026_tr": ("bostancioglu2026_tr_full_standardized.csv", "text_sha256"),
}
REPLICATES = 2000
BOOTSTRAP_SEED = 2604


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


def probabilities(coef: float, intercept: float, raw: np.ndarray) -> np.ndarray:
    z = np.clip(coef * np.asarray(raw, dtype=float) + intercept, -745.0, 745.0)
    p = 1.0 / (1.0 + np.exp(-z))
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "invalid probability")
    return p


def metrics(y: np.ndarray, p: np.ndarray, thresholds: dict[str, Any]) -> dict[str, float]:
    require(set(np.unique(y)) == {0, 1}, "bootstrap sample must contain both classes")
    out = {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(np.clip(p, 1e-15, 1.0)) +
                                     (1 - y) * np.log(np.clip(1 - p, 1e-15, 1.0)))),
        "ece_10_equal_width": ece_equal_width(y, p, 10),
    }
    for alpha in ALPHAS:
        tag = f"{alpha:.2f}"
        pred = p > float(thresholds[tag]["threshold"])
        human = y == 0
        ai = y == 1
        out[f"fpr_{tag}"] = float(pred[human].mean())
        out[f"tpr_{tag}"] = float(pred[ai].mean())
        out[f"fnr_{tag}"] = float((~pred[ai]).mean())
    return out


def sample_unit_indices(
    rng: np.random.Generator,
    unit_indices: list[np.ndarray],
    unit_labels: np.ndarray,
    class_stratified: bool,
) -> np.ndarray:
    if class_stratified:
        sampled_units = np.concatenate([
            rng.choice(np.flatnonzero(unit_labels == cls),
                       size=int((unit_labels == cls).sum()), replace=True)
            for cls in (0, 1)
        ])
    else:
        sampled_units = rng.integers(0, len(unit_indices), size=len(unit_indices))
    return np.concatenate([unit_indices[i] for i in sampled_units])


def verify_inner_and_parameters() -> dict[str, Any]:
    detail = read_json(BASE / "calibration_parameters_detail.json")
    parameter_manifest = read_json(BASE / "PARAMETER_MANIFEST.json")
    require(parameter_manifest["status"] == "FROZEN_BEFORE_EXTERNAL_LABEL_JOIN",
            "parameters were not frozen before external evaluation")
    require(parameter_manifest["external_labels_read"] is False, "invalid label-order manifest")
    require(sha(CONFIG) == parameter_manifest["config"]["sha256"], "config hash mismatch")
    require(sha(SEAL) == parameter_manifest["sealed_prediction_manifest"]["sha256"],
            "sealed-manifest hash mismatch")
    rows = detail["rows"]
    require(len(rows) == 30, "expected 30 calibration rows")
    checked = 0
    threshold_rows = []
    for row in rows:
        objective = row["objective"]
        parts = []
        for inner in INNERS:
            pred = INNER / objective / f"seed_{row['seed']}" / f"outer_{row['outer_fold']}_inner_{inner}_predictions.csv"
            side = pred.with_name(pred.stem.replace("_predictions", "_run") + ".json")
            sidecar = read_json(side)
            require(sha(pred) == sidecar["prediction_sha256"], f"inner hash mismatch: {pred}")
            require(sidecar["objective"] == objective and int(sidecar["seed"]) == row["seed"] and
                    int(sidecar["outer_fold"]) == row["outer_fold"] and
                    int(sidecar["inner_fold"]) == inner, f"wrong inner identity: {side}")
            require(sidecar["outer_test_fit_overlap"] == 0 and sidecar["outer_test_heldout_overlap"] == 0,
                    f"outer-test overlap recorded: {side}")
            frame = pd.read_csv(pred)
            require(len(frame) == 600 and frame.text_id.is_unique, f"inner coverage mismatch: {pred}")
            require(frame.objective.eq(objective).all() and frame.seed.eq(row["seed"]).all() and
                    frame.outer_fold.eq(row["outer_fold"]).all() and frame.inner_fold.eq(inner).all(),
                    f"inner prediction identity mismatch: {pred}")
            parts.append(frame)
        oof = pd.concat(parts, ignore_index=True)
        require(len(oof) == 2400 and oof.text_id.is_unique, "OOF rows are not unique")
        y = oof.label.astype(str).str.upper().eq("AI").astype(int).to_numpy()
        require(int((y == 0).sum()) == 1200 and int((y == 1).sum()) == 1200,
                "internal class balance mismatch")
        require(np.isfinite(oof.raw_decision_score).all(), "non-finite internal score")
        refit = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000).fit(
            oof[["raw_decision_score"]], y)
        require(np.isclose(refit.coef_[0, 0], row["coefficient"], rtol=1e-10, atol=1e-10) and
                np.isclose(refit.intercept_[0], row["intercept"], rtol=1e-10, atol=1e-10),
                f"Platt coefficient mismatch: {row['method']}/{row['seed']}/{row['outer_fold']}")
        p = probabilities(row["coefficient"], row["intercept"], oof.raw_decision_score.to_numpy())
        for alpha in ALPHAS:
            tag = f"{alpha:.2f}"
            human = np.sort(p[y == 0])[::-1]
            k = np_umbrella_k(len(human), alpha=alpha, delta=0.05)
            saved = row["thresholds"][tag]
            require(saved["negative_n"] == 1200 and saved["k_exceed"] == k and
                    np.isclose(saved["threshold"], human[k], rtol=1e-10, atol=1e-12),
                    f"threshold mismatch: {row['method']}/{row['seed']}/{row['outer_fold']}/{tag}")
            threshold_rows.append({"alpha": alpha, "n_negative": len(human), "k": k})
        checked += 1

    seal = read_json(SEAL)
    require(seal["status"] == "SEALED_COMPLETE_LABEL_BLIND_PREDICTION_GRID" and
            len(seal["cells"]) == 135, "sealed prediction grid incomplete")
    for cell in seal["cells"]:
        pred = ROOT / cell["prediction"]["path"]
        side = ROOT / cell["sidecar"]["path"]
        require(sha(pred) == cell["prediction"]["sha256"] and sha(side) == cell["sidecar"]["sha256"],
                f"sealed cell changed: {pred}")
        require(read_json(side).get("external_labels_read") is False, f"label-blind sidecar failed: {side}")
    return {
        "status": "PASS",
        "calibration_rows_checked": checked,
        "sealed_cells_checked": len(seal["cells"]),
        "thresholds_checked": len(threshold_rows),
        "threshold_sample_sizes": sorted({(x["alpha"], x["n_negative"], x["k"]) for x in threshold_rows}),
        "score_orientation": "higher raw score and higher calibrated probability indicate AI",
        "decision_rule": "strict probability > threshold",
        "leakage_check": "all inner sidecars report zero outer-test overlap; each OOF row is unique",
        "finite_sample_scope": "NP order-statistic bound applies to the internal calibration negative-class sampling model; it is not an external dataset-shift guarantee.",
    }


def load_realization(dataset: str, method: str, seed: int, outer: int,
                     detail_map: dict[tuple[str, int, int], dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    seal = read_json(SEAL)
    cell = next(c for c in seal["cells"] if c["dataset_id"] == dataset and c["method"] == method
                and c["seed"] == seed and c["outer_fold"] == outer)
    pred_path = ROOT / cell["prediction"]["path"]
    require(sha(pred_path) == cell["prediction"]["sha256"], f"sealed prediction changed: {pred_path}")
    pred = pd.read_csv(pred_path)
    source_file, unit_col = DATASETS[dataset]
    lab = pd.read_csv(EXTERNAL / source_file)
    require(len(pred) == len(lab) and pred.external_id.is_unique and lab.external_id.is_unique,
            f"external cardinality/identity mismatch: {dataset}")
    label_columns = ["external_id", "binary_label", unit_col]
    label_frame = lab[label_columns].rename(columns={unit_col: "_label_unit"})
    joined = pred.merge(label_frame, on="external_id",
                        how="inner", validate="one_to_one")
    require(len(joined) == len(pred), f"incomplete external join: {dataset}")
    if unit_col in pred.columns:
        require(joined[unit_col].astype(str).to_numpy().tolist() ==
                joined["_label_unit"].astype(str).to_numpy().tolist(),
                f"external unit identity mismatch: {dataset}")
    else:
        require("_label_unit" in joined.columns, f"missing external unit: {dataset}")
    detail = detail_map[(method, seed, outer)]
    y = joined.binary_label.to_numpy(int)
    p = probabilities(detail["coefficient"], detail["intercept"], joined.raw_score_or_logit.to_numpy(float))
    units = joined["_label_unit"].astype(str).to_numpy()
    unique_units, inverse = np.unique(units, return_inverse=True)
    unit_indices = [np.flatnonzero(inverse == i) for i in range(len(unique_units))]
    return y, p, unit_indices


def bootstrap() -> None:
    audit = verify_inner_and_parameters()
    details = {(r["method"], int(r["seed"]), int(r["outer_fold"])): r
               for r in read_json(BASE / "calibration_parameters_detail.json")["rows"]}
    all_summary = []
    all_bootstrap = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for dataset in DATASETS:
        arrays = {}
        unit_indices = None
        for method in METHODS:
            for seed in SEEDS:
                for outer in FOLDS:
                    y, p, indices = load_realization(dataset, method, seed, outer, details)
                    arrays[(method, seed, outer)] = (y, p, details[(method, seed, outer)])
                    if unit_indices is None:
                        unit_indices = indices
                    else:
                        require(len(indices) == len(unit_indices) and
                                all(np.array_equal(a, b) for a, b in zip(indices, unit_indices)),
                                f"paired unit mismatch: {dataset}")
        unit_count = len(unit_indices)
        source_file, unit_col = DATASETS[dataset]
        labels = arrays[(METHODS[0], SEEDS[0], FOLDS[0])][0]
        unit_labels = np.array([labels[idx[0]] for idx in unit_indices])
        mixed_units = sum(len(np.unique(labels[idx])) > 1 for idx in unit_indices)
        class_stratified = mixed_units == 0
        point_values = {}
        metric_names = ["brier", "log_loss", "ece_10_equal_width"] + [
            f"{x}_{a:.2f}" for a in ALPHAS for x in ("fpr", "tpr", "fnr")]
        for method in METHODS:
            realization_values = []
            for seed in SEEDS:
                for outer in FOLDS:
                    y, p, detail = arrays[(method, seed, outer)]
                    realization_values.append(metrics(y, p, detail["thresholds"]))
            point_values[method] = {name: float(np.mean([x[name] for x in realization_values]))
                                    for name in metric_names}
        for rep in range(REPLICATES):
            row_indices = sample_unit_indices(rng, unit_indices, unit_labels, class_stratified)
            rep_values = {}
            for method in METHODS:
                vals = []
                for seed in SEEDS:
                    for outer in FOLDS:
                        y, p, detail = arrays[(method, seed, outer)]
                        vals.append(metrics(y[row_indices], p[row_indices], detail["thresholds"]))
                rep_values[method] = {name: float(np.mean([x[name] for x in vals])) for name in metric_names}
            for name in metric_names:
                all_bootstrap.append({
                    "dataset": dataset, "replicate": rep, "metric": name,
                    "ce": rep_values["CE_ONLY_MATCHED"][name],
                    "supcon": rep_values["SUPCON"][name],
                    "supcon_minus_ce": rep_values["SUPCON"][name] - rep_values["CE_ONLY_MATCHED"][name],
                })
        boot = pd.DataFrame([x for x in all_bootstrap if x["dataset"] == dataset])
        for name in metric_names:
            for estimand, point in [
                ("CE_ONLY_MATCHED", point_values["CE_ONLY_MATCHED"][name]),
                ("SUPCON", point_values["SUPCON"][name]),
                ("SUPCON_MINUS_CE", point_values["SUPCON"][name] - point_values["CE_ONLY_MATCHED"][name]),
            ]:
                col = "supcon_minus_ce" if estimand == "SUPCON_MINUS_CE" else estimand.lower().replace("_", "")
                if estimand == "CE_ONLY_MATCHED":
                    col = "ce"
                elif estimand == "SUPCON":
                    col = "supcon"
                values = boot.loc[boot.metric == name, col].to_numpy()
                all_summary.append({
                    "dataset": dataset, "estimand": estimand, "metric": name,
                    "point_estimate": point, "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)), "replicates": REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED, "unit": unit_col,
                    "unit_count": unit_count, "mixed_label_units": mixed_units,
                    "class_stratified": class_stratified,
                    "interpretation": "conditional on fixed 30 model realizations; external-unit sampling uncertainty only",
                })
    atomic_csv(pd.DataFrame(all_summary), OUT / "bootstrap_uncertainty_summary.csv")
    atomic_csv(pd.DataFrame(all_bootstrap), OUT / "bootstrap_replicates.csv")
    report = {
        "status": "SUPPLEMENTARY_UNCERTAINTY_COMPLETE",
        "validation": audit,
        "bootstrap": {
            "replicates": REPLICATES, "seed": BOOTSTRAP_SEED,
            "resampling": "paired cluster bootstrap; identical sampled units for CE and SupCon and all realizations",
            "point_estimate": "mean across fixed 15 seed/fold realizations per method",
            "ci": "2.5th and 97.5th percentiles",
            "training_variability": "not included; seeds/folds are fixed model realizations, not iid experiments",
            "dataset_specific": True,
        },
        "outputs": {
            "INDEPENDENT_VALIDATION.json": sha(OUT / "INDEPENDENT_VALIDATION.json")
            if (OUT / "INDEPENDENT_VALIDATION.json").is_file() else None,
            "bootstrap_uncertainty_summary.csv": sha(OUT / "bootstrap_uncertainty_summary.csv"),
            "bootstrap_replicates.csv": sha(OUT / "bootstrap_replicates.csv"),
        },
    }
    atomic_json(OUT / "VALIDATION_AND_BOOTSTRAP_STATUS.json", report)


def validate_only() -> None:
    report = verify_inner_and_parameters()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "INDEPENDENT_VALIDATION.json", report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("validate", "bootstrap"))
    args = parser.parse_args()
    (validate_only if args.phase == "validate" else bootstrap)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
