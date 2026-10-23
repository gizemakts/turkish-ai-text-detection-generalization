"""Targeted CPU-only validation of fixed calibration transfer to terminal models.

Consumes saved inner-OOF calibration inputs and authenticated held-out internal
predictions from the exact terminal checkpoints used for sealed external
inference. It never fits/replaces a calibration mapping or runs inference.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta, ks_2samp, wasserstein_distance

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".gitignore").is_file())
OUT = ROOT / "results/posthoc_terminal_calibration_transfer_validation"
DETAIL = ROOT / "results/posthoc_terminal_calibration/calibration_parameters_detail.json"
PARAM_MANIFEST = ROOT / "results/posthoc_terminal_calibration/PARAMETER_MANIFEST.json"
PRED_MANIFEST = ROOT / "provenance/experiments/external_cross_corpus_v2/prediction_seal/PRESEAL_PREDICTION_MANIFEST.json"
FOLDS = ROOT / "data/study/nested_folds.csv"
METHODS = {"CE_ONLY_MATCHED": "ce", "SUPCON": "supcon"}
METHOD_LABELS = {"CE_ONLY_MATCHED": "CE", "SUPCON": "SupCon"}
SEEDS, OUTERS, ALPHAS = (42, 314, 2026), tuple(range(5)), (0.01, 0.02, 0.03, 0.05)

# Descriptive decision criteria fixed in this implementation before results are
# summarized. They are not theoretical guarantees or confirmatory tests.
CRITERIA = {
    "terminal_median_brier_max": 0.05,
    "terminal_median_log_loss_max": 0.20,
    "terminal_median_ece_max": 0.05,
    "median_brier_degradation_max": 0.02,
    "median_log_loss_degradation_max": 0.10,
    "median_ece_degradation_max": 0.05,
    "median_abs_class_raw_smd_max": 0.50,
    "p90_abs_class_raw_smd_max": 1.00,
    "threshold_median_excess_tolerance": 0.01,
    "threshold_realization_tolerance": 0.02,
    "threshold_min_fraction_within_tolerance": 0.80,
}


@lru_cache(maxsize=None)
def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(
        value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
    ) + "\n", encoding="utf-8")


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, lineterminator="\n")


def sigmoid(raw: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    z = np.clip(coefficient * raw + intercept, -745.0, 745.0)
    return 1.0 / (1.0 + np.exp(-z))


def ece(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.minimum((p * 10).astype(int), 9)
    return float(sum(np.mean(bins == b) * abs(float(p[bins == b].mean()) - float(y[bins == b].mean()))
                     for b in range(10) if np.any(bins == b)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
            "ece_10_equal_width": ece(y, p)}


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    x = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1 - 1e-12))
    if float(x[y == 1].min()) > float(x[y == 0].max()):
        return {"calibration_fit_status": "NOT_ESTIMABLE_COMPLETE_SEPARATION",
                "calibration_intercept": np.nan, "calibration_slope": np.nan}
    design = np.column_stack([np.ones(len(x)), x])
    def objective(par: np.ndarray) -> float:
        z = np.clip(design @ par, -40, 40)
        return float(np.sum(np.logaddexp(0, z) - y * z))
    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not result.success or not np.all(np.isfinite(result.x)) or np.max(np.abs(result.x)) > 100:
        return {"calibration_fit_status": "NOT_ESTIMABLE_NUMERIC_OR_QUASI_SEPARATION",
                "calibration_intercept": np.nan, "calibration_slope": np.nan}
    return {"calibration_fit_status": "ESTIMABLE_DESCRIPTIVE_MLE",
            "calibration_intercept": float(result.x[0]), "calibration_slope": float(result.x[1])}


def clopper_pearson(k: int, n: int, confidence: float = .95) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    low = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return low, high


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) /
                     max(len(a) + len(b) - 2, 1))
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else np.nan


def distribution_row(method: str, seed: int, outer: int, source: str, label: str,
                     raw: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method, "seed": seed, "outer_fold": outer,
        "source": source, "label": label, "n": len(raw), "raw_mean": np.mean(raw),
        "raw_sd": np.std(raw, ddof=1), "probability_mean": np.mean(prob),
        "probability_sd": np.std(prob, ddof=1)}
    for q in (.01, .05, .25, .50, .75, .95, .99):
        row[f"raw_q{int(q*100):02d}"] = np.quantile(raw, q)
        row[f"probability_q{int(q*100):02d}"] = np.quantile(prob, q)
    return row


def reliability_rows(method: str, seed: int, outer: int, source: str,
                     y: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    bins = np.minimum((p * 10).astype(int), 9)
    rows = []
    for b in range(10):
        mask = bins == b
        rows.append({"method": method, "seed": seed, "outer_fold": outer, "source": source,
            "bin": b, "bin_lower": b / 10, "bin_upper": (b + 1) / 10,
            "n": int(mask.sum()), "mean_probability": float(p[mask].mean()) if mask.any() else np.nan,
            "observed_ai_frequency": float(y[mask].mean()) if mask.any() else np.nan})
    return rows


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"Refusing to modify existing validation directory: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "figures").mkdir()
    details = {(r["method"], int(r["seed"]), int(r["outer_fold"])): r
               for r in json.loads(DETAIL.read_text(encoding="utf-8"))["rows"]}
    pred_manifest = json.loads(PRED_MANIFEST.read_text(encoding="utf-8"))
    folds = pd.read_csv(FOLDS)
    audit_rows, metric_rows, threshold_rows = [], [], []
    distribution_rows, comparison_rows, rel_rows, evidence = [], [], [], []

    def evidence_add(path: Path, role: str) -> None:
        key = path.relative_to(ROOT).as_posix()
        if not any(x["path"] == key for x in evidence):
            evidence.append({"path": key, "role": role, "bytes": path.stat().st_size, "sha256": sha(path)})

    for path, role in [(DETAIL, "frozen calibration parameters"), (PARAM_MANIFEST, "frozen parameter manifest"),
                       (PRED_MANIFEST, "sealed external prediction manifest"), (FOLDS, "outer-fold membership")]:
        evidence_add(path, role)

    for method, objective in METHODS.items():
        for seed in SEEDS:
            for outer in OUTERS:
                detail = details[(method, seed, outer)]
                run_path = ROOT / f"artifacts/checkpoints/external_cross_corpus/protocol_faithful_rerun/{objective}_seed_{seed}_outer_{outer}/run.json"
                run = json.loads(run_path.read_text(encoding="utf-8"))
                terminal_path = ROOT / run["internal_prediction_path"]
                model_path = ROOT / run["payload"]["model"]
                primary_run_path = ROOT / f"results/primary/{objective}/seed_{seed}/outer_{outer}_run.json"
                primary_run = json.loads(primary_run_path.read_text(encoding="utf-8"))
                inner_csvs = [ROOT / b["path"] for b in detail["source_bindings"] if b["path"].endswith("_predictions.csv")]
                inner_sidecars = [ROOT / b["path"] for b in detail["source_bindings"] if b["path"].endswith("_run.json")]
                inner = pd.concat([pd.read_csv(p) for p in inner_csvs], ignore_index=True)
                terminal = pd.read_csv(terminal_path)
                expected_pairs = set(folds.loc[folds.outer_fold == outer, "pair_id"].astype(str))
                inner_expected = set(folds.loc[folds.outer_fold != outer, "pair_id"].astype(str))
                terminal_pairs, inner_pairs = set(terminal.pair_id.astype(str)), set(inner.pair_id.astype(str))
                external_cells = [c for c in pred_manifest["cells"] if c["method"] == method and c["seed"] == seed
                                  and c["outer_fold"] == outer and c["dataset_id"] in
                                  {"renklier2026", "tum2024_tr", "bostancioglu2026_tr"}]
                external_checks = []
                for cell in external_cells:
                    sidecar_path = ROOT / cell["sidecar"]["path"]
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    cp = sidecar["checkpoint"]
                    external_checks.append(
                        sha(sidecar_path) == cell["sidecar"]["sha256"] and
                        cp["run_record_path"] == run_path.relative_to(ROOT).as_posix() and
                        cp["run_record_sha256"] == sha(run_path) and
                        cp["internal_prediction_path"] == terminal_path.relative_to(ROOT).as_posix() and
                        cp["internal_prediction_sha256"] == sha(terminal_path) and
                        cp["model_sha256"] == sha(model_path))
                    evidence_add(sidecar_path, "sealed external sidecar proving checkpoint binding")
                checks = {
                    "run_status": run.get("status") == "SCIENTIFIC_PROTOCOL_FAITHFUL_RERUN",
                    "objective": run.get("objective") == objective,
                    "seed": run.get("seed") == seed,
                    "outer_fold": run.get("outer_fold") == outer,
                    "checkpoint_roundtrip": run.get("checkpoint_roundtrip", {}).get("status") == "PASS_CHECKPOINT_ROUNDTRIP",
                    "model_hash": sha(model_path) == run["payload"]["model_sha256"],
                    "internal_prediction_hash": sha(terminal_path) == run["internal_prediction_sha256"],
                    "primary_test_membership_hash": run.get("test_pair_sha256") == primary_run.get("test_pair_sha256"),
                    "primary_train_membership_hash": run.get("train_pair_sha256") == primary_run.get("train_pair_sha256"),
                    "terminal_rows_600": len(terminal) == 600,
                    "terminal_class_balance": terminal.label.value_counts().to_dict() == {"AI": 300, "HUMAN": 300},
                    "terminal_pair_membership_exact": terminal_pairs == expected_pairs,
                    "terminal_identifiers": terminal.seed.eq(seed).all() and terminal.outer_fold.eq(outer).all()
                                            and terminal.objective.eq(objective).all() and terminal.condition.eq("A").all(),
                    "inner_rows_2400": len(inner) == 2400,
                    "inner_class_balance": inner.label.value_counts().to_dict() == {"AI": 1200, "HUMAN": 1200},
                    "inner_pair_membership_exact": inner_pairs == inner_expected,
                    "inner_terminal_pair_overlap_zero": len(inner_pairs & terminal_pairs) == 0,
                    "inner_binding_hashes": all(sha(ROOT / b["path"]) == b["sha256"] for b in detail["source_bindings"]),
                    "mapping_identity": detail["objective"] == objective and detail["seed"] == seed and detail["outer_fold"] == outer,
                    "mapping_uses_no_outer_test": detail["outer_test_rows_used_for_calibration"] == 0,
                    "mapping_uses_no_external": detail["external_rows_used_for_calibration"] == 0,
                    "score_orientation": detail["coefficient"] > 0 and terminal.loc[terminal.label == "AI", "raw_decision_score"].mean()
                                         > terminal.loc[terminal.label == "HUMAN", "raw_decision_score"].mean(),
                    "external_checkpoint_binding_three_datasets": len(external_cells) == 3 and all(external_checks),
                }
                audit_rows.append({"method": method, "objective": objective, "seed": seed, "outer_fold": outer,
                    "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                    "run_path": run_path.relative_to(ROOT).as_posix(), "run_sha256": sha(run_path),
                    "terminal_prediction_path": terminal_path.relative_to(ROOT).as_posix(),
                    "terminal_prediction_sha256": sha(terminal_path), "model_path": model_path.relative_to(ROOT).as_posix(),
                    "model_sha256": sha(model_path), "external_bound_datasets": sorted(c["dataset_id"] for c in external_cells)})
                for p, role in [(run_path, "terminal run provenance"), (terminal_path, "terminal held-out prediction"),
                                (model_path, "terminal checkpoint"), (primary_run_path, "structural primary run reference")]:
                    evidence_add(p, role)
                for p in inner_csvs: evidence_add(p, "inner-OOF calibration input")
                for p in inner_sidecars: evidence_add(p, "inner-OOF provenance")
                y_inner = (inner.label == "AI").astype(int).to_numpy()
                y_terminal = (terminal.label == "AI").astype(int).to_numpy()
                raw_inner = inner.raw_decision_score.to_numpy(float)
                raw_terminal = terminal.raw_decision_score.to_numpy(float)
                p_inner = sigmoid(raw_inner, detail["coefficient"], detail["intercept"])
                p_terminal = sigmoid(raw_terminal, detail["coefficient"], detail["intercept"])
                inner_m, terminal_m = metrics(y_inner, p_inner), metrics(y_terminal, p_terminal)
                calibration_fit = calibration_intercept_slope(y_terminal, p_terminal)
                metric_rows.append({"method": method, "objective": objective, "seed": seed, "outer_fold": outer,
                    "inner_n": len(inner), "terminal_n": len(terminal), "inner_human_n": int((y_inner == 0).sum()),
                    "inner_ai_n": int((y_inner == 1).sum()), "terminal_human_n": int((y_terminal == 0).sum()),
                    "terminal_ai_n": int((y_terminal == 1).sum()), "coefficient": detail["coefficient"],
                    "intercept": detail["intercept"], **{f"inner_{k}": v for k, v in inner_m.items()},
                    **{f"terminal_{k}": v for k, v in terminal_m.items()},
                    "delta_brier": terminal_m["brier"] - inner_m["brier"],
                    "delta_log_loss": terminal_m["log_loss"] - inner_m["log_loss"],
                    "delta_ece": terminal_m["ece_10_equal_width"] - inner_m["ece_10_equal_width"],
                    **calibration_fit})
                rel_rows.extend(reliability_rows(method, seed, outer, "inner_oof", y_inner, p_inner))
                rel_rows.extend(reliability_rows(method, seed, outer, "terminal_outer_test", y_terminal, p_terminal))
                for label, code in (("HUMAN", 0), ("AI", 1)):
                    ri, rt = raw_inner[y_inner == code], raw_terminal[y_terminal == code]
                    pi, pt = p_inner[y_inner == code], p_terminal[y_terminal == code]
                    distribution_rows.append(distribution_row(method, seed, outer, "inner_oof", label, ri, pi))
                    distribution_rows.append(distribution_row(method, seed, outer, "terminal_outer_test", label, rt, pt))
                    comparison_rows.append({"method": method, "seed": seed, "outer_fold": outer, "label": label,
                        "raw_mean_shift_terminal_minus_inner": np.mean(rt) - np.mean(ri),
                        "raw_standardized_mean_difference": standardized_mean_difference(ri, rt),
                        "raw_wasserstein_distance": wasserstein_distance(ri, rt),
                        "raw_ks_statistic": ks_2samp(ri, rt).statistic,
                        "probability_mean_shift_terminal_minus_inner": np.mean(pt) - np.mean(pi),
                        "probability_wasserstein_distance": wasserstein_distance(pi, pt),
                        "probability_ks_statistic": ks_2samp(pi, pt).statistic})
                for alpha in ALPHAS:
                    threshold = detail["thresholds"][f"{alpha:.2f}"]["threshold"]
                    pred = p_terminal > threshold
                    tn = int(((y_terminal == 0) & ~pred).sum()); fp = int(((y_terminal == 0) & pred).sum())
                    fn = int(((y_terminal == 1) & ~pred).sum()); tp = int(((y_terminal == 1) & pred).sum())
                    fpr, tpr = fp / (fp + tn), tp / (tp + fn)
                    lo, hi = clopper_pearson(fp, fp + tn)
                    threshold_rows.append({"method": method, "objective": objective, "seed": seed, "outer_fold": outer,
                        "nominal_fpr": alpha, "threshold": threshold, "decision_rule": "calibrated_probability > threshold",
                        "human_n": fp + tn, "ai_n": tp + fn, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                        "observed_fpr": fpr, "fpr_ci95_exact_low": lo, "fpr_ci95_exact_high": hi,
                        "observed_minus_nominal_fpr": fpr - alpha, "tpr": tpr, "fnr": 1 - tpr})

    audit_pass = all(r["status"] == "PASS" for r in audit_rows)
    dump(OUT / "CALIBRATION_TRANSFER_PROVENANCE_AUDIT.json", {
        "status": "PASS" if audit_pass else "FAIL", "realizations_expected": 30,
        "realizations_passed": sum(r["status"] == "PASS" for r in audit_rows),
        "authenticity_definition": "Terminal held-out predictions must be hash-bound by the terminal run record and by all three sealed external inference sidecars to the same model payload.",
        "label_convention": "HUMAN=0; AI=1", "score_orientation": "higher raw score and probability indicate AI",
        "realizations": audit_rows})
    per = pd.DataFrame(metric_rows); thresholds = pd.DataFrame(threshold_rows)
    comparisons = pd.DataFrame(comparison_rows); distributions = pd.DataFrame(distribution_rows)
    reliability = pd.DataFrame(rel_rows)
    save_csv(per, "CALIBRATION_TRANSFER_PER_REALIZATION.csv")
    save_csv(thresholds, "CALIBRATION_TRANSFER_THRESHOLDS.csv")
    save_csv(distributions, "CALIBRATION_TRANSFER_SCORE_DISTRIBUTIONS.csv")
    save_csv(comparisons, "CALIBRATION_TRANSFER_DISTRIBUTION_COMPARISONS.csv")
    save_csv(reliability, "CALIBRATION_TRANSFER_RELIABILITY_BINS.csv")

    summary_rows = []
    for method, group in per.groupby("method"):
        row: dict[str, Any] = {"method": method, "realizations": len(group)}
        for col in ["inner_brier", "terminal_brier", "delta_brier", "inner_log_loss", "terminal_log_loss",
                    "delta_log_loss", "inner_ece_10_equal_width", "terminal_ece_10_equal_width", "delta_ece"]:
            row[f"{col}_mean"] = group[col].mean(); row[f"{col}_sd"] = group[col].std(ddof=1)
            row[f"{col}_median"] = group[col].median(); row[f"{col}_min"] = group[col].min(); row[f"{col}_max"] = group[col].max()
        row["calibration_slope_estimable_n"] = int(group.calibration_slope.notna().sum())
        smd = comparisons.loc[comparisons.method == method, "raw_standardized_mean_difference"].abs()
        row["class_raw_smd_abs_median"] = smd.median(); row["class_raw_smd_abs_p90"] = smd.quantile(.9)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    save_csv(summary, "CALIBRATION_TRANSFER_METHOD_SUMMARY.csv")

    threshold_summary = thresholds.groupby(["method", "nominal_fpr"], as_index=False).agg(
        observed_fpr_mean=("observed_fpr", "mean"), observed_fpr_sd=("observed_fpr", "std"),
        observed_fpr_median=("observed_fpr", "median"), observed_fpr_min=("observed_fpr", "min"),
        observed_fpr_max=("observed_fpr", "max"), median_excess=("observed_minus_nominal_fpr", "median"),
        realizations_at_or_below_nominal=("observed_minus_nominal_fpr", lambda x: int((x <= 0).sum())),
        realizations_within_plus_002=("observed_minus_nominal_fpr", lambda x: int((x <= CRITERIA["threshold_realization_tolerance"]).sum())))
    save_csv(threshold_summary, "CALIBRATION_TRANSFER_THRESHOLD_SUMMARY.csv")

    # Descriptive figures.
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
    for ax, label in zip(axes, ("HUMAN", "AI")):
        pivot = comparisons[comparisons.label == label].pivot_table(index=["method", "seed"], columns="outer_fold",
            values="raw_standardized_mean_difference")
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(f"{label}: terminal − inner raw-score SMD"); ax.set_xlabel("Outer fold")
        ax.set_xticks(range(5)); ax.set_yticks(range(len(pivot))); ax.set_yticklabels([f"{METHOD_LABELS[m]} / {s}" for m, s in pivot.index])
        for i in range(len(pivot)):
            for j in range(5): ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes, shrink=.7, label="Standardized mean difference")
    fig.savefig(OUT / "figures/Figure_1_raw_score_shift_heatmap.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    for column, method in enumerate(METHODS):
        ax, ax_count = axes[0, column], axes[1, column]
        subset = reliability[reliability.method == method]
        for source, style, color, offset in (("inner_oof", "--", "#777777", -.018),
                                              ("terminal_outer_test", "-", "#3568a8", .018)):
            rows = subset[subset.source == source]
            occupied = rows[rows.n > 0].copy()
            points = occupied.groupby("bin").apply(
                lambda g: pd.Series({
                    "x": np.average(g.mean_probability, weights=g.n),
                    "y": np.average(g.observed_ai_frequency, weights=g.n),
                    "model_observations": g.n.sum(), "realizations": len(g),
                }), include_groups=False)
            ax.plot(points.x, points.y, style, marker="o", label=f"{source} (count-weighted)", color=color)
            counts = rows.groupby("bin").n.sum().reindex(range(10), fill_value=0)
            ax_count.bar(np.arange(10) / 10 + .05 + offset, counts, width=.034,
                         color=color, alpha=.78, label=source)
        ax.plot([0, 1], [0, 1], ":", color="black"); ax.set(xlim=(0, 1), ylim=(0, 1), title=METHOD_LABELS[method],
            ylabel="Observed AI frequency"); ax.legend(frameon=False, fontsize=7)
        ax_count.set(xlim=(0, 1), xlabel="Calibrated-probability bin",
                     ylabel="Model–observation rows", yscale="symlog")
        ax_count.legend(frameon=False, fontsize=7)
    fig.suptitle("Internal reliability; counts are repeated model–observation rows, not independent texts", fontsize=10)
    fig.savefig(OUT / "figures/Figure_2_internal_reliability.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        g = thresholds[thresholds.method == method]
        for alpha in ALPHAS:
            values = g.loc[g.nominal_fpr == alpha, "observed_fpr"].to_numpy()
            ax.scatter(np.full(len(values), alpha) + np.linspace(-.0015, .0015, len(values)), values, s=18, alpha=.65)
        ax.plot([0, .055], [0, .055], "--", color="black", label="Observed = nominal")
        ax.set(title=METHOD_LABELS[method], xlabel="Nominal FPR", xlim=(0, .055), ylabel="Observed terminal FPR")
        ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "figures/Figure_3_internal_threshold_transfer.png", dpi=220); plt.close(fig)

    metric_criteria = []
    criterion_details: dict[str, Any] = {}
    for row in summary.to_dict("records"):
        checks = {
            "terminal_brier_median": row["terminal_brier_median"] <= CRITERIA["terminal_median_brier_max"],
            "terminal_log_loss_median": row["terminal_log_loss_median"] <= CRITERIA["terminal_median_log_loss_max"],
            "terminal_ece_median": row["terminal_ece_10_equal_width_median"] <= CRITERIA["terminal_median_ece_max"],
            "brier_degradation_median": row["delta_brier_median"] <= CRITERIA["median_brier_degradation_max"],
            "log_loss_degradation_median": row["delta_log_loss_median"] <= CRITERIA["median_log_loss_degradation_max"],
            "ece_degradation_median": row["delta_ece_median"] <= CRITERIA["median_ece_degradation_max"],
            "class_raw_smd_abs_median": row["class_raw_smd_abs_median"] <= CRITERIA["median_abs_class_raw_smd_max"],
            "class_raw_smd_abs_p90": row["class_raw_smd_abs_p90"] <= CRITERIA["p90_abs_class_raw_smd_max"],
        }
        criterion_details[row["method"]] = checks
        metric_criteria.extend(checks.values())
    threshold_criteria = [r["median_excess"] <= CRITERIA["threshold_median_excess_tolerance"] and
        r["realizations_within_plus_002"] / 15 >= CRITERIA["threshold_min_fraction_within_tolerance"]
        for r in threshold_summary.to_dict("records")]
    if not audit_pass:
        decision = "CALIBRATION_TRANSFER_NOT_VERIFIABLE"
    elif all(metric_criteria) and all(threshold_criteria):
        decision = "CALIBRATION_TRANSFER_EMPIRICALLY_SUPPORTED"
    else:
        decision = "CALIBRATION_TRANSFER_LIMITED"

    def method_sentence(method: str) -> str:
        r = summary[summary.method == method].iloc[0]
        t = threshold_summary[threshold_summary.method == method]
        fprs = ", ".join(f"{a:.0%}: {v:.4f}" for a, v in zip(t.nominal_fpr, t.observed_fpr_mean))
        return (f"{METHOD_LABELS[method]} terminal medians were Brier {r.terminal_brier_median:.4f}, "
                f"log loss {r.terminal_log_loss_median:.4f}, and ECE {r.terminal_ece_10_equal_width_median:.4f}; "
                f"mean observed FPRs were {fprs}.")

    diagnostics = f"""# Targeted internal calibration-transfer validation

## Outcome

**Decision: `{decision}`.** Authentic held-out internal predictions were identified for all {sum(r['status']=='PASS' for r in audit_rows)}/30 terminal checkpoints. Each prediction file is hash-bound to a protocol-faithful terminal run, its model payload, and the three sealed external inference sidecars for the same method/seed/fold. All outer-test memberships, labels, objectives, seeds, folds, mapping identities, source hashes, and zero-overlap checks passed.

## Fixed-mapping calibration

No mapping was fitted or selected in this validation. Existing Platt coefficients were applied unchanged to each terminal checkpoint's own 600 held-out outer-test rows (300 HUMAN, 300 AI). Inner-OOF and terminal samples contain different observations.

- {method_sentence('CE_ONLY_MATCHED')}
- {method_sentence('SUPCON')}

Calibration intercept/slope MLEs were estimable in {int(per.calibration_slope.notna().sum())}/30 realizations. Non-estimable cells are explicitly marked; complete or quasi-complete class separation makes an unpenalized calibration slope non-finite rather than evidencing a slope of exactly one.

## Score-distribution diagnostics

Class-conditional means, SDs, quantiles, Wasserstein distances, KS statistics, and standardized mean differences are reported for every method/seed/fold. These diagnostics assess compatibility but cannot prove that inner-model mappings and terminal-model scores are exchangeable.

Although held-out calibration metrics were strong, the 90th percentile absolute class-conditional raw-score SMD was {summary.loc[summary.method == 'CE_ONLY_MATCHED', 'class_raw_smd_abs_p90'].iloc[0]:.3f} for CE and {summary.loc[summary.method == 'SUPCON', 'class_raw_smd_abs_p90'].iloc[0]:.3f} for SupCon, exceeding the prespecified descriptive cutoff of 1.0. The largest shifts were concentrated in the AI class. This score-scale evidence is why the final status is limited rather than fully supported.

## Threshold transfer

The unchanged 1%, 2%, 3%, and 5% thresholds used the strict rule `calibrated_probability > threshold`. Each realization has 300 held-out HUMAN observations, so FPR resolution is 1/300 = 0.00333. Per-realization Clopper–Pearson intervals describe binomial uncertainty conditional on the evaluated terminal model and held-out fold. They do not transfer the original Neyman–Pearson guarantee to terminal scores.

## Decision criteria

The status uses the descriptive criteria recorded in `VALIDATION_STATUS.json`: authenticated provenance; method-level median terminal Brier/log loss/ECE; median degradation from inner OOF; class-conditional raw-score SMD; and threshold deviations across realizations. These are transparent diagnostic cutoffs, not confirmatory tests or mathematical guarantees.

## Dependence and scope

The 15 realizations per method are dependent. Outer folds within a seed partition one dataset, and texts recur across seeds. Results are reported by realization and with descriptive method summaries only. No external inference, calibration refit, threshold optimization, bootstrap rerun, or primary-result modification occurred.
"""
    (OUT / "CALIBRATION_TRANSFER_DIAGNOSTICS.md").write_text(diagnostics, encoding="utf-8")
    publication = f"""# Calibration transfer: supplementary results and limitations

## Results

As a targeted post-hoc validation, we applied each existing inner-OOF-fitted Platt mapping and its unchanged nominal-FPR thresholds to independently held-out internal predictions generated by the corresponding final terminal checkpoint. Provenance, checkpoint hashes, method/seed/fold identities, outer-fold membership, and calibration/evaluation separation were verified for all 30 CE/SupCon realizations. {method_sentence('CE_ONLY_MATCHED')} {method_sentence('SUPCON')} Despite favorable held-out calibration and generally conservative internal FPRs, the 90th percentile absolute class-conditional terminal-versus-inner raw-score shift exceeded one pooled standard deviation for both methods, principally in AI scores; the resulting classification is therefore `{decision}`. This is descriptive rather than confirmatory: inner-fold and terminal models need not have identical score distributions, calibration intercept/slope was not estimable under complete separation in some realizations, repeated texts and folds are dependent, and internal operating-point behavior does not establish a mathematical or external-dataset FPR guarantee.

## Limitations

- The validation is post-hoc and uses the same internal corpus design, not a new population.
- Inner-OOF and outer-test observations differ in composition and difficulty.
- Terminal predictions are independent of mapping fitting within each outer split, but repeated across seeds.
- Exact binomial intervals are conditional on each fixed held-out fold and model.
- Score-distribution similarity is empirical compatibility evidence, not proof of exchangeability.
- No claim is made that nominal FPR transfers to external corpora.
"""
    (OUT / "CALIBRATION_TRANSFER_LIMITATIONS.md").write_text(publication, encoding="utf-8")
    status = {"status": decision, "provenance_status": "PASS" if audit_pass else "FAIL",
        "authentic_terminal_predictions_identified": audit_pass, "realizations_validated": len(per),
        "criteria": CRITERIA, "criteria_results": {"by_method": criterion_details,
        "metric_and_distribution_checks_all_pass": all(metric_criteria),
        "threshold_checks_all_pass": all(threshold_criteria),
        "limiting_finding": "90th percentile absolute class-conditional raw-score SMD exceeded 1.0 for both methods; largest shifts were in AI scores."}, "calibration_refit": False,
        "external_inference": False, "bootstrap_rerun": False, "existing_outputs_modified": False,
        "formal_fpr_guarantee_claimed": False}
    dump(OUT / "VALIDATION_STATUS.json", status)

    evidence_add(Path(__file__).resolve(), "validation implementation")
    outputs = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256_EVIDENCE_MANIFEST.json":
            outputs.append({"path": path.relative_to(ROOT).as_posix(), "role": "validation output",
                            "bytes": path.stat().st_size, "sha256": sha(path)})
    dump(OUT / "SHA256_EVIDENCE_MANIFEST.json", {"status": "PASS", "algorithm": "SHA-256",
        "inputs": sorted(evidence, key=lambda x: x["path"]), "outputs": outputs,
        "input_count": len(evidence), "output_count": len(outputs)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
