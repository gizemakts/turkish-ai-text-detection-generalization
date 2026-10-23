# -*- coding: utf-8 -*-
"""Prespecified one-time unblinding/statistics for GENERATOR_HOLDOUT_V1."""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def _resolve_project_root() -> Path:
    for _candidate in Path(__file__).resolve().parents:
        if (_candidate / ".gitignore").is_file():
            return _candidate
    raise RuntimeError("Turkish AI-Generated Text Detection Study project root could not be resolved")

ROOT = _resolve_project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SCRIPTS = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from turkish_ai_detection.data.study import resolve_frozen_record_path, sha256_file, write_json  # noqa: E402
from turkish_ai_detection.evaluation.metrics import _pauc  # noqa: E402
from turkish_ai_detection.reproducibility.reproduction_policy import expected_run_hashes, validate_current_migrated_reproduction  # noqa: E402

CONFIG = ROOT / "configs" / "generator_holdout.json"
RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/generator_holdout/run_fit.py"
STATS = Path(__file__)
FREEZE = ROOT / "provenance" / "experiments" / "generator_holdout" / "PREEXECUTION_FREEZE.json"
OUT = ROOT / "results" / "generator_holdout"
RESULTS = OUT / "final_statistics"

PRIMARY_OUT = ROOT / "results" / "primary"
PRIMARY_RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/primary/run_fit.py"
PRIMARY_CONFIG = ROOT / "configs" / "primary_evaluation.json"
PRIMARY_FREEZE = ROOT / "provenance" / "experiments" / "primary" / "PREEXECUTION_FREEZE.json"
PRIMARY_MANIFEST = PRIMARY_OUT / "final_statistics" / "final_statistics_manifest.json"

GENERATORS = ["Claude", "GPT", "Gemini", "Kimi", "Qwen"]
SEEDS = [42, 314, 2026]
OBJECTIVES = ["ce", "supcon"]
METRICS = ["roc_auc", "average_precision", "standardized_pauc_003"]


def slug(x: str) -> str:
    return x.lower().replace(" ", "_")


def metric_triplet(frame: pd.DataFrame):
    y = frame.label.astype(str).eq("AI").astype(int).to_numpy()
    s = frame.raw_decision_score.to_numpy(float)
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "standardized_pauc_003": float(_pauc(y, s)),
    }


def verify_freeze(cfg):
    if not FREEZE.is_file():
        raise RuntimeError("generator-holdout freeze missing")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("invalid generator-holdout freeze")
    policy = validate_current_migrated_reproduction(ROOT, "generator_holdout", RUNNER)
    return freeze, policy


def load_ood(cfg, freeze, policy):
    frames = {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            for generator in GENERATORS:
                base = OUT / objective / f"seed_{seed}"
                pred = base / f"holdout_{slug(generator)}_predictions.csv"
                side = base / f"holdout_{slug(generator)}_run.json"
                if not pred.is_file() or not side.is_file():
                    raise RuntimeError(f"missing OOD fit: {objective}/{seed}/{generator}")
                run = json.loads(side.read_text(encoding="utf-8"))
                if run.get("status") != "COMPLETE":
                    raise RuntimeError(f"incomplete OOD fit: {objective}/{seed}/{generator}")
                if run["prediction_sha256"] != sha256_file(pred):
                    raise RuntimeError(f"OOD prediction hash mismatch: {objective}/{seed}/{generator}")
                expected_config, expected_runner = expected_run_hashes(run, freeze, policy, config_key="config", runner_key="runner")
                if run["runner_sha256"] != expected_runner or run["config_sha256"] != expected_config:
                    raise RuntimeError(f"OOD runner/config mismatch: {objective}/{seed}/{generator}")
                if run["preexecution_freeze_sha256"] != sha256_file(FREEZE):
                    raise RuntimeError(f"OOD freeze mismatch: {objective}/{seed}/{generator}")

                f = pd.read_csv(pred).sort_values("text_id").reset_index(drop=True)
                if len(f) != 600 or f.text_id.nunique() != 600 or f.pair_id.nunique() != 300:
                    raise RuntimeError(f"OOD shape mismatch: {objective}/{seed}/{generator}")
                if set(f.generator.astype(str)) != {generator}:
                    raise RuntimeError(f"OOD generator mismatch: {objective}/{seed}/{generator}")
                if not np.isfinite(f.raw_decision_score.to_numpy(float)).all():
                    raise RuntimeError(f"nonfinite OOD score: {objective}/{seed}/{generator}")
                frames[(objective, seed, generator)] = f
    return frames


def load_primary_id(cfg):
    # Use the exact generator subset under the already-frozen primary pair-disjoint ID evaluation.
    manifest = json.loads(PRIMARY_MANIFEST.read_text(encoding="utf-8"))
    lineage = cfg["primary_rq1_lineage"]
    checks = {
        "runner": sha256_file(PRIMARY_RUNNER) == lineage["primary_runner_sha256"],
        "config": sha256_file(PRIMARY_CONFIG) == lineage["primary_config_sha256"],
        "freeze": sha256_file(PRIMARY_FREEZE) == lineage["primary_preexecution_freeze_sha256"],
        "manifest": sha256_file(PRIMARY_MANIFEST) == lineage["primary_final_statistics_manifest_sha256"],
        "manifest_status": manifest.get("status") == "COMPLETE" and manifest.get("fits_verified") == 50,
    }
    if not all(checks.values()):
        raise RuntimeError(f"primary ID lineage verification failed: {checks}")

    frames = {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            parts = []
            for outer in range(5):
                base = PRIMARY_OUT / objective / f"seed_{seed}"
                pred = base / f"outer_{outer}_predictions.csv"
                side = base / f"outer_{outer}_run.json"
                if not pred.is_file() or not side.is_file():
                    raise RuntimeError(f"missing primary ID prediction: {objective}/{seed}/{outer}")
                run = json.loads(side.read_text(encoding="utf-8"))
                if run["prediction_sha256"] != sha256_file(pred):
                    raise RuntimeError(f"primary ID prediction hash mismatch: {objective}/{seed}/{outer}")
                parts.append(pd.read_csv(pred))
            pooled = pd.concat(parts, ignore_index=True)
            if len(pooled) != 3000 or pooled.text_id.nunique() != 3000:
                raise RuntimeError(f"primary ID OOF incompleteness: {objective}/{seed}")
            for generator in GENERATORS:
                sub = pooled[pooled.generator.astype(str).eq(generator)].sort_values("text_id").reset_index(drop=True)
                if len(sub) != 600 or sub.pair_id.nunique() != 300:
                    raise RuntimeError(f"primary ID generator subset mismatch: {objective}/{seed}/{generator}")
                frames[(objective, seed, generator)] = sub
    return frames


def alignment(ood, primary):
    ref_keys = ["text_id", "pair_id", "label", "domain", "generator"]
    for seed in SEEDS:
        for generator in GENERATORS:
            base = ood[("ce", seed, generator)][ref_keys]
            for source, frames in (("ood_supcon", ood), ("id_ce", primary), ("id_supcon", primary)):
                objective = "supcon" if "supcon" in source else "ce"
                cur = frames[(objective, seed, generator)][ref_keys]
                if not cur.reset_index(drop=True).equals(base.reset_index(drop=True)):
                    raise RuntimeError(f"paired alignment failed: {source}/{seed}/{generator}")


def observed_tables(ood, primary):
    rows = []
    for objective in OBJECTIVES:
        for seed in SEEDS:
            for generator in GENERATORS:
                ood_m = metric_triplet(ood[(objective, seed, generator)])
                id_m = metric_triplet(primary[(objective, seed, generator)])
                row = {"objective": objective, "seed": seed, "heldout_generator": generator}
                for metric in METRICS:
                    row[f"id_{metric}"] = id_m[metric]
                    row[f"ood_{metric}"] = ood_m[metric]
                    row[f"degradation_{metric}"] = id_m[metric] - ood_m[metric]
                rows.append(row)
    per = pd.DataFrame(rows)

    macro_rows = []
    for (objective, seed), g in per.groupby(["objective", "seed"], sort=True):
        row = {"objective": objective, "seed": int(seed)}
        for metric in METRICS:
            for prefix in ("id_", "ood_", "degradation_"):
                row[f"macro_{prefix}{metric}"] = float(g[f"{prefix}{metric}"].mean())
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)

    delta_rows = []
    for seed in SEEDS:
        for generator in GENERATORS:
            ce = per[(per.objective == "ce") & (per.seed == seed) & (per.heldout_generator == generator)].iloc[0]
            sc = per[(per.objective == "supcon") & (per.seed == seed) & (per.heldout_generator == generator)].iloc[0]
            row = {"seed": seed, "heldout_generator": generator}
            for metric in METRICS:
                row[f"ood_supcon_minus_ce_{metric}"] = sc[f"ood_{metric}"] - ce[f"ood_{metric}"]
                row[f"id_supcon_minus_ce_{metric}"] = sc[f"id_{metric}"] - ce[f"id_{metric}"]
                row[f"relative_robustness_{metric}"] = (
                    ce[f"degradation_{metric}"] - sc[f"degradation_{metric}"]
                )
            delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)

    method_summary = []
    for objective, g in macro.groupby("objective", sort=True):
        for metric in METRICS:
            for quantity in ("id", "ood", "degradation"):
                values = g[f"macro_{quantity}_{metric}"].to_numpy(float)
                method_summary.append({
                    "objective": objective,
                    "quantity": quantity,
                    "metric": metric,
                    "mean_across_3_seeds": float(values.mean()),
                    "sd_across_3_seeds": float(values.std(ddof=1)),
                })

    generator_summary = []
    for (objective, generator), g in per.groupby(["objective", "heldout_generator"], sort=True):
        for metric in METRICS:
            generator_summary.append({
                "objective": objective,
                "heldout_generator": generator,
                "metric": metric,
                "mean_id_across_3_seeds": float(g[f"id_{metric}"].mean()),
                "mean_ood_across_3_seeds": float(g[f"ood_{metric}"].mean()),
                "mean_degradation_across_3_seeds": float(g[f"degradation_{metric}"].mean()),
                "sd_ood_across_3_seeds": float(g[f"ood_{metric}"].std(ddof=1)),
            })

    paired_summary = []
    for metric in METRICS:
        vals = deltas[f"ood_supcon_minus_ce_{metric}"].to_numpy(float)
        rob = deltas[f"relative_robustness_{metric}"].to_numpy(float)
        paired_summary.append({
            "metric": metric,
            "mean_ood_supcon_minus_ce_across_15_generator_seed_cells": float(vals.mean()),
            "sd_ood_supcon_minus_ce_across_15_generator_seed_cells": float(vals.std(ddof=1)),
            "mean_relative_robustness_across_15_generator_seed_cells": float(rob.mean()),
            "sd_relative_robustness_across_15_generator_seed_cells": float(rob.std(ddof=1)),
        })

    return (
        per,
        macro,
        deltas,
        pd.DataFrame(method_summary),
        pd.DataFrame(generator_summary),
        pd.DataFrame(paired_summary),
    )


def bootstrap(cfg, ood, primary):
    B = int(cfg["statistics"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))

    # Precompute pair -> row positions independently for each generator.
    pair_index = {}
    y = {}
    score = {}
    for generator in GENERATORS:
        ref = ood[("ce", SEEDS[0], generator)]
        ids = np.asarray(sorted(ref.pair_id.unique()))
        pair_index[generator] = (ids, {p: np.flatnonzero(ref.pair_id.to_numpy() == p) for p in ids})
        y[generator] = ref.label.astype(str).eq("AI").astype(int).to_numpy()
        for source_name, frames in (("ood", ood), ("id", primary)):
            for objective in OBJECTIVES:
                for seed in SEEDS:
                    score[(source_name, objective, seed, generator)] = (
                        frames[(objective, seed, generator)].raw_decision_score.to_numpy(float)
                    )

    overall_records = []
    generator_records = []

    for b in range(1, B + 1):
        overall_ood = {m: [] for m in METRICS}
        overall_rob = {m: [] for m in METRICS}
        per_generator = {g: {m: {"ood": [], "rob": []} for m in METRICS} for g in GENERATORS}

        for generator in GENERATORS:
            ids, idx_map = pair_index[generator]
            sampled = rng.choice(ids, size=len(ids), replace=True)
            idx = np.concatenate([idx_map[p] for p in sampled])
            yy = y[generator][idx]

            for seed in SEEDS:
                id_ce = metric_triplet(pd.DataFrame({
                    "label": np.where(yy == 1, "AI", "HUMAN"),
                    "raw_decision_score": score[("id", "ce", seed, generator)][idx],
                }))
                id_sc = metric_triplet(pd.DataFrame({
                    "label": np.where(yy == 1, "AI", "HUMAN"),
                    "raw_decision_score": score[("id", "supcon", seed, generator)][idx],
                }))
                od_ce = metric_triplet(pd.DataFrame({
                    "label": np.where(yy == 1, "AI", "HUMAN"),
                    "raw_decision_score": score[("ood", "ce", seed, generator)][idx],
                }))
                od_sc = metric_triplet(pd.DataFrame({
                    "label": np.where(yy == 1, "AI", "HUMAN"),
                    "raw_decision_score": score[("ood", "supcon", seed, generator)][idx],
                }))
                for metric in METRICS:
                    ood_delta = od_sc[metric] - od_ce[metric]
                    robustness = (id_ce[metric] - od_ce[metric]) - (id_sc[metric] - od_sc[metric])
                    overall_ood[metric].append(ood_delta)
                    overall_rob[metric].append(robustness)
                    per_generator[generator][metric]["ood"].append(ood_delta)
                    per_generator[generator][metric]["rob"].append(robustness)

        row = {"bootstrap_id": b}
        for metric in METRICS:
            row[f"ood_delta_{metric}"] = float(np.mean(overall_ood[metric]))
            row[f"relative_robustness_{metric}"] = float(np.mean(overall_rob[metric]))
        overall_records.append(row)

        for generator in GENERATORS:
            rowg = {"bootstrap_id": b, "heldout_generator": generator}
            for metric in METRICS:
                rowg[f"ood_delta_{metric}"] = float(np.mean(per_generator[generator][metric]["ood"]))
                rowg[f"relative_robustness_{metric}"] = float(np.mean(per_generator[generator][metric]["rob"]))
            generator_records.append(rowg)

    overall = pd.DataFrame(overall_records)
    bygen = pd.DataFrame(generator_records)

    summary = []
    for metric in METRICS:
        for quantity, col in (
            ("ood_supcon_minus_ce", f"ood_delta_{metric}"),
            ("relative_robustness", f"relative_robustness_{metric}"),
        ):
            vals = overall[col].to_numpy(float)
            summary.append({
                "scope": "macro_over_5_generators_and_3_seeds",
                "quantity": quantity,
                "metric": metric,
                "bootstrap_unit": "pair_id within held-out generator",
                "replicates": B,
                "bootstrap_seed": int(cfg["statistics"]["bootstrap_seed"]),
                "bootstrap_mean": float(vals.mean()),
                "ci95_lower": float(np.quantile(vals, 0.025)),
                "ci95_upper": float(np.quantile(vals, 0.975)),
                "inference_scope": cfg["statistics"]["inference_scope"],
            })

        for generator in GENERATORS:
            g = bygen[bygen.heldout_generator.eq(generator)]
            vals = g[f"ood_delta_{metric}"].to_numpy(float)
            summary.append({
                "scope": generator,
                "quantity": "ood_supcon_minus_ce",
                "metric": metric,
                "bootstrap_unit": "pair_id within held-out generator",
                "replicates": B,
                "bootstrap_seed": int(cfg["statistics"]["bootstrap_seed"]),
                "bootstrap_mean": float(vals.mean()),
                "ci95_lower": float(np.quantile(vals, 0.025)),
                "ci95_upper": float(np.quantile(vals, 0.975)),
                "inference_scope": cfg["statistics"]["inference_scope"],
            })
    return overall, bygen, pd.DataFrame(summary)


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    freeze, policy = verify_freeze(cfg)
    ood = load_ood(cfg, freeze, policy)
    primary = load_primary_id(cfg)
    alignment(ood, primary)

    per, macro, deltas, method_summary, generator_summary, paired_summary = observed_tables(ood, primary)
    boot, boot_gen, boot_summary = bootstrap(cfg, ood, primary)

    RESULTS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_generator_seed_metrics.csv": per,
        "macro_seed_metrics.csv": macro,
        "generator_seed_deltas.csv": deltas,
        "method_macro_summary.csv": method_summary,
        "generator_summary.csv": generator_summary,
        "paired_delta_summary.csv": paired_summary,
        "paired_pair_bootstrap_draws.csv": boot,
        "paired_pair_bootstrap_generator_draws.csv": boot_gen,
        "paired_pair_bootstrap_summary.csv": boot_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(RESULTS / name, index=False)

    manifest = {
        "schema_version": "combined-v2-generator-holdout-final-statistics-v1",
        "status": "COMPLETE",
        "unblinded": True,
        "fits_verified": 30,
        "heldout_generators": GENERATORS,
        "seeds": SEEDS,
        "statistics_config": cfg["statistics"],
        "config_sha256": sha256_file(CONFIG),
        "runner_sha256": sha256_file(RUNNER),
        "statistics_sha256": sha256_file(STATS),
        "freeze_sha256": sha256_file(FREEZE),
        "primary_reference_manifest_sha256": sha256_file(PRIMARY_MANIFEST),
        "outputs": {name: sha256_file(RESULTS / name) for name in outputs},
    }
    write_json(RESULTS / "final_statistics_manifest.json", manifest)
    print(json.dumps({
        "status": "COMPLETE",
        "fits_verified": 30,
        "results_directory": str(RESULTS),
        "results_are_now_unblinded": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
