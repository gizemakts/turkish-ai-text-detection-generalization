# -*- coding: utf-8 -*-
"""Prespecified one-time unblinding/statistics for DOMAIN_SHIFT_V1."""
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

CONFIG = ROOT / "configs" / "domain_shift.json"
RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/domain_shift/run_fit.py"
STATS = Path(__file__)
FREEZE = ROOT / "provenance" / "experiments" / "domain_shift" / "PREEXECUTION_FREEZE.json"
OUT = ROOT / "results" / "domain_shift"
RESULTS = OUT / "final_statistics"

PRIMARY_OUT = ROOT / "results" / "primary"
PRIMARY_RUNNER = ROOT / "src" / "turkish_ai_detection/experiments/primary/run_fit.py"
PRIMARY_CONFIG = ROOT / "configs" / "primary_evaluation.json"
PRIMARY_FREEZE = ROOT / "provenance" / "experiments" / "primary" / "PREEXECUTION_FREEZE.json"
PRIMARY_MANIFEST = PRIMARY_OUT / "final_statistics" / "final_statistics_manifest.json"

DIRECTIONS = {
    "abstract_to_newspaper": ("ABSTRACT", "NEWSPAPER"),
    "newspaper_to_abstract": ("NEWSPAPER", "ABSTRACT"),
}
SEEDS = [42, 314, 2026]
OBJECTIVES = ["ce", "supcon"]
METRICS = ["roc_auc", "average_precision", "standardized_pauc_003"]


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
        raise RuntimeError("domain-shift freeze missing")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_READY_TO_RUN":
        raise RuntimeError("invalid domain-shift freeze")
    policy = validate_current_migrated_reproduction(ROOT, "domain_shift", RUNNER)
    return freeze, policy


def load_cross_domain(freeze, policy):
    frames = {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            for direction, (_train_domain, test_domain) in DIRECTIONS.items():
                base = OUT / objective / f"seed_{seed}"
                pred = base / f"{direction}_predictions.csv"
                side = base / f"{direction}_run.json"
                if not pred.is_file() or not side.is_file():
                    raise RuntimeError(f"missing cross-domain fit: {objective}/{seed}/{direction}")
                run = json.loads(side.read_text(encoding="utf-8"))
                if run.get("status") != "COMPLETE":
                    raise RuntimeError(f"incomplete cross-domain fit: {objective}/{seed}/{direction}")
                if run["prediction_sha256"] != sha256_file(pred):
                    raise RuntimeError(f"prediction hash mismatch: {objective}/{seed}/{direction}")
                expected_config, expected_runner = expected_run_hashes(run, freeze, policy, config_key="config", runner_key="runner")
                if run["runner_sha256"] != expected_runner or run["config_sha256"] != expected_config:
                    raise RuntimeError(f"runner/config mismatch: {objective}/{seed}/{direction}")
                if run["preexecution_freeze_sha256"] != sha256_file(FREEZE):
                    raise RuntimeError(f"freeze mismatch: {objective}/{seed}/{direction}")

                f = pd.read_csv(pred).sort_values("text_id").reset_index(drop=True)
                expected_rows = 1000 if test_domain == "NEWSPAPER" else 2000
                expected_pairs = expected_rows // 2
                if len(f) != expected_rows or f.text_id.nunique() != expected_rows or f.pair_id.nunique() != expected_pairs:
                    raise RuntimeError(f"cross-domain shape mismatch: {objective}/{seed}/{direction}")
                if set(f.domain.astype(str)) != {test_domain}:
                    raise RuntimeError(f"target-domain mismatch: {objective}/{seed}/{direction}")
                if not np.isfinite(f.raw_decision_score.to_numpy(float)).all():
                    raise RuntimeError(f"non-finite score: {objective}/{seed}/{direction}")
                frames[(objective, seed, direction)] = f
    return frames


def load_primary_id(cfg):
    lineage = cfg["primary_rq1_lineage"]
    manifest = json.loads(PRIMARY_MANIFEST.read_text(encoding="utf-8"))
    checks = {
        "runner": sha256_file(PRIMARY_RUNNER) == lineage["primary_runner_sha256"],
        "config": sha256_file(PRIMARY_CONFIG) == lineage["primary_config_sha256"],
        "freeze": sha256_file(PRIMARY_FREEZE) == lineage["primary_preexecution_freeze_sha256"],
        "manifest": sha256_file(PRIMARY_MANIFEST) == lineage["primary_final_statistics_manifest_sha256"],
        "manifest_status": manifest.get("status") == "COMPLETE" and manifest.get("fits_verified") == 50,
    }
    if not all(checks.values()):
        raise RuntimeError(f"primary ID lineage failed: {checks}")

    pooled = {}
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
                    raise RuntimeError(f"primary prediction hash mismatch: {objective}/{seed}/{outer}")
                parts.append(pd.read_csv(pred))
            f = pd.concat(parts, ignore_index=True)
            if len(f) != 3000 or f.text_id.nunique() != 3000:
                raise RuntimeError(f"primary OOF incomplete: {objective}/{seed}")
            pooled[(objective, seed)] = f

    frames = {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            f = pooled[(objective, seed)]
            for direction, (_train_domain, test_domain) in DIRECTIONS.items():
                sub = f[f.domain.astype(str).eq(test_domain)].sort_values("text_id").reset_index(drop=True)
                expected_rows = 1000 if test_domain == "NEWSPAPER" else 2000
                if len(sub) != expected_rows or sub.pair_id.nunique() != expected_rows // 2:
                    raise RuntimeError(f"primary target-domain subset mismatch: {objective}/{seed}/{direction}")
                frames[(objective, seed, direction)] = sub
    return frames


def check_alignment(cross, primary):
    keys = ["text_id", "pair_id", "label", "domain", "generator"]
    for seed in SEEDS:
        for direction in DIRECTIONS:
            ref = cross[("ce", seed, direction)][keys].reset_index(drop=True)
            for source, frames in (("cross_supcon", cross), ("id_ce", primary), ("id_supcon", primary)):
                obj = "supcon" if "supcon" in source else "ce"
                cur = frames[(obj, seed, direction)][keys].reset_index(drop=True)
                if not cur.equals(ref):
                    raise RuntimeError(f"paired target-domain alignment failed: {source}/{seed}/{direction}")


def observed_tables(cross, primary):
    rows = []
    for objective in OBJECTIVES:
        for seed in SEEDS:
            for direction, (train_domain, test_domain) in DIRECTIONS.items():
                cross_m = metric_triplet(cross[(objective, seed, direction)])
                id_m = metric_triplet(primary[(objective, seed, direction)])
                row = {
                    "objective": objective,
                    "seed": seed,
                    "direction": direction,
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                }
                for metric in METRICS:
                    row[f"id_{metric}"] = id_m[metric]
                    row[f"cross_domain_{metric}"] = cross_m[metric]
                    row[f"degradation_{metric}"] = id_m[metric] - cross_m[metric]
                rows.append(row)
    per = pd.DataFrame(rows)

    macro_rows = []
    for (objective, seed), g in per.groupby(["objective", "seed"], sort=True):
        row = {"objective": objective, "seed": int(seed)}
        for metric in METRICS:
            for prefix in ("id_", "cross_domain_", "degradation_"):
                row[f"macro_{prefix}{metric}"] = float(g[f"{prefix}{metric}"].mean())
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)

    delta_rows = []
    for seed in SEEDS:
        for direction in DIRECTIONS:
            ce = per[(per.objective == "ce") & (per.seed == seed) & (per.direction == direction)].iloc[0]
            sc = per[(per.objective == "supcon") & (per.seed == seed) & (per.direction == direction)].iloc[0]
            row = {"seed": seed, "direction": direction}
            for metric in METRICS:
                row[f"cross_supcon_minus_ce_{metric}"] = sc[f"cross_domain_{metric}"] - ce[f"cross_domain_{metric}"]
                row[f"id_supcon_minus_ce_{metric}"] = sc[f"id_{metric}"] - ce[f"id_{metric}"]
                row[f"relative_robustness_{metric}"] = ce[f"degradation_{metric}"] - sc[f"degradation_{metric}"]
            delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)

    method_summary = []
    for objective, g in macro.groupby("objective", sort=True):
        for metric in METRICS:
            for quantity in ("id", "cross_domain", "degradation"):
                vals = g[f"macro_{quantity}_{metric}"].to_numpy(float)
                method_summary.append({
                    "objective": objective,
                    "quantity": quantity,
                    "metric": metric,
                    "mean_across_3_seeds": float(vals.mean()),
                    "sd_across_3_seeds": float(vals.std(ddof=1)),
                })

    direction_summary = []
    for (objective, direction), g in per.groupby(["objective", "direction"], sort=True):
        for metric in METRICS:
            direction_summary.append({
                "objective": objective,
                "direction": direction,
                "metric": metric,
                "mean_id_across_3_seeds": float(g[f"id_{metric}"].mean()),
                "mean_cross_domain_across_3_seeds": float(g[f"cross_domain_{metric}"].mean()),
                "mean_degradation_across_3_seeds": float(g[f"degradation_{metric}"].mean()),
                "sd_cross_domain_across_3_seeds": float(g[f"cross_domain_{metric}"].std(ddof=1)),
            })

    paired_summary = []
    for metric in METRICS:
        d = deltas[f"cross_supcon_minus_ce_{metric}"].to_numpy(float)
        r = deltas[f"relative_robustness_{metric}"].to_numpy(float)
        paired_summary.append({
            "metric": metric,
            "mean_cross_supcon_minus_ce_across_6_direction_seed_cells": float(d.mean()),
            "sd_cross_supcon_minus_ce_across_6_direction_seed_cells": float(d.std(ddof=1)),
            "mean_relative_robustness_across_6_direction_seed_cells": float(r.mean()),
            "sd_relative_robustness_across_6_direction_seed_cells": float(r.std(ddof=1)),
        })
    return per, macro, deltas, pd.DataFrame(method_summary), pd.DataFrame(direction_summary), pd.DataFrame(paired_summary)


def metric_from(y, score):
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "standardized_pauc_003": float(_pauc(y, score)),
    }


def bootstrap(cfg, cross, primary):
    B = int(cfg["statistics"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))

    pair_index, y, score = {}, {}, {}
    for direction in DIRECTIONS:
        ref = cross[("ce", SEEDS[0], direction)]
        ids = np.asarray(sorted(ref.pair_id.unique()))
        idx_map = {p: np.flatnonzero(ref.pair_id.to_numpy() == p) for p in ids}
        if any(len(v) != 2 for v in idx_map.values()):
            raise RuntimeError(f"target-domain pair does not contain exactly two rows: {direction}")
        pair_index[direction] = (ids, idx_map)
        y[direction] = ref.label.astype(str).eq("AI").astype(int).to_numpy()
        for source_name, frames in (("cross", cross), ("id", primary)):
            for objective in OBJECTIVES:
                for seed in SEEDS:
                    score[(source_name, objective, seed, direction)] = (
                        frames[(objective, seed, direction)].raw_decision_score.to_numpy(float)
                    )

    overall_rows, direction_rows = [], []
    for b in range(1, B + 1):
        overall_delta = {m: [] for m in METRICS}
        overall_rob = {m: [] for m in METRICS}
        direction_acc = {
            d: {m: {"delta": [], "rob": []} for m in METRICS}
            for d in DIRECTIONS
        }

        for direction in DIRECTIONS:
            ids, idx_map = pair_index[direction]
            sampled = rng.choice(ids, size=len(ids), replace=True)
            idx = np.concatenate([idx_map[p] for p in sampled])
            yy = y[direction][idx]

            for seed in SEEDS:
                id_ce = metric_from(yy, score[("id", "ce", seed, direction)][idx])
                id_sc = metric_from(yy, score[("id", "supcon", seed, direction)][idx])
                cd_ce = metric_from(yy, score[("cross", "ce", seed, direction)][idx])
                cd_sc = metric_from(yy, score[("cross", "supcon", seed, direction)][idx])

                for metric in METRICS:
                    delta = cd_sc[metric] - cd_ce[metric]
                    robustness = (id_ce[metric] - cd_ce[metric]) - (id_sc[metric] - cd_sc[metric])
                    overall_delta[metric].append(delta)
                    overall_rob[metric].append(robustness)
                    direction_acc[direction][metric]["delta"].append(delta)
                    direction_acc[direction][metric]["rob"].append(robustness)

        row = {"bootstrap_id": b}
        for metric in METRICS:
            row[f"cross_delta_{metric}"] = float(np.mean(overall_delta[metric]))
            row[f"relative_robustness_{metric}"] = float(np.mean(overall_rob[metric]))
        overall_rows.append(row)

        for direction in DIRECTIONS:
            rd = {"bootstrap_id": b, "direction": direction}
            for metric in METRICS:
                rd[f"cross_delta_{metric}"] = float(np.mean(direction_acc[direction][metric]["delta"]))
                rd[f"relative_robustness_{metric}"] = float(np.mean(direction_acc[direction][metric]["rob"]))
            direction_rows.append(rd)

    overall = pd.DataFrame(overall_rows)
    bydir = pd.DataFrame(direction_rows)

    summary = []
    for metric in METRICS:
        for quantity, col in (
            ("cross_domain_supcon_minus_ce", f"cross_delta_{metric}"),
            ("relative_robustness", f"relative_robustness_{metric}"),
        ):
            vals = overall[col].to_numpy(float)
            summary.append({
                "scope": "macro_over_2_directions_and_3_seeds",
                "quantity": quantity,
                "metric": metric,
                "bootstrap_unit": "pair_id within target domain",
                "replicates": B,
                "bootstrap_seed": int(cfg["statistics"]["bootstrap_seed"]),
                "bootstrap_mean": float(vals.mean()),
                "ci95_lower": float(np.quantile(vals, 0.025)),
                "ci95_upper": float(np.quantile(vals, 0.975)),
                "inference_scope": cfg["statistics"]["inference_scope"],
            })

        for direction in DIRECTIONS:
            g = bydir[bydir.direction.eq(direction)]
            for quantity, col in (
                ("cross_domain_supcon_minus_ce", f"cross_delta_{metric}"),
                ("relative_robustness", f"relative_robustness_{metric}"),
            ):
                vals = g[col].to_numpy(float)
                summary.append({
                    "scope": direction,
                    "quantity": quantity,
                    "metric": metric,
                    "bootstrap_unit": "pair_id within target domain",
                    "replicates": B,
                    "bootstrap_seed": int(cfg["statistics"]["bootstrap_seed"]),
                    "bootstrap_mean": float(vals.mean()),
                    "ci95_lower": float(np.quantile(vals, 0.025)),
                    "ci95_upper": float(np.quantile(vals, 0.975)),
                    "inference_scope": cfg["statistics"]["inference_scope"],
                })
    return overall, bydir, pd.DataFrame(summary)


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    freeze, policy = verify_freeze(cfg)
    cross = load_cross_domain(freeze, policy)
    primary = load_primary_id(cfg)
    check_alignment(cross, primary)

    per, macro, deltas, method_summary, direction_summary, paired_summary = observed_tables(cross, primary)
    boot, boot_dir, boot_summary = bootstrap(cfg, cross, primary)

    RESULTS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_direction_seed_metrics.csv": per,
        "macro_seed_metrics.csv": macro,
        "direction_seed_deltas.csv": deltas,
        "method_macro_summary.csv": method_summary,
        "direction_summary.csv": direction_summary,
        "paired_delta_summary.csv": paired_summary,
        "paired_pair_bootstrap_draws.csv": boot,
        "paired_pair_bootstrap_direction_draws.csv": boot_dir,
        "paired_pair_bootstrap_summary.csv": boot_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(RESULTS / name, index=False)

    manifest = {
        "schema_version": "combined-v2-domain-shift-final-statistics-v1",
        "status": "COMPLETE",
        "unblinded": True,
        "fits_verified": 12,
        "directions": list(DIRECTIONS),
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
        "fits_verified": 12,
        "results_directory": str(RESULTS),
        "results_are_now_unblinded": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
