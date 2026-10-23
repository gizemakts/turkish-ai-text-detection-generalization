# -*- coding: utf-8 -*-
"""Prespecified unblinding/statistics for COMBINED_V2_PRIMARY_ABLATION_V1.

Run only after all 50 confirmatory fits have completed.
Uses raw decision scores only; no calibration and no threshold optimization.
"""
from __future__ import annotations

import json
import sys
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

from turkish_ai_detection.data.study import write_json  # noqa: E402
from turkish_ai_detection.evaluation.metrics import _pauc  # noqa: E402
from turkish_ai_detection.experiments.primary.integrity import (  # noqa: E402
    load_active_folds, verify_fit_integrity,
)

CONFIG = ROOT / "configs" / "primary_evaluation.json"
OUT = ROOT / "results" / "primary"
RESULTS = OUT / "final_statistics"


def metric_triplet(y, score):
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "standardized_pauc_003": float(_pauc(y, score)),
    }


def load_all():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = load_active_folds(ROOT, cfg)
    frames = {}
    sidecars = {}
    for objective in ("ce", "supcon"):
        for seed in cfg["seeds"]:
            for outer in cfg["outer_folds"]:
                key = (objective, int(seed), int(outer))
                base = OUT / objective / f"seed_{seed}"
                pred_path = base / f"outer_{outer}_predictions.csv"
                run_path = base / f"outer_{outer}_run.json"
                if not pred_path.is_file() or not run_path.is_file():
                    raise RuntimeError(f"missing confirmatory fit: {key}")
                run = json.loads(run_path.read_text(encoding="utf-8"))
                verify_fit_integrity(
                    root=ROOT, cfg=cfg, config_path=CONFIG, folds=folds,
                    objective=objective, seed=int(seed), outer=int(outer),
                    sidecar=run, prediction_path=pred_path,
                )
                frame = pd.read_csv(pred_path)
                required = {
                    "text_id", "pair_id", "label", "domain", "generator",
                    "outer_fold", "seed", "objective", "condition",
                    "raw_decision_score",
                }
                if not required <= set(frame.columns):
                    raise RuntimeError(f"prediction schema mismatch: {key}")
                if len(frame) != 600 or frame.text_id.nunique() != 600:
                    raise RuntimeError(f"prediction row/ID mismatch: {key}")
                if set(frame.outer_fold.astype(int)) != {int(outer)}:
                    raise RuntimeError(f"outer identity mismatch: {key}")
                if set(frame.seed.astype(int)) != {int(seed)}:
                    raise RuntimeError(f"seed identity mismatch: {key}")
                if set(frame.objective.astype(str)) != {objective}:
                    raise RuntimeError(f"objective identity mismatch: {key}")
                if not np.isfinite(frame.raw_decision_score.to_numpy(float)).all():
                    raise RuntimeError(f"non-finite score: {key}")
                frames[key] = frame
                sidecars[key] = run
    return cfg, frames, sidecars


def pooled_by_seed(cfg, frames):
    rows = []
    pooled = {}
    for objective in ("ce", "supcon"):
        for seed in cfg["seeds"]:
            parts = [frames[(objective, int(seed), int(outer))] for outer in cfg["outer_folds"]]
            frame = pd.concat(parts, ignore_index=True).sort_values("text_id").reset_index(drop=True)
            if len(frame) != 3000 or frame.text_id.nunique() != 3000:
                raise RuntimeError(f"OOF completeness failed: {objective}, seed={seed}")
            pooled[(objective, int(seed))] = frame
            y = frame.label.astype(str).eq("AI").astype(int).to_numpy()
            score = frame.raw_decision_score.to_numpy(float)
            values = metric_triplet(y, score)
            rows.append({"objective": objective, "seed": int(seed), **values})
    return pd.DataFrame(rows), pooled


def check_alignment(cfg, pooled):
    reference = pooled[("ce", int(cfg["seeds"][0]))][["text_id", "pair_id", "label"]]
    reference = reference.reset_index(drop=True)
    for key, frame in pooled.items():
        aligned = frame[["text_id", "pair_id", "label"]].reset_index(drop=True)
        if not aligned.equals(reference):
            raise RuntimeError(f"paired alignment failed: {key}")


def seed_summary(seed_metrics):
    rows = []
    for objective, group in seed_metrics.groupby("objective", sort=True):
        for metric in ("roc_auc", "average_precision", "standardized_pauc_003"):
            values = group[metric].to_numpy(float)
            rows.append({
                "objective": objective,
                "metric": metric,
                "mean_across_seeds": float(values.mean()),
                "sd_across_seeds": float(values.std(ddof=1)),
                "min_seed_value": float(values.min()),
                "max_seed_value": float(values.max()),
                "n_seeds": int(len(values)),
            })

    ce = seed_metrics[seed_metrics.objective.eq("ce")].sort_values("seed")
    sc = seed_metrics[seed_metrics.objective.eq("supcon")].sort_values("seed")
    if ce.seed.tolist() != sc.seed.tolist():
        raise RuntimeError("seed pairing failed")
    deltas = []
    for metric in ("roc_auc", "average_precision", "standardized_pauc_003"):
        values = sc[metric].to_numpy(float) - ce[metric].to_numpy(float)
        deltas.append({
            "comparison": "supcon_minus_ce",
            "metric": metric,
            "mean_paired_delta_across_seeds": float(values.mean()),
            "sd_paired_delta_across_seeds": float(values.std(ddof=1)),
            "n_seeds": int(len(values)),
        })
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def pair_bootstrap(cfg, pooled):
    B = int(cfg["statistics"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))
    seed_values = [int(x) for x in cfg["seeds"]]

    ref = pooled[("ce", seed_values[0])]
    pair_ids = np.asarray(sorted(ref.pair_id.unique()))
    index_map = {
        pair: np.flatnonzero(ref.pair_id.to_numpy() == pair)
        for pair in pair_ids
    }
    if len(pair_ids) != 1500:
        raise RuntimeError("expected exactly 1500 bootstrap pair clusters")
    if any(len(v) != 2 for v in index_map.values()):
        raise RuntimeError("each bootstrap pair must contain exactly two rows")

    # All pooled frames are already sorted/aligned by text_id.
    y_full = ref.label.astype(str).eq("AI").astype(int).to_numpy()
    scores = {
        (objective, seed): pooled[(objective, seed)].raw_decision_score.to_numpy(float)
        for objective in ("ce", "supcon")
        for seed in seed_values
    }

    records = []
    for b in range(1, B + 1):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        idx = np.concatenate([index_map[pair] for pair in sampled])
        y = y_full[idx]

        seed_delta = {
            "roc_auc": [],
            "average_precision": [],
            "standardized_pauc_003": [],
        }
        for seed in seed_values:
            ce_m = metric_triplet(y, scores[("ce", seed)][idx])
            sc_m = metric_triplet(y, scores[("supcon", seed)][idx])
            for metric in seed_delta:
                seed_delta[metric].append(sc_m[metric] - ce_m[metric])

        records.append({
            "bootstrap_id": b,
            **{
                f"delta_{metric}": float(np.mean(values))
                for metric, values in seed_delta.items()
            },
        })

    draws = pd.DataFrame(records)
    summary = []
    for metric in ("roc_auc", "average_precision", "standardized_pauc_003"):
        values = draws[f"delta_{metric}"].to_numpy(float)
        summary.append({
            "comparison": "supcon_minus_ce",
            "metric": metric,
            "bootstrap_unit": "pair_id",
            "replicates": B,
            "bootstrap_seed": int(cfg["statistics"]["bootstrap_seed"]),
            "observed_bootstrap_mean_delta": float(values.mean()),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
        })
    return draws, pd.DataFrame(summary)


def main() -> int:
    cfg, frames, sidecars = load_all()
    seed_metrics, pooled = pooled_by_seed(cfg, frames)
    check_alignment(cfg, pooled)
    method_summary, paired_seed_delta = seed_summary(seed_metrics)
    draws, bootstrap_summary = pair_bootstrap(cfg, pooled)

    RESULTS.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(RESULTS / "per_seed_pooled_metrics.csv", index=False)
    method_summary.to_csv(RESULTS / "method_seed_summary.csv", index=False)
    paired_seed_delta.to_csv(RESULTS / "paired_seed_delta_summary.csv", index=False)
    draws.to_csv(RESULTS / "paired_pair_bootstrap_draws.csv", index=False)
    bootstrap_summary.to_csv(RESULTS / "paired_pair_bootstrap_summary.csv", index=False)

    manifest = {
        "schema_version": "combined-v2-primary-ablation-final-statistics-v1",
        "status": "COMPLETE",
        "unblinded": True,
        "fits_verified": 50,
        "seeds": cfg["seeds"],
        "outer_folds": cfg["outer_folds"],
        "primary_metric": cfg["primary_metric"],
        "statistics_config": cfg["statistics"],
        "outputs": {
            path.name: {"size_bytes": path.stat().st_size}
            for path in RESULTS.iterdir()
            if path.is_file()
        },
    }
    write_json(RESULTS / "final_statistics_manifest.json", manifest)
    print(json.dumps({
        "status": "COMPLETE",
        "fits_verified": 50,
        "results_directory": str(RESULTS),
        "results_are_now_unblinded": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
