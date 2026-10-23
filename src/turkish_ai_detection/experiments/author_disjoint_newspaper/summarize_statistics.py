"""Frozen author-disjoint statistics. Refuses to run without explicit --execute."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import CONFIG, FREEZE, OBJECTIVES, OUT, OUTERS, SEEDS, require, sha256_file
from .validate import validate_predictions


RESULTS = OUT / "final_statistics"


def load_predictions() -> dict[tuple[str, int, int], pd.DataFrame]:
    validate_predictions()
    frames = {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            for outer in OUTERS:
                path = OUT / objective / f"seed_{seed}" / f"fold_{outer}_predictions.csv"
                frames[(objective, seed, outer)] = pd.read_csv(path)
    return frames


def auc(frame: pd.DataFrame) -> float:
    y = frame.label.astype(str).str.upper().eq("AI").astype(int).to_numpy()
    return float(roc_auc_score(y, frame.raw_decision_score.to_numpy(float)))


def compute(frames):
    fold_rows, seed_rows, pooled = [], [], {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            parts = []
            for outer in OUTERS:
                frame = frames[(objective, seed, outer)]
                fold_rows.append({"objective": objective, "seed": seed, "author_fold": outer,
                                  "test_pairs": 100, "roc_auc": auc(frame)})
                parts.append(frame)
            combined = pd.concat(parts, ignore_index=True).sort_values("text_id").reset_index(drop=True)
            require(len(combined) == 1000 and combined.pair_id.nunique() == 500, "pooled completeness failed")
            pooled[(objective, seed)] = combined
            seed_rows.append({"objective": objective, "seed": seed, "test_pairs": 500,
                              "pooled_author_disjoint_roc_auc": auc(combined)})
    fold_metrics = pd.DataFrame(fold_rows)
    seed_metrics = pd.DataFrame(seed_rows)
    method_rows = []
    for objective, group in seed_metrics.groupby("objective", sort=True):
        values = group.pooled_author_disjoint_roc_auc.to_numpy(float)
        method_rows.append({"objective": objective, "mean_across_seeds": float(values.mean()),
                            "sd_across_seeds": float(values.std(ddof=1)), "n_seeds": len(values)})
    ce = seed_metrics[seed_metrics.objective.eq("ce")].sort_values("seed")
    sc = seed_metrics[seed_metrics.objective.eq("supcon")].sort_values("seed")
    require(ce.seed.tolist() == sc.seed.tolist(), "seed pairing failed")
    delta = sc.pooled_author_disjoint_roc_auc.to_numpy() - ce.pooled_author_disjoint_roc_auc.to_numpy()
    paired_seed = pd.DataFrame({"seed": ce.seed.to_numpy(int), "supcon_minus_ce_roc_auc": delta})
    return fold_metrics, seed_metrics, pd.DataFrame(method_rows), paired_seed, pooled


def bootstrap(pooled, replicates: int, bootstrap_seed: int):
    reference = pooled[("ce", SEEDS[0])]
    pair_ids = np.asarray(sorted(reference.pair_id.astype(str).unique()))
    require(len(pair_ids) == 500, "bootstrap requires 500 pairs")
    by_key = {key: frame.set_index("text_id", drop=False) for key, frame in pooled.items()}
    pair_text_ids = {pair: reference.loc[reference.pair_id.astype(str).eq(pair), "text_id"].tolist() for pair in pair_ids}
    require(all(len(x) == 2 for x in pair_text_ids.values()), "bootstrap pair size mismatch")
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for b in range(1, replicates + 1):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        ids = [text_id for pair in sampled for text_id in pair_text_ids[pair]]
        ce_auc, sc_auc = [], []
        for seed in SEEDS:
            ce_frame = by_key[("ce", seed)].loc[ids]
            sc_frame = by_key[("supcon", seed)].loc[ids]
            ce_auc.append(auc(ce_frame))
            sc_auc.append(auc(sc_frame))
        ce_mean, sc_mean = float(np.mean(ce_auc)), float(np.mean(sc_auc))
        rows.append({"bootstrap_id": b, "ce_mean_auc": ce_mean, "supcon_mean_auc": sc_mean,
                     "supcon_minus_ce": sc_mean - ce_mean})
    draws = pd.DataFrame(rows)
    summary = []
    for name in ("ce_mean_auc", "supcon_mean_auc", "supcon_minus_ce"):
        values = draws[name].to_numpy(float)
        summary.append({"estimand": name, "bootstrap_unit": "pair_id", "replicates": replicates,
                        "bootstrap_seed": bootstrap_seed, "bootstrap_mean": float(values.mean()),
                        "ci95_lower": float(np.quantile(values, 0.025)),
                        "ci95_upper": float(np.quantile(values, 0.975))})
    return draws, pd.DataFrame(summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("DRY_RUN_ONLY: add --execute after all 30 prediction runs pass validation")
        return 0
    if RESULTS.exists() and any(RESULTS.iterdir()):
        manifest = RESULTS / "FINAL_STATISTICS_MANIFEST.json"
        require(manifest.is_file(), "partial statistics directory exists; refusing overwrite")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        require(saved.get("status") == "COMPLETE", "existing statistics are not complete")
        print(json.dumps({"status": "SKIPPED_VERIFIED_COMPLETE", "path": str(manifest)}))
        return 0
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frames = load_predictions()
    fold_metrics, seed_metrics, method_summary, paired_seed, pooled = compute(frames)
    settings = config["statistics"]
    draws, bootstrap_summary = bootstrap(pooled, int(settings["bootstrap_replicates"]), int(settings["bootstrap_seed"]))
    RESULTS.mkdir(parents=True, exist_ok=False)
    outputs = {
        "fold_metrics.csv": fold_metrics,
        "seed_metrics.csv": seed_metrics,
        "method_summary.csv": method_summary,
        "paired_seed_differences.csv": paired_seed,
        "paired_pair_bootstrap_draws.csv": draws,
        "paired_pair_bootstrap_summary.csv": bootstrap_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(RESULTS / name, index=False)
    manifest = {
        "schema_version": "author-disjoint-newspaper-final-statistics-v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_runs_verified": 30,
        "freeze_sha256": sha256_file(FREEZE),
        "config_sha256": sha256_file(CONFIG),
        "statistics": settings,
        "outputs": {name: sha256_file(RESULTS / name) for name in outputs},
    }
    path = RESULTS / "FINAL_STATISTICS_MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "manifest": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

