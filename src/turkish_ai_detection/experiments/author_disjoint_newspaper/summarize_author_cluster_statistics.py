"""Author-cluster bootstrap and descriptive author-disjoint ROC-AUC statistics."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import CONFIG, FREEZE, OBJECTIVES, OUT, OUTERS, PROV, ROOT, SEEDS, require, sha256_file
from .validate import validate_predictions


STATISTICS_CONFIG = ROOT / "configs" / "author_disjoint_statistics.json"
RESULTS = OUT / "final_statistics"


def load_statistics_config() -> dict:
    require(STATISTICS_CONFIG.is_file(), "author-disjoint statistical settings are required")
    config = json.loads(STATISTICS_CONFIG.read_text(encoding="utf-8"))
    for key in ("primary_bootstrap", "secondary_bootstrap"):
        settings = config[key]
        require(int(settings["replicates"]) == 5000 and int(settings["seed"]) == 2602,
                "bootstrap settings differ from the study protocol")
    return config


def auc(frame: pd.DataFrame) -> float:
    y = frame.label.astype(str).str.upper().eq("AI").astype(int).to_numpy()
    return float(roc_auc_score(y, frame.raw_decision_score.to_numpy(float)))


def load_predictions() -> dict[tuple[str, int, int], pd.DataFrame]:
    validate_predictions()
    frames = {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            for outer in OUTERS:
                path = OUT / objective / f"seed_{seed}" / f"fold_{outer}_predictions.csv"
                frames[(objective, seed, outer)] = pd.read_csv(path)
    return frames


def compute_point_tables(frames):
    fold_rows, author_rows, seed_rows, pooled = [], [], [], {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            parts = []
            for outer in OUTERS:
                frame = frames[(objective, seed, outer)]
                fold_rows.append({"objective": objective, "seed": seed, "author_fold": outer,
                                  "test_pairs": 100, "roc_auc": auc(frame)})
                for author, group in frame.groupby("author", sort=True):
                    require(group.pair_id.nunique() == 50 and len(group) == 100, "author cell count mismatch")
                    author_rows.append({"objective": objective, "seed": seed, "author": author,
                                        "author_fold": outer, "test_pairs": 50, "roc_auc": auc(group)})
                parts.append(frame)
            combined = pd.concat(parts, ignore_index=True).sort_values("text_id").reset_index(drop=True)
            require(len(combined) == 1000 and combined.pair_id.nunique() == 500, "pooled completeness failed")
            require(combined.author.nunique() == 10, "pooled author count failed")
            pooled[(objective, seed)] = combined
            seed_rows.append({"objective": objective, "seed": seed, "test_pairs": 500,
                              "authors": 10, "pooled_author_disjoint_roc_auc": auc(combined)})

    fold_metrics = pd.DataFrame(fold_rows)
    author_seed_metrics = pd.DataFrame(author_rows)
    seed_metrics = pd.DataFrame(seed_rows)
    author_summary = author_seed_metrics.groupby(["objective", "author", "author_fold"], as_index=False).agg(
        mean_across_seeds=("roc_auc", "mean"), sd_across_seeds=("roc_auc", "std"), n_seeds=("seed", "nunique"))
    method_summary = seed_metrics.groupby("objective", as_index=False).agg(
        mean_across_seeds=("pooled_author_disjoint_roc_auc", "mean"),
        sd_across_seeds=("pooled_author_disjoint_roc_auc", "std"), n_seeds=("seed", "nunique"))
    ce = seed_metrics[seed_metrics.objective.eq("ce")].sort_values("seed")
    sc = seed_metrics[seed_metrics.objective.eq("supcon")].sort_values("seed")
    require(ce.seed.tolist() == sc.seed.tolist(), "seed pairing failed")
    paired_seed = pd.DataFrame({"seed": ce.seed.to_numpy(int),
                                "supcon_minus_ce_roc_auc": sc.pooled_author_disjoint_roc_auc.to_numpy() - ce.pooled_author_disjoint_roc_auc.to_numpy()})
    return fold_metrics, author_seed_metrics, author_summary, seed_metrics, method_summary, paired_seed, pooled


def verify_paired_alignment(pooled) -> pd.DataFrame:
    reference = pooled[("ce", SEEDS[0])][["text_id", "pair_id", "label", "author", "author_fold"]].reset_index(drop=True)
    for key, frame in pooled.items():
        current = frame[["text_id", "pair_id", "label", "author", "author_fold"]].reset_index(drop=True)
        require(current.equals(reference), f"paired author/fold/observation alignment failed: {key}")
    return reference


def bootstrap_summary(draws: pd.DataFrame, label: str, unit: str, replicates: int, seed: int) -> pd.DataFrame:
    rows = []
    for name in ("ce_mean_auc", "supcon_mean_auc", "supcon_minus_ce"):
        values = draws[name].to_numpy(float)
        rows.append({"analysis_label": label, "estimand": name, "top_level_sampling_unit": unit,
                     "replicates": replicates, "bootstrap_seed": seed,
                     "bootstrap_mean": float(values.mean()), "ci95_lower": float(np.quantile(values, 0.025)),
                     "ci95_upper": float(np.quantile(values, 0.975))})
    return pd.DataFrame(rows)


def author_cluster_bootstrap(pooled, settings):
    reference = verify_paired_alignment(pooled)
    authors = np.asarray(sorted(reference.author.astype(str).unique(), key=lambda x: x.encode("utf-8")))
    require(len(authors) == 10, "primary bootstrap requires exactly 10 author clusters")
    text_ids = {author: reference.loc[reference.author.astype(str).eq(author), "text_id"].tolist() for author in authors}
    require(all(len(ids) == 100 for ids in text_ids.values()), "author cluster size mismatch")
    indexed = {key: frame.set_index("text_id", drop=False) for key, frame in pooled.items()}
    rng = np.random.default_rng(int(settings["seed"]))
    rows = []
    for replicate in range(1, int(settings["replicates"]) + 1):
        sampled_authors = rng.choice(authors, size=10, replace=True)
        ids = [text_id for author in sampled_authors for text_id in text_ids[author]]
        ce_values, supcon_values = [], []
        for seed in SEEDS:
            ce_values.append(auc(indexed[("ce", seed)].loc[ids]))
            supcon_values.append(auc(indexed[("supcon", seed)].loc[ids]))
        ce_mean, supcon_mean = float(np.mean(ce_values)), float(np.mean(supcon_values))
        rows.append({"bootstrap_id": replicate, "sampled_author_count": 10,
                     "unique_sampled_authors": int(len(set(sampled_authors.tolist()))),
                     "ce_mean_auc": ce_mean, "supcon_mean_auc": supcon_mean,
                     "supcon_minus_ce": supcon_mean - ce_mean})
    draws = pd.DataFrame(rows)
    return draws, bootstrap_summary(draws, settings["label"], "author", int(settings["replicates"]), int(settings["seed"]))


def secondary_pair_bootstrap(pooled, settings):
    reference = verify_paired_alignment(pooled)
    pair_ids = np.asarray(sorted(reference.pair_id.astype(str).unique()))
    require(len(pair_ids) == 500, "secondary bootstrap requires 500 pairs")
    text_ids = {pair: reference.loc[reference.pair_id.astype(str).eq(pair), "text_id"].tolist() for pair in pair_ids}
    require(all(len(ids) == 2 for ids in text_ids.values()), "pair cluster size mismatch")
    indexed = {key: frame.set_index("text_id", drop=False) for key, frame in pooled.items()}
    rng = np.random.default_rng(int(settings["seed"]))
    rows = []
    for replicate in range(1, int(settings["replicates"]) + 1):
        sampled_pairs = rng.choice(pair_ids, size=500, replace=True)
        ids = [text_id for pair in sampled_pairs for text_id in text_ids[pair]]
        ce_values = [auc(indexed[("ce", seed)].loc[ids]) for seed in SEEDS]
        supcon_values = [auc(indexed[("supcon", seed)].loc[ids]) for seed in SEEDS]
        ce_mean, supcon_mean = float(np.mean(ce_values)), float(np.mean(supcon_values))
        rows.append({"bootstrap_id": replicate, "ce_mean_auc": ce_mean,
                     "supcon_mean_auc": supcon_mean, "supcon_minus_ce": supcon_mean - ce_mean})
    draws = pd.DataFrame(rows)
    return draws, bootstrap_summary(draws, settings["label"], "pair_id", int(settings["replicates"]), int(settings["seed"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    statistics_config = load_statistics_config()
    if not args.execute:
        print("Author-cluster statistics: pass --execute to compute results from the 30 model realizations")
        return 0
    if RESULTS.exists() and any(RESULTS.iterdir()):
        manifest_path = RESULTS / "FINAL_STATISTICS_MANIFEST.json"
        require(manifest_path.is_file(), "partial statistics directory exists; refusing overwrite")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(saved.get("status") == "COMPLETE" and "primary_uncertainty" in saved, "existing statistics are incomplete")
        print(json.dumps({"status": "SKIPPED_VERIFIED_COMPLETE", "path": str(manifest_path)}))
        return 0

    frames = load_predictions()
    tables = compute_point_tables(frames)
    fold_metrics, author_seed_metrics, author_summary, seed_metrics, method_summary, paired_seed, pooled = tables
    verify_paired_alignment(pooled)
    primary_draws, primary_summary = author_cluster_bootstrap(pooled, statistics_config["primary_bootstrap"])
    secondary_draws, secondary_summary = secondary_pair_bootstrap(pooled, statistics_config["secondary_bootstrap"])
    RESULTS.mkdir(parents=True, exist_ok=False)
    outputs = {
        "fold_metrics.csv": fold_metrics,
        "per_author_seed_metrics.csv": author_seed_metrics,
        "per_author_method_summary.csv": author_summary,
        "seed_metrics.csv": seed_metrics,
        "method_summary.csv": method_summary,
        "paired_seed_differences.csv": paired_seed,
        "primary_author_cluster_bootstrap_draws.csv": primary_draws,
        "primary_author_cluster_bootstrap_summary.csv": primary_summary,
        "secondary_pair_bootstrap_draws.csv": secondary_draws,
        "secondary_pair_bootstrap_summary.csv": secondary_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(RESULTS / name, index=False)
    manifest = {
        "schema_version": "author-disjoint-newspaper-final-statistics-v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_runs_verified": 30,
        "observed_author_clusters": 10,
        "small_cluster_limitation": statistics_config["mandatory_limitation"],
        "primary_uncertainty": statistics_config["primary_bootstrap"],
        "secondary_uncertainty": statistics_config["secondary_bootstrap"],
        "original_freeze_sha256": sha256_file(FREEZE),
        "original_config_sha256": sha256_file(CONFIG),
        "outputs": {name: sha256_file(RESULTS / name) for name in outputs},
    }
    manifest_path = RESULTS / "FINAL_STATISTICS_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

