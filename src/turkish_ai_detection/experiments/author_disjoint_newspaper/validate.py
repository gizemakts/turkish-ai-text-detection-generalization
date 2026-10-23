"""Preflight/fold and post-run integrity validation; computes no performance metric."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .common import (
    CONFIG, DATA, FOLDS, FREEZE, OBJECTIVES, OUT, OUTERS, ROOT, SEEDS,
    load_config_and_freeze, load_data_and_folds, require, sha256_file,
    split_pair_ids,
)


def validate_design() -> dict:
    config, freeze = load_config_and_freeze()
    data, folds = load_data_and_folds()
    require(sha256_file(DATA) == config["source_dataset_sha256"], "dataset/config hash mismatch")
    require(sha256_file(FOLDS) == config["fold_assignment_sha256"], "fold/config hash mismatch")
    require(len(data) == 3000 and data.pair_id.nunique() == 1500, "Combined-v2 count changed")
    news = data[data.domain.eq("NEWSPAPER")].copy()
    abstract = data[data.domain.eq("ABSTRACT")].copy()
    require(len(news) == 1000 and news.pair_id.nunique() == 500, "NEWSPAPER count mismatch")
    require(len(abstract) == 2000 and abstract.pair_id.nunique() == 1000, "ABSTRACT count mismatch")
    require(len(folds) == 500 and folds.pair_id.nunique() == 500, "fold assignment must contain 500 unique pairs")

    human = news[news.label.astype(str).str.upper().eq("HUMAN")].copy()
    ai = news[news.label.astype(str).str.upper().eq("AI")].copy()
    require(len(human) == 500 and len(ai) == 500, "NEWSPAPER class count mismatch")
    pair_sizes = news.groupby("pair_id").size()
    require(pair_sizes.eq(2).all(), "each NEWSPAPER pair must have two rows")
    label_counts = news.groupby("pair_id").label.nunique()
    require(label_counts.eq(2).all(), "each pair must contain HUMAN and AI")
    inherited = human[["pair_id", "author"]].merge(ai[["pair_id", "author"]], on="pair_id", suffixes=("_human", "_ai"), validate="one_to_one")
    require(inherited.author_human.eq(inherited.author_ai).all(), "AI did not inherit HUMAN author")
    require(set(folds.pair_id.astype(str)) == set(news.pair_id.astype(str)), "fold pair set mismatch")

    human_author = human[["pair_id", "author"]].rename(columns={"author": "human_author"})
    joined = folds.merge(human_author, on="pair_id", validate="one_to_one")
    require(joined.author.astype(str).eq(joined.human_author.astype(str)).all(), "fold author/HUMAN author mismatch")
    authors = sorted(human.author.astype(str).unique(), key=lambda x: x.encode("utf-8"))
    require(len(authors) == 10, "10 unique authors required")
    require(folds.author.nunique() == 10, "all 10 authors must be assigned")
    require(set(folds.author_fold.astype(int)) == set(OUTERS), "fold coverage mismatch")

    generator_expected = {"Claude": 20, "Gemini": 20, "GPT": 20, "Kimi": 20, "Qwen": 20}
    fold_records = []
    for outer in OUTERS:
        subset = folds[folds.author_fold.eq(outer)]
        held_authors = sorted(subset.author.unique(), key=lambda x: x.encode("utf-8"))
        require(len(held_authors) == 2, f"fold {outer}: expected 2 authors")
        require(len(subset) == 100, f"fold {outer}: expected 100 pairs")
        train_pairs, test_pairs = split_pair_ids(data, folds, outer)
        require(len(train_pairs) == 1400 and len(test_pairs) == 100, f"fold {outer}: pair counts")
        train = data[data.pair_id.astype(str).isin(train_pairs)]
        test = data[data.pair_id.astype(str).isin(test_pairs)]
        require(len(train) == 2800 and len(test) == 200, f"fold {outer}: row counts")
        require(set(test.domain.astype(str)) == {"NEWSPAPER"}, f"fold {outer}: test domain")
        train_authors = set(train.loc[train.domain.eq("NEWSPAPER"), "author"].astype(str))
        test_authors = set(test.author.astype(str))
        require(not (train_authors & test_authors), f"fold {outer}: author leakage")
        require(set(train.pair_id.astype(str)).isdisjoint(set(test.pair_id.astype(str))), f"fold {outer}: pair leakage")
        require(test.groupby("pair_id").size().eq(2).all(), f"fold {outer}: split pair")
        ai_test = test[test.label.astype(str).str.upper().eq("AI")]
        generators = {str(k): int(v) for k, v in ai_test.generator.value_counts().sort_index().items()}
        require(generators == generator_expected, f"fold {outer}: generator imbalance {generators}")
        fold_records.append({
            "fold": outer,
            "authors": held_authors,
            "train_pairs": 1400,
            "test_pairs": 100,
            "train_rows": 2800,
            "test_rows": 200,
            "author_leakage": 0,
            "pair_leakage": 0,
            "generator_family_counts": generators,
        })

    return {
        "status": "PASS",
        "freeze_status": freeze["status"],
        "authors_assigned": 10,
        "newspaper_pairs": 500,
        "human_ai_pair_members_split": False,
        "additional_generator_stratification_applied": False,
        "folds": fold_records,
        "scientific_results_computed": False,
    }


def validate_predictions() -> dict:
    validate_design()
    expected_columns = ["text_id", "pair_id", "label", "domain", "generator", "author",
                        "author_fold", "seed", "objective", "condition", "raw_decision_score"]
    reference = {}
    verified = 0
    for objective in OBJECTIVES:
        for seed in SEEDS:
            for outer in OUTERS:
                base = OUT / objective / f"seed_{seed}"
                pred_path = base / f"fold_{outer}_predictions.csv"
                run_path = base / f"fold_{outer}_run.json"
                require(pred_path.is_file() and run_path.is_file(), f"missing run: {objective}/{seed}/{outer}")
                run = json.loads(run_path.read_text(encoding="utf-8"))
                require(run.get("status") == "COMPLETE", f"incomplete run: {objective}/{seed}/{outer}")
                require(run.get("prediction_sha256") == sha256_file(pred_path), f"prediction hash mismatch: {objective}/{seed}/{outer}")
                require(run.get("outer_test_rows_used_for_training") == 0, f"test use: {objective}/{seed}/{outer}")
                require(run.get("train_test_author_overlap") == 0, f"author overlap: {objective}/{seed}/{outer}")
                frame = pd.read_csv(pred_path)
                require(list(frame.columns) == expected_columns, f"schema mismatch: {objective}/{seed}/{outer}")
                require(len(frame) == 200 and frame.text_id.nunique() == 200, f"row mismatch: {objective}/{seed}/{outer}")
                require(frame.pair_id.nunique() == 100, f"pair mismatch: {objective}/{seed}/{outer}")
                require(set(frame.author_fold.astype(int)) == {outer}, f"fold mismatch: {objective}/{seed}/{outer}")
                require(set(frame.seed.astype(int)) == {seed}, f"seed mismatch: {objective}/{seed}/{outer}")
                require(set(frame.objective.astype(str)) == {objective}, f"objective mismatch: {objective}/{seed}/{outer}")
                require(set(frame.domain.astype(str)) == {"NEWSPAPER"}, f"domain mismatch: {objective}/{seed}/{outer}")
                require(np.isfinite(frame.raw_decision_score.to_numpy(float)).all(), f"non-finite score: {objective}/{seed}/{outer}")
                identity = frame[["text_id", "pair_id", "label", "author", "author_fold"]].sort_values("text_id").reset_index(drop=True)
                if outer not in reference:
                    reference[outer] = identity
                else:
                    require(identity.equals(reference[outer]), f"paired alignment failed: {objective}/{seed}/{outer}")
                verified += 1
    return {"status": "PASS", "verified_runs": verified, "expected_runs": 30,
            "paired_alignment": True, "scientific_results_computed": False}


def finalize() -> dict:
    integrity = validate_predictions()
    stats_manifest = OUT / "final_statistics" / "FINAL_STATISTICS_MANIFEST.json"
    require(stats_manifest.is_file(), "statistics manifest missing")
    stats = json.loads(stats_manifest.read_text(encoding="utf-8"))
    require(stats.get("status") == "COMPLETE", "statistics incomplete")
    target = OUT / "FINAL_RUN_MANIFEST.json"
    record = {
        "schema_version": "author-disjoint-newspaper-final-run-v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verified_training_runs": 30,
        "prediction_integrity": integrity,
        "freeze_sha256": sha256_file(FREEZE),
        "statistics_manifest_sha256": sha256_file(stats_manifest),
    }
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        require(old.get("freeze_sha256") == record["freeze_sha256"], "existing final manifest freeze mismatch")
        require(old.get("statistics_manifest_sha256") == record["statistics_manifest_sha256"], "existing final manifest statistics mismatch")
        return {"status": "SKIPPED_VERIFIED_COMPLETE", "path": str(target)}
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "COMPLETE", "path": str(target)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    result = finalize() if args.finalize else validate_predictions() if args.predictions else validate_design()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

