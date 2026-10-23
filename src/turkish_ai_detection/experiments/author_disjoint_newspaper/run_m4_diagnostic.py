"""Optional M4 diagnostic corpus-separability baseline; dry-run by default."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import OUT, OUTERS, load_config_and_freeze, load_data_and_folds, require, sha256_file, split_pair_ids
from turkish_ai_detection.evaluation.metrics import C_GRID, condition_text, labels, make_tfidf, _pauc
from turkish_ai_detection.features.core import turkish_lower


def inner_assignment(pair_ids, outer: int) -> dict[str, int]:
    ranked = sorted(map(str, pair_ids), key=lambda p: hashlib.sha256(f"author-disjoint-inner-v1|{outer}|{p}".encode()).digest())
    return {pair_id: index % 4 for index, pair_id in enumerate(ranked)}


def run() -> None:
    config, _freeze = load_config_and_freeze()
    data, folds = load_data_and_folds()
    data["model_text"] = condition_text(data, "A").map(turkish_lower)
    target = OUT / "m4_diagnostic"
    if target.exists() and any(target.iterdir()):
        raise RuntimeError("M4 output directory is non-empty; refusing silent overwrite")
    target.mkdir(parents=True, exist_ok=False)
    predictions, tuning, run_records = [], [], []
    for outer in OUTERS:
        train_pairs, test_pairs = split_pair_ids(data, folds, outer)
        train = data[data.pair_id.astype(str).isin(train_pairs)].copy()
        test = data[data.pair_id.astype(str).isin(test_pairs)].copy()
        assignment = inner_assignment(train_pairs, outer)
        y_train = labels(train)
        candidate = []
        for c_value in C_GRID:
            inner_oof = np.full(len(train), np.nan)
            for inner in range(4):
                validation = train.pair_id.astype(str).map(assignment).eq(inner).to_numpy()
                require(validation.any() and (~validation).any(), "empty M4 inner partition")
                model = make_tfidf(c_value).fit(train.loc[~validation, "model_text"], y_train[~validation])
                inner_oof[validation] = model.decision_function(train.loc[validation, "model_text"])
            value = float(_pauc(y_train, inner_oof))
            candidate.append((value, float(c_value)))
            tuning.append({"author_fold": outer, "C": float(c_value), "training_inner_oof_std_pauc_003": value})
        best_c = sorted(candidate, key=lambda item: (-item[0], item[1]))[0][1]
        model = make_tfidf(best_c).fit(train.model_text, y_train)
        score = model.decision_function(test.model_text)
        frame = test[["text_id", "pair_id", "label", "domain", "generator", "author"]].copy()
        frame["author_fold"] = outer
        frame["model"] = "M4_DIAGNOSTIC"
        frame["raw_decision_score"] = score
        predictions.append(frame)
        run_records.append({"author_fold": outer, "best_C": best_c, "train_pairs": 1400, "test_pairs": 100,
                            "tfidf_fit_population": "training_only", "test_transform_only": True})
    prediction = pd.concat(predictions, ignore_index=True)
    require(len(prediction) == 1000 and prediction.pair_id.nunique() == 500, "M4 prediction completeness")
    pred_path = target / "m4_author_disjoint_predictions.csv"
    tune_path = target / "m4_training_only_tuning.csv"
    prediction.to_csv(pred_path, index=False)
    pd.DataFrame(tuning).to_csv(tune_path, index=False)
    record = {
        "schema_version": "author-disjoint-m4-diagnostic-v1",
        "status": "COMPLETE_PREDICTIONS_NOT_SUMMARIZED",
        "label": "diagnostic corpus-separability baseline",
        "main_ce_supcon_comparison_member": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": config["m4_diagnostic"],
        "folds": run_records,
        "prediction_sha256": sha256_file(pred_path),
        "tuning_sha256": sha256_file(tune_path),
        "final_auc_computed": False,
    }
    (target / "M4_RUN_MANIFEST.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("DRY_RUN_ONLY: M4 diagnostic is prepared; add --execute manually")
        return 0
    run()
    print(json.dumps({"status": "COMPLETE_PREDICTIONS_NOT_SUMMARIZED", "final_auc_computed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

