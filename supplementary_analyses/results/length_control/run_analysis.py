#!/usr/bin/env python
"""Post-hoc Combined-v2 length-control analysis; no transformer fitting/inference."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = (42, 137, 314, 2026, 3407)
FOLDS = tuple(range(5))
METHODS = {"CE_ONLY_MATCHED": "ce", "SUPCON": "supcon"}
DOMAINS = ("COMBINED", "ABSTRACT", "NEWSPAPER")
RULES = ("UNRESTRICTED", "ABS_LE_5", "ABS_LE_10", "ABS_LE_15", "REL_LE_10PCT")
BOOT_SEED = 261015
N_BOOT = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def auc(labels: pd.Series, scores: pd.Series) -> float:
    return float(roc_auc_score(labels, scores)) if labels.nunique() == 2 else np.nan


def ci(values: list[float]) -> tuple[float, float]:
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def main() -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    study_path = ROOT / "data/study/paired_study.csv"
    folds_path = ROOT / "data/study/nested_folds.csv"
    a2_dir = ROOT / "results/supplementary_exploratory_2026/02_length_truncation"
    a2_doc_path = a2_dir / "length_truncation_document_features.csv"
    study = pd.read_csv(study_path)
    folds = pd.read_csv(folds_path)
    if len(study) != 3000 or study.pair_id.nunique() != 1500:
        raise RuntimeError("Combined-v2 does not contain the authoritative 3,000 texts / 1,500 pairs")
    if study.groupby("pair_id").size().ne(2).any() or study.groupby("pair_id").label.nunique().ne(2).any():
        raise RuntimeError("Every pair must contain exactly one HUMAN and one AI text")
    if len(folds) != 1500 or not folds.pair_id.is_unique or set(folds.outer_fold) != set(FOLDS):
        raise RuntimeError("Unexpected outer-fold manifest")
    data = study.merge(folds[["pair_id", "outer_fold"]], on="pair_id", validate="many_to_one")
    data["binary_label"] = data.label.map({"HUMAN": 0, "AI": 1})
    if data.binary_label.isna().any() or not pd.api.types.is_numeric_dtype(data.word_count):
        raise RuntimeError("Invalid labels or word counts")

    a2_doc = pd.read_csv(a2_doc_path)
    a2_internal = a2_doc[a2_doc.dataset.eq("combined_v2")].copy()
    cross = data[["text_id", "word_count"]].merge(
        a2_internal[["document_id", "word_count"]], left_on="text_id", right_on="document_id",
        suffixes=("_study", "_a2"), validate="one_to_one"
    )
    if len(cross) != 3000 or not cross.word_count_study.eq(cross.word_count_a2).all():
        raise RuntimeError("Authoritative word counts do not agree with validated A2")

    # Pair-level length table and fixed inclusion rules.
    wide = data.pivot(index=["pair_id", "domain", "outer_fold"], columns="label", values="word_count").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"HUMAN": "human_word_count", "AI": "ai_word_count"})
    wide["absolute_word_count_difference"] = (wide.ai_word_count - wide.human_word_count).abs()
    wide["relative_word_count_difference"] = wide.absolute_word_count_difference / np.maximum(
        (wide.ai_word_count + wide.human_word_count) / 2, 1
    )
    wide["signed_ai_minus_human_words"] = wide.ai_word_count - wide.human_word_count
    wide["UNRESTRICTED"] = True
    wide["ABS_LE_5"] = wide.absolute_word_count_difference.le(5)
    wide["ABS_LE_10"] = wide.absolute_word_count_difference.le(10)
    wide["ABS_LE_15"] = wide.absolute_word_count_difference.le(15)
    wide["REL_LE_10PCT"] = wide.relative_word_count_difference.le(0.10)
    wide.to_csv(HERE / "PAIRWISE_LENGTH_DIFFERENCES.csv", index=False, float_format="%.12f")

    pair_summary = []
    for domain in DOMAINS:
        base = wide if domain == "COMBINED" else wide[wide.domain.eq(domain)]
        for rule in RULES:
            retained = int(base[rule].sum())
            pair_summary.append(
                {"domain": domain, "selection_rule": rule, "total_pairs": len(base),
                 "retained_pairs": retained, "excluded_pairs": len(base) - retained,
                 "retained_proportion": retained / len(base),
                 "median_absolute_difference": base.absolute_word_count_difference.median(),
                 "mean_absolute_difference": base.absolute_word_count_difference.mean(),
                 "median_relative_difference": base.relative_word_count_difference.median(),
                 "mean_signed_ai_minus_human_words": base.signed_ai_minus_human_words.mean()}
            )
    pair_summary = pd.DataFrame(pair_summary)
    pair_summary.to_csv(HERE / "PAIRWISE_LENGTH_SUMMARY.csv", index=False, float_format="%.12f")

    # Word-count-only pair-disjoint OOF baselines. Domain-specific models train only in-domain.
    baseline_predictions = []
    baseline_folds = []
    baseline_summary = []
    for domain in DOMAINS:
        population = data if domain == "COMBINED" else data[data.domain.eq(domain)]
        for fold in FOLDS:
            train = population[population.outer_fold.ne(fold)]
            test = population[population.outer_fold.eq(fold)]
            model = Pipeline([
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                                                  random_state=BOOT_SEED, max_iter=1000)),
            ])
            model.fit(train[["word_count"]], train.binary_label)
            scores = model.decision_function(test[["word_count"]])
            q = test[["text_id", "pair_id", "label", "binary_label", "domain", "outer_fold"]].copy()
            q["training_population"] = domain
            q["word_count_score"] = scores
            baseline_predictions.append(q)
            baseline_folds.append(
                {"population": domain, "outer_fold": fold, "train_n": len(train), "test_n": len(test),
                 "test_pairs": test.pair_id.nunique(), "roc_auc": auc(test.binary_label, pd.Series(scores))}
            )
        oof = pd.concat([q for q in baseline_predictions if q.training_population.iloc[0] == domain], ignore_index=True)
        pooled_auc = auc(oof.binary_label, oof.word_count_score)
        fold_frame = pd.DataFrame([r for r in baseline_folds if r["population"] == domain])
        pairs = np.sort(oof.pair_id.unique())
        rng = np.random.default_rng(BOOT_SEED + sum(map(ord, domain)))
        pair_index = {pair: index for index, pair in enumerate(pairs)}
        pair_codes = oof.pair_id.map(pair_index).to_numpy()
        labels = oof.binary_label.to_numpy()
        scores = oof.word_count_score.to_numpy()
        draws = []
        for _ in range(N_BOOT):
            counts = rng.multinomial(len(pairs), np.full(len(pairs), 1 / len(pairs)))
            draws.append(float(roc_auc_score(labels, scores, sample_weight=counts[pair_codes])))
        low, high = ci(draws)
        baseline_summary.append(
            {"population": domain, "n_texts": len(oof), "n_pairs": len(pairs), "pooled_oof_roc_auc": pooled_auc,
             "fold_auc_mean": fold_frame.roc_auc.mean(), "fold_auc_sample_sd": fold_frame.roc_auc.std(ddof=1),
             "pair_bootstrap_ci_low": low, "pair_bootstrap_ci_high": high,
             "bootstrap_unit": "pair_id", "bootstrap_replicates": N_BOOT}
        )
    pd.concat(baseline_predictions, ignore_index=True).to_csv(
        HERE / "LENGTH_ONLY_OOF_PREDICTIONS.csv", index=False, float_format="%.12f"
    )
    pd.DataFrame(baseline_folds).to_csv(HERE / "LENGTH_ONLY_PER_FOLD.csv", index=False, float_format="%.12f")
    baseline_summary = pd.DataFrame(baseline_summary)
    baseline_summary.to_csv(HERE / "LENGTH_ONLY_SUMMARY.csv", index=False, float_format="%.12f")

    # Primary prediction provenance and integrity verification.
    expected_data_hash = sha256(study_path)
    expected_fold_hash = sha256(folds_path)
    provenance = [
        {"kind": "dataset", "method": "", "seed": "", "outer_fold": "", "path": str(study_path.resolve()),
         "sha256": expected_data_hash, "rows": len(study), "status": "PASS"},
        {"kind": "fold_manifest", "method": "", "seed": "", "outer_fold": "", "path": str(folds_path.resolve()),
         "sha256": expected_fold_hash, "rows": len(folds), "status": "PASS"},
        {"kind": "validated_A2_features", "method": "", "seed": "", "outer_fold": "", "path": str(a2_doc_path.resolve()),
         "sha256": sha256(a2_doc_path), "rows": len(a2_doc), "status": "PASS"},
    ]
    predictions = []
    expected_test = data[["text_id", "pair_id", "label", "domain", "outer_fold"]]
    for method, objective in METHODS.items():
        for seed in SEEDS:
            seed_frames = []
            for fold in FOLDS:
                path = ROOT / f"results/primary/{objective}/seed_{seed}/outer_{fold}_predictions.csv"
                run_path = ROOT / f"results/primary/{objective}/seed_{seed}/outer_{fold}_run.json"
                frame = pd.read_csv(path)
                run = json.loads(run_path.read_text(encoding="utf-8"))
                expected = expected_test[expected_test.outer_fold.eq(fold)].sort_values("text_id").reset_index(drop=True)
                observed = frame[["text_id", "pair_id", "label", "domain", "outer_fold"]].sort_values("text_id").reset_index(drop=True)
                checks = [
                    len(frame) == 600, frame.text_id.is_unique, observed.equals(expected),
                    frame.seed.nunique() == 1 and int(frame.seed.iloc[0]) == seed,
                    frame.outer_fold.nunique() == 1 and int(frame.outer_fold.iloc[0]) == fold,
                    frame.objective.nunique() == 1 and frame.objective.iloc[0] == objective,
                    frame.condition.nunique() == 1 and frame.condition.iloc[0] == "A",
                    np.isfinite(frame.raw_decision_score).all(), run.get("status") == "COMPLETE",
                    run.get("experiment") == "COMBINED_V2_PRIMARY_ABLATION_V1",
                    run.get("dataset_sha256") == expected_data_hash, run.get("fold_sha256") == expected_fold_hash,
                    run.get("prediction_sha256") == sha256(path), run.get("train_test_pair_overlap") == 0,
                    run.get("lambda_supcon") == (0.0 if objective == "ce" else 0.1),
                ]
                status = "PASS" if all(checks) else "FAIL"
                provenance.append(
                    {"kind": "primary_oof_prediction", "method": method, "seed": seed, "outer_fold": fold,
                     "path": str(path.resolve()), "sha256": sha256(path), "rows": len(frame), "status": status}
                )
                if status != "PASS":
                    raise RuntimeError(f"Primary prediction provenance failure: {path}")
                frame["method"] = method
                seed_frames.append(frame)
            pooled = pd.concat(seed_frames, ignore_index=True)
            if len(pooled) != 3000 or not pooled.text_id.is_unique:
                raise RuntimeError(f"Incomplete pooled OOF grid: {method}, seed {seed}")
            predictions.append(pooled)
    provenance = pd.DataFrame(provenance)
    provenance.to_csv(HERE / "INPUT_PREDICTION_PROVENANCE.csv", index=False)
    pred = pd.concat(predictions, ignore_index=True)
    pred["binary_label"] = pred.label.map({"HUMAN": 0, "AI": 1})

    selected_pairs = {(domain, rule): set(
        (wide if domain == "COMBINED" else wide[wide.domain.eq(domain)]).loc[lambda x: x[rule], "pair_id"]
    ) for domain in DOMAINS for rule in RULES}
    fold_rows = []
    seed_rows = []
    for domain in DOMAINS:
        domain_pred = pred if domain == "COMBINED" else pred[pred.domain.eq(domain)]
        for rule in RULES:
            subset = domain_pred[domain_pred.pair_id.isin(selected_pairs[(domain, rule)])]
            for (method, seed, fold), group in subset.groupby(["method", "seed", "outer_fold"], sort=True):
                n_human = int(group.binary_label.eq(0).sum())
                n_ai = int(group.binary_label.eq(1).sum())
                fold_rows.append(
                    {"domain": domain, "selection_rule": rule, "method": method, "seed": seed,
                     "outer_fold": fold, "n_pairs": group.pair_id.nunique(), "n_human": n_human, "n_ai": n_ai,
                     "roc_auc": auc(group.binary_label, group.raw_decision_score),
                     "estimability": "UNDEFINED_MISSING_CLASS" if min(n_human, n_ai) == 0 else
                                     ("SPARSE_LT20_PER_CLASS" if min(n_human, n_ai) < 20 else "ADEQUATE")}
                )
            for (method, seed), group in subset.groupby(["method", "seed"], sort=True):
                n_human = int(group.binary_label.eq(0).sum())
                n_ai = int(group.binary_label.eq(1).sum())
                seed_rows.append(
                    {"domain": domain, "selection_rule": rule, "method": method, "seed": seed,
                     "n_pairs": group.pair_id.nunique(), "n_human": n_human, "n_ai": n_ai,
                     "roc_auc_pooled_oof": auc(group.binary_label, group.raw_decision_score),
                     "aggregation": "five pair-disjoint outer folds pooled within seed"}
                )
    fold_metrics = pd.DataFrame(fold_rows)
    seed_metrics = pd.DataFrame(seed_rows)
    fold_metrics.to_csv(HERE / "LENGTH_MATCHED_PER_FOLD.csv", index=False, float_format="%.12f")
    seed_metrics.to_csv(HERE / "LENGTH_MATCHED_SEED_LEVEL.csv", index=False, float_format="%.12f")
    aggregate = seed_metrics.groupby(["domain", "selection_rule", "method"]).agg(
        n_seeds=("seed", "size"), retained_pairs=("n_pairs", "first"),
        roc_auc_mean=("roc_auc_pooled_oof", "mean"), roc_auc_sample_sd=("roc_auc_pooled_oof", "std"),
        roc_auc_min=("roc_auc_pooled_oof", "min"), roc_auc_max=("roc_auc_pooled_oof", "max")
    ).reset_index()
    total_pairs = {"COMBINED": 1500, "ABSTRACT": 1000, "NEWSPAPER": 500}
    aggregate["excluded_pairs"] = aggregate.apply(lambda row: total_pairs[row.domain] - row.retained_pairs, axis=1)
    aggregate.to_csv(HERE / "LENGTH_MATCHED_AGGREGATE.csv", index=False, float_format="%.12f")
    unrestricted = seed_metrics[seed_metrics.selection_rule.eq("UNRESTRICTED")][
        ["domain", "method", "seed", "roc_auc_pooled_oof"]
    ].rename(columns={"roc_auc_pooled_oof": "unrestricted_roc_auc"})
    contrasts = seed_metrics.merge(unrestricted, on=["domain", "method", "seed"], validate="many_to_one")
    contrasts["matched_minus_unrestricted"] = contrasts.roc_auc_pooled_oof - contrasts.unrestricted_roc_auc
    contrasts.to_csv(HERE / "LENGTH_MATCHED_VS_UNRESTRICTED.csv", index=False, float_format="%.12f")

    # Verify unrestricted pooled OOF AUCs against completed A2.
    a2_disc = pd.read_csv(a2_dir / "length_subgroup_discrimination.csv")
    a2_all = a2_disc[(a2_disc.population.eq("internal")) & (a2_disc.dataset.eq("combined_v2")) &
                     (a2_disc.subgroup.eq("all")) & (a2_disc.method.isin(METHODS))]
    verify = seed_metrics[(seed_metrics.domain.eq("COMBINED")) & seed_metrics.selection_rule.eq("UNRESTRICTED")].merge(
        a2_all[["method", "seed", "roc_auc"]], on=["method", "seed"], validate="one_to_one"
    )
    verify["absolute_difference"] = (verify.roc_auc_pooled_oof - verify.roc_auc).abs()
    verify["status"] = np.where(verify.absolute_difference.le(1e-12), "PASS", "FAIL")
    verify.to_csv(HERE / "A2_UNRESTRICTED_METRIC_RECONCILIATION.csv", index=False, float_format="%.12f")
    if not verify.status.eq("PASS").all():
        raise RuntimeError("Unrestricted primary metrics do not reproduce validated A2")

    # Existing validated tokenization/truncation features, explicitly split by domain and class.
    token = data[["text_id", "domain", "label"]].merge(
        a2_internal.rename(columns={"document_id": "text_id"})[
            ["text_id", "word_count", "wordpiece_count", "effective_berturk_tokens", "truncated_berturk",
             "excluded_wordpieces", "excluded_wordpiece_proportion"]
        ], on="text_id", validate="one_to_one"
    )
    token_rows = []
    for domain in DOMAINS:
        domain_frame = token if domain == "COMBINED" else token[token.domain.eq(domain)]
        for label in ("HUMAN", "AI", "ALL"):
            group = domain_frame if label == "ALL" else domain_frame[domain_frame.label.eq(label)]
            truncated = group[group.truncated_berturk.eq(1)]
            token_rows.append(
                {"domain": domain, "class": label, "n": len(group), "word_count_mean": group.word_count.mean(),
                 "word_count_median": group.word_count.median(), "word_count_q25": group.word_count.quantile(.25),
                 "word_count_q75": group.word_count.quantile(.75), "wordpiece_mean": group.wordpiece_count.mean(),
                 "wordpiece_median": group.wordpiece_count.median(), "truncated_n": int(group.truncated_berturk.sum()),
                 "truncated_proportion": group.truncated_berturk.mean(),
                 "excluded_fraction_mean_all": group.excluded_wordpiece_proportion.mean(),
                 "excluded_fraction_mean_if_truncated": truncated.excluded_wordpiece_proportion.mean(),
                 "excluded_wordpieces_median_if_truncated": truncated.excluded_wordpieces.median(),
                 "source": "validated existing A2 document features"}
            )
    token_summary = pd.DataFrame(token_rows)
    token_summary.to_csv(HERE / "DOMAIN_TRUNCATION_COMPARISON.csv", index=False, float_format="%.12f")

    # Compact evidence-led interpretation.
    baseline_lookup = baseline_summary.set_index("population")
    abs10 = aggregate[(aggregate.domain.eq("COMBINED")) & aggregate.selection_rule.eq("ABS_LE_10")]
    abs10_pairs = pair_summary[(pair_summary.domain.eq("COMBINED")) & pair_summary.selection_rule.eq("ABS_LE_10")].iloc[0]
    trunc_lookup = token_summary.set_index(["domain", "class"])
    lines = [
        "# Scientific interpretation and limitations", "",
        "Status: **COMPLETE / VALIDATED**. This is a post-hoc exploratory sensitivity analysis.", "",
        "## Findings", "",
        f"- Word count alone was weak for Combined-v2 (pooled OOF ROC-AUC {baseline_lookup.loc['COMBINED','pooled_oof_roc_auc']:.3f}) and differed by domain: ABSTRACT {baseline_lookup.loc['ABSTRACT','pooled_oof_roc_auc']:.3f}, NEWSPAPER {baseline_lookup.loc['NEWSPAPER','pooled_oof_roc_auc']:.3f}.",
        f"- The ±10-word rule retained {int(abs10_pairs.retained_pairs)}/{int(abs10_pairs.total_pairs)} pairs. Across five seeds, CE/SupCon mean pooled-OOF AUCs remained {abs10.set_index('method').loc['CE_ONLY_MATCHED','roc_auc_mean']:.6f}/{abs10.set_index('method').loc['SUPCON','roc_auc_mean']:.6f}.",
        f"- Combined-v2 truncation affected {trunc_lookup.loc[('COMBINED','HUMAN'),'truncated_proportion']:.1%} of HUMAN and {trunc_lookup.loc[('COMBINED','AI'),'truncated_proportion']:.1%} of AI texts. Domain × class results are reported separately; domain differences alone are not evidence of a class shortcut.",
        "- Near-ceiling CE/SupCon discrimination persists under every prespecified pair-preserving length criterion. This reduces concern that gross pairwise word-count differences alone explain internal performance, but does not establish independence from length or correlated construction cues.", "",
        "## Limitations", "",
        "- Pair filtering changes the target evaluation population.",
        "- Outer-fold estimates and repeated training seeds are dependent; fold rows are diagnostics, not independent replicates.",
        "- Seed SD/ranges describe training-realization variability. The baseline pair bootstrap is conditional on the fixed folds and specification.",
        "- High matched-subset performance cannot rule out subtler length effects or correlated corpus artifacts.",
        "- This analysis does not resolve historical-period or generator-prompt confounding.",
    ]
    (HERE / "SCIENTIFIC_INTERPRETATION_AND_LIMITATIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme = """# Task 4 reproduction

Run from the project root:

```powershell
.\\.venv-gpu\\Scripts\\python.exe .\\supplementary_analyses\\results\\length_control\\run_analysis.py
```

This is CPU-only and reads existing Combined-v2, folds, A2 diagnostics, and primary OOF predictions. It performs no transformer training or inference and writes only in this directory.
"""
    (HERE / "README.md").write_text(readme, encoding="utf-8")
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS_LENGTH_CONTROL_VALIDATIONS",
        "dataset_sha256": expected_data_hash, "fold_sha256": expected_fold_hash,
        "prediction_cells": 50, "prediction_cells_passed": int((provenance.kind == "primary_oof_prediction").sum()),
        "a2_unrestricted_cells_reconciled": len(verify), "a2_max_absolute_metric_difference": verify.absolute_difference.max(),
        "length_only_oof_rows": int(sum(len(x) for x in baseline_predictions)),
        "sparse_fold_cells": int(fold_metrics.estimability.str.startswith("SPARSE").sum()),
        "undefined_fold_cells": int(fold_metrics.estimability.str.startswith("UNDEFINED").sum()),
        "config_sha256": sha256(HERE / "config.json"), "script_sha256": sha256(Path(__file__)),
    }
    (HERE / "VALIDATION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("PASS_LENGTH_CONTROL_VALIDATIONS")


if __name__ == "__main__":
    main()
